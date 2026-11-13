"""
Automatic model selection (model="auto") with a local decision model.

A Router looks at the task an agent is about to run and picks one of its
presets. D1Router asks Liquid AI's d1-2B decision model, served by
llama-server (v0.8.0+) on its /v1/systemone endpoint. Two methods:

  "choice" (default)  "Which assistant should handle this task?", one option
                      per preset with a description of the work it is for:
                        quick    -> fast:    lookups, listing, running commands...
                        standard -> capable: explaining, writing code, analysis...
                        thorough -> expert:  open-ended design, subtle reasoning...
  "complexity"        "How complex is it to answer this prompt well?" on 5
                      levels; the mean level maps to a preset with cutoffs.

Each agent can restrict the presets it may get (Agent(auto_presets=...)): d1
always sees every option, then its pick is clamped to the nearest allowed
preset, so a task d1 calls "expert" goes to a worker's most capable preset.

The labelled prompts in evals/routing.jsonl check the descriptions and
defaults against the real model: python scripts/routing_eval.py

Server: D1Router uses a running llama-server when given a URL (or D1_URL).
Otherwise it starts one on first use and keeps it for the life of the process:
  binary: llama_server= > $LLAMA_SERVER > llama-server on PATH
  model:  model_path= > $D1_MODEL > the Hugging Face cache, which you fill with
          hf download LiquidAI/d1-2B-GGUF d1-2B-Q8_0.gguf

When d1 cannot be reached, routing falls back to a preset (the agent's
auto_fallback, else the router's fallback_preset) with a warning instead of
failing the run.
"""

import asyncio
import atexit
import contextlib
import glob
import json
import logging
import os
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path

from .result import RouteDecision

log = logging.getLogger("omniharness.routing")

# ── "choice" method ────────────────────────────────────────────────────────

CHOICE_INSTRUCTIONS = "Which assistant should handle this task?"

# preset -> (option label shown to d1, description), cheapest first. The order
# is the tier order used to clamp a pick to an agent's allowed presets.
DEFAULT_CHOICES: dict[str, tuple[str, str]] = {
    "quick": ("fast", ("A fast, inexpensive assistant: lookups, listing, running commands "
                       "and reporting results, short or mechanical tasks")),
    "standard": ("capable", ("A capable assistant: explaining, writing or changing code, "
                             "comparing options, multi-step analysis")),
    "thorough": ("expert", ("An expert assistant: open-ended design, subtle concurrency or "
                            "correctness reasoning, deep expertise")),
}

# ── "complexity" method ────────────────────────────────────────────────────

COMPLEXITY_LEVELS = [
    "trivial: one fact or a one-word answer",
    "simple: a short, direct task",
    "moderate: a few steps, or some explanation",
    "complex: several connected steps, or specialist knowledge",
    "very complex: an open-ended, multi-part problem needing expert design or deep reasoning",
]
LEVEL_NAMES = ["trivial", "simple", "moderate", "complex", "very complex"]
COMPLEXITY_INSTRUCTIONS = "How complex is it to answer this prompt well?"

# (exclusive upper bound on the mean level, preset), cheapest first; the last
# bound must be None. Calibrated on chat prompts: agent tasks with tools tend
# to rate 1.5-3.0 whatever their difficulty, hence "choice" by default.
DEFAULT_TIERS: list[tuple[float | None, str]] = [
    (1.5, "quick"),
    (3.5, "standard"),
    (None, "thorough"),
]

D1_REPO_DIR = "models--LiquidAI--d1-2B-GGUF"
# llama-server reads a question in one batch: -b/-ub must hold the whole state.
UBATCH = 8192
SLOTS = 4
# ~4 chars per token keeps the state well inside one UBATCH.
MAX_STATE_CHARS = 20000


def clamp(preset: str, order: Sequence[str], allowed: Sequence[str] | None) -> str:
    """The allowed preset nearest to `preset` in tier order (ties: the more capable)."""
    if not allowed or preset in allowed:
        return preset
    rank = {p: i for i, p in enumerate(order)}
    target = rank[preset]
    return min(allowed, key=lambda p: (abs(rank[p] - target), -rank[p]))


class Router(ABC):
    """Picks a preset for a task. route() must not raise: fall back instead."""

    # Presets this router can pick, cheapest first.
    tier_order: list[str]
    fallback_preset: str = "standard"

    @abstractmethod
    async def route(
        self,
        prompt: str,
        system_prompt: str = "",
        *,
        allowed: Sequence[str] | None = None,
        fallback: str | None = None,
    ) -> RouteDecision:
        """
        Args:
            allowed: Presets the agent may use (a subset of tier_order);
                None means all.
            fallback: Preset when routing fails; defaults to fallback_preset.
        """
        raise NotImplementedError


class FixedRouter(Router):
    """Always the same preset (clamped to the allowed ones). For tests, or to disable routing."""

    def __init__(self, preset: str, tier_order: Sequence[str] = tuple(DEFAULT_CHOICES)):
        self.preset = preset
        self.tier_order = list(tier_order)

    async def route(self, prompt, system_prompt="", *, allowed=None, fallback=None):
        return RouteDecision(preset=clamp(self.preset, self.tier_order, allowed),
                             source="fixed", choice=self.preset)


class D1Error(RuntimeError):
    pass


class D1Router(Router):
    def __init__(
        self,
        url: str | None = None,
        *,
        method: str = "choice",
        choices: dict[str, tuple[str, str]] | None = None,
        tiers: list[tuple[float | None, str]] | None = None,
        fallback_preset: str = "standard",
        include_system_prompt: bool = False,
        autostart: bool = True,
        llama_server: str | None = None,
        model_path: str | None = None,
        quant: str = "Q8_0",
        startup_timeout: float = 120.0,
        request_timeout: float = 60.0,
    ):
        """
        Args:
            url: A running llama-server serving d1-2B. Defaults to $D1_URL.
            method: "choice" (default) or "complexity".
            choices: For "choice": {preset: (label, description)}, cheapest first.
            tiers: For "complexity": [(upper bound on the mean level, preset),
                ..., (None, preset)], cheapest first.
            fallback_preset: Used when d1 fails and the agent sets no fallback.
            include_system_prompt: Also show d1 the agent's system prompt.
                Off by default: d1 judges the task itself.
            autostart: Start llama-server when no URL is configured.
            llama_server / model_path / quant: How to start it (see module doc).
        """
        if method not in ("choice", "complexity"):
            raise ValueError(f"unknown method '{method}': use 'choice' or 'complexity'")
        self.url = (url or os.getenv("D1_URL") or "").rstrip("/") or None
        self.method = method
        self.choices = dict(choices or DEFAULT_CHOICES)
        if len(self.choices) < 2:
            raise ValueError("choices needs at least 2 presets")
        self.tiers = tiers or DEFAULT_TIERS
        if self.tiers[-1][0] is not None:
            raise ValueError("the last tier must have no upper bound: (None, preset)")
        if method == "choice":
            self.tier_order = list(self.choices)
        else:
            self.tier_order = [preset for _, preset in self.tiers]
        self.fallback_preset = fallback_preset
        self.include_system_prompt = include_system_prompt
        self.autostart = autostart
        self.llama_server = llama_server
        self.model_path = model_path
        self.quant = quant
        self.startup_timeout = startup_timeout
        self.request_timeout = request_timeout
        self._process: subprocess.Popen | None = None
        # llama-server's output; kept after a failed start for debugging.
        self.log_path: str | None = None
        # Thread lock, not asyncio: run_sync() calls each use their own event loop.
        self._start_lock = threading.Lock()
        # Set when d1 cannot be started: stop retrying (and warning) every run.
        self._unavailable: str | None = None

    # ── routing ────────────────────────────────────────────────────────────

    async def route(self, prompt, system_prompt="", *, allowed=None, fallback=None):
        fallback = fallback or clamp(self.fallback_preset, self.tier_order, allowed)
        if self._unavailable:
            return self._fallback(fallback, self._unavailable, warn=False)
        try:
            url = await self._ensure_server()
            state = self._state(prompt, system_prompt)
            started = time.perf_counter()
            answer = await asyncio.to_thread(self._ask, url, state)
            elapsed_ms = (time.perf_counter() - started) * 1000
            decision = self._decide(answer)
        except D1Error as e:
            # Startup failures were already logged once by _start_once().
            return self._fallback(fallback, str(e), warn=not self._unavailable)
        except (OSError, ValueError, KeyError, TypeError) as e:
            return self._fallback(fallback, f"d1 routing failed: {e}")
        decision.elapsed_ms = elapsed_ms
        decision.preset = clamp(decision.choice, self.tier_order, allowed)
        return decision

    def route_sync(self, prompt: str, system_prompt: str = "", **kwargs) -> RouteDecision:
        return asyncio.run(self.route(prompt, system_prompt, **kwargs))

    def preset_for(self, mean_level: float) -> str:
        """For "complexity": the preset for a mean level, before clamping."""
        for bound, preset in self.tiers:
            if bound is None or mean_level < bound:
                return preset
        raise AssertionError("unreachable: the last tier has no bound")

    def _state(self, prompt: str, system_prompt: str) -> str:
        if len(prompt) > MAX_STATE_CHARS:
            half = MAX_STATE_CHARS // 2
            prompt = f"{prompt[:half]}\n[...]\n{prompt[-half:]}"
        state = f"User prompt:\n{prompt}"
        if self.include_system_prompt and system_prompt:
            # Bounded so instructions + prompt still fit in one batch.
            state = f"Agent instructions:\n{system_prompt[:6000]}\n\n{state}"
        return state

    def _question(self) -> dict:
        if self.method == "choice":
            return {
                "type": "choice",
                "instructions": CHOICE_INSTRUCTIONS,
                "criteria": {label: text for label, text in self.choices.values()},
            }
        return {"type": "score", "instructions": COMPLEXITY_INSTRUCTIONS,
                "criteria": COMPLEXITY_LEVELS}

    def _decide(self, answer: dict) -> RouteDecision:
        if self.method == "choice":
            preset_by_label = {label: preset for preset, (label, _) in self.choices.items()}
            choice = preset_by_label[answer["choice"]]
            return RouteDecision(
                preset=choice,
                choice=choice,
                source="d1",
                method="choice",
                label=answer["choice"],
                confidence=answer.get("confidence"),
                probabilities={preset_by_label[k]: float(v)
                               for k, v in answer["probabilities"].items()},
            )
        probs = {int(k): float(v) for k, v in answer["probabilities"].items()}
        mean = sum(level * p for level, p in probs.items())
        level = min(round(mean), len(LEVEL_NAMES) - 1)
        choice = self.preset_for(mean)
        return RouteDecision(
            preset=choice,
            choice=choice,
            source="d1",
            method="complexity",
            label=LEVEL_NAMES[level],
            complexity=mean,
            confidence=answer.get("confidence"),
            probabilities={str(k): v for k, v in probs.items()},
        )

    def _fallback(self, preset: str, reason: str, warn: bool = True) -> RouteDecision:
        if warn:
            log.warning("%s; using preset '%s'", reason, preset)
        return RouteDecision(preset=preset, source="fallback", reason=reason)

    # ── d1 client ──────────────────────────────────────────────────────────

    def _ask(self, url: str, state: str) -> dict:
        body = json.dumps({"state": state, "questions": {"route": self._question()}}).encode()
        req = urllib.request.Request(f"{url}/v1/systemone", data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.request_timeout) as resp:
                result = json.loads(resp.read().decode("utf-8", errors="replace"))
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")[:300]
            raise D1Error(f"d1 routing failed: HTTP {e.code}: {body}") from e
        return result["answers"]["route"]

    # ── llama-server management ────────────────────────────────────────────

    async def _ensure_server(self) -> str:
        if self.url:
            return self.url
        if not self.autostart:
            raise D1Error("no d1 server: set D1_URL or pass url=")
        return await asyncio.to_thread(self._start_once)

    def _start_once(self) -> str:
        # Concurrent first runs (e.g. parallel subagents) start one server.
        with self._start_lock:
            if self.url:
                return self.url
            if self._unavailable:
                raise D1Error(self._unavailable)
            try:
                self.url = self._start_server()
            except (D1Error, OSError, ValueError) as e:
                self._unavailable = f"d1 unavailable: {e}"
                log.warning("%s; auto models use their fallback preset", self._unavailable)
                raise D1Error(self._unavailable) from e
            return self.url

    def _start_server(self) -> str:
        binary = find_llama_server(self.llama_server)
        model = find_d1_model(self.model_path, self.quant)
        port = _free_port()
        cmd = [
            binary, "-m", model, "--host", "127.0.0.1", "--port", str(port),
            "-c", str(UBATCH * SLOTS), "-np", str(SLOTS),
            "-b", str(UBATCH), "-ub", str(UBATCH),
        ]
        # Output goes to a file, not a pipe: llama-server logs a lot, and an
        # unread pipe fills up and blocks the server.
        with tempfile.NamedTemporaryFile(prefix="omniharness-d1-", suffix=".log",
                                         delete=False) as logfile:
            self.log_path = logfile.name
            log.info("starting d1 (log: %s): %s", self.log_path, " ".join(cmd))
            self._process = subprocess.Popen(
                cmd, stdin=subprocess.DEVNULL, stdout=logfile, stderr=subprocess.STDOUT
            )
        atexit.register(self.close)
        url = f"http://127.0.0.1:{port}"
        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                code, self._process = self._process.returncode, None
                raise D1Error(f"{binary} exited with code {code}: {self._log_tail()}"
                              f"{self._log_hint()} (full log: {self.log_path})")
            try:
                with urllib.request.urlopen(f"{url}/health", timeout=2) as resp:
                    if resp.status == 200:
                        return url
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(0.25)
        self.close(keep_log=True)
        raise D1Error(f"llama-server not ready after {self.startup_timeout:.0f}s: "
                      f"{self._log_tail()} (full log: {self.log_path})")

    def _log_hint(self) -> str:
        """Advice for failures we can recognize in the log."""
        try:
            with open(self.log_path, "rb") as f:
                text = f.read().decode("utf-8", errors="replace")
        except (OSError, TypeError):
            return ""
        if "unknown model architecture" in text:
            return (". This llama-server is too old for d1-2B (lfm3 architecture, "
                    "/v1/systemone): use llama.cpp v0.8.0+ and point LLAMA_SERVER at it")
        return ""

    def _log_tail(self, lines: int = 5) -> str:
        """Last lines of llama-server's log, decoded leniently (it is not always UTF-8)."""
        try:
            with open(self.log_path, "rb") as f:
                f.seek(0, os.SEEK_END)
                f.seek(max(0, f.tell() - 4096))
                text = f.read().decode("utf-8", errors="replace")
        except (OSError, TypeError):
            return "(no log)"
        tail = [line.strip() for line in text.splitlines() if line.strip()][-lines:]
        return " | ".join(tail) or "(empty log)"

    def close(self, keep_log: bool = False) -> None:
        """Stop the llama-server this router started, if any, and remove its log."""
        proc, self._process = self._process, None
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        if proc and self.log_path and not keep_log:
            with contextlib.suppress(OSError):
                os.unlink(self.log_path)
            self.log_path = None


def find_llama_server(explicit: str | None = None) -> str:
    for candidate in (explicit, os.getenv("LLAMA_SERVER")):
        if candidate:
            if not os.access(candidate, os.X_OK):
                raise D1Error(f"llama-server not executable: {candidate}")
            return candidate
    found = shutil.which("llama-server")
    if not found:
        raise D1Error("llama-server not found: put it on PATH or set LLAMA_SERVER")
    return found


def find_d1_model(explicit: str | None = None, quant: str = "Q8_0") -> str:
    path = explicit or os.getenv("D1_MODEL")
    if path:
        if not Path(path).is_file():
            raise D1Error(f"d1 model not found: {path}")
        return path
    hub = os.getenv("HF_HUB_CACHE") or os.path.join(
        os.getenv("HF_HOME") or os.path.expanduser("~/.cache/huggingface"), "hub"
    )
    matches = sorted(glob.glob(os.path.join(hub, D1_REPO_DIR, "snapshots", "*",
                                            f"d1-2B-{quant}.gguf")))
    if not matches:
        raise D1Error(
            f"d1-2B-{quant}.gguf not in {hub}: run "
            f"`hf download LiquidAI/d1-2B-GGUF d1-2B-{quant}.gguf` or set D1_MODEL"
        )
    return matches[-1]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


_default_router: Router | None = None


def get_default_router() -> Router:
    """The process-wide router used by Agent(model="auto"): one d1 server per process."""
    global _default_router
    if _default_router is None:
        _default_router = D1Router()
    return _default_router


def set_default_router(router: Router | None) -> None:
    global _default_router
    _default_router = router
