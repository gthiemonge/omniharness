"""
Built-in filesystem / shell / web tools.

The Claude Agent SDK ships Read/Write/Bash/Grep/Glob/WebFetch natively, so the
claude provider uses the SDK's versions of these (claude_native=True). For
providers without their own toolset (OpenAI and OpenAI-compatible endpoints),
these Python implementations are executed locally with matching names and
semantics, so the same prompt works with either provider.

Relative paths resolve against the working directory. Writes are allowed
anywhere the process can write, like the Claude SDK's acceptEdits mode: run
untrusted prompts in a container or VM.
"""

import glob as _glob
import os
import re
import shutil
import subprocess
import urllib.request
from pathlib import Path

from .base import Tool, ToolContext, tool


def _resolve(ctx: ToolContext, path: str) -> Path:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = Path(ctx.work_dir) / p
    return p


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated, {len(text) - limit} more chars]"


@tool(name="Read", claude_native=True,
      description="Read a file from the filesystem. Returns line-numbered content.")
def read(ctx: ToolContext, file_path: str, offset: int | None = None,
         limit: int | None = None) -> str:
    """
    Args:
        file_path: Path to the file (absolute or relative to the working directory).
        offset: 1-based line to start from.
        limit: Max number of lines to read.
    """
    path = _resolve(ctx, file_path)
    try:
        with open(path, "r", errors="replace") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return f"Error: file not found: {path}"
    except IsADirectoryError:
        return f"Error: {path} is a directory"

    start = (offset - 1) if offset and offset > 0 else 0
    end = (start + limit) if limit else len(lines)
    numbered = [
        f"{i}\t{line.rstrip(chr(10))}"
        for i, line in enumerate(lines[start:end], start=start + 1)
    ]
    return _truncate("\n".join(numbered), ctx.max_output_chars)


@tool(name="Write", claude_native=True,
      description="Write (or overwrite) a file with the given content.")
def write(ctx: ToolContext, file_path: str, content: str) -> str:
    """
    Args:
        file_path: Path to write (absolute or relative to the working directory).
        content: Full file content to write.
    """
    path = _resolve(ctx, file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return f"Wrote {len(content)} chars to {path}"


@tool(name="Bash", claude_native=True,
      description="Run a shell command in the working directory and return its output.")
def bash(ctx: ToolContext, command: str, description: str = "", timeout: int = 300) -> str:
    """
    Args:
        command: The shell command to run.
        description: One short sentence on why you run it.
        timeout: Timeout in seconds.
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=ctx.work_dir,
            env={**os.environ, **ctx.env},
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {timeout}s"

    parts = []
    if result.stdout:
        parts.append(result.stdout)
    if result.stderr:
        parts.append(f"[stderr]\n{result.stderr}")
    if result.returncode != 0:
        parts.append(f"[exit code: {result.returncode}]")
    return _truncate("\n".join(parts) if parts else "(no output)", ctx.max_output_chars)


@tool(name="Grep", claude_native=True,
      description="Search file contents for a regular expression. Returns file:line:match rows.")
def grep(ctx: ToolContext, pattern: str, path: str | None = None,
         glob: str | None = None) -> str:
    """
    Args:
        pattern: Regular expression to search for.
        path: Directory or file to search (defaults to the working directory).
        glob: Only search files matching this glob, e.g. '*.py'.
    """
    search_path = str(_resolve(ctx, path)) if path else ctx.work_dir

    if shutil.which("rg"):
        cmd = ["rg", "--line-number", "--no-heading", "--color", "never"]
        if glob:
            cmd += ["--glob", glob]
        cmd += [pattern, search_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=False)
        output = result.stdout or ""
        if not output and result.returncode not in (0, 1):
            output = result.stderr or ""
        return _truncate(output or "(no matches)", ctx.max_output_chars)

    # Fallback: pure-Python recursive search.
    try:
        regex = re.compile(pattern)
    except re.error as e:
        return f"Error: invalid pattern: {e}"

    matches: list[str] = []
    files = [search_path] if os.path.isfile(search_path) else None
    if files is None:
        files = []
        for root, dirs, names in os.walk(search_path):
            dirs[:] = [d for d in dirs if d != ".git"]
            files += [os.path.join(root, n) for n in names
                      if not glob or Path(n).match(glob)]
    for fpath in files:
        try:
            with open(fpath, "r", errors="ignore") as f:
                for lineno, line in enumerate(f, 1):
                    if regex.search(line):
                        matches.append(f"{fpath}:{lineno}:{line.rstrip()}")
        except OSError:
            continue
    return _truncate("\n".join(matches) if matches else "(no matches)", ctx.max_output_chars)


@tool(name="Glob", claude_native=True,
      description="Find files matching a glob pattern (supports ** for recursion).")
def glob(ctx: ToolContext, pattern: str, path: str | None = None) -> str:
    """
    Args:
        pattern: Glob pattern, e.g. '**/*.py'.
        path: Base directory (defaults to the working directory).
    """
    base = str(_resolve(ctx, path)) if path else ctx.work_dir
    results = sorted(_glob.glob(os.path.join(base, pattern), recursive=True))
    return _truncate("\n".join(results) if results else "(no matches)", ctx.max_output_chars)


@tool(name="WebFetch", claude_native=True,
      description="Fetch the contents of a URL and return the text.")
def web_fetch(ctx: ToolContext, url: str, prompt: str = "") -> str:
    """
    Args:
        url: The URL to fetch.
        prompt: What you are looking for in the page.
    """
    req = urllib.request.Request(url, headers={"User-Agent": "omniharness"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = resp.read().decode(errors="replace")
    # Some JSON APIs (Gerrit, ...) prepend an XSSI guard like )]}' — strip it so
    # the model gets clean JSON.
    stripped = raw.lstrip()
    if stripped.startswith(")]}'"):
        raw = stripped.split("\n", 1)[1] if "\n" in stripped else stripped[4:]
    return _truncate(raw, ctx.max_output_chars)


BUILTIN_TOOLS: dict[str, Tool] = {t.name: t for t in (read, write, bash, grep, glob, web_fetch)}

# Handy presets.
READ_ONLY_TOOLS: list[Tool] = [read, grep, glob]
CODING_TOOLS: list[Tool] = [read, write, bash, grep, glob]
ALL_TOOLS: list[Tool] = list(BUILTIN_TOOLS.values())


def get_builtin(name: str) -> Tool:
    try:
        return BUILTIN_TOOLS[name]
    except KeyError:
        available = ", ".join(BUILTIN_TOOLS)
        raise ValueError(f"Unknown built-in tool '{name}'. Available: {available}") from None
