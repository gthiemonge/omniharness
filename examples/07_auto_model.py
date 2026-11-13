#!/usr/bin/env python3
"""
model="auto": a local decision model picks the model for each run.

Liquid AI's d1-2B (served by llama-server) answers "Which assistant should
handle this task?" (fast / capable / expert), which maps to the quick /
standard / thorough preset of the agent's provider. Once the server is up, a
rating takes under a second on CPU (~0.7-0.9 s measured on a 20-thread
machine); starting the server takes a few seconds.

Setup (once):
    hf download LiquidAI/d1-2B-GGUF d1-2B-Q8_0.gguf      # ~1.7 GB
    # llama-server (llama.cpp v0.8.0+) on PATH, or LLAMA_SERVER=/path/to/llama-server
    # or reuse a running one: export D1_URL=http://127.0.0.1:8080

    # Only rate prompts, no LLM call: check the routing before spending tokens
    python examples/07_auto_model.py --rate-only
    python examples/07_auto_model.py --rate-only "Refactor our auth module to support OIDC"
    python examples/07_auto_model.py --rate-only --method complexity

    # Run an agent with model="auto"
    python examples/07_auto_model.py "What is the capital of Australia?"
    AGENT_PROVIDER=openai python examples/07_auto_model.py "Design a rate limiter for a public API"

Subagents can use it too: each delegated task is routed on its own.
    python examples/06_subagents.py . --worker-model auto
To check the routing on a labelled set: python scripts/routing_eval.py
"""

import argparse
import logging
import sys

from omniharness import Agent, D1Router

SAMPLES = [
    "What is the capital of Australia?",
    "Write a Python function that checks whether a string is a palindrome.",
    "Explain the difference between TCP and UDP, with an example of when to use each.",
    "Implement an LRU cache in Python with O(1) operations and write unit tests for it.",
    ("Design a multi-region, strongly consistent key-value store: replication, failure "
     "detection, membership changes and client routing, with the latency trade-offs."),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("prompt", nargs="*", help="Prompt(s); default: built-in samples")
    parser.add_argument("--rate-only", action="store_true",
                        help="Print the routing decision for each prompt and exit")
    parser.add_argument("--d1-url", help="Running llama-server with d1-2B (env D1_URL)")
    parser.add_argument("--method", choices=["choice", "complexity"], default="choice")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    router = D1Router(args.d1_url, method=args.method)
    try:
        if args.rate_only:
            for prompt in args.prompt or SAMPLES:
                d = router.route_sync(prompt)
                if d.source == "fallback":
                    print(f"fallback -> {d.preset}: {d.reason}")
                    return 1
                probs = "  ".join(f"{p:.2f}" for p in d.probabilities.values())
                print(f"{d.preset:9s} {d.label:13s} [{probs}]  "
                      f"{d.elapsed_ms:4.0f} ms  {prompt[:70]}")
            return 0

        prompt = " ".join(args.prompt) or SAMPLES[0]
        agent = Agent(model="auto", router=router, system_prompt="Answer concisely.")
        result = agent.run_sync(prompt)
        return 0 if result.success else 1
    finally:
        router.close()


if __name__ == "__main__":
    sys.exit(main())
