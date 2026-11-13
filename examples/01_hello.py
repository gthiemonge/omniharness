#!/usr/bin/env python3
"""
Smallest possible agent: no tools, one question.

    python examples/01_hello.py
    AGENT_PROVIDER=openai python examples/01_hello.py "Explain RAFT in 3 sentences"
"""

import sys

from omniharness import Agent

prompt = " ".join(sys.argv[1:]) or "Give me three name ideas for a Python agent library."

agent = Agent(preset="quick", system_prompt="You are concise. Answer in plain text.")
result = agent.run_sync(prompt)
sys.exit(0 if result.success else 1)
