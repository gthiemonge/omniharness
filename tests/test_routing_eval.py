"""The labelled routing eval against the real d1 model (skipped without one).

    LLAMA_SERVER=/path/to/llama-server pytest -m d1     # or D1_URL=http://...
"""

import importlib.util
import os
from pathlib import Path

import pytest

from omniharness import D1Router

pytestmark = [
    pytest.mark.d1,
    pytest.mark.skipif(not (os.getenv("D1_URL") or os.getenv("LLAMA_SERVER")),
                       reason="needs d1: set D1_URL or LLAMA_SERVER"),
]

ROOT = Path(__file__).resolve().parent.parent


def _eval_module():
    spec = importlib.util.spec_from_file_location("routing_eval",
                                                  ROOT / "scripts" / "routing_eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_choice_routing_matches_labels():
    ev = _eval_module()
    router = D1Router()
    try:
        res = ev.evaluate(router, ev.load(ev.DEFAULT_FILE), verbose=False)
    finally:
        router.close()
    # Measured: 29/34 correct, 0 too cheap (d1-2B Q8_0, llama.cpp v0.8.0).
    # Too cheap risks quality; too expensive only costs money.
    assert res["correct"] >= 0.8 * res["n"], res["confusion"]
    assert res["under"] <= 1, res["confusion"]
