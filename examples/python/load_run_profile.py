#!/usr/bin/env python3
"""Phase 7 exit-criteria example: load IR, run inference, profile from Python."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bindings" / "python"))

import uaii  # noqa: E402


def main() -> int:
    ir = ROOT / "examples" / "ir" / "toy_mlp.uaii.json"
    if not ir.is_file():
        print(f"missing IR fixture: {ir}", file=sys.stderr)
        return 1

    print("uaii", uaii.version(), "c_api", uaii.c_api_version())

    trace = ROOT / "uaii_py_profile.json"
    session = uaii.Session.from_path(
        str(ir),
        backend="cpu",
        weight_init="ones",
        profile=True,
        trace_path=str(trace),
        fusion=True,
    )
    session.set_tensor("x", [1.0, 2.0, 3.0, 4.0])
    session.run()
    out = session.get_tensor("y_prob")
    print("y_prob", out)
    print("profile", session.profile_summary())
    print("trace", trace)
    print("stats", session.debug_stats())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
