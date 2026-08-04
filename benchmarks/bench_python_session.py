#!/usr/bin/env python3
"""Simple Python session micro-benchmark (Phase 7)."""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bindings" / "python"))

import uaii  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=30)
    args = ap.parse_args()

    ir = ROOT / "examples" / "ir" / "toy_mlp.uaii.json"
    times = []
    for _ in range(args.iters):
        s = uaii.Session.from_path(str(ir), weight_init="ones", fusion=True)
        s.set_tensor("x", [1.0, 2.0, 3.0, 4.0])
        t0 = time.perf_counter()
        s.run()
        times.append((time.perf_counter() - t0) * 1000.0)

    print(f"iters={args.iters}")
    print(f"mean_ms={statistics.mean(times):.4f}")
    print(f"median_ms={statistics.median(times):.4f}")
    print(f"stdev_ms={statistics.pstdev(times):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
