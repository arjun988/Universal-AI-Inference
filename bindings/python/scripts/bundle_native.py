#!/usr/bin/env python3
"""Copy built uaii_capi shared library into uaii/_native for wheel packaging.

Usage (from repo root, after CMake build):
  python bindings/python/scripts/bundle_native.py [--build-dir build]
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-dir", default="build")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[3]
    build = (root / args.build_dir).resolve()
    dest = Path(__file__).resolve().parents[1] / "uaii" / "_native"
    dest.mkdir(parents=True, exist_ok=True)

    names = ["uaii_capi.dll", "libuaii_capi.dll", "libuaii_capi.so", "libuaii_capi.dylib"]
    found: Path | None = None
    for p in build.rglob("*"):
        if p.is_file() and p.name in names:
            found = p
            break
    if found is None:
        print(f"uaii_capi shared library not found under {build}", file=sys.stderr)
        return 1
    target = dest / found.name
    shutil.copy2(found, target)
    print(f"bundled {found} -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
