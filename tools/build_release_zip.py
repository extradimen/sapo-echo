#!/usr/bin/env python3
"""Export a clean ZIP from one committed Git revision."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", default="HEAD")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefix", default="sapo-echo/")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "git", "archive", "--format=zip", f"--prefix={args.prefix}",
            f"--output={args.output.resolve()}", args.revision,
        ],
        cwd=ROOT,
        check=True,
    )
    print(args.output.resolve())


if __name__ == "__main__":
    main()
