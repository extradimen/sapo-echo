#!/usr/bin/env python3
"""Verify every file listed in SHA256SUMS and reject missing payload files."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "SHA256SUMS"
IGNORED = {"MANIFEST.csv", "SHA256SUMS"}
IGNORED_DIRECTORIES = {".git", "__pycache__", "build", "dist", ".venv"}


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def main() -> None:
    expected: dict[str, str] = {}
    for line in INDEX.read_text(encoding="utf-8").splitlines():
        checksum, relative = line.split("  ", 1)
        expected[relative] = checksum

    actual = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file()
        and path.name not in IGNORED
        and not IGNORED_DIRECTORIES.intersection(path.parts)
    }
    missing = sorted(set(expected) - actual)
    unlisted = sorted(actual - set(expected))
    mismatched = sorted(
        relative
        for relative in set(expected) & actual
        if digest(ROOT / relative) != expected[relative]
    )
    if missing or unlisted or mismatched:
        raise SystemExit(
            f"verification failed: missing={missing}, unlisted={unlisted}, "
            f"mismatched={mismatched}"
        )
    print(f"verified {len(expected)} files")


if __name__ == "__main__":
    main()
