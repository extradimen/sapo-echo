#!/usr/bin/env python3
"""Regenerate the release inventory and SHA-256 checksum list."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {"MANIFEST.csv", "SHA256SUMS"}
EXCLUDED_DIRECTORIES = {".git", "__pycache__", "build", "dist", ".venv"}


def main() -> None:
    payload = sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and path.name not in EXCLUDED
        and not EXCLUDED_DIRECTORIES.intersection(path.parts)
    )
    rows: list[tuple[str, int, str]] = []
    checksums: list[str] = []
    for path in payload:
        relative = path.relative_to(ROOT).as_posix()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append((relative, path.stat().st_size, digest))
        checksums.append(f"{digest}  {relative}")

    with (ROOT / "MANIFEST.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["path", "bytes", "sha256"])
        writer.writerows(rows)
    (ROOT / "SHA256SUMS").write_text(
        "\n".join(checksums) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
