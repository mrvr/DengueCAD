#!/usr/bin/env python3
"""Compute the next semver release tag from existing git tags + VERSION baseline."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION_FILE = ROOT / "VERSION"
TAG_RE = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)$")


def read_baseline() -> tuple[int, int, int]:
    text = VERSION_FILE.read_text(encoding="utf-8").strip()
    m = TAG_RE.match(text)
    if not m:
        raise SystemExit(f"VERSION must be MAJOR.MINOR.PATCH, got {text!r}")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def list_tags() -> list[tuple[int, int, int]]:
    proc = subprocess.run(
        ["git", "tag", "--list", "v*"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    versions: list[tuple[int, int, int]] = []
    for line in proc.stdout.splitlines():
        m = TAG_RE.match(line.strip())
        if m:
            versions.append((int(m.group(1)), int(m.group(2)), int(m.group(3))))
    return versions


def next_version(bump: str = "patch") -> str:
    base = read_baseline()
    tags = list_tags()
    # First release: publish the baseline VERSION as-is.
    if not tags:
        return f"{base[0]}.{base[1]}.{base[2]}"
    current = max([base, *tags])
    major, minor, patch = current
    if bump == "major":
        major, minor, patch = major + 1, 0, 0
    elif bump == "minor":
        minor, patch = minor + 1, 0
    else:
        patch += 1
    return f"{major}.{minor}.{patch}"


def write_version(version: str) -> None:
    VERSION_FILE.write_text(version + "\n", encoding="utf-8")
    init_path = ROOT / "denguecad" / "__init__.py"
    text = init_path.read_text(encoding="utf-8")
    updated, n = re.subn(
        r'__version__\s*=\s*["\'][^"\']+["\']',
        f'__version__ = "{version}"',
        text,
        count=1,
    )
    if n != 1:
        raise SystemExit("Could not update denguecad/__init__.py __version__")
    init_path.write_text(updated, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bump", choices=("patch", "minor", "major"), default="patch")
    parser.add_argument("--write", action="store_true", help="Update VERSION and __init__.py")
    parser.add_argument("--tag", action="store_true", help="Print v-prefixed tag")
    args = parser.parse_args()
    version = next_version(args.bump)
    if args.write:
        write_version(version)
    print(f"v{version}" if args.tag else version)
    return 0


if __name__ == "__main__":
    sys.exit(main())
