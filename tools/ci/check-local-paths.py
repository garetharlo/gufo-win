#!/usr/bin/env python3
"""
check-local-paths.py — Refuse personal machine paths in text that would be published.

The tracked tree must stay reproducible anywhere: notes and scripts name
directories with placeholders (%TEMP%, C:\\dev, ...) instead of one account name
and one workstation layout. This check flags

  * Windows user-profile paths (a drive letter, then \\Users\\<name>), and
  * any path under a parent directory of this checkout,

in staged files (--staged, used by the pre-commit hook), in named files, or in
every tracked file.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path


def workspace_patterns(root: Path) -> list:
    """Profile paths plus the Windows ancestors of this checkout."""
    patterns = [re.compile(r"[A-Za-z]:\\Users\\[^\\\s\"'`<>|]+", re.IGNORECASE)]
    for parent in root.parents:
        text = str(parent)
        if re.fullmatch(r"[A-Za-z]:[\\/]", text):
            continue  # a drive root is nobody's personal directory
        if not re.match(r"[A-Za-z]:\\", text):
            continue  # a POSIX path, or the relative tail of the checkout
        patterns.append(re.compile(re.escape(text) + r"[\\/]", re.IGNORECASE))
    return patterns


def names_to_check(root: Path, args) -> list:
    if args.staged:
        argv = ["diff", "--cached", "--name-only", "--diff-filter=ACM", "-z"]
    elif args.files:
        return args.files
    else:
        argv = ["ls-files", "-z"]
    out = subprocess.run(["git", "-C", str(root), *argv], capture_output=True)
    return [name for name in out.stdout.decode("utf-8", "replace").split("\0") if name]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="repository root (default: .)")
    parser.add_argument("--staged", action="store_true", help="check staged files only")
    parser.add_argument("files", nargs="*", help="files to check (default: all tracked)")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    patterns = workspace_patterns(root)
    findings = []
    for name in names_to_check(root, args):
        try:
            data = (root / name).read_bytes()
        except OSError:
            continue
        if b"\x00" in data[:8192]:
            continue
        text = data.decode("utf-8", "replace")
        matches = []
        for pattern in patterns:
            matches.extend(pattern.finditer(text))
        end = -1
        # Longest form first, so one path is reported once even when two patterns cover it.
        for match in sorted(matches, key=lambda found: (found.start(), -found.end())):
            if match.start() < end:
                continue
            end = match.end()
            findings.append(f"{name}:{text.count(chr(10), 0, match.start()) + 1}: {match.group(0)}")

    if findings:
        print(f"[check-local-paths] FAILED: {len(findings)} personal machine path(s):")
        for finding in findings:
            print(f"  - {finding}")
        print("Replace them with a placeholder such as %TEMP% or C:\\dev.")
        return 1
    print(f"[check-local-paths] PASSED: no personal machine paths in {len(patterns)} pattern(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
