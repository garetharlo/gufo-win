#!/usr/bin/env python3
"""
check-local-paths.py — Refuse personal machine paths in what would be published.

The tracked tree must stay reproducible anywhere: notes and scripts name
directories with placeholders (%TEMP%, C:\\dev, ...) instead of one account name
and one workstation layout. This check flags

  * this account's Windows user-profile path (a drive letter, then
    `\\Users\\<name>` or `/Users/<name>`), and
  * any path under a parent directory of this checkout,

in every file version a push would publish (--pushed, used by the pre-push
hook), in staged files (--staged, used by the pre-commit hook), in named files,
or in every tracked file. Either slash counts, and UTF-16 companions
(PowerShell redirection writes its logs that way) are read by dropping the NUL
bytes. Other accounts stay allowed: upstream sources carry fixtures with their
own profile paths.
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


def workspace_patterns(root: Path) -> list:
    """This account's profile plus the Windows ancestors of this checkout, either slash."""
    names = {Path(os.environ.get(variable, "")).name for variable in ("USERPROFILE", "HOME")}
    patterns = [
        re.compile(r"[A-Za-z]:[\\/]Users[\\/]" + re.escape(name) + r"[\\/]", re.IGNORECASE)
        for name in sorted(names)
        if name
    ]
    for parent in root.parents:
        text = str(parent)
        if re.fullmatch(r"[A-Za-z]:[\\/]", text):
            continue  # a drive root is nobody's personal directory
        if re.fullmatch(r"[A-Za-z]:[\\/]Users", text, re.IGNORECASE):
            continue  # the shared users folder, not one person's directory
        if not re.match(r"[A-Za-z]:[\\/]", text):
            continue  # a POSIX path, or the relative tail of the checkout
        for form in dict.fromkeys([text, text.replace("\\", "/")]):
            patterns.append(re.compile(re.escape(form) + r"[\\/]", re.IGNORECASE))
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


def findings_for(name: str, data: bytes, patterns: list) -> list:
    # latin-1 never fails on odd bytes; dropping the NUL bytes turns a
    # PowerShell-redirected UTF-16 log back into searchable ASCII.
    text = data.replace(b"\x00", b"").decode("latin-1") if b"\x00" in data else data.decode("latin-1")
    matches = []
    for pattern in patterns:
        matches.extend(pattern.finditer(text))
    end = -1
    findings = []
    # Longest form first, so one path is reported once even when two patterns cover it.
    for match in sorted(matches, key=lambda found: (found.start(), -found.end())):
        if match.start() < end:
            continue
        end = match.end()
        findings.append(f"{name}:{text.count(chr(10), 0, match.start()) + 1}: {match.group(0)}")
    return findings


def pushed_findings(root: Path, rev: str, patterns: list) -> list:
    """Every file version in the commits this push would publish."""
    log = subprocess.run(
        ["git", "-C", str(root), "log", "--format=@@%H", "--name-only", rev, "--not", "--remotes"],
        capture_output=True,
    ).stdout.decode("utf-8", "replace")
    refs = []
    commit = None
    for line in log.splitlines():
        if line.startswith("@@"):
            commit = line[2:]
        elif line.strip():
            refs.append(f"{commit}:{line}")
    if not refs:
        return []

    # run(input=...) writes and reads at the same time; feeding a long history
    # through a hand-written stdin.write deadlocks once the batch output fills
    # the pipe.
    out = subprocess.run(
        ["git", "-C", str(root), "cat-file", "--batch"],
        input="\n".join(refs).encode() + b"\n",
        capture_output=True,
    ).stdout
    findings = []
    seen = set()
    at = 0
    for ref in refs:
        line_end = out.find(b"\n", at)
        header = out[at:line_end]
        at = line_end + 1
        if b" blob " not in header:
            continue  # missing object, or not a blob
        size = int(header.split()[2])
        data = out[at : at + size]
        at += size + 1  # the batch adds a newline after each object
        blob = header.split()[0]
        if blob in seen:
            continue
        seen.add(blob)
        findings.extend(findings_for(f"{ref[:7]}{ref[40:]}", data, patterns))
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="repository root (default: .)")
    parser.add_argument("--staged", action="store_true", help="check staged files only")
    parser.add_argument("--pushed", metavar="REV", help="check the commits REV would publish")
    parser.add_argument("files", nargs="*", help="files to check (default: all tracked)")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    patterns = workspace_patterns(root)
    if args.pushed:
        findings = pushed_findings(root, args.pushed, patterns)
    else:
        findings = []
        for name in names_to_check(root, args):
            try:
                data = (root / name).read_bytes()
            except OSError:
                continue
            findings.extend(findings_for(name, data, patterns))

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
