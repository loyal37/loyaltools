#!/usr/bin/env python3
"""Build and validate an installable LoyalTools Blender add-on archive."""

from __future__ import annotations

import argparse
import ast
import hashlib
import subprocess
import sys
import zipfile
from pathlib import Path


PACKAGE_ROOT = "LoyalTools"


def build_archive(repository: Path, git_ref: str, output: Path) -> None:
    if not PACKAGE_ROOT.isidentifier():
        raise ValueError(f"Invalid Python package name: {PACKAGE_ROOT}")

    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "git",
        "-C",
        str(repository),
        "archive",
        "--format=zip",
        f"--prefix={PACKAGE_ROOT}/",
        f"--output={output.resolve()}",
        git_ref,
    ]
    subprocess.run(command, check=True)


def validate_archive(output: Path) -> tuple[int, str]:
    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()
        expected_init = f"{PACKAGE_ROOT}/__init__.py"

        if expected_init not in names:
            raise ValueError(f"Archive is missing {expected_init}")
        if any(not name.startswith(f"{PACKAGE_ROOT}/") for name in names):
            raise ValueError("Archive contains entries outside the package root")
        if any("/__pycache__/" in name or name.endswith((".pyc", ".pyo")) for name in names):
            raise ValueError("Archive contains Python cache files")

        for name in names:
            if name.endswith(".py"):
                ast.parse(archive.read(name), filename=name)

    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    return len(names), digest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--git-ref", default="HEAD")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repository = args.repository.resolve()
    output = args.output.resolve()

    build_archive(repository, args.git_ref, output)
    entry_count, digest = validate_archive(output)
    print(f"Built: {output}")
    print(f"Entries: {entry_count}")
    print(f"SHA256: {digest}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, subprocess.CalledProcessError, ValueError, zipfile.BadZipFile) as exc:
        print(f"Build failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
