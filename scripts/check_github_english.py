#!/usr/bin/env python3
"""Reject CJK text in public GitHub-facing repository content."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable


CJK_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
PUBLIC_TEXT_SUFFIXES = {".md", ".txt", ".yml", ".yaml"}


def find_cjk(label: str, text: str) -> list[str]:
    failures = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if CJK_PATTERN.search(line):
            failures.append(f"{label}:{line_number}: {line.strip()}")
    return failures


def public_files(repository: Path) -> Iterable[Path]:
    readme = repository / "README.md"
    if readme.is_file():
        yield readme

    github_directory = repository / ".github"
    if github_directory.is_dir():
        for path in sorted(github_directory.rglob("*")):
            if path.is_file() and path.suffix.lower() in PUBLIC_TEXT_SUFFIXES:
                yield path


def gh_json(arguments: list[str]) -> dict:
    result = subprocess.run(
        ["gh", *arguments],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return json.loads(result.stdout)


def check_repository_files(repository: Path) -> list[str]:
    failures = []
    for path in public_files(repository):
        text = path.read_text(encoding="utf-8")
        failures.extend(find_cjk(str(path.relative_to(repository)), text))
    return failures


def check_remote_metadata(release_tag: str) -> list[str]:
    failures = []

    repository_data = gh_json(["repo", "view", "--json", "description"])
    failures.extend(find_cjk("repository.description", repository_data.get("description") or ""))

    release_data = gh_json(
        ["release", "view", release_tag, "--json", "name,body,assets"]
    )
    failures.extend(find_cjk(f"release[{release_tag}].name", release_data.get("name") or ""))
    failures.extend(find_cjk(f"release[{release_tag}].body", release_data.get("body") or ""))
    for asset in release_data.get("assets") or []:
        asset_name = asset.get("name") or ""
        asset_label = asset.get("label") or ""
        failures.extend(find_cjk(f"release[{release_tag}].asset.name", asset_name))
        failures.extend(find_cjk(f"release[{release_tag}].asset.label", asset_label))

    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--release-tag")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repository = args.repository.resolve()

    failures = check_repository_files(repository)
    if args.release_tag:
        failures.extend(check_remote_metadata(args.release_tag))

    if failures:
        print("GitHub English-only check failed:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1

    print("GitHub English-only check passed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, subprocess.CalledProcessError, UnicodeError, json.JSONDecodeError) as exc:
        print(f"GitHub English-only check could not complete: {exc}", file=sys.stderr)
        raise SystemExit(1)
