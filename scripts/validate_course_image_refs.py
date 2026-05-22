#!/usr/bin/env python3
"""Validate course image references before an Astro Starlight build.

The build already fails on broken Markdown images, but this script makes the
failure faster and easier to read. It also reports currently unused course
assets so image cleanup can stay visible without blocking normal commits.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_ROOT = PROJECT_ROOT / "public"
COURSE_IMAGE_DIR = PUBLIC_ROOT / "img" / "course"
GENERATION_ERRORS = PROJECT_ROOT / "reports" / "course-images" / "generation-errors.json"

SOURCE_ROOTS = [
    PROJECT_ROOT / "src" / "content" / "docs",
    PROJECT_ROOT / "src",
    PROJECT_ROOT / "README.md",
    PROJECT_ROOT / "astro.config.mjs",
    PROJECT_ROOT / "src" / "content.config.ts",
]

SOURCE_SUFFIXES = {".md", ".mdx", ".js", ".jsx", ".ts", ".tsx", ".json"}
COURSE_IMAGE_RE = re.compile(r"/img/course/[^)\s\"']+")


def iter_source_files() -> list[Path]:
    files: list[Path] = []
    for root in SOURCE_ROOTS:
        if not root.exists():
            continue
        if root.is_file():
            if root.suffix.lower() in SOURCE_SUFFIXES:
                files.append(root)
            continue
        files.extend(
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in SOURCE_SUFFIXES
        )
    return sorted(files)


def rel(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def find_refs() -> tuple[list[tuple[Path, str]], list[tuple[Path, str]]]:
    refs: list[tuple[Path, str]] = []
    missing: list[tuple[Path, str]] = []
    for source in iter_source_files():
        text = source.read_text(encoding="utf-8", errors="ignore")
        for match in COURSE_IMAGE_RE.finditer(text):
            url = match.group(0).rstrip(".,;`")
            refs.append((source, url))
            image_path = PUBLIC_ROOT / url.lstrip("/")
            if not image_path.exists():
                missing.append((source, url))
    return refs, missing


def load_generation_errors() -> list[object]:
    if not GENERATION_ERRORS.exists():
        return []
    data = json.loads(GENERATION_ERRORS.read_text(encoding="utf-8"))
    errors = data.get("errors", [])
    if not isinstance(errors, list):
        return [{"error": "reports/course-images/generation-errors.json has non-list errors"}]
    return errors


def find_unused(referenced_filenames: set[str]) -> list[Path]:
    if not COURSE_IMAGE_DIR.exists():
        return []
    return sorted(
        path
        for path in COURSE_IMAGE_DIR.iterdir()
        if path.is_file() and path.name not in referenced_filenames
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fail-on-unused",
        action="store_true",
        help="Treat unreferenced files in public/img/course as errors.",
    )
    parser.add_argument(
        "--show-unused",
        action="store_true",
        help="Print the full unused image list instead of only a count and sample.",
    )
    args = parser.parse_args()

    refs, missing = find_refs()
    generation_errors = load_generation_errors()
    referenced_filenames = {Path(url).name for _, url in refs}
    unused = find_unused(referenced_filenames)

    print(f"course_image_refs={len(refs)} missing={len(missing)} unused={len(unused)}")
    print(f"generation_errors={len(generation_errors)}")

    if missing:
        print("\nMissing course image references:")
        for source, url in missing:
            print(f"- {rel(source)}: {url}")

    if generation_errors:
        print("\nImage generation errors:")
        for error in generation_errors[:50]:
            print(f"- {error}")
        if len(generation_errors) > 50:
            print(f"... {len(generation_errors) - 50} more")

    if unused:
        print("\nUnused course image assets:")
        shown = unused if args.show_unused else unused[:50]
        for path in shown:
            print(f"- {rel(path)}")
        if not args.show_unused and len(unused) > len(shown):
            print(f"... {len(unused) - len(shown)} more")

    failed = bool(missing or generation_errors or (args.fail_on_unused and unused))
    if failed:
        return 1

    print("PASS course image references: all referenced course images exist")
    return 0


if __name__ == "__main__":
    sys.exit(main())
