#!/usr/bin/env python3
"""Create localized course-image placeholders and rewrite localized docs.

This intentionally does not call any image API. It gives English/Japanese docs
locale-specific image files first, so the site stays usable while final image2
assets are generated later and overwrite the placeholders.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COURSE_IMG_ROOT = PROJECT_ROOT / "static" / "img" / "course"
DOCS_I18N_ROOT = PROJECT_ROOT / "i18n"
IMAGE_RE = re.compile(r"!\[([^\]]*)]\((/img/course/[^)\s]+)([^)]*)\)")
HOMEPAGE_BASE_RE = re.compile(r"^homepage-ai-history-comic-(\d{2}-.+\.png)$")
HOMEPAGE_LOCALIZED_RE = re.compile(r"^homepage-ai-history-comic-(en|ja)-(\d{2}-.+\.png)$")


def localized_name(filename: str, locale: str) -> str:
    path = Path(filename)
    localized_homepage_match = HOMEPAGE_LOCALIZED_RE.match(path.name)
    if localized_homepage_match:
        return f"homepage-ai-history-comic-{locale}-{localized_homepage_match.group(2)}"
    homepage_match = HOMEPAGE_BASE_RE.match(path.name)
    if homepage_match:
        return f"homepage-ai-history-comic-{locale}-{homepage_match.group(1)}"
    if path.stem.endswith(f"-{locale}"):
        return filename
    return f"{path.stem}-{locale}{path.suffix}"


def is_localized(filename: str, locale: str) -> bool:
    if HOMEPAGE_LOCALIZED_RE.match(filename):
        return filename.startswith(f"homepage-ai-history-comic-{locale}-")
    return Path(filename).stem.endswith(f"-{locale}")


def iter_locale_markdown(locale: str) -> list[Path]:
    root = DOCS_I18N_ROOT / locale / "docusaurus-plugin-content-docs" / "current"
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in {".md", ".mdx"})


def image_dimensions(path: Path) -> tuple[int, int]:
    try:
        from PIL import Image
    except ImportError:
        return 1536, 1024
    try:
        with Image.open(path) as image:
            return image.size
    except Exception:
        return 1536, 1024


def create_placeholder(filename: str, locale: str, alt: str, source_filename: str, overwrite: bool) -> None:
    from generate_course_images import write_placeholder

    output_path = COURSE_IMG_ROOT / filename
    if output_path.exists() and not overwrite:
        return

    source_path = COURSE_IMG_ROOT / source_filename
    width, height = image_dimensions(source_path)
    language_name = {"en": "English", "ja": "Japanese"}.get(locale, locale)
    localized_title = alt.strip() or f"{language_name} localized image"
    job: dict[str, Any] = {
        "filename": filename,
        "size": f"{width}x{height}",
        "title": localized_title[:80],
        "alt": f"{language_name} placeholder. Replace with final image2 asset. Source: {source_filename}",
        "preview_label": f"{language_name} Placeholder",
    }
    write_placeholder(COURSE_IMG_ROOT, job, overwrite=overwrite)


def rewrite_locale(locale: str, overwrite: bool, dry_run: bool) -> dict[str, int]:
    changed_files = 0
    rewritten_refs = 0
    created_placeholders = 0
    missing_sources = 0
    seen_targets: set[str] = set()

    for md_path in iter_locale_markdown(locale):
        original = md_path.read_text(encoding="utf-8", errors="ignore")

        def replace(match: re.Match[str]) -> str:
            nonlocal rewritten_refs, created_placeholders, missing_sources
            alt, url, suffix = match.groups()
            filename = url.removeprefix("/img/course/")
            if is_localized(filename, locale):
                return match.group(0)

            target = localized_name(filename, locale)
            source_path = COURSE_IMG_ROOT / filename
            if not source_path.exists():
                missing_sources += 1
                return match.group(0)

            rewritten_refs += 1
            if target not in seen_targets:
                seen_targets.add(target)
                if not (COURSE_IMG_ROOT / target).exists() or overwrite:
                    created_placeholders += 1
                    if not dry_run:
                        create_placeholder(target, locale, alt, filename, overwrite=overwrite)
            return f"![{alt}](/img/course/{target}{suffix})"

        updated = IMAGE_RE.sub(replace, original)
        if updated != original:
            changed_files += 1
            if not dry_run:
                md_path.write_text(updated, encoding="utf-8")

    return {
        "changed_files": changed_files,
        "rewritten_refs": rewritten_refs,
        "created_placeholders": created_placeholders,
        "missing_sources": missing_sources,
        "unique_targets": len(seen_targets),
    }


def run_audit() -> None:
    subprocess.run([sys.executable, "scripts/audit_image_i18n.py", "--skip-ocr"], cwd=PROJECT_ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--locales", nargs="+", default=["en", "ja"])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-audit", action="store_true")
    args = parser.parse_args()

    summary: dict[str, dict[str, int]] = {}
    for locale in args.locales:
        summary[locale] = rewrite_locale(locale, overwrite=args.overwrite, dry_run=args.dry_run)

    for locale, counts in summary.items():
        print(f"{locale}: {counts}")

    if not args.dry_run and not args.skip_audit:
        run_audit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
