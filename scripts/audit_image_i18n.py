#!/usr/bin/env python3
"""Audit course images for localization work.

This script does not modify images. It maps Markdown image references to files
under public/img/course and reports whether English/Japanese localized variants
already exist. If tesseract is available, it also tries English OCR as a weak
signal for images that likely contain only English text.

Chinese OCR language data is not bundled by default on this machine, so images
without OCR text are still marked "needs_visual_review" rather than "safe".
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = PROJECT_ROOT / "src" / "content" / "docs"
COURSE_IMG_ROOT = PROJECT_ROOT / "public" / "img" / "course"
REPORT_DIR = PROJECT_ROOT / "reports" / "course-images"
IMAGE_RE = re.compile(r"!\[[^\]]*]\((/img/course/[^)\s]+)[^)]*\)")
HOMEPAGE_BASE_RE = re.compile(r"^homepage-ai-history-comic-(\d{2}-.+\.png)$")
HOMEPAGE_LOCALIZED_RE = re.compile(r"^homepage-ai-history-comic-(en|ja)-(\d{2}-.+\.png)$")


def iter_markdown_files() -> list[Path]:
    roots = [
        DOCS_ROOT,
        DOCS_ROOT / "zh-cn",
        DOCS_ROOT / "ja",
    ]
    files: list[Path] = []
    for root in roots:
        if root.exists():
            files.extend(
                path
                for path in root.rglob("*")
                if path.is_file() and path.suffix.lower() in {".md", ".mdx"}
            )
    return sorted(files)


def referenced_images() -> dict[str, set[str]]:
    refs: dict[str, set[str]] = {}
    for md_path in iter_markdown_files():
        text = md_path.read_text(encoding="utf-8", errors="ignore")
        for match in IMAGE_RE.finditer(text):
            image_url = match.group(1)
            if image_url.startswith("/img/course/"):
                filename = image_url.removeprefix("/img/course/")
                refs.setdefault(filename, set()).add(md_path.relative_to(PROJECT_ROOT).as_posix())
    return refs


def variant_name(filename: str, locale: str) -> str:
    path = Path(filename)
    localized_homepage_match = HOMEPAGE_LOCALIZED_RE.match(path.name)
    if localized_homepage_match:
        return str(path.with_name(f"homepage-ai-history-comic-{locale}-{localized_homepage_match.group(2)}"))
    homepage_match = HOMEPAGE_BASE_RE.match(path.name)
    if homepage_match:
        return str(path.with_name(f"homepage-ai-history-comic-{locale}-{homepage_match.group(1)}"))
    stem = path.stem
    suffix = path.suffix
    if stem.endswith(f"-{locale}"):
        return filename
    return str(path.with_name(f"{stem}-{locale}{suffix}"))


def is_localized_variant(filename: str) -> bool:
    path = Path(filename)
    if HOMEPAGE_LOCALIZED_RE.match(path.name):
        return True
    return path.stem.endswith("-en") or path.stem.endswith("-ja")


def has_tesseract() -> bool:
    try:
        subprocess.run(["tesseract", "--version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False


def english_ocr(path: Path, enabled: bool) -> str:
    if not enabled:
        return ""
    with tempfile.NamedTemporaryFile(prefix="course-image-ocr-", suffix=".txt") as tmp:
        base = tmp.name.removesuffix(".txt")
        try:
            subprocess.run(
                ["tesseract", str(path), base, "-l", "eng", "--psm", "6"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=60,
            )
            txt_path = Path(base + ".txt")
            if txt_path.exists():
                return txt_path.read_text(encoding="utf-8", errors="ignore").strip()
        except Exception:
            return ""
    return ""


def classify(filename: str, ocr_text: str, en_exists: bool, ja_exists: bool) -> str:
    if is_localized_variant(filename):
        return "localized_variant"
    if en_exists and ja_exists:
        return "localized_variants_exist"
    if ocr_text:
        return "has_english_ocr_review_if_chinese_needed"
    return "needs_visual_review"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", default=str(REPORT_DIR / "image-i18n-audit.json"))
    parser.add_argument("--output-csv", default=str(REPORT_DIR / "image-i18n-audit.csv"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-ocr", action="store_true")
    args = parser.parse_args()

    refs = referenced_images()
    filenames = sorted(refs)
    if args.limit:
        filenames = filenames[: args.limit]

    ocr_enabled = has_tesseract() and not args.skip_ocr
    rows: list[dict[str, object]] = []
    for filename in filenames:
        image_path = COURSE_IMG_ROOT / filename
        if not image_path.exists():
            rows.append(
                {
                    "filename": filename,
                    "exists": False,
                    "status": "missing_source_image",
                    "references": sorted(refs[filename]),
                }
            )
            continue

        en_name = variant_name(filename, "en")
        ja_name = variant_name(filename, "ja")
        en_exists = (COURSE_IMG_ROOT / en_name).exists()
        ja_exists = (COURSE_IMG_ROOT / ja_name).exists()
        ocr_text = english_ocr(image_path, ocr_enabled)
        rows.append(
            {
                "filename": filename,
                "exists": True,
                "size_bytes": image_path.stat().st_size,
                "english_variant": en_name,
                "english_variant_exists": en_exists,
                "japanese_variant": ja_name,
                "japanese_variant_exists": ja_exists,
                "english_ocr_sample": " ".join(ocr_text.split())[:240],
                "status": classify(filename, ocr_text, en_exists, ja_exists),
                "references": sorted(refs[filename]),
            }
        )

    json_path = Path(args.output_json)
    csv_path = Path(args.output_csv)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    fieldnames = [
        "filename",
        "exists",
        "size_bytes",
        "english_variant",
        "english_variant_exists",
        "japanese_variant",
        "japanese_variant_exists",
        "status",
        "english_ocr_sample",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    counts: dict[str, int] = {}
    for row in rows:
        status = str(row["status"])
        counts[status] = counts.get(status, 0) + 1
    print(json.dumps({"images": len(rows), "counts": counts, "json": str(json_path), "csv": str(csv_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
