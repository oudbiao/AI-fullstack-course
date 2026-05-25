#!/usr/bin/env python3
"""Generate a lightweight course quality dashboard for AI Roads.

This script is intentionally read-only. It checks the multilingual course tree for
mirror coverage, evidence blocks, folded answer blocks, image references, direct
PNG/JPG references, duplicate summaries, unbalanced details tags, common
localized visible-text residue patterns, English alt text in localized pages,
and advisory legacy-context terms.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "src" / "content" / "docs"
LOCALES = {
    "en": DOCS,
    "zh-cn": DOCS / "zh-cn",
    "ja": DOCS / "ja",
}

EVIDENCE_HEADINGS = {
    "en": "## Evidence to Keep",
    "zh-cn": "## 留下的证据",
    "ja": "## 残す証拠",
}

VISIBLE_RESIDUE_PATTERNS = [
    r"\bCore Evidence\b",
    r"\bPass Check\b",
    r"\bReference answers\b",
    r"\bProject Type\b",
    r"\bEvidence \|",
    r"\bAdvanced / Elective\b",
    r"\bDeep Dive\b",
]

LEGACY_CONTEXT_PATTERNS = [
    r"\bstudent\b",
    r"\bteacher\b",
    r"\bclassroom\b",
    r"\bcourseware\b",
    r"\bdiscount\b",
    r"\bbanana\b",
    r"\bAlice\b",
    r"\bBob\b",
    r"\bBMI\b",
    r"\bshopping\b",
    r"\bweather\b",
    r"\bmovie\b",
]

# These matches are known intentional technical/product contexts, not old classroom
# examples. The dashboard reports legacy terms as advisory samples only, but this
# allowlist keeps the samples useful and prevents noisy maintenance churn.
LEGACY_CONTEXT_ALLOWLIST = [
    r"ch12-multimodal/.*",
    r"ch13-open-source-llm/.*",
    r"ch08-rag/ch05-projects/04-courseware-assistant\.md",
    r"appendix/.*",
]

MARKDOWN_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
HTML_IMAGE_RE = re.compile(r"<img\b[^>]*\bsrc=[\"']([^\"']+)[\"'][^>]*>", re.I)
CJK_RE = re.compile(r"[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]")

TECHNICAL_ALT_TERMS = {
    "ai",
    "api",
    "aigc",
    "agent",
    "ann",
    "autodl",
    "axes",
    "axis",
    "batch",
    "bert",
    "cache",
    "checkpoint",
    "cli",
    "colab",
    "cpu",
    "csv",
    "ctc",
    "cuda",
    "dataloader",
    "dataset",
    "decision",
    "diffusion",
    "docker",
    "embedding",
    "encoder",
    "estimator",
    "fastapi",
    "feature",
    "flowchart",
    "gan",
    "gpt",
    "gpu",
    "head",
    "html",
    "http",
    "https",
    "hybrid",
    "input_ids",
    "iou",
    "jupyter",
    "json",
    "kaggle",
    "kernel",
    "latent",
    "learned",
    "loss",
    "llm",
    "lora",
    "mask",
    "matplotlib",
    "mel_spectrogram",
    "mini",
    "mha",
    "model",
    "mode",
    "mcp",
    "mqa",
    "mlx",
    "moe",
    "nlp",
    "notebook",
    "numpy",
    "ocr",
    "output",
    "pandas",
    "pdf",
    "pipeline",
    "ppt",
    "pred",
    "pytorch",
    "qlora",
    "rag",
    "readme",
    "rerank",
    "rlhf",
    "runpod",
    "scikit",
    "search",
    "seq2seq",
    "shape",
    "sglang",
    "sklearn",
    "sop",
    "sql",
    "stable",
    "standardscaler",
    "svm",
    "t5",
    "terminal",
    "textual",
    "token",
    "tokenizer",
    "training_log",
    "transformer",
    "transformers",
    "ui",
    "url",
    "uvicorn",
    "vllm",
    "vrag",
    "vs",
    "vscode",
    "web",
    "word",
}

ENGLISH_PROSE_HINTS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "before",
    "between",
    "by",
    "for",
    "from",
    "how",
    "in",
    "into",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}


def iter_markdown(locale: str) -> list[Path]:
    root = LOCALES[locale]
    if not root.exists():
        return []
    files = [p for p in root.rglob("*.md") if p.is_file()]
    if locale == "en":
        files = [p for p in files if "zh-cn" not in p.parts and "ja" not in p.parts]
    return sorted(files)


def rel_key(path: Path, locale: str) -> str:
    root = LOCALES[locale]
    return path.relative_to(root).as_posix()


def count(pattern: str, text: str) -> int:
    return len(re.findall(pattern, text, flags=re.MULTILINE))


def strip_fenced_code(text: str) -> str:
    return re.sub(r"```.*?```", "", text, flags=re.DOTALL)


def is_allowlisted(path_key: str, patterns: list[str]) -> bool:
    return any(re.fullmatch(pattern, path_key) for pattern in patterns)


def localized_alt_looks_english(alt: str) -> bool:
    words = re.findall(r"[A-Za-z][A-Za-z0-9_+\-.]*", alt)
    if not words:
        return False
    has_local_script = bool(CJK_RE.search(alt))
    if not has_local_script:
        return True

    lower_words = [word.lower().strip("._-") for word in words]
    prose_words = [
        word
        for word in lower_words
        if word not in TECHNICAL_ALT_TERMS and not re.fullmatch(r"[a-z]\d*|v\d+|x\d+", word)
    ]
    if any(word in ENGLISH_PROSE_HINTS for word in prose_words):
        return True

    return False


def analyze_file(path: Path, locale: str) -> dict[str, object]:
    text = path.read_text(encoding="utf-8")
    visible = strip_fenced_code(text)
    path_key = rel_key(path, locale)
    summaries = re.findall(r"<summary>(.*?)</summary>", text, flags=re.DOTALL)
    markdown_images = MARKDOWN_IMAGE_RE.findall(text)
    html_images = [(None, match.group(1)) for match in HTML_IMAGE_RE.finditer(text)]
    image_refs = [ref for _, ref in markdown_images] + [ref for _, ref in html_images]
    direct_raster_refs = [ref for ref in image_refs if re.search(r"\.(png|jpe?g)(?:$|[?#])", ref, re.I)]
    residue_hits: list[str] = []
    english_alt_hits: list[str] = []
    legacy_context_hits: list[str] = []
    if locale in {"zh-cn", "ja"}:
        for pattern in VISIBLE_RESIDUE_PATTERNS:
            if re.search(pattern, visible):
                residue_hits.append(pattern)
        english_alt_hits = [alt for alt, _ in MARKDOWN_IMAGE_RE.findall(visible) if localized_alt_looks_english(alt)]
    if not is_allowlisted(path_key, LEGACY_CONTEXT_ALLOWLIST):
        for pattern in LEGACY_CONTEXT_PATTERNS:
            if re.search(pattern, visible, flags=re.IGNORECASE):
                legacy_context_hits.append(pattern)
    return {
        "path": path_key,
        "has_evidence": EVIDENCE_HEADINGS[locale] in text,
        "has_summary": bool(summaries),
        "duplicate_summaries": [item for item, n in Counter(summaries).items() if n > 1],
        "details_delta": count(r"<details>", text) - count(r"</details>", text),
        "summary_delta": count(r"<summary>", text) - count(r"</summary>", text),
        "image_count": len(image_refs),
        "direct_raster_refs": direct_raster_refs,
        "visible_residue_hits": residue_hits,
        "english_alt_hits": english_alt_hits,
        "legacy_context_hits": legacy_context_hits,
    }


def main() -> int:
    files = {locale: iter_markdown(locale) for locale in LOCALES}
    keys = {locale: {rel_key(path, locale) for path in paths} for locale, paths in files.items()}
    english_keys = keys["en"]

    report: dict[str, object] = {
        "root": str(ROOT),
        "locales": {},
        "mirror_gaps": {},
        "status": "passed",
    }

    for locale, locale_files in files.items():
        analyses = [analyze_file(path, locale) for path in locale_files]
        missing_evidence = [a["path"] for a in analyses if not a["has_evidence"]]
        missing_answers = [a["path"] for a in analyses if not a["has_summary"]]
        missing_images = [a["path"] for a in analyses if a["image_count"] == 0]
        unbalanced = [a["path"] for a in analyses if a["details_delta"] or a["summary_delta"]]
        duplicate_summaries = [a["path"] for a in analyses if a["duplicate_summaries"]]
        direct_raster = [a for a in analyses if a["direct_raster_refs"]]
        visible_residue = [a for a in analyses if a["visible_residue_hits"]]
        english_alt = [a for a in analyses if a["english_alt_hits"]]
        legacy_context = [a for a in analyses if a["legacy_context_hits"]]

        locale_report = {
            "pages": len(locale_files),
            "missing_evidence": len(missing_evidence),
            "missing_answer_summary": len(missing_answers),
            "missing_image": len(missing_images),
            "unbalanced_details_or_summary": len(unbalanced),
            "duplicate_summary_pages": len(duplicate_summaries),
            "direct_png_jpg_reference_pages": len(direct_raster),
            "visible_residue_pages": len(visible_residue),
            "english_alt_text_pages": len(english_alt),
            "legacy_context_advisory_pages": len(legacy_context),
            "samples": {
                "missing_evidence": missing_evidence[:10],
                "missing_answer_summary": missing_answers[:10],
                "missing_image": missing_images[:10],
                "unbalanced": unbalanced[:10],
                "direct_png_jpg": [a["path"] for a in direct_raster[:10]],
                "visible_residue": [a["path"] for a in visible_residue[:10]],
                "english_alt_text": [a["path"] for a in english_alt[:10]],
                "legacy_context_advisory": [a["path"] for a in legacy_context[:10]],
            },
        }
        report["locales"][locale] = locale_report

        if any(
            locale_report[key]
            for key in [
                "unbalanced_details_or_summary",
                "duplicate_summary_pages",
                "direct_png_jpg_reference_pages",
                "visible_residue_pages",
                "english_alt_text_pages",
            ]
        ):
            report["status"] = "needs_attention"

    for locale in ["zh-cn", "ja"]:
        missing = sorted(english_keys - keys[locale])
        extra = sorted(keys[locale] - english_keys)
        report["mirror_gaps"][locale] = {
            "missing_from_locale": len(missing),
            "extra_in_locale": len(extra),
            "missing_samples": missing[:20],
            "extra_samples": extra[:20],
        }
        if missing or extra:
            report["status"] = "needs_attention"

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
