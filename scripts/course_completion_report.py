#!/usr/bin/env python3
"""Create a completion scorecard for the multilingual course.

The report is advisory. It helps decide where to improve next by combining
page structure, runnable signals, image teaching warnings, and readability
warnings into per-page and per-chapter scores.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = ROOT / "src" / "content" / "docs"
REPORT_JSON = ROOT / "reports" / "course-completion-report.json"
REPORT_MD = ROOT / "reports" / "course-completion-report.md"
LOCALE_ROOTS = {
    "en": DOCS_ROOT,
    "zh-cn": DOCS_ROOT / "zh-cn",
    "ja": DOCS_ROOT / "ja",
}
EVIDENCE_HEADINGS = {
    "en": "## Evidence to Keep",
    "zh-cn": "## 留下的证据",
    "ja": "## 残す証拠",
}

FENCE_START_RE = re.compile(r"^(`{3,}|~{3,})([^\n]*)$")
FRONTMATTER_RE = re.compile(r"^---\n.*?\n---\n", re.DOTALL)
MARKDOWN_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
HTML_TAG_RE = re.compile(r"<[^>]+>")
HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+)$", re.MULTILINE)
DETAIL_RE = re.compile(r"<details\b", re.I)
SUMMARY_RE = re.compile(r"<summary\b", re.I)
PASS_CHECK_RE = re.compile(
    r"(pass check|checklist|expected_output|stage completion|通过检查|通关|检查|通過条件|確認)",
    re.I,
)
PRACTICE_RE = re.compile(
    r"(exercise|practice|project|workshop|lab|your turn|练习|实操|实践|项目|工作坊|実践|練習|プロジェクト|ワークショップ)",
    re.I,
)
COMMAND_RE = re.compile(
    r"(python3?\s+\S+|pip\s+install|uv\s+run|npm\s+run|streamlit\s+run|pytest\b|curl\s+|docker\s+|colab|cuda|nvidia-smi)",
    re.I,
)
RUNNABLE_LANGS = {"bash", "sh", "zsh", "powershell", "python", "javascript", "sql"}


@dataclass(frozen=True)
class PageScore:
    locale: str
    path: str
    chapter: str
    kind: str
    score: int
    status: str
    content_units: int
    h2_count: int
    image_count: int
    code_blocks: int
    runnable_blocks: int
    details_blocks: int
    image_warnings: int
    readability_warnings: int
    deductions: list[str]


def load_script_module(name: str) -> Any:
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def iter_markdown(locale: str) -> list[Path]:
    root = LOCALE_ROOTS[locale]
    files = [path for path in root.rglob("*.md") if path.is_file()]
    if locale == "en":
        files = [path for path in files if "zh-cn" not in path.parts and "ja" not in path.parts]
    return sorted(files)


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def path_key(path: Path, locale: str) -> str:
    return path.relative_to(LOCALE_ROOTS[locale]).as_posix()


def chapter_for(key: str) -> str:
    first = key.split("/", 1)[0]
    if re.fullmatch(r"ch\d{2}-[a-z0-9-]+", first):
        return first
    if first in {"intro", "appendix", "electives"}:
        return first
    return "root"


def page_kind(key: str) -> str:
    name = Path(key).name
    parts = key.split("/")
    if key == "index.md":
        return "site-index"
    if name == "study-guide.md":
        return "study-guide"
    if name == "00-roadmap.md":
        return "roadmap"
    if name == "index.md" and len(parts) == 2:
        return "chapter-index"
    if "workshop" in key or "project" in key or "lab" in key:
        return "hands-on"
    if parts[0] == "appendix":
        return "appendix"
    if parts[0] == "intro":
        return "intro"
    return "lesson"


def parse_code_blocks(text: str) -> list[tuple[str, str]]:
    blocks: list[tuple[str, str]] = []
    in_block = False
    fence = ""
    lang = ""
    body: list[str] = []

    for line in text.splitlines():
        match = FENCE_START_RE.match(line)
        if not in_block and match:
            in_block = True
            fence = match.group(1)
            info = match.group(2).strip()
            lang = info.split()[0].strip("`").lower() if info else ""
            body = []
            continue
        if in_block and line.startswith(fence):
            blocks.append((lang, "\n".join(body)))
            in_block = False
            fence = ""
            lang = ""
            body = []
            continue
        if in_block:
            body.append(line)

    return blocks


def visible_text(text: str) -> str:
    text = FRONTMATTER_RE.sub("", text)
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = MARKDOWN_IMAGE_RE.sub(" ", text)
    text = HTML_TAG_RE.sub(" ", text)
    return text


def content_units(value: str) -> int:
    latin_words = re.findall(r"[A-Za-z][A-Za-z0-9_+\-.]*", value)
    cjk_chars = re.findall(r"[\u3400-\u9fff\u3040-\u30ff]", value)
    numbers = re.findall(r"\b\d+(?:\.\d+)?\b", value)
    return int(len(latin_words) + len(numbers) + len(cjk_chars) / 1.8)


def count_h2(text: str) -> int:
    return sum(1 for marker, _ in HEADING_RE.findall(text) if marker == "##")


def collect_warning_counts(files: list[Path]) -> tuple[Counter[str], Counter[str]]:
    image_counts: Counter[str] = Counter()
    readability_counts: Counter[str] = Counter()

    image_module = load_script_module("audit_image_teaching")
    readability_module = load_script_module("audit_readability")

    for path in files:
        for finding in image_module.audit_file(path):
            image_counts[finding.path] += 1
        for finding in readability_module.audit_file(path):
            readability_counts[finding.path] += 1

    return image_counts, readability_counts


def score_page(path: Path, locale: str, image_counts: Counter[str], readability_counts: Counter[str]) -> PageScore:
    text = path.read_text(encoding="utf-8", errors="ignore")
    visible = visible_text(text)
    key = path_key(path, locale)
    kind = page_kind(key)
    chapter = chapter_for(key)
    code_blocks = parse_code_blocks(text)
    h2_count = count_h2(text)
    images = MARKDOWN_IMAGE_RE.findall(text)
    runnable_blocks = sum(1 for lang, body in code_blocks if lang in RUNNABLE_LANGS and body.strip())
    details_blocks = len(DETAIL_RE.findall(text))
    summaries = len(SUMMARY_RE.findall(text))
    units = content_units(visible)
    has_evidence = EVIDENCE_HEADINGS[locale] in text
    has_pass_check = bool(PASS_CHECK_RE.search(visible))
    has_practice = bool(PRACTICE_RE.search(visible))
    has_command = bool(COMMAND_RE.search(text))
    rel_path = rel(path)
    image_warning_count = image_counts[rel_path]
    readability_warning_count = readability_counts[rel_path]

    deductions: list[tuple[int, str]] = []

    def deduct(points: int, reason: str) -> None:
        deductions.append((points, reason))

    if not has_evidence and kind not in {"site-index"}:
        deduct(20, "missing evidence section")
    if DETAIL_RE.findall(text) and details_blocks != text.count("</details>"):
        deduct(25, "unbalanced details tags")
    if SUMMARY_RE.findall(text) and summaries != text.count("</summary>"):
        deduct(20, "unbalanced summary tags")

    if kind in {"lesson", "hands-on"}:
        if units < 220:
            deduct(22, f"very short visible content ({units} units)")
        elif units < 420:
            deduct(10, f"thin visible content ({units} units)")
        if h2_count < 2:
            deduct(8, "few teaching sections")
    elif kind in {"chapter-index", "study-guide", "roadmap"}:
        if units < 160:
            deduct(12, f"thin navigation content ({units} units)")

    if kind in {"lesson", "hands-on", "chapter-index", "roadmap", "study-guide"} and not images:
        deduct(10, "no course image reference")

    if kind == "hands-on" and not (runnable_blocks or has_command):
        deduct(12, "hands-on page has weak runnable signal")
    if kind == "lesson" and has_practice and not (runnable_blocks or has_command or details_blocks):
        deduct(8, "practice page has weak runnable or folded-answer signal")
    if has_practice and not (details_blocks or has_pass_check):
        deduct(6, "practice signal without pass check or folded answer")

    if image_warning_count:
        deduct(min(12, image_warning_count * 4), f"{image_warning_count} image teaching warning(s)")
    if readability_warning_count:
        deduct(min(10, readability_warning_count * 2), f"{readability_warning_count} readability warning(s)")

    score = max(0, 100 - sum(points for points, _ in deductions))
    status = "solid" if score >= 90 else "review" if score >= 80 else "needs_work"
    return PageScore(
        locale=locale,
        path=key,
        chapter=chapter,
        kind=kind,
        score=score,
        status=status,
        content_units=units,
        h2_count=h2_count,
        image_count=len(images),
        code_blocks=len(code_blocks),
        runnable_blocks=runnable_blocks,
        details_blocks=details_blocks,
        image_warnings=image_warning_count,
        readability_warnings=readability_warning_count,
        deductions=[reason for _, reason in deductions],
    )


def summarize_pages(pages: list[PageScore]) -> dict[str, Any]:
    if not pages:
        return {"pages": 0, "average_score": 0, "solid": 0, "review": 0, "needs_work": 0}
    status_counts = Counter(page.status for page in pages)
    return {
        "pages": len(pages),
        "average_score": round(mean(page.score for page in pages), 1),
        "solid": status_counts["solid"],
        "review": status_counts["review"],
        "needs_work": status_counts["needs_work"],
        "image_warnings": sum(page.image_warnings for page in pages),
        "readability_warnings": sum(page.readability_warnings for page in pages),
    }


def chapter_summaries(pages: list[PageScore]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[PageScore]] = defaultdict(list)
    for page in pages:
        grouped[(page.locale, page.chapter)].append(page)

    summaries: list[dict[str, Any]] = []
    for (locale, chapter), group in sorted(grouped.items()):
        base = summarize_pages(group)
        low = sorted(group, key=lambda page: (page.score, page.path))[:5]
        summaries.append(
            {
                "locale": locale,
                "chapter": chapter,
                **base,
                "lowest_pages": [
                    {"path": page.path, "score": page.score, "status": page.status}
                    for page in low
                    if page.score < 90
                ],
            }
        )
    return summaries


def write_markdown(report: dict[str, Any]) -> None:
    lines = [
        "# Course Completion Report",
        "",
        "Generated by `scripts/course_completion_report.py`. Scores are advisory and are meant to prioritize review work.",
        "",
        "## Locale Summary",
        "",
        "| Locale | Pages | Average | Solid | Review | Needs work | Image warnings | Readability warnings |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for locale, summary in report["locale_summary"].items():
        lines.append(
            "| {locale} | {pages} | {average_score} | {solid} | {review} | {needs_work} | {image_warnings} | {readability_warnings} |".format(
                locale=locale,
                **summary,
            )
        )

    lines.extend(
        [
            "",
            "## Chapter Summary",
            "",
            "| Locale | Chapter | Pages | Average | Needs work | Image warnings | Readability warnings |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for item in report["chapter_summary"]:
        lines.append(
            "| {locale} | {chapter} | {pages} | {average_score} | {needs_work} | {image_warnings} | {readability_warnings} |".format(
                **item
            )
        )

    lines.extend(["", "## Pages Needing Attention", ""])
    if report["lowest_pages"]:
        lines.extend(
            [
                "| Locale | Score | Status | Page | Main reasons |",
                "|---|---:|---|---|---|",
            ]
        )
        for page in report["lowest_pages"]:
            reasons = "; ".join(page["deductions"][:3])
            lines.append(f"| {page['locale']} | {page['score']} | {page['status']} | `{page['path']}` | {reasons} |")
    else:
        lines.append("No pages currently have completion deductions.")

    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_report() -> dict[str, Any]:
    files_by_locale = {locale: iter_markdown(locale) for locale in LOCALE_ROOTS}
    all_files = [path for files in files_by_locale.values() for path in files]
    image_counts, readability_counts = collect_warning_counts(all_files)
    pages = [
        score_page(path, locale, image_counts, readability_counts)
        for locale, files in files_by_locale.items()
        for path in files
    ]

    locale_summary = {
        locale: summarize_pages([page for page in pages if page.locale == locale])
        for locale in LOCALE_ROOTS
    }
    low_pages = sorted(
        (page for page in pages if page.deductions),
        key=lambda page: (page.score, page.locale, page.path),
    )[:40]
    return {
        "root": str(ROOT),
        "page_count": len(pages),
        "locale_summary": locale_summary,
        "chapter_summary": chapter_summaries(pages),
        "attention_page_count": len(low_pages),
        "lowest_pages": [asdict(page) for page in low_pages],
        "pages": [asdict(page) for page in sorted(pages, key=lambda page: (page.locale, page.path))],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print the full JSON report.")
    parser.add_argument("--max-pages", type=int, default=20, help="Lowest scoring pages to print.")
    parser.add_argument("--fail-under", type=int, default=0, help="Fail if any page score is below this threshold.")
    args = parser.parse_args()

    report = build_report()
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(report)

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"audited_pages={report['page_count']}")
        print(f"report_json={rel(REPORT_JSON)}")
        print(f"report_markdown={rel(REPORT_MD)}")
        for locale, summary in report["locale_summary"].items():
            print(
                f"{locale}: average={summary['average_score']} "
                f"solid={summary['solid']} review={summary['review']} needs_work={summary['needs_work']}"
            )
        print(f"attention_pages={report['attention_page_count']}")
        for page in report["lowest_pages"][: args.max_pages]:
            reasons = "; ".join(page["deductions"][:3])
            label = "LOW" if page["score"] < 90 else "ATTN"
            print(f"{label} {page['locale']} {page['score']:>3} {page['path']} :: {reasons}")

    if args.fail_under and any(page["score"] < args.fail_under for page in report["pages"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
