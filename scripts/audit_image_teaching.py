#!/usr/bin/env python3
"""Audit whether course images are used as teaching material, not decoration.

The existing image reference validator checks that files exist. This script
adds softer quality signals: useful alt text, locale-specific image variants,
nearby explanation, and accidental repeated image use on the same page.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = ROOT / "src" / "content" / "docs"
REPORT_PATH = ROOT / "reports" / "course-images" / "image-teaching-audit.json"
IMAGE_RE = re.compile(r"!\[([^\]]*)\]\((/img/course/[^)\s]+)[^)]*\)")
HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+")
FENCE_RE = re.compile(r"^(`{3,}|~{3,})")
GENERIC_ALT_PATTERNS = [
    r"^image$",
    r"^diagram$",
    r"^chart$",
    r"^figure$",
    r"^图片$",
    r"^图$",
    r"^示意图$",
    r"^画像$",
    r"^図$",
]
TEACHING_CONTEXT_TERMS = [
    "read the picture",
    "read the image",
    "read the diagram",
    "read the graph",
    "read the flow",
    "read the timeline",
    "read the map",
    "read it like this",
    "follows this path",
    "this path",
    "is stopping at",
    "the code has",
    "upgrade",
    "run:",
    "run ",
    "look like this",
    "should look like",
    "start with the map",
    "start with the diagram",
    "keep this loop",
    "keep this mental model",
    "training rhythm",
    "practical order",
    "first part computes",
    "full pattern",
    "key parts",
    "important things",
    "clearly separates",
    "separates two things",
    "already shows",
    "already has",
    "expresses this",
    "reminds you",
    "no longer just",
    "plain model",
    "open the html report",
    "copy the code",
    "loss is",
    "choose topics",
    "class boundaries",
    "baseline",
    "max_depth",
    "contamination",
    "safe pattern",
    "weights",
    "imshow",
    "clean directory",
    "tokenizer",
    "how to read",
    "look at",
    "notice",
    "this figure",
    "this diagram",
    "the image",
    "evidence",
    "output",
    "result",
    "workflow",
    "route",
    "check",
    "first question",
    "mental model",
    "对照",
    "观察",
    "注意",
    "先读图",
    "先看图",
    "读这张图",
    "读图",
    "看图",
    "阅读这张图",
    "保留这个循环",
    "先记住",
    "这张图",
    "这幅图",
    "图里",
    "证据",
    "输出",
    "结果",
    "流程",
    "路线",
    "检查",
    "掌握",
    "添加",
    "打开",
    "记住",
    "闭环",
    "第一",
    "最重要",
    "计算",
    "训练",
    "节奏",
    "顺序",
    "完整",
    "已经",
    "具备",
    "表达",
    "提醒",
    "分开",
    "随机切分",
    "安全模式",
    "泄漏",
    "选题",
    "类别边界",
    "未来 token",
    "创建干净",
    "保存结果",
    "見る",
    "この図",
    "図を読む",
    "図を見る",
    "流れを見る",
    "まず図",
    "この流れ",
    "図では",
    "注目",
    "確認",
    "覚え",
    "追加",
    "開き",
    "計算",
    "重要",
    "順番",
    "リズム",
    "そろ",
    "表して",
    "分かる",
    "分け",
    "baseline",
    "max_depth",
    "loss",
    "ミニ PyTorch",
    "全体の形",
    "実行",
    "保存",
    "Tokenizer",
    "証拠",
    "出力",
    "結果",
    "流れ",
]


@dataclass(frozen=True)
class ImageRef:
    path: Path
    line: int
    alt: str
    url: str


@dataclass(frozen=True)
class Finding:
    severity: str
    kind: str
    path: str
    line: int
    message: str
    sample: str


def iter_markdown_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.md") if path.is_file())


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def locale_for_path(path: Path) -> str:
    parts = path.parts
    if "zh-cn" in parts:
        return "zh-cn"
    if "ja" in parts:
        return "ja"
    return "en"


def image_variant(url: str) -> str:
    stem = Path(url).stem
    if stem.endswith("-en"):
        return "en"
    if stem.endswith("-ja"):
        return "ja"
    return "base"


def display_len(value: str) -> int:
    return sum(2 if "\u2e80" <= char <= "\uffff" else 1 for char in value)


def sample(value: str, limit: int = 160) -> str:
    return " ".join(value.strip().split())[:limit]


def extract_image_refs(path: Path, text: str) -> list[ImageRef]:
    refs: list[ImageRef] = []
    for match in IMAGE_RE.finditer(text):
        line = text[: match.start()].count("\n") + 1
        refs.append(ImageRef(path, line, match.group(1).strip(), match.group(2)))
    return refs


def paragraph_context(lines: list[str], image_line: int) -> str:
    snippets: list[str] = []
    start_line = max(1, image_line - 4)
    in_fence = False

    for line in lines[: start_line - 1]:
        if FENCE_RE.match(line):
            in_fence = not in_fence

    end_line = min(len(lines), image_line + 8)
    for cursor in range(start_line, end_line + 1):
        line = lines[cursor - 1]
        if FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence or HEADING_RE.match(line):
            continue
        cleaned = IMAGE_RE.sub(" ", line)
        if re.fullmatch(r"\s*---\s*", cleaned):
            continue
        if re.match(r"^\s*(title|description|sidebar|order):\s*", cleaned):
            continue
        snippets.append(cleaned)

    return "\n".join(snippets)


def has_teaching_context(context: str) -> bool:
    lowered = context.lower()
    if any(term.lower() in lowered for term in TEACHING_CONTEXT_TERMS):
        return True
    # A short explicit sentence near the image is usually enough to avoid a
    # decorative image drop, even when it does not use one of the marker words.
    visible_words = re.findall(r"[\w\u3400-\u9fff\u3040-\u30ff]+", context)
    cjk_chars = re.findall(r"[\u3400-\u9fff\u3040-\u30ff]", context)
    return len(visible_words) >= 8 or len(cjk_chars) >= 18


def is_generic_alt(alt: str) -> bool:
    lowered = alt.strip().lower()
    return any(re.fullmatch(pattern, lowered) for pattern in GENERIC_ALT_PATTERNS)


def audit_ref(ref: ImageRef, lines: list[str]) -> list[Finding]:
    findings: list[Finding] = []
    locale = locale_for_path(ref.path)
    variant = image_variant(ref.url)
    context = paragraph_context(lines, ref.line)

    if not ref.alt:
        findings.append(
            Finding("warn", "missing-alt", rel(ref.path), ref.line, "image has empty alt text", ref.url)
        )
    elif display_len(ref.alt) < 10 or is_generic_alt(ref.alt):
        findings.append(
            Finding(
                "warn",
                "thin-alt",
                rel(ref.path),
                ref.line,
                "image alt text is very short or generic",
                ref.alt,
            )
        )

    if locale == "en" and variant != "en":
        findings.append(
            Finding(
                "warn",
                "locale-image-mismatch",
                rel(ref.path),
                ref.line,
                "English page should use an -en course image variant",
                ref.url,
            )
        )
    elif locale == "ja" and variant != "ja":
        findings.append(
            Finding(
                "warn",
                "locale-image-mismatch",
                rel(ref.path),
                ref.line,
                "Japanese page should use a -ja course image variant",
                ref.url,
            )
        )
    elif locale == "zh-cn" and variant != "base":
        findings.append(
            Finding(
                "warn",
                "locale-image-mismatch",
                rel(ref.path),
                ref.line,
                "Chinese page should usually use the base course image variant",
                ref.url,
            )
        )

    if not has_teaching_context(context):
        findings.append(
            Finding(
                "warn",
                "missing-nearby-explanation",
                rel(ref.path),
                ref.line,
                "image has little nearby explanation; add a sentence that tells learners what to inspect",
                sample(context or ref.alt),
            )
        )

    return findings


def audit_file(path: Path) -> list[Finding]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    refs = extract_image_refs(path, text)
    lines = text.splitlines()
    findings: list[Finding] = []

    seen: dict[str, int] = {}
    for ref in refs:
        seen[ref.url] = seen.get(ref.url, 0) + 1
        findings.extend(audit_ref(ref, lines))

    for url, count in sorted(seen.items()):
        if count > 1:
            first_ref = next(ref for ref in refs if ref.url == url)
            findings.append(
                Finding(
                    "warn",
                    "repeated-image-on-page",
                    rel(path),
                    first_ref.line,
                    f"same image appears {count} times on one page",
                    url,
                )
            )

    return findings


def collect_image_usage(files: list[Path]) -> dict[str, list[str]]:
    usage: dict[str, list[str]] = {}
    for path in files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        urls = {ref.url for ref in extract_image_refs(path, text)}
        for url in urls:
            usage.setdefault(url, []).append(rel(path))
    return {url: paths for url, paths in sorted(usage.items()) if len(paths) > 1}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="*", type=Path, help="Markdown files or roots to audit.")
    parser.add_argument("--max-findings", type=int, default=60)
    parser.add_argument("--fail-on-warn", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    files: list[Path] = []
    roots = args.roots or [DOCS_ROOT]
    for root in roots:
        if root.is_file():
            files.append(root)
        elif root.exists():
            files.extend(iter_markdown_files(root))
    files = sorted(set(files))

    findings = [finding for path in files for finding in audit_file(path)]
    repeated_across_pages = collect_image_usage(files)
    high_reuse_images = {url: paths for url, paths in repeated_across_pages.items() if len(paths) > 2}
    counts: dict[str, int] = {}
    for finding in findings:
        counts[finding.kind] = counts.get(finding.kind, 0) + 1

    report = {
        "audited_files": len(files),
        "finding_count": len(findings),
        "counts": dict(sorted(counts.items())),
        "repeated_across_pages_count": len(repeated_across_pages),
        "high_reuse_image_count": len(high_reuse_images),
        "high_reuse_images": high_reuse_images,
        "repeated_across_pages": repeated_across_pages,
        "findings": [asdict(finding) for finding in findings],
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"audited_files={len(files)}")
        print(f"image_teaching_findings={len(findings)}")
        print("counts=" + ", ".join(f"{key}:{value}" for key, value in sorted(counts.items())))
        print(f"repeated_across_pages={len(repeated_across_pages)} high_reuse_images={len(high_reuse_images)}")
        print(f"report={rel(REPORT_PATH)}")
        for finding in findings[: args.max_findings]:
            print(
                f"{finding.severity.upper()} {finding.path}:{finding.line} "
                f"[{finding.kind}] {finding.message} :: {finding.sample}"
            )
        remaining = len(findings) - args.max_findings
        if remaining > 0:
            print(f"... {remaining} more findings omitted")

    return 1 if args.fail_on_warn and findings else 0


if __name__ == "__main__":
    sys.exit(main())
