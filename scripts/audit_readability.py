#!/usr/bin/env python3
"""Audit course Markdown for layout patterns that often render poorly.

This is a warning-oriented report. It is meant to reveal pages that deserve
manual review, not to block every existing lesson. Pass `--fail-on-warn` when
you want to tighten the gate after fixing the backlog.
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
REPORT_PATH = ROOT / "reports" / "readability-audit.json"

TEXT_LANGS = {"", "text", "txt", "plain", "plaintext"}
STRUCTURED_LABEL_RE = re.compile(r"^\s*[-*]?\s*[^:：\n]{1,48}[:：]\s+\S")
PIPE_TABLE_RE = re.compile(r"^\s*\|.*\|\s*$")
FENCE_START_RE = re.compile(r"^(`{3,}|~{3,})([^\n]*)$")
EVIDENCE_CONTEXT_MARKERS = [
    "evidence card",
    "evidence to keep",
    "proof of learning",
    "keep evidence",
    "stage deliverable",
    "deliverable rhythm",
    "留下的证据",
    "证据卡",
    "阶段交付",
    "残す証拠",
    "証拠カード",
    "段階ごとの提出",
]
TERMINAL_CONTEXT_MARKERS = [
    "expected output",
    "example output",
    "sample output",
    "output:",
    "输出",
    "输出示例",
    "运行结果",
    "実行結果",
    "想定出力",
    "出力例",
    "期待される出力",
]
EVIDENCE_LABEL_HINTS = {
    "artifact",
    "baseline",
    "deliverable",
    "evidence",
    "failure",
    "project_goal",
    "run_command",
    "target_role",
    "what_changed",
    "what_failed",
    "what_proves_it",
    "how_to_rerun",
    "next_step",
    "产出",
    "交付",
    "岗位",
    "证据",
    "新增能力",
    "如何重跑",
    "证明材料",
    "失败样本",
    "下一步",
    "証拠",
    "職種",
    "成果物",
    "何が変わったか",
    "どう再実行するか",
    "何が証明するか",
    "何が失敗したか",
    "次に何をするか",
}
OUTPUT_LINE_RE = re.compile(
    r"^(?:"
    r"step|epoch|loss|accuracy|device|checkpoint|output|result|traceback|"
    r"on branch|no commits|untracked files|changes to be committed|commit\s+[0-9a-f]|"
    r"author:|date:|nothing added|total\b|loaded\b|group\b|shape\b"
    r")\b",
    re.I,
)


@dataclass(frozen=True)
class Finding:
    severity: str
    kind: str
    path: str
    line: int
    message: str
    sample: str


def iter_markdown_files(roots: list[Path]) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        if root.is_file() and root.suffix in {".md", ".mdx"}:
            files.append(root)
        elif root.exists():
            files.extend(path for path in root.rglob("*") if path.suffix in {".md", ".mdx"})
    return sorted(files)


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def display_width(value: str) -> int:
    width = 0
    for char in value:
        width += 2 if "\u2e80" <= char <= "\uffff" else 1
    return width


def sample(value: str, limit: int = 140) -> str:
    cleaned = " ".join(value.strip().split())
    return cleaned[:limit]


def table_blocks(lines: list[str]) -> list[tuple[int, list[str]]]:
    blocks: list[tuple[int, list[str]]] = []
    start: int | None = None
    current: list[str] = []
    in_fence = False
    fence = ""

    def flush() -> None:
        nonlocal start, current
        if start is not None and len(current) >= 2:
            blocks.append((start, current))
        start = None
        current = []

    for index, line in enumerate(lines, start=1):
        fence_match = FENCE_START_RE.match(line)
        if fence_match and not in_fence:
            flush()
            in_fence = True
            fence = fence_match.group(1)
            continue
        if in_fence:
            if line.startswith(fence):
                in_fence = False
                fence = ""
            continue

        if PIPE_TABLE_RE.match(line):
            if start is None:
                start = index
            current.append(line)
            continue

        if start is not None:
            flush()

    flush()

    return blocks


def split_table_row(line: str) -> list[str]:
    stripped = line.strip()
    if not stripped.startswith("|") or not stripped.endswith("|"):
        return []

    cells: list[str] = []
    current: list[str] = []
    escaped = False
    for char in stripped[1:-1]:
        if char == "|" and not escaped:
            cells.append("".join(current).strip())
            current = []
            continue
        current.append(char)
        escaped = char == "\\" and not escaped
        if char != "\\":
            escaped = False

    cells.append("".join(current).strip())
    return cells


def audit_table(path: Path, start_line: int, rows: list[str]) -> list[Finding]:
    findings: list[Finding] = []
    max_cells = max(len(split_table_row(row)) for row in rows)
    max_width = max(display_width(row) for row in rows)
    header = rows[0]
    header_cells = split_table_row(header)
    long_header_cells = [cell for cell in header_cells if display_width(cell) > 34]
    long_body_rows = [row for row in rows[1:] if display_width(row) > 180]

    if max_cells >= 5:
        findings.append(
            Finding(
                "warn",
                "wide-table",
                rel(path),
                start_line,
                f"table has {max_cells} columns; consider cards or a narrower comparison",
                sample(header),
            )
        )

    if long_header_cells:
        findings.append(
            Finding(
                "warn",
                "long-table-header",
                rel(path),
                start_line,
                "table header has long labels that may wrap or truncate",
                sample(" | ".join(long_header_cells)),
            )
        )

    if max_width > 220 or len(long_body_rows) >= 2:
        findings.append(
            Finding(
                "warn",
                "dense-table-row",
                rel(path),
                start_line,
                "table contains very dense rows; consider splitting the content",
                sample(max(rows, key=display_width)),
            )
        )

    return findings


def parse_code_blocks(path: Path, lines: list[str]) -> list[tuple[str, int, str, str]]:
    blocks: list[tuple[str, int, str, str]] = []
    in_block = False
    fence = ""
    lang = ""
    start_line = 0
    body: list[str] = []

    for index, line in enumerate(lines, start=1):
        match = FENCE_START_RE.match(line)
        if not in_block and match:
            in_block = True
            fence = match.group(1)
            info = match.group(2).strip()
            lang = info.split()[0].strip("`").lower() if info else ""
            start_line = index
            body = []
            continue

        if in_block and line.startswith(fence):
            previous_context = "\n".join(lines[max(0, start_line - 9) : start_line - 1])
            blocks.append((lang, start_line, "\n".join(body), previous_context))
            in_block = False
            fence = ""
            lang = ""
            start_line = 0
            body = []
            continue

        if in_block:
            body.append(line)

    return blocks


def has_marker(text: str, markers: list[str]) -> bool:
    lowered = text.lower()
    return any(marker.lower() in lowered for marker in markers)


def label_for(line: str) -> str:
    match = STRUCTURED_LABEL_RE.match(line)
    if not match:
        return ""
    return re.split(r"[:：]", line.strip().lstrip("-* "), maxsplit=1)[0].strip().lower()


def evidence_like_labels(lines: list[str]) -> bool:
    labels = [label_for(line).replace(" ", "_") for line in lines if label_for(line)]
    if not labels:
        return False
    return any(any(hint in label for hint in EVIDENCE_LABEL_HINTS) for label in labels)


def audit_text_block(path: Path, lang: str, start_line: int, value: str, previous_context: str) -> list[Finding]:
    if lang not in TEXT_LANGS:
        return []

    lines = [line for line in value.splitlines() if line.strip()]
    if not lines:
        return []

    findings: list[Finding] = []
    structured_lines = [line for line in lines if STRUCTURED_LABEL_RE.match(line)]
    has_many_labels = len(structured_lines) >= 4
    has_arrow_flow = len(lines) == 1 and re.search(r"(?:->|→)", lines[0])
    looks_like_output = has_marker(previous_context, TERMINAL_CONTEXT_MARKERS) or any(
        OUTPUT_LINE_RE.search(line.strip()) for line in lines
    )
    looks_like_rendered_evidence = has_marker(previous_context, EVIDENCE_CONTEXT_MARKERS) or (
        has_many_labels and evidence_like_labels(structured_lines)
    )

    if looks_like_rendered_evidence:
        return []

    if has_many_labels and not looks_like_output:
        findings.append(
            Finding(
                "warn",
                "structured-text-block",
                rel(path),
                start_line,
                "plain text block looks like a card/list; consider a rendered component or table",
                sample(value),
            )
        )
    elif len(lines) >= 8 and not looks_like_output:
        findings.append(
            Finding(
                "warn",
                "long-text-block",
                rel(path),
                start_line,
                "plain text block is long; make sure it is intended as terminal output",
                sample(value),
            )
        )

    if has_arrow_flow and display_width(lines[0]) > 120:
        findings.append(
            Finding(
                "warn",
                "long-flow-line",
                rel(path),
                start_line,
                "single-line flow may become hard to read on mobile",
                sample(lines[0]),
            )
        )

    return findings


def audit_file(path: Path) -> list[Finding]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    findings: list[Finding] = []

    for start_line, rows in table_blocks(lines):
        findings.extend(audit_table(path, start_line, rows))

    for lang, start_line, value, previous_context in parse_code_blocks(path, lines):
        findings.extend(audit_text_block(path, lang, start_line, value, previous_context))

    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="*", type=Path, help="Markdown files or roots to audit.")
    parser.add_argument("--max-findings", type=int, default=60)
    parser.add_argument("--fail-on-warn", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a text summary.")
    args = parser.parse_args()

    roots = args.roots or [DOCS_ROOT]
    files = iter_markdown_files(roots)
    findings = [finding for path in files for finding in audit_file(path)]
    counts: dict[str, int] = {}
    for finding in findings:
        counts[finding.kind] = counts.get(finding.kind, 0) + 1

    report = {
        "audited_files": len(files),
        "finding_count": len(findings),
        "counts": dict(sorted(counts.items())),
        "findings": [asdict(finding) for finding in findings],
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"audited_files={len(files)}")
        print(f"readability_findings={len(findings)}")
        print("counts=" + ", ".join(f"{key}:{value}" for key, value in sorted(counts.items())))
        print(f"report={rel(REPORT_PATH)}")
        for finding in findings[: args.max_findings]:
            print(
                f"{finding.severity.upper()} {finding.path}:{finding.line} "
                f"[{finding.kind}] {finding.message} :: {finding.sample}"
            )
        remaining = len(findings) - args.max_findings
        if remaining > 0:
            print(f"... {remaining} more findings omitted")

    if args.fail_on_warn and findings:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
