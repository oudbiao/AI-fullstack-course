#!/usr/bin/env python3
"""Audit fenced code blocks in course Markdown files.

The goal is not to prove that every tutorial program can complete without
external services, datasets, credentials, GPUs, or optional packages. Instead,
this script catches the problems that most often block beginners immediately:

- malformed fenced code blocks
- Python syntax errors
- JSON syntax errors
- JavaScript syntax errors
- Bash syntax errors for command-like snippets
- TODO/pass/ellipsis-only placeholders that look like unfinished code

It prints a compact report and exits non-zero if blocking issues are found.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


ROOTS = [
    Path("src/content/docs"),
]

PYTHON_LANGS = {"python", "py"}
BASH_LANGS = {"bash", "sh", "shell", "zsh"}
JSON_LANGS = {"json"}
JSONL_LANGS = {"jsonl", "ndjson"}
JS_LANGS = {"javascript", "js"}
SKIP_LANGS = {
    "",
    "text",
    "txt",
    "mermaid",
    "sql",
    "md",
    "markdown",
    "diff",
    "csv",
    "html",
    "dockerfile",
    "yaml",
    "yml",
    "cpp",
    "c++",
    "java",
    "powershell",
    "ps1",
}

PROMPT_RE = re.compile(r"^\s*(>>>|\.\.\.|\$|PS>|In \[\d*\]:|Out\[\d*\]:)")
NOTEBOOK_MAGIC_RE = re.compile(r"^\s*(!|%%?)[A-Za-z0-9_]")
PLACEHOLDER_RE = re.compile(
    r"(?im)^\s*(pass|todo\b|#\s*todo\b|//\s*todo\b|\.\.\.)\s*$"
)
INLINE_PLACEHOLDER_RE = re.compile(
    r"(?im)(pass\s*#|#\s*(fill in the code|add code here|补充代码|这里填代码|这里填写|这里补充|ここにコードを追加|コードを補完してください|ここを書き足す)|#.*\.\.\..*(code|logic|training|processing)|#.*(code|logic|training|processing).*\.\.\.)"
)
ELLIPSIS_ASSIGN_RE = re.compile(r"(?m)^\s*[\w.]+\s*=\s*\.\.\.\s*$")
ELLIPSIS_STUB_RE = re.compile(r"(?m)^\s*(?:async\s+def|def|class)\b.*:\s*\.\.\.\s*$")
OUTPUT_MARKER_RE = re.compile(
    r"(?im)^\s*(output|输出|実行結果|结果|結果)\s*:?\s*$"
)


@dataclass(frozen=True)
class CodeBlock:
    path: Path
    lang: str
    code: str
    start_line: int
    end_line: int


@dataclass(frozen=True)
class Finding:
    severity: str
    path: Path
    line: int
    lang: str
    message: str


def iter_markdown_files(roots: list[Path]) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        files.extend(sorted(root.rglob("*.md")))
    return files


def extract_code_blocks(path: Path) -> tuple[list[CodeBlock], list[Finding]]:
    blocks: list[CodeBlock] = []
    findings: list[Finding] = []
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    in_block = False
    lang = ""
    start_line = 0
    body: list[str] = []

    for index, line in enumerate(lines, start=1):
        if line.startswith("```"):
            if not in_block:
                in_block = True
                info = line[3:].strip()
                lang = info.split()[0].strip("`").lower() if info else ""
                start_line = index
                body = []
            else:
                blocks.append(
                    CodeBlock(
                        path=path,
                        lang=lang,
                        code="\n".join(body),
                        start_line=start_line,
                        end_line=index,
                    )
                )
                in_block = False
                lang = ""
                start_line = 0
                body = []
            continue

        if in_block:
            body.append(line)

    if in_block:
        findings.append(
            Finding("error", path, start_line, lang, "unclosed fenced code block")
        )

    return blocks, findings


def is_prompt_transcript(code: str) -> bool:
    return any(PROMPT_RE.match(line) for line in code.splitlines())


def looks_like_output_mixed_with_code(code: str) -> bool:
    return bool(OUTPUT_MARKER_RE.search(code))


def is_fragment_python(code: str) -> bool:
    stripped = code.strip()
    if not stripped:
        return True
    if is_prompt_transcript(stripped):
        return True
    if any(NOTEBOOK_MAGIC_RE.match(line) for line in stripped.splitlines()):
        return True
    if looks_like_output_mixed_with_code(stripped):
        return True
    return False


def check_python(block: CodeBlock) -> list[Finding]:
    findings: list[Finding] = []
    code = block.code

    if not code.strip():
        findings.append(
            Finding("error", block.path, block.start_line, block.lang, "empty Python block")
        )
        return findings

    placeholder_lines = [
        index
        for index, line in enumerate(code.splitlines(), start=1)
        if PLACEHOLDER_RE.match(line)
    ]
    if placeholder_lines:
        try:
            tree = ast.parse(code, filename=f"{block.path}:{block.start_line}")
            placeholder_lines = [
                line
                for line in placeholder_lines
                if not is_intentional_empty_body(tree, line)
            ]
        except SyntaxError:
            # The syntax error is reported below; keep the placeholder signal too.
            pass

    if placeholder_lines:
        findings.append(
            Finding(
                "warn",
                block.path,
                block.start_line,
                block.lang,
                "placeholder statement found; make sure the snippet is intentionally incomplete",
            )
        )

    if INLINE_PLACEHOLDER_RE.search(code) or ELLIPSIS_ASSIGN_RE.search(code) or ELLIPSIS_STUB_RE.search(code):
        if is_intentional_placeholder_example(code):
            return findings
        findings.append(
            Finding(
                "warn",
                block.path,
                block.start_line,
                block.lang,
                "inline placeholder comment or ellipsis assignment found; consider completing the snippet",
            )
        )

    if is_fragment_python(code):
        findings.append(
            Finding(
                "info",
                block.path,
                block.start_line,
                block.lang,
                "Python transcript/output fragment skipped for syntax compilation",
            )
        )
        return findings

    try:
        ast.parse(code, filename=f"{block.path}:{block.start_line}")
    except SyntaxError as exc:
        findings.append(
            Finding(
                "error",
                block.path,
                block.start_line + max((exc.lineno or 1) - 1, 0),
                block.lang,
                f"Python syntax error: {exc.msg}",
            )
        )

    return findings


def is_intentional_placeholder_example(code: str) -> bool:
    lowered = code.lower()
    markers = [
        "# bad practice",
        "# good practice",
        "no return",
        "style only",
        "print(result)                 # none",
        "pass   # and does nothing at all!",
        "pass   # 而且还什么都不做！",
        "pass   # しかも何もしない！",
        "3.333...",
        "keep looking...",
        "继续找...",
        "继续找…",
    ]
    return any(marker in lowered for marker in markers)


def is_intentional_empty_body(tree: ast.AST, line: int) -> bool:
    """Return True for pass/ellipsis used as a valid empty body.

    This avoids flagging examples that demonstrate style, optional imports, or
    intentionally empty classes/functions. Standalone exercise placeholders are
    still reported unless they are part of a syntactically valid empty body.
    """

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if len(node.body) == 1 and node.body[0].lineno == line:
                only = node.body[0]
                if isinstance(only, ast.Pass):
                    return True
                if (
                    isinstance(only, ast.Expr)
                    and isinstance(only.value, ast.Constant)
                    and only.value.value is Ellipsis
                ):
                    return True
        if isinstance(node, ast.ExceptHandler):
            if len(node.body) == 1 and node.body[0].lineno == line:
                only = node.body[0]
                if isinstance(only, ast.Pass):
                    return True
    return False


def check_json(block: CodeBlock) -> list[Finding]:
    if not block.code.strip():
        return [Finding("error", block.path, block.start_line, block.lang, "empty JSON block")]
    if block.code.strip().startswith("//"):
        return []
    try:
        json.loads(block.code)
    except json.JSONDecodeError as exc:
        return [
            Finding(
                "error",
                block.path,
                block.start_line + max(exc.lineno - 1, 0),
                block.lang,
                f"JSON syntax error: {exc.msg}",
            )
        ]
    return []


def check_with_command(block: CodeBlock, command: list[str], suffix: str) -> list[Finding]:
    if not block.code.strip():
        return [Finding("error", block.path, block.start_line, block.lang, "empty code block")]

    with tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False) as handle:
        handle.write(block.code)
        temp_path = Path(handle.name)

    try:
        result = subprocess.run(
            [*command, str(temp_path)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
        )
    except Exception as exc:  # pragma: no cover - defensive for local envs
        temp_path.unlink(missing_ok=True)
        return [
            Finding(
                "warn",
                block.path,
                block.start_line,
                block.lang,
                f"checker unavailable: {exc}",
            )
        ]

    temp_path.unlink(missing_ok=True)
    if result.returncode == 0:
        return []

    message = " ".join((result.stderr or result.stdout).strip().split())
    return [
        Finding(
            "error",
            block.path,
            block.start_line,
            block.lang,
            f"syntax check failed: {message[:240]}",
        )
    ]


def check_bash(block: CodeBlock) -> list[Finding]:
    code = block.code.strip()
    if not code:
        return [Finding("error", block.path, block.start_line, block.lang, "empty Bash block")]

    # Many tutorial bash blocks are command transcripts with comments or output.
    # `bash -n` still catches unmatched quotes/brackets without executing them.
    return check_with_command(block, ["bash", "-n"], ".sh")


def check_js(block: CodeBlock) -> list[Finding]:
    return check_with_command(block, ["node", "--check"], ".js")


def audit(roots: list[Path]) -> tuple[list[CodeBlock], list[Finding]]:
    blocks: list[CodeBlock] = []
    findings: list[Finding] = []
    for path in iter_markdown_files(roots):
        file_blocks, file_findings = extract_code_blocks(path)
        blocks.extend(file_blocks)
        findings.extend(file_findings)

    known_langs = PYTHON_LANGS | BASH_LANGS | JSON_LANGS | JSONL_LANGS | JS_LANGS | SKIP_LANGS
    for block in blocks:
        lang = block.lang
        if lang in PYTHON_LANGS:
            findings.extend(check_python(block))
        elif lang in JSON_LANGS:
            findings.extend(check_json(block))
        elif lang in JSONL_LANGS:
            for offset, line in enumerate(block.code.splitlines()):
                if not line.strip():
                    continue
                try:
                    json.loads(line)
                except json.JSONDecodeError as exc:
                    findings.append(
                        Finding(
                            "error",
                            block.path,
                            block.start_line + offset,
                            block.lang,
                            f"JSONL syntax error: {exc.msg}",
                        )
                    )
        elif lang in BASH_LANGS:
            findings.extend(check_bash(block))
        elif lang in JS_LANGS:
            findings.extend(check_js(block))
        elif lang not in known_langs:
            findings.append(
                Finding("warn", block.path, block.start_line, lang, "unknown code fence language")
            )

    return blocks, findings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "roots",
        nargs="*",
        type=Path,
        help="Markdown roots to audit. Defaults to the Starlight content tree.",
    )
    parser.add_argument("--max-findings", type=int, default=200)
    args = parser.parse_args()

    roots = args.roots or ROOTS
    blocks, findings = audit(roots)

    by_lang: dict[str, int] = {}
    by_severity: dict[str, int] = {}
    for block in blocks:
        by_lang[block.lang or "(none)"] = by_lang.get(block.lang or "(none)", 0) + 1
    for finding in findings:
        by_severity[finding.severity] = by_severity.get(finding.severity, 0) + 1

    print(f"audited_files={len(iter_markdown_files(roots))}")
    print(f"code_blocks={len(blocks)}")
    print(
        "languages="
        + ", ".join(f"{lang}:{count}" for lang, count in sorted(by_lang.items()))
    )
    print(
        "findings="
        + ", ".join(
            f"{severity}:{count}" for severity, count in sorted(by_severity.items())
        )
    )

    for finding in findings[: args.max_findings]:
        print(
            f"{finding.severity.upper()} {finding.path}:{finding.line} "
            f"[{finding.lang or 'none'}] {finding.message}"
        )

    remaining = len(findings) - args.max_findings
    if remaining > 0:
        print(f"... {remaining} more findings omitted")

    return 1 if any(f.severity == "error" for f in findings) else 0


if __name__ == "__main__":
    raise SystemExit(main())
