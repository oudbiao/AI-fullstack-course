#!/usr/bin/env python3
"""Audit localized build output for leftover source-language text."""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HAN_RUN_RE = re.compile(r"[\u4e00-\u9fff]+")
SIMPLIFIED_CHINESE_RE = re.compile(
    r"[这们习说让给还试变图据读页导么吗项课识]"
    r"|课堂讲解|小学高年级"
)
SOURCE_PHRASES_RE = re.compile(
    r"怎么|为什么|什么|如果|然后|这个|一个|没有|可以"
    r"|应用模式|学习路线|课程内容|机器学习|深度学习|人工智能"
    r"|学前导读|数据读写|数据选择|数据清洗|数据转换|数据合并"
    r"|AI项目|AI作品集|RAG项目|Agent项目|AI全栈项目|图像识别"
    r"|课程问答|课程問答|全栈课程|全栈学习|全栈学習"
)
SCRIPT_STYLE_RE = re.compile(r"<(script|style)\b[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")
LOCALIZED_BUILD_DIRS = {"zh-cn", "ja"}

ALLOW_TERMS = {
    "中文",
    "中国",
}


def strip_html(source: str) -> str:
    source = SCRIPT_STYLE_RE.sub(" ", source)
    source = TAG_RE.sub(" ", source)
    source = html.unescape(source)
    return SPACE_RE.sub(" ", source)


def is_allowed(snippet: str) -> bool:
    return any(term in snippet for term in ALLOW_TERMS)


def should_flag(locale: str, match_text: str, snippet: str) -> bool:
    if is_allowed(snippet):
        return False
    if locale == "ja":
        # Japanese normally uses Kanji. Only flag characters/phrases that are
        # strong signals of leftover Simplified Chinese rather than normal
        # Japanese text.
        if SIMPLIFIED_CHINESE_RE.search(match_text):
            return True
        return any(
            phrase in snippet
            for phrase in (
                "怎么",
                "为什么",
                "什么",
                "如果",
                "然后",
                "这个",
                "一个",
                "没有",
                "可以",
                "应用模式",
                "学习路线",
                "课程内容",
                "机器学习",
                "深度学习",
                "学前导读",
                "数据读写",
            )
        )
    return True


def audit_japanese_file(path: Path) -> list[dict[str, object]]:
    text = strip_html(path.read_text(encoding="utf-8", errors="ignore"))
    findings: list[dict[str, object]] = []
    seen: set[str] = set()
    for pattern in (SIMPLIFIED_CHINESE_RE, SOURCE_PHRASES_RE):
        for match in pattern.finditer(text):
            start = max(0, match.start() - 45)
            end = min(len(text), match.end() + 45)
            snippet = text[start:end].strip()
            if is_allowed(snippet):
                continue
            if snippet in seen:
                continue
            seen.add(snippet)
            findings.append({"offset": match.start(), "snippet": snippet})
            if len(findings) >= 20:
                return findings
    return findings


def audit_file(path: Path, locale: str) -> list[dict[str, object]]:
    if locale == "ja":
        return audit_japanese_file(path)

    text = strip_html(path.read_text(encoding="utf-8", errors="ignore"))
    findings: list[dict[str, object]] = []
    seen: set[str] = set()
    for match in HAN_RUN_RE.finditer(text):
        start = max(0, match.start() - 45)
        end = min(len(text), match.end() + 45)
        snippet = text[start:end].strip()
        if not should_flag(locale, match.group(0), snippet):
            continue
        if snippet in seen:
            continue
        seen.add(snippet)
        findings.append({"offset": match.start(), "snippet": snippet})
        if len(findings) >= 20:
            break
    return findings


def is_under_localized_build_dir(path: Path, build_dir: Path) -> bool:
    relative_parts = path.relative_to(build_dir).parts
    return bool(relative_parts) and relative_parts[0] in LOCALIZED_BUILD_DIRS


def iter_locale_html(build_dir: Path, locale: str) -> tuple[Path, list[Path]] | tuple[None, list[Path]]:
    if locale == "en":
        if not build_dir.exists():
            return None, []
        paths = [
            path
            for path in sorted(build_dir.rglob("*.html"))
            if not is_under_localized_build_dir(path, build_dir)
        ]
        return build_dir, paths

    locale_dir = build_dir / locale
    if not locale_dir.exists():
        return None, []
    return locale_dir, sorted(locale_dir.rglob("*.html"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", default="dist")
    parser.add_argument("--locales", nargs="+", default=["en", "ja"])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    build_dir = PROJECT_ROOT / args.build_dir
    result: dict[str, list[dict[str, object]]] = {}

    for locale in args.locales:
        locale_dir, html_paths = iter_locale_html(build_dir, locale)
        if locale_dir is None:
            result[locale] = [{"file": str(build_dir / locale), "missing": True}]
            continue
        entries: list[dict[str, object]] = []
        for path in html_paths:
            findings = audit_file(path, locale)
            if findings:
                entries.append(
                    {
                        "file": path.relative_to(PROJECT_ROOT).as_posix(),
                        "findings": findings,
                    }
                )
        result[locale] = entries

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        for locale, entries in result.items():
            print(f"{locale}: {len(entries)} files with Chinese text")
            for entry in entries[:50]:
                print(f"  {entry['file']}")
                for finding in entry.get("findings", [])[:3]:
                    print(f"    - {finding['snippet']}")

    return 1 if any(result.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
