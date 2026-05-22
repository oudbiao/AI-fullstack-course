#!/usr/bin/env python3
"""Clean leftover source-language text from localized Markdown files."""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import translate_docs_i18n as batch

HAN_RE = re.compile(r"[\u4e00-\u9fff]")
JA_SOURCE_RESIDUAL_RE = re.compile(
    r"[这们习说让给还没试变图据读页导么吗]"
    r"|怎么|为什么|什么|如果|然后|这个|一个|没有|可以"
    r"|应用模式|学习路线|课程内容|机器学习|深度学习|人工智能"
    r"|学前导读|数据读写|数据选择|数据清洗|数据转换|数据合并"
)


def normalize_base_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        return base
    return f"{base}/v1"


def split_by_heading(text: str, max_bytes: int) -> list[str]:
    if len(text.encode("utf-8")) <= max_bytes:
        return [text]

    lines = text.splitlines(keepends=True)
    chunks: list[str] = []
    start = 0
    for index, line in enumerate(lines):
        if index > 0 and line.startswith("## "):
            chunk = "".join(lines[start:index])
            if chunk:
                chunks.append(chunk)
            start = index
    tail = "".join(lines[start:])
    if tail:
        chunks.append(tail)

    merged: list[str] = []
    current = ""
    for chunk in chunks:
        if current and len((current + chunk).encode("utf-8")) > max_bytes:
            merged.append(current)
            current = chunk
        else:
            current += chunk
    if current:
        merged.append(current)
    return merged


def build_cleanup_prompt(text: str, locale: str, rel_path: str) -> list[dict[str, str]]:
    if locale == "en":
        system = """
You are cleaning an already-English Markdown/MDX localization for an AI course.

Goal:
- Replace leftover Simplified Chinese text with natural English.
- Translate Chinese strings inside examples, comments, dictionaries, tables, Mermaid labels, frontmatter values, image alt text, and prose.

Rules:
1. Return only the cleaned Markdown/MDX. Do not add wrappers or explanations.
2. Preserve Markdown structure, frontmatter keys, heading levels, tables, admonitions, MDX syntax, code fences, links, anchors, and image file paths.
3. Keep code syntax valid. You may translate data literals and printed strings if doing so does not break the example.
4. Do not translate package names, command names, API field names, variable names, environment variable names, filenames, URLs, or image paths.
5. If a Chinese phrase is intentionally used as a language sample, translate the surrounding explanation and use an English equivalent sample unless the lesson explicitly needs a Chinese-language example.
6. Keep mathematical formulas, shell commands, Python/SQL/JSON/YAML syntax valid.
""".strip()
        user = f"Clean leftover Chinese from this English localization. Relative path: {rel_path}\n\n{text}"
    elif locale == "ja":
        system = """
You are cleaning an already-Japanese Markdown/MDX localization for an AI course.

Goal:
- Replace leftover Simplified Chinese text with natural Japanese.
- Translate Chinese strings inside examples, comments, dictionaries, tables, Mermaid labels, frontmatter values, image alt text, and prose.

Rules:
1. Return only the cleaned Markdown/MDX. Do not add wrappers or explanations.
2. Preserve Markdown structure, frontmatter keys, heading levels, tables, admonitions, MDX syntax, code fences, links, anchors, and image file paths.
3. Keep code syntax valid. You may translate data literals and printed strings if doing so does not break the example.
4. Do not translate package names, command names, API field names, variable names, environment variable names, filenames, URLs, image paths, or markdown anchors.
5. If a Chinese phrase is intentionally used as a language sample, translate the surrounding explanation and use a Japanese equivalent sample unless the lesson explicitly needs a Chinese-language example.
6. Keep mathematical formulas, shell commands, Python/SQL/JSON/YAML syntax valid.
7. Natural Japanese Kanji is allowed. Only remove leftover Simplified Chinese phrasing.
""".strip()
        user = f"Clean leftover Chinese from this Japanese localization. Relative path: {rel_path}\n\n{text}"
    else:
        raise ValueError(f"Unsupported locale: {locale}")
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def clean_chunk(
    *,
    text: str,
    locale: str,
    rel_path: str,
    base_url: str,
    api_key: str,
    model: str,
    timeout: int,
    retries: int,
) -> str:
    messages = build_cleanup_prompt(text, locale, rel_path)
    last_error: str | None = None
    for attempt in range(1, retries + 2):
        try:
            cleaned = batch.chat_completion(
                base_url=base_url,
                api_key=api_key,
                model=model,
                messages=messages,
                timeout=timeout,
            ).strip()
            issues = batch.validate_translation(text, cleaned)
            if issues:
                raise RuntimeError("; ".join(issues))
            return cleaned
        except Exception as exc:  # noqa: BLE001 - logged and retried.
            last_error = str(exc)
            print(
                f"[warn] {locale}: {rel_path} cleanup attempt {attempt} failed: {last_error}",
                file=sys.stderr,
                flush=True,
            )
            if attempt <= retries:
                time.sleep(min(30, 2**attempt))
    raise RuntimeError(last_error or "unknown cleanup error")


def localized_root(locale: str) -> Path:
    return batch.DOCS_ROOT / locale


def iter_targets(locale: str, only: list[str]) -> list[Path]:
    root = localized_root(locale)
    if only:
        return [
            root
            / Path(path)
            .as_posix()
            .removeprefix("src/content/docs/")
            .removeprefix("docs/")
            for path in only
        ]
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in {".md", ".mdx"})


def needs_cleanup(locale: str, text: str) -> bool:
    if locale == "ja":
        return bool(JA_SOURCE_RESIDUAL_RE.search(text))
    return bool(HAN_RE.search(text))


def main() -> int:
    batch.load_local_env_file(batch.PROJECT_ROOT / ".env.local")
    batch.load_local_env_file(batch.PROJECT_ROOT / ".env")

    parser = argparse.ArgumentParser()
    parser.add_argument("--locale", default="en", choices=["en", "ja"])
    parser.add_argument("--only", nargs="*", default=[])
    parser.add_argument("--model", default=batch.os.environ.get("OPENAI_TEXT_MODEL", "gpt-5.4-mini"))
    parser.add_argument("--base-url", default=batch.os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--max-chunk-bytes", type=int, default=12000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    api_key = batch.os.environ.get("OPENAI_API_KEY", "")
    if not api_key and not args.dry_run:
        raise SystemExit("OPENAI_API_KEY is not set.")

    root = localized_root(args.locale)
    base_url = normalize_base_url(args.base_url)
    counts = {"cleaned": 0, "skipped": 0, "failed": 0}
    for path in iter_targets(args.locale, args.only):
        if not path.exists():
            print(f"[missing] {path.relative_to(batch.PROJECT_ROOT)}", file=sys.stderr)
            counts["failed"] += 1
            continue
        text = path.read_text(encoding="utf-8")
        if not needs_cleanup(args.locale, text):
            counts["skipped"] += 1
            continue
        rel_path = path.relative_to(root).as_posix()
        print(f"[start] {args.locale}: {rel_path}", flush=True)
        if args.dry_run:
            counts["cleaned"] += 1
            continue
        try:
            chunks = split_by_heading(text, args.max_chunk_bytes)
            cleaned_chunks = [
                clean_chunk(
                    text=chunk,
                    locale=args.locale,
                    rel_path=f"{rel_path} chunk {index}/{len(chunks)}",
                    base_url=base_url,
                    api_key=api_key,
                    model=args.model,
                    timeout=args.timeout,
                    retries=args.retries,
                )
                for index, chunk in enumerate(chunks, 1)
            ]
            cleaned = "\n\n".join(chunk.rstrip() for chunk in cleaned_chunks).rstrip() + "\n"
            path.write_text(cleaned, encoding="utf-8")
            counts["cleaned"] += 1
        except Exception as exc:  # noqa: BLE001 - continue through all targets.
            print(f"[failed] {args.locale}: {rel_path}: {exc}", file=sys.stderr, flush=True)
            counts["failed"] += 1
    print(counts)
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
