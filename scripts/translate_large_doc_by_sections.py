#!/usr/bin/env python3
"""Translate one large Markdown document by top-level sections.

This is a fallback for long docs that are too large for a single chat
completion request. It writes into the Starlight localized content tree and
reuses the same translation prompt/validation helpers as the main batch
translator.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import translate_docs_i18n as batch


def split_by_heading(text: str, heading_prefix: str) -> list[str]:
    lines = text.splitlines(keepends=True)
    chunks: list[str] = []
    start = 0
    for index, line in enumerate(lines):
        if index > 0 and line.startswith(heading_prefix):
            chunks.append("".join(lines[start:index]))
            start = index
    chunks.append("".join(lines[start:]))
    return [chunk for chunk in chunks if chunk]


def translate_chunk(
    *,
    chunk: str,
    locale: str,
    rel_path: str,
    chunk_index: int,
    total_chunks: int,
    base_url: str,
    api_key: str,
    model: str,
    timeout: int,
    retries: int,
) -> str:
    chunk_rel = f"{rel_path} section {chunk_index}/{total_chunks}"
    messages = batch.build_prompt(chunk, locale, chunk_rel)
    last_error: str | None = None
    for attempt in range(1, retries + 2):
        try:
            translated = batch.chat_completion(
                base_url=base_url,
                api_key=api_key,
                model=model,
                messages=messages,
                timeout=timeout,
            ).strip()
            issues = batch.validate_translation(chunk, translated)
            if issues:
                raise RuntimeError("; ".join(issues))
            return translated
        except Exception as exc:  # noqa: BLE001 - logged and retried.
            last_error = str(exc)
            print(
                f"[warn] {locale}: {rel_path} section {chunk_index}/{total_chunks} "
                f"attempt {attempt} failed: {last_error}",
                file=sys.stderr,
                flush=True,
            )
            if attempt <= retries:
                time.sleep(min(30, 2**attempt))
    raise RuntimeError(last_error or "unknown translation error")


def main() -> int:
    batch.load_local_env_file(batch.PROJECT_ROOT / ".env.local")
    batch.load_local_env_file(batch.PROJECT_ROOT / ".env")

    parser = argparse.ArgumentParser()
    parser.add_argument("--locale", required=True, choices=sorted(batch.LOCALE_CONFIG))
    parser.add_argument("--path", required=True)
    parser.add_argument("--heading-prefix", default="## ")
    parser.add_argument("--model", default=batch.os.environ.get("OPENAI_TEXT_MODEL", "gpt-5.4-mini"))
    parser.add_argument("--base-url", default=batch.os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--timeout", type=int, default=int(batch.os.environ.get("OPENAI_TEXT_TIMEOUT", "180")))
    parser.add_argument("--retries", type=int, default=int(batch.os.environ.get("OPENAI_TEXT_RETRIES", "2")))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    api_key = batch.os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set.")

    rel_path = (
        Path(args.path)
        .as_posix()
        .removeprefix("src/content/docs/")
        .removeprefix("docs/")
    )
    src_path = batch.DOCS_ROOT / rel_path
    out_path = batch.DOCS_ROOT / args.locale / rel_path
    if out_path.exists() and not args.overwrite:
        print(f"{args.locale}: skipped: {rel_path}")
        return 0

    source = src_path.read_text(encoding="utf-8")
    chunks = split_by_heading(source, args.heading_prefix)
    print(f"[start] {args.locale}: {rel_path} split into {len(chunks)} chunks", flush=True)

    translated_chunks: list[str] = []
    for index, chunk in enumerate(chunks, 1):
        print(
            f"[chunk] {args.locale}: {rel_path} {index}/{len(chunks)} "
            f"({len(chunk.encode('utf-8'))} bytes)",
            flush=True,
        )
        translated_chunks.append(
            translate_chunk(
                chunk=chunk,
                locale=args.locale,
                rel_path=rel_path,
                chunk_index=index,
                total_chunks=len(chunks),
                base_url=args.base_url,
                api_key=api_key,
                model=args.model,
                timeout=args.timeout,
                retries=args.retries,
            )
        )

    translated = "\n\n".join(part.rstrip() for part in translated_chunks).rstrip() + "\n"
    issues = batch.validate_translation(source, translated)
    if issues:
        raise SystemExit("; ".join(issues))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(translated, encoding="utf-8")
    print(f"{args.locale}: translated: {rel_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
