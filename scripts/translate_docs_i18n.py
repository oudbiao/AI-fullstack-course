#!/usr/bin/env python3
"""Translate Docusaurus docs into locale folders with an OpenAI-compatible API.

The script is intentionally resumable:
- Existing translated files are skipped unless --overwrite is used.
- Failed files are recorded in static/i18n-translation-failures.jsonl.
- Markdown fences are checked before writing to avoid obvious broken output.

It translates visible Markdown/MDX text, including headings, paragraphs,
frontmatter title/description/keywords, table text, Mermaid labels, comments,
and learner-facing strings. It asks the model to preserve code syntax, paths,
URLs, image references, and MDX/Docusaurus structure.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = PROJECT_ROOT / "docs"
I18N_DOCS_DIR = Path("docusaurus-plugin-content-docs") / "current"
FAILURE_LOG = PROJECT_ROOT / "static" / "i18n-translation-failures.jsonl"

LOCALE_CONFIG = {
    "en": {
        "name": "English",
        "instruction": (
            "Translate all Simplified Chinese learner-facing text into natural, clear English. "
            "Keep common AI terms such as Transformer, RAG, Agent, Prompt, PyTorch, NumPy, "
            "Pandas, SQL, LLM, API, and Git in standard English form."
        ),
    },
    "ja": {
        "name": "Japanese",
        "instruction": (
            "Translate all Simplified Chinese learner-facing text into natural, beginner-friendly Japanese. "
            "Keep common AI terms such as Transformer, RAG, Agent, Prompt, PyTorch, NumPy, "
            "Pandas, SQL, LLM, API, and Git in their standard Japanese/English technical form."
        ),
    },
}


def load_local_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def iter_docs() -> list[Path]:
    return sorted(
        path
        for path in DOCS_ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in {".md", ".mdx"}
    )


def sort_docs(paths: list[Path], mode: str) -> list[Path]:
    if mode == "size":
        return sorted(paths, key=lambda path: (path.stat().st_size, path.as_posix()))
    return sorted(paths)


def markdown_fence_count(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.lstrip().startswith("```"))


def has_frontmatter(text: str) -> bool:
    return text.startswith("---\n") or text.startswith("---\r\n")


def validate_translation(source: str, translated: str) -> list[str]:
    issues: list[str] = []
    if not translated.strip():
        issues.append("empty output")
    if markdown_fence_count(source) != markdown_fence_count(translated):
        issues.append(
            f"code fence count changed: {markdown_fence_count(source)} -> {markdown_fence_count(translated)}"
        )
    if has_frontmatter(source) and not has_frontmatter(translated):
        issues.append("frontmatter missing")
    if "```" in translated and markdown_fence_count(translated) % 2 != 0:
        issues.append("unbalanced code fences")
    return issues


def build_prompt(source: str, locale: str, relative_path: str) -> list[dict[str, str]]:
    cfg = LOCALE_CONFIG[locale]
    system = f"""
You are a senior technical translator for an AI full-stack course.
Target language: {cfg["name"]}.
{cfg["instruction"]}

Rules:
1. Return only the translated Markdown/MDX. Do not wrap it in extra fences.
2. Preserve the document structure, heading levels, lists, tables, admonitions, MDX syntax, imports, JSX tags, links, anchors, image paths, and URLs.
3. Preserve fenced code block languages and executable code syntax. Translate learner-facing comments, docstrings, printed messages, Mermaid labels, diagram labels, and explanatory strings when safe.
4. Do not translate file paths, package names, command names, API field names, variable names, environment variable names, identifiers, URLs, image filenames, or markdown anchors unless they are pure natural-language prose.
5. Translate frontmatter values such as title, description, keywords, sidebar_label, and tags, but preserve frontmatter keys.
6. Keep mathematical formulas, LaTeX, shell commands, Python/R/SQL syntax, JSON keys, and YAML keys valid.
7. Maintain a beginner-friendly teaching tone.
""".strip()
    user = f"Translate this file for locale {locale}. Relative path: {relative_path}\n\n{source}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def chat_completion(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    timeout: int,
) -> str:
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.1,
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected API response: {data}") from exc


def log_failure(record: dict[str, Any]) -> None:
    FAILURE_LOG.parent.mkdir(parents=True, exist_ok=True)
    with FAILURE_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def translate_one(
    *,
    src_path: Path,
    locale: str,
    base_url: str,
    api_key: str,
    model: str,
    timeout: int,
    retries: int,
    overwrite: bool,
    dry_run: bool,
) -> str:
    rel = src_path.relative_to(DOCS_ROOT)
    out_path = PROJECT_ROOT / "i18n" / locale / I18N_DOCS_DIR / rel
    if out_path.exists() and not overwrite:
        return "skipped"

    source = src_path.read_text(encoding="utf-8")
    if dry_run:
        print(f"[dry-run] {locale}: {rel} -> {out_path.relative_to(PROJECT_ROOT)}")
        return "dry-run"

    print(
        f"[start] {locale}: {rel} ({src_path.stat().st_size} bytes)",
        flush=True,
    )
    messages = build_prompt(source, locale, rel.as_posix())
    last_error: str | None = None
    for attempt in range(1, retries + 2):
        try:
            translated = chat_completion(
                base_url=base_url,
                api_key=api_key,
                model=model,
                messages=messages,
                timeout=timeout,
            ).strip()
            issues = validate_translation(source, translated)
            if issues:
                raise RuntimeError("; ".join(issues))
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(translated + "\n", encoding="utf-8")
            return "translated"
        except Exception as exc:  # noqa: BLE001 - errors are logged and retried.
            last_error = str(exc)
            print(
                f"[warn] {locale}: {rel} attempt {attempt} failed: {last_error}",
                file=sys.stderr,
                flush=True,
            )
            if attempt <= retries:
                time.sleep(min(30, 2**attempt))

    log_failure(
        {
            "locale": locale,
            "path": rel.as_posix(),
            "model": model,
            "error": last_error,
            "time": int(time.time()),
        }
    )
    return "failed"


def main() -> int:
    load_local_env_file(PROJECT_ROOT / ".env.local")
    load_local_env_file(PROJECT_ROOT / ".env")

    parser = argparse.ArgumentParser()
    parser.add_argument("--locales", nargs="+", default=["en", "ja"], choices=sorted(LOCALE_CONFIG))
    parser.add_argument("--model", default=os.environ.get("OPENAI_TEXT_MODEL", "gpt-5.4-mini"))
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--timeout", type=int, default=int(os.environ.get("OPENAI_TEXT_TIMEOUT", "180")))
    parser.add_argument("--retries", type=int, default=int(os.environ.get("OPENAI_TEXT_RETRIES", "2")))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-bytes", type=int, default=0)
    parser.add_argument("--sort", choices=["path", "size"], default="path")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--only", nargs="*", default=[])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key and not args.dry_run:
        raise SystemExit("OPENAI_API_KEY is not set.")

    docs = iter_docs()
    if args.only:
        wanted = {Path(p).as_posix().removeprefix("docs/") for p in args.only}
        docs = [p for p in docs if p.relative_to(DOCS_ROOT).as_posix() in wanted]
    if args.max_bytes:
        docs = [p for p in docs if p.stat().st_size <= args.max_bytes]
    docs = sort_docs(docs, args.sort)
    if args.shard_count < 1:
        raise SystemExit("--shard-count must be >= 1")
    if not 0 <= args.shard_index < args.shard_count:
        raise SystemExit("--shard-index must be between 0 and shard-count - 1")
    if args.shard_count > 1:
        docs = [p for idx, p in enumerate(docs) if idx % args.shard_count == args.shard_index]
    if args.limit:
        docs = docs[: args.limit]

    summary: dict[str, dict[str, int]] = {}
    for locale in args.locales:
        counts = {"translated": 0, "skipped": 0, "failed": 0, "dry-run": 0}
        for src_path in docs:
            status = translate_one(
                src_path=src_path,
                locale=locale,
                base_url=args.base_url,
                api_key=api_key,
                model=args.model,
                timeout=args.timeout,
                retries=args.retries,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
            )
            counts[status] = counts.get(status, 0) + 1
            rel = src_path.relative_to(DOCS_ROOT).as_posix()
            print(f"{locale}: {status}: {rel}", flush=True)
        summary[locale] = counts

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if all(v.get("failed", 0) == 0 for v in summary.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
