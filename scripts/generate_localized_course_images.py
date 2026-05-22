#!/usr/bin/env python3
"""Generate localized course images over existing placeholder files.

This script consumes `reports/course-images/localized-placeholder-pending.txt`.
It starts from the localized placeholder filenames, derives an image job from the
original course image job, and overwrites the placeholder with an image2 result.
Successful filenames are removed from the pending list so the process can be
resumed safely.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from generate_course_images import (
    DEFAULT_BASE_URL,
    DEFAULT_COURSE_IMAGE_QUALITY,
    DEFAULT_COURSE_IMAGE_SIZE,
    DEFAULT_IMAGE_RETRIES,
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_REPORT_DIR,
    DEFAULT_REQUEST_TIMEOUT,
    IMAGE_JOBS,
    VERTICAL_REFINEMENT_INSTRUCTIONS,
    available_api_keys,
    generate_image_with_http,
    set_user_readable_permissions,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PENDING_FILE = PROJECT_ROOT / "reports" / "course-images" / "localized-placeholder-pending.txt"
IMAGE_RE = re.compile(r"!\[([^\]]*)]\((/img/course/[^)\s]+)")
HOMEPAGE_LOCALIZED_RE = re.compile(r"^homepage-ai-history-comic-(en|ja)-(\d{2}-.+\.png)$")
LOCALE_NAMES = {
    "en": "English",
    "ja": "Japanese",
}
LOCALE_INSTRUCTIONS = {
    "en": """
Create the English localized final image for this course illustration.
All user-facing labels, captions, bubbles, and callouts in the image must be clear natural English.
Keep standard technical terms, code identifiers, formulas, API names, and model names in English as appropriate.
If the source prompt asks for Chinese labels, replace those labels with English.
Recompose the page vertically; do not stretch, squeeze, or warp the source image or its text. If the localized text is too long, shorten labels while keeping the same teaching meaning.
Avoid dense tiny text, gibberish, watermark, and real brand logos.
""".strip(),
    "ja": """
この教材図解の日本語ローカライズ版の最終画像を作成してください。
画像内の見出し、ラベル、吹き出し、注釈は自然な日本語にしてください。
標準的な技術用語、コード、数式、API 名、モデル名は必要に応じて英語表記のまま残してください。
元のプロンプトに中国語ラベル指定がある場合は、日本語ラベルに置き換えてください。
縦長ページとして再構成してください。元画像や文字を縦方向に引き伸ばしたり、横幅を圧縮したり、歪ませたりしないでください。日本語ラベルが長すぎる場合は、意味を保ったまま短くしてください。
小さすぎる文字、文字化け、透かし、実在ブランドロゴは禁止です。
""".strip(),
}


def read_pending(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_pending(path: Path, filenames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(filenames) + ("\n" if filenames else ""), encoding="utf-8")


def base_filename_for(target: str, locale: str) -> str:
    homepage_match = HOMEPAGE_LOCALIZED_RE.match(target)
    if homepage_match:
        return f"homepage-ai-history-comic-{homepage_match.group(2)}"
    suffix = f"-{locale}.png"
    if target.endswith(suffix):
        return f"{target[: -len(suffix)]}.png"
    return target


def image_job_map() -> dict[str, dict[str, Any]]:
    return {str(job["filename"]): job for job in IMAGE_JOBS}


def referenced_alt_texts(locale: str) -> dict[str, str]:
    root = PROJECT_ROOT / "src" / "content" / "docs" / locale
    alt_map: dict[str, str] = {}
    if not root.exists():
        return alt_map
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in {".md", ".mdx"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for match in IMAGE_RE.finditer(text):
            alt, url = match.groups()
            filename = url.removeprefix("/img/course/")
            alt_map.setdefault(filename, alt.strip())
    return alt_map


def direct_job_for(target: str, jobs: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    job = jobs.get(target)
    if not job:
        return None
    return dict(job)


def derived_job_for(target: str, locale: str, jobs: dict[str, dict[str, Any]], alt_map: dict[str, str]) -> dict[str, Any]:
    direct_job = direct_job_for(target, jobs)
    if direct_job:
        return direct_job

    base_filename = base_filename_for(target, locale)
    base_job = jobs.get(base_filename)
    if not base_job:
        raise KeyError(f"No source IMAGE_JOBS entry for {target} (base: {base_filename})")

    language = LOCALE_NAMES.get(locale, locale)
    alt = alt_map.get(target, base_job.get("alt", ""))
    source_prompt = str(base_job["prompt"])
    prompt = f"""
{LOCALE_INSTRUCTIONS[locale]}

Target language: {language}
Target filename: {target}
Localized alt text: {alt}

Use the same teaching idea, visual structure, and level of detail as the source course image prompt below, but localize visible user-facing text into {language}.

Source image prompt:
{source_prompt}
""".strip()

    return {
        **base_job,
        "filename": target,
        "title": f"{language} localized image: {base_job.get('title', base_filename)}",
        "alt": alt,
        "prompt": prompt,
    }


def select_targets(args: argparse.Namespace) -> list[str]:
    pending = read_pending(Path(args.pending_file))
    if args.only:
        selected = [name for name in args.only if name in pending or args.ignore_pending]
    else:
        selected = [
            name
            for name in pending
            if name.endswith(f"-{args.locale}.png") or name.startswith(f"homepage-ai-history-comic-{args.locale}-")
        ]

    if args.start_after:
        try:
            selected = selected[selected.index(args.start_after) + 1 :]
        except ValueError:
            raise SystemExit(f"--start-after target not found: {args.start_after}") from None

    if args.limit:
        selected = selected[: args.limit]
    return selected


def append_progress(report_dir: Path, row: dict[str, Any]) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    log_path = report_dir / "localized-generation-progress.jsonl"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def remove_successes_from_pending(pending_file: Path, successes: set[str]) -> None:
    if not successes:
        return
    pending = read_pending(pending_file)
    write_pending(pending_file, [name for name in pending if name not in successes])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--locale", choices=sorted(LOCALE_NAMES), default="ja")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument(
        "--allow-batch",
        action="store_true",
        help="Allow --limit greater than 1. By default this runner only generates one image per invocation.",
    )
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--start-after")
    parser.add_argument("--pending-file", default=str(PENDING_FILE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--request-timeout", type=int, default=DEFAULT_REQUEST_TIMEOUT)
    parser.add_argument("--retries", type=int, default=DEFAULT_IMAGE_RETRIES)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--ignore-pending", action="store_true")
    parser.add_argument(
        "--force-vertical",
        action="store_true",
        help="Force selected localized jobs to render as native 1024x1792 vertical course images for QA refinement.",
    )
    parser.add_argument("--parallel-per-key", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.limit != 1 and not args.allow_batch:
        raise SystemExit("This runner is single-image by default. Use --allow-batch only when the image backend supports batches.")

    targets = select_targets(args)
    jobs = image_job_map()
    alt_map = referenced_alt_texts(args.locale)
    output_dir = Path(args.output_dir)
    report_dir = Path(args.report_dir)

    print(f"locale: {args.locale}", flush=True)
    print(f"model: {args.model}", flush=True)
    print(f"base_url: {args.base_url}", flush=True)
    print(f"targets: {len(targets)}", flush=True)
    print(f"pending_file: {args.pending_file}", flush=True)
    api_keys = available_api_keys()
    worker_count = min(len(api_keys), len(targets)) if args.parallel_per_key and targets else 1
    print(f"api_keys: {len(api_keys)}", flush=True)
    print(f"workers: {worker_count}", flush=True)

    successes: set[str] = set()
    errors: list[dict[str, str]] = []

    def generate_target(target: str, request_api_keys: list[str]) -> tuple[str, str | None]:
        try:
            job = derived_job_for(target, args.locale, jobs, alt_map)
        except Exception as exc:
            return target, str(exc)
        if args.force_vertical:
            job["size"] = DEFAULT_COURSE_IMAGE_SIZE
            job["quality"] = DEFAULT_COURSE_IMAGE_QUALITY
            job["prompt"] = f"{job['prompt']}\n\n{VERTICAL_REFINEMENT_INSTRUCTIONS}"

        if args.dry_run:
            print(f"DRY RUN: {target} ({job.get('size')}, {job.get('quality')})", flush=True)
            return target, None

        if not request_api_keys:
            return target, "OPENAI_API_KEY is not set."

        output_path = output_dir / target
        print(f"Generating {target}...", flush=True)
        try:
            output_path.write_bytes(
                generate_image_with_http(
                    api_keys=request_api_keys,
                    base_url=args.base_url,
                    model=args.model,
                    job=job,
                    retries=args.retries,
                    request_timeout=args.request_timeout,
                )
            )
            set_user_readable_permissions(output_path)
            append_progress(
                report_dir,
                {
                    "at": datetime.now(timezone.utc).isoformat(),
                    "filename": target,
                    "locale": args.locale,
                    "status": "generated",
                },
            )
            print(f"Saved {output_path}", flush=True)
            return target, None
        except Exception as exc:
            append_progress(
                report_dir,
                {
                    "at": datetime.now(timezone.utc).isoformat(),
                    "filename": target,
                    "locale": args.locale,
                    "status": "error",
                    "error": str(exc),
                },
            )
            return target, str(exc)

    if worker_count > 1 and not args.dry_run:
        target_queue: queue.Queue[str] = queue.Queue()
        for target in targets:
            target_queue.put(target)

        result_lock = threading.Lock()
        stop_event = threading.Event()

        def key_worker(worker_index: int, api_key: str) -> None:
            while not stop_event.is_set():
                try:
                    target = target_queue.get_nowait()
                except queue.Empty:
                    return
                try:
                    print(f"worker {worker_index + 1}: {target}", flush=True)
                    target, error = generate_target(target, [api_key])
                    with result_lock:
                        if error:
                            errors.append({"filename": target, "error": error})
                            print(f"Failed {target}: {error}", flush=True)
                            if not args.continue_on_error:
                                stop_event.set()
                        else:
                            successes.add(target)
                finally:
                    target_queue.task_done()

        threads = [
            threading.Thread(target=key_worker, args=(index, api_key), daemon=False)
            for index, api_key in enumerate(api_keys[:worker_count])
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        remove_successes_from_pending(Path(args.pending_file), successes)
    else:
        for target in targets:
            target, error = generate_target(target, api_keys)
            if error:
                errors.append({"filename": target, "error": error})
                print(f"Failed {target}: {error}", flush=True)
                if not args.continue_on_error:
                    break
            else:
                successes.add(target)
                remove_successes_from_pending(Path(args.pending_file), successes)

    print(json.dumps({"successes": len(successes), "errors": errors}, ensure_ascii=False, indent=2), flush=True)
    return 1 if errors and not args.continue_on_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
