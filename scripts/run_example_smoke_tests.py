#!/usr/bin/env python3
"""Run selected course code examples as real smoke tests.

The normal code-block audit checks syntax. This script goes one level deeper:
it extracts full tutorial scripts, runs them in temporary workspaces, and checks
that the expected outputs or files appear. The manifest intentionally avoids
network calls, credentials, and long training jobs.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "reports" / "code-example-smoke-tests.json"
FENCE_RE = re.compile(r"^```(?P<info>[^\n]*)\n(?P<body>.*?)(?:^```)", re.M | re.S)


@dataclass(frozen=True)
class Example:
    name: str
    kind: str
    command: list[str]
    markdown: str | None = None
    marker: str | None = None
    source: str | None = None
    script_name: str | None = None
    required_imports: list[str] = field(default_factory=list)
    expected_stdout: list[str] = field(default_factory=list)
    expected_files: list[str] = field(default_factory=list)
    timeout_seconds: int = 60


@dataclass
class Result:
    name: str
    status: str
    duration_seconds: float
    command: list[str]
    reason: str = ""
    stdout_tail: str = ""
    stderr_tail: str = ""
    workspace: str = ""


EXAMPLES: list[Example] = [
    Example(
        name="ch02_learning_assistant_cli",
        kind="markdown_script",
        markdown="src/content/docs/zh-cn/ch02-python/ch03-projects/05-hands-on-python-workshop.md",
        marker="创建 `learning_assistant_cli.py`",
        script_name="learning_assistant_cli.py",
        command=[
            "{python}",
            "learning_assistant_cli.py",
            "seed",
            "&&",
            "{python}",
            "learning_assistant_cli.py",
            "done",
            "2",
            "&&",
            "{python}",
            "learning_assistant_cli.py",
            "stats",
            "&&",
            "{python}",
            "learning_assistant_cli.py",
            "export",
        ],
        expected_stdout=["Completion rate: 33.3%", "Exported report"],
        expected_files=["ch02_output/tasks.json", "ch02_output/learning_report.md"],
    ),
    Example(
        name="ch04_math_workshop",
        kind="markdown_script",
        markdown="src/content/docs/zh-cn/ch04-ai-math/hands-on-math-workshop.md",
        marker="把下面代码保存到 `math_workshop.py`",
        script_name="math_workshop.py",
        command=["{python}", "math_workshop.py"],
        expected_stdout=["STEP 1: Vector similarity", "best_match=vector_similarity"],
        expected_files=[
            "ch04_math_workshop_evidence/vector_similarity.csv",
            "ch04_math_workshop_evidence/probability_simulation.csv",
            "ch04_math_workshop_evidence/gradient_descent.svg",
        ],
    ),
    Example(
        name="ch05_ml_workshop",
        kind="markdown_script",
        markdown="src/content/docs/zh-cn/ch05-machine-learning/ch06-projects/05-hands-on-ml-workshop.md",
        marker="把下面代码保存为 `ml_workshop.py`",
        script_name="ml_workshop.py",
        command=["{python}", "ml_workshop.py"],
        required_imports=["numpy", "pandas", "sklearn"],
        expected_stdout=["STEP 1: data prepared", "STEP 3: evidence files"],
        expected_files=[
            "ml_workshop_run/data/learning_tasks.csv",
            "ml_workshop_run/outputs/model_comparison.csv",
            "ml_workshop_run/outputs/error_samples.csv",
        ],
        timeout_seconds=90,
    ),
    Example(
        name="ch07_tokenizer_embedding_lab",
        kind="markdown_script",
        markdown="src/content/docs/zh-cn/ch07-llm-principles/ch01-nlp-crash/05-tokenizer-embedding-lab.md",
        marker="tokenizer_embedding_lab.py",
        script_name="tokenizer_embedding_lab.py",
        command=["{python}", "tokenizer_embedding_lab.py"],
        expected_stdout=["sentence_vec", "similarity(text 1, text 2): 0.949"],
    ),
    Example(
        name="ch07_prompt_eval_lab",
        kind="markdown_script",
        markdown="src/content/docs/zh-cn/ch07-llm-principles/ch05-prompt/05-prompt-evaluation-lab.md",
        marker="prompt_eval_lab.py",
        script_name="prompt_eval_lab.py",
        command=["{python}", "prompt_eval_lab.py"],
        expected_stdout=["v1_goal_only", "v3_with_examples", "pass_rate: 100%"],
    ),
    Example(
        name="ch07_alignment_safety_lab",
        kind="markdown_script",
        markdown="src/content/docs/zh-cn/ch07-llm-principles/ch07-alignment/04-safety-evaluation-lab.md",
        marker="alignment_safety_lab.py",
        script_name="alignment_safety_lab.py",
        command=["{python}", "alignment_safety_lab.py"],
        expected_stdout=["v1_too_permissive", "v3_balanced", "pass_rate: 100%"],
    ),
    Example(
        name="ch07_domain_finetune_demo",
        kind="markdown_script",
        markdown="src/content/docs/zh-cn/ch07-llm-principles/ch08-projects/01-domain-finetuning.md",
        marker="domain_finetune_demo.py",
        script_name="domain_finetune_demo.py",
        command=["{python}", "domain_finetune_demo.py"],
        expected_stdout=["coverage= 0.0", "coverage= 1.0", "sft_sample:"],
    ),
    Example(
        name="ch09_agent_workshop",
        kind="markdown_script",
        markdown="src/content/docs/zh-cn/ch09-agent/ch10-projects/04-stage-hands-on-workshop.md",
        marker="把下面代码复制到 `agent_workshop.py`",
        script_name="agent_workshop.py",
        command=["{python}", "agent_workshop.py"],
        expected_stdout=["status: completed", "status: blocked_by_approval", "passed: 3/3"],
        expected_files=["logs/agent_traces.jsonl"],
    ),
    Example(
        name="ch10_vision_workshop",
        kind="markdown_script",
        markdown="src/content/docs/zh-cn/ch10-computer-vision/ch06-projects/03-hands-on-vision-workshop.md",
        marker="vision_workshop.py",
        script_name="vision_workshop.py",
        command=["{python}", "vision_workshop.py"],
        required_imports=["cv2", "numpy"],
        expected_stdout=["classification_accuracy: 1.000", "failure_cases: 7"],
        expected_files=[
            "cv_workshop_run/data/labels.csv",
            "cv_workshop_run/reports/predictions.csv",
            "cv_workshop_run/reports/failure_cases.md",
        ],
        timeout_seconds=90,
    ),
    Example(
        name="ch11_nlp_workshop",
        kind="markdown_script",
        markdown="src/content/docs/zh-cn/ch11-nlp/ch07-projects/05-hands-on-nlp-workshop.md",
        marker="nlp_workshop.py",
        script_name="nlp_workshop.py",
        command=["{python}", "nlp_workshop.py"],
        expected_stdout=["classification_accuracy: 0.917", "retrieval_accuracy: 1.000"],
        expected_files=[
            "nlp_workshop_run/outputs/classification_predictions.csv",
            "nlp_workshop_run/outputs/qa_predictions.jsonl",
            "nlp_workshop_run/reports/failure_cases.md",
        ],
    ),
    Example(
        name="ch12_multimodal_workshop",
        kind="markdown_script",
        markdown="src/content/docs/zh-cn/ch12-multimodal/ch05-projects/02-hands-on-multimodal-workshop.md",
        marker="multimodal_workshop.py",
        script_name="multimodal_workshop.py",
        command=["{python}", "multimodal_workshop.py"],
        expected_stdout=["svg_assets: 3", "review_passed: 2/3", "failure_cases: 1"],
        expected_files=[
            "multimodal_workshop_run/prompts/prompt_plan.json",
            "multimodal_workshop_run/reports/asset_manifest.csv",
            "multimodal_workshop_run/outputs/export_preview.html",
        ],
    ),
    Example(
        name="ch13_mini_gpt2_train_cpu_smoke",
        kind="source_script",
        source="public/examples/ch13-open-llm-lab/mini_gpt2_train.py",
        script_name="mini_gpt2_train.py",
        command=[
            "{python}",
            "mini_gpt2_train.py",
            "--output-dir",
            "mini_gpt2_smoke",
            "--device",
            "cpu",
            "--steps",
            "3",
            "--batch-size",
            "4",
            "--sample-tokens",
            "16",
        ],
        required_imports=["torch"],
        expected_stdout=["step 0001", "checkpoint:", "--- sample ---"],
        expected_files=[
            "mini_gpt2_smoke/environment_report.json",
            "mini_gpt2_smoke/training_log.csv",
            "mini_gpt2_smoke/mini_gpt2_checkpoint.pt",
            "mini_gpt2_smoke/sample.txt",
        ],
        timeout_seconds=120,
    ),
]


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def tail(value: str, limit: int = 1800) -> str:
    return value[-limit:]


def has_import(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def python_blocks_after_marker(path: Path, marker: str) -> list[str]:
    text = path.read_text(encoding="utf-8")
    marker_index = text.find(marker)
    if marker_index == -1:
        raise ValueError(f"marker not found: {marker!r}")

    blocks: list[str] = []
    for match in FENCE_RE.finditer(text, marker_index):
        lang = match.group("info").strip().split()[0].lower() if match.group("info").strip() else ""
        lang = lang.removeprefix("{").removesuffix("}")
        if lang in {"python", "py"}:
            blocks.append(match.group("body").rstrip() + "\n")
    return blocks


def extract_markdown_script(example: Example) -> str:
    if not example.markdown or not example.marker:
        raise ValueError(f"{example.name}: markdown and marker are required")
    path = ROOT / example.markdown
    blocks = python_blocks_after_marker(path, example.marker)
    if not blocks:
        raise ValueError(f"{example.name}: no Python block after marker")

    runnable = [
        block
        for block in blocks
        if "if __name__" in block or len(block.splitlines()) >= 30
    ]
    return max(runnable or blocks, key=lambda block: len(block.splitlines()))


def materialize_example(example: Example, workspace: Path) -> Path:
    script_name = example.script_name or "example.py"
    target = workspace / script_name
    if example.kind == "markdown_script":
        target.write_text(extract_markdown_script(example), encoding="utf-8")
    elif example.kind == "source_script":
        if not example.source:
            raise ValueError(f"{example.name}: source is required")
        shutil.copy2(ROOT / example.source, target)
    else:
        raise ValueError(f"{example.name}: unknown kind {example.kind!r}")
    return target


def expand_command(command: list[str]) -> tuple[list[str], bool]:
    expanded = [sys.executable if part == "{python}" else part for part in command]
    if "&&" not in expanded:
        return expanded, False
    shell_command = " ".join(subprocess.list2cmdline([part]) for part in expanded)
    return [shell_command], True


def run_example(example: Example, keep_workspaces: bool, strict: bool) -> Result:
    missing = [name for name in example.required_imports if not has_import(name)]
    if missing and not strict:
        return Result(
            name=example.name,
            status="skip",
            duration_seconds=0.0,
            command=example.command,
            reason=f"missing optional imports: {', '.join(missing)}",
        )
    if missing:
        return Result(
            name=example.name,
            status="fail",
            duration_seconds=0.0,
            command=example.command,
            reason=f"missing required imports: {', '.join(missing)}",
        )

    workspace = Path(tempfile.mkdtemp(prefix=f"{example.name}-"))
    start = time.monotonic()
    command, use_shell = expand_command(example.command)
    env = os.environ.copy()
    env.setdefault("PYTHONHASHSEED", "0")
    env.setdefault("MPLBACKEND", "Agg")

    try:
        materialize_example(example, workspace)
        completed = subprocess.run(
            command[0] if use_shell else command,
            cwd=workspace,
            env=env,
            shell=use_shell,
            text=True,
            capture_output=True,
            timeout=example.timeout_seconds,
            check=False,
        )
        duration = time.monotonic() - start
        stdout = completed.stdout
        stderr = completed.stderr

        if completed.returncode != 0:
            return Result(
                name=example.name,
                status="fail",
                duration_seconds=duration,
                command=command,
                reason=f"exit code {completed.returncode}",
                stdout_tail=tail(stdout),
                stderr_tail=tail(stderr),
                workspace=str(workspace) if keep_workspaces else "",
            )

        missing_stdout = [needle for needle in example.expected_stdout if needle not in stdout]
        if missing_stdout:
            return Result(
                name=example.name,
                status="fail",
                duration_seconds=duration,
                command=command,
                reason=f"missing stdout markers: {missing_stdout}",
                stdout_tail=tail(stdout),
                stderr_tail=tail(stderr),
                workspace=str(workspace) if keep_workspaces else "",
            )

        missing_files = [path for path in example.expected_files if not (workspace / path).exists()]
        if missing_files:
            return Result(
                name=example.name,
                status="fail",
                duration_seconds=duration,
                command=command,
                reason=f"missing files: {missing_files}",
                stdout_tail=tail(stdout),
                stderr_tail=tail(stderr),
                workspace=str(workspace) if keep_workspaces else "",
            )

        return Result(
            name=example.name,
            status="pass",
            duration_seconds=duration,
            command=command,
            stdout_tail=tail(stdout, 900),
            stderr_tail=tail(stderr, 900),
            workspace=str(workspace) if keep_workspaces else "",
        )
    except subprocess.TimeoutExpired as exc:
        duration = time.monotonic() - start
        return Result(
            name=example.name,
            status="fail",
            duration_seconds=duration,
            command=command,
            reason=f"timeout after {example.timeout_seconds}s",
            stdout_tail=tail(exc.stdout or ""),
            stderr_tail=tail(exc.stderr or ""),
            workspace=str(workspace) if keep_workspaces else "",
        )
    except Exception as exc:
        duration = time.monotonic() - start
        return Result(
            name=example.name,
            status="fail",
            duration_seconds=duration,
            command=command,
            reason=f"{type(exc).__name__}: {exc}",
            workspace=str(workspace) if keep_workspaces else "",
        )
    finally:
        if not keep_workspaces:
            shutil.rmtree(workspace, ignore_errors=True)


def write_report(results: list[Result], elapsed: float) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "generated_by": rel(Path(__file__)),
        "elapsed_seconds": round(elapsed, 3),
        "summary": {
            "total": len(results),
            "passed": sum(result.status == "pass" for result in results),
            "skipped": sum(result.status == "skip" for result in results),
            "failed": sum(result.status == "fail" for result in results),
        },
        "results": [asdict(result) for result in results],
    }
    REPORT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List example names and exit.")
    parser.add_argument("--only", action="append", default=[], help="Run only examples whose name contains this text. Repeatable.")
    parser.add_argument("--strict", action="store_true", help="Fail instead of skipping examples with missing optional imports.")
    parser.add_argument("--keep-workspaces", action="store_true", help="Keep temporary workspaces for failed or passed examples.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected = EXAMPLES
    if args.only:
        selected = [
            example
            for example in EXAMPLES
            if any(token in example.name for token in args.only)
        ]

    if args.list:
        for example in selected:
            deps = f" deps={','.join(example.required_imports)}" if example.required_imports else ""
            print(f"{example.name}{deps}")
        return 0

    if not selected:
        print("No examples selected.")
        return 2

    start = time.monotonic()
    results: list[Result] = []
    for example in selected:
        result = run_example(example, keep_workspaces=args.keep_workspaces, strict=args.strict)
        results.append(result)
        duration = f"{result.duration_seconds:.2f}s"
        if result.status == "pass":
            print(f"PASS {example.name} ({duration})")
        elif result.status == "skip":
            print(f"SKIP {example.name}: {result.reason}")
        else:
            print(f"FAIL {example.name}: {result.reason}")
            if result.stdout_tail:
                print("--- stdout tail ---")
                print(result.stdout_tail)
            if result.stderr_tail:
                print("--- stderr tail ---")
                print(result.stderr_tail)

    elapsed = time.monotonic() - start
    write_report(results, elapsed)
    passed = sum(result.status == "pass" for result in results)
    skipped = sum(result.status == "skip" for result in results)
    failed = sum(result.status == "fail" for result in results)
    print(f"example_smoke_tests={len(results)} passed={passed} skipped={skipped} failed={failed}")
    print(f"report={rel(REPORT_PATH)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
