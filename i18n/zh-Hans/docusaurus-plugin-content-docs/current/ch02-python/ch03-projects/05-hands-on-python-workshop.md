---
title: "2.3.5 跟做工作坊：构建本地学习任务助手"
sidebar_position: 24
description: "第 2 章 Python 跟做实操：用 argparse、dataclass、JSON 持久化、异常处理和报告导出，构建一个可运行的命令行学习任务助手。"
keywords: [Python 实操, 命令行应用, argparse, JSON, dataclass, 文件读写, Python 项目]
---

# 2.3.5 跟做工作坊：构建本地学习任务助手

![Python 跟做工作坊路线图](/img/course/ch02-hands-on-python-workshop-route.webp)

:::tip 工作坊目标
这一页是第 2 章的实操桥梁。你不只是阅读语法说明，而是会做出一个小工具：创建学习任务、保存到 JSON、标记完成、查看统计，并导出 Markdown 报告。
:::

## 你会做出什么

你会构建一个名为 `learning_assistant_cli.py` 的命令行学习任务助手。它只使用 Python 标准库，不需要安装第三方包。

跟着步骤完成后，你可以运行这些命令：

```bash
python3 learning_assistant_cli.py seed
python3 learning_assistant_cli.py list
python3 learning_assistant_cli.py add "Practice command-line arguments" --stage 2.3 --tag argparse
python3 learning_assistant_cli.py done 2
python3 learning_assistant_cli.py stats
python3 learning_assistant_cli.py export
```

项目会生成：

| 文件 | 用途 |
|---|---|
| `learning_assistant_cli.py` | 可运行的 Python 程序 |
| `ch02_output/tasks.json` | 保存的学习任务 |
| `ch02_output/learning_report.md` | 导出的作品集证据 |

## Step 0：创建干净的练习文件夹

在终端运行：

```bash
mkdir ch02-learning-assistant-workshop
cd ch02-learning-assistant-workshop
python3 --version
```

预期输出类似下面这样，版本号不同没关系。

```text
Python 3.12.3
```

本工作坊使用了 `dataclass`、`list[str]`、`str | None` 等现代 Python 标准库写法。请使用 Python 3.10 或更新版本。

## Step 1：先看懂整段程序要怎么跑

![CLI 命令执行流程](/img/course/ch02-hands-on-cli-command-flow.webp)

程序会沿着一条简单路线执行：

| 步骤 | 发生了什么 | 对应 Python 知识 |
|---|---|---|
| 用户输入命令 | `add`、`list`、`done`、`stats` 或 `export` | 命令行参数 |
| `argparse` 解析命令 | 命令变成结构化数据 | 函数和模块 |
| 程序读取 JSON | 从磁盘读取已有任务 | 文件读写和异常 |
| 命令函数执行 | 修改或汇总数据 | 列表、字典、循环 |
| 程序保存输出 | 写回 JSON 或 Markdown | 持久化 |

读代码时先记住这张图。你正在做的是一个完整小程序，而不是单独练某个语法点。

## Step 2：创建完整脚本

创建 `learning_assistant_cli.py` 文件，然后粘贴下面代码。

```python
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

OUTPUT_DIR = Path("ch02_output")
DATA_FILE = OUTPUT_DIR / "tasks.json"
REPORT_FILE = OUTPUT_DIR / "learning_report.md"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class Task:
    id: int
    title: str
    stage: str
    tags: list[str]
    done: bool = False
    created_at: str = field(default_factory=utc_now)
    completed_at: str | None = None


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)


def load_tasks() -> list[Task]:
    if not DATA_FILE.exists():
        return []
    try:
        raw_tasks = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Cannot read {DATA_FILE}: invalid JSON at line {exc.lineno}. Fix or remove the file, then rerun.") from exc
    return [Task(**item) for item in raw_tasks]


def save_tasks(tasks: list[Task]) -> None:
    ensure_output_dir()
    data = [asdict(task) for task in tasks]
    DATA_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def next_id(tasks: list[Task]) -> int:
    if not tasks:
        return 1
    return max(task.id for task in tasks) + 1


def seed_tasks(_: argparse.Namespace) -> None:
    tasks = [
        Task(id=1, title="Read Python functions", stage="2.1", tags=["functions"]),
        Task(id=2, title="Practice JSON file saving", stage="2.2", tags=["json", "file-io"]),
        Task(id=3, title="Build the first CLI command", stage="2.3", tags=["cli"]),
    ]
    save_tasks(tasks)
    print(f"Wrote {len(tasks)} sample tasks to {DATA_FILE}")


def add_task(args: argparse.Namespace) -> None:
    title = args.title.strip()
    if not title:
        raise SystemExit("Task title cannot be empty.")
    tasks = load_tasks()
    task = Task(id=next_id(tasks), title=title, stage=args.stage, tags=args.tag)
    tasks.append(task)
    save_tasks(tasks)
    print(f"Added task #{task.id}: {task.title}")


def list_tasks(_: argparse.Namespace) -> None:
    tasks = load_tasks()
    if not tasks:
        print("No tasks yet. Run: python learning_assistant_cli.py add \"Read functions\"")
        return
    print("ID  Status  Stage  Title")
    print("--  ------  -----  -----")
    for task in tasks:
        status = "done" if task.done else "todo"
        print(f"{task.id:<2}  {status:<6}  {task.stage:<5}  {task.title}")


def complete_task(args: argparse.Namespace) -> None:
    tasks = load_tasks()
    for task in tasks:
        if task.id == args.id:
            task.done = True
            task.completed_at = utc_now()
            save_tasks(tasks)
            print(f"Completed task #{task.id}: {task.title}")
            return
    raise SystemExit(f"Task #{args.id} was not found.")


def show_stats(_: argparse.Namespace) -> None:
    tasks = load_tasks()
    total = len(tasks)
    done = sum(task.done for task in tasks)
    todo = total - done
    by_stage: dict[str, int] = {}
    for task in tasks:
        by_stage[task.stage] = by_stage.get(task.stage, 0) + 1
    rate = (done / total * 100) if total else 0
    print(f"Total tasks: {total}")
    print(f"Done: {done}")
    print(f"Todo: {todo}")
    print(f"Completion rate: {rate:.1f}%")
    print("Tasks by stage:")
    for stage, count in sorted(by_stage.items()):
        print(f"- {stage}: {count}")


def export_report(_: argparse.Namespace) -> None:
    tasks = load_tasks()
    done = sum(task.done for task in tasks)
    total = len(tasks)
    lines = [
        "# Python Learning Assistant Report",
        "",
        f"Generated at: {utc_now()}",
        f"Total tasks: {total}",
        f"Completed tasks: {done}",
        "",
        "## Tasks",
        "",
    ]
    for task in tasks:
        checkbox = "x" if task.done else " "
        tags = ", ".join(task.tags) if task.tags else "-"
        lines.append(f"- [{checkbox}] #{task.id} {task.title} (stage {task.stage}; tags: {tags})")
    ensure_output_dir()
    REPORT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Exported report to {REPORT_FILE}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local learning-task assistant for Chapter 2 Python practice.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    seed_parser = subparsers.add_parser("seed", help="Create sample tasks.")
    seed_parser.set_defaults(func=seed_tasks)

    add_parser = subparsers.add_parser("add", help="Add one learning task.")
    add_parser.add_argument("title", help="Task title, wrapped in quotes if it contains spaces.")
    add_parser.add_argument("--stage", default="2.1", help="Course stage or section, such as 2.1 or 2.3.")
    add_parser.add_argument("--tag", action="append", default=[], help="Repeatable tag, such as --tag functions --tag json.")
    add_parser.set_defaults(func=add_task)

    list_parser = subparsers.add_parser("list", help="List tasks.")
    list_parser.set_defaults(func=list_tasks)

    done_parser = subparsers.add_parser("done", help="Mark one task as complete.")
    done_parser.add_argument("id", type=int, help="Task id to complete.")
    done_parser.set_defaults(func=complete_task)

    stats_parser = subparsers.add_parser("stats", help="Show task statistics.")
    stats_parser.set_defaults(func=show_stats)

    export_parser = subparsers.add_parser("export", help="Export a Markdown report.")
    export_parser.set_defaults(func=export_report)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
```

## Step 3：运行第一个命令

```bash
python3 learning_assistant_cli.py seed
```

预期输出：

```text
Wrote 3 sample tasks to ch02_output/tasks.json
```

再查看任务：

```bash
python3 learning_assistant_cli.py list
```

预期输出：

```text
ID  Status  Stage  Title
--  ------  -----  -----
1   todo    2.1    Read Python functions
2   todo    2.2    Practice JSON file saving
3   todo    2.3    Build the first CLI command
```

## Step 4：添加并完成任务

![JSON 持久化流程](/img/course/ch02-hands-on-json-persistence-flow.webp)

添加一条新任务：

```bash
python3 learning_assistant_cli.py add "Practice command-line arguments" --stage 2.3 --tag argparse
```

预期输出：

```text
Added task #4: Practice command-line arguments
```

把任务 `2` 标记为完成：

```bash
python3 learning_assistant_cli.py done 2
```

预期输出：

```text
Completed task #2: Practice JSON file saving
```

这时打开 `ch02_output/tasks.json`，你应该能看到正常的 JSON 数据。时间戳会不同，但任务 `2` 的 `done` 字段应该是 `true`。

## Step 5：查看统计并导出报告

```bash
python3 learning_assistant_cli.py stats
```

预期输出：

```text
Total tasks: 4
Done: 1
Todo: 3
Completion rate: 25.0%
Tasks by stage:
- 2.1: 1
- 2.2: 1
- 2.3: 2
```

导出 Markdown 报告：

```bash
python3 learning_assistant_cli.py export
```

预期输出：

```text
Exported report to ch02_output/learning_report.md
```

现在你已经有一个可运行项目，以及一份可以当作作品集证据的小报告。

## Step 6：理解关键代码

| 代码片段 | 它在训练什么 | 后续为什么重要 |
|---|---|---|
| `argparse` | 把终端命令转换成结构化值 | CLI、脚本和自动化工具都需要清晰输入 |
| `@dataclass` | 用字段描述一个任务 | 后续 API 模型、数据库行、配置对象都是类似思路 |
| `load_tasks()` | 读取 JSON，并处理坏 JSON | 真实程序必须能面对缺失或损坏的文件 |
| `save_tasks()` | 把 Python 对象转成 JSON | 这是持久化的最小版本 |
| 命令函数 | 一个命令对应一个函数 | 大项目依赖清楚的函数边界 |
| `export_report()` | 把内部数据变成用户能看的输出 | AI 和数据工具经常要生成报告、日志和证据 |

## 常见错误与修复

![错误与调试地图](/img/course/ch02-hands-on-error-debug-map.webp)

| 问题 | 可能原因 | 修复 |
|---|---|---|
| `python3: command not found` | 你的系统使用 `python` 而不是 `python3` | 先试 `python --version`，再运行 `python learning_assistant_cli.py seed` |
| `Task #99 was not found.` | 你想完成的任务 id 不存在 | 先运行 `python3 learning_assistant_cli.py list` |
| `invalid JSON` 错误 | 手动编辑 `tasks.json` 时把格式弄坏了 | 修复 JSON 文件，或删除后重新运行 `seed` |
| 报告是空的 | 还没有创建任务 | 先运行 `seed` 或 `add`，再运行 `export` |
| 能看懂代码但不会改 | 整个脚本一次看太大 | 一次只改一个命令，然后只重跑对应命令 |

## 小练习

1. 新增一个 `delete` 命令，根据 id 删除任务。
2. 新增一个 `search` 命令，根据关键词查找任务。
3. 给 `list` 增加 `--tag` 过滤。
4. 修改 `export_report()`，让未完成任务排在前面。
5. 故意弄坏 `tasks.json`，运行 `list`，记录错误信息和修复过程。

## 作品集证据清单

![Python 项目证据包](/img/course/ch02-hands-on-evidence-pack.webp)

请保留这些材料作为证据：

- `learning_assistant_cli.py`
- `ch02_output/tasks.json`
- `ch02_output/learning_report.md`
- 一张截图或复制的终端输出，展示 `seed`、`list`、`done`、`stats`、`export`
- 一份简短 `README.md`，说明如何运行工具，以及你处理了哪些错误

这是第 2 章的核心习惯：**不要停在语法，必须把语法变成能运行、能保存数据、能处理错误、能讲清楚的小工具。**
