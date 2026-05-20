---
title: "2.3.5 ハンズオンワークショップ：ローカル学習タスクアシスタントを作る"
sidebar_position: 24
description: "第 2 章の Python ハンズオン。argparse、dataclass、JSON 永続化、例外処理、レポート出力を使って、実行できるコマンドライン学習タスクアシスタントを作る。"
keywords: [Python ハンズオン, コマンドラインアプリ, argparse, JSON, dataclass, ファイル入出力, Python プロジェクト]
---

# 2.3.5 ハンズオンワークショップ：ローカル学習タスクアシスタントを作る

![Python ハンズオンワークショップのルート](/img/course/ch02-hands-on-python-workshop-route-ja.webp)

:::tip ワークショップの目標
このページは第 2 章の実践の橋渡しです。構文説明を読むだけでなく、学習タスクを作成し、JSON に保存し、完了扱いにし、統計を表示し、Markdown レポートを書き出す小さなツールを作ります。
:::

## 何を作るのか

`learning_assistant_cli.py` というコマンドライン学習タスクアシスタントを作ります。Python 標準ライブラリだけを使うため、サードパーティパッケージをインストールする必要はありません。

手順どおりに進めると、次のようなコマンドを実行できるようになります。

```bash
python3 learning_assistant_cli.py seed
python3 learning_assistant_cli.py list
python3 learning_assistant_cli.py add "Practice command-line arguments" --stage 2.3 --tag argparse
python3 learning_assistant_cli.py done 2
python3 learning_assistant_cli.py stats
python3 learning_assistant_cli.py export
```

プロジェクトは次のファイルを作ります。

| ファイル | 目的 |
|---|---|
| `learning_assistant_cli.py` | 実行できる Python プログラム |
| `ch02_output/tasks.json` | 保存された学習タスク |
| `ch02_output/learning_report.md` | ポートフォリオ証拠として使える出力レポート |

## ステップ 0：きれいな練習フォルダを作る

ターミナルで実行します。

```bash
mkdir ch02-learning-assistant-workshop
cd ch02-learning-assistant-workshop
python3 --version
```

出力は次のようになります。バージョン番号は違っていてかまいません。

```text
Python 3.12.3
```

このワークショップでは `dataclass`、`list[str]`、`str | None` などの現代的な Python 標準ライブラリ構文を使います。Python 3.10 以降を使ってください。

## ステップ 1：まずプログラム全体の流れを見る

![CLI コマンド実行フロー](/img/course/ch02-hands-on-cli-command-flow-ja.webp)

プログラムは単純な流れで動きます。

| 手順 | 何が起きるか | 対応する Python 概念 |
|---|---|---|
| ユーザーがコマンドを入力 | `add`、`list`、`done`、`stats`、`export` のどれか | コマンドライン引数 |
| `argparse` が解析 | コマンドが構造化された値になる | 関数とモジュール |
| プログラムが JSON を読む | 保存済みタスクをディスクから読む | ファイル入出力と例外 |
| コマンド関数が動く | データを変更または集計する | リスト、辞書、ループ |
| プログラムが出力を保存 | JSON または Markdown を書き戻す | 永続化 |

コードを読むときは、この図を頭に置いてください。作っているのは孤立した構文練習ではなく、小さいけれど完成したプログラムです。

## ステップ 2：完全なスクリプトを作る

`learning_assistant_cli.py` というファイルを作り、次のコードを貼り付けます。

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

## ステップ 3：最初のコマンドを実行する

```bash
python3 learning_assistant_cli.py seed
```

期待される出力：

```text
Wrote 3 sample tasks to ch02_output/tasks.json
```

次にタスクを一覧表示します。

```bash
python3 learning_assistant_cli.py list
```

期待される出力：

```text
ID  Status  Stage  Title
--  ------  -----  -----
1   todo    2.1    Read Python functions
2   todo    2.2    Practice JSON file saving
3   todo    2.3    Build the first CLI command
```

## ステップ 4：タスクを追加し、完了にする

![JSON 永続化フロー](/img/course/ch02-hands-on-json-persistence-flow-ja.webp)

新しいタスクを追加します。

```bash
python3 learning_assistant_cli.py add "Practice command-line arguments" --stage 2.3 --tag argparse
```

期待される出力：

```text
Added task #4: Practice command-line arguments
```

タスク `2` を完了にします。

```bash
python3 learning_assistant_cli.py done 2
```

期待される出力：

```text
Completed task #2: Practice JSON file saving
```

この時点で `ch02_output/tasks.json` を開くと、通常の JSON データが見えるはずです。タイムスタンプは環境ごとに違いますが、タスク `2` の `done` フィールドは `true` になっているはずです。

## ステップ 5：統計を表示し、レポートを書き出す

```bash
python3 learning_assistant_cli.py stats
```

期待される出力：

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

Markdown レポートを書き出します。

```bash
python3 learning_assistant_cli.py export
```

期待される出力：

```text
Exported report to ch02_output/learning_report.md
```

これで、実行できるプロジェクトと、ポートフォリオ証拠として使える小さなレポートができました。

## ステップ 6：重要な部分を理解する

| コード部分 | 何を練習しているか | 後でなぜ重要か |
|---|---|---|
| `argparse` | ターミナルのコマンドを構造化された値に変換する | CLI、スクリプト、自動化ツールには明確な入力が必要 |
| `@dataclass` | タスクをフィールドで説明する | 後の API モデル、データベース行、設定オブジェクトと同じ考え方 |
| `load_tasks()` | JSON を読み、壊れた JSON に対応する | 実際のプログラムは、存在しないファイルや壊れたファイルにも耐える必要がある |
| `save_tasks()` | Python オブジェクトを JSON に変換する | 永続化の最小版 |
| コマンド関数 | 1 つのコマンドを 1 つの関数に分ける | 大きなプロジェクトは明確な関数境界に依存する |
| `export_report()` | 内部データをユーザー向けの出力にする | AI ツールやデータツールでは、レポート、ログ、証拠がよく必要になる |

## 残す証拠

このページを終えたら、この evidence card を残します。

```text
project_goal: CLI, scraper, API, AI API call, or integrated Python workshop target
run_command: exact command used to start the project
artifact: output file, API response, JSON record, screenshot, or README note
failure_check: dependency, network, parsing, route, input validation, or API-key issue
Expected_output: reproducible mini project folder with run result and one failure case
```

## よくあるエラーと直し方

![エラーとデバッグのマップ](/img/course/ch02-hands-on-error-debug-map-ja.webp)

| 問題 | よくある原因 | 修正 |
|---|---|---|
| `python3: command not found` | 環境では `python` を使う設定になっている | `python --version` を試し、`python learning_assistant_cli.py seed` を実行する |
| `Task #99 was not found.` | 存在しないタスク id を完了にしようとしている | 先に `python3 learning_assistant_cli.py list` を実行する |
| `invalid JSON` エラー | `tasks.json` を手動編集して形式を壊した | JSON ファイルを修正するか、削除してから `seed` を実行する |
| レポートが空 | まだタスクを作っていない | `seed` または `add` を実行してから `export` する |
| コードは読めるが変更できない | スクリプト全体を一度に見て大きく感じている | 1 回に 1 つのコマンドだけ変更し、そのコマンドだけ再実行する |

## ミニ演習

1. `delete` コマンドを追加し、id でタスクを削除する。
2. `search` コマンドを追加し、キーワードを含むタスクを探す。
3. `list` に `--tag` フィルタを追加する。
4. `export_report()` を変更し、未完了タスクを先に出す。
5. わざと `tasks.json` を壊し、`list` を実行して、エラーメッセージと修正方法を記録する。

<details>
<summary>参考解答と解説</summary>

1. `delete` は id を受け取り、`tasks.json` から該当アイテムを削除し、明確な確認メッセージを出します。もう一度 `list` を実行して、行が本当に消えたことを確認します。
2. `search` は `title` をキーワードで絞り込み、必要なら `tags` も対象にし、大文字小文字を区別せずに一致したものだけを表示します。
3. `--tag` は `list` の `argparse` フィルタにするのが最適です。保存データを変更せずにコマンドを再利用できます。
4. レポートで今やる作業を先に目立たせたいなら、`export_report()` で未完了タスクを完了済みより前に並べます。形式を安定させておくと diff が読みやすくなります。
5. わざと `tasks.json` を壊して `list` を実行し、スクリプトがクラッシュせずに分かりやすい JSON エラーを出すことを確認します。その後、ファイルを修復するか削除してから `seed` を再実行します。
</details>

## ポートフォリオ用の証拠チェックリスト

![Python プロジェクト証拠パック](/img/course/ch02-hands-on-evidence-pack-ja.webp)

証拠として次のファイルを残しましょう。

- `learning_assistant_cli.py`
- `ch02_output/tasks.json`
- `ch02_output/learning_report.md`
- `seed`、`list`、`done`、`stats`、`export` を実行したスクリーンショット、またはコピーしたターミナル出力
- ツールの実行方法と、対応したエラーを書いた短い `README.md`

第 2 章の核心はこれです。**構文で止まらず、実行でき、データを保存でき、エラーを扱え、説明できる小さなツールに変えましょう。**
