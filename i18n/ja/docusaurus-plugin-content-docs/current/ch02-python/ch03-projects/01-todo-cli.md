---
title: "1.1 プロジェクト：コマンドラインタスクマネージャー"
sidebar_position: 1
description: "Python の基礎知識を総合的に活用して、コマンドラインのタスク管理ツールを作る"
---

# プロジェクト：コマンドラインタスクマネージャー

![コマンドラインタスクマネージャーのアーキテクチャ図](/img/course/ch02-todo-cli-architecture.png)

## プロジェクトの位置づけ

これは、Python 基礎段階で最初に取り組む、ひと通り完結した小さなプロジェクトです。データ構造、関数、ファイルの読み書き、例外処理を組み合わせて、タスクを保存したり、一覧表示したり、状態を変更したりできる、実用的なコマンドラインツールを作ります。

## プロジェクトの目標

- Python の基礎知識（データ構造、関数、ファイル操作、例外処理）を総合的に使う
- 完全なプロジェクト開発の流れを体験する：要件分析 → 設計 → コーディング → テスト
- **本当に使える**コマンドラインツールを作る

---

## プロジェクト概要

これから作るのは、**コマンドラインタスクマネージャー**（簡易版の Todoist のようなもの）です。次の機能をサポートします。

- タスクを追加する
- すべてのタスクを表示する
- タスクを完了済みにする
- タスクを削除する
- データを永続化する（プログラムを終了してもデータが消えない）

完成イメージ：

```
===== タスクマネージャー =====
1. すべてのタスクを表示
2. タスクを追加
3. タスクを完了
4. タスクを削除
5. 終了

操作を選んでください (1-5): 1

📋 タスクリスト:
  1. [ ] Python の基礎を学ぶ        (作成日: 2026-02-09)
  2. [✓] 第 1 章のツール基礎を完了     (作成日: 2026-02-08)
  3. [ ] 機械学習プロジェクトを始める        (作成日: 2026-02-09)

合計 3 件のタスク、完了済み 1 件
```

---

## 第1ステップ：プロジェクト計画

### データ設計

1件のタスクには、どんな情報が必要でしょうか？

```python
task = {
    "id": 1,
    "title": "Python の基礎を学ぶ",
    "done": False,
    "created_at": "2026-02-09 14:30:00"
}
```

すべてのタスクは1つのリストに入れ、JSON ファイルに保存します。

### 機能モジュール

| モジュール | 機能 |
|------|------|
| データ管理 | ファイルへのタスクの読み込み/保存 |
| タスク操作 | 追加・削除・更新・参照 |
| ユーザーインターフェース | メニュー表示、入力処理 |

---

## 第2ステップ：基本版

まずは、ファイル保存なしのいちばんシンプルな版を実装してみましょう。

```python
# todo.py —— コマンドラインタスクマネージャー

from datetime import datetime


def show_menu():
    """メニューを表示する"""
    print("\n===== タスクマネージャー =====")
    print("1. すべてのタスクを表示")
    print("2. タスクを追加")
    print("3. タスクを完了")
    print("4. タスクを削除")
    print("5. 終了")
    print()


def show_tasks(tasks: list[dict]) -> None:
    """すべてのタスクを表示する"""
    if not tasks:
        print("📭 まだタスクがありません。さっそく追加してみましょう！")
        return

    print("\n📋 タスクリスト:")
    for i, task in enumerate(tasks, 1):
        status = "✓" if task["done"] else " "
        print(f'  {i}. [{status}] {task["title"]}  '
              f'(作成日: {task["created_at"][:10]})')

    done_count = sum(1 for t in tasks if t["done"])
    print(f"\n合計 {len(tasks)} 件のタスク、完了済み {done_count} 件")


def add_task(tasks: list[dict]) -> None:
    """新しいタスクを追加する"""
    title = input("タスクのタイトルを入力してください: ").strip()
    if not title:
        print("❌ タスクのタイトルは空にできません！")
        return

    task = {
        "id": len(tasks) + 1,
        "title": title,
        "done": False,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    tasks.append(task)
    print(f"✅ タスク「{title}」を追加しました！")


def complete_task(tasks: list[dict]) -> None:
    """タスクを完了済みにする"""
    show_tasks(tasks)
    if not tasks:
        return

    try:
        num = int(input("完了にしたいタスク番号を入力してください: "))
        if 1 <= num <= len(tasks):
            task = tasks[num - 1]
            if task["done"]:
                print(f"⚠️ タスク「{task['title']}」はすでに完了しています")
            else:
                task["done"] = True
                print(f"✅ タスク「{task['title']}」を完了済みにしました！")
        else:
            print("❌ 無効なタスク番号です！")
    except ValueError:
        print("❌ 数字を入力してください！")


def delete_task(tasks: list[dict]) -> None:
    """タスクを削除する"""
    show_tasks(tasks)
    if not tasks:
        return

    try:
        num = int(input("削除したいタスク番号を入力してください: "))
        if 1 <= num <= len(tasks):
            removed = tasks.pop(num - 1)
            print(f"🗑️ タスク「{removed['title']}」を削除しました！")
        else:
            print("❌ 無効なタスク番号です！")
    except ValueError:
        print("❌ 数字を入力してください！")


def main():
    """メイン関数"""
    tasks = []

    print("タスクマネージャーへようこそ！")

    while True:
        show_menu()
        choice = input("操作を選んでください (1-5): ").strip()

        if choice == "1":
            show_tasks(tasks)
        elif choice == "2":
            add_task(tasks)
        elif choice == "3":
            complete_task(tasks)
        elif choice == "4":
            delete_task(tasks)
        elif choice == "5":
            print("👋 さようなら！")
            break
        else:
            print("❌ 無効な選択です。1-5 を入力してください")


if __name__ == "__main__":
    main()
```

**やってみよう：** 上のコードを `todo.py` として保存し、`python todo.py` を実行してみましょう。

---

## 第3ステップ：ファイルの永続化を追加する

このままだと、プログラムを終了するとデータが消えてしまいます。ファイル保存機能を追加しましょう。

```python
import json
from pathlib import Path

DATA_FILE = Path("tasks.json")


def load_tasks() -> list[dict]:
    """ファイルからタスクを読み込む"""
    if DATA_FILE.exists():
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                tasks = json.load(f)
            print(f"📂 {len(tasks)} 件のタスクを読み込みました")
            return tasks
        except (json.JSONDecodeError, IOError) as e:
            print(f"⚠️ データの読み込みに失敗しました: {e}。空のリストを使います")
    return []


def save_tasks(tasks: list[dict]) -> None:
    """タスクをファイルに保存する"""
    try:
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(tasks, f, ensure_ascii=False, indent=2)
    except IOError as e:
        print(f"⚠️ データの保存に失敗しました: {e}")
```

次に `main()` 関数を次のように変更します。

```python
def main():
    tasks = load_tasks()  # 起動時に読み込む

    print("タスクマネージャーへようこそ！")

    while True:
        show_menu()
        choice = input("操作を選んでください (1-5): ").strip()

        if choice == "1":
            show_tasks(tasks)
        elif choice == "2":
            add_task(tasks)
            save_tasks(tasks)  # 追加後に保存
        elif choice == "3":
            complete_task(tasks)
            save_tasks(tasks)  # 変更後に保存
        elif choice == "4":
            delete_task(tasks)
            save_tasks(tasks)  # 削除後に保存
        elif choice == "5":
            save_tasks(tasks)  # 終了前に保存
            print("👋 さようなら！")
            break
        else:
            print("❌ 無効な選択です。1-5 を入力してください")
```

---

## 第4ステップ：発展チャレンジ

基本版が完成したら、次の機能を追加してレベルアップしてみましょう。

### チャレンジ 1：タスクの優先度

タスクに優先度（高/中/低）を追加し、優先度順に表示できるようにする。

### チャレンジ 2：検索機能

キーワードでタスクのタイトルを検索できるようにする。

### チャレンジ 3：集計機能

次のような統計情報を表示する：総タスク数、完了率、本日の追加数 など。

### チャレンジ 4：クラスでリファクタリング

プロジェクト全体をオブジェクト指向で書き直してみましょう。

```python
class Task:
    """1件のタスク"""
    def __init__(self, title: str, priority: str = "中"):
        self.title = title
        self.priority = priority
        self.done = False
        self.created_at = datetime.now()

class TaskManager:
    """タスクマネージャー"""
    def __init__(self, filename: str = "tasks.json"):
        self.filename = filename
        self.tasks: list[Task] = []
        self.load()

    def add(self, title: str, priority: str = "中") -> None: ...
    def complete(self, index: int) -> None: ...
    def delete(self, index: int) -> None: ...
    def search(self, keyword: str) -> list[Task]: ...
    def save(self) -> None: ...
    def load(self) -> None: ...
```

---

## プロジェクト自己チェックリスト

プロジェクトが完成したら、次の項目を確認してみましょう。

- [ ] プログラムが正常に動作し、不正な入力でクラッシュしない
- [ ] データがファイルに保存され、再起動しても残っている
- [ ] コードが関数ごとに分かれていて、ひとまとまりになっていない
- [ ] 適切なエラー処理がある（try/except）
- [ ] 関数に docstring がある
- [ ] 変数名が分かりやすい（PEP 8 に沿っている）
- [ ] Git でプロジェクトコードを管理している

:::tip プロジェクトの経験
このプロジェクトはシンプルですが、ソフトウェア開発の核心となる要素である**ユーザーとのやり取り、データ処理、ファイル保存、エラー処理**をすべて含んでいます。これからのすべてのプロジェクト（Web アプリでも AI システムでも）は、これらの要素の拡張と組み合わせです。このプロジェクトをしっかり作り上げることが、実践的なプログラミングへの第一歩です。
:::

## バージョンの進め方のおすすめ

| バージョン | 目標 | 重点的に仕上げること |
|---|---|---|
| 基本版 | 最小限の一連の流れを動かす | 入力できる、処理できる、出力できる、そして一組のサンプルを残す |
| 標準版 | 発表できる形にする | 設定、ログ、エラー処理、README とスクリーンショットを追加する |
| チャレンジ版 | ポートフォリオ品質に近づける | 評価、比較実験、失敗例の分析、次のステップの計画を追加する |

まずは基本版を完成させることをおすすめします。最初から何でも入れようとしなくて大丈夫です。バージョンを1つ上げるたびに、「何が新しくできるようになったか」「どうやって確認したか」「まだ何が課題か」を README に書き足していきましょう。
