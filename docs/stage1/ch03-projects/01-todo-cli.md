---
title: "1.1 项目：命令行任务管理器"
sidebar_position: 1
description: "综合运用 Python 基础知识，构建一个命令行任务管理工具"
---

# 项目：命令行任务管理器

## 项目目标

- 综合运用 Python 基础知识（数据结构、函数、文件操作、异常处理）
- 体验完整的项目开发流程：需求分析 → 设计 → 编码 → 测试
- 构建一个**真正可用的**命令行工具

---

## 项目简介

我们要构建一个**命令行任务管理器**（类似简易版的 Todoist），支持：

- 添加任务
- 查看所有任务
- 标记任务完成
- 删除任务
- 数据持久化（关闭程序后数据不丢失）

最终效果：

```
===== 任务管理器 =====
1. 查看所有任务
2. 添加任务
3. 完成任务
4. 删除任务
5. 退出

请选择操作 (1-5): 1

📋 任务列表:
  1. [ ] 学习 Python 基础        (创建于: 2026-02-09)
  2. [✓] 完成 Stage 0           (创建于: 2026-02-08)
  3. [ ] 开始机器学习项目        (创建于: 2026-02-09)

共 3 个任务，已完成 1 个
```

---

## 第一步：项目规划

### 数据设计

每个任务需要哪些信息？

```python
task = {
    "id": 1,
    "title": "学习 Python 基础",
    "done": False,
    "created_at": "2026-02-09 14:30:00"
}
```

所有任务存在一个列表中，并保存到 JSON 文件。

### 功能模块

| 模块 | 功能 |
|------|------|
| 数据管理 | 加载/保存任务到文件 |
| 任务操作 | 增删改查 |
| 用户界面 | 菜单显示、输入处理 |

---

## 第二步：基础版本

先实现一个最简单的版本，不带文件保存：

```python
# todo.py —— 命令行任务管理器

from datetime import datetime


def show_menu():
    """显示菜单"""
    print("\n===== 任务管理器 =====")
    print("1. 查看所有任务")
    print("2. 添加任务")
    print("3. 完成任务")
    print("4. 删除任务")
    print("5. 退出")
    print()


def show_tasks(tasks: list[dict]) -> None:
    """显示所有任务"""
    if not tasks:
        print("📭 暂无任务，快去添加一个吧！")
        return

    print("\n📋 任务列表:")
    for i, task in enumerate(tasks, 1):
        status = "✓" if task["done"] else " "
        print(f'  {i}. [{status}] {task["title"]}  '
              f'(创建于: {task["created_at"][:10]})')

    done_count = sum(1 for t in tasks if t["done"])
    print(f"\n共 {len(tasks)} 个任务，已完成 {done_count} 个")


def add_task(tasks: list[dict]) -> None:
    """添加新任务"""
    title = input("请输入任务标题: ").strip()
    if not title:
        print("❌ 任务标题不能为空！")
        return

    task = {
        "id": len(tasks) + 1,
        "title": title,
        "done": False,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    tasks.append(task)
    print(f"✅ 任务「{title}」已添加！")


def complete_task(tasks: list[dict]) -> None:
    """标记任务为完成"""
    show_tasks(tasks)
    if not tasks:
        return

    try:
        num = int(input("请输入要完成的任务编号: "))
        if 1 <= num <= len(tasks):
            task = tasks[num - 1]
            if task["done"]:
                print(f"⚠️ 任务「{task['title']}」已经完成过了")
            else:
                task["done"] = True
                print(f"✅ 任务「{task['title']}」已标记为完成！")
        else:
            print("❌ 无效的任务编号！")
    except ValueError:
        print("❌ 请输入数字！")


def delete_task(tasks: list[dict]) -> None:
    """删除任务"""
    show_tasks(tasks)
    if not tasks:
        return

    try:
        num = int(input("请输入要删除的任务编号: "))
        if 1 <= num <= len(tasks):
            removed = tasks.pop(num - 1)
            print(f"🗑️ 任务「{removed['title']}」已删除！")
        else:
            print("❌ 无效的任务编号！")
    except ValueError:
        print("❌ 请输入数字！")


def main():
    """主函数"""
    tasks = []

    print("欢迎使用任务管理器！")

    while True:
        show_menu()
        choice = input("请选择操作 (1-5): ").strip()

        if choice == "1":
            show_tasks(tasks)
        elif choice == "2":
            add_task(tasks)
        elif choice == "3":
            complete_task(tasks)
        elif choice == "4":
            delete_task(tasks)
        elif choice == "5":
            print("👋 再见！")
            break
        else:
            print("❌ 无效的选择，请输入 1-5")


if __name__ == "__main__":
    main()
```

**试一试：** 把上面的代码保存为 `todo.py`，运行 `python todo.py`。

---

## 第三步：添加文件持久化

现在程序一关数据就没了。让我们加上文件保存功能：

```python
import json
from pathlib import Path

DATA_FILE = Path("tasks.json")


def load_tasks() -> list[dict]:
    """从文件加载任务"""
    if DATA_FILE.exists():
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                tasks = json.load(f)
            print(f"📂 已加载 {len(tasks)} 个任务")
            return tasks
        except (json.JSONDecodeError, IOError) as e:
            print(f"⚠️ 加载数据失败: {e}，将使用空列表")
    return []


def save_tasks(tasks: list[dict]) -> None:
    """保存任务到文件"""
    try:
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(tasks, f, ensure_ascii=False, indent=2)
    except IOError as e:
        print(f"⚠️ 保存数据失败: {e}")
```

然后修改 `main()` 函数：

```python
def main():
    tasks = load_tasks()  # 启动时加载

    print("欢迎使用任务管理器！")

    while True:
        show_menu()
        choice = input("请选择操作 (1-5): ").strip()

        if choice == "1":
            show_tasks(tasks)
        elif choice == "2":
            add_task(tasks)
            save_tasks(tasks)  # 添加后保存
        elif choice == "3":
            complete_task(tasks)
            save_tasks(tasks)  # 修改后保存
        elif choice == "4":
            delete_task(tasks)
            save_tasks(tasks)  # 删除后保存
        elif choice == "5":
            save_tasks(tasks)  # 退出前保存
            print("👋 再见！")
            break
        else:
            print("❌ 无效的选择，请输入 1-5")
```

---

## 第四步：扩展挑战

基础版完成后，试着添加以下功能来提升自己：

### 挑战 1：任务优先级

给任务添加优先级（高/中/低），并支持按优先级排序显示。

### 挑战 2：搜索功能

支持按关键词搜索任务标题。

### 挑战 3：统计功能

显示统计信息：总任务数、完成率、今日新增等。

### 挑战 4：用类重构

把整个项目用面向对象的方式重构：

```python
class Task:
    """单个任务"""
    def __init__(self, title: str, priority: str = "中"):
        self.title = title
        self.priority = priority
        self.done = False
        self.created_at = datetime.now()

class TaskManager:
    """任务管理器"""
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

## 项目自查清单

完成项目后，对照检查：

- [ ] 程序能正常运行，不会因为非法输入而崩溃
- [ ] 数据保存到文件，重启后数据还在
- [ ] 代码有函数分层，不是一大坨
- [ ] 有适当的错误处理（try/except）
- [ ] 函数有文档字符串
- [ ] 变量命名清晰（符合 PEP 8）
- [ ] 用 Git 管理了项目代码

:::tip 项目经验
这个项目虽然简单，但涵盖了软件开发的核心要素：**用户交互、数据处理、文件存储、错误处理**。后面的所有项目（无论是 Web 应用还是 AI 系统）都是这些要素的扩展和组合。把这个项目做好，你就迈出了实战编程的第一步。
:::
