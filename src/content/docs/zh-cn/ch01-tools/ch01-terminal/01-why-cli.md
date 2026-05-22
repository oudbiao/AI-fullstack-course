---
title: "1.1.1 为什么要学命令行"
description: "理解命令行为什么是 AI 开发里可重复、可排查的控制层。"
sidebar:
  order: 1
---

# 1.1.1 为什么要学命令行

![命令行自动化工作流图](/img/course/ch01-cli-automation-workflow.webp)

命令行就是用文字给电脑下精确指令。AI 项目里，你会用它进入项目文件夹、运行 Python、安装依赖、保存 Git 提交、连接服务器和启动服务。

先不要背命令。先把一个最小闭环跑通：

```text
进入文件夹 -> 运行命令 -> 看输出 -> 修下一步
```

## 命令行 vs 图形界面

| 需求 | 图形界面 | 命令行 |
|---|---|---|
| 做一次简单操作 | 好上手 | 也可以 |
| 重复同一件事 | 容易漏点一步 | 复制命令重跑 |
| 批量处理 | 慢 | 快 |
| 操作服务器 | 经常没有界面 | 标准方式 |
| 留下排错证据 | 不好记录 | 命令和输出都能保存 |

关键不是终端看起来专业，而是命令本身就是**可重复的证据**。

## 先跑一次

在练习文件夹里打开终端，运行：

```bash
pwd
mkdir ai-cli-practice
cd ai-cli-practice
python -c "from pathlib import Path; Path('hello_terminal.py').write_text('print(\"hello from terminal\")\\n', encoding='utf-8')"
python hello_terminal.py
ls
```

预期信号：

```text
hello from terminal
hello_terminal.py
```

如果你在 Windows 上 `python` 不能用，可以试：

```bash
py hello_terminal.py
```

到这里，你已经完成最小终端闭环：查看当前位置、创建文件夹、创建脚本、运行脚本、检查结果。

## AI 项目会在哪里用到

| AI 任务 | 示例命令 |
|---|---|
| 安装依赖 | `python -m pip install pandas` |
| 运行脚本 | `python train.py` |
| 保存代码 | `git add .` 和 `git commit -m "message"` |
| 启动 API | `uvicorn main:app --reload` |
| 连接服务器 | `ssh user@server` |
| 构建部署应用 | `docker build -t my-ai-app .` |

今天不用全会，只要先知道：后面很多 AI 工作流都会从终端开始。

## 先认识这 10 个命令

| 命令 | 作用 |
|---|---|
| `pwd` | 查看当前文件夹 |
| `ls` | 查看文件列表 |
| `cd` | 切换文件夹 |
| `mkdir` | 创建文件夹 |
| `cp` | 复制 |
| `mv` | 移动或重命名 |
| `rm` | 删除 |
| `python` | 运行 Python |
| `git` | 保存和查看代码历史 |
| `pip` / `conda` | 安装依赖、管理环境 |

## 如果失败，先查这里

| 现象 | 先检查 |
|---|---|
| `command not found` | 工具是否安装好；安装后是否重新打开终端 |
| `python` 版本不对 | 运行 `python --version`，确认当前环境 |
| 找不到文件 | 运行 `pwd` 和 `ls`，你可能不在目标文件夹 |
| Permission denied | 文件或文件夹是否属于其他用户 |
| 命令太长不想重打 | 按上箭头调出历史命令 |

当你能说清 `pwd`、`cd`、`ls` 和 `python hello_terminal.py` 在自己的文件夹里做了什么，就可以进入下一节。下一节会更慢地练文件操作。

## 留下的证据

学完这一页，至少保留这张证据卡：

```text
命令：你运行过的精确终端命令
工作目录：pwd/当前文件夹及重要文件列表
输出：复制的命令输出或结果截图
失败检查：错误的路径、缺少命令、权限问题，或 shell 不匹配
期望产出：可复现的终端操作，命令和结果并排展示
```
