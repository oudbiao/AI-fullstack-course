#!/usr/bin/env python3
"""Generate AI course illustrations with an OpenAI-compatible image API.

Security notes:
- Do not hard-code API keys in this file.
- Set OPENAI_API_KEY in your local shell or a local .env.local file that is not committed.
- Optional: set OPENAI_BASE_URL if you need a proxy endpoint.

Example:
    export OPENAI_API_KEY="your_new_key"
    export OPENAI_BASE_URL="https://cliproxy.airoads.org/v1"
    python3 scripts/generate_course_images.py --dry-run
    python3 scripts/generate_course_images.py
"""

from __future__ import annotations

import argparse
import base64
import http.client
import json
import os
import stat
import textwrap
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_local_env_file(path: Path) -> None:
    """Load simple KEY=VALUE pairs from an ignored local env file."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


load_local_env_file(PROJECT_ROOT / ".env.local")
load_local_env_file(PROJECT_ROOT / ".env")

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "static" / "img" / "course"
DEFAULT_MODEL = os.environ.get("OPENAI_IMAGE_MODEL", "gpt-image-2")
DEFAULT_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://cliproxy.airoads.org/v1")
FALLBACK_PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAFgwJ/lLwVRwAAAABJRU5ErkJggg=="

IMAGE_JOBS: list[dict[str, Any]] = [
    {
        "filename": "ai-fullstack-hero.png",
        "size": "1536x1024",
        "quality": "high",
        "title": "AI 全栈学习教程主视觉",
        "suggested_page": "docs/index.md",
        "alt": "AI 全栈学习教程主视觉：学习者沿着编程、数据、模型、RAG 和 Agent 路线成长。",
        "prompt": """
高质感中文在线课程网站主视觉插图，主题是“AI 全栈学习教程”。
画面表现一个学习者从开发工具、Python、数据分析、机器学习、大模型应用、RAG、AI Agent 到毕业项目的成长路径。
整体感觉温暖、新手友好、现代科技感，蓝紫渐变背景，轻微 3D 插画质感，清晰层次，适合放在课程首页顶部。
画面右侧预留标题文字空间，但图片里不要生成具体文字，不要出现真实品牌 logo，不要出现乱码小字。
""".strip(),
    },
    {
        "filename": "ai-learning-assistant-roadmap.png",
        "size": "1536x1024",
        "quality": "high",
        "title": "AI 学习助手版本路线图",
        "suggested_page": "docs/intro/ai-learning-assistant-version-roadmap.md",
        "alt": "AI 学习助手从项目骨架逐步升级到 RAG 和 Agent 毕业作品的路线图。",
        "prompt": """
一张适合 AI 编程课程的项目路线图插图，主题是“AI 学习助手从 v0.1 成长到 v1.0”。
画面中有一个友好的 AI 学习助手角色，沿着阶段路线逐步获得能力：空项目、JSON 记事本、数据分析、Prompt、RAG 知识库、Agent 工具调用、毕业作品。
风格轻松、有趣、现代扁平插画结合轻微 3D，色彩统一为蓝紫和暖黄色，适合中文课程网站。
不要出现真实品牌 logo，不要生成难以阅读的小字，不要出现英文乱码。
""".strip(),
    },
    {
        "filename": "prompt-rag-agent-progression.png",
        "size": "1536x1024",
        "quality": "high",
        "title": "Prompt 到 RAG 到 Agent 能力进阶图",
        "suggested_page": "docs/index.md",
        "alt": "Prompt、RAG、Agent 三阶段能力进阶：表达、查资料和分步行动。",
        "prompt": """
一张解释 Prompt、RAG、Agent 能力进阶的科技教育插图。
左侧代表 Prompt：用户输入问题，AI 生成结构化回答；中间代表 RAG：AI 从知识库文档中检索证据并引用；右侧代表 Agent：AI 调用工具、执行步骤、保存日志并完成任务。
风格清晰、流程图感、现代科技蓝紫色，适合中文 AI 全栈课程网站。
可以使用抽象图标和箭头表达流程，但不要生成具体小字，不要出现真实品牌 logo，不要出现乱码文字。
""".strip(),
    },
    {
        "filename": "boss-challenge-map.png",
        "size": "1536x1024",
        "quality": "high",
        "title": "课程 Boss 战挑战地图",
        "suggested_page": "docs/intro/boss-challenge-map.md",
        "alt": "课程 Boss 战挑战地图：工作台守门人、JSON 地牢管理员、脏数据侦探、引用幻觉龙和无限循环魔王。",
        "prompt": """
一张游戏化学习地图插图，主题是“AI 全栈课程 Boss 战挑战地图”。
画面像轻量冒险地图，但保持教育课程的专业感。包含几个象征性关卡：开发工具工作台、JSON 地牢、脏数据侦探办公室、RAG 引用幻觉龙、Agent 无限循环魔王。
风格新颖、有趣、新手友好，适合技术课程网页，色彩明亮但不幼稚。
不要出现真实品牌 logo，不要生成具体文字，不要出现乱码小字。
""".strip(),
    },
    {
        "filename": "debug-detective-missions.png",
        "size": "1536x1024",
        "quality": "high",
        "title": "Debug 侦探任务集",
        "suggested_page": "docs/intro/debug-detective-missions.md",
        "alt": "Debug 侦探任务集：学习者像侦探一样收集线索、定位错误并写下修复记录。",
        "prompt": """
一张“Debug 侦探任务集”主题插图。
画面表现一个学习者像侦探一样调查代码错误：终端、日志、JSON 文件、数据表、检索结果、Agent trace 线索分布在桌面上，旁边有放大镜和线索板。
风格现代、轻松、略带悬疑但不阴暗，适合新手编程课程，用来把报错变成破案练习。
不要出现真实品牌 logo，不要生成具体文字，不要出现乱码小字。
""".strip(),
    },
    {
        "filename": "ch01-tools-foundation.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "开发者工具基础主视觉",
        "suggested_page": "docs/ch01-tools/index.md",
        "alt": "开发者工具基础主视觉：终端、Git、编辑器和 Python 环境组成稳定工作台。",
        "prompt": """
一张适合中文技术课程的阶段主视觉，主题是“开发者工具基础”。
画面表现一个整洁的开发工作台：终端窗口、Git 分支图、代码编辑器、Python 环境、文件夹和检查清单，整体感觉新手友好、清爽、可靠。
适合放在课程阶段首页，画面留出一定呼吸空间，不要生成具体文字，不要出现真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch01-task-list-workflow.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "开发者工具阶段任务链",
        "suggested_page": "docs/ch01-tools/task-list.md",
        "alt": "开发者工具阶段任务链：终端、项目目录、Python 环境、编辑器、Git 和远程仓库串成完整工作流。",
        "prompt": """
一张适合开发者工具入门课程的教学插图，主题是“从空项目到可复现仓库的任务链”。
画面表现六个清晰环节：打开终端、创建项目目录、配置 Python 环境、用编辑器写代码、用 Git 保存版本、推送到远程仓库。
风格清爽、工程化、新手友好，像一张课程闯关路线图；用图标和箭头表达，不要生成具体文字，不要出现真实品牌 logo，不要出现乱码小字。
""".strip(),
    },
    {
        "filename": "ch01-cli-automation-workflow.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "命令行自动化工作流图",
        "suggested_page": "docs/ch01-tools/ch01-terminal/01-why-cli.md",
        "alt": "命令行自动化工作流图：命令行把文件操作、环境管理、脚本运行、远程服务器和 Git 自动化串起来。",
        "prompt": """
一张适合命令行入门课程的教学插图，主题是“命令行为什么是开发者的自动化控制台”。
画面表现一个终端窗口连接到文件目录、Python 环境、脚本运行、远程服务器、Git 提交和批处理任务，形成一条可重复执行的自动化链路。
重点让新手看懂：命令行不是黑盒，而是把重复操作变成可保存、可复制、可自动化的流程。
风格现代开发者工作台，清晰、友好、科技感适中；不要生成具体文字，不要出现真实品牌 logo，不要出现乱码小字。
""".strip(),
    },
    {
        "filename": "ch01-package-manager-flow.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "包管理器安装依赖流程图",
        "suggested_page": "docs/ch01-tools/ch01-terminal/03-package-managers.md",
        "alt": "包管理器安装依赖流程图：输入安装命令后，包管理器查找软件、下载依赖、安装并统一更新。",
        "prompt": """
一张适合包管理器入门课程的教学插图，主题是“包管理器是开发者版应用商店”。
画面表现用户输入安装命令，包管理器从软件仓库查找工具，自动下载软件和依赖，安装到系统，并支持后续更新和卸载。
可以用货架、包裹、下载箭头、依赖积木和终端窗口表达，但不要出现真实品牌 logo，不要生成具体文字，不要出现乱码小字。
风格简洁、明亮、适合新人理解 Homebrew、apt、winget、pip、conda 的共同思想。
""".strip(),
    },
    {
        "filename": "ch01-git-daily-loop.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Git 日常最小循环图",
        "suggested_page": "docs/ch01-tools/ch02-git/02-core-operations.md",
        "alt": "Git 日常最小循环图：修改文件、查看状态、查看差异、暂存、提交和回看历史。",
        "prompt": """
一张适合 Git 入门课程的教学图，主题是“Git 每天的最小工作循环”。
画面表现开发者修改文件后，依次经过查看状态、查看差异、放入暂存区、提交版本、回看历史这几个步骤，形成一个循环。
重点突出工作区、暂存区、版本历史之间的流动关系，新手一眼能看懂 add、commit、status、diff、log 的位置。
风格清晰、工程图解、蓝绿色科技教育风；不要出现真实品牌 logo，不要生成密集小字或乱码文字。
""".strip(),
    },
    {
        "filename": "ch01-git-remote-sync.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Git 本地远程同步图",
        "suggested_page": "docs/ch01-tools/ch02-git/03-remote-repos.md",
        "alt": "Git 本地远程同步图：本地仓库通过 push、pull、clone 和 GitHub 远程仓库同步。",
        "prompt": """
一张适合 Git 远程仓库入门课程的教学图，主题是“本地仓库和远程仓库如何同步”。
画面表现一台本地电脑上的代码仓库，与云端远程仓库之间通过上传、拉取、克隆三种方向箭头同步；另一台电脑可以从远程仓库克隆项目。
重点表达远程仓库既是备份，也是协作和作品集展示位置。
风格清晰、现代、适合新手理解 push、pull、clone 和 SSH key；不要出现真实品牌 logo，不要生成具体文字，不要出现乱码小字。
""".strip(),
    },
    {
        "filename": "ch01-vscode-workspace-flow.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "VS Code 项目工作流图",
        "suggested_page": "docs/ch01-tools/ch03-devenv/02-vscode.md",
        "alt": "VS Code 项目工作流图：打开项目文件夹、选择解释器、编辑代码、内置终端运行、调试并提交到 Git。",
        "prompt": """
一张适合 VS Code 入门课程的教学插图，主题是“VS Code 是项目工作台，不只是文本编辑器”。
画面表现 VS Code 工作区连接项目文件树、Python 解释器选择、代码编辑区、内置终端、调试面板和 Git 变更列表。
重点让新手看懂：要打开整个项目文件夹、选择正确解释器、用内置终端运行代码、查看 Git 改动。
风格像清爽的开发者桌面和界面示意结合，不要出现真实品牌 logo，不要生成具体文字，不要出现乱码小字。
""".strip(),
    },
    {
        "filename": "ch02-python-foundation.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Python 编程基础主视觉",
        "suggested_page": "docs/ch02-python/index.md",
        "alt": "Python 编程基础主视觉：变量、函数、数据结构和小项目逐步组成程序。",
        "prompt": """
一张适合中文编程课程的阶段主视觉，主题是“Python 编程基础”。
画面表现变量、函数、列表、字典、文件读写和命令行小工具像积木一样组成一个可运行程序，风格现代、轻松、清晰。
适合新手学习页面，不要生成具体文字，不要出现真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch02-learning-quest-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Python 学习闯关地图",
        "suggested_page": "docs/ch02-python/index.md",
        "alt": "Python 学习闯关地图：第一段程序、数据组织、函数封装、文件保存、异常处理、API 调用和小作品逐步推进。",
        "prompt": """
一张适合 Python 入门首页的学习闯关地图，主题是“从第一段程序到 Python 小作品”。
画面表现新手从写出第一段程序开始，逐步经过组织数据、封装函数、读写文件、处理异常、调用 Web API、接入 AI API，最后完成一个小作品。
风格像清爽的编程冒险地图，用图标、路径和节点表达进阶关系；不要生成具体文字，不要出现真实品牌 logo，不要出现乱码小字。
""".strip(),
    },
    {
        "filename": "ch02-python-ai-backbone.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Python AI 主线能力链",
        "suggested_page": "docs/ch02-python/index.md",
        "alt": "Python AI 主线能力链：语法、数据结构、函数模块、文件异常、第三方库、Web API 和 AI API 串成后续项目基础。",
        "prompt": """
一张适合 Python 编程基础首页的能力链路图，主题是“为什么 Python 是 AI 全栈主线语言”。
画面表现 Python 语法连接数据结构、函数与模块、文件与异常、第三方库、Web API 和 AI API，最后延伸到数据分析、机器学习、RAG 和 Agent。
重点让新手看到每个语法点后面都通向真实 AI 项目能力。
风格现代、工程化、清晰友好；不要生成具体代码文字，不要出现真实品牌 logo，不要出现乱码小字。
""".strip(),
    },
    {
        "filename": "ch02-study-guide-program-loop.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Python 学习指南最小闭环",
        "suggested_page": "docs/ch02-python/study-guide.md",
        "alt": "Python 学习指南最小闭环：输入进入程序，经过变量、条件循环、数据结构、函数模块，最后输出文件、API 或项目结果。",
        "prompt": """
一张适合 Python 学习指南的教学图，主题是“新手第一遍只要抓住输入、处理、输出闭环”。
画面表现输入进入程序，依次经过变量与数据类型、条件和循环、数据结构、函数和模块，最后输出到屏幕、文件、API 或项目作品。
重点帮助初学者把零散语法看成一条稳定主线，而不是背知识点清单。
风格温和、清晰、像老师手绘的现代流程图；不要生成具体文字，不要出现真实品牌 logo，不要出现乱码小字。
""".strip(),
    },
    {
        "filename": "ch02-task-list-workflow.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Python 阶段任务链",
        "suggested_page": "docs/ch02-python/task-list.md",
        "alt": "Python 阶段任务链：基础语法、数据结构、函数、文件保存、异常处理和阶段项目逐步串成编程能力。",
        "prompt": """
一张适合 Python 入门课程的任务路线图，主题是“从语法到项目的 Python 学习任务链”。
画面表现基础语法、数据结构、函数拆分、文件保存、异常处理、阶段项目六个环节逐步连接，最后形成一个能运行的小工具。
风格清爽、现代、适合新手学习，像一张编程闯关路线图；用图标和箭头表达，不要生成具体文字，不要出现真实品牌 logo，不要出现乱码小字。
""".strip(),
    },
    {
        "filename": "ch02-python-ai-workflow.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Python 到 AI 应用工作流",
        "suggested_page": "docs/ch02-python/ch01-basics/01-intro.md",
        "alt": "Python 到 AI 应用工作流：小脚本、文件数据、模型调用、API 封装、RAG 和 Agent 逐步连接。",
        "prompt": """
一张适合 Python 简介页面的教学插图，主题是“Python 如何一路连接到 AI 应用”。
画面表现一个 Python 小脚本逐步扩展为文件处理、数据分析、模型调用、Web API、RAG 知识库和 Agent 工具调用。
重点让新手看到 Python 不是孤立语法，而是一条通向 AI 全栈项目的主线。
风格明亮、友好、工程化插图；不要出现具体代码文字，不要出现真实品牌 logo，不要出现乱码小字。
""".strip(),
    },
    {
        "filename": "ch02-operators-decision-flow.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "运算符与条件判断流程图",
        "suggested_page": "docs/ch02-python/ch01-basics/03-operators.md",
        "alt": "运算符与条件判断流程图：原始数据经过算术、比较和逻辑组合，进入条件分支。",
        "prompt": """
一张适合 Python 运算符入门课程的教学图，主题是“运算符帮助程序计算和做判断”。
画面表现原始数据经过算术运算、比较运算、逻辑组合，最终进入不同条件分支，像一个简单决策流水线。
适合解释准确率、阈值、条件筛选、成员判断等编程场景。
风格简洁、白板教学与科技插画结合；不要生成复杂公式，不要出现真实品牌 logo，不要出现乱码文字。
""".strip(),
    },
    {
        "filename": "ch02-input-output-flow.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Python 输入处理输出流程图",
        "suggested_page": "docs/ch02-python/ch01-basics/04-io.md",
        "alt": "Python 输入处理输出流程图：input 接收用户输入，程序处理后用 f-string 和 print 展示结果。",
        "prompt": """
一张适合 Python 输入输出入门课程的教学图，主题是“程序的最小闭环：输入、处理、输出”。
画面表现用户在终端输入信息，Python 程序进行类型转换和计算，再用格式化文本输出清晰结果。
重点让新手理解 CLI、API、RAG 和 Agent 后面都会重复这个输入到输出的模式。
风格亲切、清晰、像交互式终端示意图；不要生成具体文字，不要出现真实品牌 logo，不要出现乱码小字。
""".strip(),
    },
    {
        "filename": "ch02-modules-package-structure.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "模块与包项目结构图",
        "suggested_page": "docs/ch02-python/ch01-basics/08-modules.md",
        "alt": "模块与包项目结构图：一个 main.py 拆分为 utils、data、api 等模块，并通过 import 复用。",
        "prompt": """
一张适合 Python 模块与包课程的教学图，主题是“代码如何从一个文件拆成可维护项目”。
画面表现一个拥挤的 main.py 文件被拆分成多个清晰模块：工具函数、数据处理、API 调用、配置文件和包目录，它们通过 import 连接。
重点表达模块化让代码更清楚、更容易复用、更适合后续 AI 项目。
风格像项目文件树和积木结构结合，清晰、现代；不要出现真实品牌 logo，不要生成密集小字或乱码文字。
""".strip(),
    },
    {
        "filename": "ch02-file-io-serialization-flow.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "文件读写与序列化流程图",
        "suggested_page": "docs/ch02-python/ch02-advanced/03-file-io.md",
        "alt": "文件读写与序列化流程图：内存中的 Python 数据被序列化写入文件，再读取并还原成对象。",
        "prompt": """
一张适合 Python 文件操作课程的教学图，主题是“数据如何从内存保存到文件再读回来”。
画面表现 Python 内存里的列表、字典或对象，通过序列化变成 JSON/CSV/文本文件，写入磁盘；下次程序启动时再读取并还原。
适合新手理解持久化、配置文件、训练日志和任务数据保存。
风格清晰、实用、像数据仓库和程序内存之间的桥梁；不要出现真实品牌 logo，不要生成密集小字或乱码文字。
""".strip(),
    },
    {
        "filename": "ch02-functional-pipeline.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "函数式数据流水线图",
        "suggested_page": "docs/ch02-python/ch02-advanced/04-functional.md",
        "alt": "函数式数据流水线图：一组数据经过 map 转换、filter 筛选、sorted 排序得到结果。",
        "prompt": """
一张适合 Python 函数式编程入门课程的教学图，主题是“函数式写法像数据流水线”。
画面表现一组数据卡片进入流水线，先被批量转换，再被筛选，再按规则排序，最后输出干净结果。
重点让新手理解 map、filter、sorted key、lambda 的用途，而不是追求复杂技巧。
风格清爽、数据流水线、现代教育插图；不要出现真实品牌 logo，不要生成具体文字或乱码小字。
""".strip(),
    },
    {
        "filename": "ch02-generator-streaming-data.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "生成器流式数据处理图",
        "suggested_page": "docs/ch02-python/ch02-advanced/05-iterators-generators.md",
        "alt": "生成器流式数据处理图：生成器一次只产出一个元素，适合大文件、流式数据和训练数据加载。",
        "prompt": """
一张适合 Python 迭代器与生成器课程的教学图，主题是“生成器一次只产出一个元素，所以更省内存”。
画面左右对比：一边是把巨大数据一次性塞进内存导致拥堵，另一边是生成器像水龙头一样一滴一滴输出数据，供循环逐个处理。
连接到大文件读取、日志流、训练数据加载等 AI 场景。
风格直观、有类比感、教学友好；不要出现真实品牌 logo，不要生成密集文字或乱码。
""".strip(),
    },
    {
        "filename": "ch02-type-hints-quality-flow.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "类型注解与代码质量流程图",
        "suggested_page": "docs/ch02-python/ch02-advanced/06-type-hints.md",
        "alt": "类型注解与代码质量流程图：函数输入输出通过类型注解、编辑器提示、格式化和检查工具变得更易维护。",
        "prompt": """
一张适合 Python 类型注解与代码质量课程的教学图，主题是“从能跑到好维护”。
画面表现一个函数的输入输出被类型注解标清，编辑器提前提示错误，格式化工具整理代码，检查工具发现潜在问题，最后形成更可靠的项目代码。
重点表达类型注解不是给机器看的负担，而是给未来自己和团队看的说明书。
风格现代、清晰、像代码质量仪表盘；不要出现真实品牌 logo，不要生成具体代码文字或乱码。
""".strip(),
    },
    {
        "filename": "ch02-todo-cli-architecture.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "命令行任务管理器架构图",
        "suggested_page": "docs/ch02-python/ch03-projects/01-todo-cli.md",
        "alt": "命令行任务管理器架构图：用户命令被解析后读取 tasks.json，修改任务列表并保存结果。",
        "prompt": """
一张适合 Python 命令行项目课程的架构图，主题是“任务管理器如何从用户命令到文件保存”。
画面表现用户在终端输入新增、查看、完成、删除任务，程序解析命令，读取 tasks.json，修改任务列表，再保存回文件并输出结果。
重点突出数据结构、函数拆分、文件持久化和异常处理如何组合成真正可用的小工具。
风格像小型软件架构图，清晰、亲切；不要出现真实品牌 logo，不要生成具体文字或乱码小字。
""".strip(),
    },
    {
        "filename": "ch02-web-scraper-pipeline.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "网络爬虫数据采集流程图",
        "suggested_page": "docs/ch02-python/ch03-projects/02-web-scraper.md",
        "alt": "网络爬虫数据采集流程图：HTTP 请求获取网页，解析 HTML，清洗结构化数据并保存。",
        "prompt": """
一张适合 Python 爬虫项目课程的流程图，主题是“网络爬虫如何把网页变成结构化数据”。
画面表现发送 HTTP 请求、拿到网页 HTML、解析目标内容、清洗成表格字段、保存到 CSV 或 JSON 的完整链路。
适合新手理解数据不是凭空出现的，而是需要采集、解析、清洗和保存。
风格清爽、网络数据流、教育插图；不要出现真实网站品牌，不要生成具体文字或乱码小字。
""".strip(),
    },
    {
        "filename": "ch02-web-api-request-response.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Web API 请求响应架构图",
        "suggested_page": "docs/ch02-python/ch03-projects/03-web-api.md",
        "alt": "Web API 请求响应架构图：客户端发送请求，FastAPI 路由接收，函数处理后返回 JSON 响应。",
        "prompt": """
一张适合 FastAPI 入门项目课程的架构图，主题是“Web API 如何连接用户、程序和 AI 服务”。
画面表现客户端请求进入后端 API，路由分发给 Python 函数，函数处理数据或调用模型，最后返回 JSON 响应给客户端。
重点让新手理解 API 是把 Python 能力提供给其他程序调用的桥梁，为后续 AI 服务、RAG 和 Agent 打基础。
风格现代后端架构图、清晰友好；不要出现真实品牌 logo，不要生成具体文字或乱码小字。
""".strip(),
    },
    {
        "filename": "ch03-data-visualization.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "数据分析与可视化主视觉",
        "suggested_page": "docs/ch03-data-analysis/index.md",
        "alt": "数据分析与可视化主视觉：表格数据经过清洗、聚合和图表表达形成分析报告。",
        "prompt": """
一张适合数据分析课程的阶段主视觉，主题是“数据分析与可视化”。
画面表现一张凌乱数据表逐步被清洗、聚合、转成折线图、柱状图和分析报告，风格清晰、亲切、现代，适合新手建立数据工作流直觉。
不要生成具体文字，不要出现真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch03-learning-quest-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "数据分析学习闯关地图",
        "suggested_page": "docs/ch03-data-analysis/index.md",
        "alt": "数据分析学习闯关地图：原始数据经过字段观察、清洗、聚合、可视化、解释和报告输出。",
        "prompt": """
一张适合数据分析与可视化首页的学习闯关地图，主题是“从原始数据到可信分析报告”。
画面表现学习者拿到原始数据，依次经过观察字段、检查缺失异常、清洗转换、统计聚合、可视化探索、解释发现、写成报告。
风格像数据侦探路线图，清爽、现代、新手友好；用图标、箭头和节点表达，不要生成具体文字，不要出现真实品牌 logo，不要出现乱码小字。
""".strip(),
    },
    {
        "filename": "ch03-data-analysis-backbone.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "数据分析主线闭环",
        "suggested_page": "docs/ch03-data-analysis/index.md",
        "alt": "数据分析主线闭环：获取数据、理解字段、清洗异常、转换聚合、可视化探索并服务建模或业务决策。",
        "prompt": """
一张适合数据分析课程首页的主线流程图，主题是“AI 项目为什么离不开数据闭环”。
画面表现数据从文件、日志、数据库进入分析流程，经过字段理解、缺失异常清洗、转换聚合、图表探索、结论沉淀，最后服务机器学习建模、RAG 评估或业务决策。
重点让新手看到数据处理不是背 API，而是一个可重复的工作流。
风格专业、清晰、有数据工作台质感；不要生成具体文字，不要出现真实品牌 logo，不要出现乱码小字。
""".strip(),
    },
    {
        "filename": "ch03-study-guide-data-loop.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "数据分析学习指南最小闭环",
        "suggested_page": "docs/ch03-data-analysis/study-guide.md",
        "alt": "数据分析学习指南最小闭环：读取数据、理解字段、清洗整理、统计分析、可视化和写出结论。",
        "prompt": """
一张适合数据分析学习指南的教学图，主题是“新手第一遍只要抓住数据分析最小闭环”。
画面表现数据读进来，先理解字段，再清洗整理，接着统计分析，最后用图表表达结论并写入报告。
重点帮助初学者把 NumPy、Pandas 和可视化看成一条连续数据流，而不是零散库函数。
风格温和、清晰、像老师手绘的现代流程图；不要生成具体文字，不要出现真实品牌 logo，不要出现乱码小字。
""".strip(),
    },
    {
        "filename": "ch03-task-list-workflow.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "数据分析阶段任务链",
        "suggested_page": "docs/ch03-data-analysis/task-list.md",
        "alt": "数据分析阶段任务链：读取数据、检查质量、清洗转换、统计分析、可视化表达和写出结论。",
        "prompt": """
一张适合数据分析阶段任务单的任务链插图，主题是“把原始数据变成可解释结论的六步任务”。
画面表现读取数据、检查质量、清洗转换、统计分析、可视化表达、写出结论和局限六个环节逐步连接。
风格像课程通关路线图，清爽、实用、适合新人执行；不要生成具体文字，不要出现真实品牌 logo，不要出现乱码小字。
""".strip(),
    },
    {
        "filename": "ch03-pure-python-data-flow.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "纯 Python 数据处理痛点图",
        "suggested_page": "docs/ch03-data-analysis/ch01-warmup/01-pure-python-data.md",
        "alt": "纯 Python 数据处理痛点图：CSV 解析、字典列表、手写循环、清洗统计和输出结果串成繁琐流程。",
        "prompt": """
一张适合纯 Python 数据处理预热课的教学图，主题是“为什么处理数据需要 NumPy 和 Pandas”。
画面表现一份 CSV 数据进入纯 Python 脚本后，需要手写解析、列表字典循环、类型转换、缺失处理、统计汇总和结果输出，流程显得笨重但可理解。
右侧隐约展示后续 NumPy/Pandas 会把同样流程变得更清晰高效。
风格像数据侦探工作台，亲切、具体；不要生成具体代码文字，不要出现真实品牌 logo，不要出现乱码小字。
""".strip(),
    },
    {
        "filename": "ch03-numpy-overview-array-engine.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "NumPy 科学计算引擎图",
        "suggested_page": "docs/ch03-data-analysis/ch02-numpy/01-overview.md",
        "alt": "NumPy 科学计算引擎图：ndarray 作为底层数组能力，支撑 Pandas、可视化、机器学习和深度学习。",
        "prompt": """
一张适合 NumPy 概述课程的教学图，主题是“NumPy 是 Python 数据科学的数组引擎”。
画面表现 ndarray 多维数组作为中央引擎，向外支撑 Pandas 表格处理、Matplotlib 可视化、机器学习、深度学习和科学计算。
重点让新手理解 NumPy 不是孤立工具，而是后续数据和 AI 库的共同底座。
风格现代、工程化、清晰友好；不要生成具体文字，不要出现真实品牌 logo，不要出现乱码小字。
""".strip(),
    },
    {
        "filename": "ch03-numpy-indexing-slicing-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "NumPy 索引切片地图",
        "suggested_page": "docs/ch03-data-analysis/ch02-numpy/03-indexing-slicing.md",
        "alt": "NumPy 索引切片地图：基本索引、切片、布尔索引、花式索引、视图和拷贝帮助选择数组子集。",
        "prompt": """
一张适合 NumPy 索引与切片课程的教学图，主题是“从数组里精准取出想要的数据”。
画面表现一个二维数组网格，分别用不同颜色高亮单个元素、连续切片、整行整列、布尔条件选中的格子、花式索引选中的离散格子，并用视图与拷贝做轻量对比。
重点让新手看到索引不是背语法，而是在问“我想取哪一块数据”。
风格清晰、白板教学和数据网格结合；不要生成具体代码文字，不要出现真实品牌 logo，不要出现乱码小字。
""".strip(),
    },
    {
        "filename": "ch03-numpy-reshape-axis-flow.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "NumPy 变形与轴操作图",
        "suggested_page": "docs/ch03-data-analysis/ch02-numpy/05-reshaping.md",
        "alt": "NumPy 变形与轴操作图：同一组元素通过 reshape、转置、拼接和分割变成不同数组形状。",
        "prompt": """
一张适合 NumPy 数组变形课程的教学图，主题是“数据不变，形状在变”。
画面表现一串数字积木先变成二维矩阵，再变成三维数据块，并通过转置、拼接、分割等操作改变组织方式。
重点突出 reshape 不改变元素总数，axis 决定操作方向，适合后续批量数据和图像数据理解。
风格像数学积木和数据管道结合，准确、清晰；不要生成具体文字，不要出现真实品牌 logo，不要出现乱码小字。
""".strip(),
    },
    {
        "filename": "ch03-numpy-linear-algebra-toolkit.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "NumPy 线性代数工具箱",
        "suggested_page": "docs/ch03-data-analysis/ch02-numpy/06-linear-algebra.md",
        "alt": "NumPy 线性代数工具箱：矩阵乘法、点积、范数、解方程、特征值和余弦相似度服务 AI 计算。",
        "prompt": """
一张适合 NumPy 线性代数课程的教学图，主题是“矩阵运算是 AI 计算的工具箱”。
画面表现矩阵乘法、向量点积、范数长度、线性方程组、特征方向、余弦相似度这些工具被放在一个清晰工具箱里，连接到推荐、相似度、降维和模型计算场景。
重点让新手把抽象矩阵操作和真实数据任务联系起来。
风格现代数学白板、直观、有几何感；不要生成具体公式文字，不要出现真实品牌 logo，不要出现乱码小字。
""".strip(),
    },
    {
        "filename": "ch03-numpy-random-statistics-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "NumPy 随机数与统计地图",
        "suggested_page": "docs/ch03-data-analysis/ch02-numpy/07-random-stats.md",
        "alt": "NumPy 随机数与统计地图：随机种子、分布、抽样、描述统计、相关性和蒙特卡洛模拟组成统计实验流程。",
        "prompt": """
一张适合 NumPy 随机数与统计课程的教学图，主题是“用随机数做可重复的统计实验”。
画面表现随机种子控制实验复现，随机分布产生样本，抽样和打乱制造实验数据，描述统计、百分位数、相关性和直方图帮助理解数据，最后连接到蒙特卡洛模拟。
风格像实验室和数据图表结合，清晰、具体、新手友好；不要生成具体文字，不要出现真实品牌 logo，不要出现乱码小字。
""".strip(),
    },
    {
        "filename": "ch03-pandas-roadmap.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Pandas 数据处理路线图",
        "suggested_page": "docs/ch03-data-analysis/ch03-pandas/00-roadmap.md",
        "alt": "Pandas 数据处理路线图：读入数据、看结构、选择过滤、清洗、转换、聚合、合并和输出给图表或模型。",
        "prompt": """
一张适合 Pandas 导读页的路线图，主题是“真实表格数据如何一步步变成分析结果”。
画面表现原始 CSV/Excel/JSON 数据被 Pandas 读入，经过查看结构、选择过滤、清洗缺失异常、转换派生列、分组聚合、多表合并，最后输出给图表、报告或机器学习模型。
重点让新手先建立数据流顺序，再去学习 API。
风格像数据工厂流水线，表格和箭头清晰；不要生成具体文字，不要出现真实品牌 logo，不要出现乱码小字。
""".strip(),
    },
    {
        "filename": "ch03-pandas-read-write-first-look.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Pandas 数据读写初次见面流程",
        "suggested_page": "docs/ch03-data-analysis/ch03-pandas/02-read-write.md",
        "alt": "Pandas 数据读写初次见面流程：读取文件、检查 info/head/shape、确认编码类型、分块处理和导出结果。",
        "prompt": """
一张适合 Pandas 数据读写课程的教学图，主题是“第一次拿到新数据文件该怎么稳稳读进来”。
画面表现 CSV、Excel、JSON、SQL 数据源进入 Pandas，先检查行列规模、字段类型、缺失值、编码和样例行，再根据需要分块读取或导出为新文件。
重点帮助新手避免一上来就分析，而是先和数据做“初次见面”。
风格清晰、实用、像数据入口检查站；不要生成具体代码文字，不要出现真实品牌 logo，不要出现乱码小字。
""".strip(),
    },
    {
        "filename": "ch03-pandas-selection-filter-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Pandas 数据选择与过滤地图",
        "suggested_page": "docs/ch03-data-analysis/ch03-pandas/03-selection-filter.md",
        "alt": "Pandas 数据选择与过滤地图：loc、iloc、布尔条件、isin、between、query 帮助从 DataFrame 中挑出目标数据。",
        "prompt": """
一张适合 Pandas 数据选择与过滤课程的教学图，主题是“从大表里挑出你真正想看的那部分”。
画面表现一张 DataFrame 表格，分别用标签索引、位置索引、布尔条件、多条件组合、范围筛选和 query 查询选出不同数据子集。
重点突出 loc 看标签、iloc 看位置、布尔索引看条件。
风格像数据筛选控制台，清楚、亲切；不要生成具体代码文字，不要出现真实品牌 logo，不要出现乱码小字。
""".strip(),
    },
    {
        "filename": "ch03-pandas-cleaning-workflow.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Pandas 数据清洗工作流",
        "suggested_page": "docs/ch03-data-analysis/ch03-pandas/04-data-cleaning.md",
        "alt": "Pandas 数据清洗工作流：缺失值、重复值、异常值、类型错误和字符串脏数据被检查、记录和修复。",
        "prompt": """
一张适合 Pandas 数据清洗课程的教学图，主题是“把脏数据变成可信数据”。
画面表现一张混乱表格中存在缺失值、重复行、异常值、错误类型、空格和大小写问题，经过检查、记录、修复和验证后变成干净数据集。
重点提醒新手清洗不是随便删除，而是要有原因、有记录、有复核。
风格像数据医院或清洗工作台，直观、友好；不要生成具体文字，不要出现真实品牌 logo，不要出现乱码小字。
""".strip(),
    },
    {
        "filename": "ch03-pandas-transform-pipeline.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Pandas 数据转换流水线",
        "suggested_page": "docs/ch03-data-analysis/ch03-pandas/05-data-transform.md",
        "alt": "Pandas 数据转换流水线：映射、替换、apply、派生列、分箱和标准化让原始字段变成可分析特征。",
        "prompt": """
一张适合 Pandas 数据转换课程的教学图，主题是“把原始字段加工成更有用的分析特征”。
画面表现原始列进入转换流水线，经过映射替换、类型转换、apply 函数、派生新列、分箱、标准化和重命名，最后输出更适合统计和建模的表格。
重点让新手理解转换不是炫技，而是把数据变成问题需要的形状。
风格现代数据流水线，清晰、有工程感；不要生成具体代码文字，不要出现真实品牌 logo，不要出现乱码小字。
""".strip(),
    },
    {
        "filename": "ch03-pandas-merge-concat-join.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Pandas 合并拼接连接图",
        "suggested_page": "docs/ch03-data-analysis/ch03-pandas/07-merge.md",
        "alt": "Pandas 合并拼接连接图：merge 按键匹配，concat 上下或左右拼接，join 按索引连接多张表。",
        "prompt": """
一张适合 Pandas 数据合并课程的教学图，主题是“多张表怎样变成一张可分析的大表”。
画面表现用户表、订单表、商品表通过共同键进行 merge，不同月份数据通过 concat 上下拼接，按索引的数据通过 join 连接。
重点对比 inner、left、outer 的直觉：保留交集、保留左表、保留全集。
风格像表格拼图和关系图结合，清楚、实用；不要生成具体文字，不要出现真实品牌 logo，不要出现乱码小字。
""".strip(),
    },
    {
        "filename": "ch03-pandas-time-series-analysis.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Pandas 时间序列分析图",
        "suggested_page": "docs/ch03-data-analysis/ch03-pandas/08-time-series.md",
        "alt": "Pandas 时间序列分析图：日期解析、时间索引、重采样、滑动窗口和趋势对比帮助分析随时间变化的数据。",
        "prompt": """
一张适合 Pandas 时间序列课程的教学图，主题是“让表格沿着时间轴动起来”。
画面表现日期字段被解析成时间索引，数据按天、周、月重采样，滑动窗口平滑趋势，并用折线图观察周期、峰值和异常点。
连接到学习记录、销售数据、日志监控和模型训练曲线场景。
风格清晰、时间轴与数据图表结合；不要生成具体文字，不要出现真实品牌 logo，不要出现乱码小字。
""".strip(),
    },
    {
        "filename": "ch03-visualization-roadmap.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "数据可视化学习路线图",
        "suggested_page": "docs/ch03-data-analysis/ch04-visualization/00-roadmap.md",
        "alt": "数据可视化学习路线图：先明确表达目标，再选择图表、绘图库、标注样式并用于分析或汇报。",
        "prompt": """
一张适合数据可视化导读页的路线图，主题是“先学会选图，再学会美化”。
画面表现整理好的数据先进入表达目标判断：趋势、对比、分布、关系、相关性，再选择折线图、柱状图、直方图、散点图、热力图等图表，最后优化标题、坐标轴、图例和注释用于分析报告。
重点让新手知道图表是为了回答问题，不是为了装饰。
风格清爽、图表卡片丰富但不拥挤；不要生成具体文字，不要出现真实品牌 logo，不要出现乱码小字。
""".strip(),
    },
    {
        "filename": "ch03-plotly-interactive-dashboard.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Plotly 交互式仪表盘图",
        "suggested_page": "docs/ch03-data-analysis/ch04-visualization/03-plotly.md",
        "alt": "Plotly 交互式仪表盘图：筛选器、悬停提示、缩放、动态图表和网页展示帮助探索数据。",
        "prompt": """
一张适合 Plotly 交互式可视化课程的教学图，主题是“当图表需要被探索，而不只是被观看”。
画面表现一个数据仪表盘，包含筛选器、悬停提示、缩放框选、动态图表联动和网页展示区域，让用户可以自己探索数据。
重点对比静态图适合汇报，交互图适合探索、演示和产品页面。
风格现代仪表盘、清晰、产品感强；不要出现真实品牌 logo，不要生成具体文字或乱码小字。
""".strip(),
    },
    {
        "filename": "ch03-database-roadmap.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "数据库选修学习路线图",
        "suggested_page": "docs/ch03-data-analysis/ch05-database/00-roadmap.md",
        "alt": "数据库选修学习路线图：本地文件、关系型数据库、SQL 查询、Python 连接和数据分析协作逐步连接。",
        "prompt": """
一张适合数据库导读页的学习路线图，主题是“为什么数据分析后面还要补数据库”。
画面表现小规模本地文件适合 Pandas，数据变多、多人协作、需要长期保存时进入关系型数据库，再通过 SQL 查询和 Python 连接回到数据分析流程。
重点让新手理解数据库不是替代 Pandas，而是现实数据来源和持久化系统。
风格像数据档案室和分析工作台连接图，清晰、稳重；不要生成具体文字，不要出现真实品牌 logo，不要出现乱码小字。
""".strip(),
    },
    {
        "filename": "ch03-relational-database-foundation.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "关系型数据库基础图",
        "suggested_page": "docs/ch03-data-analysis/ch05-database/01-relational-db.md",
        "alt": "关系型数据库基础图：数据库、表、行、列、主键、外键、索引和权限共同支撑可靠数据管理。",
        "prompt": """
一张适合关系型数据库入门课程的教学图，主题是“从 Excel 表格到可靠数据库系统”。
画面表现数据库里有多张表，每张表由行、列、字段类型组成，通过主键和外键建立关系，并用索引、事务、权限和备份支撑多人协作和可靠存储。
重点让新手理解数据库不是一个更大的 CSV，而是管理长期数据的系统。
风格清晰、数据库蓝图感、适合教学；不要生成具体文字，不要出现真实品牌 logo，不要出现乱码小字。
""".strip(),
    },
    {
        "filename": "ch03-python-database-bridge.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Python 与数据库协作桥梁图",
        "suggested_page": "docs/ch03-data-analysis/ch05-database/03-python-db.md",
        "alt": "Python 与数据库协作桥梁图：Python 连接数据库、执行 SQL、取回结果、交给 Pandas 分析并写回数据。",
        "prompt": """
一张适合 Python 数据库操作课程的教学图，主题是“代码怎样真正和数据库协作”。
画面表现 Python 程序连接数据库，执行参数化 SQL，取回结果集，交给 Pandas 做分析，再把清洗或统计结果写回数据库。
重点突出连接、游标、查询、防注入、事务提交和 Pandas read_sql/to_sql 的协作关系。
风格像桥梁和数据管道结合，清晰、工程化；不要生成具体代码文字，不要出现真实品牌 logo，不要出现乱码小字。
""".strip(),
    },
    {
        "filename": "ch03-database-design-erd-normalization.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "数据库设计与范式图",
        "suggested_page": "docs/ch03-data-analysis/ch05-database/04-db-design.md",
        "alt": "数据库设计与范式图：实体拆表、主键外键、范式、索引和查询场景共同减少重复和维护风险。",
        "prompt": """
一张适合数据库设计课程的教学图，主题是“先分表，再连表，再补索引”。
画面表现一个混乱的大宽表被拆成用户、订单、商品等实体表，通过主键和外键连接，减少重复和更新冲突；旁边展示索引像目录一样加速查询。
重点让新手理解范式不是背概念，而是为了减少重复、减少冲突、减少维护事故。
风格像仓库货架规划图和 ERD 结构图结合，清晰、实用；不要生成具体文字，不要出现真实品牌 logo，不要出现乱码小字。
""".strip(),
    },
    {
        "filename": "ch04-ai-math.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "AI 数学基础主视觉",
        "suggested_page": "docs/ch04-ai-math/index.md",
        "alt": "AI 数学基础主视觉：向量、矩阵、概率分布和梯度下降连接到模型训练。",
        "prompt": """
一张适合 AI 数学入门课程的阶段主视觉，主题是“AI 数学最小必要基础”。
画面把向量箭头、矩阵网格、概率分布曲线、梯度下降路径和小模型训练连接在一起，像一张温和的学习地图。
风格清晰、具体、帮助新手降低公式恐惧，不要生成具体文字，不要出现真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch04-learning-quest-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "AI 数学学习闯关地图",
        "suggested_page": "docs/ch04-ai-math/index.md",
        "alt": "AI 数学学习闯关地图：向量、矩阵、概率、损失、梯度和优化逐步连接到模型训练。",
        "prompt": """
一张适合 AI 数学基础首页的学习闯关地图，主题是“从数学直觉到模型训练”。
画面表现学习者从向量表示一个样本开始，经过矩阵表示一批数据、概率表达不确定性、损失衡量错误、梯度指出改进方向、优化让模型逐步变好。
重点降低公式恐惧，让新人看到数学概念是一组可用工具，而不是孤立定理。
风格温和、清晰、现代教育插图；不要生成具体公式文字，不要出现真实品牌 logo，不要出现乱码小字。
""".strip(),
    },
    {
        "filename": "ch04-ai-math-backbone.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "AI 数学最小必要主线图",
        "suggested_page": "docs/ch04-ai-math/index.md",
        "alt": "AI 数学最小必要主线图：线性代数负责表示，概率统计负责不确定性，微积分负责损失和优化。",
        "prompt": """
一张适合 AI 数学基础首页的主线关系图，主题是“为什么这里叫最小必要数学基础”。
画面分成三条支柱：线性代数负责数据和参数表示，概率统计负责不确定性、评估和预测置信度，微积分与优化负责损失函数、梯度和参数更新；三条支柱共同支撑机器学习、深度学习和大模型。
重点让新手知道第一遍不用学完整数学体系，只要先抓住最高频的模型语言。
风格清爽、结构化、像课程能力架构图；不要生成具体文字，不要出现真实品牌 logo，不要出现乱码小字。
""".strip(),
    },
    {
        "filename": "ch04-study-guide-math-minimum-loop.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "AI 数学学习指南最小闭环",
        "suggested_page": "docs/ch04-ai-math/study-guide.md",
        "alt": "AI 数学学习指南最小闭环：向量矩阵表示数据，概率统计衡量不确定性，导数梯度优化参数。",
        "prompt": """
一张适合 AI 数学学习指南的教学图，主题是“第一遍数学只抓三个模型直觉”。
画面表现向量和矩阵把数据变成可计算表示，概率和统计衡量不确定性与评估，导数和梯度告诉模型如何更新参数，最后汇入机器学习模型。
重点帮助初学者避免陷入证明细节，先用图和代码建立直觉。
风格温和、清晰、像老师手绘的现代流程图；不要生成具体公式文字，不要出现真实品牌 logo，不要出现乱码小字。
""".strip(),
    },
    {
        "filename": "ch04-linear-algebra-chapter-flow.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "线性代数章节关系图",
        "suggested_page": "docs/ch04-ai-math/ch01-linear-algebra/00-roadmap.md",
        "alt": "线性代数章节关系图：现实数据先表示成向量，再堆成矩阵，经过变换找到特征方向并连接 PCA 和神经网络。",
        "prompt": """
一张适合线性代数导读页的章节关系图，主题是“向量、矩阵、特征值和向量空间如何串起来”。
画面表现现实世界的数据先被写成向量，很多向量堆成矩阵，矩阵完成批量变换，特征向量找到特殊方向，进一步连接 PCA 降维、神经网络矩阵乘法和向量空间视角。
重点让新手先看懂整章关系，再进入公式。
风格像数学地图和 AI 数据流结合，清晰、具体；不要生成具体文字，不要出现真实品牌 logo，不要出现乱码小字。
""".strip(),
    },
    {
        "filename": "ch04-vector-ai-meaning-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "向量 AI 含义地图",
        "suggested_page": "docs/ch04-ai-math/ch01-linear-algebra/01-vectors.md",
        "alt": "向量 AI 含义地图：现实对象被写成数字向量，通过点积和余弦相似度连接推荐、搜索、RAG 和语义匹配。",
        "prompt": """
一张适合向量入门课程的教学图，主题是“向量是 AI 世界的信息卡片”。
画面表现用户资料、词语、图片或文档被编码成数字向量，然后通过点积、夹角和余弦相似度比较方向是否接近，最终用于推荐、搜索、RAG 检索和语义匹配。
重点让新手把向量从抽象箭头理解成“可计算的对象表示”。
风格直观、友好、带几何箭头和数据卡片；不要生成具体文字，不要出现真实品牌 logo，不要出现乱码小字。
""".strip(),
    },
    {
        "filename": "ch04-matrix-batch-transform-flow.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "矩阵批量变换流程图",
        "suggested_page": "docs/ch04-ai-math/ch01-linear-algebra/02-matrices.md",
        "alt": "矩阵批量变换流程图：单个样本向量堆成样本矩阵，乘权重矩阵后得到批量输出。",
        "prompt": """
一张适合矩阵入门课程的教学图，主题是“矩阵让模型一次处理一批样本”。
画面表现单个样本向量像信息卡片，很多样本堆成样本矩阵 X，经过权重矩阵 W 的批量变换，输出预测结果或下一层表示。
重点让新手理解为什么机器学习和神经网络代码里到处都是 X @ W。
风格像数据表格、矩阵网格和神经网络层结合，清晰、工程化；不要生成具体代码文字，不要出现真实品牌 logo，不要出现乱码小字。
""".strip(),
    },
    {
        "filename": "ch04-eigen-pca-direction-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "特征方向与 PCA 地图",
        "suggested_page": "docs/ch04-ai-math/ch01-linear-algebra/03-eigenvalues.md",
        "alt": "特征方向与 PCA 地图：矩阵变换中少数方向不改变方向，特征值衡量拉伸倍数，PCA 用它找重要方向。",
        "prompt": """
一张适合特征值与特征向量课程的教学图，主题是“在变化中找到不变的特殊方向”。
画面表现一个矩阵变换让大多数箭头方向改变，但少数特殊方向只被拉伸或缩短，这些方向连接到特征向量和特征值；右侧展示 PCA 沿最重要方向压缩数据。
重点降低特征值概念的陌生感，让新人先抓住“特殊方向”和“信息保留”。
风格几何直觉强、清晰、有动画感；不要生成具体公式文字，不要出现真实品牌 logo，不要出现乱码小字。
""".strip(),
    },
    {
        "filename": "ch04-vector-space-high-level-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "向量空间高层理解图",
        "suggested_page": "docs/ch04-ai-math/ch01-linear-algebra/04-vector-spaces.md",
        "alt": "向量空间高层理解图：向量、线性无关、基、维度、线性变换和 SVD 串成更高层的线性代数视角。",
        "prompt": """
一张适合向量空间选修课的高层理解图，主题是“把向量、矩阵和变换放进同一个框架”。
画面表现一组向量是否冗余，如何形成基和维度，矩阵如何表示线性变换，SVD 如何把复杂变换拆成可理解的方向和尺度。
重点让新手知道这一节是整理视角，不是必须一次吃透的高级理论。
风格像概念地图和空间网格结合，清晰、抽象但亲切；不要生成具体公式文字，不要出现真实品牌 logo，不要出现乱码小字。
""".strip(),
    },
    {
        "filename": "ch04-probability-chapter-flow.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "概率统计章节关系图",
        "suggested_page": "docs/ch04-ai-math/ch02-probability/00-roadmap.md",
        "alt": "概率统计章节关系图：概率描述事件，分布描述随机现象，统计推断从数据反推规律，信息论衡量不确定性和预测误差。",
        "prompt": """
一张适合概率与统计导读页的章节关系图，主题是“从不确定事件到模型损失函数”。
画面表现现实世界有不确定性，概率描述单个事件，概率分布描述随机变量整体规律，统计推断从观测数据反推参数，信息论衡量不确定性和预测误差，最后连接分类损失、贝叶斯推断、A/B 测试、决策树和语言模型。
重点让新手看到概率统计不是公式集合，而是 AI 处理不确定性的语言。
风格现代、清晰、数据科学感；不要生成具体文字，不要出现真实品牌 logo，不要出现乱码小字。
""".strip(),
    },
    {
        "filename": "ch04-probability-bayes-update-flow.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "概率与贝叶斯更新流程图",
        "suggested_page": "docs/ch04-ai-math/ch02-probability/01-probability-basics.md",
        "alt": "概率与贝叶斯更新流程图：事件概率在获得新证据后通过条件概率和贝叶斯法则更新判断。",
        "prompt": """
一张适合概率基础课程的教学图，主题是“有了新证据后，判断会怎样更新”。
画面表现一个不确定事件先有初始概率，新证据进入后，通过条件概率和贝叶斯更新改变判断，最后连接到垃圾邮件判断、医疗检测和分类模型概率输出。
重点让新手理解概率不是 0 或 1，而是可以随着证据更新的信心。
风格像侦探证据板和概率仪表盘结合，清晰、具体；不要生成具体公式文字，不要出现真实品牌 logo，不要出现乱码小字。
""".strip(),
    },
    {
        "filename": "ch04-distribution-random-world-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "概率分布随机现象地图",
        "suggested_page": "docs/ch04-ai-math/ch02-probability/02-distributions.md",
        "alt": "概率分布随机现象地图：很多次随机结果堆起来形成分布，离散分布处理计数标签，连续分布处理噪声误差和连续特征。",
        "prompt": """
一张适合概率分布课程的教学图，主题是“随机现象整体长什么样”。
画面表现很多次随机结果堆叠形成分布形状，左侧是离散结果的柱状分布，右侧是连续变量的曲线分布，并连接到标签计数、噪声、误差、连续特征和随机初始化。
重点让新手先看懂分布形状和使用场景，而不是背所有分布名称。
风格像统计实验可视化，清晰、亲切；不要生成具体文字，不要出现真实品牌 logo，不要出现乱码小字。
""".strip(),
    },
    {
        "filename": "ch04-statistical-inference-data-to-parameter.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "统计推断从数据到参数图",
        "suggested_page": "docs/ch04-ai-math/ch02-probability/03-statistical-inference.md",
        "alt": "统计推断从数据到参数图：观测数据用于估计参数，MLE、MAP、假设检验和 A/B 测试帮助做判断。",
        "prompt": """
一张适合统计推断课程的教学图，主题是“看到数据后，怎样反推出背后的参数和结论”。
画面表现观测样本进入推断流程，先提出参数假设，通过 MLE 找到最能解释数据的参数，通过 MAP 加入先验，再通过假设检验和 A/B 测试判断差异是否可靠。
重点让新手理解 MLE/MAP 不是孤立公式，而是在回答“什么解释最合理”。
风格像数据实验室和推理流程结合，清晰、专业；不要生成具体公式文字，不要出现真实品牌 logo，不要出现乱码小字。
""".strip(),
    },
    {
        "filename": "ch04-information-theory-loss-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "信息论到损失函数地图",
        "suggested_page": "docs/ch04-ai-math/ch02-probability/04-information-theory.md",
        "alt": "信息论到损失函数地图：信息量、熵、交叉熵和 KL 散度连接到分类损失和语言模型训练。",
        "prompt": """
一张适合信息论入门课程的教学图，主题是“从惊讶程度到模型损失函数”。
画面表现越意外的事件信息量越大，一个分布越不确定熵越高，预测分布和真实分布之间的差距由交叉熵和 KL 散度衡量，最后连接到分类损失函数和语言模型训练。
重点让新手理解交叉熵不是突然出现的 API，而是衡量预测分布差距的工具。
风格像概率分布和模型训练面板结合，清晰、现代；不要生成具体公式文字，不要出现真实品牌 logo，不要出现乱码小字。
""".strip(),
    },
    {
        "filename": "ch04-calculus-training-flow.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "微积分优化章节关系图",
        "suggested_page": "docs/ch04-ai-math/ch03-calculus/00-roadmap.md",
        "alt": "微积分优化章节关系图：导数衡量变化，梯度指明方向，梯度下降更新参数，反向传播高效计算所有梯度。",
        "prompt": """
一张适合微积分与优化导读页的章节关系图，主题是“模型到底是怎么学起来的”。
画面表现模型有损失函数，导数先看一个变量如何影响损失，梯度把多个参数的影响合成方向，梯度下降沿负梯度更新参数，反向传播高效算出深层网络中所有参数的梯度。
重点让新手把 loss.backward 和 optimizer.step 背后的直觉先看懂。
风格像训练过程剖面图，清晰、有运动方向；不要生成具体公式文字，不要出现真实品牌 logo，不要出现乱码小字。
""".strip(),
    },
    {
        "filename": "ch04-derivative-change-rate-bridge.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "导数变化率桥梁图",
        "suggested_page": "docs/ch04-ai-math/ch03-calculus/01-derivatives.md",
        "alt": "导数变化率桥梁图：一个量的变化率连接到梯度、梯度下降和反向传播的优化主线。",
        "prompt": """
一张适合导数入门课程的教学图，主题是“导数是变化率，也是优化的第一块地基”。
画面表现一条曲线上的切线斜率表示一个量变化得多快，再从单变量变化率延伸到多变量梯度、负梯度方向、梯度下降和反向传播。
重点让新手知道导数不是孤立知识点，而是在为模型参数更新做准备。
风格像曲线、切线和训练路线结合，直观、温和；不要生成具体公式文字，不要出现真实品牌 logo，不要出现乱码小字。
""".strip(),
    },
    {
        "filename": "ch04-gradient-parameter-knobs-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "偏导与梯度调参旋钮图",
        "suggested_page": "docs/ch04-ai-math/ch03-calculus/02-partial-derivatives-gradient.md",
        "alt": "偏导与梯度调参旋钮图：固定其他变量只看一个旋钮的影响，所有偏导合成梯度给出整体调整方向。",
        "prompt": """
一张适合偏导数与梯度课程的教学图，主题是“多参数模型像一台有很多旋钮的机器”。
画面表现水温、研磨、粉量、时间等旋钮影响咖啡味道的类比，偏导数逐个检查每个旋钮的影响，梯度把所有影响合成一张整体调参指南，负梯度指向让损失下降最快的方向。
重点让新手理解偏导和梯度不是抽象符号，而是多变量调参信息。
风格像机器控制台和数学等高线结合，清晰、形象；不要生成具体文字，不要出现真实品牌 logo，不要出现乱码小字。
""".strip(),
    },
    {
        "filename": "ch04-gradient-descent-iteration-loop.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "梯度下降迭代闭环图",
        "suggested_page": "docs/ch04-ai-math/ch03-calculus/03-gradient-descent.md",
        "alt": "梯度下降迭代闭环图：当前位置计算梯度，沿负梯度方向走一步，损失变小并重复直到收敛。",
        "prompt": """
一张适合梯度下降课程的教学图，主题是“蒙着眼睛下山的模型训练过程”。
画面表现参数点在损失地形上，从当前位置计算坡度方向，沿负梯度方向走一步，来到损失更小的新位置，不断重复直到接近低谷。
同时表现学习率太大容易越过低谷、太小会走得慢的直觉。
风格像山谷地形和训练循环结合，清晰、直观；不要生成具体公式文字，不要出现真实品牌 logo，不要出现乱码小字。
""".strip(),
    },
    {
        "filename": "ch04-backprop-chain-rule-training-bridge.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "反向传播链式法则桥梁图",
        "suggested_page": "docs/ch04-ai-math/ch03-calculus/04-chain-rule-backprop.md",
        "alt": "反向传播链式法则桥梁图：前向传播算出输出和损失，损失沿计算图反向传递，用链式法则得到每个参数梯度。",
        "prompt": """
一张适合链式法则与反向传播课程的教学图，主题是“从损失往回追责，算出每个参数该怎么改”。
画面表现前向传播先经过多层计算得到输出和损失，随后损失沿计算图反向流动，每一层用链式法则把梯度传回去，最终得到所有参数的梯度并交给优化器更新。
重点让新手理解 PyTorch backward 本质是在自动执行这条反向计算链。
风格像计算图流水线和回流箭头结合，清晰、工程化；不要生成具体代码文字，不要出现真实品牌 logo，不要出现乱码小字。
""".strip(),
    },
    {
        "filename": "ch04-linear-algebra-roadmap.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "线性代数学习地图",
        "suggested_page": "docs/ch04-ai-math/ch01-linear-algebra/00-roadmap.md",
        "alt": "线性代数学习地图：向量、矩阵、特殊方向和 AI 场景串成学习顺序。",
        "prompt": """
一张适合 AI 数学入门课程的学习地图插图，主题是“线性代数从向量到矩阵再到 AI 应用”。
画面用清晰的视觉路径表达：向量像箭头表示对象和方向，矩阵像网格变换表示批量改变空间，特征方向像不被旋转的特殊箭头，最后连接到 embedding、attention、PCA 等 AI 场景。
风格温和、现代、课堂白板与科技插画结合，帮助新手先建立直觉再看公式。
不要出现真实品牌 logo，不要生成密集小字，不要生成乱码文字，可以用图标、箭头和分区表达结构。
""".strip(),
    },
    {
        "filename": "eigenvalue-special-directions.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "特征向量特殊方向示意图",
        "suggested_page": "docs/ch04-ai-math/ch01-linear-algebra/03-eigenvalues.md",
        "alt": "特征向量特殊方向示意图：矩阵变换后方向不变，只发生拉伸或压缩。",
        "prompt": """
一张适合线性代数入门课程的教学插图，主题是“特征向量是矩阵变换中方向不变的特殊箭头”。
画面表现一个坐标网格经过矩阵作用被拉伸或剪切，大多数箭头方向发生改变，只有一两根高亮箭头仍沿原方向，只是变长或变短。
构图要让新手一眼看懂“方向不变，长度变化”，风格清晰、数学可视化、颜色区分明显。
不要出现真实品牌 logo，不要生成复杂公式，不要生成密集小字或乱码文字。
""".strip(),
    },
    {
        "filename": "vector-space-basis-span.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "向量空间基向量张成示意图",
        "suggested_page": "docs/ch04-ai-math/ch01-linear-algebra/04-vector-spaces.md",
        "alt": "向量空间基向量张成示意图：基向量通过线性组合覆盖可到达空间。",
        "prompt": """
一张适合线性代数入门课程的教学图，主题是“基向量如何张成一个向量空间”。
画面表现两个不同颜色的基向量从原点出发，通过平移网格和线性组合覆盖整个二维平面；同时用一个目标点展示“沿着两个基向量走几步就能到达”。
风格像新手友好的数学白板，几何直觉强，空间感清楚。
不要出现真实品牌 logo，不要生成复杂公式，不要生成难以阅读的小字。
""".strip(),
    },
    {
        "filename": "ch04-probability-roadmap.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "概率与统计学习地图",
        "suggested_page": "docs/ch04-ai-math/ch02-probability/00-roadmap.md",
        "alt": "概率与统计学习地图：不确定性、概率、分布、推断和信息论如何连接到模型。",
        "prompt": """
一张适合 AI 概率统计入门课程的学习地图插图，主题是“从不确定性到模型判断”。
画面表现现实世界里有不确定事件，经过概率描述、概率分布、样本统计、参数估计、贝叶斯更新和信息熵，最后连接到模型预测置信度。
风格清晰、温和、适合新人降低概率论畏难感，像一张由图标和箭头组成的学习路线图。
不要出现真实品牌 logo，不要生成密集小字，不要生成乱码文字。
""".strip(),
    },
    {
        "filename": "distribution-family-comparison.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "概率分布家族对比图",
        "suggested_page": "docs/ch04-ai-math/ch02-probability/02-distributions.md",
        "alt": "概率分布家族对比图：二项、泊松、正态和中心极限定理对应不同随机现象。",
        "prompt": """
一张适合概率分布课程的教学插图，主题是“不同分布描述不同类型的随机现象”。
画面用四个并列卡片表达：抛硬币次数的离散柱状分布、单位时间事件次数的稀疏柱状分布、测量误差的钟形曲线、许多随机因素叠加后趋近钟形曲线。
整体要像数据科学课堂上的可视化对比图，图形准确、留白充足、颜色区分清楚。
不要出现真实品牌 logo，不要生成复杂公式，不要生成密集小字或乱码文字。
""".strip(),
    },
    {
        "filename": "mle-likelihood-curve.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "最大似然估计似然曲线图",
        "suggested_page": "docs/ch04-ai-math/ch02-probability/03-statistical-inference.md",
        "alt": "最大似然估计似然曲线图：让观测数据最可能出现的参数就是 MLE 选择。",
        "prompt": """
一张中文 AI 数学课程白板风格教学图，主题是“最大似然估计：找到最能解释数据的参数”。
画面表现一条平滑的似然曲线，中间有最高点；下方有观测样本点，若干候选参数像滑块一样比较，最高点被高亮，表达“哪个参数让这些数据最可能出现”。
请使用中文标题“最大似然估计”，少量中文标签即可，例如“候选参数”“似然最高”“观测数据”。不要使用英文标题，不要出现英文大段说明。
风格像现代数学课程插图，曲线和峰值清楚，帮助新手从直觉理解 MLE。
不要出现真实品牌 logo，不要生成复杂公式，不要生成难以阅读的小字或乱码文字。
""".strip(),
    },
    {
        "filename": "ch04-calculus-roadmap.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "微积分与优化学习地图",
        "suggested_page": "docs/ch04-ai-math/ch03-calculus/00-roadmap.md",
        "alt": "微积分与优化学习地图：导数、偏导、梯度、梯度下降和反向传播组成模型学习链路。",
        "prompt": """
一张适合 AI 微积分与优化入门课程的学习地图插图，主题是“模型如何根据 loss 的方向学习”。
画面用视觉链路表达：函数曲线上的切线斜率、二维曲面上的偏导、等高线上的梯度方向、参数点沿坡下降、神经网络反向传播梯度。
风格温和、清晰、把抽象公式变成地形和路线，适合新手理解优化。
不要出现真实品牌 logo，不要生成密集小字，不要生成乱码文字。
""".strip(),
    },
    {
        "filename": "derivative-tangent-slope.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "导数切线斜率示意图",
        "suggested_page": "docs/ch04-ai-math/ch03-calculus/01-derivatives.md",
        "alt": "导数切线斜率示意图：某一点的切线越陡，函数在附近变化越快。",
        "prompt": """
一张适合微积分入门课程的教学插图，主题是“导数就是某一点附近变化有多快”。
画面表现一条平滑曲线，曲线上选中一点并画出切线，旁边用小车上坡或温度曲线的类比表达斜率越陡变化越快。
构图清爽、几何关系明确、颜色温和，帮助新手把导数看成局部速度。
不要出现真实品牌 logo，不要生成复杂公式，不要生成难以阅读的小字。
""".strip(),
    },
    {
        "filename": "gradient-contour-field.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "梯度等高线方向场图",
        "suggested_page": "docs/ch04-ai-math/ch03-calculus/02-partial-derivatives-gradient.md",
        "alt": "梯度等高线方向场图：负梯度方向引导参数一步步走向更低 loss。",
        "prompt": """
一张适合梯度课程的教学图，主题是“梯度指向上坡最快方向，负梯度指向下降方向”。
画面表现像地图等高线一样的损失函数地形，多个箭头组成方向场，参数点沿着负梯度箭头一步步走向低谷。
风格清晰、空间直觉强，像机器学习优化的地形地图，颜色区分高低区域。
不要出现真实品牌 logo，不要生成复杂公式，不要生成密集小字或乱码文字。
""".strip(),
    },
    {
        "filename": "chain-rule-backprop-graph.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "链式法则计算图与反向传播示意图",
        "suggested_page": "docs/ch04-ai-math/ch03-calculus/04-chain-rule-backprop.md",
        "alt": "链式法则计算图与反向传播示意图：前向计算损失，反向逐层传播梯度。",
        "prompt": """
一张适合神经网络数学基础课程的教学图，主题是“链式法则如何支撑反向传播”。
画面表现一个简洁计算图：输入经过若干计算节点得到预测和损失，蓝色箭头表示前向计算，红色箭头从损失反向传回每个节点表示梯度影响。
风格现代、工程白板感强，重点突出“前向算结果，反向分责任”。
不要出现真实品牌 logo，不要生成复杂公式，不要生成乱码文字或密集小字。
""".strip(),
    },
    {
        "filename": "math-study-loop.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "AI 数学学习循环图",
        "suggested_page": "docs/ch04-ai-math/study-guide.md",
        "alt": "AI 数学学习循环图：直觉解释、小数字例子、代码实验、模型连接和误区复盘形成循环。",
        "prompt": """
一张适合 AI 数学学习指南的流程插图，主题是“数学学习不要死磕证明，先建立可运行的理解循环”。
画面表现一个学习者围绕五个环节循环：直觉类比、小数字手算、图形可视化、代码实验、连接到模型训练和复盘误区。
风格鼓励、温暖、清晰，有学习陪伴感，适合新人看到后觉得数学可以拆小步学习。
不要出现真实品牌 logo，不要生成密集小字，不要生成乱码文字。
""".strip(),
    },
    {
        "filename": "math-task-checklist.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "数学最小通关任务单图",
        "suggested_page": "docs/ch04-ai-math/task-list.md",
        "alt": "数学最小通关任务单图：向量相似度、概率分布和梯度下降三个小实验组成最低通关。",
        "prompt": """
一张适合 AI 数学阶段任务清单的插图，主题是“三个最小实验通关数学基础”。
画面表现三个并列小任务：画向量并比较相似度、采样随机数并观察分布、让一个点沿损失曲面下降。每个任务像可完成的实验卡片，带有轻量 check 标记和代码工作台氛围。
风格新手友好、实战感强、让学习者愿意动手。
不要出现真实品牌 logo，不要生成密集小字，不要生成乱码文字。
""".strip(),
    },
    {
        "filename": "ch05-machine-learning.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "机器学习主视觉",
        "suggested_page": "docs/ch05-machine-learning/index.md",
        "alt": "机器学习主视觉：数据、baseline、评估、特征工程和模型复盘组成建模闭环。",
        "prompt": """
一张适合机器学习课程的阶段主视觉，主题是“机器学习入门到实战”。
画面表现数据表进入建模流水线，经过训练集测试集、baseline、评估指标、错误分析和特征工程，最终形成一份模型报告。
风格专业但新手友好，不要生成具体文字，不要出现真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch05-learning-quest-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "机器学习学习闯关地图",
        "suggested_page": "docs/ch05-machine-learning/index.md",
        "alt": "机器学习学习闯关地图：定义问题、准备数据、建立 baseline、选择指标、训练模型、错误分析和建模报告。",
        "prompt": """
一张适合机器学习课程首页的学习闯关地图，主题是“从问题到可复盘的模型项目”。
画面表现定义问题、准备数据、建立 baseline、选择指标、训练模型、分析错误、改进特征和模型、形成建模报告这些环节逐步连接。
风格延续课程之前的视觉优先路线，图形清晰、流程感强、新手友好。
文字不是主体；如确实需要标签，中英文自然混用：中文写概念提示，标准机器学习术语、API、变量名和公式保留英文或数学形式，例如 baseline、fit、predict、AUC、X/y、loss。不要出现整段英文说明、无意义英文、乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch05-modeling-loop-backbone.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "机器学习建模主线闭环",
        "suggested_page": "docs/ch05-machine-learning/index.md",
        "alt": "机器学习建模主线闭环：业务问题、任务定义、数据准备、baseline、指标评估、特征工程、模型改进和误差分析。",
        "prompt": """
一张适合机器学习课程首页的建模闭环图，主题是“机器学习不是模型名称大全，而是一套可复盘流程”。
画面表现业务问题进入任务定义，经过数据准备、baseline 模型、指标评估、特征工程、模型改进、误差分析与解释，最后回到下一轮实验。
风格像数据科学实验台和项目看板结合，强调迭代、评估和复盘。
文字不是主体；如确实需要标签，中英文自然混用：中文写概念提示，标准术语、API、变量名和公式保留英文或数学形式，例如 baseline、train/test、AUC、RMSE、Pipeline。不要出现整段英文说明、无意义英文、乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch05-study-guide-project-loop.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "机器学习学习指南项目循环",
        "suggested_page": "docs/ch05-machine-learning/study-guide.md",
        "alt": "机器学习学习指南项目循环：问题、任务定义、数据划分、baseline、指标评估、特征工程、模型改进和误差分析。",
        "prompt": """
一张适合机器学习学习指南的流程插图，主题是“第一遍机器学习只抓完整项目循环”。
画面表现学习者围绕项目循环前进：理解问题、定义任务、划分数据、训练 baseline、查看指标、做特征工程、改进模型、做误差分析。
风格温暖、清晰、有陪伴感，帮助新人不要被算法名称吓住。
文字不是主体；如确实需要标签，中英文自然混用：中文写概念提示，标准术语、API、变量名和公式保留英文或数学形式，例如 baseline、fit、predict、score、X/y。不要出现整段英文说明、无意义英文、乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch05-basics-chapter-flow.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "机器学习基础章节关系图",
        "suggested_page": "docs/ch05-machine-learning/ch01-ml-basics/00-roadmap.md",
        "alt": "机器学习基础章节关系图：Python 数据分析、AI 数学、机器学习基础、sklearn、监督学习、无监督学习、评估与特征工程逐步连接。",
        "prompt": """
一张适合机器学习基础导读页的章节关系图，主题是“从数据和数学进入机器学习基础”。
画面表现 Python 与数据分析提供数据处理能力，AI 数学提供向量、概率和优化直觉，随后进入机器学习基础、sklearn 最小流程、监督学习、无监督学习、评估与特征工程。
风格像课程路线图和桥梁结合，清晰、有层次。
文字不是主体；如确实需要标签，中英文自然混用：中文写概念提示，标准术语和 API 保留英文，例如 sklearn、fit、predict、score。不要出现整段英文说明、无意义英文、乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch05-task-type-decision-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "机器学习任务类型判断图",
        "suggested_page": "docs/ch05-machine-learning/ch01-ml-basics/01-what-is-ml.md",
        "alt": "机器学习任务类型判断图：先问有没有标签，再判断监督学习、无监督学习、分类、回归、聚类、降维和异常检测。",
        "prompt": """
一张适合“什么是机器学习”页面的任务判断图，主题是“先问问题类型，再选学习方法”。
画面表现一个决策树：先问有没有标签；有标签进入监督学习，再分分类和回归；没有标签进入无监督学习，再分聚类、降维和异常检测。
风格像清晰的路线选择图，适合新人快速判断任务类型。
文字不是主体；如确实需要标签，中英文自然混用：中文写判断问题，标准术语保留英文，例如 Regression、Classification、Clustering，但不要整张图全英文。不要出现无意义英文、乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch05-sklearn-fit-predict-loop.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "sklearn fit predict 统一流程图",
        "suggested_page": "docs/ch05-machine-learning/ch01-ml-basics/02-sklearn-intro.md",
        "alt": "sklearn fit predict 统一流程图：准备 X 和 y，创建模型，fit 学参数，predict 或 transform 用到新数据，score 或 metric 评估结果。",
        "prompt": """
一张适合 sklearn 入门课程的流程图，主题是“sklearn 把不同模型统一成同一套操作习惯”。
画面表现准备 X 和 y，创建模型或变换器，fit 从训练数据中学习参数，predict 或 transform 用到新数据，score 或 metric 判断效果。
风格像工程白板和数据流水线结合，简洁、清楚。
文字不是主体；这里应该保留英文 API 和变量名：X、y、fit、predict、transform、score、metric。其他说明可以用少量中文短标签。不要出现整段英文说明、乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch05-math-to-ml-training-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "数学到机器学习训练地图",
        "suggested_page": "docs/ch05-machine-learning/ch01-ml-basics/03-math-to-ml-bridge.md",
        "alt": "数学到机器学习训练地图：线性代数组织 X 和 w，概率统计描述不确定性和 loss，微积分提供梯度和参数更新方向。",
        "prompt": """
一张适合“数学到机器学习桥梁”页面的概念图，主题是“第 4 章数学如何进入第 5 章建模流程”。
画面分成三股力量汇入训练循环：线性代数把数据组织成 X、参数组织成 w；概率统计描述不确定性、损失和评估；微积分通过梯度告诉参数怎么更新。
风格像三条管道汇入模型训练引擎，直观、有桥梁感。
文字不是主体；公式、变量和标准术语保留英文或数学符号，例如 X、w、loss、gradient。其他说明可以用少量中文短标签。不要出现整段英文说明、乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch05-supervised-chapter-flow.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "监督学习章节关系图",
        "suggested_page": "docs/ch05-machine-learning/ch02-supervised/00-roadmap.md",
        "alt": "监督学习章节关系图：线性回归、逻辑回归、决策树和集成学习由简单到复杂逐步连接。",
        "prompt": """
一张适合监督学习导读页的章节关系图，主题是“从最简单模型到更强模型的监督学习主线”。
画面表现带标签数据进入模型学习路径：线性回归预测连续值，逻辑回归输出分类概率，决策树做规则分裂，集成学习把多个模型组合成更稳结果。
风格像课程路线图和模型进化图结合，清晰、有层次。
文字不是主体；如确实需要标签，中英文自然混用：中文写概念提示，标准术语保留英文，例如 Regression、Classification、Bagging、Boosting。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch05-linear-regression-learning-flow.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "线性回归学习主线图",
        "suggested_page": "docs/ch05-machine-learning/ch02-supervised/01-linear-regression.md",
        "alt": "线性回归学习主线图：连续值预测、模型形式、误差目标、正规方程或梯度下降、残差分析、正则化和调参。",
        "prompt": """
一张适合线性回归课程的学习主线图，主题是“先看任务，再看模型、损失、求解和改进”。
画面表现连续值预测任务从散点数据开始，建立一条拟合线，计算误差和 loss，用正规方程或梯度下降求解，再通过残差分析、多项式和正则化改进。
风格像数学白板和实战流程结合，清晰、温和。
文字不是主体；公式和标准术语保留英文或符号，例如 y = wx + b、loss、Ridge、Lasso。其他说明可以用少量中文短标签。不要出现整段英文说明、乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch05-logistic-classification-flow.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "逻辑回归分类主线图",
        "suggested_page": "docs/ch05-machine-learning/ch02-supervised/02-logistic-regression.md",
        "alt": "逻辑回归分类主线图：分类任务、线性分数、Sigmoid 概率、交叉熵损失、决策边界和多分类扩展。",
        "prompt": """
一张适合逻辑回归课程的分类主线图，主题是“从线性分数到分类概率和决策边界”。
画面表现分类数据点先经过线性打分，再通过 Sigmoid 变成 0 到 1 的概率，使用交叉熵 loss 训练，最后形成决策边界并可扩展到多分类。
风格像二维分类图和概率仪表盘结合，直观、清晰。
文字不是主体；公式和标准术语保留英文或符号，例如 Sigmoid、Cross Entropy、p、threshold。其他说明可以用少量中文短标签。不要出现整段英文说明、乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch05-decision-tree-learning-flow.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "决策树学习主线图",
        "suggested_page": "docs/ch05-machine-learning/ch02-supervised/03-decision-trees.md",
        "alt": "决策树学习主线图：树是一串 if-else 规则，每次分裂让节点更纯，树越深越容易过拟合，通过剪枝控制复杂度。",
        "prompt": """
一张适合决策树课程的学习主线图，主题是“树模型从规则分裂到复杂度控制”。
画面表现一棵树用一串 if-else 规则分裂数据，每次分裂让子节点更纯；树太深会记住训练集噪声，通过剪枝和深度限制控制复杂度，最后连接随机森林和 Boosting。
风格像规则树、样本点和剪枝工具结合，清晰、具体。
文字不是主体；标准术语保留英文，例如 if-else、Gini、Entropy、max_depth。其他说明可以用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch05-ensemble-bagging-boosting-flow.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "集成学习 Bagging Boosting 对比图",
        "suggested_page": "docs/ch05-machine-learning/ch02-supervised/04-ensemble-learning.md",
        "alt": "集成学习 Bagging Boosting 对比图：Bagging 并行训练多棵树降低方差，Boosting 串行纠错逐步提升模型。",
        "prompt": """
一张适合集成学习课程的对比图，主题是“并行投票和串行纠错是两条主线”。
画面左侧表现 Bagging：多棵树并行训练、投票或平均、降低方差；右侧表现 Boosting：模型按顺序训练，每一步关注上一步错的样本，逐步纠错。
风格像双路线技术白板，图形清楚、对比强。
文字不是主体；标准术语保留英文，例如 Bagging、Boosting、Random Forest、GBDT、XGBoost。其他说明可以用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch05-unsupervised-chapter-flow.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "无监督学习章节关系图",
        "suggested_page": "docs/ch05-machine-learning/ch03-unsupervised/00-roadmap.md",
        "alt": "无监督学习章节关系图：没有标签的数据通过聚类、降维和异常检测发现结构。",
        "prompt": """
一张适合无监督学习导读页的章节关系图，主题是“没有标签时，先发现结构”。
画面表现一堆未标注样本进入三条路径：聚类发现自然分组，降维把高维数据压缩成可视化平面，异常检测找出少数不寻常样本。
风格像数据探索地图，清晰、温和、有发现感。
文字不是主体；标准术语保留英文，例如 Clustering、PCA、t-SNE、UMAP、Anomaly。其他说明可以用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch05-clustering-decision-flow.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "聚类算法选择流程图",
        "suggested_page": "docs/ch05-machine-learning/ch03-unsupervised/01-clustering.md",
        "alt": "聚类算法选择流程图：先看目标和数据形状，再选择 K-Means、层次聚类或 DBSCAN，并用轮廓系数和业务解释评估。",
        "prompt": """
一张适合聚类课程的选择流程图，主题是“聚类不是自动真相，要看目标、形状和解释”。
画面表现未标注数据点进入判断流程：先问目标是分群还是探索，再看数据形状是否球形、链状或有噪声，再选择 K-Means、层次聚类或 DBSCAN，最后用轮廓系数和业务解释评估。
风格像数据侦探地图，清晰、具体。
文字不是主体；标准术语保留英文，例如 K-Means、DBSCAN、silhouette。其他说明可以用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch05-dimensionality-reduction-purpose-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "降维目的选择图",
        "suggested_page": "docs/ch05-machine-learning/ch03-unsupervised/02-dimensionality-reduction.md",
        "alt": "降维目的选择图：建模预处理优先 PCA，可视化探索再考虑 t-SNE 或 UMAP，并关注主成分数量和解释方式。",
        "prompt": """
一张适合降维课程的目的选择图，主题是“先问为什么降维，再选方法”。
画面表现高维数据进入两个方向：为了建模预处理时优先考虑 PCA，关注主成分数量和信息保留；为了可视化探索时考虑 t-SNE 或 UMAP，关注局部结构和解释方式。
风格像高维数据云被投影到二维平面，清晰、有空间感。
文字不是主体；标准术语和公式保留英文，例如 PCA、t-SNE、UMAP、variance。其他说明可以用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch05-anomaly-detection-decision-flow.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "异常检测决策流程图",
        "suggested_page": "docs/ch05-machine-learning/ch03-unsupervised/03-anomaly-detection.md",
        "alt": "异常检测决策流程图：先定义异常类型，再看数据维度、样本量和异常比例，选择统计方法、Isolation Forest、One-Class SVM 或 LOF。",
        "prompt": """
一张适合异常检测课程的决策流程图，主题是“先定义异常是什么，再选择方法和阈值”。
画面表现数据点中有少数离群样本，流程先判断异常是极端值、局部稀疏点还是边界外样本，再看维度、样本量和异常比例，最后选择统计方法、Isolation Forest、One-Class SVM 或 LOF，并思考误报漏报代价。
风格像风险雷达和数据点地图结合，清晰、实用。
文字不是主体；标准术语保留英文，例如 Isolation Forest、One-Class SVM、LOF。其他说明可以用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch05-evaluation-chapter-flow.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "模型评估章节关系图",
        "suggested_page": "docs/ch05-machine-learning/ch04-evaluation/00-roadmap.md",
        "alt": "模型评估章节关系图：指标、交叉验证、偏差方差和超参数调优串成判断模型是否可靠的闭环。",
        "prompt": """
一张适合模型评估导读页的章节关系图，主题是“模型到底好不好，不能只看一次分数”。
画面表现评估指标先回答看什么，交叉验证让分数更稳定，偏差方差诊断欠拟合或过拟合，超参数调优在正确验证流程内改进模型。
风格像模型体检路线图，清晰、专业。
文字不是主体；标准术语保留英文，例如 Accuracy、Recall、AUC、RMSE、K-fold。其他说明可以用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch05-metrics-selection-flow.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "评估指标选择流程图",
        "suggested_page": "docs/ch05-machine-learning/ch04-evaluation/01-metrics.md",
        "alt": "评估指标选择流程图：先看分类或回归，再看错误代价，选择 Accuracy、Recall、Precision、F1、AUC、RMSE 等指标。",
        "prompt": """
一张适合评估指标课程的选择流程图，主题是“指标不是训练结束后顺手看的分数，而是模型设计的一部分”。
画面表现先判断任务是分类还是回归，再判断错误代价是误报更贵还是漏报更贵，最后选择 Accuracy、Recall、Precision、F1、AUC、RMSE 等指标，并连接阈值选择和模型对比。
风格像决策树和仪表盘结合，清晰、实用。
文字不是主体；指标名称保留英文，例如 Accuracy、Recall、Precision、F1、AUC、RMSE。其他说明可以用少量中文短标签。不要出现整段英文说明、乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch05-cross-validation-stability-flow.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "交叉验证稳定评估流程图",
        "suggested_page": "docs/ch05-machine-learning/ch04-evaluation/02-cross-validation.md",
        "alt": "交叉验证稳定评估流程图：一次 train/test 划分不稳定，K 折重复评估，分类任务保持类别比例，特殊任务使用特殊切分。",
        "prompt": """
一张适合交叉验证课程的流程图，主题是“为什么一次随机划分不够可信”。
画面表现一次 train/test 划分像单次考试，K-fold 交叉验证像多轮轮流验证，分类任务需要 Stratified 保持类别比例，时间序列或分组数据需要特殊切法。
风格像数据切分条带和评估仪表盘结合，直观、清晰。
文字不是主体；标准术语保留英文，例如 train/test、K-fold、Stratified、Group、TimeSeries。其他说明可以用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch05-bias-variance-action-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "偏差方差行动诊断图",
        "suggested_page": "docs/ch05-machine-learning/ch04-evaluation/03-bias-variance.md",
        "alt": "偏差方差行动诊断图：模型效果不好时先分辨欠拟合还是过拟合，再决定加复杂度、加数据或加正则化。",
        "prompt": """
一张适合偏差方差课程的行动诊断图，主题是“模型效果不好时，不要乱试，要先诊断”。
画面表现模型效果差先进入诊断：训练分和验证分都差是欠拟合，可尝试加复杂度或更好特征；训练分高验证分差是过拟合，可尝试加数据、正则化、简化模型或交叉验证。
风格像医生诊断流程和学习曲线结合，清晰、实用。
文字不是主体；标准术语保留英文，例如 bias、variance、regularization。其他说明可以用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch05-hyperparameter-tuning-workflow.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "超参数调优验证流程图",
        "suggested_page": "docs/ch05-machine-learning/ch04-evaluation/04-hyperparameter-tuning.md",
        "alt": "超参数调优验证流程图：先有 baseline 和验证方式，再选搜索方法、比较参数组合，最后只用测试集做最终评估。",
        "prompt": """
一张适合超参数调优课程的工作流图，主题是“调参必须放在正确评估流程里”。
画面表现先建立 baseline，确定验证方式，再设计搜索空间，选择网格搜索、随机搜索或贝叶斯优化，比较参数组合，最后只在最终测试集上评估一次。
风格像实验控制台和搜索空间地图结合，强调不要用测试集反复调参。
文字不是主体；标准术语保留英文，例如 baseline、Grid Search、Random Search、Bayesian、validation、test set。其他说明可以用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch05-feature-engineering-chapter-flow.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "特征工程章节关系图",
        "suggested_page": "docs/ch05-machine-learning/ch05-feature-engineering/00-roadmap.md",
        "alt": "特征工程章节关系图：特征理解、预处理、构造、选择和 Pipeline 逐步把原始数据变成模型容易学习的输入。",
        "prompt": """
一张适合特征工程导读页的章节关系图，主题是“给模型看的数据，决定模型能学到什么”。
画面表现原始数据先经过特征理解，再进入预处理、特征构造、特征选择，最后用 Pipeline 固化成可复用、可验证、少泄漏的建模流程。
风格像数据加工工坊和流水线结合，清晰、工程感强。
文字不是主体；标准术语保留英文，例如 Pipeline、One-Hot、scaler。其他说明可以用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch05-feature-understanding-workflow.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "特征理解与泄漏检查图",
        "suggested_page": "docs/ch05-machine-learning/ch05-feature-engineering/01-feature-understanding.md",
        "alt": "特征理解与泄漏检查图：原始表格先识别特征类型、查看缺失分布、分析与目标关系、检查冗余和目标泄漏。",
        "prompt": """
一张适合特征理解课程的工作流图，主题是“先认识特征，再决定怎样处理”。
画面表现原始表格进入特征检查流程：识别数值、类别、时间、文本、ID 等特征类型，查看缺失和分布，分析与目标变量关系，检查冗余和目标泄漏，再决定预处理和构造策略。
风格像数据侦探桌面和检查清单结合，清晰、实用。
文字不是主体；标准术语保留英文，例如 ID、target、leakage、correlation。其他说明可以用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch05-projects-portfolio-loop.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "机器学习项目作品集闭环图",
        "suggested_page": "docs/ch05-machine-learning/ch06-projects/00-roadmap.md",
        "alt": "机器学习项目作品集闭环图：业务问题、任务定义、数据准备、baseline、评估指标、特征工程、错误分析、结论解释和作品集报告。",
        "prompt": """
一张适合机器学习项目实战导读页的作品集闭环图，主题是“项目不是跑完代码，而是可复盘、可解释、可交付”。
画面表现业务问题进入任务定义，经过数据准备、baseline、评估指标、特征工程和模型改进、错误分析、结论解释，最后输出 README、报告、图表和作品集材料。
风格像项目看板和实验记录台结合，专业但不死板。
文字不是主体；标准术语保留英文，例如 baseline、README、report、metric。其他说明可以用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ml-basics-roadmap.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "机器学习基础学习地图",
        "suggested_page": "docs/ch05-machine-learning/ch01-ml-basics/00-roadmap.md",
        "alt": "机器学习基础学习地图：问题定义、数据、模型、训练、评估和复盘组成入门主线。",
        "prompt": """
一张中文机器学习入门课程的学习地图插图，主题是“机器学习基础从问题到模型闭环”。
画面用清晰路线表达：定义问题、准备数据、划分训练测试、选择 baseline、训练模型、评估指标、错误分析和迭代改进。
风格像现代数据科学白板，温和、清晰、适合新人先建立全局地图。
可以有少量中文标签和箭头，但不要出现密集小字，不要乱码，不要真实品牌 logo。
""".strip(),
    },
    {
        "filename": "math-to-ml-bridge.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "数学到机器学习桥梁图",
        "suggested_page": "docs/ch05-machine-learning/ch01-ml-basics/03-math-to-ml-bridge.md",
        "alt": "数学到机器学习桥梁图：线性代数组织数据，概率统计定义不确定性，微积分指导参数更新。",
        "prompt": """
一张中文 AI 数学到机器学习的桥梁图，主题是“线性代数、概率统计、微积分如何流入模型训练”。
画面分成三条河流或三条管道：线性代数把样本和参数组织成矩阵，概率统计描述不确定性和损失，微积分给出梯度方向，三条线汇入一个机器学习训练循环。
构图要像新人友好的概念桥梁，抽象但容易懂，颜色区分三条线。
不要出现真实品牌 logo，不要复杂公式，不要密集小字或乱码。
""".strip(),
    },
    {
        "filename": "supervised-learning-roadmap.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "监督学习路线图",
        "suggested_page": "docs/ch05-machine-learning/ch02-supervised/00-roadmap.md",
        "alt": "监督学习路线图：带标签数据进入回归、分类、决策树和集成模型。",
        "prompt": """
一张中文机器学习课程路线图，主题是“监督学习：带答案的数据如何训练模型”。
画面表现带标签数据进入监督学习流水线，分成回归预测数值、分类判断类别、决策树做规则分裂、集成学习汇总多个模型。
风格清晰、现代、像课程白板和数据工作台结合，帮助新人理解本章学习顺序。
可以有少量中文标签，不要出现真实品牌 logo，不要生成密集小字或乱码。
""".strip(),
    },
    {
        "filename": "unsupervised-learning-roadmap.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "无监督学习路线图",
        "suggested_page": "docs/ch05-machine-learning/ch03-unsupervised/00-roadmap.md",
        "alt": "无监督学习路线图：没有标签的数据通过聚类、降维和异常检测发现结构。",
        "prompt": """
一张中文机器学习课程路线图，主题是“无监督学习：没有标签也能发现结构”。
画面表现一堆未标注数据点进入三条路径：聚类把相似样本分组，降维把高维数据压到二维图，异常检测找出离群点。
风格像数据探索地图，清晰、温和、帮助新人理解无监督学习不是预测答案而是发现结构。
不要出现真实品牌 logo，不要密集小字，不要乱码。
""".strip(),
    },
    {
        "filename": "anomaly-detection-outliers.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "异常检测离群点示意图",
        "suggested_page": "docs/ch05-machine-learning/ch03-unsupervised/03-anomaly-detection.md",
        "alt": "异常检测离群点示意图：正常样本聚成群，离群样本被统计方法、Isolation Forest 和 LOF 识别。",
        "prompt": """
一张中文异常检测课程教学图，主题是“从正常点群里找出不寻常样本”。
画面表现大多数蓝色数据点聚成正常区域，少数橙色或红色点远离人群；旁边用三种小卡片表达统计阈值、Isolation Forest 随机切分、LOF 局部密度异常。
风格像机器学习课堂图解，重点突出正常区域、边界、离群点和告警。
不要真实品牌 logo，不要复杂公式，不要密集小字或乱码。
""".strip(),
    },
    {
        "filename": "ml-evaluation-roadmap.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "模型评估学习地图",
        "suggested_page": "docs/ch05-machine-learning/ch04-evaluation/00-roadmap.md",
        "alt": "模型评估学习地图：指标、交叉验证、偏差方差和调参共同判断模型是否可靠。",
        "prompt": """
一张中文机器学习评估课程学习地图，主题是“不要只看一次分数，要判断模型是否可靠”。
画面表现评估指标、训练测试切分、K 折交叉验证、偏差方差诊断、超参数调优、最终报告形成一条质量检查路线。
风格像模型体检报告和数据科学白板结合，清晰、专业、新手友好。
不要出现真实品牌 logo，不要生成密集小字或乱码。
""".strip(),
    },
    {
        "filename": "cross-validation-kfold.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "K 折交叉验证切分图",
        "suggested_page": "docs/ch05-machine-learning/ch04-evaluation/02-cross-validation.md",
        "alt": "K 折交叉验证切分图：数据被分成多折，每一折轮流作为验证集，其余作为训练集。",
        "prompt": """
一张中文机器学习课程教学图，主题是“K 折交叉验证如何更稳定地估计模型效果”。
画面表现一条数据长条被切成 K 份，每一轮不同颜色的一份作为验证集，其余作为训练集，最后多个分数取平均。
构图要非常直观，像时间轴和表格结合，突出“轮流验证”和“平均更稳”。
不要出现真实品牌 logo，不要密集小字，不要乱码。
""".strip(),
    },
    {
        "filename": "bias-variance-tradeoff.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "偏差方差权衡三联图",
        "suggested_page": "docs/ch05-machine-learning/ch04-evaluation/03-bias-variance.md",
        "alt": "偏差方差权衡三联图：欠拟合、刚好和过拟合对应不同训练误差与泛化表现。",
        "prompt": """
一张中文机器学习课程教学图，主题是“偏差-方差权衡：欠拟合、刚好、过拟合”。
画面用三联图展示：左边模型太简单无法贴合趋势，中间模型刚好捕捉规律，右边模型过度弯曲记住噪声；下方可用靶心或训练/验证误差曲线辅助说明。
风格清晰、白板感、适合新人一眼理解泛化问题。
不要出现真实品牌 logo，不要复杂公式，不要密集小字或乱码。
""".strip(),
    },
    {
        "filename": "hyperparameter-tuning-search.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "超参数搜索方法对比图",
        "suggested_page": "docs/ch05-machine-learning/ch04-evaluation/04-hyperparameter-tuning.md",
        "alt": "超参数搜索方法对比图：网格搜索、随机搜索和贝叶斯优化用不同策略寻找更优参数。",
        "prompt": """
一张中文机器学习调参课程对比图，主题是“网格搜索、随机搜索、贝叶斯优化如何找好参数”。
画面分成三块：网格搜索像规则棋盘逐点尝试，随机搜索像随机撒点探索，贝叶斯优化像根据前面结果逐步靠近高分区域。
风格现代数据科学插图，重点突出搜索空间、候选点、最佳区域。
不要出现真实品牌 logo，不要复杂公式，不要密集小字或乱码。
""".strip(),
    },
    {
        "filename": "feature-engineering-roadmap.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "特征工程路线图",
        "suggested_page": "docs/ch05-machine-learning/ch05-feature-engineering/00-roadmap.md",
        "alt": "特征工程路线图：理解特征、预处理、构造、选择和 Pipeline 组成建模前处理主线。",
        "prompt": """
一张中文机器学习课程学习地图，主题是“特征工程：把原始数据变成模型更容易学习的输入”。
画面表现原始数据表进入五个环节：特征理解、缺失异常处理、缩放编码、构造新特征、筛选特征、进入 Pipeline 和模型。
风格像数据加工工坊，清晰、工程化、适合新人理解特征工程为什么重要。
不要真实品牌 logo，不要密集小字，不要乱码。
""".strip(),
    },
    {
        "filename": "feature-type-target-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "特征类型与目标关系图",
        "suggested_page": "docs/ch05-machine-learning/ch05-feature-engineering/01-feature-understanding.md",
        "alt": "特征类型与目标关系图：数值、类别、时间、文本等特征需要先看分布、关系和泄漏风险。",
        "prompt": """
一张中文特征理解课程教学图，主题是“先认识特征，再让模型学习”。
画面把一张数据表拆成不同类型特征：数值、类别、时间、文本、ID；再用箭头连接到分布分析、和目标变量关系、相关性冗余、目标泄漏检查。
风格像数据侦探工作台，清晰、可视化强，帮助新人知道探索特征该看什么。
不要真实品牌 logo，不要密集小字，不要乱码。
""".strip(),
    },
    {
        "filename": "feature-preprocessing-pipeline.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "特征预处理流水线图",
        "suggested_page": "docs/ch05-machine-learning/ch05-feature-engineering/02-preprocessing.md",
        "alt": "特征预处理流水线图：缺失值、异常值、数值缩放、类别编码和防泄漏 Pipeline 依次处理数据。",
        "prompt": """
一张中文机器学习特征预处理教学图，主题是“把脏乱特征整理成模型可用输入”。
画面表现原始数据表通过流水线依次处理：缺失值填补、异常值检查、数值标准化、类别编码、训练集参数只用于测试集 transform，最后进入模型。
风格工程化、清晰、像数据清洗流水线，突出防止数据泄漏。
不要真实品牌 logo，不要密集小字，不要乱码。
""".strip(),
    },
    {
        "filename": "feature-construction-workshop.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "特征构造工作坊图",
        "suggested_page": "docs/ch05-machine-learning/ch05-feature-engineering/03-feature-construction.md",
        "alt": "特征构造工作坊图：原始特征通过交互、时间、分组统计和领域知识生成更有用的新特征。",
        "prompt": """
一张中文机器学习特征构造课程插图，主题是“从原始字段加工出更有意义的新特征”。
画面像一个特征工坊：原始字段进入加工台，产出交互特征、时间特征、分组统计特征、领域知识特征；旁边有模型效果对比小图表示构造后更容易学习。
风格明亮、实战感强、新手友好。
不要真实品牌 logo，不要复杂公式，不要密集小字或乱码。
""".strip(),
    },
    {
        "filename": "feature-selection-methods.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "特征选择方法对比图",
        "suggested_page": "docs/ch05-machine-learning/ch05-feature-engineering/04-feature-selection.md",
        "alt": "特征选择方法对比图：过滤法、包裹法、嵌入法从不同角度保留有用特征。",
        "prompt": """
一张中文机器学习特征选择课程对比图，主题是“不是特征越多越好，要筛出真正有用的特征”。
画面分成三种方法：过滤法像筛网按统计关系初筛，包裹法像反复试模型组合，嵌入法像模型自己给出重要性；最后汇总到更简洁的特征集合。
风格清晰、工程白板感，帮助新人理解删特征也是建模能力。
不要真实品牌 logo，不要密集小字，不要乱码。
""".strip(),
    },
    {
        "filename": "column-transformer-pipeline.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "ColumnTransformer 与 Pipeline 工作流图",
        "suggested_page": "docs/ch05-machine-learning/ch05-feature-engineering/05-pipeline.md",
        "alt": "ColumnTransformer 与 Pipeline 工作流图：数值列和类别列分别处理后合并，再进入模型和调参流程。",
        "prompt": """
一张中文 Scikit-learn 工程课程结构图，主题是“ColumnTransformer + Pipeline 把预处理和模型串起来”。
画面表现数据表按列分流：数值列进入缩放器，类别列进入编码器，文本或其他列进入对应处理器，最后合并成特征矩阵进入模型；外层接 GridSearch 或交叉验证。
风格工程化、模块清楚、适合新人理解为什么真实项目要用 Pipeline。
不要真实品牌 logo，不要密集小字，不要乱码。
""".strip(),
    },
    {
        "filename": "ml-projects-roadmap.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "机器学习项目实战路线图",
        "suggested_page": "docs/ch05-machine-learning/ch06-projects/00-roadmap.md",
        "alt": "机器学习项目实战路线图：房价预测、客户流失、用户分群和 Kaggle 训练完整项目能力。",
        "prompt": """
一张中文机器学习项目实战路线图，主题是“从教程代码走向可交付项目”。
画面表现四个项目关卡：房价预测练回归，客户流失练分类，用户分群练聚类，Kaggle 练竞赛提交；中间贯穿数据探索、特征工程、模型对比、评估报告、作品集交付。
风格像课程闯关地图和数据科学工作流结合，专业但有趣。
不要真实品牌 logo，不要密集小字，不要乱码。
""".strip(),
    },
    {
        "filename": "house-price-project-flow.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "房价预测项目流程图",
        "suggested_page": "docs/ch05-machine-learning/ch06-projects/01-house-price.md",
        "alt": "房价预测项目流程图：房屋特征经过探索、特征工程、回归建模和误差分析得到价格预测。",
        "prompt": """
一张中文机器学习回归项目流程图，主题是“房价预测从数据到模型报告”。
画面表现房屋数据表包含面积、房间数、位置、楼层等特征，经过 EDA、缺失处理、特征工程、回归模型对比、误差分析，最后输出预测价格和项目报告。
风格像实战项目看板，清晰、新手友好、有作品集感。
不要真实品牌 logo，不要密集小字，不要乱码。
""".strip(),
    },
    {
        "filename": "customer-churn-project-flow.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "客户流失预测项目流程图",
        "suggested_page": "docs/ch05-machine-learning/ch06-projects/02-customer-churn.md",
        "alt": "客户流失预测项目流程图：用户行为数据经过分类建模、阈值选择和业务动作转成流失预警。",
        "prompt": """
一张中文机器学习分类项目流程图，主题是“客户流失预测从行为数据到预警名单”。
画面表现用户使用时长、消费频次、投诉记录、最近登录等特征进入分类模型，输出流失概率；再通过阈值选择、混淆矩阵、业务干预策略形成闭环。
风格清晰、业务感强、适合新人理解分类项目不只是准确率。
不要真实品牌 logo，不要密集小字，不要乱码。
""".strip(),
    },
    {
        "filename": "user-segmentation-rfm.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "用户分群 RFM 聚类图",
        "suggested_page": "docs/ch05-machine-learning/ch06-projects/03-user-segmentation.md",
        "alt": "用户分群 RFM 聚类图：Recency、Frequency、Monetary 特征经过标准化和聚类形成可解释用户群体。",
        "prompt": """
一张中文机器学习聚类项目教学图，主题是“RFM 用户分群：把用户分成可行动的人群”。
画面表现 Recency、Frequency、Monetary 三个用户特征进入标准化和 K-Means 聚类，得到高价值客户、沉睡客户、潜力客户等群体，并用雷达图或二维散点解释群体特征。
风格像数据分析项目展示页，清晰、业务可解释、新手友好。
不要真实品牌 logo，不要密集小字，不要乱码。
""".strip(),
    },
    {
        "filename": "kaggle-submission-loop.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Kaggle 竞赛提交闭环图",
        "suggested_page": "docs/ch05-machine-learning/ch06-projects/04-kaggle.md",
        "alt": "Kaggle 竞赛提交闭环图：读取数据、建立 baseline、交叉验证、生成提交文件和复盘榜单形成训练闭环。",
        "prompt": """
一张中文机器学习竞赛训练图，主题是“Kaggle 入门竞赛的提交闭环”。
画面表现下载训练集和测试集、建立 baseline、特征工程、交叉验证、本地分数、生成 submission.csv、提交榜单、复盘误差并迭代。
风格像竞赛工作台和项目看板结合，专业、清晰、让新人知道不要盲目刷榜。
不要出现真实品牌 logo，不要密集小字，不要乱码。
""".strip(),
    },
    {
        "filename": "ml-study-loop.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "机器学习学习循环图",
        "suggested_page": "docs/ch05-machine-learning/study-guide.md",
        "alt": "机器学习学习循环图：先跑 baseline，再解释结果、做错误分析、改特征和复盘。",
        "prompt": """
一张中文机器学习学习指南插图，主题是“机器学习不要只背算法，要形成学习循环”。
画面表现学习者围绕一个循环：先理解问题，跑最小 baseline，看指标，做错误分析，改特征或模型，写复盘，再进入下一个实验。
风格温暖、鼓励、像学习陪伴型课程图，让新人觉得可以一步步学会建模。
不要真实品牌 logo，不要密集小字，不要乱码。
""".strip(),
    },
    {
        "filename": "ml-task-checklist.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "机器学习通关任务清单图",
        "suggested_page": "docs/ch05-machine-learning/task-list.md",
        "alt": "机器学习通关任务清单图：回归、分类、聚类、评估和特征工程任务组成阶段通关作品。",
        "prompt": """
一张中文机器学习阶段任务清单插图，主题是“完成这些小任务，就真正入门机器学习”。
画面用任务卡片展示：训练一个回归模型、训练一个分类模型、做一次聚类分析、画混淆矩阵和学习曲线、搭建 Pipeline、完成一个项目报告。
风格像课程闯关任务板，现代、清晰、有成就感，适合放在任务单页面。
不要真实品牌 logo，不要密集小字，不要乱码。
""".strip(),
    },
    {
        "filename": "ch06-deep-learning.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "深度学习与 Transformer 主视觉",
        "suggested_page": "docs/ch06-deep-learning/index.md",
        "alt": "深度学习与 Transformer 主视觉：张量、训练循环、CNN、RNN、Attention 和 Transformer 串成学习路径。",
        "prompt": """
一张适合深度学习课程的阶段主视觉，主题是“深度学习与 Transformer 基础”。
画面表现张量数据进入神经网络训练循环，再连接到 CNN、RNN、Attention 和 Transformer 模块，像打开模型发动机舱。
风格现代科技感但不过度炫目，适合课程首页，不要生成具体文字，不要出现真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch06-learning-quest-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "深度学习学习闯关地图",
        "suggested_page": "docs/ch06-deep-learning/index.md",
        "alt": "深度学习学习闯关地图：张量、前向传播、损失、反向传播、优化器、CNN、序列模型、Attention 和 Transformer 逐步连接。",
        "prompt": """
一张适合深度学习课程首页的学习闯关地图，主题是“从训练循环走向 Transformer”。
画面表现张量和数据加载进入前向传播，计算 loss，反向传播得到 gradient，optimizer 更新参数，再进入 CNN、RNN、Attention 和 Transformer 结构。
风格视觉优先、流程清晰、新手友好，像一张模型发动机舱路线图。
文字不是主体；如需标签，中英文自然混用，公式、API、变量名和标准术语保留英文或数学形式，例如 Tensor、loss、gradient、optimizer、CNN、RNN、Attention、Transformer。不要整段英文说明、无意义英文、乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch06-training-loop-backbone.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "深度学习训练闭环主线图",
        "suggested_page": "docs/ch06-deep-learning/index.md",
        "alt": "深度学习训练闭环主线图：神经网络基础、PyTorch 训练闭环、CNN、RNN、Attention、Transformer 和大模型基础逐步连接。",
        "prompt": """
一张适合深度学习课程首页的主线图，主题是“深度学习为什么是后续大模型的发动机”。
画面从神经网络基础出发，经过 PyTorch 训练闭环、CNN 图像任务、RNN 序列任务、Attention，再进入 Transformer 和大模型基础。
风格像课程技术栈演进图，层次清楚、结构感强。
文字不是主体；如需标签，中英文自然混用，标准术语保留英文，例如 PyTorch、CNN、RNN、Attention、Transformer、LLM。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch06-study-guide-training-loop.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "深度学习学习指南训练闭环",
        "suggested_page": "docs/ch06-deep-learning/study-guide.md",
        "alt": "深度学习学习指南训练闭环：数据、模型前向传播、损失、反向传播、优化器更新、评估和调参组成学习主线。",
        "prompt": """
一张适合深度学习学习指南的训练闭环图，主题是“第一遍深度学习只抓训练循环”。
画面表现数据进入模型，forward 得到输出，loss 衡量差距，backward 计算梯度，optimizer step 更新参数，然后进入评估、调参和下一轮实验。
风格温暖、清晰、有陪伴感，帮助新人不要被长代码和模型名吓住。
文字不是主体；如需标签，中英文自然混用，API 和标准术语保留英文，例如 forward、loss、backward、optimizer.step、eval。不要整段英文说明、乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch06-nn-basics-chapter-flow.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "神经网络基础章节关系图",
        "suggested_page": "docs/ch06-deep-learning/ch01-nn-basics/00-roadmap.md",
        "alt": "神经网络基础章节关系图：神经元、激活函数、前向反向传播、优化器、正则化和初始化逐步连接。",
        "prompt": """
一张适合神经网络基础导读页的章节关系图，主题是“神经网络为什么能学起来”。
画面表现第 5 章的模型、损失、评估进入第 6 章内部机制：神经元和激活函数、前向传播、反向传播、优化器、正则化、参数初始化。
风格像把模型外壳打开，看见内部零件如何协作，清晰、具体。
文字不是主体；如需标签，中英文自然混用，标准术语和公式保留英文或数学形式，例如 activation、forward、backward、optimizer、regularization、initialization。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch06-ml-to-dl-bridge-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "从机器学习到深度学习桥接图",
        "suggested_page": "docs/ch06-deep-learning/ch01-nn-basics/00-ml-to-dl-bridge.md",
        "alt": "从机器学习到深度学习桥接图：经典机器学习的任务、baseline 和评估，过渡到深度学习的自动表示学习和训练闭环。",
        "prompt": """
一张适合“从经典机器学习到深度学习”过渡页的桥接图，主题是“第 6 章不是推翻第 5 章，而是打开训练内部结构”。
画面左侧是经典 ML：任务定义、baseline、metric、feature engineering；右侧是深度学习：representation learning、layers、loss、gradient、training loop，中间用桥梁连接。
风格像两座学习岛之间的桥，帮助新人自然过渡。
文字不是主体；如需标签，中英文自然混用，标准术语保留英文，例如 baseline、metric、feature engineering、representation、loss、gradient。不要整段英文说明、乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch06-weight-init-signal-stability-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "权重初始化信号稳定图",
        "suggested_page": "docs/ch06-deep-learning/ch01-nn-basics/05-weight-init.md",
        "alt": "权重初始化信号稳定图：初始化太小导致信号衰减，初始化太大导致梯度爆炸或激活饱和，合适初始化让前向信号和反向梯度稳定。",
        "prompt": """
一张适合权重初始化课程的概念图，主题是“训练开始前，第一步棋要摆稳”。
画面分三条路径：初始化太小导致信号逐层衰减；初始化太大导致激活饱和或梯度爆炸；合适初始化让 forward signal 和 backward gradient 稳定流过多层网络。
风格像神经网络信号管道和仪表盘结合，直观、具体。
文字不是主体；如需标签，中英文自然混用，公式和标准术语保留英文，例如 Xavier、He/Kaiming、vanishing gradient、exploding gradient。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch06-pytorch-chapter-flow.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "PyTorch 章节关系图",
        "suggested_page": "docs/ch06-deep-learning/ch02-pytorch/00-roadmap.md",
        "alt": "PyTorch 章节关系图：Tensor、Autograd、nn.Module、DataLoader、Training Loop 和实践技巧组成最小工程闭环。",
        "prompt": """
一张适合 PyTorch 导读页的章节关系图，主题是“把能学习的模型写成可训练代码”。
画面表现 Tensor 作为数据容器，Autograd 计算梯度，nn.Module 组织模型，DataLoader 批量喂数据，Training Loop 串起 forward、loss、backward、step，最后进入实践技巧。
风格像深度学习工程流水线，简洁、清楚。
文字不是主体；API、类名和标准术语保留英文，例如 Tensor、Autograd、nn.Module、DataLoader、Training Loop、forward、backward。其他说明可用少量中文短标签。不要整段英文说明、乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch06-sklearn-to-pytorch-shift-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "sklearn 到 PyTorch 换挡图",
        "suggested_page": "docs/ch06-deep-learning/ch02-pytorch/00-sklearn-to-pytorch-bridge.md",
        "alt": "sklearn 到 PyTorch 换挡图：sklearn 的 fit 和 predict 被拆成 Tensor、模型、loss、backward、optimizer step 和评估。",
        "prompt": """
一张适合“从 sklearn 到 PyTorch”过渡页的对照图，主题是“从自动挡 fit 到手动挡训练循环”。
画面左侧是 sklearn：X/y、model.fit、predict；右侧是 PyTorch：Tensor、model、loss_fn、optimizer、forward、loss、backward、step，强调细节被拆开但控制力更强。
风格像驾驶换挡类比和训练流水线结合，新手友好。
文字不是主体；API 和变量名保留英文，例如 sklearn.fit、predict、Tensor、loss_fn、optimizer.step。其他说明可用少量中文短标签。不要整段英文说明、乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch06-pytorch-tensor-lifecycle-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "PyTorch Tensor 生命周期图",
        "suggested_page": "docs/ch06-deep-learning/ch02-pytorch/01-pytorch-basics.md",
        "alt": "PyTorch Tensor 生命周期图：真实数据变成 Tensor，检查 shape 和 dtype，完成索引、变形和运算，再送进模型。",
        "prompt": """
一张适合 PyTorch 基础页的张量生命周期图，主题是“数据进入深度学习前，要先成为 Tensor”。
画面表现真实数据先转换成 Tensor，检查 shape、dtype 和 device，再做 indexing、reshape、broadcasting 和矩阵运算，最后进入模型。
风格像数据加工台和张量立方体结合，清晰、实用。
文字不是主体；API、变量和标准术语保留英文，例如 Tensor、shape、dtype、device、reshape、broadcasting、GPU。其他说明可用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch06-nn-module-parameter-flow.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "nn.Module 参数组织流程图",
        "suggested_page": "docs/ch06-deep-learning/ch02-pytorch/03-nn-module.md",
        "alt": "nn.Module 参数组织流程图：Tensor 进入 Layer，Layer 被 Module 组织，forward 定义数据流，parameters 交给优化器更新。",
        "prompt": """
一张适合 nn.Module 课程的模型组织图，主题是“nn.Module 是可训练模型的容器”。
画面表现 Tensor 输入到 Linear/ReLU 等 layer，多个 layer 被 nn.Module 或 nn.Sequential 组织起来，forward 定义数据流，parameters 交给 optimizer 更新。
风格像乐高模型盒和训练管线结合，帮助新人理解模型对象。
文字不是主体；API 和标准术语保留英文，例如 nn.Module、Linear、ReLU、forward、parameters、train、eval。其他说明可用少量中文短标签。不要整段英文说明、乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch06-cnn-chapter-flow.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "CNN 章节关系图",
        "suggested_page": "docs/ch06-deep-learning/ch03-cnn/00-roadmap.md",
        "alt": "CNN 章节关系图：图像空间结构、卷积、CNN 结构、经典架构、迁移学习和图像分类项目逐步连接。",
        "prompt": """
一张适合 CNN 导读页的章节关系图，主题是“图片不是普通表格，网络结构要跟着数据变”。
画面表现 MLP 擅长固定向量，但图像有空间结构；CNN 通过 local connection、parameter sharing、receptive field 看图，再进入卷积基础、CNN 结构、经典架构、迁移学习和图像分类项目。
风格像图像网格、卷积核滑动和模型演进路线结合，直观、清晰。
文字不是主体；标准术语保留英文，例如 MLP、CNN、kernel、feature map、ResNet、transfer learning。其他说明可用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch06-rnn-chapter-flow.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "RNN 序列模型章节关系图",
        "suggested_page": "docs/ch06-deep-learning/ch04-rnn/00-roadmap.md",
        "alt": "RNN 序列模型章节关系图：序列顺序、隐藏状态、RNN、LSTM/GRU 和序列建模实战逐步连接。",
        "prompt": """
一张适合 RNN 与序列模型导读页的章节关系图，主题是“顺序本身就是信息”。
画面表现静态输入进入序列输入，时间步逐个展开，hidden state 边读边记，再连接到 RNN、LSTM/GRU、序列建模实战，并为 Attention 铺路。
风格像时间轴、记忆流和模型结构结合，温和、清晰。
文字不是主体；标准术语保留英文，例如 sequence、time step、hidden state、RNN、LSTM、GRU。其他说明可用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch06-transformer-chapter-flow.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Transformer 章节关系图",
        "suggested_page": "docs/ch06-deep-learning/ch05-transformer/00-roadmap.md",
        "alt": "Transformer 章节关系图：RNN 痛点、Attention、Q/K/V、Self-Attention、Transformer 模块、BERT/GPT 和大模型逐步连接。",
        "prompt": """
一张适合 Transformer 导读页的章节关系图，主题是“序列建模从顺序传递走向全局关联”。
画面表现 RNN 在长距离依赖和并行训练上遇到瓶颈，Attention 让每个位置直接关注相关位置，Q/K/V 拆解角色，Self-Attention 汇聚上下文，最后堆叠成 Transformer 并连接 BERT、GPT 和 LLM。
风格像序列 token 星图和模块架构图结合，清楚、有历史过渡感。
文字不是主体；标准术语保留英文，例如 RNN、Attention、Q/K/V、Self-Attention、Transformer、BERT、GPT、LLM。其他说明可用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch06-transformer-global-context-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Transformer 全局上下文建模图",
        "suggested_page": "docs/ch06-deep-learning/ch05-transformer/00-roadmap.md",
        "alt": "Transformer 全局上下文建模图：输入序列经过词向量和位置编码，Self-Attention 建立位置关系，前馈网络、残差和归一化稳定训练，堆叠成 Transformer。",
        "prompt": """
一张适合 Transformer 学习主线的全局上下文图，主题是“每个 token 都能看向相关 token”。
画面表现输入序列变成 embedding 和 positional encoding，进入 Self-Attention 建立全局关系，再经过 feed-forward、residual、LayerNorm，多个 block 堆叠形成 Transformer，支撑预训练语言模型。
风格像 token 网络和深度模块堆叠结合，强调关系、上下文和稳定训练。
文字不是主体；标准术语和变量保留英文，例如 token、embedding、positional encoding、Self-Attention、FFN、residual、LayerNorm。其他说明可用少量中文短标签。不要整段英文说明、乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch06-generative-chapter-flow.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "生成模型章节关系图",
        "suggested_page": "docs/ch06-deep-learning/ch06-generative/00-roadmap.md",
        "alt": "生成模型章节关系图：从分类预测转向生成新样本，GAN 对抗式生成和 VAE 潜空间生成组成两条经典路线。",
        "prompt": """
一张适合生成模型导读页的章节关系图，主题是“从判对走向生成得像”。
画面表现前面模型主要做分类、回归和表示学习，随后转向学习数据分布并生成新样本；分成两条路线：VAE 的 latent space、sampling、decoder，GAN 的 generator 和 discriminator 对抗训练。
风格像创作工作室和模型实验台结合，清晰、有探索感。
文字不是主体；标准术语保留英文，例如 VAE、latent space、sampling、GAN、generator、discriminator。其他说明可用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch06-training-tips-chapter-flow.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "深度学习训练技巧章节关系图",
        "suggested_page": "docs/ch06-deep-learning/ch07-training-tips/00-roadmap.md",
        "alt": "深度学习训练技巧章节关系图：模型能跑起来后，通过超参数调优、训练监控诊断和模型压缩走向稳定训练与落地。",
        "prompt": """
一张适合训练技巧导读页的章节关系图，主题是“从能训练走到会排障、会迭代、会落地”。
画面表现模型已经能跑起来后，进入 hyperparameter tuning、training monitoring、diagnosis、model compression 和部署准备，像一条深度学习工程排障路线。
风格像训练控制台、曲线监控和工具箱结合，实用、清晰。
文字不是主体；标准术语保留英文，例如 learning rate、batch size、loss curve、overfitting、compression、quantization。其他说明可用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch06-projects-portfolio-loop.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "深度学习项目作品集路线图",
        "suggested_page": "docs/ch06-deep-learning/ch08-projects/00-roadmap.md",
        "alt": "深度学习项目作品集路线图：图像分类、文本情感分析和生成模型项目连接到训练曲线、错误分析和作品集输出。",
        "prompt": """
一张适合深度学习项目导读页的作品集路线图，主题是“项目不是代码跑完，而是训练过程可解释”。
画面表现神经网络、PyTorch、CNN、RNN、Transformer 和生成模型汇入三个项目：图像分类、文本情感分析、生成模型实战；最后输出训练曲线、错误样例、模型报告和作品集材料。
风格像项目看板和实验记录台结合，专业但新手友好。
文字不是主体；标准术语保留英文，例如 PyTorch、CNN、Transformer、loss curve、report、portfolio。其他说明可用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch06-deep-learning-project-cycle.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "深度学习项目训练复盘闭环图",
        "suggested_page": "docs/ch06-deep-learning/ch08-projects/00-roadmap.md",
        "alt": "深度学习项目训练复盘闭环图：任务定义、数据集准备、DataLoader、模型结构、训练循环、验证评估、调整数据模型超参数、保存模型和作品集输出。",
        "prompt": """
一张适合深度学习项目章的训练复盘闭环图，主题是“数据、模型、训练、验证和错误分析循环”。
画面表现任务定义进入数据集准备和 DataLoader，模型结构进入 training loop，验证评估后根据表现调整数据、模型和超参数；如果表现足够好，就保存模型、展示样例、输出 README 和作品集。
风格像实验循环和发布流程结合，帮助新人形成可复现项目意识。
文字不是主体；标准术语保留英文，例如 DataLoader、training loop、validation、checkpoint、README。其他说明可用少量中文短标签。不要整段英文说明、乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch10-computer-vision.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "计算机视觉主视觉",
        "suggested_page": "docs/ch10-computer-vision/index.md",
        "alt": "计算机视觉主视觉：图像分类、目标检测、分割、OCR 和医学影像组成视觉任务地图。",
        "prompt": """
一张适合计算机视觉课程的阶段主视觉，主题是“计算机视觉”。
画面表现同一张图片被模型逐步理解：分类标签、目标框、分割 mask、OCR 文本框和医学影像热力图，突出从粗到细的视觉任务层级。
风格清晰、工程感、适合新手课程，不要生成具体文字，不要出现真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch11-nlp.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "自然语言处理主视觉",
        "suggested_page": "docs/ch11-nlp/index.md",
        "alt": "自然语言处理主视觉：文本经过分词、词向量、分类、序列标注和预训练模型处理。",
        "prompt": """
一张适合自然语言处理课程的阶段主视觉，主题是“NLP 自然语言处理”。
画面表现一段文本被分词、转成向量、进行情感分类、实体标注、摘要和问答，最后连接到 BERT/GPT 风格的预训练模型抽象模块。
风格温和、清晰、有学习地图感，不要生成具体文字，不要出现真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch07-llm-principles.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "大模型原理主视觉",
        "suggested_page": "docs/ch07-llm-principles/index.md",
        "alt": "大模型原理主视觉：Token、Embedding、Transformer、预训练、Prompt、微调和对齐组成能力链路。",
        "prompt": """
一张适合大模型课程的阶段主视觉，主题是“大模型原理、Prompt 与微调”。
画面表现文本被切成 Token，进入 Embedding 和 Transformer 层，经过预训练、Prompt、微调和对齐，形成可控的大模型能力。
风格现代、抽象但易懂，不要生成具体文字，不要出现真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch07-learning-quest-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "大模型学习闯关地图",
        "suggested_page": "docs/ch07-llm-principles/index.md",
        "alt": "大模型学习闯关地图：Token、Embedding、Attention、Transformer、预训练、Prompt、微调和对齐逐步连接。",
        "prompt": """
一张适合大模型课程首页的学习闯关地图，主题是“从文本进入可控的大模型能力”。
画面表现文本先切成 Token，进入 Embedding 和 Attention，再进入 Transformer、预训练、Prompt、微调、对齐与安全，最后形成可控的大模型应用能力。
风格视觉优先、路线清晰、新手友好，像把聊天机器人的魔法盒拆开。
文字不是主体；如需标签，中英文自然混用，标准术语保留英文，例如 Token、Embedding、Attention、Transformer、Pretraining、Prompt、Fine-tuning、RLHF。不要整段英文说明、无意义英文、乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch07-llm-capability-backbone.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "大模型能力来源主线图",
        "suggested_page": "docs/ch07-llm-principles/index.md",
        "alt": "大模型能力来源主线图：NLP 基础、Token 与 Embedding、Transformer、大规模预训练、Prompt、PEFT、RLHF 和可控应用逐步连接。",
        "prompt": """
一张适合大模型课程首页的能力来源主线图，主题是“大模型不是魔法，而是结构、数据、训练和对齐共同作用”。
画面从 NLP 基础和 Token/Embedding 出发，连接 Transformer、大规模预训练、Prompt Engineering、PEFT/LoRA、RLHF 与对齐，最终形成可控的大模型能力。
风格像技术栈演进图和能力管线结合，清晰、有层次。
文字不是主体；如需标签，中英文自然混用，标准术语保留英文，例如 NLP、Token、Embedding、Transformer、Prompt、PEFT、LoRA、RLHF。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch07-study-guide-evolution-line.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "大模型学习指南演进线",
        "suggested_page": "docs/ch07-llm-principles/study-guide.md",
        "alt": "大模型学习指南演进线：token、embedding、Transformer、预训练、Prompt、微调、对齐和可控应用组成第一遍学习主线。",
        "prompt": """
一张适合大模型学习指南的演进线插图，主题是“第一遍大模型只抓能力从哪里来到如何被控制”。
画面表现 token 变成 embedding，Transformer 建模上下文，预训练获得通用能力，Prompt、微调和对齐让能力更可用，最后进入可控应用。
风格温暖、清晰、有陪伴感，帮助新人不要被模型榜单和术语吓住。
文字不是主体；如需标签，中英文自然混用，标准术语保留英文，例如 token、embedding、Transformer、Prompt、fine-tuning、alignment。不要整段英文说明、乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch07-nlp-crash-chapter-flow.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "NLP 速成章节关系图",
        "suggested_page": "docs/ch07-llm-principles/ch01-nlp-crash/00-roadmap.md",
        "alt": "NLP 速成章节关系图：Tokenizer、Embedding、预训练模型速览和 Hugging Face 快速上手组成大模型文本底座。",
        "prompt": """
一张适合 NLP 核心速成导读页的章节关系图，主题是“进入大模型前，先压实最小文本底座”。
画面表现原始文本进入 Tokenizer，被切成 token 和 id，再进入 Embedding 变成向量，连接预训练模型速览和 Hugging Face 快速上手。
风格像文本处理流水线和学习路线图结合，清晰、实用。
文字不是主体；标准术语和库名保留英文，例如 Tokenizer、Embedding、token id、pretrained model、Hugging Face。其他说明可用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch07-llm-overview-chapter-flow.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "LLM 概览章节关系图",
        "suggested_page": "docs/ch07-llm-principles/ch02-llm-overview/00-roadmap.md",
        "alt": "LLM 概览章节关系图：NLP 基础、Token、Embedding、Transformer、预训练语言模型、大语言模型、Prompt、微调、RAG 和 Agent 逐步连接。",
        "prompt": """
一张适合 LLM 概览导读页的章节关系图，主题是“先建立看大模型的坐标系”。
画面表现 NLP 基础、Token/Embedding、Transformer、预训练语言模型逐步发展到 LLM，再连接 Prompt、微调、RAG 和 Agent 应用路线。
风格像大模型时代地图，层次清楚，避免榜单式堆模型名。
文字不是主体；标准术语保留英文，例如 NLP、Token、Embedding、Transformer、LLM、Prompt、RAG、Agent。其他说明可用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch07-llm-capability-stack.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "大模型能力栈与应用生态图",
        "suggested_page": "docs/ch07-llm-principles/ch02-llm-overview/00-roadmap.md",
        "alt": "大模型能力栈与应用生态图：大规模数据、Transformer、预训练、指令微调、对齐、API、RAG、Agent 和多模态组成能力底座。",
        "prompt": """
一张适合 LLM 概览页的能力栈图，主题是“大模型能力由底座和应用系统共同组成”。
画面从大规模数据和 Transformer 架构开始，经过预训练、指令微调与对齐，形成对话和生成能力，再通过 API、RAG、Agent、多模态和监控进入应用生态。
风格像分层技术栈和生态地图结合，清楚、专业。
文字不是主体；标准术语保留英文，例如 data scale、Transformer、pretraining、SFT、alignment、API、RAG、Agent、multimodal。其他说明可用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch07-transformer-deep-chapter-flow.png",
        "size": "1024x1024",
        "quality": "medium",
        "title": "Transformer 深入章节关系图",
        "suggested_page": "docs/ch07-llm-principles/ch03-transformer-deep/00-roadmap.md",
        "alt": "Transformer 深入章节关系图：Attention、Transformer 架构、架构变体、高效注意力、规模计算、预训练、微调和部署逐步连接。",
        "prompt": """
一张适合 Transformer 深入导读页的简洁章节关系图，主题是“从 Attention 到大模型工程”。
画面用一条清楚的学习路径串起 Attention、Transformer block、架构变体、计算成本、预训练、微调和部署。
风格像高质量课程插画，中心有简化的 Transformer block，周围用箭头连接关键概念，层级少、留白充足。
文字不是主体；标准术语保留英文，例如 Attention、Transformer、KV Cache、pretraining、fine-tuning、deployment。其他说明只用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch07-transformer-cost-task-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Transformer 信息流计算成本任务适配图",
        "suggested_page": "docs/ch07-llm-principles/ch03-transformer-deep/00-roadmap.md",
        "alt": "Transformer 信息流计算成本任务适配图：结构组件分别影响信息流、计算成本和任务适配，连接上下文长度、显存吞吐和 BERT/GPT/T5。",
        "prompt": """
一张适合 Transformer 深入页的三轴概念图，主题是“结构、计算和任务适配要一起看”。
画面中心是 Transformer block，向外分出三条线：information flow 解释 token 如何互相关注；compute cost 连接 context length、memory、throughput；task fit 连接 BERT、GPT、T5 和不同任务。
风格像技术白板和系统架构图结合，帮助新人建立工程直觉。
文字不是主体；标准术语保留英文，例如 information flow、compute cost、context length、memory、throughput、BERT、GPT、T5。其他说明可用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch07-pretraining-chapter-flow.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "预训练技术章节关系图",
        "suggested_page": "docs/ch07-llm-principles/ch04-pretraining/00-roadmap.md",
        "alt": "预训练技术章节关系图：Transformer 结构、预训练数据、预训练目标、训练工程、Prompt、微调和 RAG 逐步连接。",
        "prompt": """
一张适合预训练技术导读页的章节关系图，主题是“模型通用能力从数据、目标和工程中来”。
画面表现 Transformer 结构连接预训练数据、预训练目标和训练工程，训练出通用能力，再进入 Prompt、微调、RAG 和 Agent 应用。
风格像大规模训练工厂和能力管线结合，清晰、专业。
文字不是主体；标准术语保留英文，例如 Transformer、pretraining data、training objective、training engineering、Prompt、fine-tuning、RAG。其他说明可用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch07-pretraining-data-objective-engineering-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "预训练数据目标工程三角图",
        "suggested_page": "docs/ch07-llm-principles/ch04-pretraining/00-roadmap.md",
        "alt": "预训练数据目标工程三角图：数据来源、清洗去重、质量安全、语言建模、掩码预测、多任务目标、分布式训练、混合精度和监控共同决定模型能力。",
        "prompt": """
一张适合预训练技术页的三角关系图，主题是“数据决定上限，目标决定学习方式，工程决定能不能训完”。
画面分成三块：data 包含来源、清洗、去重、质量与安全；objective 包含 language modeling、masked prediction、多任务目标；engineering 包含 distributed training、mixed precision、checkpoint、monitoring。
风格像训练控制中心和数据治理图结合，清晰、实用。
文字不是主体；标准术语保留英文，例如 data quality、dedup、language modeling、masked prediction、distributed training、checkpoint。其他说明可用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch07-prompt-chapter-flow.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Prompt 工程章节关系图",
        "suggested_page": "docs/ch07-llm-principles/ch05-prompt/00-roadmap.md",
        "alt": "Prompt 工程章节关系图：大模型能力、任务目标、Prompt 组织、模型输出、结构化结果和应用功能逐步连接。",
        "prompt": """
一张适合 Prompt 工程导读页的章节关系图，主题是“Prompt 是应用层和模型层之间的接口设计”。
画面表现大模型能力进入任务目标、上下文组织、约束条件、输出格式、模型调用、结构化结果和应用功能。
风格像产品接口设计图和模型调用流程结合，清爽、实用。
文字不是主体；标准术语保留英文，例如 Prompt、context、constraints、JSON、Markdown、model call、structured output。其他说明可用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch07-prompt-iteration-loop.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Prompt 迭代测试闭环图",
        "suggested_page": "docs/ch07-llm-principles/ch05-prompt/00-roadmap.md",
        "alt": "Prompt 迭代测试闭环图：用户需求、任务定义、输入材料、约束条件、输出格式、示例检查、模型调用、结果验证和 Prompt 迭代组成可复用调用。",
        "prompt": """
一张适合 Prompt 工程页的迭代闭环图，主题是“Prompt 不是问一句，而是设计一次可复用模型调用”。
画面表现用户需求进入任务定义、输入材料、约束条件、输出格式、示例与检查，调用模型后做结果验证，再根据失败样例迭代 Prompt。
风格像实验记录和产品测试流程结合，强调可测试、可解析、可维护。
文字不是主体；标准术语保留英文，例如 Prompt version、test cases、structured output、JSON、validation、iteration。其他说明可用少量中文短标签。不要整段英文说明、乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch07-finetuning-chapter-flow.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "大模型微调章节关系图",
        "suggested_page": "docs/ch07-llm-principles/ch06-finetuning/00-roadmap.md",
        "alt": "大模型微调章节关系图：预训练、通用语言能力、Prompt、微调、数据调整行为、领域任务和稳定格式逐步连接。",
        "prompt": """
一张适合大模型微调导读页的章节关系图，主题是“微调不是万能按钮，而是用数据塑造行为”。
画面表现预训练获得通用语言能力，Prompt 不改参数地调用能力，微调用任务数据调整行为，最后服务领域任务、稳定格式和固定任务模式。
风格像模型行为调校台，清晰、克制、帮助新人判断边界。
文字不是主体；标准术语保留英文，例如 pretraining、Prompt、fine-tuning、behavior、domain task、format stability。其他说明可用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch07-finetuning-decision-loop.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "微调决策与评估闭环图",
        "suggested_page": "docs/ch07-llm-principles/ch06-finetuning/00-roadmap.md",
        "alt": "微调决策与评估闭环图：判断是否需要微调、定义任务和评估标准、准备样本、选择 LoRA/QLoRA/PEFT、训练验证、对比 Prompt/RAG 基线并决定上线。",
        "prompt": """
一张适合大模型微调页的决策闭环图，主题是“先判断是否需要微调，再准备数据和评估”。
画面表现先判断问题类型，定义任务和评估标准，准备高质量样本，选择 LoRA、QLoRA 或 PEFT，训练与验证，再和 Prompt/RAG baseline 对比，决定是否上线。
风格像技术路线决策树和实验闭环结合，实用、清楚。
文字不是主体；标准术语保留英文，例如 LoRA、QLoRA、PEFT、baseline、train/validation/test、Prompt、RAG。其他说明可用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch07-alignment-chapter-flow.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "大模型对齐章节关系图",
        "suggested_page": "docs/ch07-llm-principles/ch07-alignment/00-roadmap.md",
        "alt": "大模型对齐章节关系图：预训练、指令微调、人类反馈、RLHF/DPO、安全边界和可靠应用逐步连接。",
        "prompt": """
一张适合大模型对齐导读页的章节关系图，主题是“有能力不等于好用、可靠和安全”。
画面表现预训练和指令微调后，加入人类反馈、偏好比较、RLHF/DPO、安全边界和可靠应用评估，让模型更有帮助、更诚实、更安全。
风格像模型行为校准仪和安全护栏结合，清晰、专业。
文字不是主体；标准术语保留英文，例如 SFT、human feedback、RLHF、DPO、safety boundary、helpfulness、honesty。其他说明可用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch07-alignment-app-safety-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "对齐与应用安全边界图",
        "suggested_page": "docs/ch07-llm-principles/ch07-alignment/00-roadmap.md",
        "alt": "对齐与应用安全边界图：模型对齐让模型更愿意遵循指令、承认不确定和遵守安全边界，并连接 Prompt 应用、RAG 引用和 Agent 工具权限。",
        "prompt": """
一张适合大模型对齐页的应用安全图，主题是“对齐会影响 Prompt、RAG 和 Agent 的行为边界”。
画面表现模型对齐带来三个倾向：follow instructions、admit uncertainty、respect safety boundaries；它们分别连接 Prompt 应用、RAG 引用、Agent 工具权限、人工确认和日志审计。
风格像安全控制台和系统边界图结合，强调可靠应用。
文字不是主体；标准术语保留英文，例如 Prompt、RAG、Agent、tool permission、human confirmation、audit log。其他说明可用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch07-projects-route-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "大模型综合项目路线图",
        "suggested_page": "docs/ch07-llm-principles/ch08-projects/00-roadmap.md",
        "alt": "大模型综合项目路线图：领域任务、Prompt baseline、失败分析、Prompt 优化、RAG、微调、评估对比和项目展示组成方案选择闭环。",
        "prompt": """
一张适合大模型综合项目导读页的路线图，主题是“项目不是直接微调，而是先判断问题类型”。
画面表现领域任务先建立 Prompt baseline，再分析失败类型：任务表达不清则优化 Prompt，知识不足则考虑 RAG，行为不稳则考虑微调，最后进入评估对比和项目展示。
风格像项目看板和技术路线决策图结合，清晰、工程化。
文字不是主体；标准术语保留英文，例如 domain task、Prompt baseline、failure analysis、RAG、fine-tuning、evaluation。其他说明可用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch07-project-method-choice-loop.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "大模型项目方法选择闭环图",
        "suggested_page": "docs/ch07-llm-principles/ch08-projects/00-roadmap.md",
        "alt": "大模型项目方法选择闭环图：任务定义、数据样例、baseline、失败分析、方法选择、实现方案、评估对比、结论取舍和作品集输出组成项目闭环。",
        "prompt": """
一张适合大模型项目章的方法选择闭环图，主题是“围绕任务、数据、方法和评估做取舍”。
画面表现任务定义进入数据和样例，建立 baseline，做失败分析，选择 Prompt、RAG 或 fine-tuning，完成实现方案、评估对比、结论取舍，最后输出 README、报告和作品集材料。
风格像实验闭环、决策树和作品集交付看板结合，专业但新手友好。
文字不是主体；标准术语保留英文，例如 baseline、Prompt、RAG、fine-tuning、evaluation set、README、portfolio。其他说明可用少量中文短标签。不要整段英文说明、乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch08-rag-engineering.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "LLM 应用与 RAG 主视觉",
        "suggested_page": "docs/ch08-rag/index.md",
        "alt": "LLM 应用与 RAG 主视觉：文档处理、向量库、检索、Prompt、API 和日志组成知识库系统。",
        "prompt": """
一张适合 LLM 应用工程课程的阶段主视觉，主题是“LLM 应用开发与 RAG”。
画面表现 PDF、Word、PPT 和网页资料进入文档处理流水线，写入向量数据库，再通过检索、Prompt、API、日志和评估生成带来源答案。
风格工程化、清晰、适合新手理解系统结构，不要生成具体文字，不要出现真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch08-learning-quest-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "RAG 学习闯关地图",
        "suggested_page": "docs/ch08-rag/index.md",
        "alt": "RAG 学习闯关地图：准备文档、切分、Embedding、向量库、检索、Prompt、生成带来源答案、评估和优化逐步连接。",
        "prompt": """
一张适合 LLM 应用与 RAG 首页的学习闯关地图，主题是“从资料到带来源答案”。
画面表现准备文档、切分 Chunk、生成 Embedding、写入 vector database、retrieval、组织 Prompt、生成带来源答案、评估和优化逐步连接。
风格像清晰的课程路线图和工程流水线结合，适合新人一眼看懂顺序。
文字不是主体；标准术语保留英文，例如 Chunk、Embedding、vector database、retrieval、Prompt、evaluation。其他说明可用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch08-rag-system-backbone.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "RAG 应用系统主干图",
        "suggested_page": "docs/ch08-rag/index.md",
        "alt": "RAG 应用系统主干图：用户问题经过查询改写、向量检索、文档召回、重排、上下文拼接、LLM 生成、引用评估和日志记录。",
        "prompt": """
一张适合解释现代 RAG 应用的系统主干图，主题是“RAG 不是只接向量库，而是一条可调试链路”。
画面表现用户问题进入 query rewrite，经过 vector search、document retrieval、rerank、context assembly、LLM answer，最后产生 citation、evaluation 和 logs。
风格像产品架构图和调试链路图结合，突出每一步都可以检查。
文字不是主体；标准术语保留英文，例如 query rewrite、vector search、rerank、context、LLM、citation、logs。其他说明可用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch08-ragops-improvement-loop.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "RAGOps 持续改进闭环图",
        "suggested_page": "docs/ch08-rag/index.md",
        "alt": "RAGOps 持续改进闭环图：文档更新、重新解析、重建索引、固定问题评估、上线监控、失败样本、调整切分检索和效果对比形成循环。",
        "prompt": """
一张适合 RAGOps 章节的持续改进闭环图，主题是“知识库上线后还要持续维护质量”。
画面表现文档更新、重新解析、重建索引、固定问题评估、上线监控、记录失败样本、调整 chunk/retrieval、比较优化效果，形成一个闭环。
风格像运维控制台和学习反馈飞轮结合，强调版本、日志和评估。
文字不是主体；标准术语保留英文，例如 RAGOps、index version、eval set、retrieval logs、chunk、monitoring。其他说明可用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch08-study-guide-four-layer-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "LLM 应用四层学习地图",
        "suggested_page": "docs/ch08-rag/study-guide.md",
        "alt": "LLM 应用四层学习地图：知识层、模型层、应用层和工程层逐步连接，形成可运行的 RAG 应用。",
        "prompt": """
一张适合第八章学习指南的四层学习地图，主题是“LLM 应用不是只调接口，而是四层系统”。
画面用清晰分层表现知识层、模型层、应用层和工程层：文档与向量库、LLM 与 Embedding、对话与工具、API 日志部署评估。
风格像课程分层地图，温和、清楚、对新人友好。
文字不是主体；标准术语保留英文，例如 LLM、Embedding、RAG、API、logs、deployment、evaluation。其他说明可用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch08-rag-position-bridge.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "RAG 在大模型应用中的位置桥接图",
        "suggested_page": "docs/ch08-rag/ch01-rag/00-roadmap.md",
        "alt": "RAG 位置桥接图：大模型原理、Prompt、微调、RAG、知识库问答、工具调用和 Agent 逐步连接。",
        "prompt": """
一张适合 RAG 学前导读的桥接图，主题是“从理解模型到组织应用系统”。
画面表现 LLM 原理连接 Prompt，Prompt 连接 fine-tuning，fine-tuning 连接 RAG，RAG 连接知识库问答，再连接 tool calling 和 Agent。
风格像学习阶段桥梁和路线图结合，帮助新人知道 RAG 在课程中的位置。
文字不是主体；标准术语保留英文，例如 LLM、Prompt、fine-tuning、RAG、tool calling、Agent。其他说明可用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch08-rag-core-chapter-flow.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "RAG 核心章节学习顺序图",
        "suggested_page": "docs/ch08-rag/ch01-rag/00-roadmap.md",
        "alt": "RAG 核心章节学习顺序图：最小闭环、文档解析清洗、Chunk 与元数据、Embedding 与向量库、召回过滤重排、上下文生成、评估优化逐步连接。",
        "prompt": """
一张适合 RAG 核心章节导读的学习顺序图，主题是“先跑通闭环，再优化每一层”。
画面表现最小 RAG 闭环、文档解析清洗、Chunk 与 metadata、Embedding 与 vector database、召回过滤重排、context 组装与答案生成、evaluation 优化。
风格像任务路线图和调试看板结合，帮助新人知道先学什么后学什么。
文字不是主体；标准术语保留英文，例如 Chunk、metadata、Embedding、vector database、retrieval、rerank、context、evaluation。其他说明可用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch08-rag-data-to-answer-pipeline.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "RAG 从资料到答案流水线图",
        "suggested_page": "docs/ch08-rag/ch01-rag/00-roadmap.md",
        "alt": "RAG 从资料到答案流水线图：原始文档解析清洗切块向量化入库，用户问题向量化检索重排，拼接上下文后由 LLM 生成带来源答案。",
        "prompt": """
一张适合 RAG 导读页的完整流水线图，主题是“把资料变成可检索知识，再变成可引用答案”。
画面分成上下两条流：上半部分是原始文档解析、清洗、chunk、embedding、index；下半部分是用户问题、query embedding、retrieval、rerank、context、LLM answer、citation。
风格像双通道工程管线，重点突出资料端和问题端在哪里汇合。
文字不是主体；标准术语保留英文，例如 chunk、embedding、index、query、retrieval、rerank、context、LLM answer、citation。其他说明可用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch08-deployment-chapter-flow.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "模型部署章节学习顺序图",
        "suggested_page": "docs/ch08-rag/ch02-deployment/00-roadmap.md",
        "alt": "模型部署章节学习顺序图：本地模型运行、推理服务、统一 API 入口逐步连接，让模型调用从实验变成稳定接口。",
        "prompt": """
一张适合模型部署导读页的章节学习顺序图，主题是“模型从实验运行到稳定服务入口”。
画面表现本地模型运行进入 inference server，再进入 unified API，最后被 RAG、应用开发和工程监控调用。
风格像服务化架构图，清楚表达从单机运行到可复用接口的递进关系。
文字不是主体；标准术语保留英文，例如 local model、inference server、unified API、RAG、monitoring。其他说明可用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch08-model-serving-decision-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "模型服务选型决策图",
        "suggested_page": "docs/ch08-rag/ch02-deployment/00-roadmap.md",
        "alt": "模型服务选型决策图：效果、延迟、成本、隐私、显存、量化、路由和推理优化共同决定云端、本地或混合部署方式。",
        "prompt": """
一张适合模型工程精讲的选型决策图，主题是“不是永远调用最强模型，而是平衡效果、成本和约束”。
画面中心是 model serving，周围连接 quality、latency、cost、privacy、GPU memory、quantization、routing、caching、streaming，最后分向 cloud API、local model 和 hybrid deployment。
风格像决策罗盘和系统架构图结合，帮助新人理解部署取舍。
文字不是主体；标准术语保留英文，例如 model serving、quality、latency、cost、privacy、quantization、routing、caching、streaming、hybrid deployment。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch08-app-dev-chapter-flow.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "LLM 应用开发章节关系图",
        "suggested_page": "docs/ch08-rag/ch03-app-dev/00-roadmap.md",
        "alt": "LLM 应用开发章节关系图：模型 API、RAG 知识库、工具函数、应用服务、对话系统、用户界面、反馈日志和评估逐步连接。",
        "prompt": """
一张适合 LLM 应用开发导读页的章节关系图，主题是“把一次模型调用升级成应用功能”。
画面表现 model API、RAG knowledge base、tools/functions 汇入 application service，再连接 dialogue system、user interface、feedback、logs 和 evaluation。
风格像产品架构图和学习路线图结合，强调模型只是应用的一层。
文字不是主体；标准术语保留英文，例如 model API、RAG、tools、Function Calling、dialogue system、feedback、logs、evaluation。其他说明可用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch08-app-dev-learning-order-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "LLM 应用开发学习顺序图",
        "suggested_page": "docs/ch08-rag/ch03-app-dev/00-roadmap.md",
        "alt": "LLM 应用开发学习顺序图：LLM API 调用、错误处理日志、模型层 Prompt 层抽象、Function Calling、多轮对话状态、文档代码业务流程逐步连接。",
        "prompt": """
一张适合 LLM 应用开发章的新手学习顺序图，主题是“先把调用跑稳，再逐步接入工具和业务流程”。
画面表现 LLM API 调用、error handling 和 logs、模型层和 Prompt 层抽象、Function Calling、多轮 dialogue state、文档处理、代码助手和业务流程逐步升级。
风格像课程路径图和工程模块图结合，层级清楚，适合新人跟着学。
文字不是主体；标准术语保留英文，例如 LLM API、error handling、logs、Prompt layer、Function Calling、dialogue state。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch08-llm-app-capability-loop.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "LLM 应用能力闭环图",
        "suggested_page": "docs/ch08-rag/ch03-app-dev/00-roadmap.md",
        "alt": "LLM 应用能力闭环图：用户输入、意图识别、RAG 检索、Function Calling、Prompt 组装、LLM 输出、解析校验、展示日志反馈形成闭环。",
        "prompt": """
一张适合应用开发主线的能力闭环图，主题是“可用的 LLM 应用必须处理输入、外部能力、输出和反馈”。
画面表现用户输入进入 intent detection 和 context整理，分流到 RAG retrieval、Function Calling 或 direct generation，再进入 Prompt assembly、LLM output、parse/validate、display、logs、feedback。
风格像应用状态机和调试闭环结合，清晰、工程化。
文字不是主体；标准术语保留英文，例如 intent detection、RAG retrieval、Function Calling、Prompt assembly、LLM output、parse、validate、logs、feedback。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch08-engineering-chapter-flow.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "LLM 工程化章节学习顺序图",
        "suggested_page": "docs/ch08-rag/ch04-engineering/00-roadmap.md",
        "alt": "LLM 工程化章节学习顺序图：异步并发、API 设计、日志监控、容器化部署逐步连接，让能跑的应用变成可维护系统。",
        "prompt": """
一张适合 LLM 工程化导读页的章节学习顺序图，主题是“从能跑到能上线、能排障、能维护”。
画面表现 async/concurrency、API design、logging/monitoring、Docker deployment 逐步连接，并回到评估和告警。
风格像工程升级路线和系统运行仪表盘结合，新手友好。
文字不是主体；标准术语保留英文，例如 async、concurrency、API design、logging、monitoring、Docker、deployment、alert。其他说明可用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch08-llmops-trace-loop.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "LLMOps Trace 复盘闭环图",
        "suggested_page": "docs/ch08-rag/ch04-engineering/00-roadmap.md",
        "alt": "LLMOps Trace 复盘闭环图：输入、检索、Prompt 版本、模型输出、工具结果、最终答案、耗时成本和失败样本支持排障复盘。",
        "prompt": """
一张适合 LLMOps 精讲的 Trace 复盘闭环图，主题是“系统答错时要能查清每一层发生了什么”。
画面表现 user input、retrieval results、Prompt version、model output、tool result、final answer、latency、cost、failure sample 被记录成 trace，并进入 evaluation 和 rollback。
风格像可观测性控制台和事故复盘白板结合，专业但不压迫。
文字不是主体；标准术语保留英文，例如 trace、Prompt version、retrieval results、model output、tool result、latency、cost、evaluation、rollback。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch08-projects-route-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "LLM 应用综合项目路线图",
        "suggested_page": "docs/ch08-rag/ch05-projects/00-roadmap.md",
        "alt": "LLM 应用综合项目路线图：文档资料、解析切分向量化、RAG 检索、LLM 调用、应用功能、日志评估错误处理、部署展示逐步连接。",
        "prompt": """
一张适合第八章综合项目导读页的项目路线图，主题是“把知识、模型、应用和工程做成系统”。
画面表现文档资料进入解析切分向量化，经过 RAG retrieval 和 LLM call，形成应用功能，再补 logs、evaluation、error handling、deployment 和 demo。
风格像作品集项目路线图和工程架构图结合，清楚、可讲解。
文字不是主体；标准术语保留英文，例如 RAG retrieval、LLM call、logs、evaluation、error handling、deployment、demo、portfolio。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch08-project-learning-order-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "LLM 应用项目学习顺序图",
        "suggested_page": "docs/ch08-rag/ch05-projects/00-roadmap.md",
        "alt": "LLM 应用项目学习顺序图：知识库问答、文档到答案链路、来源引用评估、智能问答助手、会话状态工具调用、RAG 微调组合和课件生成应用逐步升级。",
        "prompt": """
一张适合综合项目章的新手学习顺序图，主题是“先做知识库问答，再升级复杂应用”。
画面表现先做知识库问答，跑通文档到答案链路，加入来源引用和评估样例，扩展为智能问答助手，加入 conversation state 和 tools，再尝试 RAG + fine-tuning，最后做课件生成等复杂应用。
风格像项目升级阶梯和学习路线图结合，鼓励新人一步步完成。
文字不是主体；标准术语保留英文，例如 conversation state、tools、RAG + fine-tuning、evaluation、courseware generation。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch08-project-delivery-loop.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "LLM 应用项目交付闭环图",
        "suggested_page": "docs/ch08-rag/ch05-projects/00-roadmap.md",
        "alt": "LLM 应用项目交付闭环图：用户任务、意图判断、RAG 检索或直接模型调用、结构化输出、校验引用、日志反馈、评估迭代形成作品集闭环。",
        "prompt": """
一张适合综合项目章的项目交付闭环图，主题是“项目不是一次生成，而是知识、模型、功能和工程的闭环”。
画面表现用户问题或任务进入 intent 判断，需要知识时走 RAG retrieval，不需要时走 direct LLM call，再生成 structured output，进行 validation、citation、logs、feedback、evaluation iteration，最后沉淀 README 和作品集展示。
风格像项目交付看板和系统闭环图结合，强调证据、日志和评估。
文字不是主体；标准术语保留英文，例如 intent、RAG retrieval、direct LLM call、structured output、validation、citation、logs、feedback、evaluation、README、portfolio。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch09-agent-systems.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "AI Agent 系统主视觉",
        "suggested_page": "docs/ch09-agent/index.md",
        "alt": "AI Agent 系统主视觉：目标、计划、工具、记忆、观察和评估组成智能体执行闭环。",
        "prompt": """
一张适合 AI Agent 课程的阶段主视觉，主题是“AI Agent 与智能体系统”。
画面表现一个 AI 助手围绕目标拆解计划、调用工具、读取资料、更新记忆、记录 trace、做安全检查并输出结果。
风格清晰、系统设计感、适合技术课程，不要生成具体文字，不要出现真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch12-multimodal-aigc.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "AIGC 与多模态主视觉",
        "suggested_page": "docs/ch12-multimodal/index.md",
        "alt": "AIGC 与多模态主视觉：文字、图像、语音、视频和审核导出组成创意工作流。",
        "prompt": """
一张适合多模态与 AIGC 课程的阶段主视觉，主题是“AIGC 与多模态”。
画面表现文字、图片、语音、视频、分镜、素材版本和内容审核被组织成一个创意工作流，像一张现代 AI 创作工作台。
风格明亮、专业、有产品感，不要生成具体文字，不要出现真实品牌 logo。
""".strip(),
    },
    {
        "filename": "git-four-areas.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Git 四区工作流图",
        "suggested_page": "docs/ch01-tools/ch02-git/01-git-basics.md",
        "alt": "Git 四区工作流图：工作区、暂存区、本地仓库和远程仓库之间的 add、commit、push、pull 关系。",
        "prompt": """
一张适合 Git 新手课程的教学示意图，主题是“Git 四区工作流”。
画面用四个清晰区域表现工作区、暂存区、本地仓库、远程仓库，并用箭头表达 add、commit、push、pull 的方向。
风格简洁、现代、适合中文技术课程，不要生成难以阅读的小字，不要出现真实品牌 logo。
""".strip(),
    },
    {
        "filename": "pandas-dataframe-structure.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Pandas DataFrame 结构图",
        "suggested_page": "docs/ch03-data-analysis/ch03-pandas/01-core-structures.md",
        "alt": "Pandas DataFrame 结构图：行、列、索引、Series 和 DataFrame 的关系。",
        "prompt": """
一张适合 Pandas 入门课程的教学示意图，主题是“DataFrame 结构”。
画面表现一张表格，突出行、列、索引、列名、单列 Series、多列 DataFrame 的关系。
风格清晰、像教学白板和现代数据分析工具结合，不要生成真实品牌 logo，不要出现乱码文字。
""".strip(),
    },
    {
        "filename": "chart-selection-decision-tree.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "图表选择决策树",
        "suggested_page": "docs/ch03-data-analysis/ch04-visualization/04-best-practices.md",
        "alt": "图表选择决策树：比较大小、看趋势、看分布、看关系时选择不同图表。",
        "prompt": """
一张适合数据可视化课程的教学图，主题是“如何选图表”。
画面像一棵简洁决策树：比较大小、看趋势、看分布、看关系、看构成，分别连接到柱状图、折线图、直方图、散点图、堆叠图等图形图标。
风格清爽、专业、适合新手学习，不要生成真实品牌 logo，不要出现难以阅读的小字。
""".strip(),
    },
    {
        "filename": "gradient-descent-path.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "梯度下降路径图",
        "suggested_page": "docs/ch04-ai-math/ch03-calculus/03-gradient-descent.md",
        "alt": "梯度下降路径图：参数点沿着损失函数曲面一步步走向较低位置。",
        "prompt": """
一张适合 AI 数学课程的教学插图，主题是“梯度下降”。
画面表现一个点在损失函数山坡或曲面上，根据箭头一步步往低处移动，旁边有简洁的路径轨迹和学习率步长感觉。
风格直观、温和、帮助新手理解优化，不要生成具体公式文字，不要出现真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ml-modeling-loop.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "机器学习建模闭环图",
        "suggested_page": "docs/ch05-machine-learning/ch01-ml-basics/01-what-is-ml.md",
        "alt": "机器学习建模闭环图：任务定义、数据准备、baseline、训练、评估和错误分析形成循环。",
        "prompt": """
一张适合机器学习入门课程的系统流程图，主题是“建模闭环”。
画面表现任务定义、数据准备、baseline、训练、评估、错误分析和改进形成循环，像一份模型侦探报告流程。
风格专业、新手友好、清晰，不要生成真实品牌 logo，不要出现乱码小字。
""".strip(),
    },
    {
        "filename": "confusion-matrix-error-cost.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "混淆矩阵与错误代价图",
        "suggested_page": "docs/ch05-machine-learning/ch04-evaluation/01-metrics.md",
        "alt": "混淆矩阵与错误代价图：TP、FP、FN、TN 对应不同业务后果。",
        "prompt": """
一张适合机器学习评估课程的教学图，主题是“混淆矩阵和错误代价”。
画面用 2x2 矩阵表现预测和真实的四种组合，并用简洁图标表达误报、漏报、正确识别和正确忽略的不同业务后果。
风格清晰、教学感强，不要生成真实品牌 logo，不要出现难以阅读的小字。
""".strip(),
    },
    {
        "filename": "pytorch-training-loop.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "PyTorch 训练循环图",
        "suggested_page": "docs/ch06-deep-learning/ch02-pytorch/05-training-loop.md",
        "alt": "PyTorch 训练循环图：DataLoader、model、loss、backward、optimizer.step 组成训练闭环。",
        "prompt": """
一张适合 PyTorch 入门课程的教学图，主题是“训练循环”。
画面表现 batch 数据进入模型，输出计算 loss，loss backward 产生梯度，optimizer 更新参数，然后进入下一轮。
风格像现代工程白板，结构清晰，不要生成真实品牌 logo，不要出现乱码文字。
""".strip(),
    },
    {
        "filename": "cnn-convolution-kernel.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "CNN 卷积核滑动示意图",
        "suggested_page": "docs/ch06-deep-learning/ch03-cnn/01-convolution-basics.md",
        "alt": "CNN 卷积核滑动示意图：小卷积核在图像网格上滑动并生成特征图。",
        "prompt": """
一张适合 CNN 课程的教学示意图，主题是“卷积核滑动”。
画面表现一个小卷积核在像素网格上移动，提取边缘或纹理模式，并输出右侧特征图。
风格清晰、具体、适合新手理解图像卷积，不要生成真实品牌 logo，不要出现乱码文字。
""".strip(),
    },
    {
        "filename": "self-attention-qkv.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Self-Attention QKV 结构图",
        "suggested_page": "docs/ch06-deep-learning/ch05-transformer/01-attention-mechanism.md",
        "alt": "Self-Attention QKV 结构图：Query、Key、Value 计算相关性并汇总上下文信息。",
        "prompt": """
一张适合 Transformer 课程的教学结构图，主题是“Self-Attention 的 QKV”。
画面表现多个 token 生成 Query、Key、Value，Query 和 Key 计算相关性权重，再加权汇总 Value 得到上下文表示。
风格现代、清晰、适合教学，不要出现真实品牌 logo，不要生成难以阅读的小字。
""".strip(),
    },
    {
        "filename": "object-detection-output.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "目标检测输出拆解图",
        "suggested_page": "docs/ch10-computer-vision/ch03-detection/01-detection-overview.md",
        "alt": "目标检测输出拆解图：图片中的目标框、类别和置信度组成检测结果。",
        "prompt": """
一张适合计算机视觉课程的教学图，主题是“目标检测输出”。
画面表现一张图片中多个物体被框出，每个框对应类别、位置和置信度，用清晰图标表达检测比分类多了什么。
风格专业、直观，不要出现真实品牌 logo，不要生成难以阅读的小字。
""".strip(),
    },
    {
        "filename": "semantic-segmentation-mask.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "语义分割 Mask 对比图",
        "suggested_page": "docs/ch10-computer-vision/ch04-segmentation/01-semantic-segmentation.md",
        "alt": "语义分割 Mask 对比图：原图、像素级类别 mask 和边界错误对照。",
        "prompt": """
一张适合图像分割课程的教学图，主题是“语义分割 mask”。
画面左侧是原图抽象示意，右侧是不同颜色覆盖的像素级分割 mask，并突出边界、小目标和类别不平衡这些常见问题。
风格清晰、教学感强，不要出现真实品牌 logo，不要生成乱码文字。
""".strip(),
    },
    {
        "filename": "bio-ner-recovery.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "BIO 标签到实体恢复图",
        "suggested_page": "docs/ch11-nlp/ch04-sequence-labeling/01-ner-overview.md",
        "alt": "BIO 标签到实体恢复图：词序列上的 B、I、O 标签被合并成命名实体。",
        "prompt": """
一张适合 NLP 序列标注课程的教学图，主题是“BIO 到实体恢复”。
画面表现一句话被分成 token，每个 token 有 B、I、O 标签，相邻标签最终合并成姓名、地点、机构等实体块。
风格清爽、像标注工具界面，不要出现真实品牌 logo，不要生成乱码文字。
""".strip(),
    },
    {
        "filename": "bert-gpt-t5-comparison.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "BERT GPT T5 对比图",
        "suggested_page": "docs/ch11-nlp/ch06-pretrained/00-roadmap.md",
        "alt": "BERT GPT T5 对比图：理解型、续写型和 text-to-text 统一任务范式的差异。",
        "prompt": """
一张适合预训练模型课程的对比图，主题是“BERT、GPT、T5 的差异”。
画面用三个并列模块表现 BERT 更像双向理解，GPT 更像顺着前文续写，T5 更像把所有任务统一成 text-to-text。
风格清晰、现代、适合新手对比，不要出现真实品牌 logo，不要生成乱码小字。
""".strip(),
    },
    {
        "filename": "prompt-before-after.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Prompt 改写前后对比卡",
        "suggested_page": "docs/ch07-llm-principles/ch05-prompt/01-prompt-basics.md",
        "alt": "Prompt 改写前后对比卡：模糊请求加入角色、任务、约束、格式和示例后变得可执行。",
        "prompt": """
一张适合 Prompt 工程课程的教学图，主题是“坏 Prompt 到好 Prompt”。
画面用左右对比卡片表现：左边是模糊请求，右边加入角色、任务、背景、约束、输出格式和示例后变得清晰。
风格像产品设计评审卡片，清爽易懂，不要出现真实品牌 logo，不要生成具体小字。
""".strip(),
    },
    {
        "filename": "lora-parameter-update.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "LoRA 参数更新对比图",
        "suggested_page": "docs/ch07-llm-principles/ch06-finetuning/02-lora-qlora.md",
        "alt": "LoRA 参数更新对比图：冻结大模型主体，只训练小型低秩适配器参数。",
        "prompt": """
一张适合大模型微调课程的结构图，主题是“LoRA 参数更新”。
画面表现一个大模型主体被冻结，旁边插入小型低秩适配器模块，训练时只更新适配器，和全量微调形成对比。
风格专业、清晰、适合新人理解，不要出现真实品牌 logo，不要生成乱码文字。
""".strip(),
    },
    {
        "filename": "rag-document-answer-loop.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "RAG 文档到答案闭环图",
        "suggested_page": "docs/ch08-rag/ch01-rag/01-rag-basics.md",
        "alt": "RAG 文档到答案闭环图：文档切块、向量化、检索、重排、Prompt 和带来源答案组成闭环。",
        "prompt": """
一张适合 RAG 课程的系统图，主题是“文档到答案闭环”。
画面表现 PDF、Word、网页资料被切块、向量化、写入向量库，用户问题触发检索和重排，最后生成带来源引用的答案。
风格工程化、清晰、适合中文课程，不要出现真实品牌 logo，不要生成乱码文字。
""".strip(),
    },
    {
        "filename": "courseware-assistant-workflow.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "课件生成助手工作流图",
        "suggested_page": "docs/ch08-rag/ch05-projects/04-courseware-assistant.md",
        "alt": "课件生成助手工作流图：主题输入、资料检索、例题抽取、结构化生成和 Word 模板导出。",
        "prompt": """
一张适合 AI 项目课程的系统工作流图，主题是“知识库驱动的课件生成助手”。
画面表现用户输入主题，系统查找 PDF、Word、PPT 和外部资料，抽取例题，生成结构化课件，再套 Word 模板导出。
风格像产品架构图和工作台结合，清晰、专业，不要出现真实品牌 logo，不要生成乱码文字。
""".strip(),
    },
    {
        "filename": "agent-tool-trace.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Agent 工具调用 Trace 图",
        "suggested_page": "docs/ch09-agent/ch03-tools/08-multi-tool-practice.md",
        "alt": "Agent 工具调用 Trace 图：计划、工具选择、参数、观察结果和最终回复串成可回放轨迹。",
        "prompt": """
一张适合 AI Agent 课程的执行轨迹图，主题是“多工具调用 trace”。
画面表现 Agent 从计划开始，依次选择工具、传入参数、接收观察结果、更新状态，最终生成回复，整个过程可回放。
风格系统设计感、清晰、技术课程友好，不要出现真实品牌 logo，不要生成难以阅读的小字。
""".strip(),
    },
    {
        "filename": "agent-guardrails-layers.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Agent 护栏分层图",
        "suggested_page": "docs/ch09-agent/ch08-eval-safety/04-guardrails.md",
        "alt": "Agent 护栏分层图：输入、检索、工具、输出和人工确认构成多层安全边界。",
        "prompt": """
一张适合 Agent 安全课程的教学图，主题是“护栏分层”。
画面表现输入检查、资料来源检查、工具权限检查、输出格式校验、高风险人工确认等多层护栏围绕 Agent 执行链路。
风格清晰、稳重、像系统安全架构图，不要出现真实品牌 logo，不要生成乱码文字。
""".strip(),
    },
    {
        "filename": "diffusion-noise-denoise.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "扩散模型加噪去噪图",
        "suggested_page": "docs/ch12-multimodal/ch02-image-gen/01-diffusion-models.md",
        "alt": "扩散模型加噪去噪图：图像逐步加噪，再从噪声中一步步去噪生成图像。",
        "prompt": """
一张适合图像生成课程的教学图，主题是“扩散模型加噪与去噪”。
画面表现一张清晰图像逐步变成噪声，再从噪声经过多步去噪恢复成图像，中间有时间步和方向感。
风格现代、直观、适合新手理解生成过程，不要出现真实品牌 logo，不要生成乱码文字。
""".strip(),
    },
    {
        "filename": "matplotlib-figure-axes.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Matplotlib Figure 与 Axes 结构图",
        "suggested_page": "docs/ch03-data-analysis/ch04-visualization/01-matplotlib.md",
        "alt": "Matplotlib Figure 与 Axes 结构图：画布、子图、坐标轴、标题、图例和数据线之间的关系。",
        "prompt": """
一张适合 Matplotlib 入门课程的教学图，主题是“Figure、Axes 和图表元素的关系”。
画面表现一个大画布 Figure 内部有多个 Axes 子图，每个子图包含坐标轴、标题、图例、网格、数据线和标注。
风格清晰、像现代教学白板，帮助新手建立对象层级，不要出现真实品牌 logo，不要生成难以阅读的小字。
""".strip(),
    },
    {
        "filename": "seaborn-statistical-plots.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Seaborn 统计图选择图",
        "suggested_page": "docs/ch03-data-analysis/ch04-visualization/02-seaborn.md",
        "alt": "Seaborn 统计图选择图：分布、关系、类别、矩阵和分面图对应不同数据分析问题。",
        "prompt": """
一张适合 Seaborn 课程的教学图，主题是“统计可视化图谱”。
画面用清晰卡片展示直方图、KDE、散点图、箱线图、热力图、FacetGrid 等图表分别回答分布、关系、类别差异、相关性和分组对比问题。
风格现代、数据分析感强，适合新手快速选图，不要出现真实品牌 logo，不要生成乱码文字。
""".strip(),
    },
    {
        "filename": "sql-table-join-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "SQL 表连接关系图",
        "suggested_page": "docs/ch03-data-analysis/ch05-database/02-sql-basics.md",
        "alt": "SQL 表连接关系图：用户表、订单表和商品表通过主键外键关联，并通过查询得到分析结果。",
        "prompt": """
一张适合 SQL 入门课程的教学图，主题是“表、主键、外键和 JOIN”。
画面表现用户表、订单表、商品表通过主键和外键连接，查询语句把多张表合成一张分析结果表。
风格清晰、数据库关系图与数据分析工作流结合，不要出现真实品牌 logo，不要生成难以阅读的小字。
""".strip(),
    },
    {
        "filename": "eda-analysis-workflow.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "EDA 探索性数据分析流程图",
        "suggested_page": "docs/ch03-data-analysis/ch06-projects/01-eda-project.md",
        "alt": "EDA 探索性数据分析流程图：数据概览、质量检查、特征构造、统计分析、可视化和结论汇报。",
        "prompt": """
一张适合 EDA 实战项目的教学流程图，主题是“探索性数据分析完整路径”。
画面表现数据加载、数据概览、缺失异常检查、特征构造、统计分析、可视化洞察和结论报告形成闭环。
风格像数据分析师工作台，清晰、实战感强，不要出现真实品牌 logo，不要生成乱码文字。
""".strip(),
    },
    {
        "filename": "vector-dot-cosine-geometry.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "向量点积与余弦相似度几何图",
        "suggested_page": "docs/ch04-ai-math/ch01-linear-algebra/01-vectors.md",
        "alt": "向量点积与余弦相似度几何图：两个向量的夹角越小，方向越接近，相似度越高。",
        "prompt": """
一张适合线性代数入门课程的教学图，主题是“向量、夹角、点积和余弦相似度”。
画面表现二维坐标系中的多个向量，突出两个向量之间的夹角、投影和方向相似程度，用颜色表达相似和不相似。
风格直观、温和、帮助新手把公式变成几何直觉，不要出现真实品牌 logo，不要生成难以阅读的小字。
""".strip(),
    },
    {
        "filename": "matrix-linear-transform-grid.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "矩阵线性变换网格图",
        "suggested_page": "docs/ch04-ai-math/ch01-linear-algebra/02-matrices.md",
        "alt": "矩阵线性变换网格图：矩阵把原始网格旋转、缩放、拉伸或压缩成新的空间。",
        "prompt": """
一张适合矩阵课程的教学图，主题是“矩阵就是空间变换”。
画面左侧是规则网格和一个简单形状，右侧展示经过矩阵作用后的旋转、缩放、剪切或压缩效果，并用箭头连接。
风格清晰、数学可视化、适合新手理解矩阵乘法，不要出现真实品牌 logo，不要生成乱码文字。
""".strip(),
    },
    {
        "filename": "probability-distribution-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "概率分布与贝叶斯更新图",
        "suggested_page": "docs/ch04-ai-math/ch02-probability/01-probability-basics.md",
        "alt": "概率分布与贝叶斯更新图：先验概率遇到新证据后，更新为后验概率。",
        "prompt": """
一张适合概率论入门课程的教学图，主题是“概率分布、条件概率和贝叶斯更新”。
画面表现一个不确定事件的概率分布，加入新证据后分布发生更新，形成先验到后验的直观变化。
风格柔和、清晰、像新手友好的数学插图，不要出现真实品牌 logo，不要生成难以阅读的小字。
""".strip(),
    },
    {
        "filename": "information-entropy-uncertainty.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "信息熵与不确定性图",
        "suggested_page": "docs/ch04-ai-math/ch02-probability/04-information-theory.md",
        "alt": "信息熵与不确定性图：越平均的分布越难猜，熵越高；越确定的分布熵越低。",
        "prompt": """
一张适合信息论课程的教学图，主题是“熵、不确定性和交叉熵损失”。
画面用不同概率分布的小卡片对比：确定事件、偏斜分布、均匀分布，并用视觉高度表达不确定性大小。
风格清晰、现代、帮助新手理解信息量和预测损失，不要出现真实品牌 logo，不要生成乱码文字。
""".strip(),
    },
    {
        "filename": "sklearn-estimator-pipeline.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Scikit-learn Estimator 与 Pipeline 图",
        "suggested_page": "docs/ch05-machine-learning/ch01-ml-basics/02-sklearn-intro.md",
        "alt": "Scikit-learn Estimator 与 Pipeline 图：数据经过预处理、模型训练、预测和评估，形成统一 fit/predict 工作流。",
        "prompt": """
一张适合 Scikit-learn 入门课程的工程流程图，主题是“统一的 Estimator 和 Pipeline 工作流”。
画面表现数据集进入 train/test split，经过 scaler、feature transformer、model，再调用 fit、predict、score 完成训练评估。
风格工程化、清晰、适合新手建立 sklearn 使用套路，不要出现真实品牌 logo，不要生成难以阅读的小字。
""".strip(),
    },
    {
        "filename": "linear-regression-loss-landscape.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "线性回归拟合与损失曲面图",
        "suggested_page": "docs/ch05-machine-learning/ch02-supervised/01-linear-regression.md",
        "alt": "线性回归拟合与损失曲面图：直线拟合散点，参数在损失曲面上寻找更小误差。",
        "prompt": """
一张适合线性回归课程的教学图，主题是“拟合直线和最小化误差”。
画面左侧表现散点和多条候选拟合线，右侧表现参数点在损失曲面中移动寻找最低误差。
风格直观、教学感强，帮助新手连接模型、损失和优化，不要出现真实品牌 logo，不要生成乱码文字。
""".strip(),
    },
    {
        "filename": "logistic-regression-boundary.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "逻辑回归决策边界图",
        "suggested_page": "docs/ch05-machine-learning/ch02-supervised/02-logistic-regression.md",
        "alt": "逻辑回归决策边界图：Sigmoid 把线性得分转成概率，并用边界分开两个类别。",
        "prompt": """
一张适合逻辑回归课程的教学图，主题是“概率输出与决策边界”。
画面表现二维散点被一条清晰决策边界分开，旁边展示线性得分经过 S 形曲线转成概率的直觉。
风格清晰、现代、帮助新手理解分类不是回归名字误导，不要出现真实品牌 logo，不要生成难以阅读的小字。
""".strip(),
    },
    {
        "filename": "decision-tree-split-path.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "决策树分裂路径图",
        "suggested_page": "docs/ch05-machine-learning/ch02-supervised/03-decision-trees.md",
        "alt": "决策树分裂路径图：样本沿着特征判断节点一步步走到叶子结论。",
        "prompt": """
一张适合决策树课程的教学图，主题是“从特征问题到叶子结论”。
画面表现一个样本从根节点开始，根据特征条件一路向下分裂，最终到达不同叶子类别，并展示过深树和剪枝的对比直觉。
风格像可视化流程卡片，清晰、适合新手，不要出现真实品牌 logo，不要生成乱码文字。
""".strip(),
    },
    {
        "filename": "ensemble-learning-voting-forest.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "集成学习投票与森林图",
        "suggested_page": "docs/ch05-machine-learning/ch02-supervised/04-ensemble-learning.md",
        "alt": "集成学习投票与森林图：多个弱模型通过投票、平均或逐步修正残差得到更稳结果。",
        "prompt": """
一张适合集成学习课程的教学图，主题是“多个模型一起做决定”。
画面表现多棵不同决策树和多个基础模型分别给出判断，最后通过投票、平均或残差修正汇总成更稳定预测。
风格清晰、像机器学习团队协作，帮助新手理解 Bagging、Boosting、Stacking，不要出现真实品牌 logo。
""".strip(),
    },
    {
        "filename": "clustering-kmeans-centroids.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "K-Means 聚类中心迭代图",
        "suggested_page": "docs/ch05-machine-learning/ch03-unsupervised/01-clustering.md",
        "alt": "K-Means 聚类中心迭代图：样本点分配给最近中心，中心再移动到簇的平均位置。",
        "prompt": """
一张适合聚类课程的教学图，主题是“K-Means 如何找簇中心”。
画面表现散点被不同颜色分组，多个中心点从初始位置逐步移动到簇中心，形成分配和更新的循环。
风格直观、数据可视化感强，适合新手理解无监督学习，不要出现真实品牌 logo，不要生成乱码文字。
""".strip(),
    },
    {
        "filename": "pca-dimensionality-reduction.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "PCA 降维投影图",
        "suggested_page": "docs/ch05-machine-learning/ch03-unsupervised/02-dimensionality-reduction.md",
        "alt": "PCA 降维投影图：高维数据沿主成分方向投影到低维平面，尽量保留主要变化。",
        "prompt": """
一张适合降维课程的教学图，主题是“PCA 把高维数据投影到主方向”。
画面表现三维点云沿着最大变化方向投影到二维平面，旁边用箭头表示第一主成分和第二主成分。
风格清晰、数学直觉强、帮助新手理解保留方差，不要出现真实品牌 logo，不要生成难以阅读的小字。
""".strip(),
    },
    {
        "filename": "neural-network-forward-backward.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "神经网络前向与反向传播图",
        "suggested_page": "docs/ch06-deep-learning/ch01-nn-basics/02-forward-backward.md",
        "alt": "神经网络前向与反向传播图：数据向前得到预测和损失，梯度向后更新每层参数。",
        "prompt": """
一张适合神经网络课程的教学图，主题是“前向传播和反向传播”。
画面表现输入经过多层神经网络得到预测和 loss，梯度再从 loss 反向流回每一层并更新参数。
风格清晰、像训练流程剖面图，帮助新手理解网络如何学习，不要出现真实品牌 logo，不要生成乱码文字。
""".strip(),
    },
    {
        "filename": "imagenet-cnn-evolution.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "经典 CNN 架构演进图",
        "suggested_page": "docs/ch06-deep-learning/ch03-cnn/03-classic-architectures.md",
        "alt": "经典 CNN 架构演进图：LeNet、AlexNet、VGG、ResNet 从浅层到深层逐步演进。",
        "prompt": """
一张适合 CNN 架构课程的历史演进图，主题是“经典 CNN 从 LeNet 到 ResNet”。
画面用时间轴和层级块展示 LeNet、AlexNet、VGG、ResNet 的架构变化，突出更深网络、残差连接和 ImageNet 推动。
风格现代、教学感强，不要出现真实品牌 logo，不要生成难以阅读的小字。
""".strip(),
    },
    {
        "filename": "lstm-gate-memory-flow.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "LSTM 门控记忆流图",
        "suggested_page": "docs/ch06-deep-learning/ch04-rnn/02-lstm-gru.md",
        "alt": "LSTM 门控记忆流图：遗忘门、输入门、输出门共同控制长期记忆的保留、写入和读取。",
        "prompt": """
一张适合 RNN 课程的教学结构图，主题是“LSTM 如何用门控制记忆”。
画面表现 cell state 像一条记忆传送带，遗忘门、输入门、输出门分别控制丢掉旧信息、写入新信息和输出当前状态。
风格清晰、机制感强，适合新手理解长短期记忆，不要出现真实品牌 logo，不要生成乱码文字。
""".strip(),
    },
    {
        "filename": "transformer-block-architecture.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Transformer Block 架构图",
        "suggested_page": "docs/ch06-deep-learning/ch05-transformer/02-transformer-architecture.md",
        "alt": "Transformer Block 架构图：注意力、残差连接、LayerNorm 和前馈网络组成一个可堆叠模块。",
        "prompt": """
一张适合 Transformer 架构课程的教学结构图，主题是“一个 Transformer Block 的内部结构”。
画面表现 token embedding 进入 self-attention、残差连接、LayerNorm、前馈网络，再堆叠成多层模型，并对比 encoder 和 decoder 路线。
风格工程化、清晰、适合新手拆解复杂架构，不要出现真实品牌 logo，不要生成难以阅读的小字。
""".strip(),
    },
    {
        "filename": "word2vec-embedding-neighborhood.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "词向量语义邻域图",
        "suggested_page": "docs/ch11-nlp/ch02-embeddings/01-word-embedding.md",
        "alt": "词向量语义邻域图：语义相近的词在向量空间里更靠近，并形成可计算的方向关系。",
        "prompt": """
一张适合 NLP 词嵌入课程的教学图，主题是“词语在向量空间里的语义邻居”。
画面表现二维或三维向量空间中的词点云，相近含义的词聚在一起，并用箭头展示语义方向和类比关系。
风格清晰、现代、帮助新手理解 embedding 不是普通编号，不要出现真实品牌 logo，不要生成乱码文字。
""".strip(),
    },
    {
        "filename": "bert-masked-language-model.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "BERT Masked Language Model 图",
        "suggested_page": "docs/ch11-nlp/ch06-pretrained/02-bert.md",
        "alt": "BERT Masked Language Model 图：模型同时看左右上下文，预测被遮住的 token。",
        "prompt": """
一张适合 BERT 课程的教学图，主题是“Masked Language Model”。
画面表现一句话中部分 token 被遮住，BERT 同时利用左侧和右侧上下文预测 mask，并输出用于分类或抽取的表示。
风格清晰、NLP 教学感强，不要出现真实品牌 logo，不要生成难以阅读的小字。
""".strip(),
    },
    {
        "filename": "gpt-autoregressive-generation.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "GPT 自回归生成图",
        "suggested_page": "docs/ch11-nlp/ch06-pretrained/03-gpt-series.md",
        "alt": "GPT 自回归生成图：模型根据已有上下文一步一步预测下一个 token。",
        "prompt": """
一张适合 GPT 课程的教学图，主题是“自回归下一词预测”。
画面表现模型只看已经出现的上下文，逐步预测下一个 token，生成结果像一串从左到右展开的文本链。
风格现代、清晰、帮助新手理解 GPT 为什么擅长生成，不要出现真实品牌 logo，不要生成乱码文字。
""".strip(),
    },
    {
        "filename": "rlhf-three-stage-loop.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "RLHF 三阶段流程图",
        "suggested_page": "docs/ch07-llm-principles/ch07-alignment/02-rlhf.md",
        "alt": "RLHF 三阶段流程图：监督微调、奖励模型和强化学习优化共同把模型行为调得更符合人类偏好。",
        "prompt": """
一张适合大模型对齐课程的流程图，主题是“RLHF 三阶段”。
画面表现监督微调、人工偏好比较训练奖励模型、再用强化学习优化模型输出，形成对齐闭环。
风格清晰、稳重、适合解释复杂训练流程，不要出现真实品牌 logo，不要生成难以阅读的小字。
""".strip(),
    },
    {
        "filename": "rag-evaluation-triangle.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "RAG 评估三角图",
        "suggested_page": "docs/ch08-rag/ch01-rag/07-rag-evaluation.md",
        "alt": "RAG 评估三角图：检索质量、答案忠实度和用户可用性共同决定系统效果。",
        "prompt": """
一张适合 RAG 评估课程的教学图，主题是“RAG 评估不是只看答案对不对”。
画面用三角结构表现检索命中、上下文相关性、答案忠实度、引用来源、用户可用性和延迟成本共同影响系统质量。
风格工程化、清晰、适合新人建立评估框架，不要出现真实品牌 logo，不要生成乱码文字。
""".strip(),
    },
    {
        "filename": "agent-memory-system.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Agent 记忆系统分层图",
        "suggested_page": "docs/ch09-agent/ch04-memory/01-memory-overview.md",
        "alt": "Agent 记忆系统分层图：短期上下文、长期记忆、情景记忆和程序记忆共同支持持续任务。",
        "prompt": """
一张适合 AI Agent 课程的系统结构图，主题是“Agent 记忆系统”。
画面表现短期上下文、长期知识库、用户偏好、情景记录、程序经验和检索机制如何共同支持 Agent 执行长期任务。
风格清晰、像智能体大脑剖面图，不要出现真实品牌 logo，不要生成难以阅读的小字。
""".strip(),
    },
    {
        "filename": "mlp-neuron-activation.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "神经元到 MLP 结构图",
        "suggested_page": "docs/ch06-deep-learning/ch01-nn-basics/01-neurons-activation.md",
        "alt": "神经元到 MLP 结构图：输入、权重、偏置、激活函数和多层连接组成可学习模型。",
        "prompt": """
一张适合深度学习入门课程的教学图，主题是“从单个神经元到多层感知机”。
画面表现输入特征经过权重和偏置汇总，再通过激活函数输出；右侧扩展成输入层、隐藏层、输出层的 MLP。
重点突出“线性组合 + 非线性激活 + 多层堆叠”这三个直觉，风格清晰、现代教学白板感。
不要出现真实品牌 logo，不要生成密集小字，不要生成乱码文字。
""".strip(),
    },
    {
        "filename": "optimizer-comparison.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "优化器路径对比图",
        "suggested_page": "docs/ch06-deep-learning/ch01-nn-basics/03-optimizers.md",
        "alt": "优化器路径对比图：SGD、Momentum 和 Adam 用不同下降轨迹寻找更低损失。",
        "prompt": """
一张适合深度学习优化器课程的教学图，主题是“SGD、Momentum、Adam 的下降路径差异”。
画面表现同一个损失地形上有三条不同颜色的下降轨迹：SGD 抖动前进，Momentum 带惯性更平滑，Adam 自适应调整步长。
构图要让新手一眼看懂“学习率和更新策略会改变训练路径”，风格清晰、带一点工程实验感。
不要出现真实品牌 logo，不要生成复杂公式，不要生成密集小字或乱码文字。
""".strip(),
    },
    {
        "filename": "regularization-overfitting-controls.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "正则化控制过拟合图",
        "suggested_page": "docs/ch06-deep-learning/ch01-nn-basics/04-regularization.md",
        "alt": "正则化控制过拟合图：Dropout、Weight Decay、数据增强和早停共同限制模型记死训练集。",
        "prompt": """
一张适合深度学习正则化课程的教学图，主题是“如何防止模型过拟合”。
画面左侧表现训练集被模型死记硬背，右侧用 Dropout、Weight Decay、数据增强、Early Stopping 四个工具把模型拉回更稳的泛化区域。
风格像模型训练诊断白板，清晰、直观、适合新手理解“不要只追训练集分数”。
不要出现真实品牌 logo，不要生成难以阅读的小字或乱码文字。
""".strip(),
    },
    {
        "filename": "pytorch-autograd-graph.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "PyTorch Autograd 计算图",
        "suggested_page": "docs/ch06-deep-learning/ch02-pytorch/02-autograd.md",
        "alt": "PyTorch Autograd 计算图：张量运算形成计算图，loss.backward 沿图反向计算梯度。",
        "prompt": """
一张适合 PyTorch 自动求导课程的教学图，主题是“Autograd 计算图”。
画面表现多个 tensor 运算节点从输入一路组成 loss，调用 backward 后梯度沿着计算图反向流回需要更新的参数。
突出 requires_grad、loss、backward、grad 四个概念之间的关系，风格工程化、清晰、适合新人理解。
不要出现真实品牌 logo，不要生成密集代码或乱码文字。
""".strip(),
    },
    {
        "filename": "dataset-dataloader-batch-flow.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Dataset DataLoader Batch 流程图",
        "suggested_page": "docs/ch06-deep-learning/ch02-pytorch/04-data-loading.md",
        "alt": "Dataset DataLoader Batch 流程图：原始样本经过 Dataset、Sampler、DataLoader 组成可训练 batch。",
        "prompt": """
一张适合 PyTorch 数据加载课程的教学图，主题是“Dataset 到 DataLoader 再到 Batch”。
画面表现原始图片或文本样本进入 Dataset，经过索引、变换、打乱、分批，最终输出一个 batch 给训练循环。
重点突出 batch 维度、shuffle、transform、collate 的角色，风格清晰、像工程流水线。
不要出现真实品牌 logo，不要生成密集小字或乱码文字。
""".strip(),
    },
    {
        "filename": "cnn-feature-map-pipeline.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "CNN 特征图流水线",
        "suggested_page": "docs/ch06-deep-learning/ch03-cnn/02-cnn-structure.md",
        "alt": "CNN 特征图流水线：图像经过卷积、激活、池化和全连接层逐步提取局部到整体特征。",
        "prompt": """
一张适合 CNN 结构课程的教学图，主题是“CNN 从像素到类别的特征提取流水线”。
画面表现输入图像经过卷积层、激活、池化，多层后得到更抽象的特征图，最后进入分类头输出类别。
强调浅层看边缘纹理、深层看局部结构和整体语义，风格清晰、层次分明。
不要出现真实品牌 logo，不要生成复杂小字或乱码文字。
""".strip(),
    },
    {
        "filename": "rnn-unrolled-hidden-state.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "RNN 时间展开隐藏状态图",
        "suggested_page": "docs/ch06-deep-learning/ch04-rnn/01-rnn-basics.md",
        "alt": "RNN 时间展开隐藏状态图：序列逐步输入，同一单元反复使用隐藏状态传递上下文。",
        "prompt": """
一张适合 RNN 入门课程的教学图，主题是“RNN 在时间上展开”。
画面表现一个 RNN 单元沿时间轴复制展开，输入 token 或时间点逐步进入，隐藏状态像记忆接力棒一样从前一步传到后一步。
重点突出共享参数、隐藏状态、many-to-one 和 many-to-many 的直觉，风格清晰、适合新手。
不要出现真实品牌 logo，不要生成密集小字或乱码文字。
""".strip(),
    },
    {
        "filename": "gan-adversarial-loop.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "GAN 生成器判别器对抗图",
        "suggested_page": "docs/ch06-deep-learning/ch06-generative/01-gan.md",
        "alt": "GAN 生成器判别器对抗图：生成器尝试造假样本，判别器尝试分辨真假，双方一起进步。",
        "prompt": """
一张适合生成模型课程的教学图，主题是“GAN 的生成器与判别器博弈”。
画面表现生成器从随机噪声制造样本，判别器同时看到真实样本和生成样本并判断真假，反馈信号推动生成器改进。
用“造假者与鉴别师”的直觉表达对抗训练，但保持专业技术课程风格。
不要出现真实品牌 logo，不要生成密集文字或乱码文字。
""".strip(),
    },
    {
        "filename": "vae-latent-space-flow.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "VAE 潜空间生成流程图",
        "suggested_page": "docs/ch06-deep-learning/ch06-generative/02-vae.md",
        "alt": "VAE 潜空间生成流程图：编码器把输入压到分布化潜空间，采样后由解码器重建或生成样本。",
        "prompt": """
一张适合 VAE 入门课程的教学图，主题是“编码器、潜空间、采样、解码器”。
画面表现输入样本经过编码器变成均值和方差描述的潜空间分布，从潜空间采样后再由解码器生成或重建样本。
突出潜空间连续、可采样、可插值的直觉，风格清晰、柔和、适合新人理解。
不要出现真实品牌 logo，不要生成复杂公式或乱码文字。
""".strip(),
    },
    {
        "filename": "training-curve-diagnosis.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "训练曲线诊断图",
        "suggested_page": "docs/ch06-deep-learning/ch07-training-tips/02-training-diagnosis.md",
        "alt": "训练曲线诊断图：欠拟合、过拟合、学习率过大和正常收敛对应不同 loss 曲线形态。",
        "prompt": """
一张适合深度学习训练诊断课程的教学图，主题是“从 loss 曲线看训练问题”。
画面用四个并列小面板表现正常收敛、欠拟合、过拟合、学习率过大震荡这些典型训练曲线，并用颜色区分训练集和验证集。
风格像实验记录看板，清晰、准确、帮助新手把曲线和问题对应起来。
不要出现真实品牌 logo，不要生成密集小字或乱码文字。
""".strip(),
    },
    {
        "filename": "tokenizer-subword-flow.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Tokenizer 子词切分流程图",
        "suggested_page": "docs/ch07-llm-principles/ch01-nlp-crash/01-tokenizer.md",
        "alt": "Tokenizer 子词切分流程图：原始文本被清洗、切成 token、映射为 id，再进入模型。",
        "prompt": """
一张适合大模型入门课程的教学图，主题是“Tokenizer 如何把文本变成模型能读的 token id”。
画面表现原始句子经过规范化、子词切分、词表查找，变成 token 序列和数字 id，最后进入 embedding 层。
重点突出 token 不等于汉字或单词、切分会影响上下文长度和成本，风格清晰、现代。
不要出现真实品牌 logo，不要生成密集小字或乱码文字。
""".strip(),
    },
    {
        "filename": "embedding-semantic-space.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Embedding 语义空间图",
        "suggested_page": "docs/ch07-llm-principles/ch01-nlp-crash/02-embeddings.md",
        "alt": "Embedding 语义空间图：token 被映射到向量空间，语义相近的表示在空间中更接近。",
        "prompt": """
一张适合大模型 embedding 课程的教学图，主题是“语义空间里的 token 向量”。
画面表现 token id 经过 embedding 表变成向量点，语义相近的点聚在一起，不同方向代表不同语义维度。
强调“模型不是直接理解文字，而是在向量空间里计算关系”，风格清晰、适合新手建立直觉。
不要出现真实品牌 logo，不要生成乱码文字。
""".strip(),
    },
    {
        "filename": "llm-history-timeline.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "大模型发展时间线图",
        "suggested_page": "docs/ch07-llm-principles/ch02-llm-overview/01-development-history.md",
        "alt": "大模型发展时间线图：从统计语言模型、词向量、Transformer 到指令微调和多模态模型逐步演进。",
        "prompt": """
一张适合大模型发展史课程的时间线插图，主题是“从统计 NLP 到大模型时代”。
画面用时间轴串起统计语言模型、词向量、Seq2Seq、Transformer、预训练模型、指令微调、RAG 与 Agent、多模态等里程碑。
风格像技术史展板，清晰、有故事感，帮助新人把技术放进历史脉络。
不要出现真实品牌 logo，不要生成密集小字或乱码文字。
""".strip(),
    },
    {
        "filename": "pretraining-data-pipeline.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "预训练数据流水线图",
        "suggested_page": "docs/ch07-llm-principles/ch04-pretraining/01-pretraining-data.md",
        "alt": "预训练数据流水线图：原始网页、书籍、代码和论文经过清洗、去重、过滤、切分后进入训练集。",
        "prompt": """
一张适合大模型预训练课程的工程流程图，主题是“预训练数据从原始资料到训练样本”。
画面表现网页、书籍、代码、论文等资料进入清洗、去重、质量过滤、隐私安全过滤、tokenization、切分打包，最后进入训练集。
风格工程化、清晰、适合新手理解“数据质量决定模型底座质量”。
不要出现真实品牌 logo，不要生成难以阅读的小字或乱码文字。
""".strip(),
    },
    {
        "filename": "finetuning-alignment-pipeline.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "微调与对齐总流程图",
        "suggested_page": "docs/ch07-llm-principles/ch06-finetuning/01-finetuning-overview.md",
        "alt": "微调与对齐总流程图：基础模型经过 SFT、LoRA、偏好数据和对齐评估变成更适合任务的助手。",
        "prompt": """
一张适合大模型微调课程的总流程图，主题是“从基础模型到领域助手”。
画面表现基础模型经过任务数据准备、SFT、LoRA 或 QLoRA、偏好数据、对齐评估、上线监控，逐步变成适合业务场景的模型。
重点突出“不是只训练一次，而是数据、训练、评估、反馈的循环”，风格清晰、专业。
不要出现真实品牌 logo，不要生成密集小字或乱码文字。
""".strip(),
    },
    {
        "filename": "document-processing-vectorization.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "文档解析与向量化流程图",
        "suggested_page": "docs/ch08-rag/ch01-rag/02-document-processing.md",
        "alt": "文档解析与向量化流程图：PDF、Word、PPT 经过解析、清洗、切块、embedding 后写入向量库。",
        "prompt": """
一张适合 RAG 文档处理课程的流程图，主题是“文档解析到向量化”。
画面表现 PDF、Word、PPT、网页被解析成文本和结构，经过清洗、切块、元数据标注、embedding，最后写入向量库。
重点突出 chunk、metadata、embedding 三个新人最容易混的概念，风格工程化、清晰。
不要出现真实品牌 logo，不要生成密集小字或乱码文字。
""".strip(),
    },
    {
        "filename": "vector-database-similarity-search.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "向量数据库相似度检索图",
        "suggested_page": "docs/ch08-rag/ch01-rag/03-vector-databases.md",
        "alt": "向量数据库相似度检索图：查询向量在向量空间中寻找最接近的文档片段。",
        "prompt": """
一张适合向量数据库课程的教学图，主题是“相似度检索如何找到相关文档片段”。
画面表现用户问题变成查询向量，在向量空间中寻找距离最近的 chunk 点，并返回 Top-K 结果给 RAG 系统。
重点突出向量空间、距离、Top-K、metadata filter 的关系，风格清晰、现代。
不要出现真实品牌 logo，不要生成乱码文字。
""".strip(),
    },
    {
        "filename": "hybrid-search-rerank-flow.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Hybrid Search 与 Rerank 流程图",
        "suggested_page": "docs/ch08-rag/ch01-rag/04-retrieval-strategies.md",
        "alt": "Hybrid Search 与 Rerank 流程图：关键词检索和向量检索召回候选，再由重排模型筛选上下文。",
        "prompt": """
一张适合 RAG 检索策略课程的流程图，主题是“Hybrid Search + Rerank”。
画面表现同一个问题同时走关键词检索和向量检索，合并候选文档后进入 reranker 重排，最后选择最相关片段进入上下文。
重点突出召回和精排的分工，风格工程化、清晰、适合新手理解检索优化。
不要出现真实品牌 logo，不要生成密集小字或乱码文字。
""".strip(),
    },
    {
        "filename": "function-calling-workflow.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Function Calling 工作流图",
        "suggested_page": "docs/ch08-rag/ch03-app-dev/03-function-calling.md",
        "alt": "Function Calling 工作流图：模型根据工具 schema 选择函数、生成参数、执行工具并整合结果。",
        "prompt": """
一张适合大模型应用开发课程的流程图，主题是“Function Calling 如何让模型调用工具”。
画面表现用户请求进入模型，模型根据工具 schema 选择函数并生成结构化参数，应用执行函数，返回观察结果，再由模型整合最终回答。
重点突出 schema、参数校验、工具结果和最终回复之间的边界，风格清晰、工程化。
不要出现真实品牌 logo，不要生成复杂小字或乱码文字。
""".strip(),
    },
    {
        "filename": "template-doc-generation-pipeline.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Word PPT 模板生成流水线图",
        "suggested_page": "docs/ch08-rag/ch03-app-dev/08-template-doc-generation.md",
        "alt": "Word PPT 模板生成流水线图：结构化内容经过模板变量、版式校验和资源插入后导出文档。",
        "prompt": """
一张适合大模型文档生成课程的工程流程图，主题是“结构化内容如何生成 Word 或 PPT”。
画面表现主题和资料先变成大纲、段落、例题和图表资源，再进入模板变量填充、样式校验、引用检查，最后导出 Word 或 PPT 文件。
重点突出结构化 JSON、模板、资源、校验、导出五个环节，风格清晰、像产品工作流。
不要出现真实品牌 logo，不要生成密集小字或乱码文字。
""".strip(),
    },
    {
        "filename": "agent-vs-chatbot-comparison.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Agent 与普通 Chatbot 对比图",
        "suggested_page": "docs/ch09-agent/ch01-agent-basics/01-what-is-agent.md",
        "alt": "Agent 与普通 Chatbot 对比图：聊天机器人主要回答，Agent 会规划、调用工具、观察结果并推进任务。",
        "prompt": """
一张适合 AI Agent 入门课程的对比图，主题是“普通 Chatbot 和 Agent 的区别”。
画面左侧表现聊天机器人接收问题并回答，右侧表现 Agent 有目标、计划、工具调用、观察结果、记忆和最终交付物。
重点突出 Agent 更像“会行动的任务执行系统”，风格清晰、友好、适合新人建立边界。
不要出现真实品牌 logo，不要生成密集小字或乱码文字。
""".strip(),
    },
    {
        "filename": "agent-system-architecture.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Agent 系统架构图",
        "suggested_page": "docs/ch09-agent/ch01-agent-basics/04-system-architecture.md",
        "alt": "Agent 系统架构图：目标、规划器、模型、工具、记忆、环境和评估模块共同组成执行闭环。",
        "prompt": """
一张适合 AI Agent 架构课程的系统图，主题是“一个 Agent 系统由哪些模块组成”。
画面表现用户目标进入规划器和模型，大模型连接工具、记忆、环境观察、执行日志、评估与安全护栏，形成持续执行闭环。
重点突出模块边界和数据流，风格专业、清晰、像系统架构白板。
不要出现真实品牌 logo，不要生成密集小字或乱码文字。
""".strip(),
    },
    {
        "filename": "react-reason-act-observe-loop.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "ReAct 推理行动观察循环图",
        "suggested_page": "docs/ch09-agent/ch02-reasoning/03-react.md",
        "alt": "ReAct 推理行动观察循环图：模型在思考、行动、观察之间循环，逐步接近任务答案。",
        "prompt": """
一张适合 Agent 推理课程的流程图，主题是“ReAct：Reason、Act、Observe 的循环”。
画面表现 Agent 根据任务先推理下一步，再调用工具或执行动作，读取观察结果后更新判断并进入下一轮，直到完成任务。
重点突出循环、可回放轨迹和工具观察，风格清晰、像任务执行仪表盘。
不要出现真实品牌 logo，不要生成密集小字或乱码文字。
""".strip(),
    },
    {
        "filename": "mcp-host-client-server.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "MCP Host Client Server 架构图",
        "suggested_page": "docs/ch09-agent/ch05-mcp/02-mcp-architecture.md",
        "alt": "MCP Host Client Server 架构图：Host 通过 Client 连接多个 Server，统一发现资源、工具和提示能力。",
        "prompt": """
一张适合 MCP 协议课程的系统架构图，主题是“MCP Host、Client、Server 如何协作”。
画面表现应用 Host 内部有 MCP Client，连接多个 MCP Server，每个 Server 提供工具、资源、提示模板，Agent 通过协议统一发现和调用能力。
重点突出协议边界、连接关系和能力发现，风格工程化、清晰。
不要出现真实品牌 logo，不要生成难以阅读的小字或乱码文字。
""".strip(),
    },
    {
        "filename": "multi-agent-message-flow.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "多 Agent 协作消息流图",
        "suggested_page": "docs/ch09-agent/ch07-multi-agent/01-architecture-patterns.md",
        "alt": "多 Agent 协作消息流图：规划者、研究者、执行者和评审者通过消息与共享状态协作完成任务。",
        "prompt": """
一张适合多 Agent 课程的协作架构图，主题是“多个 Agent 如何分工协作”。
画面表现规划者、研究者、执行者、评审者等角色围绕共享任务板和消息通道协作，任务被拆分、分派、执行、复核、合并。
重点突出角色分工、消息流、共享状态和冲突处理，风格清晰、像团队作战室。
不要出现真实品牌 logo，不要生成密集小字或乱码文字。
""".strip(),
    },
    {
        "filename": "cv-pixel-rgb-grid.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "像素网格与 RGB 通道图",
        "suggested_page": "docs/ch10-computer-vision/ch01-cv-basics/01-image-fundamentals.md",
        "alt": "像素网格与 RGB 通道图：图像由像素矩阵组成，RGB 三个通道共同表示颜色。",
        "prompt": """
一张适合计算机视觉入门课程的教学图，主题是“图像其实是像素矩阵和 RGB 通道”。
画面表现一张小图被放大成像素网格，再拆分为红、绿、蓝三个通道矩阵，并重新合成为彩色图像。
重点突出 height、width、channel 这三个维度，风格清晰、像现代教学白板，帮助新手理解图片如何变成张量。
不要出现真实品牌 logo，不要生成密集小字或乱码文字。
""".strip(),
    },
    {
        "filename": "cv-image-processing-pipeline.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "图像处理流水线图",
        "suggested_page": "docs/ch10-computer-vision/ch01-cv-basics/03-image-processing.md",
        "alt": "图像处理流水线图：原图经过灰度化、滤波、边缘检测和形态学操作得到更适合模型使用的特征。",
        "prompt": """
一张适合 OpenCV 和图像处理课程的流程图，主题是“从原图到可分析特征的处理流水线”。
画面表现原始图像依次经过灰度化、去噪滤波、阈值分割、边缘检测、形态学操作，最终得到轮廓或候选区域。
风格清晰、工程感强，像一条视觉预处理流水线，适合新手理解图像处理不是魔法而是步骤组合。
不要出现真实品牌 logo，不要生成难以阅读的小字或乱码文字。
""".strip(),
    },
    {
        "filename": "cv-data-augmentation-gallery.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "图像数据增强样例墙",
        "suggested_page": "docs/ch10-computer-vision/ch02-classification/01-data-augmentation.md",
        "alt": "图像数据增强样例墙：翻转、裁剪、旋转、颜色扰动和噪声让模型看到更多变化。",
        "prompt": """
一张适合图像分类课程的教学图，主题是“数据增强如何让模型见过更多变化”。
画面用样例墙展示同一张训练图片经过随机裁剪、水平翻转、旋转、颜色扰动、模糊、噪声等增强后的多个版本。
重点突出增强应该保持标签不变，同时提升泛化能力，风格清晰、直观、适合新人理解。
不要出现真实品牌 logo，不要生成密集小字或乱码文字。
""".strip(),
    },
    {
        "filename": "yolo-grid-detection-flow.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "YOLO 网格检测流程图",
        "suggested_page": "docs/ch10-computer-vision/ch03-detection/03-yolo-series.md",
        "alt": "YOLO 网格检测流程图：图片被划分为网格，每个位置预测候选框、类别和置信度，再经过 NMS 得到结果。",
        "prompt": """
一张适合目标检测课程的教学图，主题是“YOLO 如何一次看完整张图并预测检测框”。
画面表现输入图片被划分成网格，每个网格预测多个候选框、类别和置信度，最后通过 NMS 去掉重复框得到最终检测结果。
重点突出 one-stage、grid、confidence、NMS 的直觉，风格清晰、现代、适合新手理解。
不要出现真实品牌 logo，不要生成难以阅读的小字或乱码文字。
""".strip(),
    },
    {
        "filename": "ocr-layout-recognition-pipeline.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "OCR 版面解析与文字识别流程图",
        "suggested_page": "docs/ch10-computer-vision/ch05-advanced/03-ocr.md",
        "alt": "OCR 版面解析与文字识别流程图：文档图像经过版面检测、文本框定位、识别和结构化输出。",
        "prompt": """
一张适合 OCR 课程的工程流程图，主题是“从文档图片到结构化文字”。
画面表现扫描件或截图先经过版面分析，识别标题、段落、表格和图片区域，再做文本框检测、文字识别、阅读顺序恢复，最终输出结构化文本。
风格清晰、像文档智能处理系统架构图，适合新手理解 OCR 不只是识字。
不要出现真实品牌 logo，不要生成密集小字或乱码文字。
""".strip(),
    },
    {
        "filename": "text-preprocessing-pipeline.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "文本预处理流水线图",
        "suggested_page": "docs/ch11-nlp/ch01-text-basics/02-text-preprocessing.md",
        "alt": "文本预处理流水线图：原始文本经过清洗、分词、标准化、去停用词和编码后进入模型。",
        "prompt": """
一张适合 NLP 入门课程的教学流程图，主题是“文本预处理从脏文本到模型输入”。
画面表现原始文本包含噪声、HTML、标点、大小写或繁简差异，经过清洗、分词、标准化、过滤、编码，变成可训练的 token 或特征。
重点突出不同任务不一定需要同样的清洗策略，风格清爽、适合新人建立流程感。
不要出现真实品牌 logo，不要生成密集小字或乱码文字。
""".strip(),
    },
    {
        "filename": "bow-tfidf-representation.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "BoW 与 TF-IDF 文本表示图",
        "suggested_page": "docs/ch11-nlp/ch01-text-basics/03-text-representation.md",
        "alt": "BoW 与 TF-IDF 文本表示图：文本被转成词频向量，TF-IDF 会降低常见词权重并突出关键词。",
        "prompt": """
一张适合 NLP 文本表示课程的教学图，主题是“BoW 和 TF-IDF 如何把文本变成向量”。
画面表现多篇短文本先形成词表，再变成词频表格；旁边展示 TF-IDF 把高频但无区分度的词降权，把关键词高亮。
风格像数据表和向量空间结合的白板图，清晰、适合新手理解传统文本特征。
不要出现真实品牌 logo，不要生成难以阅读的小字或乱码文字。
""".strip(),
    },
    {
        "filename": "contextual-embedding-comparison.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "上下文词嵌入对比图",
        "suggested_page": "docs/ch11-nlp/ch02-embeddings/02-contextual-embedding.md",
        "alt": "上下文词嵌入对比图：同一个词在不同句子里会得到不同表示，帮助模型理解语境。",
        "prompt": """
一张适合 NLP 表示学习课程的教学图，主题是“上下文词嵌入为什么比静态词向量更灵活”。
画面表现同一个词出现在两个不同句子中，静态词向量只有一个固定点，而上下文模型会根据前后文生成不同位置的表示。
重点突出“一词多义”和“上下文决定表示”，风格清晰、现代、适合新手理解 BERT 类模型。
不要出现真实品牌 logo，不要生成密集小字或乱码文字。
""".strip(),
    },
    {
        "filename": "text-classification-pipeline.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "文本分类训练流程图",
        "suggested_page": "docs/ch11-nlp/ch03-classification/03-classification-practice.md",
        "alt": "文本分类训练流程图：文本经过清洗、编码、模型、概率输出和错误分析形成分类项目闭环。",
        "prompt": """
一张适合文本分类实战课程的项目流程图，主题是“从文本样本到分类模型”。
画面表现文本数据经过标注、清洗、编码，进入传统模型或深度模型，输出类别概率，再通过混淆矩阵和错误样本分析改进。
风格工程化、清晰、像 NLP 项目看板，帮助新手理解分类不只是调用模型。
不要出现真实品牌 logo，不要生成密集小字或乱码文字。
""".strip(),
    },
    {
        "filename": "seq2seq-attention-alignment.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Seq2Seq 注意力对齐图",
        "suggested_page": "docs/ch11-nlp/ch05-seq2seq/02-attention-in-nlp.md",
        "alt": "Seq2Seq 注意力对齐图：解码器每生成一个词都会关注输入序列中不同位置。",
        "prompt": """
一张适合 Seq2Seq 与注意力课程的教学图，主题是“注意力如何在翻译中对齐输入和输出”。
画面表现编码器读取源语言序列，解码器逐步生成目标语言，每一步通过注意力权重连接到源序列不同 token，并形成一张对齐热力图。
重点突出 attention 让模型不用只依赖一个固定上下文向量，风格清晰、适合新手理解。
不要出现真实品牌 logo，不要生成难以阅读的小字或乱码文字。
""".strip(),
    },
    {
        "filename": "multimodal-alignment-fusion.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "多模态对齐与融合图",
        "suggested_page": "docs/ch12-multimodal/ch01-multimodal/01-multimodal-basics.md",
        "alt": "多模态对齐与融合图：文本、图像、语音和视频被编码到可共同推理的表示空间。",
        "prompt": """
一张适合多模态入门课程的教学图，主题是“文本、图像、语音、视频如何被对齐到同一个理解空间”。
画面表现四种输入模态分别经过编码器，映射到共享表示空间，再由融合模块完成问答、检索、生成或决策。
重点突出对齐、融合、跨模态检索和联合推理，风格现代、清晰、适合新人建立整体框架。
不要出现真实品牌 logo，不要生成密集小字或乱码文字。
""".strip(),
    },
    {
        "filename": "vision-language-model-architecture.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "视觉语言模型架构图",
        "suggested_page": "docs/ch12-multimodal/ch01-multimodal/02-vision-language.md",
        "alt": "视觉语言模型架构图：图像编码器提取视觉特征，语言模型结合文本指令生成回答。",
        "prompt": """
一张适合视觉语言模型课程的系统结构图，主题是“图片如何进入语言模型并参与回答”。
画面表现图像被切成 patch 或特征，进入视觉编码器，再通过投影层连接到语言模型，结合用户文本问题生成多模态回答。
重点突出 image encoder、projector、LLM、text prompt 的连接关系，风格工程化、清晰。
不要出现真实品牌 logo，不要生成密集小字或乱码文字。
""".strip(),
    },
    {
        "filename": "stable-diffusion-components.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Stable Diffusion 组件架构图",
        "suggested_page": "docs/ch12-multimodal/ch02-image-gen/02-stable-diffusion.md",
        "alt": "Stable Diffusion 组件架构图：文本编码器、U-Net、VAE 和 Scheduler 协作完成图像生成。",
        "prompt": """
一张适合 Stable Diffusion 课程的组件架构图，主题是“文本到图像生成系统由哪些模块组成”。
画面表现 prompt 进入文本编码器，噪声 latent 在 U-Net 和 Scheduler 的多步去噪中变化，最后由 VAE 解码成图像。
重点突出 text encoder、latent、U-Net、scheduler、VAE 的分工，风格清晰、专业、适合新人拆解架构。
不要出现真实品牌 logo，不要生成复杂公式或乱码文字。
""".strip(),
    },
    {
        "filename": "video-audio-generation-pipeline.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "视频与语音生成流水线图",
        "suggested_page": "docs/ch12-multimodal/ch03-video-gen/01-video-generation.md",
        "alt": "视频与语音生成流水线图：脚本、分镜、关键帧、视频生成、TTS 和后期合成组成内容生产流程。",
        "prompt": """
一张适合 AIGC 视频课程的生产流程图，主题是“从文案到视频和配音的生成流水线”。
画面表现脚本生成、分镜设计、关键帧或参考图、视频生成、语音合成 TTS、字幕和后期合成，最终输出短视频。
重点突出视频不是单次生成，而是多资产、多步骤协同的工作流，风格清晰、像创意生产看板。
不要出现真实品牌 logo，不要生成密集小字或乱码文字。
""".strip(),
    },
    {
        "filename": "creative-platform-workflow.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "AI 创意内容平台工作流图",
        "suggested_page": "docs/ch12-multimodal/ch05-projects/01-creative-platform.md",
        "alt": "AI 创意内容平台工作流图：用户需求经过素材管理、生成、审核、版本管理和导出形成完整产品闭环。",
        "prompt": """
一张适合 AIGC 综合项目课程的产品架构图，主题是“AI 创意内容平台的完整工作流”。
画面表现用户输入创意需求，系统管理文本、图片、音频、视频素材，调用生成模型，经过人工审核、版权与安全检查、版本管理，最后导出作品包。
重点突出从玩具 demo 到可交付产品需要资产管理、审核和导出闭环，风格专业、清晰、产品感强。
不要出现真实品牌 logo，不要生成密集小字或乱码文字。
""".strip(),
    },
    {
        "filename": "elective-cpp-runtime-memory.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "C++ 运行与内存模型图",
        "suggested_page": "docs/electives/module-a/01-cpp-basics.md",
        "alt": "C++ 运行与内存模型图：源码经过编译生成程序，栈、堆、对象、引用共同影响部署代码行为。",
        "prompt": """
一张适合 C++ 与模型部署选修课的教学图，主题是“C++ 从源码到运行时内存”。
画面表现 C++ 源码经过编译器生成可执行程序，运行时有栈、堆、对象、引用、vector 和函数调用之间的关系。
重点突出 Python 背景学习者最容易卡住的编译、类型、对象生命周期和拷贝成本，风格清晰、工程化。
不要出现真实品牌 logo，不要生成密集小字或乱码文字。
""".strip(),
    },
    {
        "filename": "elective-model-optimization-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "模型优化路线图",
        "suggested_page": "docs/electives/module-a/03-model-optimization.md",
        "alt": "模型优化路线图：量化、剪枝、蒸馏、算子融合和批处理分别优化大小、速度、成本和稳定性。",
        "prompt": """
一张适合模型部署课程的教学路线图，主题是“模型优化不是只有压缩模型”。
画面表现五条优化路线：量化、剪枝、蒸馏、算子融合、批处理与调度，并分别连接到模型大小、延迟、吞吐、成本和精度影响。
重点突出优化需要看瓶颈、指标和业务约束，风格专业、清晰、像工程决策白板。
不要出现真实品牌 logo，不要生成密集小字或乱码文字。
""".strip(),
    },
    {
        "filename": "elective-inference-engine-hardware.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "推理引擎与硬件适配图",
        "suggested_page": "docs/electives/module-a/04-inference-engines.md",
        "alt": "推理引擎与硬件适配图：模型通过推理引擎适配 CPU、GPU、NPU 和边缘设备，影响延迟与吞吐。",
        "prompt": """
一张适合推理引擎课程的系统图，主题是“模型、推理引擎和硬件之间的适配关系”。
画面表现训练好的模型经过格式转换和图优化，进入 ONNX Runtime、TensorRT、OpenVINO 等推理引擎，再适配 CPU、GPU、NPU 和边缘设备。
重点突出没有绝对最强引擎，选择取决于硬件、延迟、吞吐、维护复杂度，风格工程化、清晰。
不要出现真实品牌 logo，不要生成密集小字或乱码文字。
""".strip(),
    },
    {
        "filename": "elective-model-serving-architecture.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "模型服务化架构图",
        "suggested_page": "docs/electives/module-a/06-model-serving.md",
        "alt": "模型服务化架构图：请求入口、队列、批处理、模型执行器、版本路由和监控组成线上推理系统。",
        "prompt": """
一张适合模型服务化课程的架构图，主题是“线上模型服务不只是包一层 API”。
画面表现请求入口、鉴权限流、任务队列、动态批处理器、模型执行器、版本路由、健康检查、日志监控和指标看板。
重点突出延迟、吞吐、错误率、批处理效率和模型版本管理，风格清晰、生产系统架构感。
不要出现真实品牌 logo，不要生成密集小字或乱码文字。
""".strip(),
    },
    {
        "filename": "elective-python-decorator-flow.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Python 装饰器执行流程图",
        "suggested_page": "docs/electives/module-b/01-decorators-advanced.md",
        "alt": "Python 装饰器执行流程图：装饰器把日志、计时、重试等横切逻辑包裹在原函数外层。",
        "prompt": """
一张适合 Python 进阶课程的教学图，主题是“装饰器如何包裹函数执行”。
画面表现原函数被日志、计时、重试、权限检查等装饰器逐层包裹，请求进入外层包装函数，再调用原函数并返回结果。
重点突出 wraps、带参数装饰器、横切逻辑复用和过度嵌套风险，风格清爽、工程感强。
不要出现真实品牌 logo，不要生成密集代码或乱码文字。
""".strip(),
    },
    {
        "filename": "elective-asyncio-concurrency-control.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "asyncio 并发控制流程图",
        "suggested_page": "docs/electives/module-b/03-concurrency.md",
        "alt": "asyncio 并发控制流程图：多个异步任务通过事件循环、并发上限、超时和取消机制稳定执行。",
        "prompt": """
一张适合 Python asyncio 课程的教学图，主题是“异步并发不是越多越好”。
画面表现事件循环调度多个 I/O 任务，使用 semaphore 控制并发上限，用 timeout 和 cancellation 处理慢任务，同时保护上游 API。
重点突出等待、并发、限流、超时、取消和错误收集的关系，风格清晰、像工程运行时仪表盘。
不要出现真实品牌 logo，不要生成密集小字或乱码文字。
""".strip(),
    },
    {
        "filename": "elective-svm-margin-support-vectors.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "SVM 最大间隔与支持向量图",
        "suggested_page": "docs/electives/module-c/01-svm.md",
        "alt": "SVM 最大间隔与支持向量图：支持向量决定分界线，最大间隔让分类边界更稳。",
        "prompt": """
一张适合经典机器学习课程的教学图，主题是“SVM 找最大间隔分类边界”。
画面表现两类样本点、分界超平面、两侧 margin 和贴在边界附近的支持向量；旁边用曲线投影暗示核技巧把非线性问题变得可分。
重点突出支持向量、最大间隔、特征缩放和核方法直觉，风格清晰、数学直觉强。
不要出现真实品牌 logo，不要生成复杂公式或乱码文字。
""".strip(),
    },
    {
        "filename": "elective-knn-neighbor-voting.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "KNN 邻居投票图",
        "suggested_page": "docs/electives/module-c/02-knn.md",
        "alt": "KNN 邻居投票图：新样本根据距离找到最近 K 个邻居，用多数投票决定类别。",
        "prompt": """
一张适合 KNN 课程的教学图，主题是“邻居投票如何决定新样本类别”。
画面表现一个新样本点周围有不同类别的邻居，K=3 和 K=9 两个圆圈展示不同 K 值会改变投票结果。
重点突出距离度量、特征缩放、K 值大小和计算成本，风格直观、清晰、适合新人理解。
不要出现真实品牌 logo，不要生成密集小字或乱码文字。
""".strip(),
    },
    {
        "filename": "elective-naive-bayes-evidence.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "朴素贝叶斯证据累积图",
        "suggested_page": "docs/electives/module-c/03-naive-bayes.md",
        "alt": "朴素贝叶斯证据累积图：多个特征像证据一样累积，更新每个类别的后验概率。",
        "prompt": """
一张适合朴素贝叶斯课程的教学图，主题是“特征证据如何累积成分类概率”。
画面表现一封文本或样本被拆成多个特征，每个特征像证据卡片一样给不同类别加权，最终形成后验概率柱状图。
重点突出先验、似然、后验、条件独立假设和文本分类直觉，风格像侦探证据板但保持专业。
不要出现真实品牌 logo，不要生成复杂公式或乱码文字。
""".strip(),
    },
    {
        "filename": "elective-ai-security-red-team-loop.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "AI 安全红队闭环图",
        "suggested_page": "docs/electives/module-d.md",
        "alt": "AI 安全红队闭环图：威胁建模、攻击样本、评估统计、修复和回归测试形成持续安全闭环。",
        "prompt": """
一张适合 AI 安全课程的流程图，主题是“红队测试和安全修复闭环”。
画面表现资产识别、攻击面分析、红队样本生成、模型和系统链路测试、失败模式统计、修复策略、回归测试集沉淀。
重点突出安全不是一次规则检查，而是持续评估和修复，风格稳重、系统安全架构感。
不要出现真实品牌 logo，不要生成密集小字或乱码文字。
""".strip(),
    },
    {
        "filename": "elective-ai-frontend-stack.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "AI 前端交互栈图",
        "suggested_page": "docs/electives/module-e.md",
        "alt": "AI 前端交互栈图：HTML、CSS、JavaScript、fetch、加载态和错误态共同把模型能力变成可体验产品。",
        "prompt": """
一张适合 AI 前端基础课程的教学图，主题是“AI 产品页面由哪些前端部分组成”。
画面表现 HTML 负责结构、CSS 负责外观、JavaScript 负责交互，fetch 调用后端或 AI API，并展示加载态、错误态、结果区和用户反馈按钮。
重点突出前端不只是装饰层，而是让模型能力可见、可控、可恢复的体验层，风格现代、清晰。
不要出现真实品牌 logo，不要生成密集小字或乱码文字。
""".strip(),
    },
    {
        "filename": "elective-ai-product-decision-matrix.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "AI 产品决策四象限图",
        "suggested_page": "docs/electives/module-f.md",
        "alt": "AI 产品决策四象限图：用户价值、成本、风险和体验共同决定一个 AI 功能是否值得做。",
        "prompt": """
一张适合 AI 产品设计课程的决策图，主题是“AI 产品功能值不值得做”。
画面用四象限或雷达图表现用户价值、实现成本、风险、可体验性四个维度，多个候选功能被放到评估板上排序。
重点突出不要从模型能力出发，而要从用户问题、成本、风险和体验闭环出发，风格产品策略感、清晰专业。
不要出现真实品牌 logo，不要生成密集小字或乱码文字。
""".strip(),
    },
    {
        "filename": "appendix-ai-milestones-timeline.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "AI 历史接力赛时间线图",
        "suggested_page": "docs/appendix/ai-milestones.md",
        "alt": "AI 历史接力赛时间线图：概率推断、神经网络、经典机器学习、深度学习、大模型、Agent 和多模态依次接力。",
        "prompt": """
一张适合 AI 课程附录的历史主线图，主题是“AI 历史像一场接力赛”。
画面从左到右表现概率推断、早期神经网络、经典机器学习、深度学习复兴、Transformer 与大模型、RAG 与 Agent、多模态与 AIGC 七个时代接力。
每个时代用一个清晰图标表达：概率骰子、神经元、分界线、GPU 训练实验室、注意力模块、工具调用控制台、图文音视频融合。
风格像高质量课程海报，层次清楚、适合新人建立历史感。
不要出现真实品牌 logo，不要生成密集小字或乱码文字。
""".strip(),
    },
    {
        "filename": "appendix-troubleshooting-rescue-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "学习卡点排障地图",
        "suggested_page": "docs/appendix/troubleshooting.md",
        "alt": "学习卡点排障地图：环境、代码、训练、显存、项目拆解和学习焦虑对应不同排查路径。",
        "prompt": """
一张适合 AI 学习附录的排障地图，主题是“学习卡点救援”。
画面表现一个学习者站在问题分叉路口，六条路径分别通向环境依赖、代码输入输出、训练不收敛、显存不足、项目无法落地、学习焦虑。
每条路径都有对应的工具图标：终端、放大镜、loss 曲线、显存仪表、最小闭环积木、复盘笔记。
风格温暖、鼓励、新手友好，像技术学习的救援地图。
不要出现真实品牌 logo，不要生成密集小字或乱码文字。
""".strip(),
    },
    {
        "filename": "appendix-hardware-cloud-decision-tree.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "硬件与云资源决策树",
        "suggested_page": "docs/appendix/hardware.md",
        "alt": "硬件与云资源决策树：先看学习阶段和任务，再决定本地电脑、云 GPU、API 或硬件升级。",
        "prompt": """
一张适合 AI 入门课程的硬件决策树插图，主题是“先判断任务，再决定是否买 GPU”。
画面表现从学习阶段和任务类型出发，分流到本地电脑、云 GPU、API 优先、本地硬件升级四种选择。
突出内存、硬盘、稳定环境、GPU 的优先级差异，并表现“先小实验跑通，再上云训练”的理性路径。
风格清晰、理性、科技感，不制造设备焦虑。
不要出现真实品牌 logo，不要生成密集小字或乱码文字。
""".strip(),
    },
    {
        "filename": "appendix-job-prep-funnel.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "求职准备漏斗图",
        "suggested_page": "docs/appendix/job-prep.md",
        "alt": "求职准备漏斗图：岗位定位、项目选择、README 打磨、简历表达、面试复盘逐步收敛。",
        "prompt": """
一张适合 AI 学习者求职附录的规划图，主题是“把学习成果整理成可投递作品”。
画面用漏斗或流水线表现：岗位定位、项目筛选、项目 README、效果截图、简历表达、面试复盘、持续改进。
突出 2 到 3 个能讲清楚的项目比堆课程名更重要，画面专业、鼓励、行动感强。
不要出现真实品牌 logo，不要生成密集小字或乱码文字。
""".strip(),
    },
    {
        "filename": "appendix-continuous-learning-flywheel.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "持续学习三层飞轮图",
        "suggested_page": "docs/appendix/continuous-learning.md",
        "alt": "持续学习三层飞轮图：基础学习、项目学习和前沿跟踪互相推动，形成长期学习节奏。",
        "prompt": """
一张适合 AI 持续学习方法论的飞轮图，主题是“基础、项目、前沿三层学习如何互相推动”。
画面表现三个互相连接的环：基础能力、项目实践、前沿跟踪；周围有每日推进、每周复盘、每月总结和知识库沉淀。
重点表达不要天天追热点，而是用项目和复盘把知识留下来。
风格清爽、稳定、长期主义，适合课程附录。
不要出现真实品牌 logo，不要生成密集小字或乱码文字。
""".strip(),
    },
    {
        "filename": "appendix-resource-selection-funnel.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "资源选择漏斗图",
        "suggested_page": "docs/appendix/resources.md",
        "alt": "资源选择漏斗图：先学主线课程，遇到卡点再补外部资源，最后回到项目验证。",
        "prompt": """
一张适合推荐学习资源附录的导航图，主题是“资源不是越多越好，而是服务当前卡点”。
画面用漏斗表现：主线课程优先、识别具体卡点、选择一类外部资源、回到代码或项目验证、写一句复盘。
可以用书本、视频、文档、代码编辑器和项目看板图标表达不同资源类型。
风格干净、降低信息焦虑，适合新人学习路线页面。
不要出现真实品牌 logo，不要生成密集小字或乱码文字。
""".strip(),
    },
    {
        "filename": "appendix-faq-decision-tree.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "FAQ 新人问题分流树",
        "suggested_page": "docs/appendix/faq.md",
        "alt": "FAQ 新人问题分流树：数学、GPU、阶段选择、学习时间、项目、论文和求职问题各自分流。",
        "prompt": """
一张适合 AI 新人常见问题页面的分流树插图，主题是“遇到犹豫时先判断问题类型”。
画面表现一个中心问号分出数学基础、GPU 设备、学习阶段、每周时间、项目启动、代码报错、论文阅读、求职准备等分支。
每个分支用直观图标表示，整体像友好的学习咨询台，帮助新人减少焦虑。
不要出现真实品牌 logo，不要生成密集小字或乱码文字。
""".strip(),
    },
    {
        "filename": "appendix-project-quick-reference-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "AI 项目速查总览图",
        "suggested_page": "docs/appendix/resource-quick-ref.md",
        "alt": "AI 项目速查总览图：环境命令、baseline、评估指标、训练信号、RAG、Agent 和 Prompt 串成项目检查地图。",
        "prompt": """
一张适合 AI 项目速查页的总览图，主题是“做项目时先查哪一块”。
画面像一张项目控制台地图，包含环境命令、baseline 选择、评估指标、训练信号、RAG 检查、Agent 工具安全、Prompt 输出格式七个区域。
强调快速定位、快速回查、回到正文深入学习，风格像专业工程看板，清晰但不拥挤。
不要出现真实品牌 logo，不要生成密集小字或乱码文字。
""".strip(),
    },
    {
        "filename": "appendix-course-numbering-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "展示章节号与源码目录对应地图",
        "suggested_page": "docs/appendix/course-numbering.md",
        "alt": "展示章节号与源码目录对应地图：网页第 1 到第 12 章分别对应 docs/ch01 到 docs/ch12 源码目录。",
        "prompt": """
一张适合课程维护附录的编号对应关系图，主题是“网页展示章节号和源码目录如何对应”。
画面表现左侧是学习者看到的第 1 到第 12 章课程地图，右侧是维护者看到的 docs/ch01 到 docs/ch12 目录树，中间用整齐连线对应。
强调主线分组不是目录层级，源码目录和展示章节号已经对齐。
风格清晰、文档工程感、适合课程维护者快速理解。
不要出现真实品牌 logo，不要生成密集小字或乱码文字。
""".strip(),
    },
    {
        "filename": "appendix-visual-enhancement-kanban.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "课程图片资产规划看板",
        "suggested_page": "docs/appendix/visual-enhancement-plan.md",
        "alt": "课程图片资产规划看板：P0、P1、P2、P3 不同优先级图片分批规划和生成。",
        "prompt": """
一张适合课程视觉增强规划页的资产看板图，主题是“课程图片应该按理解成本优先生成”。
画面表现一个图片资产规划看板，分为 P0 阶段首页、P1 核心概念和项目架构、P2 数学代码可视化和历史故事、P3 装饰型补充四列。
旁边有生成脚本、manifest、页面引用和质量检查的流程线，强调图片服务理解而不是单纯装饰。
风格专业、明亮、像课程内容生产控制台。
不要出现真实品牌 logo，不要生成密集小字或乱码文字。
""".strip(),
    },
    {
        "filename": "ch01-terminal-path-command-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "终端路径与命令执行关系图",
        "suggested_page": "docs/ch01-tools/ch01-terminal/02-basic-operations.md",
        "alt": "终端路径与命令执行关系图：当前目录、目录树、命令、参数和输出结果共同组成一次终端操作。",
        "prompt": """
一张适合命令行入门课程的教学图，主题是“终端操作其实是在目录树里执行命令”。
画面表现左侧是文件夹目录树，中间是终端窗口里的当前路径、命令、参数，右侧是命令执行后的文件变化或输出结果。
重点突出 cwd 当前目录、相对路径、绝对路径、命令、参数、输出之间的关系，帮助新人理解为什么同一条命令在不同目录结果不同。
风格清晰、友好、像开发者工作台，不要出现真实品牌 logo，不要生成密集小字或乱码文字。
""".strip(),
    },
    {
        "filename": "ch01-python-env-stack.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Python 环境与依赖关系图",
        "suggested_page": "docs/ch01-tools/ch03-devenv/01-python-env.md",
        "alt": "Python 环境与依赖关系图：系统 Python、虚拟环境、pip、requirements 和项目代码需要保持一致。",
        "prompt": """
一张适合 Python 环境管理课程的教学图，主题是“项目、解释器、虚拟环境和依赖要对齐”。
画面表现一个项目文件夹连接到虚拟环境，虚拟环境里有 Python 解释器和 pip 安装的依赖包，旁边有 requirements 文件作为依赖清单。
对比一个错误场景：包安装到了别的环境，IDE 和终端使用的解释器不一致。
风格像环境诊断仪表盘，清晰、实用、降低新人的环境焦虑。
不要出现真实品牌 logo，不要生成密集小字或乱码文字。
""".strip(),
    },
    {
        "filename": "ch01-git-branch-collaboration.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Git 分支协作流程图",
        "suggested_page": "docs/ch01-tools/ch02-git/04-branches.md",
        "alt": "Git 分支协作流程图：从 main 分出 feature 分支，提交、合并、解决冲突并同步远程仓库。",
        "prompt": """
一张适合 Git 入门课程的分支协作图，主题是“分支让多人协作不互相干扰”。
画面表现 main 主线、feature 分支、commit 节点、pull request 或合并点、冲突解决区和远程仓库同步。
重点突出先拉最新、开分支、提交、合并、处理冲突、push/pull 的顺序关系。
风格现代、清晰、适合新人理解 Git 分支不是抽象概念，而是协作安全通道。
不要出现真实品牌 logo，不要生成密集小字或乱码文字。
""".strip(),
    },
    {
        "filename": "ch01-jupyter-kernel-state.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Jupyter Cell 与 Kernel 状态图",
        "suggested_page": "docs/ch01-tools/ch03-devenv/03-jupyter.md",
        "alt": "Jupyter Cell 与 Kernel 状态图：多个 Cell 共享同一个 Kernel，执行顺序会影响变量状态和图表输出。",
        "prompt": """
一张适合 Jupyter Notebook 入门课程的教学图，主题是“Cell 顺序和 Kernel 状态会影响运行结果”。
画面表现多个 Notebook cell 依次或乱序执行，所有 cell 共享同一个 kernel 内存状态，变量、DataFrame、图表输出会随着执行顺序变化。
加入重启 kernel、清空输出、重新运行全部 cell 的流程提示，用视觉方式解释为什么 Notebook 有时看起来很神秘。
风格清爽、教育插图、像数据实验室工作台。
不要出现真实品牌 logo，不要生成密集小字或乱码文字。
""".strip(),
    },
    {
        "filename": "ch02-variable-object-reference.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "变量、对象与引用关系图",
        "suggested_page": "docs/ch02-python/ch01-basics/02-data-types.md",
        "alt": "变量、对象与引用关系图：变量名像标签，指向内存中的对象，对象有类型和值。",
        "prompt": """
一张适合 Python 新手课程的教学图，主题是“变量名不是盒子，而是指向对象的标签”。
画面表现变量名标签连接到内存中的对象卡片，对象卡片包含类型和值；同时展示两个变量指向同一个列表对象和重新赋值后指向新对象。
重点突出变量名、对象、类型、值、引用、可变对象和不可变对象的直觉区别。
风格温和、清晰、像白板课件，不要出现真实品牌 logo，不要生成密集小字或乱码文字。
""".strip(),
    },
    {
        "filename": "ch02-control-flow-paths.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Python 流程控制执行路径图",
        "suggested_page": "docs/ch02-python/ch01-basics/05-control-flow.md",
        "alt": "Python 流程控制执行路径图：if 负责分岔，for 负责遍历，while 负责条件循环，break 和 continue 改变循环路径。",
        "prompt": """
一张适合 Python 流程控制课程的教学图，主题是“程序像沿着路径前进”。
画面用路线图表现 if/elif/else 的分岔、for 循环遍历列表、while 根据条件重复执行，以及 break 和 continue 如何改变循环路径。
重点让新人理解执行顺序、条件判断、循环体、退出条件和死循环风险。
风格像程序流程游乐园地图，清晰、有趣但保持专业。
不要出现真实品牌 logo，不要生成密集小字或乱码文字。
""".strip(),
    },
    {
        "filename": "ch02-data-structures-comparison.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Python 数据结构对比图",
        "suggested_page": "docs/ch02-python/ch01-basics/06-data-structures.md",
        "alt": "Python 数据结构对比图：list、tuple、dict、set 分别适合顺序集合、不可变记录、键值映射和去重集合。",
        "prompt": """
一张适合 Python 数据结构课程的对比图，主题是“list、tuple、dict、set 各自适合什么场景”。
画面分成四个区域：list 像可增删的队列，tuple 像固定记录卡片，dict 像通过 key 查 value 的索引柜，set 像自动去重的集合圈。
用输入数据到结构选择的方式表达：要保顺序、要不可变、要按名字查、要去重分别选什么。
风格清晰、图标化、适合新人快速建立选择直觉。
不要出现真实品牌 logo，不要生成密集小字或乱码文字。
""".strip(),
    },
    {
        "filename": "ch02-function-call-scope.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "函数调用、参数与作用域图",
        "suggested_page": "docs/ch02-python/ch01-basics/07-functions.md",
        "alt": "函数调用、参数与作用域图：调用函数时参数进入局部作用域，return 把结果交回调用者。",
        "prompt": """
一张适合 Python 函数基础课程的教学图，主题是“函数像一台有输入和输出的小机器”。
画面表现调用者把参数传入函数，函数内部有局部作用域工作台，经过处理后用 return 把结果返回；旁边展示调用栈一层层进入和退出。
重点突出参数、返回值、局部变量、全局变量、作用域、调用栈的关系。
风格清晰、温和、像代码运行剖面图。
不要出现真实品牌 logo，不要生成密集小字或乱码文字。
""".strip(),
    },
    {
        "filename": "ch02-oop-class-object-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "类、对象、属性与方法关系图",
        "suggested_page": "docs/ch02-python/ch02-advanced/01-oop.md",
        "alt": "类、对象、属性与方法关系图：类像蓝图，对象是实例，属性保存状态，方法定义行为。",
        "prompt": """
一张适合 Python 面向对象课程的教学图，主题是“类是蓝图，对象是按蓝图造出来的实例”。
画面表现一个 Class 蓝图生成多个 Object 实例，每个对象有自己的属性状态，同时共享类定义的方法；可用任务、学生或机器人作为抽象示例。
重点突出 class、instance、attribute、method、self 的直觉关系，避免复杂继承细节。
风格像工程蓝图加教学白板，清楚、友好。
不要出现真实品牌 logo，不要生成密集小字或乱码文字。
""".strip(),
    },
    {
        "filename": "ch02-exception-flow.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "异常处理执行流程图",
        "suggested_page": "docs/ch02-python/ch02-advanced/02-exceptions.md",
        "alt": "异常处理执行流程图：try 中出错会进入 except，没有出错可进入 else，finally 总会执行清理。",
        "prompt": """
一张适合 Python 异常处理课程的流程图，主题是“错误发生后程序可以有控制地恢复”。
画面表现 try 代码块正常执行或抛出异常，异常进入对应 except 分支，没有异常进入 else，最后 finally 负责关闭文件、释放资源等清理动作。
重点突出不要吞掉错误、要保留错误信息、清理资源和用户友好提示之间的关系。
风格像安全气囊和排障流程结合的技术图，清晰专业。
不要出现真实品牌 logo，不要生成密集小字或乱码文字。
""".strip(),
    },
    {
        "filename": "ch02-ai-api-request-response.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "AI API 请求响应链路图",
        "suggested_page": "docs/ch02-python/ch03-projects/04-ai-api-experience.md",
        "alt": "AI API 请求响应链路图：用户输入经过 Python 客户端发出请求，模型返回结果，程序处理错误、重试和展示。",
        "prompt": """
一张适合 Python AI API 项目课的系统流程图，主题是“一次 AI API 调用从输入到输出发生了什么”。
画面表现用户输入、Python 客户端、请求 payload、网络调用、AI 服务、响应结果、错误处理、重试、最终展示结果的链路。
重点突出 API key 保护、请求参数、超时、错误处理、日志和成本意识。
风格现代、产品工程感、适合新人第一次连接 AI 服务。
不要出现真实品牌 logo，不要生成密集小字或乱码文字。
""".strip(),
    },
    {
        "filename": "ch03-numpy-array-shape-axis.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "NumPy 数组 Shape 与 Axis 图",
        "suggested_page": "docs/ch03-data-analysis/ch02-numpy/02-array-basics.md",
        "alt": "NumPy 数组 Shape 与 Axis 图：一维、二维、三维数组通过 shape 和 axis 描述结构与运算方向。",
        "prompt": """
一张适合 NumPy 入门课程的教学图，主题是“ndarray 的 shape 和 axis 决定数据长什么样、沿哪个方向算”。
画面表现一维数组、二维表格、三维数据块，旁边用箭头标出 axis 0、axis 1、axis 2，以及 shape 元组如何描述维度大小。
重点突出数组不是普通列表，而是规则的多维数字容器。
风格像数学积木和数据表结合，准确、清晰、适合新人理解。
不要出现真实品牌 logo，不要生成密集小字或乱码文字。
""".strip(),
    },
    {
        "filename": "ch03-numpy-broadcasting-vectorization.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "NumPy 广播与向量化运算图",
        "suggested_page": "docs/ch03-data-analysis/ch02-numpy/04-operations.md",
        "alt": "NumPy 广播与向量化运算图：小数组沿维度自动扩展，批量运算替代手写循环。",
        "prompt": """
一张适合 NumPy 数组运算课程的教学图，主题是“广播机制让小数组自动对齐大数组”。
画面表现一个矩阵和一个行向量或列向量相加，小数组沿合适方向虚拟扩展后参与批量计算；旁边对比手写 for 循环和向量化运算。
重点突出形状对齐、广播不是复制真实数据、向量化更简洁高效。
风格清晰、数据工程感、帮助新人理解广播规则。
不要出现真实品牌 logo，不要生成密集小字或乱码文字。
""".strip(),
    },
    {
        "filename": "ch03-pandas-groupby-split-apply-combine.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Pandas GroupBy 分组聚合流程图",
        "suggested_page": "docs/ch03-data-analysis/ch03-pandas/06-groupby.md",
        "alt": "Pandas GroupBy 分组聚合流程图：原始表先按字段拆分，再分别聚合，最后组合成结果表。",
        "prompt": """
一张适合 Pandas 分组聚合课程的教学图，主题是“GroupBy 的 split-apply-combine 三步”。
画面表现一张原始表按类别或城市字段拆成多个小表，每组分别求和、均值或计数，最后组合成一张聚合结果表。
重点突出分组键、聚合函数、多指标聚合和结果索引的关系。
风格像数据加工流水线，表格清楚但不要生成密集小字。
不要出现真实品牌 logo，不要出现乱码文字。
""".strip(),
    },
    {
        "filename": "ch03-multi-source-analysis-architecture.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "多数据源整合分析架构图",
        "suggested_page": "docs/ch03-data-analysis/ch06-projects/02-multi-source-analysis.md",
        "alt": "多数据源整合分析架构图：CSV、Excel、数据库和 API 数据经过清洗、对齐、合并、分析和报告输出。",
        "prompt": """
一张适合数据分析项目课的架构图，主题是“多数据源如何整合成一份分析报告”。
画面表现 CSV、Excel、数据库、API 四类数据源进入数据管道，经过字段标准化、缺失值处理、主键对齐、表连接、指标计算、可视化和报告输出。
重点突出真实项目不是只读一个文件，而是要处理来源、格式、口径和质量差异。
风格专业、清晰、像数据项目蓝图，适合新人理解项目全貌。
不要出现真实品牌 logo，不要生成密集小字或乱码文字。
""".strip(),
    },
    {
        "filename": "intro-ai-fullstack-capability-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "AI 全栈能力总地图",
        "suggested_page": "docs/intro/ai-fullstack-map.md",
        "alt": "AI 全栈能力总地图：开发基础、数据能力、模型基础、大模型理解、应用开发、Agent 系统和工程化方向逐层展开。",
        "prompt": """
一张适合 AI 全栈课程导览页的能力地图图，主题是“AI 全栈能力从基础到应用逐层叠加”。
画面表现七层能力：开发基础、数据能力、数学与模型基础、深度学习与大模型理解、RAG 应用开发、Agent 系统、工程化与方向拓展。
每层用清晰图标表达，像一张从地基到塔顶的学习结构图，强调能力之间不是平铺，而是逐层支撑。
风格现代、清晰、适合新人建立全局方向感；不要出现真实品牌 logo，不要生成密集小字或乱码文字。
""".strip(),
    },
    {
        "filename": "intro-modern-ai-stack-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "现代 AI 应用技术栈全景图",
        "suggested_page": "docs/intro/modern-ai-stack.md",
        "alt": "现代 AI 应用技术栈全景图：模型、RAG、Agent、多模态、模型工程、评估、监控和部署组成真实 AI 系统。",
        "prompt": """
一张适合 2025-2026 AI 应用技术导览的全景图，主题是“现代 AI 应用不只是调用一个模型”。
画面表现用户入口、模型 API、本地模型、RAG 知识库、Agent 工具调用、多模态输入输出、评估体系、监控日志、部署运维和安全护栏。
重点突出 RAGOps、AgentOps、LLMOps 的系统化视角，让新人看到真实 AI 产品由多个层次组合。
风格像现代技术架构地图，专业但不压迫；不要出现真实品牌 logo，不要生成密集小字或乱码文字。
""".strip(),
    },
    {
        "filename": "intro-learning-path-selection.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "推荐学习路线选择图",
        "suggested_page": "docs/intro/learning-path.md",
        "alt": "推荐学习路线选择图：根据应用开发、模型理解、作品集或时间投入选择不同学习深度。",
        "prompt": """
一张适合 AI 课程推荐学习路线页面的选择图，主题是“先选主线，再逐步补分支”。
画面表现一个学习者站在路线选择牌前，三条主路线分别通向 AI 应用工程、模型原理理解、项目作品集交付；旁边有 4 周、8 周、12 周节奏选项。
重点表达不需要第一次推平所有内容，而是先跑通闭环，再按目标补深度。
风格温暖、清晰、像学习路线导航图；不要出现真实品牌 logo，不要生成密集小字或乱码文字。
""".strip(),
    },
    {
        "filename": "intro-four-main-routes-subway.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "四条主线学习路线地铁图",
        "suggested_page": "docs/intro/main-learning-routes.md",
        "alt": "四条主线学习路线地铁图：零基础全栈、已有开发经验、数据模型方向和作品集冲刺路线在关键章节交汇。",
        "prompt": """
一张适合“四条主线学习路线”页面的地铁线路图，主题是“同一套课程可以按目标走不同主线”。
画面表现四条彩色路线：零基础全栈 AI 应用、已有开发经验转 AI、数据科学与模型理解、作品集冲刺；线路在 Python、数据分析、机器学习、RAG、Agent、毕业项目等站点交汇。
重点表达路线可以不同，但关键能力站点会交叉汇合。
风格像高质量地铁路线图和课程地图结合，清晰、有方向感；不要出现真实品牌 logo，不要生成密集小字或乱码文字。
""".strip(),
    },
    {
        "filename": "intro-blocker-diagnosis-flow.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "学习卡点诊断分流图",
        "suggested_page": "docs/intro/blocker-diagnosis-map.md",
        "alt": "学习卡点诊断分流图：从环境、代码、数学、模型、RAG、Agent 和项目表达等卡点回流到对应课程章节。",
        "prompt": """
一张适合学习卡点诊断地图页面的分流图，主题是“卡住时先判断是哪一层能力缺口”。
画面表现中心问题“我卡住了”分流到环境依赖、代码调试、数学概念、模型训练、RAG 检索、Agent 工具、项目表达七类卡点，每类连接到回看章节和最小复现实验。
重点表达卡住不是失败，而是定位能力层并回流补课。
风格像温和的故障诊断控制台，清晰、鼓励、新人友好；不要出现真实品牌 logo，不要生成密集小字或乱码文字。
""".strip(),
    },
    {
        "filename": "intro-project-portfolio-roadmap.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "项目作品集成长阶梯图",
        "suggested_page": "docs/intro/project-roadmap.md",
        "alt": "项目作品集成长阶梯图：从命令行小工具、数据分析、机器学习、RAG、Agent 到毕业项目逐步形成作品集。",
        "prompt": """
一张适合 AI 项目路线与作品集页面的成长阶梯图，主题是“每个阶段都留下一个可展示项目”。
画面表现学习者沿着项目阶梯升级：命令行小工具、网页采集、数据分析报告、机器学习模型、RAG 知识库、Agent 工具流、毕业项目作品集。
每一级都有输入、处理、输出和评估的小图标，强调项目不是炫技，而是可运行、可解释、可展示。
风格明亮、行动感强、适合作品集导览页；不要出现真实品牌 logo，不要生成密集小字或乱码文字。
""".strip(),
    },
    {
        "filename": "intro-role-based-paths-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "角色路线选择地图",
        "suggested_page": "docs/intro/role-based-paths.md",
        "alt": "角色路线选择地图：AI 应用工程师、RAG 工程师、Agent 开发者、模型工程方向和作品集求职者对应不同学习重点。",
        "prompt": """
一张适合角色路线选择页面的职业导向地图，主题是“按目标角色调整精读和项目重点”。
画面表现五个目标角色：AI 应用工程师、RAG 工程师、Agent 开发者、模型工程方向、作品集求职者；每个角色连接到不同能力模块和代表项目。
重点表达不是跳过基础，而是围绕目标角色决定哪些章节精读、哪些快读、哪些项目重点打磨。
风格专业、清爽、像职业路线咨询图；不要出现真实品牌 logo，不要生成密集小字或乱码文字。
""".strip(),
    },
    {
        "filename": "intro-graduation-project-loop.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "毕业项目闭环设计图",
        "suggested_page": "docs/intro/graduation-project-guide.md",
        "alt": "毕业项目闭环设计图：需求、数据、模型、系统、评估、部署和复盘组成可展示的 AI 全栈毕业项目。",
        "prompt": """
一张适合 AI 全栈毕业项目设计指南的闭环图，主题是“毕业项目从需求到交付是一条完整闭环”。
画面表现需求定义、数据与知识库、模型或 API、RAG/Agent 系统、前端或接口、评估与失败案例、部署与日志、作品集展示八个环节闭合。
重点表达毕业项目不是更大的练习，而是能说明问题、方案、评估、失败和迭代的完整作品。
风格像产品蓝图和工程交付看板结合，清晰、有完成感；不要出现真实品牌 logo，不要生成密集小字或乱码文字。
""".strip(),
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate course images and save them under static/img/course.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned image jobs without calling the API.")
    parser.add_argument("--only", nargs="*", help="Generate only selected filenames, such as ai-fullstack-hero.png.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for generated images.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Image model name.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenAI-compatible base URL.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing valid PNG files.")
    parser.add_argument("--ensure-placeholders", action="store_true", help="Create local preview PNG files for all planned images.")
    return parser.parse_args()


def selected_jobs(only: list[str] | None) -> list[dict[str, Any]]:
    if not only:
        return IMAGE_JOBS
    selected = {name.strip() for name in only}
    jobs = [job for job in IMAGE_JOBS if job["filename"] in selected]
    missing = selected - {job["filename"] for job in jobs}
    if missing:
        raise SystemExit(f"Unknown image filename(s): {', '.join(sorted(missing))}")
    return jobs


def write_manifest(output_dir: Path, jobs: list[dict[str, Any]]) -> None:
    manifest = output_dir / "manifest.md"
    lines = ["# AI 全栈课程配图清单", ""]
    lines.append("这些图片由 `scripts/generate_course_images.py` 生成，Markdown 页面使用 `/img/course/...` 引用。")
    lines.append("")
    lines.append("编号说明：课程源码目录已经和网页展示章节号对齐；例如 `docs/ch01-tools` 对应第 1 章“开发者工具基础”，`docs/ch05-machine-learning` 对应第 5 章“机器学习入门到实战”。")
    lines.append("")
    lines.append("| 文件 | 用途 | 建议插入页面 | Alt 文案 |")
    lines.append("|---|---|---|---|")
    for job in IMAGE_JOBS:
        lines.append(f"| `{job['filename']}` | {job['title']} | `{job['suggested_page']}` | {job['alt']} |")
    lines.append("")
    manifest.write_text("\n".join(lines), encoding="utf-8")


def ensure_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def write_placeholder(output_dir: Path, job: dict[str, Any]) -> None:
    output_path = output_dir / job["filename"]
    if output_path.exists():
        try:
            if output_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"):
                return
        except OSError:
            pass

    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        output_path.write_bytes(base64.b64decode(FALLBACK_PNG))
        return

    width, height = 1536, 1024
    image = Image.new("RGB", (width, height), "#111827")
    draw = ImageDraw.Draw(image)

    # A real generated image can replace this preview later. The preview keeps docs visually usable.
    palettes = [
        ("#0f172a", "#2563eb", "#22c55e"),
        ("#111827", "#0891b2", "#f59e0b"),
        ("#1f2937", "#4f46e5", "#ec4899"),
        ("#0b1120", "#14b8a6", "#a3e635"),
    ]
    base, accent, warm = palettes[abs(hash(job["filename"])) % len(palettes)]
    draw.rectangle([(0, 0), (width, height)], fill=base)

    for index in range(0, width, 96):
        draw.line([(index, 0), (index - 420, height)], fill=accent, width=2)
    for index in range(0, height, 96):
        draw.line([(0, index), (width, index + 320)], fill="#1e293b", width=2)

    draw.rounded_rectangle([(110, 120), (1426, 904)], radius=48, outline=accent, width=6)
    draw.rounded_rectangle([(160, 170), (700, 360)], radius=36, fill="#0f172a", outline="#334155", width=3)
    draw.rounded_rectangle([(820, 190), (1320, 360)], radius=36, fill="#0f172a", outline="#334155", width=3)
    draw.rounded_rectangle([(210, 560), (1320, 730)], radius=44, fill="#020617", outline=warm, width=5)

    for x, y, r, color in [(300, 265, 54, accent), (455, 265, 54, warm), (610, 265, 54, "#60a5fa")]:
        draw.ellipse([(x - r, y - r), (x + r, y + r)], fill=color)
    draw.line([(700, 265), (820, 265)], fill="#e5e7eb", width=8)
    draw.polygon([(820, 265), (790, 245), (790, 285)], fill="#e5e7eb")

    font_paths = [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
    ]

    def load_font(size: int):
        for font_path in font_paths:
            if Path(font_path).exists():
                try:
                    return ImageFont.truetype(font_path, size=size)
                except OSError:
                    continue
        return ImageFont.load_default()

    title_font = load_font(66)
    body_font = load_font(34)
    label_font = load_font(26)

    draw.text((210, 595), job["title"], font=title_font, fill="#f8fafc")

    alt_lines = textwrap.wrap(job["alt"], width=30)
    y = 760
    for line in alt_lines[:2]:
        draw.text((220, y), line, font=body_font, fill="#cbd5e1")
        y += 48

    draw.text((1160, 850), "Preview Asset", font=label_font, fill="#94a3b8")
    image.save(output_path, format="PNG")
    set_user_readable_permissions(output_path)


def ensure_placeholders(output_dir: Path) -> None:
    ensure_output_dir(output_dir)
    for job in IMAGE_JOBS:
        write_placeholder(output_dir, job)


def set_user_readable_permissions(output_path: Path) -> None:
    output_path.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)


def generate_image_with_http(api_key: str, base_url: str, model: str, job: dict[str, Any], retries: int = 2) -> bytes:
    """Generate one image through an OpenAI-compatible HTTP endpoint."""
    endpoint = f"{base_url.rstrip('/')}/images/generations"
    payload = {
        "model": model,
        "prompt": job["prompt"],
        "size": job["size"],
        "quality": job["quality"],
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-type": "application/json",
            "Accept": "*/*",
            "User-Agent": "curl/8.7.1",
        },
        method="POST",
    )
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                response_data = json.loads(response.read().decode("utf-8"))
                break
        except urllib.error.HTTPError as exc:
            if exc.code in {408, 429, 500, 502, 503, 504, 524} and attempt < retries:
                wait_seconds = 8 * (attempt + 1)
                print(
                    f"Image API returned HTTP {exc.code} for {job['filename']}; "
                    f"retrying in {wait_seconds}s...",
                    flush=True,
                )
                time.sleep(wait_seconds)
                continue
            raise RuntimeError(f"Image API request failed with HTTP {exc.code} for {job['filename']}.") from exc
        except (urllib.error.URLError, TimeoutError, http.client.HTTPException) as exc:
            if attempt < retries:
                wait_seconds = 8 * (attempt + 1)
                print(
                    f"Image API network error for {job['filename']}; retrying in {wait_seconds}s...",
                    flush=True,
                )
                time.sleep(wait_seconds)
                continue
            raise RuntimeError(f"Image API network error for {job['filename']}.") from exc
    else:
        raise RuntimeError(f"Image API request failed for {job['filename']}.")

    image_data = response_data.get("data", [{}])[0]
    if image_data.get("b64_json"):
        return base64.b64decode(image_data["b64_json"])
    raise RuntimeError("Image API response did not include data[0].b64_json.")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    jobs = selected_jobs(args.only)

    print(f"model: {args.model}", flush=True)
    print(f"base_url: {args.base_url}", flush=True)
    print(f"output_dir: {output_dir}", flush=True)
    print(f"jobs: {len(jobs)}", flush=True)

    if args.ensure_placeholders:
        ensure_placeholders(output_dir)
        write_manifest(output_dir, IMAGE_JOBS)
        print(f"Placeholders ensured under {output_dir}", flush=True)
        return

    if args.dry_run:
        for job in jobs:
            print(f"DRY RUN: {job['filename']} ({job['size']}, {job['quality']}) - {job['title']}", flush=True)
        return

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set. Set it in your local shell before generating images.")

    ensure_output_dir(output_dir)
    write_manifest(output_dir, jobs)

    client = None
    try:
        from openai import OpenAI
    except ImportError as exc:
        print("The Python package `openai` is not installed; using the built-in HTTP fallback.", flush=True)
    else:
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=args.base_url)

    for job in jobs:
        output_path = output_dir / job["filename"]
        if output_path.exists() and output_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") and not args.overwrite:
            print(f"Skipping existing valid PNG: {job['filename']} (use --overwrite to regenerate)", flush=True)
            continue
        print(f"Generating {job['filename']}...", flush=True)
        if client:
            result = client.images.generate(
                model=args.model,
                prompt=job["prompt"],
                size=job["size"],
                quality=job["quality"],
            )
            image_base64 = result.data[0].b64_json
            output_path.write_bytes(base64.b64decode(image_base64))
        else:
            output_path.write_bytes(
                generate_image_with_http(
                    api_key=os.environ["OPENAI_API_KEY"],
                    base_url=args.base_url,
                    model=args.model,
                    job=job,
                )
            )
        set_user_readable_permissions(output_path)
        print(f"Saved {output_path}", flush=True)


if __name__ == "__main__":
    main()
