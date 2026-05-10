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
from datetime import datetime, timezone
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
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports" / "course-images"
DEFAULT_MODEL = os.environ.get("OPENAI_IMAGE_MODEL", "gpt-image-2")
DEFAULT_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://cliproxy.airoads.org/v1")
DEFAULT_REQUEST_TIMEOUT = int(os.environ.get("OPENAI_IMAGE_TIMEOUT", "180"))
DEFAULT_IMAGE_RETRIES = int(os.environ.get("OPENAI_IMAGE_RETRIES", "2"))
FALLBACK_PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAFgwJ/lLwVRwAAAABJRU5ErkJggg=="

# Course illustrations should prefer 1024x1792 vertical comic/explainer pages.
# They read better on mobile and keep zh/en/ja variants visually consistent.
# Use landscape only by adding an explicit `allow_landscape: True` to a job.
DEFAULT_COURSE_IMAGE_SIZE = "1024x1792"
DEFAULT_COURSE_IMAGE_QUALITY = "high"
IMAGE_JOBS: list[dict[str, Any]] = [
    {
        "filename": "intro-quick-experience-loop.png",
        "size": "1024x1792",
        "quality": "high",
        "title": "30 分钟 AI 快速体验闭环图",
        "suggested_page": "docs/intro/quick-experience.md",
        "alt": "30 分钟 AI 快速体验闭环图：打开 Colab、运行图像识别、体验文本生成、尝试图像生成，然后回到主课程。",
        "prompt": """
一张竖版 9:16 中文教学漫画，主题是“30 分钟先体验 AI，再回到系统学习”。
构图固定：一张连续的学习桌面 + 蜿蜒路线，不要白底圆角卡片堆叠，不要 SVG 信息图，不要纯流程框。五个编号小场景沿着同一条蓝绿色学习路径从上到下连接，像新人跟做的迷你实验路线。
五个小场景短标签依次是：打开 Colab、运行图像识别、体验文本生成、尝试图像生成、回到主课程。
每个场景都要用画面教 input -> model -> output：提示词纸片进入模型屏幕，按下运行，右侧出现结果；再改一个词比较输出。不要只贴文字。
可视元素要具体：学习者、笔记本电脑、狗的图片被识别、文本回复卡片、由提示词生成的湖边小屋图片、最后回到课程地图。
文字只保留标题、五个步骤名和极短提示；中文必须自然简洁，只有 Colab、AI、prompt 这类关键术语可以保留英文。文字要贴近对应物体、字号大、清楚。不要大段英文说明、不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "intro-quick-experience-loop-en.png",
        "size": "1024x1792",
        "quality": "high",
        "title": "30-Minute AI Quick Experience Loop",
        "suggested_page": "docs/intro/quick-experience.md",
        "alt": "30-minute AI quick experience loop: open Colab, run image recognition, try text generation, try image generation, then return to the main course path.",
        "prompt": """
A vertical 9:16 English teaching comic for the first 30-minute AI experience.
Fixed composition: one continuous learning desk plus a winding path, not a stack of white rounded cards, not an SVG infographic, not a pure flowchart. Five numbered mini-scenes follow the same blue-green learning path from top to bottom.
Use these five short step labels: Open Colab, Run image recognition, Try text generation, Try image generation, Return to the main course path.
Each scene must teach input -> model -> output through the drawing: a prompt/input card enters a model screen, a run action happens, a result appears, then one word changes and the learner compares the output. Do not make it a text poster.
Concrete visuals: learner at a laptop, a dog photo being recognized, a generated text reply card, a lake cabin image generated from a prompt, and the learner returning to a course map.
Visible text should only be the title, five step names, and very short action notes. Text must be natural English, large, readable, and attached to the object it explains. Avoid tiny text, gibberish, fake UI text, real brand logos, and large white rounded containers.
""".strip(),
    },
    {
        "filename": "intro-quick-experience-loop-ja.png",
        "size": "1024x1792",
        "quality": "high",
        "title": "30分 AI クイック体験ループ",
        "suggested_page": "docs/intro/quick-experience.md",
        "alt": "30分AIクイック体験ループ: Colab を開き、画像認識を実行し、テキスト生成と画像生成を試してから、本編ルートに戻る。",
        "prompt": """
AI の最初の 30 分体験を説明する、縦長 9:16 の日本語教材漫画。
固定構図：連続した学習デスクと曲がる学習ルート。白い角丸カードの積み重ね、SVG 風インフォグラフィック、ただのフローチャートにはしない。5 つの番号付き小場面を、同じ青緑のルートで上から下へつなぐ。
5 つの短いステップ名は「Colab を開く」「画像認識を実行」「テキスト生成を試す」「画像生成を試す」「本編ルートに戻る」。
各場面は input -> model -> output を絵で教える。プロンプト/入力カードがモデル画面に入り、実行し、結果が出て、1 語だけ変えて出力を比べる。文字だけのポスターにはしない。
具体的な画面要素：ノートPCの前の学習者、認識される犬の写真、生成された文章カード、プロンプトから作られる湖畔の小屋画像、最後に講座マップへ戻る学習者。
画像内の文字はタイトル、5 つのステップ名、ごく短い行動メモだけ。自然な日本語で、大きく読みやすく、対応する物体の近くに置く。細かい文字、文字化け、偽 UI テキスト、実在ロゴ、大きな白い角丸コンテナは禁止。
""".strip(),
    },
    {
        "filename": "intro-minimal-setup-kit-en.png",
        "size": "1024x1792",
        "quality": "high",
        "title": "Minimal setup kit before starting the AI full-stack course",
        "suggested_page": "docs/intro/environment-setup.md",
        "alt": "Minimal setup kit before starting the AI full-stack course: a normal computer, browser and Colab, VS Code, Terminal, Python 3.11 or Miniconda, Git and GitHub, and a project folder README; GPU, Docker, vector database, and LLM API key can wait.",
        "prompt": """
A vertical English course infographic, comic-like but practical, explaining "Minimal setup kit before starting the AI full-stack course".
Use a clean step-by-step layout, not a decorative poster. The main area shows one ordinary laptop as the center, with six clear setup cards around it:
1. Browser / Colab
2. VS Code
3. Terminal
4. Python 3.11 / Miniconda
5. Git / GitHub
6. Project folder + README
At the bottom, add a separate muted section labeled "Later, not now" with four small cards: GPU, Docker, vector database, LLM API key.
Use short natural English labels only, large and readable. Avoid dense paragraphs, tiny text, gibberish, fake UI text, watermarks, and real brand logos.
Style: warm professional beginner-friendly course illustration, light workflow comic, clear icons, simple arrows, good spacing, vertical 9:16.
""".strip(),
    },
    {
        "filename": "intro-minimal-setup-kit.png",
        "size": "1024x1792",
        "quality": "high",
        "title": "AI 全栈课程最小准备包图",
        "suggested_page": "docs/intro/environment-setup.md",
        "alt": "AI 全栈课程最小准备包图：普通电脑、浏览器和 Colab、VS Code、Terminal、Python 3.11 或 Miniconda、Git 和 GitHub、项目文件夹 README；GPU、Docker、向量数据库和 LLM API Key 可以以后再准备。",
        "prompt": """
一张竖版中文课程图解，主题是“AI 全栈课程开始前，最小需要准备什么”。
画面要像可跟做的步骤漫画/流程图，不要做成装饰海报。主体是一台普通电脑，周围分成 6 张清晰准备卡：
1. 浏览器 / Colab
2. VS Code
3. Terminal
4. Python 3.11 / Miniconda
5. Git / GitHub
6. 项目文件夹 + README
底部单独做一个弱化区域，标题“以后再说”，放 4 张小卡：GPU、Docker、向量数据库、LLM API Key。
文字只用自然中文短标签，术语 VS Code、Terminal、Python 3.11、Miniconda、Git、GitHub、README、GPU、Docker、LLM API Key 可以保留英文。字体要大、清楚、少量。不要密集段落、不要小字乱码、不要真实品牌 logo、不要水印。
风格专业、温和、新手友好，竖版 9:16，图标清晰，箭头简单，留白充足。
""".strip(),
    },
    {
        "filename": "intro-minimal-setup-kit-ja.png",
        "size": "1024x1792",
        "quality": "high",
        "title": "AI フルスタック講座を始める前の最小準備セット図",
        "suggested_page": "docs/intro/environment-setup.md",
        "alt": "AI フルスタック講座を始める前の最小準備セット図：普通のPC、ブラウザと Colab、VS Code、Terminal、Python 3.11 または Miniconda、Git と GitHub、プロジェクトフォルダ README。GPU、Docker、ベクトルデータベース、LLM API Key は後でよい。",
        "prompt": """
AI フルスタック講座を始める前の「最小準備セット」を説明する、日本語の縦長コース図解。
飾りポスターではなく、手順が分かる漫画風/フロー図にする。中央に普通のノートPCを置き、その周囲に 6 枚の分かりやすい準備カードを配置：
1. ブラウザ / Colab
2. VS Code
3. Terminal
4. Python 3.11 / Miniconda
5. Git / GitHub
6. プロジェクトフォルダ + README
下部に控えめな別エリアを作り、見出しは「後でOK」。小カードは GPU、Docker、ベクトルDB、LLM API Key。
文字は自然な日本語の短いラベルだけにし、VS Code、Terminal、Python 3.11、Miniconda、Git、GitHub、README、GPU、Docker、LLM API Key などの標準用語は英語表記でよい。文字は大きく読みやすく、少なめに。細かすぎる文字、文字化け、透かし、実在ブランドロゴは禁止。
スタイルはプロフェッショナルでやさしい初心者向け教材図解、縦 9:16、明快なアイコン、簡単な矢印、余白を十分に。
""".strip(),
    },
    {
        "filename": "intro-start-here-four-step-map-en.png",
        "size": "1024x1792",
        "quality": "high",
        "title": "Start Here four-step entrance map",
        "suggested_page": "docs/intro/course-page-guide.md",
        "alt": "Start Here four-step map for new AI full-stack learners: Quick Experience, Environment Setup, Capability Map, and Learning Path into Chapter 1.",
        "prompt": """
A vertical English course infographic, comic panels plus workflow arrows, mobile-readable, theme: "Start Here: four steps into the AI full-stack course".
Use four stacked step panels with large short labels and clear icons:
1. Quick Experience: run a tiny AI example and feel input -> model -> output.
2. Environment Setup: prepare Python, Git, and a project folder; do not install advanced tools first.
3. Capability Map: understand tools, data, models, LLM, RAG, Agent, delivery in one map.
4. Learning Path: choose one route, then enter Chapter 1.
Make each step a simple mini scene, not a decorative background. Use arrows connecting the four panels downward.
Bottom rule in a large clear strip: "Read briefly, run something, keep evidence".
Text must be sparse, large, and clean. Avoid tiny text, dense paragraphs, gibberish, watermarks, and real brand logos.
Style: professional beginner-friendly technical course illustration, vertical 9:16, comic workflow, clear icons, calm colors, enough whitespace.
""".strip(),
    },
    {
        "filename": "intro-start-here-four-step-map.png",
        "size": "1024x1792",
        "quality": "high",
        "title": "先从这里开始四步入口图",
        "suggested_page": "docs/intro/course-page-guide.md",
        "alt": "新人从零开始 AI 全栈课程的四步入口图：快速体验、环境准备、能力地图、学习路线，然后进入第 1 章。",
        "prompt": """
一张竖版中文课程图解，漫画式小分镜 + 流程箭头，移动端可读，主题是“先从这里开始：新人进入 AI 全栈课程的 4 步入口”。
从上到下 4 个步骤面板，每步用大字短标签和清晰图标：
1. 快速体验：先跑一个小 AI 示例，感受 输入 -> 模型 -> 输出。
2. 环境准备：准备 Python、Git、项目文件夹，不要先装一堆高级工具。
3. 能力地图：看一张图理解 tools、data、models、LLM、RAG、Agent、delivery。
4. 学习路线：选一条路线，然后进入第 1 章。
每一步是简单小场景，不要装饰性背景。用向下箭头连接四个面板。
底部大字短规则：“短读，跑起来，留证据”。
文字要少、字号大、清晰。术语 Python、Git、tools、data、models、LLM、RAG、Agent、delivery 可保留英文。不要小字乱码、不要真实品牌 logo、不要水印、不要密集段落。
风格专业、温和、新手友好，竖版 9:16，流程漫画感，图标清楚，留白充足。
""".strip(),
    },
    {
        "filename": "intro-start-here-four-step-map-ja.png",
        "size": "1024x1792",
        "quality": "high",
        "title": "ここから始める四ステップ入口図",
        "suggested_page": "docs/intro/course-page-guide.md",
        "alt": "AI フルスタック講座をゼロから始めるための四ステップ入口図：クイック体験、環境準備、能力マップ、学習ルートから第1章へ進む。",
        "prompt": """
日本語の縦長コース図解。漫画風の小パネル + フロー矢印で、モバイルでも読める構成。テーマは「ここから始める：AI フルスタック講座への 4 ステップ入口」。
上から下へ 4 つのステップパネルを配置し、大きな短いラベルと明快なアイコンを使う：
1. クイック体験：小さな AI 例を動かし、入力 -> モデル -> 出力を体感。
2. 環境準備：Python、Git、プロジェクトフォルダを準備。高度なツールを先に大量導入しない。
3. 能力マップ：tools、data、models、LLM、RAG、Agent、delivery を一枚で理解。
4. 学習ルート：ひとつのルートを選び、第1章へ進む。
各ステップはシンプルな小場面にし、装飾背景にしない。4 パネルを下向き矢印でつなぐ。
下部の大きな短いルール：「短く読み、動かし、証拠を残す」。
文字は少なく、大きく、読みやすく。Python、Git、tools、data、models、LLM、RAG、Agent、delivery などの標準用語は英語でよい。細かい文字、文字化け、実在ブランドロゴ、透かし、長い段落は禁止。
スタイルはプロフェッショナルでやさしい初心者向け教材図解、縦 9:16、フロー漫画、明快なアイコン、十分な余白。
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
        "filename": "ch01-ai-workstation-comic.png",
        "size": "1024x1792",
        "quality": "high",
        "title": "开发者工具 AI 工作站漫画指导图",
        "suggested_page": "docs/ch01-tools/index.md",
        "alt": "开发者工具 AI 工作站漫画：新人把终端、Python 环境、VS Code、Jupyter 和 Git 组装成可复现的 AI 学习工作台。",
        "prompt": """
制作一页 9:16 竖版中文科普漫画，标题：“第一章：搭建 AI 学习工作站”。
像真正漫画书内页，不是海报；适合 AI 全栈课程新人导读。画面清晰、不拥挤，6 格分镜，统一温暖科技感。

第 1 格：新人站在凌乱桌面前，桌上有散乱文件、报错窗口和问号。气泡：“我不是不会 AI，是环境总坏。”
第 2 格：终端像控制台点亮，旁白：“终端负责发出可重复命令。”小标签：“路径、命令、输出。”
第 3 格：Python 环境像独立实验室，多个项目放在不同透明盒子里。短句：“每个项目一个环境。”
第 4 格：VS Code 和 Jupyter 像两块操作面板：一个写工程代码，一个做实验笔记。短句：“工程与探索各司其职。”
第 5 格：Git 像游戏存档机，把 README、代码和运行记录保存成时间节点。成功点标签：“能回看、能恢复、能展示。”
第 6 格：工作站变整洁，新人把第一个 ai-learning-lab 推到云端。问题标签：“不要混环境、不要乱删、不要忘记提交。”箭头：“下一章：开始写 Python 程序。”

底部历史意义/学习意义：“稳定工具链，让后面的 Python、数据、模型、RAG 和 Agent 都能真正跑起来。”
所有文字必须是简体中文，短句、清晰、融入气泡、小黑板、旁白框和标签牌。不要乱码，不要英文水印，不要真实 logo。
""".strip(),
    },
    {
        "filename": "ch01-ai-workstation-comic-en.png",
        "size": "1024x1792",
        "quality": "high",
        "title": "Developer Tools AI Workstation Comic Guide",
        "suggested_page": "docs/ch01-tools/index.md",
        "alt": "Developer tools AI workstation comic: a beginner assembles terminal, Python environment, VS Code, Jupyter, and Git into a reproducible AI learning workstation.",
        "prompt": """
Create one 9:16 vertical English educational comic page titled: “Chapter 1: Build Your AI Learning Workstation”.
It should look like a real comic-book learning page, not a poster. Beginner-friendly, warm tech style, clear layout, 6 panels, not crowded.

Panel 1: A beginner stands at a messy desk with scattered files, error windows, and question marks. Speech bubble: “Maybe AI is not the problem. My setup keeps breaking.”
Panel 2: The terminal lights up like a control console. Narration: “Terminal: repeatable commands.” Small labels: “path, command, output.”
Panel 3: Python environments appear as separate transparent labs for different projects. Short line: “One project, one environment.”
Panel 4: VS Code and Jupyter appear as two work panels: one for engineering code, one for experiments and notes. Short line: “Build in VS Code. Explore in Jupyter.”
Panel 5: Git appears as a game save machine, saving README, code, and run logs into timeline checkpoints. Success label: “review, restore, showcase.”
Panel 6: The workstation becomes clean. The beginner pushes the first ai-learning-lab to the cloud. Problem label: “Don’t mix environments. Don’t delete blindly. Don’t forget commits.” Arrow: “Next: write Python programs.”

Bottom learning meaning: “A stable toolchain lets Python, data, models, RAG, and Agents actually run.”
All visible text must be natural English, short, clear, and integrated into speech bubbles, blackboards, narration boxes, and labels. No gibberish, no watermark, no real logos.
""".strip(),
    },
    {
        "filename": "ch01-ai-workstation-comic-ja.png",
        "size": "1024x1792",
        "quality": "high",
        "title": "開発者ツール AI 作業台漫画ガイド",
        "suggested_page": "docs/ch01-tools/index.md",
        "alt": "開発者ツール AI 作業台漫画：初心者がターミナル、Python 環境、VS Code、Jupyter、Git を組み合わせて再現可能な AI 学習作業台を作る。",
        "prompt": """
9:16 縦長の日本語教育漫画を 1 ページ作成してください。タイトル：「第1章：AI 学習作業台を作る」。
ポスターではなく、本物の漫画教材の1ページのようにしてください。初心者向け、温かいテック感、6 コマ、見やすく、詰め込みすぎない構成。

第 1 コマ：初心者が散らかった机の前に立っている。机には散乱したファイル、エラー画面、疑問符。吹き出し：「AI が難しい前に、環境がすぐ壊れる。」
第 2 コマ：ターミナルが操作コンソールのように光る。ナレーション：「ターミナル：再現できる命令。」小ラベル：「パス、コマンド、出力。」
第 3 コマ：Python 環境がプロジェクトごとの透明な実験室として描かれる。短句：「1 プロジェクト、1 環境。」
第 4 コマ：VS Code と Jupyter が2つの操作パネルとして並ぶ。片方は工程コード、片方は実験ノート。短句：「VS Code で作る。Jupyter で試す。」
第 5 コマ：Git がゲームのセーブ装置のように、README、コード、実行ログを時間軸に保存する。成功ラベル：「見返せる、戻せる、見せられる。」
第 6 コマ：作業台が整い、初心者が最初の ai-learning-lab をクラウドへ push する。問題ラベル：「環境を混ぜない。むやみに消さない。コミットを忘れない。」矢印：「次：Python プログラムを書く。」

下部の学習意味：「安定したツールチェーンがあるから、Python、データ、モデル、RAG、Agent が本当に動く。」
画像内の文字はすべて自然な日本語にしてください。短く、読みやすく、吹き出し・黒板・ナレーション枠・ラベルに自然に入れてください。文字化け、透かし、実在ロゴは禁止。
""".strip(),
    },
    {
        "filename": "ch01-hands-on-workstation-route.png",
        "size": "1024x1792",
        "quality": "high",
        "title": "第 1 章工作台实操路线图",
        "suggested_page": "docs/ch01-tools/ch04-workshop/01-hands-on-tools-workshop.md",
        "alt": "第 1 章工作台实操路线图：终端、项目骨架、Python 检查脚本、VS Code、Jupyter、Git commit 和证据报告组成完整练习。",
        "prompt": """
一张 9:16 竖版中文教学漫画，主题是“第 1 章：从空文件夹到可复现 AI 学习工作台”。
构图固定：一张连续的工作台 + 蜿蜒证据路线，不要白底圆角框堆叠，不要 SVG 信息图，不要纯流程框。7 个编号小场景沿着同一条蓝绿色路径从上到下连接，像新人跟做的实操路线。
7 个短标签依次是：打开终端、创建 ai-learning-lab、编写 workstation_check.py、运行 Python 检查、打开 VS Code、在 Jupyter 中查看、提交证据到 Git。
每个场景都要把 input -> action -> output 画出来：文件夹进入终端，脚本运行后生成 reports，VS Code 和 Jupyter 看同一份结果，最后 Git 留下 commit 和证据包。不要只贴图标。
可视元素要具体：终端窗口、项目文件夹、Python 脚本、报告文件、编辑器界面、Notebook 结果、Git 提交节点、README 作品集页面。
文字只保留标题、7 个步骤名和极短提示；中文必须自然简洁，必要技术词可以保留英文，如 Python、VS Code、Jupyter、Git、commit、reports。文字要贴近对应物体，字号大、清楚。不要大段英文说明、不要乱码小字或真实品牌 logo、不要大块白色流程卡。
""".strip(),
    },
    {
        "filename": "ch01-hands-on-workstation-route-en.png",
        "size": "1024x1792",
        "quality": "high",
        "title": "Chapter 1 Hands-On Workstation Route",
        "suggested_page": "docs/ch01-tools/ch04-workshop/01-hands-on-tools-workshop.md",
        "alt": "Chapter 1 hands-on workstation route: terminal, project skeleton, Python check script, VS Code, Jupyter, Git commit, and evidence reports form one complete practice.",
        "prompt": """
A vertical 9:16 English teaching comic for Chapter 1: from an empty folder to a reproducible AI learning workstation.
Fixed composition: one continuous workbench plus a winding evidence path, not a stack of white rounded cards, not an SVG infographic, not a pure flowchart. Seven numbered mini-scenes follow the same blue-green route from top to bottom.
Use these seven short labels: open terminal, create ai-learning-lab, write workstation_check.py, run the Python check, open VS Code, review in Jupyter, commit evidence to Git.
Each scene must show input -> action -> output: a folder enters the terminal, the script runs and generates reports, VS Code and Jupyter inspect the same result, and Git leaves a commit plus an evidence pack. Do not make it a text poster.
Concrete visuals: terminal window, project folder, Python script, report file, editor interface, notebook result, Git commit node, and a README portfolio page.
Visible text should only be the title, seven step names, and very short action notes. Text must be natural English, large, readable, and attached to the object it explains. Avoid tiny text, gibberish, real brand logos, and large white flow cards.
""".strip(),
    },
    {
        "filename": "ch01-hands-on-workstation-route-ja.png",
        "size": "1024x1792",
        "quality": "high",
        "title": "第1章 ハンズオン作業台ルート",
        "suggested_page": "docs/ch01-tools/ch04-workshop/01-hands-on-tools-workshop.md",
        "alt": "第1章ハンズオン作業台ルート: ターミナル、プロジェクト骨格、Python 検査スクリプト、VS Code、Jupyter、Git commit、証拠レポートで 1 つの実習ルートを作る。",
        "prompt": """
第1章の実践フローを説明する、縦長 9:16 の日本語教材漫画。
固定構図：温かい木の作業台、吊り下げた木札のタイトル、曲がりながら進む青緑の証拠ルート。白い角丸カードの積み重ね、SVG 風インフォグラフィック、ただのフローチャートにはしない。7 つの番号付き小場面を、同じ青緑のルートで上から下へつなぐ。英語版と同じく、机の上に実物の道具やファイルが置かれている感じにする。
7 つの短いラベルは「ターミナルを開く」「ai-learning-lab を作る」「workstation_check.py を書く」「Python 検査を実行」「VS Code を開く」「Jupyter で確認」「Git に証拠を commit」。
各場面で input -> action -> output を絵で示す。フォルダがターミナルに入り、スクリプトが reports を作り、VS Code と Jupyter が同じ結果を確認し、最後に Git が commit と証拠パックを残す。文字だけのポスターにはしない。
具体的な画面要素：ターミナル画面、プロジェクトフォルダ、Python スクリプト、レポートファイル、エディタ画面、Notebook の結果、Git commit ノード、README の作品集ページ。
画像内の文字はタイトル、7 つのステップ名、ごく短い行動メモだけ。自然な日本語で、大きく読みやすく、対応する物体の近くに置く。Python、VS Code、Jupyter、Git、commit、reports などの技術用語は英語表記のままでよい。大きな白いフローカード、文字化け、実在ロゴは禁止。背景の雰囲気は、木の机、ランプ、本、観葉植物がある温かいワークスペースにそろえる。
""".strip(),
    },
    {
        "filename": "ch01-hands-on-terminal-git-loop.png",
        "size": "1024x1792",
        "quality": "high",
        "title": "终端 Python Git 最小执行循环",
        "suggested_page": "docs/ch01-tools/ch04-workshop/01-hands-on-tools-workshop.md",
        "alt": "终端 Python Git 最小执行循环：进入项目目录，运行 Python 脚本，生成报告，查看 git status，add 并 commit。",
        "prompt": """
一张 9:16 竖版中文教学图，主题是“终端、Python、Git 的最小执行循环”。
画面像分步骤漫画流程：1 pwd/ls 确认当前目录，2 python3 src/workstation_check.py 运行脚本，3 reports 生成 JSON 和 Markdown，4 git status 查看变化，5 git add 选择文件，6 git commit 保存检查点。
用箭头连接每一步，并在旁边用小放大镜突出“当前目录”“解释器”“未跟踪文件”“提交消息”这四个新人最容易卡住的检查点。
风格清爽、实操感强、像命令行课程练习卡。
文字少而大，命令和技术词保留英文，例如 pwd、ls、python3、git status、git add、git commit。不要真实品牌 logo，不要乱码小字。
""".strip(),
    },
    {
        "filename": "ch01-hands-on-env-editor-notebook-flow.png",
        "size": "1024x1792",
        "quality": "high",
        "title": "环境 编辑器 Notebook 协作流程图",
        "suggested_page": "docs/ch01-tools/ch04-workshop/01-hands-on-tools-workshop.md",
        "alt": "环境、编辑器与 Notebook 协作流程图：同一个项目环境被终端、VS Code 和 Jupyter 共享，脚本生成报告，Notebook 读取报告。",
        "prompt": """
一张 9:16 竖版中文教学流程图，主题是“同一个项目环境如何连接终端、VS Code 和 Jupyter”。
画面中心是 ai-learning-lab 项目文件夹，连接到同一个 Python 解释器 / 虚拟环境。
左侧是终端运行 src/workstation_check.py，右侧是 VS Code 选择同一个解释器，下方是 Jupyter Notebook 读取 reports/workstation-check.json 并显示结果。
加入一个错误对比角落：如果 VS Code 或 Jupyter 选错解释器，就出现 import error 或 file not found 警示。
重点帮助新人理解“不是三个工具各跑各的，而是都应该对齐到同一个项目环境”。
中文标签为主，保留 VS Code、Jupyter、Python 解释器、reports/workstation-check.json 等技术词。不要真实品牌 logo，不要乱码小字。
""".strip(),
    },
    {
        "filename": "ch01-hands-on-debug-map.png",
        "size": "1024x1792",
        "quality": "high",
        "title": "第 1 章工作台常见错误排查图",
        "suggested_page": "docs/ch01-tools/ch04-workshop/01-hands-on-tools-workshop.md",
        "alt": "第 1 章工作台常见错误排查图：命令找不到、路径错误、解释器不一致、Git 仓库缺失和 Notebook 路径错误分别对应检查命令。",
        "prompt": """
一张 9:16 竖版中文排错地图，主题是“第 1 章工作台问题先定位断在哪里”。
画面从顶部错误提示流向 6 张故障卡片：command not found、No such file or directory、ModuleNotFoundError、fatal not a git repository、wrong interpreter、Notebook cannot find report。
每张卡片连接到一个检查动作：python --version、pwd + ls、which python + python -m pip --version、git status、Python: Select Interpreter、Path.cwd()。
底部汇总成一句行动原则：保存完整命令、完整输出、修复记录到 learning-log。
风格像新人排障速查表，清晰、流程式、非装饰。
文字少而大；命令和错误名保留英文，说明标签用中文。不要真实品牌 logo，不要乱码小字。
""".strip(),
    },
    {
        "filename": "ch01-hands-on-portfolio-pack.png",
        "size": "1024x1792",
        "quality": "high",
        "title": "第 1 章作品集证据包",
        "suggested_page": "docs/ch01-tools/ch04-workshop/01-hands-on-tools-workshop.md",
        "alt": "第 1 章作品集证据包：README、工作台检查脚本、JSON 报告、Markdown 报告、学习日志、终端输出和 Git 提交历史共同证明环境可复现。",
        "prompt": """
一张 9:16 竖版中文作品集证据包图，主题是“第 1 章完成后应该留下哪些可检查证据”。
画面中心是 ai-learning-lab 文件夹，里面清楚展示 README.md、src/workstation_check.py、reports/workstation-check.json、reports/workstation-report.md、notes/learning-log.md、screenshots、Git history。
用箭头说明证据链：运行命令生成 reports，reports 进入 README，Git commit 保存版本，learning-log 记录错误和修复。
重点表达“能运行 + 有记录 + 可复现”才是工具章的真实产出。
中文标签为主，必要文件名和 Git/README/reports 等技术词保留英文。不要真实品牌 logo，不要乱码小字。
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
        "title": "文件读写与序列化流程图",
        "suggested_page": "docs/ch02-python/ch02-advanced/03-file-io.md",
        "alt": "文件读写与序列化流程图：内存中的 Python 数据被序列化写入文件，再读取并还原成对象。",
        "prompt": """
一张竖版 9:16 中文教学插画，主题标题：“程序结束后，数据怎样还在？”。
根据 Python 文件读写与序列化课程内容设计：让新手一眼看懂内存数据只能临时存在，必须经过序列化写入文件，下一次启动再读取并反序列化恢复。
固定构图：上半部分是正在运行的 Python 程序桌面，左侧有列表、字典、对象三种“内存中的数据”；中间是一个清楚的“序列化”转换机器，把数据变成 JSON、CSV、TXT 三种文件卡；右侧是磁盘文件夹。下半部分是“下次启动”的反向路径：从文件夹读取文件，经过“反序列化”，回到内存数据。用一条连续箭头形成闭环。
必须把四个关键动作画成可见操作：保存、写入文件、读取文件、恢复对象。加入一个小错误提示角落：“只存在内存，关掉程序就丢失”，并用修复路径指向“写入文件”。
可见文字只保留：标题、内存数据、序列化、JSON、CSV、TXT、磁盘文件、读取、反序列化、恢复对象、下次启动、只在内存会丢失。中文必须自然清楚；JSON、CSV、TXT、Python 可以保留英文。字号大，不要密集代码，不要英文段落，不要乱码小字。
风格是直接可学的课堂漫画 + 数据流转插画，有人物和真实操作，不要白底圆角框堆叠，不要 SVG 信息图，不要纯文字贴图，不要真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch02-file-io-serialization-flow-en.png",
        "title": "File I/O and Serialization Flow",
        "suggested_page": "docs/ch02-python/ch02-advanced/03-file-io.md",
        "alt": "File I/O and serialization flow: Python data in memory is serialized, written to files, then read and restored on the next run.",
        "prompt": """
A vertical 9:16 English teaching illustration titled: "How data survives after the program stops".
Design it for a Python file I/O and serialization lesson. A beginner should understand that data in memory is temporary, so the program must serialize it, write it to a file, then read and deserialize it on the next startup.
Fixed composition: the top half shows a running Python program workspace. On the left are three concrete in-memory data examples: list, dict, object. In the middle, a clear "Serialize" conversion machine turns them into three file cards: JSON, CSV, TXT. On the right is a disk folder. The bottom half shows the reverse path labeled "Next run": read the files from the folder, pass through "Deserialize", and restore the data back into memory. Use one continuous arrow loop.
Show four actions as visible operations, not just labels: save, write file, read file, restore object. Add a small mistake corner: "Only in memory = lost after exit", with a repair arrow pointing to "Write file".
Visible text only: title, In memory, Serialize, JSON, CSV, TXT, Files on disk, Read, Deserialize, Restore objects, Next run, Only in memory = lost after exit. Text must be natural English, large, readable, and attached to the object it explains. Avoid dense code, tiny text, gibberish, fake UI text, and real brand logos.
Style: practical classroom comic plus data-flow illustration with a learner and real operations. Do not make a stack of white rounded boxes, SVG infographic, pure flowchart, or text poster.
""".strip(),
    },
    {
        "filename": "ch02-file-io-serialization-flow-ja.png",
        "title": "ファイル入出力とシリアライズの流れ",
        "suggested_page": "docs/ch02-python/ch02-advanced/03-file-io.md",
        "alt": "ファイル入出力とシリアライズの流れ：メモリ内の Python データをシリアライズしてファイルに保存し、次回起動時に読み込んで復元する。",
        "prompt": """
縦長 9:16 の日本語教材イラスト。タイトルは「プログラム終了後もデータを残すには」。
Python のファイル入出力とシリアライズの章に合わせて、初心者が「メモリ上のデータは一時的なので、シリアライズしてファイルへ書き出し、次回起動時に読み込んで復元する」と理解できる内容にする。
固定構図：上半分は実行中の Python プログラムの作業机。左に「メモリ上のデータ」として list、dict、object の具体例。中央に「シリアライズ」変換マシンがあり、JSON、CSV、TXT のファイルカードへ変換する。右にディスク上のフォルダ。下半分は「次回起動」の逆向きルート：フォルダからファイルを読み込み、「デシリアライズ」を通ってメモリ内のデータへ戻る。一本の連続した矢印で閉ループにする。
4 つの動作を絵として見せる：保存、ファイルへ書き込み、ファイルを読み込み、オブジェクトを復元。小さな失敗コーナーに「メモリだけだと終了時に消える」と入れ、修正ルートを「ファイルへ書き込み」へ向ける。
画像内の文字はタイトル、メモリ上のデータ、シリアライズ、JSON、CSV、TXT、ディスク上のファイル、読み込み、デシリアライズ、オブジェクトを復元、次回起動、メモリだけだと消える、だけ。自然な日本語で、大きく読みやすく、対応する物体の近くに置く。細かいコード、細かい文字、文字化け、偽 UI、実在ロゴは禁止。
スタイルは実用的な授業漫画 + データ流転イラスト。学習者と実際の操作を入れる。白い角丸ボックスの積み重ね、SVG 風インフォグラフィック、ただのフローチャート、文字ポスターにはしない。
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
        "filename": "ch02-hands-on-python-workshop-route.png",
        "title": "第 2 章 Python 跟做工作坊路线图",
        "suggested_page": "docs/ch02-python/ch03-projects/05-hands-on-python-workshop.md",
        "alt": "第 2 章 Python 跟做工作坊路线图：终端命令、argparse、任务对象、JSON 保存、统计和报告导出组成完整小工具。",
        "prompt": """
制作一张 9:16 竖版课程图解，主题是“第 2 章跟着做：从 Python 语法到本地学习任务助手”。
画面像漫画式实操路线页，从上到下展示 6 个步骤：创建项目文件夹、编写 CLI 脚本、解析命令、保存 tasks.json、显示统计、导出 learning_report.md。
重点让新人看懂变量、函数、列表字典、文件、异常和类型标注如何组合成一个能运行的小工具。
可使用极短中文标签：建文件夹、写脚本、解析命令、保存 JSON、统计、导出报告。不要真实品牌 logo，不要水印，不要乱码小字。
""".strip(),
    },
    {
        "filename": "ch02-hands-on-cli-command-flow.png",
        "title": "CLI 命令执行流程图",
        "suggested_page": "docs/ch02-python/ch03-projects/05-hands-on-python-workshop.md",
        "alt": "CLI 命令执行流程图：用户输入 add/list/done/stats/export，argparse 解析后分发到对应函数执行。",
        "prompt": """
制作一张 9:16 竖版分步骤图解，主题是“命令行程序如何从用户输入走到函数执行”。
画面展示终端里输入 add、list、done、stats、export，进入 argparse 解析器，变成 Namespace 参数，再分发到 add_task、list_tasks、complete_task、show_stats、export_report 函数。
用箭头表现代码执行顺序，让新手理解 CLI 不是魔法，而是“输入文本 → 解析 → 调用函数 → 输出结果”。
可以保留 argparse、Namespace、function、CLI 等技术词，配少量中文标签。不要真实品牌 logo，不要乱码。
""".strip(),
    },
    {
        "filename": "ch02-hands-on-json-persistence-flow.png",
        "title": "JSON 文件持久化流程图",
        "suggested_page": "docs/ch02-python/ch03-projects/05-hands-on-python-workshop.md",
        "alt": "JSON 文件持久化流程图：Task 对象列表通过 asdict 转成 JSON，写入 tasks.json，下次启动再读回对象。",
        "prompt": """
制作一张 9:16 竖版教学图，主题是“为什么保存到 JSON 后程序重启还记得任务”。
画面展示 Task dataclass 对象列表，通过 asdict 变成普通 dict/list，再由 json.dumps 写入 tasks.json；下一次程序启动时 json.loads 读回，再变成 Task 对象。
需要突出“内存数据”和“磁盘文件”的前后对比：程序运行时在内存，程序退出后靠 JSON 文件保存状态。
使用少量中文标签，保留 dataclass、asdict、json.dumps、json.loads、tasks.json 等技术词。不要真实品牌 logo，不要乱码。
""".strip(),
    },
    {
        "filename": "ch02-hands-on-error-debug-map.png",
        "title": "Python CLI 常见错误与调试地图",
        "suggested_page": "docs/ch02-python/ch03-projects/05-hands-on-python-workshop.md",
        "alt": "Python CLI 常见错误与调试地图：命令找不到、任务 id 不存在、JSON 损坏、路径错误和空数据分别对应不同检查点。",
        "prompt": """
制作一张 9:16 竖版错误排查漫画流程图，主题是“Python 小工具跑不起来时先查哪里”。
画面分成 5 个新人常见卡点：python3 找不到、任务 id 不存在、tasks.json 损坏、文件路径不对、报告为空。
每个卡点旁边给出视觉化检查动作：看版本、先 list、修 JSON、打印 Path.cwd、先 seed/add 再 export。
风格像故障诊断地图，清晰、温和、不吓人。可使用短中文标签和少量技术词，不要真实品牌 logo，不要乱码。
""".strip(),
    },
    {
        "filename": "ch02-hands-on-evidence-pack.png",
        "title": "Python 项目作品集证据包",
        "suggested_page": "docs/ch02-python/ch03-projects/05-hands-on-python-workshop.md",
        "alt": "Python 项目作品集证据包：脚本、tasks.json、learning_report.md、终端输出、README 和调试记录共同证明项目可运行。",
        "prompt": """
制作一张 9:16 竖版作品集清单图，主题是“Python 阶段不要只说会语法，要留下可运行证据”。
画面像一个整洁项目文件夹展开，包含 learning_assistant_cli.py、tasks.json、learning_report.md、terminal output、README.md、debug_notes.md 六类材料。
用箭头表示从运行命令到保存数据、导出报告、记录错误，形成可检查的项目证据。
风格专业但新人友好，适合课程页结尾。可以保留文件名和 CLI、JSON、README 等标准词。不要真实品牌 logo，不要乱码。
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
        "title": "Plotly 交互式仪表盘图",
        "suggested_page": "docs/ch03-data-analysis/ch04-visualization/03-plotly.md",
        "alt": "Plotly 交互式仪表盘图：筛选器、悬停提示、缩放、动态图表和网页展示帮助探索数据。",
        "prompt": """
一张竖版 9:16 中文教学插画，主题标题：“图表需要被探索，不只是被观看”。
根据 Plotly 交互式可视化章节设计：读者先明白静态图适合汇报结论，交互图适合探索数据、演示变化、放进网页产品。
固定构图：上方放“静态图”的小场景：一张图片图表只能看趋势。中间放“交互式仪表盘”的大场景：学习者在浏览器中操作折线图、柱状图、散点图。用可见手势和鼠标箭头表现四个动作：筛选条件、悬停提示、框选缩放、联动更新。下方放“适合场景”三格：探索异常、课堂演示、网页报告。
必须让画面教会差异：同一份数据，静态图回答一个问题；交互图让读者自己追问下一步。
可见文字只保留：标题、静态图、交互图、筛选、悬停提示、缩放、联动、探索异常、演示、网页报告。中文必须自然清楚；Plotly、dashboard 可以作为小标签保留。字号大，不要英文段落，不要乱码小字。
风格是清晰的课堂漫画 + 产品仪表盘示意，真实操作感强。不要白底圆角框堆叠，不要 SVG 信息图，不要纯文字贴图，不要真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch03-plotly-interactive-dashboard-en.png",
        "title": "Plotly Interactive Dashboard Diagram",
        "suggested_page": "docs/ch03-data-analysis/ch04-visualization/03-plotly.md",
        "alt": "Plotly interactive dashboard diagram: filters, hover tooltips, zoom selection, linked charts, and a web report help users explore data.",
        "prompt": """
A vertical 9:16 English teaching illustration titled: "Charts can be explored, not just viewed".
Design it for the Plotly interactive visualization lesson. The reader should first understand: static charts are good for reporting one conclusion; interactive charts are good for exploring data, demonstrating changes, and embedding in web products.
Fixed composition: top small scene labeled "Static chart": one image chart can only show a trend. Middle large scene labeled "Interactive dashboard": a learner uses a browser dashboard with a line chart, bar chart, and scatter plot. Show four visible actions with cursor/hand gestures: filter, hover tooltip, zoom selection, linked update. Bottom has three use-case panels: find anomalies, classroom demo, web report.
The image must teach the contrast: the same data, a static chart answers one question; an interactive chart lets the reader ask the next question.
Visible text only: title, Static chart, Interactive chart, Filter, Hover tooltip, Zoom, Linked update, Find anomalies, Demo, Web report. Text must be natural English, large, readable, and attached to the visual action. Plotly and dashboard may appear as small technical labels. Avoid dense UI text, gibberish, fake brand logos, and tiny text.
Style: clear classroom comic plus practical product dashboard mockup, with real interaction. Do not make a stack of white rounded boxes, SVG infographic, pure flowchart, or text poster.
""".strip(),
    },
    {
        "filename": "ch03-plotly-interactive-dashboard-ja.png",
        "title": "Plotly インタラクティブダッシュボード図",
        "suggested_page": "docs/ch03-data-analysis/ch04-visualization/03-plotly.md",
        "alt": "Plotly インタラクティブダッシュボード図：フィルタ、ホバー表示、ズーム選択、連動更新、Web レポートでデータを探索する。",
        "prompt": """
縦長 9:16 の日本語教材イラスト。タイトルは「グラフは見るだけでなく、探索できる」。
Plotly のインタラクティブ可視化の章に合わせて、読者が「静的グラフは結論の共有向き、インタラクティブグラフはデータ探索・変化の説明・Web 製品への埋め込み向き」と理解できる内容にする。
固定構図：上部に小さく「静的グラフ」の場面。一枚の画像グラフで傾向を見るだけ。中央に大きく「インタラクティブダッシュボード」の場面。学習者がブラウザ上で折れ線グラフ、棒グラフ、散布図を操作している。カーソルや手の動きで 4 つの操作を見せる：フィルタ、ホバー表示、ズーム選択、連動更新。下部は用途 3 つ：異常を探す、授業デモ、Web レポート。
同じデータでも、静的グラフは 1 つの問いに答え、インタラクティブグラフは次の問いを探せる、という違いが絵で分かるようにする。
画像内の文字はタイトル、静的グラフ、インタラクティブ、フィルタ、ホバー表示、ズーム、連動更新、異常を探す、デモ、Web レポートだけ。自然な日本語で、大きく読みやすく、操作対象の近くに置く。Plotly、dashboard は小さな技術ラベルとして可。細かい UI 文字、文字化け、偽ブランドロゴは禁止。
スタイルは分かりやすい授業漫画 + 実用的な製品ダッシュボード。白い角丸ボックスの積み重ね、SVG 風インフォグラフィック、ただのフローチャート、文字ポスターにはしない。
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
        "filename": "ch03-study-guide-data-loop-vertical.png",
        "title": "数据分析学习指南最小闭环",
        "suggested_page": "docs/ch03-data-analysis/study-guide.md",
        "alt": "数据分析学习指南最小闭环：读取数据、检查质量、清洗问题、转换特征、可视化并解释结论。",
        "prompt": """
一张竖版 9:16 中文教学插画，主题标题：“数据分析练习的最小闭环”。
根据第 3 章学习指南设计：这不是装饰图，而是给学习者做练习时反复对照的检查路线。
固定构图：一张真实数据分析工作台，从上到下形成 6 步循环。每一步既有短标签，也有对应画面动作：1 读取数据（CSV/表格进入工作台），2 检查（放大镜看字段、类型、缺失值），3 清洗（修缺失、重复、异常），4 转换（派生列、分组统计），5 可视化（选择能回答问题的图），6 解释（写一句结论和限制）。右侧用一条回环箭头从“解释”回到“检查”，表示结论不清楚就回看数据。
底部要有一个小提醒：“从原始表到可信结论”。画面必须让读者看懂每一步要产出证据，而不是只背 API。
可见文字只保留：标题、读取、检查、清洗、转换、可视化、解释、字段/缺失值、分组统计、结论、从原始表到可信结论。中文自然清楚；CSV、API 可以保留英文。字号大，不要英文标题，不要乱码小字。
风格是清楚漂亮的课堂漫画 + 操作清单，有学习者、表格、图表和报告。不要白底圆角框堆叠，不要 SVG 信息图，不要纯文字贴图，不要真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch03-study-guide-data-loop-vertical-en.png",
        "title": "Data Analysis Study Loop",
        "suggested_page": "docs/ch03-data-analysis/study-guide.md",
        "alt": "Minimum data analysis study loop: read data, inspect quality, clean problems, transform features, visualize, and explain conclusions.",
        "prompt": """
A vertical 9:16 English teaching illustration titled: "The smallest data analysis practice loop".
Design it for the Chapter 3 study guide. It should be a repeatable checklist learners can use while practicing, not a decorative poster.
Fixed composition: a real data-analysis workbench arranged as a six-step loop from top to bottom. Each step has a short label and a visible action: 1 Read (CSV/table enters the workspace), 2 Inspect (magnifier checks fields, types, missing values), 3 Clean (fix missing values, duplicates, outliers), 4 Transform (derive columns and group statistics), 5 Visualize (choose a chart that answers a question), 6 Explain (write one conclusion and one limitation). A loop arrow returns from Explain to Inspect to show that unclear conclusions require checking the data again.
Bottom reminder: "From raw table to trustworthy conclusion". The image must teach that each step leaves evidence, not just API memorization.
Visible text only: title, Read, Inspect, Clean, Transform, Visualize, Explain, fields / missing values, group statistics, conclusion, From raw table to trustworthy conclusion. Text must be natural English, large, readable, and tied to the visual action. CSV and API may appear as technical labels. Avoid tiny text, gibberish, fake UI text, and real brand logos.
Style: clear and attractive classroom comic plus practical checklist with a learner, tables, charts, and a report. Do not make a stack of white rounded boxes, SVG infographic, pure flowchart, or text poster.
""".strip(),
    },
    {
        "filename": "ch03-study-guide-data-loop-vertical-ja.png",
        "title": "データ分析学習ガイドの最小ループ",
        "suggested_page": "docs/ch03-data-analysis/study-guide.md",
        "alt": "データ分析学習ガイドの最小ループ：データを読み、品質を確認し、問題を清掃し、特徴を変換し、可視化して結論を説明する。",
        "prompt": """
縦長 9:16 の日本語教材イラスト。タイトルは「データ分析練習の最小ループ」。
第 3 章の学習ガイドに合わせて、学習者が練習中に何度も見返すチェックルートとして描く。飾りポスターにはしない。
固定構図：本物のデータ分析ワークベンチを、上から下へ 6 ステップのループとして配置。各ステップは短いラベルと対応する動作で表す：1 読み込む（CSV/表が作業台へ入る）、2 確認する（虫眼鏡で列、型、欠損値を見る）、3 きれいにする（欠損、重複、外れ値を直す）、4 変換する（派生列とグループ集計）、5 可視化する（問いに答えるグラフを選ぶ）、6 説明する（結論と限界を一文で書く）。右側に「説明」から「確認する」へ戻る矢印を入れ、結論が曖昧ならデータを見直すことを示す。
下部の短いリマインダー：「生データから信頼できる結論へ」。各ステップで証拠を残すことが絵で分かるようにする。API 暗記だけに見せない。
画像内の文字はタイトル、読み込む、確認する、きれいにする、変換する、可視化する、説明する、列/欠損値、グループ集計、結論、生データから信頼できる結論へ、だけ。自然な日本語で、大きく読みやすく、対応する動作の近くに置く。CSV、API は技術ラベルとして可。細かい文字、文字化け、偽 UI、実在ロゴは禁止。
スタイルは分かりやすく見栄えのよい授業漫画 + 実用チェックリスト。学習者、表、グラフ、レポートを入れる。白い角丸ボックスの積み重ね、SVG 風インフォグラフィック、ただのフローチャート、文字ポスターにはしない。
""".strip(),
    },
    {
        "filename": "ch03-hands-on-data-workshop-route.png",
        "title": "第 3 章跟做数据工作坊路线图",
        "suggested_page": "docs/ch03-data-analysis/ch06-projects/03-hands-on-data-workshop.md",
        "alt": "第 3 章跟做数据工作坊路线图：脏 CSV 经过清洗、分组统计、SQLite 查询、图表和报告，形成可复现证据。",
        "prompt": """
制作一张 9:16 竖版课程图解，主题是“第 3 章跟着做：从脏 CSV 到可信数据报告”。
画面像漫画式流程课页，按从上到下 6 个步骤展示：原始学习日志 CSV、清洗与记录、按主题分组统计、写入 SQLite、生成 SVG 图表、输出 HTML 报告。
每个步骤都要有清楚的图标、箭头和小型界面卡片，让新人一眼看出这是可以照着操作的实操路线。
文字可以使用极短中文标签：原始 CSV、清洗、分组、SQLite、图表、报告。不要生成真实品牌 logo，不要水印，不要乱码小字。
""".strip(),
    },
    {
        "filename": "ch03-hands-on-cleaning-pipeline.png",
        "title": "学习日志数据清洗与校验流水线",
        "suggested_page": "docs/ch03-data-analysis/ch06-projects/03-hands-on-data-workshop.md",
        "alt": "学习日志数据清洗与校验流水线：缺失分钟、负数分钟、重复记录和主题大小写问题被检查、修复或记录。",
        "prompt": """
制作一张 9:16 竖版教学流程图，主题是“数据清洗不是随便删除，而是检查、处理、记录、复核”。
画面展示一张学习记录表进入清洗工作台，表里有四种新人常见问题：minutes 为空、minutes 为负数、topic 有多余空格和大小写不一致、重复记录。
中间用分步骤卡片表现：读取 CSV、标准化 topic、验证 minutes、去重、写 cleaning_log.json、输出 clean CSV。
风格清晰、像数据诊断漫画分镜，适合放在代码前帮助理解执行顺序。允许极短中文标签，但不要密集小字、乱码或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch03-hands-on-groupby-sql-flow.png",
        "title": "分组统计与 SQLite 查询数据流",
        "suggested_page": "docs/ch03-data-analysis/ch06-projects/03-hands-on-data-workshop.md",
        "alt": "分组统计与 SQLite 查询数据流：清洗后的学习记录按 topic 聚合，并写入 SQLite 用 SQL 查询 Top 主题。",
        "prompt": """
制作一张 9:16 竖版流程对比图，主题是“同一份干净数据，可以用 Python 分组，也可以用 SQLite 查询”。
画面上半部分展示 clean_learning_log.csv 被按 topic 分成 Python、Pandas、Visualization、SQL、RAG 五组，计算学习分钟和平均信心值。
画面下半部分展示同一份数据进入 SQLite 表 learning_logs，再通过 SQL 查询得到 Top 3 主题。
重点表现“groupby”和“SQL GROUP BY”其实都在回答按类别汇总的问题。使用少量中文标签，保留 SQL、SQLite、GROUP BY 等标准技术词。不要真实品牌 logo，不要乱码。
""".strip(),
    },
    {
        "filename": "ch03-hands-on-chart-report-flow.png",
        "title": "图表与报告输出流程",
        "suggested_page": "docs/ch03-data-analysis/ch06-projects/03-hands-on-data-workshop.md",
        "alt": "图表与报告输出流程：分组统计结果生成 SVG 条形图，并嵌入 HTML 报告形成可检查结论。",
        "prompt": """
制作一张 9:16 竖版分步骤图解，主题是“从统计结果到可展示报告”。
画面展示 topic summary 表格进入 SVG 条形图生成器，生成 topic_minutes.svg，再被嵌入 report.html，最终形成包含总时长、清洗行数、Top 主题和结论的报告。
需要突出“图表不是装饰，而是让结论可检查”的教学重点。构图像新人跟做课程截图流，包含表格、条形图、浏览器报告三个视觉层次。
可使用极短中文标签：统计表、SVG 图表、HTML 报告、结论。不要真实品牌 logo，不要水印，不要乱码小字。
""".strip(),
    },
    {
        "filename": "ch03-hands-on-evidence-pack.png",
        "title": "数据分析作品集证据包清单",
        "suggested_page": "docs/ch03-data-analysis/ch06-projects/03-hands-on-data-workshop.md",
        "alt": "数据分析作品集证据包清单：脚本、原始数据、清洗数据、清洗日志、数据库、图表、报告和复盘说明组成可信交付。",
        "prompt": """
制作一张 9:16 竖版教学清单图，主题是“数据分析作品集不只放图表，还要放证据包”。
画面像整洁的项目文件夹展开图，包含 learning_log_pipeline.py、raw CSV、clean CSV、cleaning_log.json、SQLite database、SVG chart、HTML report、short notes 八类材料。
每类材料用图标和小卡片表示，并用箭头说明它们从原始数据一路形成可信结论。
风格专业但新人友好，像作品集交付清单。可以使用短中文标签，保留文件名和 SQLite、CSV、SVG、HTML 等标准词。不要真实品牌 logo，不要乱码。
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
        "filename": "ch04-probability-history-foundations-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "概率统计历史基础地图",
        "suggested_page": "docs/ch04-ai-math/ch02-probability/05-history-foundations.md",
        "alt": "概率统计历史基础地图：Bayes 更新证据，MLE 从数据反推参数，EM 处理隐藏变量，Shannon 信息论度量不确定性。",
        "prompt": """
一张适合 AI 数学概率统计历史小节的教学地图，主题是“从不确定性到可训练模型的统计基础”。
画面分四个清晰节点：Bayes 用新证据更新判断，MLE 从数据反推最可能参数，EM 在隐藏变量下先猜再修，Shannon information theory 度量不确定性并连接 cross entropy。
风格像历史接力地图和数学白板结合，帮助新人把老概念和现代 AI loss、probability、uncertainty 连接起来。
文字不是主体；中文写概念提示，标准术语和公式保留英文或数学形式，例如 Bayes、MLE、EM、entropy、cross entropy。不要整段英文说明、乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch04-hands-on-math-workshop-route.png",
        "size": "1024x1792",
        "quality": "high",
        "title": "第 4 章 AI 数学实操工作坊路线图",
        "suggested_page": "docs/ch04-ai-math/hands-on-math-workshop.md",
        "alt": "第 4 章数学实操工作坊路线图：小数字、向量相似度、概率模拟、熵与 loss、梯度下降和证据包组成可跟做路线。",
        "prompt": """
一张竖版、漫画式、流程式的第 4 章 AI 数学实操工作坊路线图，主题是“把公式变成可运行证据”。
画面从上到下分成 6 个清晰步骤：small numbers、vector similarity、probability simulation、entropy and loss、gradient descent、evidence pack。
每一步像课程卡片，有小型终端、CSV 文件、SVG 图和学习笔记图标，用箭头串起来，强调新人可以一步步跟着做。
文字不是主体；标准术语保留英文，例如 vector similarity、probability simulation、entropy、loss、gradient descent、evidence pack。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch04-hands-on-vector-similarity-flow.png",
        "size": "1024x1792",
        "quality": "high",
        "title": "向量相似度证据流图",
        "suggested_page": "docs/ch04-ai-math/hands-on-math-workshop.md",
        "alt": "向量相似度证据流图：query vector 与 topic vectors 经过 dot、norm、cosine similarity 和 distance 后输出最相似主题。",
        "prompt": """
一张竖版教学流程图，主题是“向量相似度不是公式背诵，而是回答哪个对象更像”。
画面上方是 query vector，下面分三条 topic vector 支线；每条支线经过 dot product、norm、cosine similarity、distance 四个小检查点。
底部输出 ranking table，并标出 best match。旁边用箭头解释它会出现在 embedding retrieval、recommendation、RAG similarity 中。
文字不是主体；标准术语保留英文，例如 query vector、topic vector、dot product、norm、cosine similarity、distance、best match、embedding。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch04-hands-on-probability-simulation-flow.png",
        "size": "1024x1792",
        "quality": "high",
        "title": "概率模拟与不确定性流程图",
        "suggested_page": "docs/ch04-ai-math/hands-on-math-workshop.md",
        "alt": "概率模拟与不确定性流程图：true probability、random samples、batch rate、running rate 和 expected probability 展示采样波动。",
        "prompt": """
一张竖版分步骤图，主题是“概率要通过重复样本观察，不是看一次结果”。
画面展示 true probability 进入 random sampler，产生多个 batch cards；每个 batch 计算 batch_rate，随后进入 running_rate 折线图。
用一条虚线表示 expected probability，显示观测比例围绕它上下波动并逐渐稳定。
文字不是主体；标准术语保留英文，例如 true probability、random samples、batch_rate、running_rate、expected probability、uncertainty、sample size。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch04-hands-on-gradient-descent-loop.png",
        "size": "1024x1792",
        "quality": "high",
        "title": "梯度下降执行循环图",
        "suggested_page": "docs/ch04-ai-math/hands-on-math-workshop.md",
        "alt": "梯度下降执行循环图：x 进入 loss function，计算 gradient，乘 learning rate，更新 x 并重复直到 loss 下降。",
        "prompt": """
一张竖版循环流程图，主题是“梯度下降是模型训练的最小节奏”。
画面表现参数 x 进入 loss function，输出 loss；再计算 gradient，乘 learning_rate，得到 update step，回到新的 x。
旁边有小型 loss curve，从高到低下降；再用几个点展示 step 0、step 1、step 2 到 final。
文字不是主体；标准术语保留英文，例如 x、loss function、gradient、learning_rate、update step、loss curve、final loss。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch04-hands-on-evidence-pack.png",
        "size": "1024x1792",
        "quality": "high",
        "title": "第 4 章数学证据包图",
        "suggested_page": "docs/ch04-ai-math/hands-on-math-workshop.md",
        "alt": "第 4 章数学证据包图：README、vector_similarity.csv、probability_simulation.csv、gradient_descent.csv、math_cards.md 和 SVG 图组成可复盘证据。",
        "prompt": """
一张竖版作品集证据包图，主题是“跑完数学工作坊后要留下什么，才能证明自己真的会用”。
画面中心是 ch04_math_workshop_evidence 文件夹，里面有 README.md、vector_similarity.csv、probability_simulation.csv、gradient_descent.csv、math_cards.md、vector_similarity.svg、probability_simulation.svg、gradient_descent.svg。
每个文件连接到用途：重新运行命令、相似度证据、不确定性证据、优化轨迹、模型语言卡片、视觉复盘。
底部展示 reviewer 可以回答三个问题：计算了什么、发生了什么变化、它支持什么模型概念。
文字不是主体；标准术语保留英文，例如 README.md、CSV、SVG、math_cards.md、reviewer、evidence。其他说明尽量用中文短标签。不要乱码小字或真实品牌 logo。
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
        "filename": "ch05-ml-history-breakthrough-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "机器学习历史突破地图",
        "suggested_page": "docs/ch05-machine-learning/ch01-ml-basics/04-history-breakthroughs.md",
        "alt": "机器学习历史突破地图：Bayes、MLE、EM、线性模型、决策树、SVM、随机森林、Boosting、XGBoost 和 sklearn 工程化逐步连接。",
        "prompt": """
一张适合机器学习历史突破小节的技术演进地图，主题是“经典机器学习如何从统计推断走向可复盘工程流程”。
画面从 Bayes、MLE、EM 开始，连接到 linear model、decision tree、SVM margin、Random Forest、Boosting/XGBoost，最后汇入 sklearn Pipeline 和模型评估闭环。
风格像课程历史路线图和项目建模流程结合，层次清楚、新人友好。
文字不是主体；中文写概念提示，标准术语保留英文，例如 Bayes、MLE、EM、SVM、Random Forest、Boosting、XGBoost、sklearn Pipeline。不要整段英文说明、乱码小字或真实品牌 logo。
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
        "alt": "监督学习章节关系图：线性回归、逻辑回归、决策树、集成学习和 SVM 最大间隔路线由简单到复杂逐步连接。",
        "prompt": """
一张适合监督学习导读页的章节关系图，主题是“从最简单模型到更强模型的监督学习主线”。
画面表现带标签数据进入模型学习路径：线性回归预测连续值，逻辑回归输出分类概率，决策树做规则分裂，集成学习把多个模型组合成更稳结果，SVM 用最大间隔理解稳健分类边界。
风格像课程路线图和模型进化图结合，清晰、有层次。
文字不是主体；如确实需要标签，中英文自然混用：中文写概念提示，标准术语保留英文，例如 Regression、Classification、Bagging、Boosting、SVM、margin。不要整张图全英文，不要乱码小字或真实品牌 logo。
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
        "filename": "ch05-svm-margin-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "SVM 最大间隔直觉图",
        "suggested_page": "docs/ch05-machine-learning/ch02-supervised/05-svm.md",
        "alt": "SVM 最大间隔直觉图：支持向量决定分类边界，最大间隔让分界线离两类样本都更远，核方法帮助处理非线性边界。",
        "prompt": """
一张适合 SVM 教学页的最大间隔直觉图，主题是“分类边界不仅要分对，还要离样本足够远”。
画面表现二维分类样本，两类点之间有一条清晰 decision boundary，两侧有 margin 安全带，最近的点被标成 support vectors；旁边用小插图表现 kernel trick 把非线性数据换到更容易分开的空间。
风格像机器学习白板讲解图，边界、间隔和支持向量要一眼清楚。
文字不是主体；中文写概念提示，标准术语保留英文，例如 SVM、margin、support vectors、kernel trick。不要整段英文说明、乱码小字或真实品牌 logo。
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
        "filename": "ch05-data-split-leakage-guardrail.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "训练验证测试与数据泄漏护栏图",
        "suggested_page": "docs/ch05-machine-learning/ch01-ml-basics/01-what-is-ml.md",
        "alt": "训练验证测试与数据泄漏护栏图：训练集学习、验证集选方案、测试集最终验收，预处理必须只在训练数据中 fit。",
        "prompt": """
一张适合机器学习新人课程的防泄漏流程图，主题是“训练集、验证集、测试集之间要有护栏”。
画面表现完整数据被切成 train、validation、test 三块；train 用来 fit 模型和预处理器，validation 用来选模型和调参数，test 被锁在最后只验收一次。
旁边用红色警示标出错误做法：先对全量数据标准化、先用全量数据选特征、反复看测试集调参。
风格像数据安全闸门和课程白板结合，清晰、新手友好。
文字不是主体；标准术语和 API 保留英文，例如 train、validation、test、fit、transform、leakage。中文写短提示。不要整段英文说明、乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch05-sklearn-pipeline-anatomy.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "sklearn Pipeline 组件拆解图",
        "suggested_page": "docs/ch05-machine-learning/ch01-ml-basics/02-sklearn-intro.md",
        "alt": "sklearn Pipeline 组件拆解图：Transformer 负责 fit/transform，Estimator 负责 fit/predict，Pipeline 保证训练和预测流程一致。",
        "prompt": """
一张适合 sklearn 入门页的组件拆解图，主题是“Estimator、Transformer、Pipeline 三个角色怎么配合”。
画面像一台透明的机器学习装配线：左侧输入数据表 X/y，中间有 Transformer 零件负责 fit/transform，Estimator 零件负责 fit/predict，外层 Pipeline 把步骤锁成同一条可复现流程。
加入一个小对比：训练时 fit + transform，预测时只 transform + predict，帮助新人理解为什么测试集不能重新 fit。
风格工程化、模块清楚、温和教学感。
文字不是主体；API 和变量名保留英文，例如 X、y、fit、transform、predict、score、Pipeline。中文写短标签。不要整段英文说明、乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch05-linear-regression-residual-diagnostics.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "线性回归残差诊断图",
        "suggested_page": "docs/ch05-machine-learning/ch02-supervised/01-linear-regression.md",
        "alt": "线性回归残差诊断图：随机残差、弯曲模式、漏斗形误差和异常点分别提示模型、特征或指标需要调整。",
        "prompt": """
一张适合线性回归课程的残差诊断教学图，主题是“回归模型不要只看 R²，要看残差在说什么”。
画面分成四个小面板：随机散开的残差表示较健康；残差呈弯曲模式提示欠拟合或缺少非线性特征；残差呈漏斗形提示方差不稳定；少数极端点提示异常值或特殊样本。
每个面板旁边用小箭头连接到下一步动作：加特征、变换目标、检查异常、换指标。
风格像统计诊断白板，清晰、直观、适合新人。
文字不是主体；标准术语保留英文，例如 residual、R²、RMSE、MAE。中文写短提示。不要整段英文说明、乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch05-logistic-threshold-tradeoff.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "逻辑回归阈值权衡图",
        "suggested_page": "docs/ch05-machine-learning/ch02-supervised/02-logistic-regression.md",
        "alt": "逻辑回归阈值权衡图：模型输出概率，阈值下降提高召回但增加误报，阈值上升提高精确率但增加漏报。",
        "prompt": """
一张适合逻辑回归课程的阈值权衡图，主题是“概率输出如何变成业务决策”。
画面表现模型输出一排从 0 到 1 的概率刻度，阈值线可以左右移动；阈值低时召回更多正例但误报增加，阈值高时预测更谨慎但漏报增加。
旁边用一个小混淆矩阵和 Precision/Recall 天平辅助说明。
风格像交互式仪表盘和机器学习白板结合，直观、新人友好。
文字不是主体；标准术语保留英文，例如 probability、threshold、Precision、Recall、F1。中文写短提示。不要整段英文说明、乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch05-tree-pruning-overfit-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "决策树过拟合与剪枝图",
        "suggested_page": "docs/ch05-machine-learning/ch02-supervised/03-decision-trees.md",
        "alt": "决策树过拟合与剪枝图：过深的树把噪声切碎，预剪枝和后剪枝控制深度、叶子样本数和 ccp_alpha。",
        "prompt": """
一张适合决策树课程的过拟合与剪枝教学图，主题是“树越深越像记忆训练集，剪枝让它回到规律”。
画面左侧是一棵过深的树和碎片化决策边界，孤立噪声点被切成小区域；右侧是一棵剪枝后的树，边界更平滑、叶子更稳。
中间放剪刀或修枝工具，标出 max_depth、min_samples_leaf、ccp_alpha 三个关键控制点。
风格像园艺修枝和 ML 白板结合，形象但专业。
文字不是主体；标准术语保留英文，例如 max_depth、min_samples_leaf、ccp_alpha、overfit。中文写短提示。不要整段英文说明、乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch05-ensemble-error-correction-lab.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "集成学习纠错实验室图",
        "suggested_page": "docs/ch05-machine-learning/ch02-supervised/04-ensemble-learning.md",
        "alt": "集成学习纠错实验室图：Bagging 通过并行投票降低方差，Boosting 通过串行纠错降低偏差。",
        "prompt": """
一张适合集成学习课程的双路线实验室图，主题是“Bagging 是多人投票，Boosting 是连续订正”。
左侧表现 Bagging：多棵树并行、各自看到不同样本和特征，最后投票或平均，目标是降低 variance。
右侧表现 Boosting：第 1 个弱模型后标出错误样本，后续模型逐轮关注残差或错例，目标是降低 bias。
风格像数据科学实验室和流程白板结合，左右对比强。
文字不是主体；标准术语保留英文，例如 Bagging、Boosting、Random Forest、GBDT、variance、bias。中文写短提示。不要整段英文说明、乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch05-clustering-shape-selection-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "聚类数据形状与算法选择图",
        "suggested_page": "docs/ch05-machine-learning/ch03-unsupervised/01-clustering.md",
        "alt": "聚类数据形状与算法选择图：圆团状簇适合 K-Means，弯曲和噪声适合 DBSCAN，层级结构适合层次聚类。",
        "prompt": """
一张适合聚类课程的算法选择图，主题是“先看数据形状，再选聚类算法”。
画面分成三类二维数据形状：圆团状且大小相近，箭头指向 K-Means；弯月形或不规则形状且带噪声，箭头指向 DBSCAN；有层级嵌套关系，箭头指向 Hierarchical clustering。
旁边强调聚类没有唯一真相，需要结合 silhouette 和业务解释。
风格像数据点地图和选择指南结合，清晰、新人友好。
文字不是主体；标准术语保留英文，例如 K-Means、DBSCAN、Hierarchical、silhouette。中文写短提示。不要整段英文说明、乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch05-pca-explained-variance-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "PCA 方差解释比读图指南",
        "suggested_page": "docs/ch05-machine-learning/ch03-unsupervised/02-dimensionality-reduction.md",
        "alt": "PCA 方差解释比读图指南：主成分越往后新增信息越少，累计方差曲线的拐点帮助选择 n_components。",
        "prompt": """
一张适合 PCA 降维课程的方差解释比图解，主题是“如何读累计方差曲线选择主成分数量”。
画面左侧是高维数据云投影到 PC1、PC2、PC3；右侧是 scree plot 和 cumulative explained variance 曲线，标出 90%、95% 阈值和曲线拐点。
底部用三枚小卡片提示：压缩率、信息保留、下游模型验证三者一起决定 n_components。
风格像数学白板和数据可视化结合，干净、准确。
文字不是主体；标准术语保留英文，例如 PCA、PC1、PC2、explained variance、n_components。中文写短提示。不要整段英文说明、乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch05-anomaly-method-comparison-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "异常检测方法对比图",
        "suggested_page": "docs/ch05-machine-learning/ch03-unsupervised/03-anomaly-detection.md",
        "alt": "异常检测方法对比图：Z-score 和 IQR 适合低维极端值，Isolation Forest 适合高维孤立点，LOF 适合局部密度异常。",
        "prompt": """
一张适合异常检测课程的方法对比图，主题是“不同异常长相，对应不同检测方法”。
画面分成四块：Z-score/IQR 检测一维极端值；Isolation Forest 用随机切分快速孤立异常点；LOF 发现局部密度很低的点；One-Class SVM 学习正常边界。
每块都用简洁数据点小图表达异常形态，并用箭头连接到适用场景。
风格像风险雷达和 ML 方法卡片结合，清晰、实用。
文字不是主体；标准术语保留英文，例如 Z-score、IQR、Isolation Forest、LOF、One-Class SVM。中文写短提示。不要整段英文说明、乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch05-threshold-roc-pr-curve-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "阈值、ROC 与 PR 曲线读图指南",
        "suggested_page": "docs/ch05-machine-learning/ch04-evaluation/01-metrics.md",
        "alt": "阈值、ROC 与 PR 曲线读图指南：混淆矩阵定位错误，阈值曲线展示 Precision/Recall 权衡，PR 曲线更适合不平衡任务。",
        "prompt": """
一张适合模型评估课程的分类指标读图指南，主题是“从混淆矩阵到阈值，再到 ROC/PR 曲线”。
画面从左到右：混淆矩阵标出 TP、FP、FN、TN；中间是可移动 threshold 影响 Precision 和 Recall 的曲线；右侧是 ROC 曲线和 PR 曲线对比，强调不平衡数据更重视 PR。
风格像评估仪表盘和课程白板结合，层次清楚。
文字不是主体；标准术语保留英文，例如 TP、FP、FN、TN、threshold、ROC、AUC、PR、Precision、Recall。中文写短提示。不要整段英文说明、乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch05-cv-leakage-safe-pipeline-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "交叉验证防泄漏 Pipeline 图",
        "suggested_page": "docs/ch05-machine-learning/ch04-evaluation/02-cross-validation.md",
        "alt": "交叉验证防泄漏 Pipeline 图：每一折只在训练折 fit 预处理器，再 transform 验证折，避免全量预处理造成数据泄漏。",
        "prompt": """
一张适合交叉验证课程的防泄漏流程图，主题是“K-fold 中预处理必须包在 Pipeline 里”。
画面表现 5 折交叉验证，每一轮训练折内部 fit scaler/PCA/selector，验证折只 receive transform；外侧有 Pipeline 护栏保护流程。
旁边用红色叉号展示错误做法：先对全量数据 fit 预处理器，再做 K-fold。
风格像数据切分条带和安全护栏结合，直观、清晰。
文字不是主体；标准术语和 API 保留英文，例如 K-fold、Pipeline、fit、transform、scaler、PCA、leakage。中文写短提示。不要整段英文说明、乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch05-learning-curve-diagnosis-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "学习曲线诊断图",
        "suggested_page": "docs/ch05-machine-learning/ch04-evaluation/03-bias-variance.md",
        "alt": "学习曲线诊断图：训练和验证都低提示欠拟合，训练高验证低提示过拟合，验证曲线继续上升说明增加数据可能有帮助。",
        "prompt": """
一张适合偏差方差课程的学习曲线诊断图，主题是“看训练曲线和验证曲线决定下一步动作”。
画面分成三个小面板：训练和验证分数都低表示 high bias/欠拟合；训练高验证低且间距大表示 high variance/过拟合；验证分数随样本量继续上升表示增加数据可能有效。
每个面板下方连接动作建议：加特征或复杂度、加正则化或数据、继续收集数据。
风格像医生诊断卡和 ML 曲线结合，简洁、直观。
文字不是主体；标准术语保留英文，例如 learning curve、train score、validation score、bias、variance。中文写短提示。不要整段英文说明、乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch05-search-space-budget-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "超参数搜索空间与预算图",
        "suggested_page": "docs/ch05-machine-learning/ch04-evaluation/04-hyperparameter-tuning.md",
        "alt": "超参数搜索空间与预算图：参数维度越多组合越爆炸，先从关键超参数和可控预算开始，再逐步扩大搜索范围。",
        "prompt": """
一张适合超参数调优课程的搜索空间与预算图，主题是“调参不是乱搜，是在预算内设计实验”。
画面表现一个二维或三维搜索空间，Grid Search 像规则棋盘，Random Search 像随机撒点，Bayesian Search 像逐步靠近高分区域；旁边有计算预算沙漏和组合爆炸警示。
底部突出新人策略：先 baseline，先少数关键参数，先 cv，再最终 test。
风格像实验控制台和参数地图结合，专业、新手友好。
文字不是主体；标准术语保留英文，例如 baseline、Grid Search、Random Search、Bayesian、CV、test。中文写短提示。不要整段英文说明、乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch05-feature-leakage-red-flags-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "特征泄漏红旗检查图",
        "suggested_page": "docs/ch05-machine-learning/ch05-feature-engineering/01-feature-understanding.md",
        "alt": "特征泄漏红旗检查图：预测时刻之后才产生、由目标派生、与目标几乎完美相关或只在线下存在的字段都是高风险特征。",
        "prompt": """
一张适合特征工程课程的泄漏风险检查图，主题是“分数高得离谱时，先查特征泄漏”。
画面像数据侦探红旗板：预测时刻之后产生的字段、由 target 后续结果派生的字段、和 target 几乎完美相关的字段、线上预测时拿不到的字段，都被贴上红旗。
中间是一张数据表和时间线，强调预测时刻 cutoff。
风格像侦探线索墙和数据表结合，清楚、有记忆点。
文字不是主体；标准术语保留英文，例如 target、leakage、cutoff、baseline。中文写短提示。不要整段英文说明、乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch05-columntransformer-real-table-pipeline.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "真实表格数据 ColumnTransformer Pipeline 图",
        "suggested_page": "docs/ch05-machine-learning/ch05-feature-engineering/05-pipeline.md",
        "alt": "真实表格数据 ColumnTransformer Pipeline 图：数值、类别和自定义特征分流处理后合并，再和模型一起进入交叉验证或 GridSearch。",
        "prompt": """
一张适合 sklearn Pipeline 课程的真实表格流水线图，主题是“不同列走不同处理线，最后合成一个可复现模型流程”。
画面从一张 Titanic 风格的表格开始，数值列进入 imputer + scaler，类别列进入 imputer + one-hot，组合特征进入 custom transformer，最后由 ColumnTransformer 合并并交给 classifier。
外层连接 cross-validation 和 GridSearch，强调完整 Pipeline 一起评估。
风格像工程架构图和教学插图结合，模块清晰。
文字不是主体；标准术语和 API 保留英文，例如 ColumnTransformer、Pipeline、imputer、scaler、one-hot、classifier、GridSearch。中文写短提示。不要整段英文说明、乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch05-hands-on-ml-workshop-route.png",
        "size": "1024x1792",
        "quality": "high",
        "title": "第 5 章机器学习实操工作坊路线图",
        "suggested_page": "docs/ch05-machine-learning/ch06-projects/05-hands-on-ml-workshop.md",
        "alt": "第 5 章机器学习实操工作坊路线图：从数据表到 baseline、Pipeline、模型评估、阈值复盘、错误分析和 README 证据包。",
        "prompt": """
一张适合第 5 章机器学习综合实操工作坊的竖版路线图，主题是“从数据表到可复现 ML 证据包”。
画面从上到下分成 7 个可跟做步骤：define task、split data、baseline、Pipeline、metrics、error analysis、README evidence。
旁边小卡片显示 Dummy baseline、Logistic Regression、Random Forest、threshold review、leakage check。
风格像跟做课程的分步骤漫画流程，竖版、清晰、适合新人。
文字不是主体；标准术语保留英文，例如 baseline、Pipeline、F1、AUC、threshold、error samples、README。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch05-hands-on-evidence-pipeline.png",
        "size": "1024x1792",
        "quality": "high",
        "title": "机器学习实操证据流水线图",
        "suggested_page": "docs/ch05-machine-learning/ch06-projects/05-hands-on-ml-workshop.md",
        "alt": "机器学习实操证据流水线图：generated table、schema、model comparison、threshold review、error samples、leakage check 和 README 串成证据链。",
        "prompt": """
一张适合机器学习实操页的竖版证据流水线图，主题是“每一步都落到可检查文件”。
画面表现 generated table、schema.json、model_comparison.csv、best_model_metrics.json、threshold_review.csv、error_samples.csv、leakage_check.md、experiment_log.md、README.md 串成证据链。
用箭头强调从数据、训练、评估、阈值、错误分析到作品集交付的流转。
风格像文件夹流程图和课程漫画结合，竖版、分步骤、清楚实用。
文字不是主体；标准术语和文件名保留英文。中文只用短提示。不要整段英文说明、乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch05-hands-on-data-split-lab.png",
        "size": "1024x1792",
        "quality": "high",
        "title": "训练测试划分实操护栏图",
        "suggested_page": "docs/ch05-machine-learning/ch06-projects/05-hands-on-ml-workshop.md",
        "alt": "训练测试划分实操护栏图：target 列留作答案，X 特征进入 train/test split，训练集 fit 预处理和模型，测试集只做最终验收。",
        "prompt": """
一张适合第 5 章机器学习实操工作坊的竖版教学图，主题是“先把数据角色分清楚，再训练模型”。
画面从一张 learning_tasks.csv 数据表开始，突出 target column delayed 作为答案，其他列进入 X features。
随后数据进入 train/test split：train split 用来 fit imputer、scaler、encoder 和 model；test split 被放在带锁的 final check 区域，只用于最后 evaluate。
用红色警示卡表现错误做法：target enters X、fit scaler on all data、tune on test set。
风格像分步骤漫画流程和数据科学白板结合，竖版、清晰、适合新手跟做。
文字不是主体；标准术语保留英文，例如 X、y、target、train/test split、fit、evaluate、leakage。中文只用短提示。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch05-hands-on-pipeline-training-flow.png",
        "size": "1024x1792",
        "quality": "high",
        "title": "ColumnTransformer 与 Pipeline 训练流程图",
        "suggested_page": "docs/ch05-machine-learning/ch06-projects/05-hands-on-ml-workshop.md",
        "alt": "ColumnTransformer 与 Pipeline 训练流程图：数值列和类别列分别预处理后合并，再进入 Dummy、Logistic Regression 和 Random Forest 比较。",
        "prompt": """
一张适合机器学习综合实操的竖版流程图，主题是“Pipeline 把预处理和模型训练绑在一起，避免手滑泄漏”。
画面展示 numeric features 进入 median imputer 和 StandardScaler，categorical features 进入 most_frequent imputer 和 OneHotEncoder(handle_unknown ignore)，两路汇入 ColumnTransformer。
ColumnTransformer 再和三个模型分支相连：Dummy baseline、Logistic Regression、Random Forest，最后进入 cross_validate 和 model_comparison.csv。
强调同一套 Pipeline 在训练、验证和测试中复用。
风格像工程流水线和课程漫画结合，模块分明、竖版。
文字不是主体；API 和文件名保留英文，例如 ColumnTransformer、Pipeline、StandardScaler、OneHotEncoder、cross_validate、model_comparison.csv。中文只用短提示。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch05-hands-on-threshold-decision-lab.png",
        "size": "1024x1792",
        "quality": "high",
        "title": "阈值与指标决策实验图",
        "suggested_page": "docs/ch05-machine-learning/ch06-projects/05-hands-on-ml-workshop.md",
        "alt": "阈值与指标决策实验图：概率输出经过不同 threshold 后改变 precision、recall、F1 和 flagged_students。",
        "prompt": """
一张适合第 5 章实操工作坊的竖版决策实验图，主题是“阈值不是默认值，而是项目选择”。
画面上方是 Logistic Regression 输出的一排 probability scores；中间有可移动 threshold slider，展示 0.30、0.50、0.70 三个位置。
每个位置连接到小指标卡：Precision、Recall、F1、flagged_students，表现阈值降低时召回提高但误报增加，阈值升高时预测更谨慎但漏报可能增加。
底部展示业务规则卡：choose recall target first。
风格像交互式仪表盘和教学漫画结合，竖版、清楚。
文字不是主体；标准术语保留英文，例如 probability、threshold、Precision、Recall、F1、flagged_students。中文只用短提示。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch05-hands-on-error-bucket-review.png",
        "size": "1024x1792",
        "quality": "high",
        "title": "错误样本分桶复盘图",
        "suggested_page": "docs/ch05-machine-learning/ch06-projects/05-hands-on-ml-workshop.md",
        "alt": "错误样本分桶复盘图：error_samples.csv 被分成 false positive 和 false negative，再按 track、study_mode 和特征模式复盘。",
        "prompt": """
一张适合机器学习实操页的竖版错误分析图，主题是“不要只看分数，要把错例分桶”。
画面从 error_samples.csv 文件进入 error analysis table，分成 false positive 和 false negative 两个桶。
每个桶继续连接到 track、study_mode、high quiz low practice、many forum questions 等小标签，最后输出 next feature idea、threshold adjustment、data quality check 三类行动。
强调错误样本是下一轮实验的入口。
风格像侦探看板和数据表复盘结合，竖版、清晰、新手友好。
文字不是主体；标准术语和文件名保留英文，例如 error_samples.csv、false positive、false negative、track、study_mode、next action。中文只用短提示。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch05-hands-on-rerun-experiment-loop.png",
        "size": "1024x1792",
        "quality": "high",
        "title": "机器学习实验复跑闭环图",
        "suggested_page": "docs/ch05-machine-learning/ch06-projects/05-hands-on-ml-workshop.md",
        "alt": "机器学习实验复跑闭环图：一次只改一个变量，rerun script，比较 CV/Test 指标，记录 experiment_log，再决定保留或回滚。",
        "prompt": """
一张适合第 5 章工作坊结尾的竖版实验闭环图，主题是“记录过的复跑才是建模实践”。
画面从 experiment idea 开始，进入 change one variable，例如 max_depth=8 或 threshold target；随后 rerun script，compare CV F1 and test F1，inspect errors，update experiment_log.md，最后分支为 keep 或 rollback。
底部展示 README next steps 和 portfolio evidence。
风格像项目迭代看板和数据科学实验日志结合，竖版、专业但新手友好。
文字不是主体；标准术语和文件名保留英文，例如 experiment_log.md、change one variable、rerun、CV F1、test F1、keep、rollback。中文只用短提示。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch05-hands-on-code-execution-sequence.png",
        "size": "1024x1792",
        "quality": "high",
        "title": "ML 工作坊代码执行顺序图",
        "suggested_page": "docs/ch05-machine-learning/ch06-projects/05-hands-on-ml-workshop.md",
        "alt": "ML 工作坊代码执行顺序图：main 函数依次重置目录、生成数据、划分数据、构建 Pipeline、训练模型、保存报告和打印结果。",
        "prompt": """
一张竖版代码执行顺序图，主题是“main() 如何串起完整 ML 项目”。
步骤从上到下：reset workspace、make dataset、train/test split、build ColumnTransformer、train Dummy/Logistic/RandomForest、cross_validate、choose best、threshold table、error samples、reports、print expected output。
画面像代码调用栈和流程看板结合，突出每个函数输出的文件。
风格清晰、适合新人跟着代码运行和定位卡点。
文字不是主体；函数名、API、文件名保留英文。中文只用短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch05-hands-on-debug-loop.png",
        "size": "1024x1792",
        "quality": "high",
        "title": "ML 工作坊常见错误排查闭环图",
        "suggested_page": "docs/ch05-machine-learning/ch06-projects/05-hands-on-ml-workshop.md",
        "alt": "ML 工作坊常见错误排查闭环图：先查环境、泄漏、随机种子、类别不平衡、未知类别和阈值，再回到证据文件。",
        "prompt": """
一张竖版常见错误排查闭环图，主题是“先查泄漏、再查指标、再查阈值和错误样本”。
故障卡片包括：ModuleNotFoundError、leakage high score、unstable split、accuracy high F1 poor、unknown category、low recall。
每个故障卡连到动作：install packages、remove target from X、set random_state、check confusion matrix、handle_unknown ignore、review threshold_review.csv。
风格像排错流程漫画和工程检查表结合，红色警示但不吓人，适合新人。
文字不是主体；错误名、API、文件名保留英文。中文只用短提示。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch05-hands-on-portfolio-pack.png",
        "size": "1024x1792",
        "quality": "high",
        "title": "机器学习作品集证据包图",
        "suggested_page": "docs/ch05-machine-learning/ch06-projects/05-hands-on-ml-workshop.md",
        "alt": "机器学习作品集证据包图：README、运行命令、baseline 指标、模型对比、阈值复盘、错误样本、泄漏检查和下一步计划组成可复现交付。",
        "prompt": """
一张竖版作品集证据包图，主题是“第 5 章项目交付要让别人能复现、能质疑、能继续改”。
画面像一个打开的项目文件夹和作品集页面，文件夹卡片包括 README、运行命令、baseline 指标、模型对比、阈值复盘、错误样本、泄漏检查、下一步计划。
强调证据不是装饰，而是项目交付的一部分。
风格专业、干净、有作品集质感，适合放在实操课程末尾。
文字不是主体；标准术语和文件名保留英文。中文尽量用短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch05-project-report-storyboard.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "机器学习项目报告故事板",
        "suggested_page": "docs/ch05-machine-learning/ch06-projects/00-roadmap.md",
        "alt": "机器学习项目报告故事板：问题定义、数据说明、baseline、指标、模型对比、错误分析、结论和下一步计划组成作品集报告。",
        "prompt": """
一张适合机器学习项目实战章节的报告故事板，主题是“项目交付不是贴分数，而是讲清建模证据链”。
画面像作品集页面草图，包含问题定义、数据说明、baseline、评估指标、模型对比、错误样本、结论解释、下一步计划八个卡片模块。
中间用一条线串起“从业务问题到可复盘报告”的路径。
风格专业、清晰、有作品集质感，帮助新人知道报告该放什么。
文字不是主体；标准术语保留英文，例如 baseline、metric、error analysis、README、report。中文写短提示。不要整段英文说明、乱码小字或真实品牌 logo。
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
        "title": "机器学习阶段实操清单图",
        "suggested_page": "docs/ch05-machine-learning/study-guide.md",
        "alt": "机器学习通关任务清单图：回归、分类、聚类、评估和特征工程任务组成阶段通关作品。",
        "prompt": """
一张中文机器学习阶段实操清单插图，主题是“完成这些小任务，就真正入门机器学习”。
画面用任务卡片展示：训练一个回归模型、训练一个分类模型、做一次聚类分析、画混淆矩阵和学习曲线、搭建 Pipeline、完成一个项目报告。
风格现代、清晰、有成就感，适合放在学习指南里的任务清单区域。
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
        "filename": "ch06-dl-history-breakthrough-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "深度学习历史突破地图",
        "suggested_page": "docs/ch06-deep-learning/ch01-nn-basics/06-history-breakthroughs.md",
        "alt": "深度学习历史突破地图：感知器、XOR、反向传播、新认知机、梯度消失、LSTM、RBM/DBN、AlexNet、ResNet、Transformer 逐步连接。",
        "prompt": """
一张适合深度学习历史突破小节的技术演进地图，主题是“神经网络三次浪潮如何一步步走到 Transformer”。
画面按时间线展示 Perceptron、XOR setback、Backpropagation、Neocognitron/CNN idea、vanishing gradient、LSTM、RBM/DBN revival、AlexNet/ImageNet、ResNet、Attention/Transformer。
风格像历史接力赛和神经网络结构地图结合，突出每个节点解决上一代什么问题。
文字不是主体；中文写概念提示，标准术语保留英文，例如 Perceptron、XOR、Backpropagation、LSTM、AlexNet、ResNet、Transformer。不要整段英文说明、乱码小字或真实品牌 logo。
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
        "filename": "ch06-hands-on-dl-workshop-route.png",
        "size": "1024x1792",
        "quality": "high",
        "title": "第 6 章 PyTorch 实操工作坊路线图",
        "suggested_page": "docs/ch06-deep-learning/ch08-projects/04-hands-on-dl-workshop.md",
        "alt": "第 6 章 PyTorch 实操工作坊路线图：从 shape trace、Dataset、DataLoader、baseline、CNN、训练曲线、checkpoint 到 README 证据包。",
        "prompt": """
一张适合第 6 章深度学习综合实操工作坊的竖版路线图，主题是“从 tensor shape 到 PyTorch 训练证据包”。
画面从上到下分成 8 个可跟做步骤：shape trace、Dataset、DataLoader、Flatten baseline、Tiny CNN、training loop、validation curve、checkpoint + README evidence。
旁边小卡片显示 CrossEntropyLoss、Adam、loss_curve.png、model_comparison.csv、error_samples.csv。
风格像跟做课程的分步骤漫画流程，竖版、清晰、适合新人。
文字不是主体；标准术语保留英文，例如 tensor、Dataset、DataLoader、CNN、loss、validation、checkpoint、README。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch06-hands-on-training-evidence-pipeline.png",
        "size": "1024x1792",
        "quality": "high",
        "title": "深度学习训练证据流水线图",
        "suggested_page": "docs/ch06-deep-learning/ch08-projects/04-hands-on-dl-workshop.md",
        "alt": "深度学习训练证据流水线图：training_log、model comparison、confusion matrix、review samples、loss curve、checkpoint 和 README 串成证据链。",
        "prompt": """
一张适合 PyTorch 实操页的竖版证据流水线图，主题是“训练不是 done，而是每一步都有证据”。
画面表现 synthetic images、shape_trace.md、training_log.csv、model_comparison.csv、confusion_matrix.csv、error_samples.csv、loss_curve.png、best_model.pt、README.md 串成训练证据链。
用箭头强调从数据、训练、验证、复盘样本、曲线到作品集交付的流转。
风格像训练控制台、文件夹流程图和课程漫画结合，竖版、分步骤、清楚实用。
文字不是主体；标准术语和文件名保留英文。中文只用短提示。不要整段英文说明、乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch06-hands-on-code-execution-sequence.png",
        "size": "1024x1792",
        "quality": "high",
        "title": "PyTorch 工作坊代码执行顺序图",
        "suggested_page": "docs/ch06-deep-learning/ch08-projects/04-hands-on-dl-workshop.md",
        "alt": "PyTorch 工作坊代码执行顺序图：main 函数依次重置目录、生成数据、划分数据、构建 DataLoader、训练 baseline 和 CNN、评估、保存曲线和报告。",
        "prompt": """
一张竖版代码执行顺序图，主题是“main() 如何串起完整 PyTorch 项目”。
步骤从上到下：reset workspace、build synthetic dataset、random_split、DataLoader、shape trace、train Flatten baseline、train Tiny CNN、validate、predict samples、plot loss curve、save checkpoint、write README。
画面像代码调用栈和训练看板结合，突出每个函数输出的文件。
风格清晰、适合新人跟着代码运行和定位卡点。
文字不是主体；函数名、API、文件名保留英文。中文只用短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch06-hands-on-shape-debug-loop.png",
        "size": "1024x1792",
        "quality": "high",
        "title": "PyTorch shape 与训练排错闭环图",
        "suggested_page": "docs/ch06-deep-learning/ch08-projects/04-hands-on-dl-workshop.md",
        "alt": "PyTorch shape 与训练排错闭环图：先查 tensor shape、label format、loss curve、validation gap 和 batch size，再回到证据文件。",
        "prompt": """
一张竖版常见错误排查闭环图，主题是“PyTorch 出错时先查 shape、loss 和曲线”。
故障卡片包括：ModuleNotFoundError、Expected 4D input to conv2d、Target size error、loss not decreasing、overfitting、out of memory。
每个故障卡连到动作：install torch、print batch shape、check CrossEntropyLoss labels、lower learning rate、compare train/val curves、reduce BATCH_SIZE。
风格像排错流程漫画和工程检查表结合，红色警示但不吓人，适合新人。
文字不是主体；错误名、API、文件名保留英文。中文只用短提示。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch06-hands-on-portfolio-pack.png",
        "size": "1024x1792",
        "quality": "high",
        "title": "深度学习作品集证据包图",
        "suggested_page": "docs/ch06-deep-learning/ch08-projects/04-hands-on-dl-workshop.md",
        "alt": "深度学习作品集证据包图：运行命令、shape trace、训练日志、loss 曲线、模型对比、checkpoint、复盘样本和下一步计划组成可复现交付。",
        "prompt": """
一张竖版作品集证据包图，主题是“第 6 章项目交付要让别人能复现、能诊断、能继续改”。
画面像一个打开的深度学习项目文件夹和作品集页面，文件夹卡片包括运行命令、shape trace、training_log.csv、loss_curve.png、模型对比、checkpoint、复盘样本、debug 清单、下一步计划。
强调训练证据不是装饰，而是项目交付的一部分。
风格专业、干净、有作品集质感，适合放在实操课程末尾。
文字不是主体；标准术语和文件名保留英文。中文尽量用短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch06-hands-on-shape-split-guardrail.png",
        "size": "1024x1792",
        "quality": "high",
        "title": "PyTorch shape 与数据划分护栏图",
        "suggested_page": "docs/ch06-deep-learning/ch08-projects/04-hands-on-dl-workshop.md",
        "alt": "PyTorch shape 与数据划分护栏图：16x16 灰度图像进入 batch/channel/height/width，再划分为 train、validation 和 test，避免数据泄漏。",
        "prompt": """
一张竖版课程图解，主题是“训练前先确认 tensor shape 和数据划分”。
画面从上到下分成 5 个步骤：single image 16x16、add channel -> (1,16,16)、batch -> (32,1,16,16)、labels -> (32,)、random_split -> train / validation / test。
旁边用护栏和检查牌强调：不要把 test set 用来调参，validation 用来选模型，test 只做最后确认。
风格像新人跟做课程的漫画式流程图，清晰、竖版、有步骤编号。
文字不是主体；标准术语和 shape 保留英文，例如 tensor、batch、channel、train、validation、test、random_split。中文只用短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch06-hands-on-dataset-dataloader-batch-flow.png",
        "size": "1024x1792",
        "quality": "high",
        "title": "Dataset 到 DataLoader 批处理流程图",
        "suggested_page": "docs/ch06-deep-learning/ch08-projects/04-hands-on-dl-workshop.md",
        "alt": "Dataset 到 DataLoader 批处理流程图：__len__、__getitem__、shuffle、batch size 和 mini-batch 输出连接到训练循环。",
        "prompt": """
一张竖版 PyTorch 教学流程图，主题是“Dataset 负责拿样本，DataLoader 负责组 batch”。
画面从上到下展示：images + labels + sample_ids 进入 StripeDataset；__len__ 告诉总数；__getitem__ 拿出一个样本；DataLoader 做 shuffle 和 batch_size；最后输出 mini-batch images、labels、sample_ids 给 training loop。
突出新人容易混淆的点：Dataset 不训练模型，DataLoader 不改变标签含义，它只负责批处理和迭代。
风格像课堂白板加漫画步骤卡，竖版、简洁、适合跟着代码理解。
文字不是主体；标准术语保留英文，例如 Dataset、DataLoader、__len__、__getitem__、shuffle、batch_size、mini-batch。中文只用短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch06-hands-on-training-loop-anatomy.png",
        "size": "1024x1792",
        "quality": "high",
        "title": "PyTorch 训练循环解剖图",
        "suggested_page": "docs/ch06-deep-learning/ch08-projects/04-hands-on-dl-workshop.md",
        "alt": "PyTorch 训练循环解剖图：model.train、zero_grad、forward、CrossEntropyLoss、backward、clip_grad_norm、optimizer.step 和 validation 按顺序执行。",
        "prompt": """
一张竖版训练循环解剖图，主题是“每个 batch 的 PyTorch 更新顺序”。
画面像放大的训练循环机器，从上到下显示：model.train()、batch images/labels、optimizer.zero_grad、logits = model(images)、loss = CrossEntropyLoss、loss.backward、clip_grad_norm、optimizer.step、log train metrics。
右侧另有 validation 小通道：model.eval()、torch.no_grad()、evaluate val loss/accuracy、save best state。
强调顺序不能乱，旧梯度要清空，验证不反向传播。
文字不是主体；API 和函数名保留英文。中文只用短提示。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch06-hands-on-output-reading-lab.png",
        "size": "1024x1792",
        "quality": "high",
        "title": "深度学习输出文件阅读实验室图",
        "suggested_page": "docs/ch06-deep-learning/ch08-projects/04-hands-on-dl-workshop.md",
        "alt": "深度学习输出文件阅读实验室图：shape_trace、training_log、model_comparison、confusion_matrix、error_samples、metrics_summary 和 loss_curve 逐项阅读。",
        "prompt": """
一张竖版输出阅读实验室图，主题是“跑完代码后不要只看 accuracy，要逐个读证据文件”。
画面像一个训练结果实验台，依次打开 shape_trace.md、training_log.csv、model_comparison.csv、confusion_matrix.csv、error_samples.csv、metrics_summary.json、loss_curve.png。
每个文件旁边有一个短问题：shape 对吗、loss 降了吗、baseline 被超过了吗、哪类混淆了、哪些样本要复盘、最佳模型是谁、曲线是否健康。
风格清晰、流程式、适合新人跟做后检查输出。
文字不是主体；文件名和标准术语保留英文。中文只用短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch06-hands-on-rerun-experiment-loop.png",
        "size": "1024x1792",
        "quality": "high",
        "title": "PyTorch 小步重跑实验闭环图",
        "suggested_page": "docs/ch06-deep-learning/ch08-projects/04-hands-on-dl-workshop.md",
        "alt": "PyTorch 小步重跑实验闭环图：一次只改 batch size、learning rate、epochs 或模型宽度，重跑后比较日志、曲线、checkpoint 和复盘样本。",
        "prompt": """
一张竖版小步实验闭环图，主题是“改一个变量，重跑一次，比较证据”。
画面从上到下形成循环：choose one change、edit BATCH_SIZE or lr or epochs、rerun python dl_workshop.py、compare training_log.csv、compare loss_curve.png、read error_samples.csv、write experiment note、decide next change。
强调不要一次改很多变量，否则无法解释结果变化。
风格像工程实验笔记和课程漫画结合，竖版、清晰、可操作。
文字不是主体；标准术语和文件名保留英文。中文只用短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch06-neuron-linear-activation-gate.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "神经元线性打分与激活门图",
        "suggested_page": "docs/ch06-deep-learning/ch01-nn-basics/01-neurons-activation.md",
        "alt": "神经元线性打分与激活门图：输入特征先经过加权求和 z=x·w+b，再经过 ReLU、Sigmoid 等激活函数形成输出。",
        "prompt": """
一张面向深度学习新手的机制图，主题是“人工神经元 = 线性打分 + 非线性门”。
画面左侧是输入特征 x1、x2、x3，经过不同权重 w 和偏置 b 汇入公式 z = x·w + b；中间是一个发光的 activation gate，展示 ReLU 截断负值、Sigmoid 压到 0~1、Tanh 压到 -1~1；右侧输出 a = f(z)，再进入下一层。
风格像清晰的教学信息图，有线性打分区、激活门区、输出区三段结构，帮助新人一眼看出两步计算。
文字不是主体；中文短标签为主，公式、变量和标准术语保留英文，例如 z = x·w + b、ReLU、Sigmoid、Tanh、activation。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch06-xor-single-layer-limit-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "XOR 单层感知机局限图",
        "suggested_page": "docs/ch06-deep-learning/ch01-nn-basics/01-neurons-activation.md",
        "alt": "XOR 单层感知机局限图：AND 可以被一条直线分开，XOR 四个点无法被单条线性边界分开，多层网络可以组合出非线性边界。",
        "prompt": """
一张面向新手解释 XOR 问题的对比图，主题是“单层感知机只能画一条直线，XOR 需要非线性边界”。
画面左侧展示 AND 四个点和一条清楚的 linear boundary；中间展示 XOR 四个点交叉分布，尝试画一条直线但失败；右侧展示多层网络先做特征变换，再组合出弯曲或折线形决策边界。
整体像数学启蒙海报，用坐标点、分割线和小型网络结构说明表达能力边界。
文字不是主体；中文短标签为主，标准术语保留英文，例如 XOR、Perceptron、linear boundary、MLP。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch06-backprop-error-responsibility-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "反向传播误差责任分摊图",
        "suggested_page": "docs/ch06-deep-learning/ch01-nn-basics/02-forward-backward.md",
        "alt": "反向传播误差责任分摊图：loss 从输出层向隐藏层回传，把误差责任分配给 W2、b2、W1、b1 等参数梯度。",
        "prompt": """
一张解释反向传播的教学机制图，主题是“从 loss 往回分摊误差责任”。
画面上半部是前向传播：input -> hidden layer -> output -> loss；下半部用反方向箭头从 loss 回传到 output、hidden、parameters，标出 dL/dW2、dL/db2、dL/dW1、dL/db1。
视觉上像一条生产线出现误差后，责任沿计算图逐层追溯，强调 chain rule 和 gradient。
文字不是主体；中文短标签为主，公式和标准术语保留英文，例如 loss、gradient、chain rule、backward、dL/dW。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch06-numpy-to-pytorch-training-loop-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "NumPy 到 PyTorch 训练循环对照图",
        "suggested_page": "docs/ch06-deep-learning/ch01-nn-basics/02-forward-backward.md",
        "alt": "NumPy 到 PyTorch 训练循环对照图：手写 forward、loss、gradient、update 对应 PyTorch 的 model(x)、loss.backward、optimizer.step 和 zero_grad。",
        "prompt": """
一张面向新手的 NumPy 到 PyTorch 对照图，主题是“手写训练循环怎样变成 PyTorch API”。
画面左侧是 NumPy 手工流程：forward、compute loss、manual gradient、manual update；右侧是 PyTorch 流程：model(x)、loss_fn、loss.backward()、optimizer.step()、optimizer.zero_grad()；中间用桥梁连接每一对步骤。
风格像迁移路线图，帮助学习者知道 PyTorch 不是黑箱，而是在自动化手工训练步骤。
文字不是主体；中文短标签为主，API 和标准术语保留英文，例如 NumPy、PyTorch、model(x)、loss.backward()、optimizer.step()、zero_grad()。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch06-optimizer-gradient-to-update-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "梯度到参数更新的优化器决策图",
        "suggested_page": "docs/ch06-deep-learning/ch01-nn-basics/03-optimizers.md",
        "alt": "梯度到参数更新的优化器决策图：梯度给出坡度方向，SGD、Momentum、Adam 以不同方式把梯度转换成参数更新。",
        "prompt": """
一张解释优化器的教学图，主题是“gradient 只是坡度，optimizer 决定怎么走”。
画面左侧是 loss landscape 山谷和当前参数点，箭头表示 gradient；中间分成三条路线：SGD 直接走当前坡度、Momentum 带惯性减少左右摇摆、Adam 根据历史一阶和二阶信息自动调步长；右侧是参数更新后的更低 loss。
风格像下山路线比较图，直观表现不同优化器的行为差异。
文字不是主体；中文短标签为主，标准术语和公式保留英文，例如 gradient、SGD、Momentum、Adam、learning rate、update。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch06-regularization-overfit-action-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "过拟合问题到正则化动作选择图",
        "suggested_page": "docs/ch06-deep-learning/ch01-nn-basics/04-regularization.md",
        "alt": "过拟合问题到正则化动作选择图：先看训练验证曲线和数据划分，再选择数据增强、weight decay、early stopping、Dropout 等手段。",
        "prompt": """
一张深度学习过拟合排查与正则化选择图，主题是“不要一过拟合就只想到 Dropout”。
画面左侧是 train loss 下降但 val loss 上升的曲线；中间是排查节点：数据划分、样本量、模型容量、训练轮数；右侧是动作工具箱：data augmentation、weight decay、early stopping、Dropout、BatchNorm/LayerNorm。
风格像新手诊断流程卡片，清晰、轻松，突出先诊断再用工具。
文字不是主体；中文短标签为主，标准术语保留英文，例如 overfitting、val loss、weight decay、Dropout、early stopping、data augmentation。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch06-tensor-shape-meaning-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "PyTorch 张量 shape 语义速查图",
        "suggested_page": "docs/ch06-deep-learning/ch02-pytorch/01-pytorch-basics.md",
        "alt": "PyTorch 张量 shape 语义速查图：表格数据、图像数据和文本序列分别对应 batch/features、batch/channels/height/width、batch/seq_len/embedding_dim。",
        "prompt": """
一张 PyTorch shape 语义速查图，主题是“shape 不只是数字，每一维都有含义”。
画面分三栏：表格数据 [batch, features]，图像数据 [batch, channels, height, width]，文本序列 [batch, seq_len, embedding_dim]；每栏用小样本盒子和维度箭头标出每一维代表什么。
风格像清爽的工程备忘卡，适合新人写模型前对照。
文字不是主体；中文短标签为主，shape、API 和标准术语保留英文，例如 Tensor、shape、batch、features、channels、height、width、seq_len、embedding_dim。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch06-autograd-gradient-lifecycle-map.png",
        "size": "1024x1792",
        "quality": "medium",
        "title": "PyTorch 自动求导梯度生命周期图",
        "suggested_page": "docs/ch06-deep-learning/ch02-pytorch/02-autograd.md",
        "alt": "PyTorch 自动求导训练闭环图：先 zero_grad 清空旧梯度，再 forward 建图，backward 写入 grad，optimizer.step 更新参数，下一批重新建图。",
        "prompt": """
一张竖版 PyTorch autograd 训练闭环图，主题是“一轮训练里，计算图是临时的，.grad 缓冲区会保留到你清空”。
画面用 5 个大步骤卡片展示：1 optimizer.zero_grad() 清空旧 .grad；2 forward 创建本轮计算图并得到 loss；3 loss.backward() 按链式法则把梯度写入 .grad；4 optimizer.step() 读取 .grad 更新参数；5 下一批 batch 重新创建新图。
左右两侧用很轻的提示条区分两种生命周期：计算图 forward 创建、backward 使用、随后释放；.grad 默认累积，忘记 zero_grad 会叠加旧梯度。
风格要像清爽工程白板和流程图结合，竖向、留白充足、每一步只放一两句短标签；避免密集小字、复杂公式和拥挤箭头。
中文短标签为主，API 和标准术语保留英文，例如 autograd、loss.backward()、.grad、optimizer.step()、zero_grad()。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch06-training-loop-order-guardrail.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "PyTorch 训练循环顺序护栏图",
        "suggested_page": "docs/ch06-deep-learning/ch02-pytorch/05-training-loop.md",
        "alt": "PyTorch 训练循环顺序护栏图：训练阶段按 model.train、batch、forward、loss、zero_grad、backward、step 执行，验证阶段用 eval 和 no_grad。",
        "prompt": """
一张 PyTorch 训练循环顺序护栏图，主题是“训练循环不能随便打乱顺序”。
画面主线是一条有护栏的跑道：model.train()、取 batch、forward、loss、optimizer.zero_grad()、loss.backward()、optimizer.step()；旁边分出验证支线：model.eval()、torch.no_grad()、validation metrics。
风格像工程流程图和检查清单结合，帮助新人一眼记住训练态与验证态。
文字不是主体；中文短标签为主，API 和标准术语保留英文，例如 model.train()、forward、loss、zero_grad()、backward()、step()、model.eval()、torch.no_grad()。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch06-conv-stride-padding-size-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "卷积 stride padding 与输出尺寸变化图",
        "suggested_page": "docs/ch06-deep-learning/ch03-cnn/01-convolution-basics.md",
        "alt": "卷积 stride padding 与输出尺寸变化图：stride 控制滑动步长，padding 给边缘补框，二者共同影响输出特征图尺寸。",
        "prompt": """
一张解释卷积 stride、padding 和 output size 的教学图，主题是“滑多远、补几圈，决定输出有多大”。
画面左侧是输入图像网格和 kernel 窗口；中间分别展示 stride=1 和 stride=2 的滑动步子差异，以及 padding=0 和 padding=1 的边缘补框；右侧用 feature map 尺寸变化展示 output size。
风格像可视化卷积动画的关键帧，清晰标出滑动、补边和输出尺寸。
文字不是主体；中文短标签为主，公式和标准术语保留英文，例如 kernel、stride、padding、feature map、output size。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch06-cnn-receptive-field-growth-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "CNN 感受野逐层变大的特征组合图",
        "suggested_page": "docs/ch06-deep-learning/ch03-cnn/01-convolution-basics.md",
        "alt": "CNN 感受野逐层变大的特征组合图：浅层看边缘纹理，中层组合局部形状，深层看到物体部件和整体语义。",
        "prompt": """
一张解释 CNN 感受野和层级特征的教学图，主题是“小局部特征逐层组合成大语义”。
画面从左到右展示原图局部 patch，第一层 receptive field 看到边缘和纹理，第二层组合出角点和局部形状，第三层看到更大物体部件，最后形成整体语义判断。
风格像图像理解逐层放大镜，突出 receptive field 逐层变大。
文字不是主体；中文短标签为主，标准术语保留英文，例如 receptive field、edge、texture、feature map、semantic feature。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch06-cnn-channel-spatial-tradeoff-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "CNN 通道数与空间尺寸权衡图",
        "suggested_page": "docs/ch06-deep-learning/ch03-cnn/02-cnn-structure.md",
        "alt": "CNN 通道数与空间尺寸权衡图：网络越往后，高宽通常变小，通道数变多，空间细节减少但语义浓度上升。",
        "prompt": """
一张解释 CNN shape 变化趋势的教学图，主题是“高宽变小，通道变多，语义更浓”。
画面展示一组 feature maps 从浅层到深层：H×W 逐渐缩小，channels 从少到多，旁边用图标表现浅层保留像素细节、深层记录更多抽象特征种类。
风格像三维方块堆叠的信息图，清晰标出 height、width、channels 的变化。
文字不是主体；中文短标签为主，shape 和标准术语保留英文，例如 H、W、channels、feature map、spatial resolution、semantic feature。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch06-transfer-learning-freeze-finetune-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "迁移学习冻结 backbone 与逐步微调决策图",
        "suggested_page": "docs/ch06-deep-learning/ch03-cnn/04-transfer-learning.md",
        "alt": "迁移学习冻结 backbone 与逐步微调决策图：根据数据量和任务相似度决定只训练 head、冻结 backbone 或逐步解冻 fine-tune。",
        "prompt": """
一张迁移学习决策图，主题是“数据多少、任务像不像，决定 freeze 还是 fine-tune”。
画面上方是两个判断旋钮：数据量少/多、任务相似/差异大；下方对应三种策略：冻结 backbone 只训练 head、解冻最后几层、小学习率全模型 fine-tune。配一个预训练 CNN backbone 和新任务 classifier head。
风格像选择路线图，帮助新人理解迁移学习不是固定套路。
文字不是主体；中文短标签为主，标准术语保留英文，例如 transfer learning、backbone、head、freeze、fine-tune、learning rate。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch06-rnn-hidden-state-rolling-memory-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "RNN 隐藏状态滚动记忆图",
        "suggested_page": "docs/ch06-deep-learning/ch04-rnn/01-rnn-basics.md",
        "alt": "RNN 隐藏状态滚动记忆图：每个时间步读取当前输入 x_t 和上一隐藏状态 h_{t-1}，生成新的隐藏状态 h_t。",
        "prompt": """
一张解释 RNN hidden state 的教学图，主题是“每读一步，就更新一份滚动记忆”。
画面从左到右是 time step t1、t2、t3、t4，每一步都有输入 x_t 和上一份记忆 h_{t-1} 汇入同一个 RNN cell，输出新记忆 h_t；用滚动笔记本或记忆胶囊表现 hidden state。
风格像序列阅读流程图，强调参数共享和逐步更新。
文字不是主体；中文短标签为主，公式和标准术语保留英文，例如 x_t、h_{t-1}、h_t、RNN cell、hidden state、shared weights。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch06-rnn-long-dependency-vanishing-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "RNN 长依赖与梯度消失直觉图",
        "suggested_page": "docs/ch06-deep-learning/ch04-rnn/01-rnn-basics.md",
        "alt": "RNN 长依赖与梯度消失直觉图：早期信息沿 hidden state 传递越来越淡，反向传播梯度回到早期时间步也越来越弱。",
        "prompt": """
一张解释 RNN 长依赖困难和梯度消失的直觉图，主题是“信息和梯度在长序列里越传越淡”。
画面展示一长串 time steps，早期 token 的信息用亮色信号向后传递但逐渐变淡；反向梯度从后往前传回时也逐渐变弱。右侧标出 LSTM/GRU 用门控补救，Transformer 用 attention 直接建立远距离连接。
风格像时间隧道和信号衰减图结合，适合初学者理解痛点。
文字不是主体；中文短标签为主，标准术语保留英文，例如 long dependency、vanishing gradient、LSTM、GRU、Attention、Transformer。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch06-lstm-gates-information-control-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "LSTM 门控信息流控制图",
        "suggested_page": "docs/ch06-deep-learning/ch04-rnn/02-lstm-gru.md",
        "alt": "LSTM 门控信息流控制图：Forget Gate 控制旧记忆保留，Input Gate 控制新信息写入，Output Gate 控制当前输出。",
        "prompt": """
一张解释 LSTM gates 的教学图，主题是“LSTM 不是更复杂，而是学会管理记忆”。
画面中心是一条 cell state 记忆高速路 c_t，三道可视化闸门依次控制信息：Forget Gate 决定旧记忆留多少，Input Gate 决定新信息写多少，Output Gate 决定对外暴露多少；旁边展示 hidden state h_t。
风格像水闸或信息管道控制台，清晰、直观。
文字不是主体；中文短标签为主，标准术语和变量保留英文，例如 Forget Gate、Input Gate、Output Gate、cell state c_t、hidden state h_t。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch06-attention-qkv-library-analogy-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "注意力 QKV 图书馆检索类比图",
        "suggested_page": "docs/ch06-deep-learning/ch05-transformer/01-attention-mechanism.md",
        "alt": "注意力 QKV 图书馆检索类比图：Q 像当前问题，K 像资料索引标签，V 像真正要取出的内容，注意力按匹配分数混合 V。",
        "prompt": """
一张用图书馆检索类比解释 attention Q/K/V 的教学图，主题是“Q 是问题，K 是索引，V 是内容”。
画面中一个 token 拿着 Query 去查资料架，每本资料有 Key 标签和 Value 内容；Query 与所有 Key 打分得到 attention scores，再按权重混合 Value 形成 context vector。
风格像现代图书馆和矩阵流结合，既形象又保留模型机制。
文字不是主体；中文短标签为主，公式和标准术语保留英文，例如 Query Q、Key K、Value V、attention score、softmax、context。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch06-causal-mask-no-peeking-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Causal Mask 防止偷看未来图",
        "suggested_page": "docs/ch06-deep-learning/ch05-transformer/01-attention-mechanism.md",
        "alt": "Causal Mask 防止偷看未来图：生成任务中每个位置只能看自己和过去 token，不能看到未来答案。",
        "prompt": """
一张解释 causal mask 的教学图，主题是“生成模型训练时不能偷看未来”。
画面左侧是一串 token 逐个生成，当前位置只能看左侧历史；右侧是 attention matrix，上三角区域被红色 mask 遮住，下三角区域允许关注。底部用考试类比：当前题不能提前看后面答案。
风格像矩阵热力图和序列生成流程结合，清楚表现 no peeking。
文字不是主体；中文短标签为主，标准术语保留英文，例如 causal mask、attention matrix、token、no peeking、decoder。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch06-transformer-block-role-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Transformer Block 组件职责图",
        "suggested_page": "docs/ch06-deep-learning/ch05-transformer/02-transformer-architecture.md",
        "alt": "Transformer Block 组件职责图：Attention 混合上下文，Residual 保留原信息，LayerNorm 稳定数值，FFN 对每个位置再加工。",
        "prompt": """
一张解释 Transformer Block 组件职责的教学图，主题是“Transformer 不只有 attention，而是一套可堆深的结构”。
画面中心是一个 Encoder Block，被拆成四个职责模块：Self-Attention 负责上下文交流，Residual 负责保留原信息和梯度通道，LayerNorm 负责稳定数值，FFN 负责逐位置再加工；箭头展示数据流。
风格像透明机器结构图，每个模块像不同功能的齿轮或工位。
文字不是主体；中文短标签为主，标准术语保留英文，例如 Self-Attention、Residual、LayerNorm、FFN、Encoder Block。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch06-transformer-representation-refinement-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Transformer 层间表示逐步精炼图",
        "suggested_page": "docs/ch06-deep-learning/ch05-transformer/02-transformer-architecture.md",
        "alt": "Transformer 层间表示逐步精炼图：每层 shape 可能不变，但 token 表示不断融入更多上下文，语义信息越来越丰富。",
        "prompt": """
一张解释 Transformer 多层表示精炼的教学图，主题是“shape 不变，但表示内容在变强”。
画面展示同一串 token 经过 Block 1、Block 2、Block 3，外形尺寸始终标为 [batch, seq_len, d_model]，但每个 token 内部颜色和连接越来越丰富，表示它融入更多上下文和语义关系。
风格像逐层精炼的宝石或信号处理流水线，强调 representation 而不是尺寸变化。
文字不是主体；中文短标签为主，shape 和标准术语保留英文，例如 [batch, seq_len, d_model]、token representation、context、semantic refinement。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch06-gan-adversarial-balance-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "GAN 对抗训练平衡与模式崩塌图",
        "suggested_page": "docs/ch06-deep-learning/ch06-generative/01-gan.md",
        "alt": "GAN 对抗训练平衡与模式崩塌图：Generator 生成假样本，Discriminator 区分真假，二者需要保持平衡，否则可能反馈不足或 mode collapse。",
        "prompt": """
一张解释 GAN 对抗训练的教学图，主题是“生成器和判别器像两个同时进步的对手”。
画面左侧 Generator 从 noise 生成 fake samples，右侧 Discriminator 同时看 real samples 和 fake samples 并给出反馈；中间有平衡秤表示二者不能一边倒。角落展示 mode collapse：生成器只重复少数样本。
风格像训练擂台和实验台结合，直观但不夸张。
文字不是主体；中文短标签为主，标准术语保留英文，例如 GAN、Generator、Discriminator、real/fake、feedback、mode collapse。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch06-vae-latent-continuity-sampling-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "VAE 连续潜空间与采样区域图",
        "suggested_page": "docs/ch06-deep-learning/ch06-generative/02-vae.md",
        "alt": "VAE 连续潜空间与采样区域图：编码器输出 mu 和 sigma，把样本映射到可采样区域，潜空间连续时插值和生成更自然。",
        "prompt": """
一张解释 VAE latent space 的教学图，主题是“VAE 学的是可采样区域，不只是一个固定点”。
画面左侧输入样本经过 Encoder 输出 mu 和 sigma，中间是二维 latent space，每个样本不是单点而是一个柔和的概率云区域；从区域采样 z 后进入 Decoder，生成连续变化的输出。展示两个样本之间平滑插值。
风格像星图和概率云结合，清楚表现连续潜空间。
文字不是主体；中文短标签为主，标准术语和变量保留英文，例如 VAE、Encoder、Decoder、mu、sigma、z、latent space、sampling。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch06-training-diagnosis-dashboard-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "训练诊断仪表盘排查路线图",
        "suggested_page": "docs/ch06-deep-learning/ch07-training-tips/02-training-diagnosis.md",
        "alt": "训练诊断仪表盘排查路线图：先看 train/val 曲线，再看学习率和 batch，继续检查预测分布、梯度、数据和标签。",
        "prompt": """
一张深度学习训练诊断仪表盘图，主题是“loss 不对时，先排查流程和数据，不要马上换大模型”。
画面像一个 training dashboard，从上到下依次是 train/val loss 曲线、learning rate 与 batch size 控件、prediction distribution、gradient health、data/label check；右侧是排查路线箭头，最后才到 model architecture。
风格像清晰的工程监控台和排障路线图，帮助新人训练时不慌。
文字不是主体；中文短标签为主，标准术语保留英文，例如 train loss、val loss、learning rate、batch size、gradient、data label、model architecture。不要整张图全英文，不要乱码小字或真实品牌 logo。
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
        "filename": "ch07-tokenizer-granularity-tradeoff-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Tokenizer 粒度取舍图",
        "suggested_page": "docs/ch07-llm-principles/ch01-nlp-crash/01-tokenizer.md",
        "alt": "Tokenizer 粒度取舍图：字符级、词级和子词级在序列长度、OOV 风险、词表大小和语义粒度之间做取舍。",
        "prompt": """
一张适合大模型入门课程的科学教育信息图，主题是“Tokenizer 是粒度、成本和覆盖率的取舍”。
画面分成三栏：char-level 字符级、word-level 词级、subword/BPE 子词级；用四个维度对比 sequence length、OOV risk、vocab size、semantic granularity。
风格像高质量课程白板和清爽数据卡片结合，帮助新人理解为什么现代大模型多用子词切分。
文字不是主体；标准术语保留英文，例如 char-level、word-level、subword、BPE、WordPiece、OOV、vocab size。其他说明可用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch07-tokenizer-inputids-mask-length-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Tokenizer 到 input_ids 与 attention_mask 图",
        "suggested_page": "docs/ch07-llm-principles/ch01-nlp-crash/01-tokenizer.md",
        "alt": "Tokenizer 到张量图：原文经过 tokenization、special tokens、input_ids、padding、attention_mask 和 truncation 变成可批处理张量。",
        "prompt": """
一张适合解释 tokenizer 输出对象的流程图，主题是“文字怎样变成模型能批处理的张量”。
画面表现 raw text 进入 tokenization，加入 special tokens，例如 [CLS]、[SEP]，再转换为 input_ids；短句经过 padding，长句经过 truncation，同时生成 attention_mask。
用两条样本句子展示同一个 batch 中长短不一的输入如何被对齐，最终进入 Transformer。
文字不是主体；标准术语保留英文，例如 raw text、tokens、input_ids、padding、truncation、attention_mask、batch。其他说明可用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch07-embedding-onehot-dense-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "One-hot 到 dense embedding 语义空间图",
        "suggested_page": "docs/ch07-llm-principles/ch01-nlp-crash/02-embeddings.md",
        "alt": "One-hot 到 dense embedding 对比图：One-hot 只能区分身份，dense embedding 把语义相近词放到相近向量空间位置。",
        "prompt": """
一张适合解释 Embedding 的对比信息图，主题是“One-hot 只记身份，dense embedding 记录语义距离”。
画面左侧展示稀疏 one-hot 编码，每个词只有一个位置为 1；右侧展示二维或三维语义空间，refund 和 return 靠近，password 和 reset 靠近，banana 离技术词较远。
加入一条 cosine similarity 的距离提示，但不要堆公式，重点是空间直觉。
文字不是主体；标准术语保留英文，例如 one-hot、dense embedding、semantic space、cosine similarity。其他说明可用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch07-contextual-embedding-sense-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "上下文化表示消解一词多义图",
        "suggested_page": "docs/ch07-llm-principles/ch01-nlp-crash/02-embeddings.md",
        "alt": "上下文化表示图：bank 在 bank account 与 river bank 两种上下文中形成不同上下文化向量。",
        "prompt": """
一张适合解释 contextual embedding 的课程图，主题是“同一个词会被上下文重新定位”。
画面中央是同一个 token：bank；上方句子是 bank account，注意力线把它拉向 finance 语义簇；下方句子是 river bank，注意力线把它拉向 river/land 语义簇。
用小对比展示 static embedding 位置固定，而 contextual embedding 会随上下文移动。
文字不是主体；标准术语保留英文，例如 token、bank、static embedding、contextual embedding、attention、finance、river。其他说明可用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch07-next-token-generation-loop-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Next-token 生成循环与采样图",
        "suggested_page": "docs/ch07-llm-principles/ch02-llm-overview/02-core-concepts.md",
        "alt": "Next-token 生成循环图：上下文经过 embedding、Transformer、logits、softmax、temperature/top-p 采样得到下一个 token，再拼回上下文继续生成。",
        "prompt": """
一张适合大模型原理入门的循环图，主题是“生成文本就是不断预测下一个 token”。
画面表现 context 进入 embedding 和 Transformer，输出 logits，经 softmax 形成概率条，再通过 temperature 和 top-p sampling 选出 next token，最后把新 token append 回 context，形成下一轮循环。
风格像清晰的机制动画分镜，概率柱状条和 token 卡片要直观。
文字不是主体；标准术语保留英文，例如 context、embedding、Transformer、logits、softmax、temperature、top-p、next token、append。其他说明可用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch07-context-window-budget-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Context window 信息预算图",
        "suggested_page": "docs/ch07-llm-principles/ch02-llm-overview/02-core-concepts.md",
        "alt": "Context window 信息预算图：系统提示、历史对话、检索资料、用户问题和输出空间共同占用 token 预算。",
        "prompt": """
一张适合解释 context window 的信息预算图，主题是“上下文窗口像一张有限大小的工作台”。
画面用一个固定容量的工作台或行李箱表示 context window，内部被 system prompt、chat history、retrieved docs、user question、output budget 分段占用；边缘展示 overflow/truncation 风险。
让新人一眼理解长上下文不是无限记忆，而是有限 token 预算。
文字不是主体；标准术语保留英文，例如 context window、system prompt、chat history、retrieved docs、user question、output budget、truncation。其他说明可用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch07-huggingface-workflow-object-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "HuggingFace 标准工作流对象关系图",
        "suggested_page": "docs/ch07-llm-principles/ch01-nlp-crash/04-huggingface-quickstart.md",
        "alt": "HuggingFace 工作流图：文本经 tokenizer 变成 input_ids 与 attention_mask，config 定义结构，model.forward 输出 hidden states/logits。",
        "prompt": """
一张适合 HuggingFace 入门的对象关系图，主题是“几个对象各负责什么”。
画面像一个小实验台：raw text 进入 tokenizer，得到 input_ids 和 attention_mask；config 像模型蓝图；model.forward 像计算机器，输出 hidden states 或 logits。
不要使用真实 HuggingFace logo，用抽象工具箱和模型模块表现。
文字不是主体；标准术语保留英文，例如 tokenizer、config、model.forward、input_ids、attention_mask、hidden states、logits。其他说明可用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch07-transformer-block-dataflow-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Transformer Block 数据流拆解图",
        "suggested_page": "docs/ch07-llm-principles/ch03-transformer-deep/01-architecture-review.md",
        "alt": "Transformer Block 数据流图：token 表示经过 Self-Attention、Residual、LayerNorm、FFN 和再次 Residual/LayerNorm 完成一层 block 更新。",
        "prompt": """
一张适合解释 Transformer block 的数据流图，主题是“一层 block 如何更新 token 表示”。
画面从 token representations 开始，依次经过 Self-Attention、Residual、LayerNorm、FFN、Residual、LayerNorm；每个模块旁用短标签说明作用：Attention 交流上下文，Residual 保留原信息，LayerNorm 稳定数值，FFN 单 token 深加工。
风格像严谨但友好的课堂机制图，箭头清晰、层次分明。
文字不是主体；标准术语保留英文，例如 Self-Attention、Residual、LayerNorm、FFN、token representations。其他说明可用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch07-architecture-mask-task-fit-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "架构 mask 与任务适配图",
        "suggested_page": "docs/ch07-llm-principles/ch03-transformer-deep/02-model-variants.md",
        "alt": "架构 mask 与任务适配图：Encoder-only 双向理解，Decoder-only 因果生成，Encoder-Decoder 先读输入再生成输出，任务取决于信息流约束。",
        "prompt": """
一张适合解释 Transformer 架构变体的三栏图，主题是“信息流约束决定任务适配”。
三栏分别是 Encoder-only、Decoder-only、Encoder-Decoder；每栏展示 attention mask 矩阵：双向全可见、causal mask 只能看过去、encoder-decoder cross-attention 先读输入再生成输出。
下方连接典型任务：classification、generation、translation/summarization。可以出现 BERT、GPT、T5 作为标准术语，但不要画真实 logo。
文字不是主体；标准术语保留英文，例如 Encoder-only、Decoder-only、Encoder-Decoder、attention mask、causal mask、cross-attention、BERT、GPT、T5。其他说明可用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch07-efficient-attention-bottleneck-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "高效注意力瓶颈分流图",
        "suggested_page": "docs/ch07-llm-principles/ch03-transformer-deep/03-efficient-attention.md",
        "alt": "高效注意力瓶颈分流图：长上下文平方复杂度、KV cache 体积和显存读写效率分别对应 sliding/local attention、MQA/GQA、FlashAttention 等路线。",
        "prompt": """
一张适合讲高效注意力路线的决策分流图，主题是“先找瓶颈，再选优化方法”。
画面中央是 Attention bottleneck，分成三条问题线：long context 的 O(n^2) 计算、KV cache memory 变大、memory IO 读写慢；分别连接 sliding/local attention、MQA/GQA、FlashAttention 等方法。
风格像工程诊断仪表盘和路线图结合，强调不是所有方法解决同一个问题。
文字不是主体；标准术语保留英文，例如 Attention bottleneck、O(n^2)、KV cache、memory IO、FlashAttention、sliding attention、MQA、GQA。其他说明可用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch07-kv-cache-mqa-gqa-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "KV cache 与 MHA/GQA/MQA 对比图",
        "suggested_page": "docs/ch07-llm-principles/ch03-transformer-deep/03-efficient-attention.md",
        "alt": "KV cache 对比图：MHA 每个 head 有独立 K/V，GQA 分组共享 K/V，MQA 多个 query heads 共享一组 K/V，从而减少 cache。",
        "prompt": """
一张适合解释 MHA、GQA、MQA 的结构对比图，主题是“Query head 很多，K/V 可以更少”。
画面三栏对比：MHA 每个 query head 对应独立 K/V；GQA 多个 query heads 分组共享 K/V；MQA 所有或大部分 query heads 共享一组 K/V。旁边用 memory bars 展示 KV cache 逐步变小。
风格像模型结构剖面图，颜色区分 Q、K、V，清晰但不要过度复杂。
文字不是主体；标准术语保留英文，例如 MHA、GQA、MQA、query heads、KV heads、KV cache、memory。其他说明可用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch07-scale-cost-knobs-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "大模型规模成本旋钮图",
        "suggested_page": "docs/ch07-llm-principles/ch03-transformer-deep/04-scale-computation.md",
        "alt": "大模型规模成本旋钮图：layers、hidden size、context length、batch size、kv heads 等旋钮共同放大参数量、计算量和 KV cache。",
        "prompt": """
一张适合解释大模型规模成本的控制面板图，主题是“每个结构旋钮都会改变训练和推理成本”。
画面有多个旋钮：layers、hidden size、context length、batch size、kv heads；这些旋钮连接到 params、FLOPs、activation memory、KV cache、latency 等指标。
突出 hidden size 对参数和计算有平方级影响，用视觉上更粗的连线表现。
文字不是主体；标准术语保留英文，例如 layers、hidden size、context length、batch size、kv heads、params、FLOPs、activation memory、KV cache、latency。其他说明可用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch07-train-inference-cost-split-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "训练期与推理期成本结构对比图",
        "suggested_page": "docs/ch07-llm-principles/ch03-transformer-deep/04-scale-computation.md",
        "alt": "训练期与推理期成本对比图：训练期关注参数、梯度、优化器状态和激活，推理期关注 KV cache、延迟、吞吐和并发显存。",
        "prompt": """
一张适合解释训练成本和推理成本差异的双栏图，主题是“训练和上线不是同一种成本结构”。
左侧 training factory 展示 parameters、gradients、optimizer states、activations、checkpoint；右侧 inference serving dashboard 展示 KV cache、latency、throughput、concurrency、batching。
风格像工程对比看板，帮助新人理解为什么会有训练显存和推理显存两个问题。
文字不是主体；标准术语保留英文，例如 training、inference、parameters、gradients、optimizer states、activations、KV cache、latency、throughput、concurrency。其他说明可用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch07-pretraining-data-governance-funnel.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "预训练数据治理漏斗图",
        "suggested_page": "docs/ch07-llm-principles/ch04-pretraining/01-pretraining-data.md",
        "alt": "预训练数据治理漏斗图：原始网页书籍代码论坛数据经过清洗、去重、风险过滤、污染控制和配比后形成训练语料版本。",
        "prompt": """
一张适合解释预训练数据治理的漏斗图，主题是“不是所有互联网文本都适合直接训练”。
画面上方是 raw web、books、code、forum、docs 等来源，进入大漏斗：cleaning、dedup、quality filter、privacy/copyright risk、contamination control、mixture ratio，最后输出 corpus version 和 model base。
风格像数据工厂和治理流程结合，强调质量、风险和版本。
文字不是主体；标准术语保留英文，例如 raw web、code、cleaning、dedup、quality filter、privacy、copyright、contamination、mixture ratio、corpus version。其他说明可用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch07-pretraining-objective-comparison-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "预训练目标样本改造对比图",
        "suggested_page": "docs/ch07-llm-principles/ch04-pretraining/02-pretraining-methods.md",
        "alt": "预训练目标对比图：同一句文本分别构造成 Causal LM、Masked LM 和 Span Corruption 训练样本，对应续写、补空和恢复片段。",
        "prompt": """
一张适合解释预训练目标的三栏对比图，主题是“同一段文本可以改造成不同练习题”。
用同一句简短文本作为素材，三栏分别展示 Causal LM 预测下一个 token、Masked LM 填 [MASK]、Span Corruption 恢复被替换片段；每栏下方说明会训练出的能力倾向。
风格像练习册和模型训练板结合，直观、轻量。
文字不是主体；标准术语保留英文，例如 Causal LM、Masked LM、[MASK]、Span Corruption、next token、GPT、BERT、T5。其他说明可用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch07-pretraining-engineering-production-line.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "预训练工程生产线图",
        "suggested_page": "docs/ch07-llm-principles/ch04-pretraining/03-pretraining-engineering.md",
        "alt": "预训练工程生产线图：数据分片、流式读取、训练步骤、checkpoint、恢复训练、吞吐监控组成稳定预训练生产线。",
        "prompt": """
一张适合解释预训练工程的生产线图，主题是“大模型预训练像一条不能随便停的生产线”。
画面表现 data shards 仓库、streaming dataloader 传送带、GPU workers 训练区、checkpoint vault、resume switch、throughput monitor 和 loss curve 监控屏。
突出稳定性、恢复训练、吞吐和监控，不要画真实品牌硬件 logo。
文字不是主体；标准术语保留英文，例如 data shards、streaming dataloader、GPU workers、checkpoint、resume、throughput、loss curve。其他说明可用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch07-prompt-spec-three-layer-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Prompt 三层任务规格图",
        "suggested_page": "docs/ch07-llm-principles/ch05-prompt/01-prompt-basics.md",
        "alt": "Prompt 三层任务规格图：Prompt 基础由任务目标、输出格式和约束条件三层组成，先写清规格再谈技巧。",
        "prompt": """
一张适合 Prompt 工程入门的三层结构图，主题是“Prompt 先是任务规格，不是咒语”。
画面展示一张规格卡片分成三层：task goal、output format、constraints；三层共同进入 model call，输出更稳定的 response。
旁边用坏例子变好例子的视觉对比表现：模糊请求变成可执行规格。
文字不是主体；标准术语保留英文，例如 Prompt、task goal、output format、constraints、model call、response。其他说明可用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch07-advanced-prompt-technique-decision-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "高级 Prompt 技巧选择图",
        "suggested_page": "docs/ch07-llm-principles/ch05-prompt/02-advanced-prompting.md",
        "alt": "高级 Prompt 技巧选择图：根据标签边界模糊、风格不一致、任务多步骤、漏条件或格式错误选择 few-shot、角色设定、分步约束、自检等技巧。",
        "prompt": """
一张适合解释高级 Prompt 技巧的决策树，主题是“先看失败症状，再选技巧”。
画面从 failure symptom 开始分支：标签边界模糊选择 few-shot examples；风格不一致选择 role/style guide；多步骤任务选择 step-by-step plan；容易漏条件选择 checklist/self-check；格式错误选择 schema/validation。
风格像排障树和工具箱结合，实用、清晰。
文字不是主体；标准术语保留英文，例如 failure symptom、few-shot、role、style guide、step-by-step、checklist、self-check、schema、validation。其他说明可用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch07-structured-output-contract-validation-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "结构化输出合同与校验闭环图",
        "suggested_page": "docs/ch07-llm-principles/ch05-prompt/03-structured-output.md",
        "alt": "结构化输出合同与校验图：Prompt 定义 JSON schema 合同，模型输出后由程序解析、字段校验、类型校验和值域校验，失败时重试或转人工。",
        "prompt": """
一张适合解释结构化输出可靠性的闭环图，主题是“模型输出要经过程序合同校验”。
画面表现 Prompt 定义 JSON schema contract，model output 进入 JSON parser，再经过 field check、type check、range check；通过则进入 workflow，失败则 retry 或 human review。
风格像 API 合同、质检流水线和调试面板结合，突出可解析、可校验、可回退。
文字不是主体；标准术语保留英文，例如 JSON schema、model output、JSON parser、field check、type check、range check、retry、human review、workflow。其他说明可用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch07-finetune-decision-rag-prompt-peft-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "微调前方案选择决策图",
        "suggested_page": "docs/ch07-llm-principles/ch06-finetuning/01-finetuning-overview.md",
        "alt": "微调前方案选择图：知识问题优先 RAG，格式问题优先 Prompt/结构化输出，工具流程问题优先 Agent，稳定行为问题再考虑微调或 PEFT。",
        "prompt": """
一张适合解释“什么时候需要微调”的决策图，主题是“不要把所有问题都丢给 fine-tuning”。
画面从 problem diagnosis 开始：知识缺失走 RAG；格式不稳走 Prompt/structured output；工具流程走 Agent/tool use；少量固定风格走 Prompt examples；大量稳定行为和领域表达再走 fine-tuning/PEFT。
风格像工程路线选择器，强调先用低成本方案验证。
文字不是主体；标准术语保留英文，例如 problem diagnosis、RAG、Prompt、structured output、Agent、tool use、fine-tuning、PEFT、baseline。其他说明可用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch07-lora-qlora-low-rank-memory-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "LoRA 与 QLoRA 低秩增量和显存节省图",
        "suggested_page": "docs/ch07-llm-principles/ch06-finetuning/02-lora-qlora.md",
        "alt": "LoRA 与 QLoRA 图：冻结权重 W，加上低秩增量 ΔW=A@B；QLoRA 在此基础上量化基础模型以进一步节省显存。",
        "prompt": """
一张适合解释 LoRA 和 QLoRA 的结构图，主题是“冻结大权重，只训练小增量”。
画面展示 frozen weight W 作为大矩阵不更新，旁边有两个小矩阵 A 和 B 组成 low-rank update ΔW=A@B；右侧展示 QLoRA 把 base model quantized to 4-bit，同时训练 LoRA adapters。
用 memory bars 展示 full fine-tuning、LoRA、QLoRA 的显存占用逐步降低。
文字不是主体；公式和标准术语保留英文，例如 W、ΔW=A@B、LoRA、QLoRA、frozen weights、low-rank update、4-bit、adapters、memory。其他说明可用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch07-peft-placement-family-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "PEFT 方法可训练参数放置位置图",
        "suggested_page": "docs/ch07-llm-principles/ch06-finetuning/03-other-peft.md",
        "alt": "PEFT 方法位置图：Prompt Tuning 放输入 embedding 前，Prefix Tuning 放每层 KV 前缀，Adapter 插层间瓶颈模块，IA3 学通道缩放。",
        "prompt": """
一张适合解释 PEFT 家族差异的 Transformer 剖面图，主题是“不同 PEFT 方法把可训练参数放在不同位置”。
画面展示一个简化 Transformer layer，标出 Prompt Tuning 在 input embedding 前添加软提示，Prefix Tuning 在每层 KV 前缀添加参数，Adapter 插在层间 bottleneck 模块，IA3 学通道缩放向量。
风格像模型结构地图，用颜色区分四种方法，帮助新人建立位置感。
文字不是主体；标准术语保留英文，例如 PEFT、Prompt Tuning、Prefix Tuning、Adapter、IA3、input embedding、KV prefix、bottleneck、scaling。其他说明可用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch07-data-labeling-flywheel-review-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "数据标注质检与飞轮回流图",
        "suggested_page": "docs/ch07-llm-principles/ch06-finetuning/05-data-labeling.md",
        "alt": "数据标注飞轮图：标注一致性、Cohen kappa、低置信度和线上失败样本进入复核队列，再回流训练集和评估集。",
        "prompt": """
一张适合解释微调数据标注质量的飞轮图，主题是“好数据来自持续复核和回流”。
画面形成循环：online failures、dedup、annotation guideline、double labeling、agreement check、Cohen kappa、review queue、hard examples，再回流 train set 和 eval set。
突出低置信度样本和线上失败样本如何进入复核队列。
文字不是主体；标准术语保留英文，例如 online failures、dedup、annotation guideline、double labeling、agreement check、Cohen kappa、review queue、hard examples、train set、eval set。其他说明可用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch07-alignment-hhh-tension-guardrail-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Helpful Honest Harmless 对齐张力图",
        "suggested_page": "docs/ch07-llm-principles/ch07-alignment/01-alignment-problem.md",
        "alt": "HHH 对齐张力图：Helpful、Honest、Harmless 三目标存在张力，需要评估、策略、护栏和人工复核共同落地。",
        "prompt": """
一张适合解释大模型对齐目标的三角张力图，主题是“Helpful、Honest、Harmless 需要同时权衡”。
画面是一个三角形，三个顶点为 Helpful、Honest、Harmless；不同请求点落在三角内部，表示有时帮助性、诚实性和安全性会互相拉扯。
外围加入 guardrails、evaluation rubric、policy、human review，表现应用落地需要机制而不是一句口号。
文字不是主体；标准术语保留英文，例如 Helpful、Honest、Harmless、guardrails、evaluation rubric、policy、human review。其他说明可用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch07-rlhf-reward-kl-loop-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "RLHF 奖励模型与 KL 约束闭环图",
        "suggested_page": "docs/ch07-llm-principles/ch07-alignment/02-rlhf.md",
        "alt": "RLHF 奖励与 KL 图：偏好对训练 Reward Model，策略模型朝高奖励更新，同时 Reference Model 和 KL penalty 防止模型跑偏。",
        "prompt": """
一张适合解释 RLHF 核心机制的闭环图，主题是“奖励引导模型，KL 防止跑偏”。
画面表现 SFT model 生成回答，人类 preference pairs 标记 chosen/rejected，用来训练 Reward Model；Policy Model 朝高 reward 更新，同时 Reference Model 通过 KL penalty 像安全绳一样约束变化。
风格像训练闭环和安全绳隐喻结合，清晰展示角色关系。
文字不是主体；标准术语保留英文，例如 SFT model、preference pairs、chosen、rejected、Reward Model、Policy Model、Reference Model、KL penalty、PPO。其他说明可用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch07-dpo-rlhf-shortcut-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "DPO 相比 RLHF 的偏好优化捷径图",
        "suggested_page": "docs/ch07-llm-principles/ch07-alignment/03-alternative-methods.md",
        "alt": "DPO 与 RLHF 对比图：RLHF 长链需要奖励模型和 PPO，DPO 直接用 chosen/rejected 偏好对优化策略边距。",
        "prompt": """
一张适合解释 DPO 和 RLHF 差异的双路线图，主题是“DPO 把偏好优化链路变短”。
左侧 RLHF long path：SFT、Reward Model、PPO、Policy update；右侧 DPO shortcut：chosen/rejected preference pair 直接优化 policy margin。
用道路或管线长度对比突出 DPO 更直接，但不要暗示它总是更好。
文字不是主体；标准术语保留英文，例如 RLHF、SFT、Reward Model、PPO、Policy update、DPO、chosen、rejected、policy margin。其他说明可用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch07-domain-finetune-evaluation-board-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "垂直领域微调项目评估看板图",
        "suggested_page": "docs/ch07-llm-principles/ch08-projects/01-domain-finetuning.md",
        "alt": "垂直领域微调项目评估看板图：任务边界、SFT 数据、Prompt/RAG baseline、评估规则、before/after 和失败样例组成作品级微调项目看板。",
        "prompt": """
一张适合大模型垂直领域微调项目的评估看板图，主题是“作品级项目要能证明改进来自哪里”。
画面像项目 dashboard，分区展示 task scope、SFT data、Prompt/RAG baseline、evaluation rubric、before/after comparison、failure cases、deployment notes。
强调不仅训练模型，还要有 baseline、评估和失败样例复盘。
文字不是主体；标准术语保留英文，例如 task scope、SFT data、Prompt/RAG baseline、evaluation rubric、before/after、failure cases、deployment notes。其他说明可用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch07-hands-on-workshop-route.png",
        "size": "1024x1792",
        "quality": "high",
        "title": "第 7 章 LLM 实操工作坊路线图",
        "suggested_page": "docs/ch07-llm-principles/ch08-projects/03-stage-hands-on-workshop.md",
        "alt": "第 7 章 LLM 实操工作坊路线图：tokens、payload、Prompt 版本、结构化校验、方案选择和证据包组成一条可跟做路线。",
        "prompt": """
一张竖版、漫画式、流程式的第 7 章 LLM 实操工作坊路线图，主题是“从一句用户请求到可复盘证据包”。
画面从上到下分成 6 个步骤：token trace、request payload、prompt versions、JSON validation、route decision、evidence pack。
每一步都像课程卡片，有箭头连接，旁边有小型终端窗口和文件夹图标，强调用户可以一步一步跟着做。
文字不是主体；标准术语保留英文，例如 token trace、payload、Prompt v1/v2/v3、JSON validation、RAG、fine-tuning、evidence pack。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch07-hands-on-payload-validation-loop.png",
        "size": "1024x1792",
        "quality": "high",
        "title": "Prompt payload 与结构化校验闭环图",
        "suggested_page": "docs/ch07-llm-principles/ch08-projects/03-stage-hands-on-workshop.md",
        "alt": "Prompt payload 与结构化校验闭环图：instructions、input、model、temperature 组成请求，输出经过 JSON parser、field check、type check 和 retry/human review。",
        "prompt": """
一张竖版流程图，主题是“模型回答不能直接进业务流程，必须先通过结构化校验”。
上半部分展示 request payload 卡片，字段包括 model、instructions、input、max_output_tokens、temperature、prompt_version。
中间展示 model-like output 进入 JSON parser，然后经过 field check、type check、enum check、range check。
下半部分分成两条结果：valid output -> workflow；invalid output -> failure reason -> prompt revision / retry / human review。
文字不是主体；标准术语保留英文，例如 request payload、JSON parser、field check、type check、enum check、retry、human review。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch07-hands-on-code-execution-trace.png",
        "size": "1024x1792",
        "quality": "high",
        "title": "第 7 章工作坊代码执行顺序图",
        "suggested_page": "docs/ch07-llm-principles/ch08-projects/03-stage-hands-on-workshop.md",
        "alt": "第 7 章工作坊代码执行顺序图：main 依次执行 build_token_trace、evaluate_prompt_versions、build_route_decisions、save_evidence 并打印 STEP 1 到 STEP 4。",
        "prompt": """
一张竖版代码执行顺序图，主题是“main 函数如何一步步生成第 7 章证据包”。
画面像代码走查漫画：main() 从顶部开始，依次进入 build_token_trace()、evaluate_prompt_versions()、build_route_decisions()、save_evidence()。
右侧对应打印区域：STEP 1 token/vector、STEP 2 prompt evaluation、STEP 3 route check、STEP 4 evidence files。
底部展示 ch07_workshop_evidence 文件夹包含 token_trace.json、prompt_eval.csv、route_decisions.json、failure_cases.md、README.md。
文字不是主体；标准术语保留英文，例如 main()、build_token_trace、evaluate_prompt_versions、save_evidence、STEP 1、prompt_eval.csv。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch07-hands-on-route-decision-ladder.png",
        "size": "1024x1792",
        "quality": "high",
        "title": "Prompt、RAG 与微调路线阶梯图",
        "suggested_page": "docs/ch07-llm-principles/ch08-projects/03-stage-hands-on-workshop.md",
        "alt": "Prompt、RAG 与微调路线阶梯图：先看失败类型，格式问题用结构化输出，知识问题用 RAG，稳定行为问题再评估微调。",
        "prompt": """
一张竖版对比式路线阶梯图，主题是“不要一开始就 fine-tuning，要先根据失败类型选择路线”。
阶梯从低成本到高成本：Prompt first、Structured output、RAG first、Prompt eval then fine-tuning plan。
每一级旁边放一个失败症状卡片：unclear instruction、invalid JSON、missing source/latest policy、stable brand tone failure。
用箭头表现从 failure_cases.md 回到 route decision，再选择下一步。
文字不是主体；标准术语保留英文，例如 Prompt first、Structured output、RAG first、fine-tuning plan、failure_cases.md、route decision。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch07-hands-on-portfolio-evidence-pack.png",
        "size": "1024x1792",
        "quality": "high",
        "title": "第 7 章作品集证据包图",
        "suggested_page": "docs/ch07-llm-principles/ch08-projects/03-stage-hands-on-workshop.md",
        "alt": "第 7 章作品集证据包图：README、token_trace.json、prompt_eval.csv、route_decisions.json 和 failure_cases.md 组成可复现 LLM 工程证据。",
        "prompt": """
一张竖版作品集证据包图，主题是“跑完工作坊后要留下什么，别人才能复现你的判断”。
画面中心是 ch07_workshop_evidence 文件夹，里面清楚展示 README.md、token_trace.json、prompt_eval.csv、route_decisions.json、failure_cases.md 五个文件。
每个文件连到一个用途：how to run、token evidence、prompt scorecard、method choice、failure review。
底部展示 reviewer 看证据包后能回答三个问题：what was tested、what failed、what changes next。
文字不是主体；标准术语保留英文，例如 README.md、token_trace.json、prompt_eval.csv、route_decisions.json、failure_cases.md、reviewer。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
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
        "filename": "ch08-workshop-chunk-execution-flow-map.png",
        "size": "1024x1792",
        "quality": "high",
        "title": "第八章 RAG 工作坊 chunk_documents 执行顺序图",
        "suggested_page": "docs/ch08-rag/ch05-projects/05-stage-hands-on-workshop.md",
        "alt": "RAG 工作坊 chunk_documents 执行顺序图：DOCUMENTS 进入句子切分，按窗口组成 chunk，写入 chunk_id、source、roles 和 text。",
        "prompt": """
一张适合第八章 RAG 实操工作坊的竖版流程教学图，主题是“chunk_documents() 怎样把文档变成可检索片段”。
画面用 5 个纵向步骤表现：DOCUMENTS list、split sentences、group by sentences_per_chunk、create chunk record、append to chunks。
每个 chunk record 画成卡片，卡片字段包含 chunk_id、doc_id、title、source、roles、text；强调 source 和 roles 会跟着 chunk 走。
风格像漫画式代码执行顺序图和教学流程卡片结合，新手友好，适合先看图再看代码。
文字不是主体；标准术语保留英文，例如 DOCUMENTS、chunk_documents()、sentences_per_chunk、chunk_id、source、roles、text。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch08-workshop-retrieve-permission-branch-map.png",
        "size": "1024x1792",
        "quality": "high",
        "title": "第八章 RAG 工作坊 retrieve 权限分支图",
        "suggested_page": "docs/ch08-rag/ch05-projects/05-stage-hands-on-workshop.md",
        "alt": "RAG 工作坊 retrieve 权限分支图：query 命中 chunk 后先检查 score，再按 roles 分到 allowed_hits、blocked_hits 或 no_evidence。",
        "prompt": """
一张适合第八章 RAG 实操工作坊的竖版分支流程图，主题是“retrieve() 先检索，再按权限分流”。
画面表现 query 进入 keyword_score，score = 0 的 chunk 被丢弃；score > 0 的 chunk 进入 permission check。
permission check 分成三条清晰路径：public or role allowed -> allowed_hits；matched but role blocked -> blocked_hits；no matched allowed chunks -> no_evidence 或 blocked_by_permission。
特别突出 private roadmap 例子：public user 命中了 internal.md#roadmap，但进入 blocked_hits，不能进入 answer context。
风格像工程调试分支图和漫画路牌结合，新手一眼能看懂 allowed_hits 与 blocked_hits 的区别。
文字不是主体；标准术语保留英文，例如 retrieve()、keyword_score、score、roles、allowed_hits、blocked_hits、no_evidence、blocked_by_permission、answer context。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch08-workshop-evaluation-pass-fail-flow-map.png",
        "size": "1024x1792",
        "quality": "high",
        "title": "第八章 RAG 工作坊 evaluate 评测流转图",
        "suggested_page": "docs/ch08-rag/ch05-projects/05-stage-hands-on-workshop.md",
        "alt": "RAG 工作坊 evaluate 评测流转图：EVAL_CASES 逐条运行 rag_answer，检查 status_ok 和 citation_ok，汇总 PASS、FAIL 和 passed 计数。",
        "prompt": """
一张适合第八章 RAG 实操工作坊的竖版评测流程图，主题是“evaluate() 把一次 Demo 变成可重复检查”。
画面用 6 个步骤表现：EVAL_CASES、run rag_answer、compare expected_status、check expected_source in citations、mark PASS or FAIL、summary passed/total。
用三张小测试卡片示例 refund_window、api_key_setup、private_block；其中 private_block 显示 blocked_by_permission 且 citations = none 也可以 PASS。
风格像测试看板、代码流程图和学习卡片结合，强调评估不是主观感觉，而是固定问题集。
文字不是主体；标准术语保留英文，例如 EVAL_CASES、rag_answer()、expected_status、expected_source、citations、PASS、FAIL、passed/total。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch08-rag-layer-failure-debug-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "RAG 分层故障定位图",
        "suggested_page": "docs/ch08-rag/ch01-rag/01-rag-basics.md",
        "alt": "RAG 分层故障定位图：从文档切块、检索召回、上下文拼装到生成回答逐层定位问题。",
        "prompt": """
一张适合 RAG 基础课程的分层故障定位图，主题是“RAG 出问题时不要先怪模型”。
画面从 user query 到 document chunks、retrieval top-k、context packing、LLM answer、citation，逐层标出常见失败：chunk too coarse、missed retrieval、wrong ranking、context overflow、unsupported answer。
风格像排障仪表盘和学习流程图结合，清晰、工程化。
文字不是主体；标准术语保留英文，例如 RAG、chunk、top-k、context、LLM answer、citation。其他说明可用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch08-chunk-size-overlap-tradeoff-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Chunk 大小与 overlap 取舍图",
        "suggested_page": "docs/ch08-rag/ch01-rag/02-document-processing.md",
        "alt": "Chunk 大小与 overlap 取舍图：chunk 太大召回不精准，太小证据不完整，overlap 缓解边界信息被切断。",
        "prompt": """
一张适合文档切块教学的取舍图，主题是“chunk size 和 overlap 决定证据能不能被找准、用完整”。
画面用同一段文档展示 large chunk、tiny chunk、balanced chunk with overlap 三种切法，旁边标出 precision、evidence completeness、context cost、boundary loss。
风格像开卷考试笔记卡片和工程参数面板结合，适合新人理解。
文字不是主体；标准术语保留英文，例如 chunk size、overlap、precision、evidence completeness、context cost。其他说明可用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch08-courseware-chunk-metadata-schema-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "课件知识块元数据 schema 图",
        "suggested_page": "docs/ch08-rag/ch01-rag/02-document-processing.md",
        "alt": "课件知识块元数据 schema 图：topic、content_type、source_origin、page_or_slide 等字段支撑概念、例题和练习的稳定组装。",
        "prompt": """
一张适合课件生成项目的知识块 schema 信息图，主题是“知识块不是裸文本，而是带任务目的的卡片”。
画面展示一个 courseware chunk card，包含 text、topic、content_type、source_origin、page_or_slide、grade、source_refs；右侧连接 concept block、example block、exercise block 和 Word template。
风格像结构化数据卡片和模板映射图结合，清晰、新手友好。
文字不是主体；标准术语保留英文，例如 topic、content_type、source_origin、page_or_slide、source_refs、Word template。其他说明可用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch08-vector-record-metadata-filter-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "向量库记录与元数据过滤图",
        "suggested_page": "docs/ch08-rag/ch01-rag/03-vector-databases.md",
        "alt": "向量库记录与元数据过滤图：一条记录由 id、vector、text、metadata 组成，metadata filter 支撑权限、来源和业务范围过滤。",
        "prompt": """
一张适合解释向量数据库记录结构的教学图，主题是“vector database 存的不只是向量”。
画面展示 record = id + vector + text + metadata，query vector 先经过 metadata filter，再做 similarity search，最终返回 text、score、source。
强调 metadata 支撑权限、版本、来源引用和评估。
文字不是主体；标准术语保留英文，例如 vector database、record、metadata filter、similarity search、score、source。其他说明可用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch08-ann-exact-search-tradeoff-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "精确搜索与 ANN 取舍图",
        "suggested_page": "docs/ch08-rag/ch01-rag/03-vector-databases.md",
        "alt": "精确搜索与 ANN 取舍图：精确搜索逐个比较更准但慢，ANN 先缩小候选范围更快但可能牺牲一点最优保证。",
        "prompt": """
一张适合解释 exact search 和 ANN 的对比图，主题是“向量库为什么需要近似最近邻”。
左侧 exact search 像全量逐个比对，右侧 ANN 像先按索引区域缩小候选再找近邻；用速度、规模、准确性三条指标对比。
风格像地图索引和搜索雷达结合，直观但不堆公式。
文字不是主体；标准术语保留英文，例如 exact search、ANN、nearest neighbor、index、candidate set、speed、recall。其他说明可用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch08-hybrid-retrieval-blindspot-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "关键词检索与向量检索盲区图",
        "suggested_page": "docs/ch08-rag/ch01-rag/04-retrieval-strategies.md",
        "alt": "关键词检索与向量检索盲区图：关键词适合精确术语，向量适合同义表达，Hybrid Search 结合两者减少漏召回。",
        "prompt": """
一张适合 Hybrid Search 教学的盲区对比图，主题是“字面匹配和语义匹配各有短板”。
画面左右对比 keyword search 和 vector search：keyword 擅长 error code、product name、policy id；vector 擅长 paraphrase、synonym、colloquial query；中间合并成 hybrid score。
风格像双通道检索仪表和漏斗图结合。
文字不是主体；标准术语保留英文，例如 keyword search、vector search、hybrid score、BM25、embedding、paraphrase。其他说明可用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch08-rerank-query-rewrite-funnel-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Query Rewrite 与 Rerank 双阶段漏斗图",
        "suggested_page": "docs/ch08-rag/ch01-rag/04-retrieval-strategies.md",
        "alt": "Query Rewrite 与 Rerank 双阶段漏斗图：query rewrite 在检索前改写问题，rerank 在粗召回后精排候选。",
        "prompt": """
一张适合解释 query rewrite 和 rerank 差异的双阶段漏斗图，主题是“一个改入口，一个改排序”。
画面表现 user query 先经过 query rewrite 变成 retrieval-friendly query，进入 sparse/dense retrieval 得到 candidate set，再经过 reranker 输出 final context。
风格像检索流水线和质量筛选漏斗结合，帮助新人分清前后位置。
文字不是主体；标准术语保留英文，例如 query rewrite、retrieval-friendly query、candidate set、reranker、final context。其他说明可用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch08-rag-optimization-debug-funnel-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "RAG 优化分层排障漏斗图",
        "suggested_page": "docs/ch08-rag/ch01-rag/05-rag-optimization.md",
        "alt": "RAG 优化分层排障漏斗图：按文档处理、召回、上下文拼装、生成约束和引用检查逐层定位瓶颈。",
        "prompt": """
一张适合 RAG 优化章节的排障漏斗图，主题是“先找瓶颈，再调参数”。
画面从 symptom 进入四层诊断：document processing、retrieval、context packing、generation/citation；每层列出该看的 logs 和该调的 knobs。
风格像故障诊断看板和漏斗图结合，专业但新人友好。
文字不是主体；标准术语保留英文，例如 symptom、logs、document processing、retrieval、context packing、generation、citation、knobs。其他说明可用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch08-rag-experiment-eval-loop-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "RAG 优化实验闭环图",
        "suggested_page": "docs/ch08-rag/ch01-rag/05-rag-optimization.md",
        "alt": "RAG 优化实验闭环图：固定评估集、建立 baseline、一次改一个变量、比较指标和失败样本、决定是否保留。",
        "prompt": """
一张适合 RAG 优化实验的闭环图，主题是“优化要像做实验，不靠感觉”。
画面表现 eval set、baseline、change one variable、run evaluation、compare metrics、inspect failures、keep or rollback 形成循环。
突出 Hit@k、answer correctness、citation_ok、latency、cost 这些指标。
文字不是主体；标准术语保留英文，例如 eval set、baseline、change one variable、Hit@k、citation_ok、latency、cost、rollback。其他说明可用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch08-advanced-rag-architecture-selection-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "高级 RAG 架构选择图",
        "suggested_page": "docs/ch08-rag/ch01-rag/06-advanced-rag.md",
        "alt": "高级 RAG 架构选择图：多知识库用 Router RAG，多步骤问题用 Multi-hop RAG，自主行动用 Agentic RAG，关系链明显时考虑 Graph RAG。",
        "prompt": """
一张适合高级 RAG 架构章节的选择图，主题是“架构升级要对应问题形态”。
画面用四个分支连接 symptoms 到 architecture：multi knowledge bases -> Router RAG；multi-step question -> Multi-hop RAG；dynamic actions -> Agentic RAG；entity relations -> Graph RAG。
风格像决策树和系统架构图结合，强调不是组件越多越高级。
文字不是主体；标准术语保留英文，例如 Router RAG、Multi-hop RAG、Agentic RAG、Graph RAG、structured retrieval。其他说明可用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch08-rag-evaluation-layered-dashboard-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "RAG 分层评估仪表盘图",
        "suggested_page": "docs/ch08-rag/ch01-rag/07-rag-evaluation.md",
        "alt": "RAG 分层评估仪表盘图：检索层、生成层、引用层和系统层分别监控命中率、正确性、faithfulness、延迟和成本。",
        "prompt": """
一张适合 RAG 评估章节的分层仪表盘图，主题是“RAG 评估不能只看最终答案”。
画面分成 retrieval、generation、citation、system 四层，每层有代表指标：Recall@K、MRR、Correctness、Faithfulness、citation_ok、latency、cost、failure rate。
风格像数据仪表盘和教学分层图结合。
文字不是主体；标准术语保留英文，例如 Recall@K、MRR、Correctness、Faithfulness、citation_ok、latency、cost。其他说明可用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch08-faithfulness-citation-check-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Faithfulness 与引用真实性检查图",
        "suggested_page": "docs/ch08-rag/ch01-rag/07-rag-evaluation.md",
        "alt": "Faithfulness 与引用真实性检查图：把答案拆成关键结论，逐条连接到 evidence，区分 supported 与 unsupported。",
        "prompt": """
一张适合解释 faithfulness 和 citation check 的教学图，主题是“答案每个关键结论都要能回到证据”。
画面展示 answer claims 被拆成 claim 1、claim 2、claim 3，每条线连接到 retrieved evidence；能连接的是 supported，无法连接的是 unsupported / hallucinated。
风格像证据链白板和审计检查图结合。
文字不是主体；标准术语保留英文，例如 Faithfulness、claim、evidence、supported、unsupported、citation check、hallucination。其他说明可用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch08-local-model-api-decision-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "本地模型与云 API 部署决策图",
        "suggested_page": "docs/ch08-rag/ch02-deployment/01-local-models.md",
        "alt": "本地模型与云 API 部署决策图：隐私、成本、延迟、离线、质量和运维能力共同决定部署路线。",
        "prompt": """
一张适合模型部署入门的决策图，主题是“本地模型和云 API 是业务约束取舍”。
画面用天平或路线选择器展示 privacy、cost、latency、offline、quality、operations 六个因素，分别指向 cloud API、local model、hybrid route。
风格像产品技术决策看板，清爽实用。
文字不是主体；标准术语保留英文，例如 cloud API、local model、hybrid、privacy、latency、operations。其他说明可用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch08-inference-serving-queue-batch-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "推理服务队列与批处理图",
        "suggested_page": "docs/ch08-rag/ch02-deployment/02-inference-servers.md",
        "alt": "推理服务队列与批处理图：请求进入队列，按 batch 合并执行，在 latency 与 throughput 之间做权衡。",
        "prompt": """
一张适合高性能推理服务课程的系统图，主题是“能跑模型不等于能服务流量”。
画面表现 requests 进入 queue，scheduler 合成 batch，model server 执行，responses 返回；旁边展示 latency、throughput、batch size、timeout、concurrency 的拉扯。
风格像后厨出餐和服务调度面板结合。
文字不是主体；标准术语保留英文，例如 queue、scheduler、batch、model server、latency、throughput、concurrency、timeout。其他说明可用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch08-unified-api-provider-gateway-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "统一 API Provider Gateway 图",
        "suggested_page": "docs/ch08-rag/ch02-deployment/03-unified-api.md",
        "alt": "统一 API Provider Gateway 图：业务层发统一请求，网关内部处理 provider 适配、模型路由、fallback、日志和成本统计。",
        "prompt": """
一张适合统一 API 章节的 provider gateway 架构图，主题是“把 provider 差异隔离在一层”。
画面上方是 business code 发出 unified request，中间是 API gateway，内部有 adapter、routing、fallback、usage logging、error normalization，下方连接 provider A、provider B、local model。
风格像系统网关架构图，清晰专业。
文字不是主体；标准术语保留英文，例如 unified request、API gateway、adapter、routing、fallback、usage、error normalization、provider、local model。其他说明可用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch08-llm-api-robust-client-loop-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "LLM API 稳健客户端闭环图",
        "suggested_page": "docs/ch08-rag/ch03-app-dev/01-llm-api-practice.md",
        "alt": "LLM API 稳健客户端闭环图：配置、请求、timeout、retry、统一响应、usage、日志和 raw output 共同组成稳定调用层。",
        "prompt": """
一张适合大模型 API 工程实践的闭环图，主题是“API 调用要从 demo 变成稳定运行时”。
画面表现 config、request builder、timeout、retry policy、LLM call、response parser、usage tracking、structured log、raw output archive 组成 robust client。
风格像运行时中间层和质量护栏结合，实用、清楚。
文字不是主体；标准术语保留英文，例如 config、timeout、retry、response parser、usage tracking、structured log、raw output、robust client。其他说明可用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch08-langchain-component-pipeline-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "LangChain 组件流水线图",
        "suggested_page": "docs/ch08-rag/ch03-app-dev/02-langchain-basics.md",
        "alt": "LangChain 组件流水线图：Prompt、Retriever、Model、Output Parser 等组件通过清晰输入输出串成应用链路。",
        "prompt": """
一张适合 LangChain 入门的组件流水线图，主题是“框架的价值是拆清组件边界”。
画面展示 user query 依次经过 prompt template、retriever、context builder、model、output parser、app response，每个节点标注 input/output。
风格像模块化积木和数据流图结合，新手友好。
文字不是主体；标准术语保留英文，例如 Prompt Template、Retriever、Context Builder、Model、Output Parser、input/output、Chain。其他说明可用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch08-function-calling-validation-dispatch-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Function Calling 校验与执行闭环图",
        "suggested_page": "docs/ch08-rag/ch03-app-dev/03-function-calling.md",
        "alt": "Function Calling 校验与执行闭环图：模型产出 tool call，程序做 schema 校验、参数校验、dispatch 执行、错误处理和结果回填。",
        "prompt": """
一张适合 Function Calling 章节的工程闭环图，主题是“模型提出调用意图，程序负责安全执行”。
画面表现 user intent 进入 model，输出 structured tool call，经过 schema validation、argument validation、dispatcher、tool execution、error handling/retry，最后 tool result 回到 model 生成最终回答。
风格像桥梁和安全闸门结合，强调边界。
文字不是主体；标准术语保留英文，例如 tool call、schema validation、argument validation、dispatcher、tool execution、retry、tool result。其他说明可用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch08-huggingface-ecosystem-layers-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "HuggingFace 生态层次图",
        "suggested_page": "docs/ch08-rag/ch03-app-dev/04-huggingface-deep.md",
        "alt": "HuggingFace 生态层次图：Datasets、Tokenizers、Models、Pipelines 和 Hub 组成从数据到模型使用的完整生态链。",
        "prompt": """
一张适合 HuggingFace 生态深入页的分层图，主题是“HuggingFace 不只是模型仓库”。
画面从 Datasets 到 Tokenizers、Models、Pipelines、Hub 分层连接，表现数据处理、输入编码、模型计算、任务封装和共享协作。
不要使用真实 logo，用抽象生态平台表现。
文字不是主体；标准术语保留英文，例如 Datasets、Tokenizers、Models、Pipelines、Hub、model card、inference。其他说明可用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch08-dialog-state-slot-memory-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "对话状态、槽位与记忆管理图",
        "suggested_page": "docs/ch08-rag/ch03-app-dev/05-dialog-system.md",
        "alt": "对话状态、槽位与记忆管理图：history、topic、slots、last_tool_result 和 summary 共同支撑多轮对话状态。",
        "prompt": """
一张适合多轮对话系统的状态管理图，主题是“有 history 不等于有 state”。
画面展示 chat history 进入 state manager，提炼出 topic、slots、last retrieved doc、last_tool_result、summary，再决定 ask clarification、call tool 或 answer。
风格像客服工作台和状态机结合，清晰直观。
文字不是主体；标准术语保留英文，例如 history、state manager、topic、slots、summary、tool result、clarification。其他说明可用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch08-ai-coding-human-review-loop-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "AI 辅助编程人工验证闭环图",
        "suggested_page": "docs/ch08-rag/ch03-app-dev/06-ai-assisted-coding.md",
        "alt": "AI 辅助编程人工验证闭环图：需求约束、AI 代码草稿、diff、测试、真实样例和人工 review 共同保证代码质量。",
        "prompt": """
一张适合 AI 辅助编程章节的协作闭环图，主题是“AI 生成草稿，人类负责验证和合并”。
画面表现 requirements、AI draft、diff review、unit tests、real sample run、security check、human review、merge decision 形成循环。
风格像开发工作流看板和质量门禁结合。
文字不是主体；标准术语保留英文，例如 requirements、AI draft、diff review、unit tests、security check、human review、merge。其他说明可用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch08-document-parsing-format-router-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "PDF Word PPT 文档解析路由图",
        "suggested_page": "docs/ch08-rag/ch03-app-dev/07-document-parsing.md",
        "alt": "PDF Word PPT 文档解析路由图：不同文件类型进入不同解析链，恢复文本、结构、页码、内容类型和来源信息。",
        "prompt": """
一张适合文档解析与知识抽取章节的路由图，主题是“文档解析不是一个解析器走天下”。
画面表现 PDF、scanned PDF/OCR、DOCX、PPTX 进入 parser router，再分别经过 text extraction、layout/order recovery、metadata、content_type detection，输出 structured chunks。
风格像文件分拣流水线和知识抽取工厂结合。
文字不是主体；标准术语保留英文，例如 PDF、DOCX、PPTX、OCR、parser router、metadata、content_type、structured chunks。其他说明可用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch08-template-schema-to-render-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "结构化课件到模板渲染图",
        "suggested_page": "docs/ch08-rag/ch03-app-dev/08-template-doc-generation.md",
        "alt": "结构化课件到模板渲染图：courseware schema 先整理成 template payload，再填入 Word/PPT 模板并导出文档。",
        "prompt": """
一张适合模板化文档生成章节的工程图，主题是“先结构化，再模板渲染，不直接自由写 Word”。
画面展示 courseware schema 进入 template payload mapper，分成 title、concept_block、example_block、exercise_block、source_block，再填入 Word/PPT template，导出 docx/pptx。
风格像文档生产线和字段映射图结合。
文字不是主体；标准术语保留英文，例如 courseware schema、template payload、concept_block、example_block、source_block、Word template、docx、pptx。其他说明可用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch08-async-concurrency-semaphore-timeout-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "异步并发、Semaphore 与 timeout 控制图",
        "suggested_page": "docs/ch08-rag/ch04-engineering/01-async-programming.md",
        "alt": "异步并发控制图：gather 并发等待，Semaphore 限制同时请求数，timeout 防止单个上游卡死。",
        "prompt": """
一张适合异步编程章节的工程机制图，主题是“异步不是无限并发，而是聪明等待和受控并发”。
画面表现 asyncio.gather 同时发起多个 I/O tasks，Semaphore gate 限制并发，timeout watchdog 处理慢任务，最后汇总结果。
风格像厨房并行备菜和后端调度图结合，直观、轻松。
文字不是主体；标准术语保留英文，例如 asyncio.gather、I/O tasks、Semaphore、timeout、rate limit、results。其他说明可用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch08-api-contract-error-version-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "API 契约、错误结构与版本管理图",
        "suggested_page": "docs/ch08-rag/ch04-engineering/02-api-design.md",
        "alt": "API 契约图：request schema、response schema、error object、trace_id 和 version 共同组成稳定服务接口。",
        "prompt": """
一张适合 API 设计章节的系统契约图，主题是“API 不是随便包个 JSON，而是长期稳定的契约”。
画面展示 client 通过 /api/v1 调用 service，接口契约分成 request schema、response schema、error object、trace_id、version、idempotency。
风格像合同文档和服务接口架构图结合，清晰专业。
文字不是主体；标准术语保留英文，例如 request schema、response schema、error object、trace_id、version、idempotency、/api/v1。其他说明可用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch08-observability-logs-metrics-trace-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "日志、指标与 Trace 可观测性图",
        "suggested_page": "docs/ch08-rag/ch04-engineering/03-logging-monitoring.md",
        "alt": "日志指标 Trace 可观测性图：logs 记录事件，metrics 观察趋势，trace 还原单条请求链路。",
        "prompt": """
一张适合 LLM 日志与监控章节的可观测性图，主题是“系统坏了要能看见哪里坏”。
画面分成 logs、metrics、trace 三块：logs 记录 stage events，metrics 汇总 latency/error/token/retrieval hit，trace 串起 api、retrieval、tool、LLM、response。
风格像运维仪表盘和链路追踪图结合。
文字不是主体；标准术语保留英文，例如 logs、metrics、trace、latency、error rate、tokens、retrieval hit、LLM。其他说明可用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch08-docker-image-container-compose-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Docker 镜像、容器与 Compose 部署图",
        "suggested_page": "docs/ch08-rag/ch04-engineering/04-docker-deployment.md",
        "alt": "Docker 部署图：Dockerfile 构建 image，image 启动 container，Compose 协调 app、vector db、redis 等多个服务。",
        "prompt": """
一张适合 Docker 部署章节的概念图，主题是“容器化把应用和运行环境标准化”。
画面展示 Dockerfile -> image -> container 的关系，旁边用 Docker Compose 编排 app service、vector db、redis、logs，并标出 environment variables 和 health check。
风格像部署蓝图和服务编排图结合。
文字不是主体；标准术语保留英文，例如 Dockerfile、image、container、Docker Compose、environment variables、health check、vector db、redis。其他说明可用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch08-enterprise-kb-permission-citation-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "企业知识库权限与引用闭环图",
        "suggested_page": "docs/ch08-rag/ch05-projects/01-enterprise-kb.md",
        "alt": "企业知识库权限与引用闭环图：用户权限先过滤候选文档，检索和生成后答案必须带来源引用并记录审计日志。",
        "prompt": """
一张适合企业知识库问答项目的系统图，主题是“企业 RAG 要同时正确、合规、可追溯”。
画面表现 user role/permission 先过滤 public/internal docs，再 retrieval、rerank、answer with sources，最后进入 citation check 和 audit log。
风格像企业权限系统和 RAG 流程图结合，可靠、清晰。
文字不是主体；标准术语保留英文，例如 permission filter、public docs、internal docs、retrieval、rerank、sources、citation check、audit log。其他说明可用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch08-rag-finetune-responsibility-split-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "RAG 与微调职责拆分图",
        "suggested_page": "docs/ch08-rag/ch05-projects/02-domain-rag-finetuning.md",
        "alt": "RAG 与微调职责拆分图：RAG 负责知识更新和来源引用，fine-tuning 负责回答风格、格式稳定和业务口径。",
        "prompt": """
一张适合 RAG+微调综合项目的职责拆分图，主题是“RAG 补知识，fine-tuning 补行为”。
画面左右分栏：RAG side 包含 documents、retrieval、citations、knowledge update；fine-tuning side 包含 style、format、domain tone、task behavior；中间合成 final answer。
风格像系统职责边界图，强调两者互补不是替代。
文字不是主体；标准术语保留英文，例如 RAG、fine-tuning、knowledge update、citations、style、format、domain behavior、final answer。其他说明可用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch08-assistant-session-tool-trace-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "智能助手 session、检索与工具 Trace 图",
        "suggested_page": "docs/ch08-rag/ch05-projects/03-intelligent-assistant.md",
        "alt": "智能助手 session、检索与工具 Trace 图：多轮对话中 session state、retrieval、tool call、answer 和 state update 共同形成持续协作闭环。",
        "prompt": """
一张适合智能问答助手项目的多轮 trace 图，主题是“助手感来自状态、检索和工具协作”。
画面表现 turn 1、turn 2、turn 3 的对话流，每轮更新 session state；需要知识时 retrieval，需要用户状态时 tool call，最后 answer 后写回 state。
风格像产品对话轨迹和系统链路图结合。
文字不是主体；标准术语保留英文，例如 session state、retrieval、tool call、answer、state update、trace、multi-turn。其他说明可用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch08-courseware-assistant-production-line-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "课件生成助手生产线图",
        "suggested_page": "docs/ch08-rag/ch05-projects/04-courseware-assistant.md",
        "alt": "课件生成助手生产线图：PDF/Word/PPT 资料入库，解析成知识块，检索主题和例题，生成 courseware schema，再导出 Word。",
        "prompt": """
一张适合知识库驱动课件生成助手项目的生产线图，主题是“从资料库到 Word 课件的可调试链路”。
画面表现 PDF/Word/PPT 进入 document parsing，变成 structured chunks，按 topic/content_type retrieval，内部资料优先、外部资料补充，生成 courseware schema，最后 Word template export。
风格像教育内容工厂和系统架构图结合，适合新人一眼看懂项目闭环。
文字不是主体；标准术语保留英文，例如 PDF、Word、PPT、document parsing、structured chunks、topic、content_type、courseware schema、Word template、export。其他说明可用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
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
        "filename": "ch09-agent-vs-workflow-backbone.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Agent 与普通应用执行主线图",
        "suggested_page": "docs/ch09-agent/index.md",
        "alt": "Agent 与普通应用执行主线图：目标、任务理解、计划、工具选择、执行动作、观察结果、重新规划和输出记录组成 Agent 执行闭环。",
        "prompt": """
一张适合解释 Agent 和普通 LLM 应用区别的执行主线图，主题是“固定回答变成动态行动闭环”。
画面左侧是普通 LLM app：input、context、answer；右侧是 Agent：goal、plan、tool、action、observation、replan、result、trace。
风格像对比图和状态机结合，突出 Agent 会根据观察结果调整路线。
文字不是主体；标准术语保留英文，例如 LLM app、Agent、goal、plan、tool、observation、replan、trace。其他说明可用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch09-agentops-control-loop.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "AgentOps 可控执行闭环图",
        "suggested_page": "docs/ch09-agent/index.md",
        "alt": "AgentOps 可控执行闭环图：用户目标、任务边界、计划步骤、工具白名单、执行观察、日志成本、高风险判断、人工确认和继续完成组成可控闭环。",
        "prompt": """
一张适合 AgentOps 精讲的可控执行闭环图，主题是“把 Agent 的自由度放进边界里”。
画面表现 user goal、task boundary、plan steps、tool whitelist、execution and observation、logs and cost、risk check、human-in-the-loop、continue or finish。
风格像安全控制台和执行流程图结合，强调权限、确认和可追踪。
文字不是主体；标准术语保留英文，例如 AgentOps、tool whitelist、logs、cost、risk check、human-in-the-loop、trace。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch09-study-guide-minimal-agent-loop.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "最小 Agent 学习闭环图",
        "suggested_page": "docs/ch09-agent/study-guide.md",
        "alt": "最小 Agent 学习闭环图：目标、计划、工具调用、观察结果、更新状态、判断完成和输出结果形成单 Agent 闭环。",
        "prompt": """
一张适合 Agent 学习指南的最小闭环图，主题是“第一遍先把单 Agent 做稳”。
画面表现 goal、plan、tool call、observation、state update、done check、final output，未完成时回到 plan。
风格像轻量状态机和学习路线图结合，让新人知道 Agent 不是魔法，而是一串可检查步骤。
文字不是主体；标准术语保留英文，例如 goal、plan、tool call、observation、state update、done check、final output。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch09-basics-position-bridge.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Agent 基础位置桥接图",
        "suggested_page": "docs/ch09-agent/ch01-agent-basics/00-roadmap.md",
        "alt": "Agent 基础位置桥接图：LLM 应用、对话生成、RAG 接入知识、工具调用、Agent、目标驱动多步执行和可观察可恢复可评估逐步连接。",
        "prompt": """
一张适合 Agent 基础导读页的位置桥接图，主题是“从 LLM 应用走向目标驱动系统”。
画面表现 LLM app、dialogue generation、RAG、tool calling、Agent、goal-driven multi-step execution、observable/recoverable/evaluable 逐步连接。
风格像学习阶段桥梁，帮助新人理解 Agent 在课程中的位置。
文字不是主体；标准术语保留英文，例如 LLM app、RAG、tool calling、Agent、goal-driven、observable、recoverable、evaluable。其他说明可用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch09-basics-chapter-flow.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Agent 基础章节学习顺序图",
        "suggested_page": "docs/ch09-agent/ch01-agent-basics/00-roadmap.md",
        "alt": "Agent 基础章节学习顺序图：分清 Agent 边界、理解发展脉络、建立能力层级、拆解系统结构、跑通单 Agent 闭环再进入推理工具记忆 MCP 多 Agent。",
        "prompt": """
一张适合 Agent 基础章的新手学习顺序图，主题是“先分清边界，再跑通单 Agent 闭环”。
画面表现 Agent boundary、history、capability levels、system architecture、single Agent loop，再连接 reasoning、tools、memory、MCP、multi-agent。
风格像课程章节路线图和能力阶梯结合，简洁、清晰。
文字不是主体；标准术语保留英文，例如 Agent boundary、capability levels、system architecture、single Agent loop、reasoning、tools、memory、MCP。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch09-basics-execution-loop.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "单 Agent 执行闭环图",
        "suggested_page": "docs/ch09-agent/ch01-agent-basics/00-roadmap.md",
        "alt": "单 Agent 执行闭环图：用户目标、任务理解、制定下一步计划、调用工具或生成回答、观察结果、更新状态、判断目标完成和输出过程记录。",
        "prompt": """
一张适合 Agent 基础章的单 Agent 执行闭环图，主题是“Agent 是围绕目标组织模型、工具、状态和反馈的系统”。
画面表现 user goal、task understanding、next-step plan、tool or answer、observation、state update、goal done check、result and trace。
风格像状态机和工程白板结合，突出循环和停止条件。
文字不是主体；标准术语保留英文，例如 user goal、plan、tool、observation、state update、done check、trace。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch09-rl-agent-breakthroughs-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "强化学习到 Agent 历史突破地图",
        "suggested_page": "docs/ch09-agent/ch01-agent-basics/05-rl-to-agent-breakthroughs.md",
        "alt": "强化学习到 Agent 历史突破地图：TD-Gammon、DQN Atari、AlphaGo、RLHF 和 ReAct 共同连接行动、反馈、搜索、规划与工具调用。",
        "prompt": """
一张适合 Agent 历史补充页的突破地图，主题是“从强化学习的行动反馈到现代 LLM Agent”。
画面展示 TD-Gammon 自我对弈、DQN Atari 从画面到动作、AlphaGo 结合策略价值和搜索、RLHF 使用人类偏好、ReAct 让模型推理与行动交替，最后汇入可观察的 Agent execution loop。
风格像游戏棋盘、环境交互和工程流程图结合，帮助新人理解 state、action、reward、planning、tool use 的关系。
文字不是主体；中文写概念提示，标准术语保留英文，例如 TD-Gammon、DQN、Atari、AlphaGo、RLHF、ReAct、state、action、reward。不要整段英文说明、乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch09-reasoning-chapter-flow.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Agent 推理与规划章节学习顺序图",
        "suggested_page": "docs/ch09-agent/ch02-reasoning/00-roadmap.md",
        "alt": "Agent 推理与规划章节学习顺序图：推理能力、链式推理、ReAct、Plan-and-Execute、更复杂规划和推理评估逐步连接。",
        "prompt": """
一张适合 Agent 推理与规划导读页的章节学习顺序图，主题是“让 Agent 从会调用能力走向会组织行动”。
画面表现 reasoning ability、chain reasoning、ReAct、Plan-and-Execute、advanced planning、reasoning evaluation 逐步连接。
风格像思考路线图和工具行动轨迹结合，突出“想”和“做”的交错。
文字不是主体；标准术语保留英文，例如 reasoning、chain reasoning、ReAct、Plan-and-Execute、advanced planning、evaluation。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch09-tools-chapter-flow.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Agent 工具使用章节学习顺序图",
        "suggested_page": "docs/ch09-agent/ch03-tools/00-roadmap.md",
        "alt": "Agent 工具使用章节学习顺序图：Function Calling、工具描述、参数 schema、工具返回、错误恢复、权限安全边界和多工具实战逐步连接。",
        "prompt": """
一张适合 Agent 工具使用导读页的章节学习顺序图，主题是“从语言能力走向受控执行能力”。
画面表现 Function Calling、tool description、parameter schema、tool result、error recovery、permission boundary、multi-tool practice 逐步连接。
风格像工具箱路线图和执行 trace 结合，突出工具不是越多越好，而是越清晰越可靠。
文字不是主体；标准术语保留英文，例如 Function Calling、tool description、parameter schema、tool result、error recovery、permission boundary、trace。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch09-tools-action-layer-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Agent 工具行动层地图",
        "suggested_page": "docs/ch09-agent/ch03-tools/00-roadmap.md",
        "alt": "Agent 工具行动层地图：用户目标、Agent 规划、选择工具、生成参数、执行工具、观察结果和继续规划或输出组成行动层。",
        "prompt": """
一张适合 Agent 工具章位置说明的行动层地图，主题是“工具让 Agent 从语言层进入真实工作流”。
画面表现 user goal、Agent planning、tool selection、argument generation、tool execution、observation、continue planning or output。
风格像外部世界连接图和执行流水线结合，突出工具是 Agent 连接 API、文件、数据库和代码环境的接口。
文字不是主体；标准术语保留英文，例如 Agent planning、tool selection、argument generation、tool execution、observation、API、database。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch09-tool-control-loop.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Agent 工具受控调用闭环图",
        "suggested_page": "docs/ch09-agent/ch03-tools/00-roadmap.md",
        "alt": "Agent 工具受控调用闭环图：计划下一步、判断是否需要工具、选择工具、生成结构化参数、参数校验、执行工具、解析观察结果、失败修正或更新状态。",
        "prompt": """
一张适合 Agent 工具章主线的受控调用闭环图，主题是“工具调用不是模型想调就调，而是在边界内把计划转成动作”。
画面表现 next step、need tool check、tool selection、structured args、args validation、tool execution、observation parsing、failure repair、state update。
风格像安全阀门和流程引擎结合，强调参数校验、失败恢复和权限边界。
文字不是主体；标准术语保留英文，例如 tool selection、structured args、validation、tool execution、observation、failure repair、state update。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch09-memory-chapter-flow.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Agent 记忆系统章节学习顺序图",
        "suggested_page": "docs/ch09-agent/ch04-memory/00-roadmap.md",
        "alt": "Agent 记忆系统章节学习顺序图：记忆类型、短期上下文、长期记忆、情景与程序性信息、检索更新机制、遗忘安全校验逐步连接。",
        "prompt": """
一张适合 Agent 记忆系统导读页的章节学习顺序图，主题是“记忆服务任务，不是越多越智能”。
画面表现 memory types、short-term context、long-term memory、episodic memory、procedural memory、retrieval/update、forgetting and safety check 逐步连接。
风格像分层记忆仓库和任务助手路线图结合，清晰、温和。
文字不是主体；标准术语保留英文，例如 memory types、short-term context、long-term memory、episodic、procedural、retrieval、update、safety check。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch09-memory-write-retrieve-loop.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Agent 记忆写入检索闭环图",
        "suggested_page": "docs/ch09-agent/ch04-memory/00-roadmap.md",
        "alt": "Agent 记忆写入检索闭环图：新信息、是否值得记、当前上下文、结构化摘要、记忆存储、未来任务检索、有效性验证、用于计划回答、更新或遗忘。",
        "prompt": """
一张适合 Agent 记忆主线的写入检索闭环图，主题是“什么时候存、存成什么、什么时候取、取出来是否可信”。
画面表现 new information、worth remembering check、current context、structured summary、memory store、future retrieval、validity check、use in plan/answer、update or forget。
风格像数据生命周期图和任务上下文管理图结合，突出筛选、验证、遗忘。
文字不是主体；标准术语保留英文，例如 structured summary、memory store、retrieval、validity check、plan、answer、update、forget。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch09-mcp-chapter-flow.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "MCP 章节学习顺序图",
        "suggested_page": "docs/ch09-agent/ch05-mcp/00-roadmap.md",
        "alt": "MCP 章节学习顺序图：MCP 定位、Client 与 Server、工具资源提示词、最小 MCP Server、MCP Client 接入和真实工具生态逐步连接。",
        "prompt": """
一张适合 MCP 导读页的章节学习顺序图，主题是“把外部工具和上下文用统一协议接入 Agent”。
画面表现 MCP positioning、Client and Server、tools/resources/prompts、minimal MCP Server、MCP Client integration、real tool ecosystem 逐步连接。
风格像协议连接地图和插件生态图结合，清晰表达连接层。
文字不是主体；标准术语保留英文，例如 MCP、Client、Server、tools、resources、prompts、integration、ecosystem。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch09-mcp-capability-bridge.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "MCP 能力接入桥接图",
        "suggested_page": "docs/ch09-agent/ch05-mcp/00-roadmap.md",
        "alt": "MCP 能力接入桥接图：外部能力封装为 MCP Server，声明工具和资源，MCP Client 连接，模型应用读取能力描述，按任务调用，返回观察结果，Agent 继续决策。",
        "prompt": """
一张适合 MCP 主线的能力接入桥接图，主题是“MCP 让工具、资源和提示词模板被统一发现和调用”。
画面表现 external capability、MCP Server、tools/resources declaration、MCP Client、model app reads capability description、task call、observation result、Agent decision。
风格像桥梁架构图，突出 Server、Client 和能力描述的关系。
文字不是主体；标准术语保留英文，例如 external capability、MCP Server、resources、MCP Client、capability description、observation、Agent decision。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch09-frameworks-position-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Agent 框架位置地图",
        "suggested_page": "docs/ch09-agent/ch06-frameworks/00-roadmap.md",
        "alt": "Agent 框架位置地图：Agent 基础、推理规划、工具记忆、MCP、Agent 框架、多 Agent、部署和评估逐步连接。",
        "prompt": """
一张适合 Agent 框架章的学习位置地图，主题是“框架不是魔法，而是对状态、工具、流程和日志的抽象”。
画面表现 Agent basics、reasoning/planning、tools/memory、MCP、Agent frameworks，再连接 multi-agent、deployment、evaluation。
风格像学习路线图和工程抽象层地图结合，提醒先理解系统再选框架。
文字不是主体；标准术语保留英文，例如 Agent basics、reasoning、tools、memory、MCP、frameworks、multi-agent、deployment、evaluation。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch09-framework-selection-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Agent 框架选型地图",
        "suggested_page": "docs/ch09-agent/ch06-frameworks/00-roadmap.md",
        "alt": "Agent 框架选型地图：框架总览、LangGraph、LlamaIndex、CrewAI、AutoGen、框架选型和最小可控 Agent 项目逐步连接。",
        "prompt": """
一张适合 Agent 框架选型的地图，主题是“根据任务复杂度选框架，而不是哪个火学哪个”。
画面表现 framework overview 分向 LangGraph、LlamaIndex、CrewAI、AutoGen，再汇入 framework selection 和 minimal controllable Agent project。
风格像决策地图和技术栈路线图结合，突出不同框架侧重点。
文字不是主体；标准术语保留英文，例如 LangGraph、LlamaIndex、CrewAI、AutoGen、framework selection、state graph、RAG、role collaboration。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch09-multi-agent-chapter-flow.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "多 Agent 章节学习顺序图",
        "suggested_page": "docs/ch09-agent/ch07-multi-agent/00-roadmap.md",
        "alt": "多 Agent 章节学习顺序图：判断是否需要多 Agent、选择协作架构、定义角色职责、设计通信协议、管理共享状态、汇总审查决策、评估成本质量逐步连接。",
        "prompt": """
一张适合多 Agent 导读页的章节学习顺序图，主题是“多 Agent 是分工协作机制，不是复制多个聊天机器人”。
画面表现 need multi-agent check、collaboration architecture、roles and responsibilities、communication protocol、shared state、aggregation/review/decision、cost and quality evaluation。
风格像团队协作看板和系统架构图结合，突出边界和最终负责人。
文字不是主体；标准术语保留英文，例如 multi-agent、roles、communication protocol、shared state、review、decision、cost、quality。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch09-multi-agent-coordination-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "多 Agent 协作协调图",
        "suggested_page": "docs/ch09-agent/ch07-multi-agent/00-roadmap.md",
        "alt": "多 Agent 协作协调图：用户目标、总控 Agent、研究 Agent、执行 Agent、审查 Agent、中间结果、汇总冲突处理和最终答案或行动。",
        "prompt": """
一张适合多 Agent 主线的协作协调图，主题是“协调成本必须小于分工收益”。
画面表现 user goal 进入 orchestrator Agent，再分派给 research Agent、execution Agent、review Agent，产出 intermediate results，进入 aggregation/conflict resolution，最终输出 answer or action。
风格像协作网络图和项目指挥中心结合，强调消息、共享状态和最终决策者。
文字不是主体；标准术语保留英文，例如 orchestrator Agent、research Agent、execution Agent、review Agent、intermediate results、aggregation、conflict resolution。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch09-eval-safety-chapter-flow.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Agent 评估安全章节学习顺序图",
        "suggested_page": "docs/ch09-agent/ch08-eval-safety/00-roadmap.md",
        "alt": "Agent 评估安全章节学习顺序图：任务成功标准、结果评估、过程评估、安全边界、Guardrails、日志调用轨迹和评估结果迭代系统逐步连接。",
        "prompt": """
一张适合 Agent 评估与安全导读页的章节学习顺序图，主题是“Agent 不只要能跑，还要知道跑得好不好、安不安全”。
画面表现 success criteria、result evaluation、process evaluation、safety boundary、Guardrails、logs and traces、evaluation-driven iteration。
风格像安全评估仪表盘和课程路线图结合，突出过程评估。
文字不是主体；标准术语保留英文，例如 success criteria、process evaluation、Guardrails、logs、trace、safety boundary、iteration。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch09-agent-risk-debug-loop.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Agent 风险排障闭环图",
        "suggested_page": "docs/ch09-agent/ch08-eval-safety/00-roadmap.md",
        "alt": "Agent 风险排障闭环图：测试任务集、Agent 执行、记录轨迹、评估结果质量、检查工具调用、检查安全规则、发现失败模式、修复 Prompt 工具权限流程。",
        "prompt": """
一张适合 Agent 评估安全主线的风险排障闭环图，主题是“每次失败都要能归因到模型、计划、工具、权限或表达”。
画面表现 test task set、Agent execution、trace recording、result quality evaluation、tool call check、safety rule check、failure pattern discovery、fix Prompt/tools/permissions/workflow。
风格像事故复盘白板和质量控制台结合，强调可观测性和持续修复。
文字不是主体；标准术语保留英文，例如 test task set、trace recording、tool call check、safety rule、failure pattern、Prompt、permissions、workflow。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch09-deployment-chapter-flow.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Agent 部署运维章节学习顺序图",
        "suggested_page": "docs/ch09-agent/ch09-deployment/00-roadmap.md",
        "alt": "Agent 部署运维章节学习顺序图：服务架构、运行时任务队列、状态轨迹持久化、失败恢复、延迟成本错误监控、灰度发布和持续迭代逐步连接。",
        "prompt": """
一张适合 Agent 部署运维导读页的章节学习顺序图，主题是“从本地 Demo 走向可运行、可观察、可恢复的服务”。
画面表现 service architecture、runtime and task queue、state/trace persistence、failure recovery、latency/cost/error monitoring、canary release、continuous iteration。
风格像生产运维路线图和服务架构图结合，清晰、工程化。
文字不是主体；标准术语保留英文，例如 runtime、task queue、state persistence、trace、failure recovery、monitoring、canary release。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch09-production-runtime-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Agent 生产运行架构图",
        "suggested_page": "docs/ch09-agent/ch09-deployment/00-roadmap.md",
        "alt": "Agent 生产运行架构图：用户请求、任务创建、Agent 执行、模型工具调用、状态持久化、任务完成判断、返回结果、日志轨迹、监控告警评估组成运行闭环。",
        "prompt": """
一张适合 Agent 部署主线的生产运行架构图，主题是“生产 Agent 的关键是每一步都能记录、恢复、限制和优化”。
画面表现 user request、task creation、Agent runtime、model/tool calls、state persistence、done check、result response、logs/traces、monitoring/alerts/evaluation。
风格像云服务架构图和任务运行时控制台结合，突出状态、日志和恢复。
文字不是主体；标准术语保留英文，例如 Agent runtime、model/tool calls、state persistence、done check、logs、traces、monitoring、alerts、evaluation。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch09-deployment-observability-loop.png",
        "size": "1024x1024",
        "quality": "medium",
        "title": "Agent 部署可观测恢复闭环图",
        "suggested_page": "docs/ch09-agent/ch09-deployment/00-roadmap.md",
        "alt": "Agent 部署可观测恢复闭环图：Agent 执行、模型工具调用、日志轨迹、监控告警评估、失败恢复、状态持久化和返回结果组成生产闭环。",
        "prompt": """
一张适合 Agent 部署运维主线的简洁闭环图，主题是“上线后的 Agent 要能看见问题并恢复”。
画面中心是 Agent runtime，周围只保留四个关键环节：logs/traces、monitoring、failure recovery、state persistence，最后回到 result response。
风格像简洁的可观测性控制台和恢复流程图，层级少、留白充足。
文字不是主体；标准术语保留英文，例如 Agent runtime、logs、traces、monitoring、failure recovery、state persistence。其他说明只用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch09-projects-route-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Agent 综合项目路线图",
        "suggested_page": "docs/ch09-agent/ch10-projects/00-roadmap.md",
        "alt": "Agent 综合项目路线图：用户目标、任务规划、工具调用、观察结果、状态记忆、评估安全检查、是否完成、结果交付和过程记录形成项目闭环。",
        "prompt": """
一张适合 Agent 综合项目导读页的项目路线图，主题是“把推理、工具、记忆、评估、安全和部署装进一个可复盘项目”。
画面表现 user goal、task planning、tool call、observation、state and memory、evaluation and safety check、done check、deliver result and trace。
风格像作品集项目路线图和执行闭环图结合，强调可追踪。
文字不是主体；标准术语保留英文，例如 task planning、tool call、observation、state、memory、evaluation、safety check、trace、portfolio。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch09-project-learning-order-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Agent 项目学习顺序图",
        "suggested_page": "docs/ch09-agent/ch10-projects/00-roadmap.md",
        "alt": "Agent 项目学习顺序图：先做研究助手，练习检索引用总结，再做数据分析 Agent 练习工具调用结果解释，最后做多 Agent 开发团队练习角色分工协调审查。",
        "prompt": """
一张适合 Agent 项目章的新手学习顺序图，主题是“先做研究助手，再升级数据分析 Agent 和多 Agent 团队”。
画面表现 research assistant、retrieval/citation/summary、data analysis Agent、Python tool and chart、multi-agent dev team、role coordination、review loop。
风格像项目升级阶梯和作品集路线图结合，鼓励逐步完成。
文字不是主体；标准术语保留英文，例如 research assistant、citation、data analysis Agent、Python tool、multi-agent dev team、review loop。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch09-project-delivery-loop.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Agent 项目交付闭环图",
        "suggested_page": "docs/ch09-agent/ch10-projects/00-roadmap.md",
        "alt": "Agent 项目交付闭环图：任务输入、计划列表、执行步骤、工具调用日志、观察中间结果、失败处理、最终输出、评估复盘和部署演示形成作品集闭环。",
        "prompt": """
一张适合 Agent 项目交付的闭环图，主题是“作品集展示的是可追踪的执行闭环，不是一次模型输出”。
画面表现 task input、plan list、execution steps、tool call logs、observations and intermediate results、failure handling、final output、evaluation review、deployment/demo。
风格像项目交付看板和 trace 时间线结合，强调 README、日志、失败案例和评估。
文字不是主体；标准术语保留英文，例如 plan list、execution steps、tool call logs、observations、failure handling、evaluation review、deployment、demo、README。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch09-workshop-single-agent-loop-flow-map.png",
        "size": "1024x1792",
        "quality": "high",
        "title": "第九章 Agent 工作坊单 Agent 执行循环图",
        "suggested_page": "docs/ch09-agent/ch10-projects/04-stage-hands-on-workshop.md",
        "alt": "Agent 工作坊单 Agent 执行循环图：goal 进入状态，planner 选择工具，tool 返回 observation，state 更新并写入 trace，直到 done check。",
        "prompt": """
一张适合第九章 Agent 实操工作坊的竖版流程教学图，主题是“一个可控单 Agent 怎样从目标执行到完成”。
画面用 6 个纵向步骤表现：goal input、state、planner decision、tool call、observation、state update + trace、done check。
突出 beginner workflow：先看图，再看 Python 代码，再看命令输出。
风格像漫画式代码执行顺序图和 AgentOps 控制台结合，新手友好。
文字不是主体；标准术语保留英文，例如 goal、state、planner、tool call、observation、trace、done check。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch09-workshop-tool-schema-permission-map.png",
        "size": "1024x1792",
        "quality": "high",
        "title": "第九章 Agent 工作坊工具 Schema 与权限图",
        "suggested_page": "docs/ch09-agent/ch10-projects/04-stage-hands-on-workshop.md",
        "alt": "Agent 工作坊工具 Schema 与权限图：工具调用先过 required、type、unknown argument 校验，再按 read_only、write_limited 风险决定是否需要人工批准。",
        "prompt": """
一张适合第九章 Agent 实操工作坊的竖版分步骤图，主题是“tool call 不能直接执行，必须先校验 schema 和权限”。
画面表现 tool name、arguments 进入 validation：required fields、type check、unknown argument；通过后进入 permission gate：read_only 直接执行，write_limited 需要 human approval，destructive 默认禁用。
突出 publish_report 被 blocked_by_approval 的例子。
风格像安全检查站、代码 schema 卡片和漫画路牌结合。
文字不是主体；标准术语保留英文，例如 tool schema、required、type check、unknown argument、read_only、write_limited、human approval、blocked_by_approval。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch09-workshop-trace-jsonl-replay-map.png",
        "size": "1024x1792",
        "quality": "high",
        "title": "第九章 Agent 工作坊 Trace JSONL 复盘图",
        "suggested_page": "docs/ch09-agent/ch10-projects/04-stage-hands-on-workshop.md",
        "alt": "Agent 工作坊 Trace JSONL 复盘图：每一步记录 run_id、step、thought、action、arguments、observation 和 next_decision，支持事后 replay。",
        "prompt": """
一张适合第九章 Agent 实操工作坊的竖版调试图，主题是“trace 让 Agent 的每一步可以复盘”。
画面表现 agent_traces.jsonl 文件，每一行是 JSON record，字段包括 run_id、step、thought、action、arguments、observation、next_decision。
旁边展示 replay 流程：read JSONL、group by run_id、inspect failed step、fix tool schema or permission、rerun eval。
风格像日志查看器、侦探白板和教学卡片结合，强调不是只看最终答案。
文字不是主体；标准术语保留英文，例如 agent_traces.jsonl、run_id、step、thought、action、arguments、observation、next_decision、replay。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch09-workshop-evaluation-scorecard-map.png",
        "size": "1024x1792",
        "quality": "high",
        "title": "第九章 Agent 工作坊评估计分卡图",
        "suggested_page": "docs/ch09-agent/ch10-projects/04-stage-hands-on-workshop.md",
        "alt": "Agent 工作坊评估计分卡图：EVAL_CASES 固定运行 safe plan、approval block、no evidence 三类任务，并汇总 PASS、FAIL 与 passed 计数。",
        "prompt": """
一张适合第九章 Agent 实操工作坊的竖版评估流程图，主题是“Agent 不能只展示成功 Demo，要用固定任务集评估”。
画面用三张测试卡片展示 safe_learning_plan、publish_without_approval、unknown_topic；每张卡片进入 run_agent，比较 expected_status 与 actual status。
底部汇总 PASS、FAIL、passed/total，并提示检查 completion、permission、no_evidence 三类行为。
风格像测试看板、成绩单和执行 trace 图结合，适合新人理解评估。
文字不是主体；标准术语保留英文，例如 EVAL_CASES、run_agent、expected_status、actual status、PASS、FAIL、passed/total、permission、no_evidence。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch09-capability-level-ladder-map.png",
        "size": "1024x1792",
        "quality": "high",
        "title": "第九章 Agent 能力分级阶梯图",
        "suggested_page": "docs/ch09-agent/ch01-agent-basics/03-capability-levels.md",
        "alt": "Agent 能力分级阶梯图：L0 回答、L1 单工具、L2 多步骤工具、L3 目标驱动、L4 长周期高自治，并标出能力越高风险越高。",
        "prompt": """
一张适合第九章 Agent 能力分级小节的竖版阶梯教学图，主题是“不要把所有系统都叫高级 Agent”。
画面用从下到上的 5 层阶梯表现 L0 response only、L1 single tool、L2 multi-step tools、L3 goal-driven、L4 long-running / multi-agent / high autonomy。
每一层旁边画出小图标：回答气泡、单个工具、工具链、目标旗帜、长期任务看板；右侧用风险温度条表现 autonomy 和 risk 逐级上升。
底部加一个新人提示区：choose the smallest sufficient level。
文字不是主体；标准术语保留英文，例如 L0、L1、L2、L3、L4、response only、single tool、multi-step tools、goal-driven、high autonomy、risk。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch09-reasoning-state-checkpoint-map.png",
        "size": "1024x1792",
        "quality": "high",
        "title": "第九章 LLM 推理与中间状态图",
        "suggested_page": "docs/ch09-agent/ch02-reasoning/01-llm-reasoning.md",
        "alt": "LLM 推理与中间状态图：复杂任务需要拆解、保存中间状态、检查约束、逐步得到结论，而不是只凭记忆直接回答。",
        "prompt": """
一张适合第九章 LLM 推理能力小节的竖版流程图，主题是“reasoning 不是直接背答案，而是维护 intermediate state”。
画面上半部分对比 memory lookup 与 reasoning path：左侧直接问答，右侧 complex goal -> decompose -> intermediate state -> constraint check -> next step -> conclusion。
画面下半部分展示新人常见卡点：忘记前一步、跳过约束、工具结果没有写回 state、最后答案无法追溯。
风格像漫画式思考过程和调试检查点结合。
文字不是主体；标准术语保留英文，例如 memory lookup、reasoning path、complex goal、decompose、intermediate state、constraint check、next step、conclusion。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch09-tool-strategy-routing-map.png",
        "size": "1024x1792",
        "quality": "high",
        "title": "第九章 工具调用策略路由图",
        "suggested_page": "docs/ch09-agent/ch03-tools/03-tool-strategies.md",
        "alt": "工具调用策略路由图：先判断是否需要工具，再选择工具、校验参数、执行、验证结果，失败时进入 retry、fallback 或停止。",
        "prompt": """
一张适合第九章工具调用策略小节的竖版决策树图，主题是“有工具不等于会用工具”。
画面表现用户任务进入 routing policy：no tool、single tool、multi-step tools、fallback；每条路径都包含 argument validation、execute、verify result、stop condition。
重点画出失败分支：tool error -> retry with limit；low confidence -> fallback；unsafe action -> permission gate；loop too long -> stop。
风格像任务路由控制台、流程卡片和新人提示便签结合。
文字不是主体；标准术语保留英文，例如 routing policy、no tool、single tool、multi-step tools、fallback、argument validation、verify result、retry limit、permission gate、stop condition。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch09-short-term-memory-window-map.png",
        "size": "1024x1792",
        "quality": "high",
        "title": "第九章 短期记忆窗口与状态图",
        "suggested_page": "docs/ch09-agent/ch04-memory/02-short-term-memory.md",
        "alt": "短期记忆窗口与状态图：当前任务只保留必要上下文、运行状态和摘要，避免把全部历史无限塞进上下文窗口。",
        "prompt": """
一张适合第九章短期记忆小节的竖版教学图，主题是“short-term memory 是当前任务的 working memory”。
画面从上到下展示 user turns、recent conversation window、runtime state、temporary results、summary memory、model context window。
左侧画出错误做法：把 all history 全塞进 context，导致 token overflow、lost focus、high cost；右侧画出正确做法：keep recent turns、store state、summarize old steps、drop noise。
风格像工作台、窗口裁剪和状态白板结合，适合新人理解上下文窗口不是无限的。
文字不是主体；标准术语保留英文，例如 short-term memory、working memory、recent turns、runtime state、temporary results、summary memory、context window、token overflow。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch09-mcp-client-server-message-flow-map.png",
        "size": "1024x1792",
        "quality": "high",
        "title": "第九章 MCP 客户端服务器消息流图",
        "suggested_page": "docs/ch09-agent/ch05-mcp/01-mcp-overview.md",
        "alt": "MCP 客户端服务器消息流图：Agent 通过 MCP client 发现 tools/resources/prompts，再用统一消息格式调用 MCP server。",
        "prompt": """
一张适合第九章 MCP 概述小节的竖版协议流程图，主题是“MCP 把工具集成从胶水代码变成统一 client-server 协议”。
画面表现 Agent App、MCP client、transport、MCP server、tools、resources、prompts。
用分步骤箭头表现 initialize、list tools/resources/prompts、call tool、return result、update Agent state。
旁边对比 no protocol 的混乱接口和 MCP 的统一消息格式。
风格像协议时序图和课程漫画结合，新人友好但工程感明确。
文字不是主体；标准术语保留英文，例如 MCP、client、server、transport、tools、resources、prompts、initialize、list tools、call tool、result。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch09-multi-agent-collaboration-run-map.png",
        "size": "1024x1792",
        "quality": "high",
        "title": "第九章 多 Agent 协作实操运行图",
        "suggested_page": "docs/ch09-agent/ch07-multi-agent/06-multi-agent-practice.md",
        "alt": "多 Agent 协作实操运行图：Planner、Retriever、Writer、Reviewer 分工处理同一个任务，通过共享状态和检查点汇总结果。",
        "prompt": """
一张适合第九章多 Agent 实操小节的竖版协作流程图，主题是“多 Agent 不是多人聊天，而是有角色、状态和交付物的协作系统”。
画面展示一个用户任务进入 shared task board，然后依次或并行分配给 Planner、Retriever、Writer、Reviewer。
每个角色产出明确 artifact：plan、evidence、draft、review notes；最终由 aggregator 形成 final answer。
强调 shared state、handoff、checkpoint、quality gate，避免角色互相覆盖。
风格像项目看板、流水线和漫画协作场景结合。
文字不是主体；标准术语保留英文，例如 Planner、Retriever、Writer、Reviewer、shared state、handoff、checkpoint、quality gate、artifact、aggregator。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch09-agent-benchmark-custom-eval-map.png",
        "size": "1024x1792",
        "quality": "high",
        "title": "第九章 Agent Benchmark 与自定义评估集图",
        "suggested_page": "docs/ch09-agent/ch08-eval-safety/02-benchmarks.md",
        "alt": "Agent Benchmark 与自定义评估集图：通用 benchmark 可比较趋势，但生产项目还需要 normal、boundary、tool failure、safety 四类自定义样本。",
        "prompt": """
一张适合第九章 Agent Benchmark 小节的竖版对比图，主题是“通用 benchmark 不能替代你的项目 eval set”。
画面左侧是 general benchmark：fixed tasks、unified scoring、leaderboard、trend comparison。
画面右侧是 custom project eval set：normal tasks、boundary cases、tool failures、safety / permission cases、business success criteria。
中间用桥梁说明：use benchmark for capability boundary, use custom eval for production readiness。
风格像考试成绩单与产品验收清单对比，新人一眼能明白不要只追 leaderboard。
文字不是主体；标准术语保留英文，例如 general benchmark、custom eval set、fixed tasks、unified scoring、leaderboard、normal tasks、boundary cases、tool failures、safety cases、production readiness。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch09-runtime-management-protection-map.png",
        "size": "1024x1792",
        "quality": "high",
        "title": "第九章 Agent 运行时管理防护图",
        "suggested_page": "docs/ch09-agent/ch09-deployment/02-runtime-management.md",
        "alt": "Agent 运行时管理防护图：并发限制、timeout、retry、circuit breaker 和 metrics 共同保护生产 Agent 的稳定性。",
        "prompt": """
一张适合第九章运行时管理小节的竖版生产防护图，主题是“Agent 上线后最先崩的常常不是答案，而是运行稳定性”。
画面展示 user requests 进入 runtime manager，依次经过 queue、concurrency limit、timeout、retry with backoff、circuit breaker、metrics。
右侧画出危险路径：retry storm、slow dependency、queue buildup、cost spike；左侧画出保护结果：stable latency、bounded cost、degraded but alive。
风格像生产控制台、告警仪表盘和流程式防护墙结合。
文字不是主体；标准术语保留英文，例如 runtime manager、queue、concurrency limit、timeout、retry with backoff、circuit breaker、metrics、retry storm、cost spike。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch09-agent-boundary-workflow-chatbot-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "工作流、聊天机器人与 Agent 边界图",
        "suggested_page": "docs/ch09-agent/ch01-agent-basics/01-what-is-agent.md",
        "alt": "工作流、聊天机器人与 Agent 边界图：固定步骤、对话回复和目标驱动的持续决策三种系统边界对比。",
        "prompt": """
一张适合 AI Agent 入门课程的系统边界图，主题是“workflow、chatbot、Agent 三者不是同一种控制方式”。
画面左侧是固定地铁线路般的 workflow，中间是前台接待式 chatbot，右侧是带目标、状态、工具和反馈循环的 Agent 助理。
风格像新人友好的产品架构对比图，层级清晰、留白充足、科技感但不花哨。
文字不是主体；标准术语保留英文，例如 workflow、chatbot、Agent、goal、state、tool、feedback loop。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch09-agent-action-loop-trace-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Agent 行动闭环与 Trace 图",
        "suggested_page": "docs/ch09-agent/ch01-agent-basics/01-what-is-agent.md",
        "alt": "Agent 行动闭环与 Trace 图：用户目标进入决策器，产生行动，工具返回观察，状态更新并留下可复盘 trace。",
        "prompt": """
一张适合 AI Agent 基础课的行动闭环图，主题是“Agent 的价值在于目标、行动、观察和状态更新的可复盘 trace”。
画面表现 user goal、decision、action、tool call、observation、state update、final answer，同时旁边有一条 trace 时间线记录每轮步骤。
风格像执行回路和调试日志结合，适合新人一眼理解 Agent 不是一次性回答。
文字不是主体；标准术语保留英文，例如 user goal、decision、action、tool call、observation、state update、trace、final answer。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch09-agent-system-architecture-dataflow-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Agent 系统架构数据流图",
        "suggested_page": "docs/ch09-agent/ch01-agent-basics/04-system-architecture.md",
        "alt": "Agent 系统架构数据流图：Planner、Tool Layer、Memory、State、Guardrails、Observability 组成生产 Agent 数据流。",
        "prompt": """
一张适合 Agent 系统架构章节的生产级数据流图，主题是“Agent 不是一个模型，而是一组协作组件”。
画面表现 Planner、Tool Layer、Memory、State Store、Guardrails、Observability、Model Runtime 之间的数据流，用户请求从入口进入，最终输出答案和 trace。
风格像云架构图和课程手绘图结合，清楚展示组件职责。
文字不是主体；标准术语保留英文，例如 Planner、Tool Layer、Memory、State Store、Guardrails、Observability、Model Runtime、trace。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch09-cot-self-check-structure-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "链式推理与自检结构图",
        "suggested_page": "docs/ch09-agent/ch02-reasoning/02-chain-reasoning.md",
        "alt": "链式推理与自检结构图：问题被拆成 facts、subtasks、decision、answer，并经过 self-check 降低错误。",
        "prompt": """
一张适合链式推理课程的小节图，主题是“CoT 的核心不是长文本，而是结构化中间状态和自检”。
画面表现 question 被拆成 facts、subtasks、decision、answer，旁边有 self-check 检查关键变量、步骤顺序和最终结论。
风格像草稿纸、结构化 JSON 卡片和质量检查清单结合，直观而克制。
文字不是主体；标准术语保留英文，例如 CoT、facts、subtasks、decision、answer、self-check、scratchpad。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch09-plan-execute-monitor-replan-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Plan-and-Execute 监控重规划图",
        "suggested_page": "docs/ch09-agent/ch02-reasoning/04-plan-and-execute.md",
        "alt": "Plan-and-Execute 监控重规划图：Planner 制定计划，Executor 执行步骤，Monitor 根据观察决定继续或 replan。",
        "prompt": """
一张适合 Plan-and-Execute 小节的机制图，主题是“长任务需要计划、执行、监控和重规划分层”。
画面表现 Planner 生成 plan list，Executor 逐步执行，Monitor 检查 observation、missing info、tool failure 和 goal change，必要时触发 replan。
风格像项目施工清单和 Agent 控制台结合，突出全局规划与局部执行。
文字不是主体；标准术语保留英文，例如 Planner、Executor、Monitor、plan list、observation、tool failure、goal change、replan。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch09-advanced-planning-dag-critical-path-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "高级规划 DAG、并行与关键路径图",
        "suggested_page": "docs/ch09-agent/ch02-reasoning/05-advanced-planning.md",
        "alt": "高级规划 DAG、并行与关键路径图：任务节点、依赖关系、并行 worker、关键路径和失败重规划组成复杂任务规划。",
        "prompt": """
一张适合 Agent 高级规划章节的 DAG 任务图，主题是“复杂任务不是长清单，而是带依赖、并行和关键路径的任务图”。
画面表现 task nodes、dependencies、parallel workers、critical path、resource limit、failure and local replanning。
风格像工程调度甘特图和 DAG 白板结合，关键路径用醒目线条表示。
文字不是主体；标准术语保留英文，例如 DAG、task nodes、dependencies、parallel workers、critical path、resource limit、replanning。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch09-reasoning-eval-failure-taxonomy-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Agent 推理失败类型诊断图",
        "suggested_page": "docs/ch09-agent/ch02-reasoning/06-reasoning-evaluation.md",
        "alt": "Agent 推理失败类型诊断图：intent、plan、tool、observation、stop condition 和 final answer 六层失败归因。",
        "prompt": """
一张适合 Agent 推理评估章节的失败归因图，主题是“答案错误背后可能是意图、计划、工具、观察、停止条件或最终表达出错”。
画面表现 intent understanding、plan quality、tool choice、observation use、stop condition、final answer 六层诊断漏斗，并有 trace evidence 指向每层。
风格像质量诊断仪表盘和排障流程图结合，适合新人做问题定位。
文字不是主体；标准术语保留英文，例如 intent、plan quality、tool choice、observation、stop condition、final answer、trace evidence。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch09-tool-schema-validation-guardrail-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Function Calling Schema 校验与执行护栏图",
        "suggested_page": "docs/ch09-agent/ch03-tools/01-function-calling-deep.md",
        "alt": "Function Calling Schema 校验与执行护栏图：模型产出 tool call 后，程序做 schema 校验、权限检查、参数清洗、执行和错误恢复。",
        "prompt": """
一张适合 Function Calling 深入小节的工程链路图，主题是“模型提出调用，程序负责安全执行”。
画面表现 model tool call、schema validation、permission check、argument sanitization、dispatcher、tool execution、structured error、retry or fallback。
风格像后端调用链和安全网关结合，突出程序边界。
文字不是主体；标准术语保留英文，例如 tool call、schema validation、permission check、arguments、dispatcher、structured error、retry、fallback。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch09-tool-description-quality-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "工具描述质量与发现流程图",
        "suggested_page": "docs/ch09-agent/ch03-tools/02-tool-description.md",
        "alt": "工具描述质量与发现流程图：name、description、arguments、returns、risk_level 帮助 Agent 发现并选择正确工具。",
        "prompt": """
一张适合 Agent 工具描述章节的工具卡片图，主题是“工具描述越清楚，模型越不容易拿错工具”。
画面表现 tool registry 中的 name、description、when to use、when not to use、required args、returns、risk_level，并展示 user query 到 candidate tools 的发现流程。
风格像工具箱标签和搜索推荐系统结合，清楚、现代、适合课程插图。
文字不是主体；标准术语保留英文，例如 tool registry、name、description、when to use、required args、returns、risk_level、candidate tools。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch09-tool-safety-permission-sandbox-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "工具安全权限、沙箱与审计图",
        "suggested_page": "docs/ch09-agent/ch03-tools/05-tool-safety.md",
        "alt": "工具安全权限、沙箱与审计图：read_only、write_limited、destructive 三类工具经过权限、确认、沙箱和 audit log 防线。",
        "prompt": """
一张适合 Agent 工具安全章节的多层防线图，主题是“工具让 Agent 会行动，也让风险从语言层进入动作层”。
画面表现 read_only、write_limited、destructive 工具风险分级，以及 permission check、sandbox、human approval、timeout、idempotency、audit log。
风格像安全控制台和分层护城河，直观展示低风险到高风险的升级。
文字不是主体；标准术语保留英文，例如 read_only、write_limited、destructive、permission check、sandbox、human approval、idempotency、audit log。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch09-code-agent-sandbox-review-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "代码 Agent 沙箱、测试与 Review 闭环图",
        "suggested_page": "docs/ch09-agent/ch03-tools/07-code-agent.md",
        "alt": "代码 Agent 沙箱、测试与 Review 闭环图：Read、Plan、Patch、Run Tests、Review、Repair、Accept 形成代码执行循环。",
        "prompt": """
一张适合代码 Agent 小节的工程闭环图，主题是“代码 Agent 不是代码补全，而是在真实仓库里读、改、跑、验、再修”。
画面表现 Read context、Plan change、Generate patch、Sandbox run、Tests、Review、Repair、Accept，并用 diff 和 test report 作为中间工件。
风格像 IDE、CI 流水线和 Agent trace 结合，专业但新人友好。
文字不是主体；标准术语保留英文，例如 Read、Plan、Patch、Sandbox、Tests、Review、Repair、Accept、diff、test report。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch09-memory-layer-selection-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Agent 记忆分层选择图",
        "suggested_page": "docs/ch09-agent/ch04-memory/01-memory-overview.md",
        "alt": "Agent 记忆分层选择图：short-term、long-term、episodic、procedural memory 分别保存当前任务、稳定偏好、具体经历和可复用流程。",
        "prompt": """
一张适合 Agent 记忆系统概述的小节图，主题是“不同信息应该进入不同记忆层，而不是全塞进上下文”。
画面表现 short-term memory 像工作台，long-term memory 像档案库，episodic memory 像经历记录，procedural memory 像操作手册，并有信息流选择箭头。
风格像四层资料柜和 Agent 工作台，温和、清晰、适合新人。
文字不是主体；标准术语保留英文，例如 short-term memory、long-term memory、episodic memory、procedural memory、context、profile、workflow。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch09-long-term-memory-write-update-policy-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "长期记忆写入、更新与置信度图",
        "suggested_page": "docs/ch09-agent/ch04-memory/03-long-term-memory.md",
        "alt": "长期记忆写入、更新与置信度图：信息先经过写入判断，再带 confidence 和 version 更新，最后按任务检索。",
        "prompt": """
一张适合 Agent 长期记忆章节的策略流程图，主题是“长期记忆难点不是存，而是判断、更新和冲突处理”。
画面表现 new information、write policy、stability check、confidence score、version update、conflict resolution、task-aware retrieval。
风格像用户档案卡和数据治理流程结合，突出置信度和版本。
文字不是主体；标准术语保留英文，例如 write policy、stability check、confidence、version、conflict resolution、retrieval、user profile。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch09-memory-engineering-lifecycle-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "记忆工程生命周期图",
        "suggested_page": "docs/ch09-agent/ch04-memory/05-memory-engineering.md",
        "alt": "记忆工程生命周期图：write、index、retrieve、cleanup、compress 和 privacy control 组成记忆工程闭环。",
        "prompt": """
一张适合 Agent 记忆工程章节的生命周期图，主题是“记忆系统更像图书馆，不是储物间”。
画面表现 write、index、retrieve、cleanup、compress、privacy control、TTL、importance score，形成长期可维护闭环。
风格像图书馆编目系统和数据管道结合，清楚展示每一段策略。
文字不是主体；标准术语保留英文，例如 write、index、retrieve、cleanup、compress、privacy control、TTL、importance score。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch09-mcp-host-client-server-message-flow-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "MCP 工具发现与调用消息流图",
        "suggested_page": "docs/ch09-agent/ch05-mcp/02-mcp-architecture.md",
        "alt": "MCP 工具发现与调用消息流图：Host、Client、Transport、Server 之间先 tools/list，再 tools/call，返回结构化结果。",
        "prompt": """
一张适合 MCP 架构小节的消息流图，主题是“MCP 把工具发现和工具调用标准化”。
画面表现 Host、Client、Transport、MCP Server、Tools/Resources/Prompts，消息顺序包括 tools/list、tools/call、result response。
风格像协议时序图和架构图结合，清晰展示角色分工。
文字不是主体；标准术语保留英文，例如 Host、Client、Transport、MCP Server、Tools、Resources、Prompts、tools/list、tools/call、result。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch09-mcp-server-tool-contract-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "MCP Server 工具契约图",
        "suggested_page": "docs/ch09-agent/ch05-mcp/03-mcp-server-dev.md",
        "alt": "MCP Server 工具契约图：list_tools、validate args、call_tool、execute logic、normalize result 和 error response 形成服务端契约。",
        "prompt": """
一张适合 MCP Server 开发小节的工具契约图，主题是“Server 是能力边界和工具契约的守门人”。
画面表现 list_tools、tool schema、validate arguments、call_tool、execute logic、normalize result、error response、audit。
风格像 API 契约卡片和服务端执行管线结合，适合工程课程。
文字不是主体；标准术语保留英文，例如 list_tools、tool schema、validate arguments、call_tool、execute logic、normalize result、error response、audit。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch09-langgraph-state-machine-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "LangGraph 状态机与条件边图",
        "suggested_page": "docs/ch09-agent/ch06-frameworks/02-langchain-langgraph.md",
        "alt": "LangGraph 状态机与条件边图：state 在节点间流动，conditional edges 根据状态决定 retrieve、answer、fallback 或 retry。",
        "prompt": """
一张适合 LangChain/LangGraph 小节的状态机图，主题是“当流程有分支、回路和状态时，图式抽象比链式抽象更自然”。
画面表现 shared state、nodes、conditional edges、retrieve、answer、fallback、retry、checkpoint，旁边用一条简单 chain 对比固定步骤。
风格像流程图编辑器和状态机控制台结合，结构清楚。
文字不是主体；标准术语保留英文，例如 shared state、nodes、conditional edges、retrieve、answer、fallback、retry、checkpoint、chain。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch09-framework-selection-decision-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Agent 框架选型决策图",
        "suggested_page": "docs/ch09-agent/ch06-frameworks/08-framework-selection.md",
        "alt": "Agent 框架选型决策图：根据 stateful flow、knowledge/RAG、role collaboration、prototype speed 和 long-term maintenance 选择框架。",
        "prompt": """
一张适合 Agent 框架选型指南的决策树图，主题是“按任务结构选框架，而不是按热度选框架”。
画面表现 stateful flow、knowledge/RAG、role collaboration、prototype speed、long-term maintenance 五个分支，指向 graph workflow、retrieval framework、role framework、lightweight custom。
风格像架构决策图和课程路线图结合，简洁、有判断感。
文字不是主体；标准术语保留英文，例如 stateful flow、knowledge/RAG、role collaboration、prototype speed、long-term maintenance、graph workflow、lightweight custom。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch09-multi-agent-pattern-selection-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "多 Agent 架构模式选择图",
        "suggested_page": "docs/ch09-agent/ch07-multi-agent/01-architecture-patterns.md",
        "alt": "多 Agent 架构模式选择图：supervisor-worker、pipeline、reviewer、peer/group 根据任务分工和控制需求选择。",
        "prompt": """
一张适合多 Agent 架构模式小节的模式选择图，主题是“多 Agent 的重点不是数量，而是分工和组织方式”。
画面表现 supervisor-worker、pipeline、reviewer、peer/group 四种模式，用不同组织结构展示调度、流水线、生成评审和多方讨论。
风格像团队组织图和系统架构图结合，突出适用场景。
文字不是主体；标准术语保留英文，例如 supervisor-worker、pipeline、reviewer、peer/group、task split、coordination、review loop。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch09-multi-agent-communication-contract-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Agent 间通信契约图",
        "suggested_page": "docs/ch09-agent/ch07-multi-agent/02-communication.md",
        "alt": "Agent 间通信契约图：sender、receiver、type、task_id、payload、status、retry 和 shared state 组成可靠通信。",
        "prompt": """
一张适合 Agent 间通信小节的消息契约图，主题是“多 Agent 不能只靠随意对话，必须有可追踪消息结构”。
画面表现 sender、receiver、message type、task_id、payload、status、timestamp、retry_count，同时展示 message passing、shared state、event bus 三种通信方式。
风格像消息队列控制台和团队协作白板结合，清晰有工程感。
文字不是主体；标准术语保留英文，例如 sender、receiver、message type、task_id、payload、status、retry_count、event bus、shared state。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch09-multi-agent-coordination-cost-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "多 Agent 协调、冲突与收敛图",
        "suggested_page": "docs/ch09-agent/ch07-multi-agent/03-task-coordination.md",
        "alt": "多 Agent 协调、冲突与收敛图：任务分配、依赖顺序、共享状态、重复劳动、结论冲突和 reviewer 裁决。",
        "prompt": """
一张适合多 Agent 任务协调小节的协调成本图，主题是“多 Agent 既能分工，也会带来通信、冲突和收敛成本”。
画面表现 task assignment、dependency order、shared state、duplicate work、conflicting conclusions、reviewer arbitration、final convergence。
风格像项目调度看板和分布式系统冲突解决图结合，提醒不要滥用多 Agent。
文字不是主体；标准术语保留英文，例如 task assignment、dependency order、shared state、duplicate work、conflict、reviewer arbitration、convergence。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch09-agent-eval-layered-scorecard-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Agent 分层评估计分卡图",
        "suggested_page": "docs/ch09-agent/ch08-eval-safety/01-evaluation-methods.md",
        "alt": "Agent 分层评估计分卡图：结果层、过程层、工具层、安全层共同衡量任务成功、路径合理、工具正确和边界安全。",
        "prompt": """
一张适合 Agent 评估方法章节的分层 scorecard 图，主题是“评估 Agent 不能只看最终答案”。
画面表现 result layer、process layer、tool layer、safety layer 四层评分卡，包括 task success、step quality、tool accuracy、permission safety、cost。
风格像质量评估仪表盘和课程表格结合，适合新人复制到项目 README。
文字不是主体；标准术语保留英文，例如 scorecard、result layer、process layer、tool layer、safety layer、task success、tool accuracy、cost。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch09-agent-security-prompt-injection-risk-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Prompt Injection 与工具风险隔离图",
        "suggested_page": "docs/ch09-agent/ch08-eval-safety/03-agent-security.md",
        "alt": "Prompt Injection 与工具风险隔离图：不可信外部内容不能覆盖系统指令，高风险动作经过权限、确认、脱敏和审计。",
        "prompt": """
一张适合 Agent 安全章节的风险隔离图，主题是“外部文档可以是资料，但不能变成指令”。
画面表现 untrusted content、prompt injection、system boundary、secret redaction、permission check、human approval、high-risk tool、audit log。
风格像网络安全威胁模型和工具权限图结合，醒目但不恐吓。
文字不是主体；标准术语保留英文，例如 untrusted content、prompt injection、system boundary、secret redaction、permission check、human approval、audit log。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch09-agent-observability-trace-span-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Agent 可观测 Trace Span 图",
        "suggested_page": "docs/ch09-agent/ch08-eval-safety/05-observability.md",
        "alt": "Agent 可观测 Trace Span 图：一个 request_id 串起 planner、retriever、tool call、LLM、safety 和 final answer 多个 span。",
        "prompt": """
一张适合 Agent 可观测性章节的 trace span 图，主题是“没有 request_id 和结构化 trace，Agent 排障只能靠猜”。
画面表现 request_id 贯穿 planner span、retriever span、tool call span、LLM span、safety span、final answer，并有 metrics、logs、replay 汇聚到观测面板。
风格像分布式追踪瀑布图和 AI 控制台结合，清楚展示跨层链路。
文字不是主体；标准术语保留英文，例如 request_id、trace span、planner、retriever、tool call、LLM、safety、metrics、logs、replay。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch09-agent-runtime-state-queue-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Agent 部署架构分层数据流图",
        "suggested_page": "docs/ch09-agent/ch09-deployment/01-deployment-architecture.md",
        "alt": "Agent 部署架构分层数据流图：API Gateway、Orchestrator、Queue、Worker、Tool Executor、State Store、Trace Metrics 组成生产架构。",
        "prompt": """
一张适合 Agent 部署架构小节的生产分层数据流图，主题是“从脚本到服务，需要接入、编排、队列、执行、状态和观测分层”。
画面表现 API Gateway、Agent Orchestrator、Queue、Worker、Model Service、Tool Executor、State Store、Trace/Metrics、Alerts。
风格像云原生架构图和课程图解结合，路径清楚、组件边界明显。
文字不是主体；标准术语保留英文，例如 API Gateway、Agent Orchestrator、Queue、Worker、Model Service、Tool Executor、State Store、Trace/Metrics、Alerts。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch09-agent-persistence-checkpoint-eventlog-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Agent Checkpoint、Event Log 与恢复图",
        "suggested_page": "docs/ch09-agent/ch09-deployment/03-persistence-recovery.md",
        "alt": "Agent Checkpoint、Event Log 与恢复图：任务每步写 checkpoint 和 event log，崩溃后按 checkpoint 恢复并用幂等键避免重复副作用。",
        "prompt": """
一张适合 Agent 持久化与恢复章节的恢复机制图，主题是“长任务要能从失败处继续，而不是从零开始”。
画面表现 task steps、checkpoint、event log、crash、resume、idempotency key、skip duplicate side effect、final state。
风格像自动保存的 IDE、事件流水账和恢复时间线结合，强调可恢复和可复盘。
文字不是主体；标准术语保留英文，例如 checkpoint、event log、crash、resume、idempotency key、side effect、final state。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch09-agent-cost-routing-cache-budget-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Agent 成本路由、缓存与预算控制图",
        "suggested_page": "docs/ch09-agent/ch09-deployment/04-cost-optimization.md",
        "alt": "Agent 成本路由、缓存与预算控制图：模型调用、工具调用、缓存命中、重试、上下文长度和预算上限共同决定链路成本。",
        "prompt": """
一张适合 Agent 成本优化章节的链路成本图，主题是“Agent 成本是任务链路账单，不是单次模型价格”。
画面表现 model routing、small model filter、large model answer、context compression、cache hit、tool call cost、retry budget、cost cap。
风格像成本仪表盘和任务链路图结合，帮助新人看到钱花在哪。
文字不是主体；标准术语保留英文，例如 model routing、small model、large model、context compression、cache hit、tool cost、retry budget、cost cap。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch09-production-readiness-canary-rollback-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Agent 生产 Readiness、灰度与回滚图",
        "suggested_page": "docs/ch09-agent/ch09-deployment/05-production-best-practices.md",
        "alt": "Agent 生产 Readiness、灰度与回滚图：metrics、logs、timeout、rate limit、eval suite、canary、rollback、human override、audit log 组成上线检查。",
        "prompt": """
一张适合 Agent 生产最佳实践章节的上线检查图，主题是“能上线不等于可运维，必须可观测、可灰度、可回滚、可审计”。
画面表现 readiness checklist、metrics、structured logs、timeout、rate limit、eval suite、canary rollout、rollback plan、human override、audit log。
风格像发布控制台和安全 checklist 结合，稳定、专业、清晰。
文字不是主体；标准术语保留英文，例如 readiness checklist、metrics、structured logs、timeout、rate limit、eval suite、canary rollout、rollback plan、human override、audit log。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch09-research-assistant-citation-trace-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "研究助手引用 Trace 图",
        "suggested_page": "docs/ch09-agent/ch10-projects/01-research-assistant.md",
        "alt": "研究助手引用 Trace 图：query、retrieve、select sources、summarize claims、attach citations、verify claims 形成可信研究助手。",
        "prompt": """
一张适合研究助手 Agent 项目的引用追踪图，主题是“可信研究助手的每条 claim 都要回到 source”。
画面表现 query、retrieve documents、select sources、summarize claims、attach citations、verify claims、failure cases，并用线条连接 claim 和 source。
风格像研究工作台、文献卡片和 trace 时间线结合，适合作品集项目展示。
文字不是主体；标准术语保留英文，例如 query、retrieve、sources、claims、citations、verify、failure cases、trace。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch09-data-analysis-agent-notebook-loop-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "数据分析 Agent 可复核工作流图",
        "suggested_page": "docs/ch09-agent/ch10-projects/02-data-analysis-agent.md",
        "alt": "数据分析 Agent 可复核工作流图：load data、profile schema、compute statistics、generate insight、suggest chart、write report 构成分析闭环。",
        "prompt": """
一张适合数据分析 Agent 项目的 notebook 式工作流图，主题是“数据分析 Agent 的价值在于可复核的计算链路”。
画面表现 load data、profile schema、clean data、compute statistics、generate insight、suggest chart、write report、trace table。
风格像数据分析 notebook、图表建议卡片和 Agent 工作流结合，清爽直观。
文字不是主体；标准术语保留英文，例如 load data、profile schema、clean data、compute statistics、insight、suggest chart、write report、trace。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch09-multi-agent-dev-team-delivery-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "多 Agent 开发团队交付闭环图",
        "suggested_page": "docs/ch09-agent/ch10-projects/03-multi-agent-dev-team.md",
        "alt": "多 Agent 开发团队交付闭环图：planner 产出 plan，coder 产出 patch，reviewer 产出 issues，tester 产出 test report，失败后回退修复。",
        "prompt": """
一张适合多 Agent 开发团队项目的交付闭环图，主题是“角色数量不是重点，工件交接和验证闭环才是重点”。
画面表现 planner、TaskPlan、coder、Patch、reviewer、ReviewNote、tester、TestReport、repair loop、accepted delivery。
风格像软件团队看板、CI 检查和多 Agent 消息流结合，强调工件交接。
文字不是主体；标准术语保留英文，例如 planner、TaskPlan、coder、Patch、reviewer、ReviewNote、tester、TestReport、repair loop、accepted delivery。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
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
        "filename": "ch10-visual-task-progression-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "视觉任务输出粒度进阶图",
        "suggested_page": "docs/ch10-computer-vision/index.md",
        "alt": "视觉任务输出粒度进阶图：图像基础、OpenCV、图像分类、目标检测、图像分割、OCR 视频 3D 和视觉综合项目由粗到细连接。",
        "prompt": """
一张适合解释视觉任务由浅入深的输出粒度图，主题是“从整图类别到框，再到像素区域”。
画面从 image basics、OpenCV 到 classification、detection、segmentation，再到 OCR/video/3D 和 CV project，输出越来越精细。
风格像阶梯式任务地图，突出“这是什么、在哪里、边界在哪里”的递进关系。
文字不是主体；标准术语保留英文，例如 OpenCV、classification、detection、segmentation、OCR、video、3D。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch10-study-guide-output-granularity-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "视觉输出粒度学习指南图",
        "suggested_page": "docs/ch10-computer-vision/study-guide.md",
        "alt": "视觉输出粒度学习指南图：图像基础、分类、检测、分割、OCR 视频 3D 按输出粒度逐步升级。",
        "prompt": """
一张适合计算机视觉学习指南的学习地图，主题是“先按输出是什么来分清视觉任务”。
画面表现 image basics、classification: what、detection: where、segmentation: pixel mask、OCR/video/3D，像一条从粗到细的视觉理解路线。
风格像清爽教学卡片和任务粒度仪表盘结合。
文字不是主体；标准术语保留英文，例如 classification、detection、segmentation、pixel mask、OCR、video、3D。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch10-cv-basics-chapter-flow.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "视觉基础章节学习顺序图",
        "suggested_page": "docs/ch10-computer-vision/ch01-cv-basics/00-roadmap.md",
        "alt": "视觉基础章节学习顺序图：图像是什么、OpenCV 读写查看图像、基础图像处理逐步连接。",
        "prompt": """
一张适合视觉基础导读页的章节学习顺序图，主题是“先看懂输入图像，再谈模型”。
画面表现 what is image、pixels/channels、OpenCV read/write/view、basic image processing、model input intuition。
风格像图像数据白板和 OpenCV 工作台结合，帮助新人建立输入直觉。
文字不是主体；标准术语保留英文，例如 pixels、channels、OpenCV、image processing、model input。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch10-classification-chapter-flow.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "图像分类章节学习顺序图",
        "suggested_page": "docs/ch10-computer-vision/ch02-classification/00-roadmap.md",
        "alt": "图像分类章节学习顺序图：数据增强、现代分类架构和训练技巧共同决定整图类别判断效果。",
        "prompt": """
一张适合图像分类导读页的章节学习顺序图，主题是“给整张图输出一个主要类别”。
画面表现 data augmentation、modern CNN/ViT architectures、training tricks、validation、error samples，形成分类训练闭环。
风格像训练实验看板和图片样例墙结合。
文字不是主体；标准术语保留英文，例如 data augmentation、CNN、ViT、training tricks、validation、error samples。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch10-detection-chapter-flow.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "目标检测章节学习顺序图",
        "suggested_page": "docs/ch10-computer-vision/ch03-detection/00-roadmap.md",
        "alt": "目标检测章节学习顺序图：检测任务概览、经典检测器、YOLO 系列和检测项目实战逐步连接。",
        "prompt": """
一张适合目标检测导读页的章节学习顺序图，主题是“图里有什么，还要知道它在哪里”。
画面表现 detection overview、bounding box、IoU、mAP、classic detectors、YOLO、NMS、detection practice。
风格像带框标注的视觉项目地图，突出位置理解和指标。
文字不是主体；标准术语保留英文，例如 bounding box、IoU、mAP、YOLO、NMS、detection practice。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch10-segmentation-chapter-flow.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "图像分割章节学习顺序图",
        "suggested_page": "docs/ch10-computer-vision/ch04-segmentation/00-roadmap.md",
        "alt": "图像分割章节学习顺序图：语义分割、实例分割和分割实战逐步连接，突出像素级 mask 输出。",
        "prompt": """
一张适合图像分割导读页的章节学习顺序图，主题是“从框级定位走向像素级区域理解”。
画面表现 semantic segmentation、instance segmentation、mask、IoU/Dice、boundary errors、segmentation practice。
风格像原图与 mask 叠加的教学图，突出边界和区域。
文字不是主体；标准术语保留英文，例如 semantic segmentation、instance segmentation、mask、IoU、Dice、boundary errors。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch10-advanced-vision-route-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "高级视觉方向选择地图",
        "suggested_page": "docs/ch10-computer-vision/ch05-advanced/00-roadmap.md",
        "alt": "高级视觉方向选择地图：OCR、人脸、视频、3D 视觉和多模态行业项目逐步连接，帮助学习者选择作品方向。",
        "prompt": """
一张适合高级视觉导读页的方向选择地图，主题是“从基础视觉任务走向真实应用方向”。
画面表现 image basics、classification、detection、segmentation 之后分出 OCR、face、video、3D vision，再连接 multimodal/AIGC/industry project。
风格像方向分叉地图，强调每个方向的输入输出和应用边界。
文字不是主体；标准术语保留英文，例如 OCR、face、video、3D vision、multimodal、industry project。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch10-projects-delivery-loop.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "视觉项目交付闭环图",
        "suggested_page": "docs/ch10-computer-vision/ch06-projects/00-roadmap.md",
        "alt": "视觉项目交付闭环图：场景需求、任务定义、数据收集、标注规范、训练验证、指标评估、可视化结果、失败案例分析和项目交付形成闭环。",
        "prompt": """
一张适合计算机视觉综合项目导读页的交付闭环图，主题是“视觉项目不是只放预测图，而是数据、标注、指标和失败案例闭环”。
画面表现 scenario need、task definition、data collection、annotation rules、train/validation、metrics、visualized results、failure cases、delivery。
风格像作品集交付看板和视觉评估仪表盘结合。
文字不是主体；标准术语保留英文，例如 annotation rules、train/validation、metrics、failure cases、delivery、portfolio。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch10-workshop-vision-pipeline-map.png",
        "size": "1024x1792",
        "quality": "high",
        "title": "第十章视觉实操工作坊完整流水线图",
        "suggested_page": "docs/ch10-computer-vision/ch06-projects/03-hands-on-vision-workshop.md",
        "alt": "视觉实操工作坊完整流水线图：合成数据集、预处理、分类、检测框、分割 mask、指标评估、错误样本报告形成可复现闭环。",
        "prompt": """
一张适合第十章计算机视觉实操工作坊的竖版流程图，主题是“从一张图像到可复现视觉项目闭环”。
画面从上到下展示 synthetic dataset、preprocessing、classification、bounding box detection、segmentation mask、metrics、failure report、portfolio evidence。
强调先看图，再运行 Python 脚本，再检查 outputs 和 reports。
风格像课程工作坊路线图和视觉项目看板结合，新手友好但工程感明确。
文字不是主体；标准术语保留英文，例如 synthetic dataset、preprocessing、classification、bounding box、segmentation mask、metrics、failure report、outputs、reports。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch10-workshop-synthetic-dataset-map.png",
        "size": "1024x1792",
        "quality": "high",
        "title": "第十章视觉实操工作坊合成数据集图",
        "suggested_page": "docs/ch10-computer-vision/ch06-projects/03-hands-on-vision-workshop.md",
        "alt": "视觉实操工作坊合成数据集图：脚本自动生成 circle、square、triangle 图像、mask、标签、bbox 和 challenge 样本。",
        "prompt": """
一张适合第十章视觉实操工作坊的数据准备图，主题是“离线合成数据集让新人不用下载数据也能跑完整 CV 流程”。
画面展示 Python script 生成 circle、square、triangle 三类样本，同时生成 image、mask、label、bbox、challenge 字段，并写入 labels.csv。
旁边展示 train split 和 test split，以及 blurred、occluded、small_object、low_contrast 四种故意设计的难例。
风格像数据集工厂、图像样本墙和标注表结合。
文字不是主体；标准术语保留英文，例如 synthetic dataset、circle、square、triangle、mask、label、bbox、labels.csv、train split、test split、challenge。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch10-workshop-metrics-iou-confusion-map.png",
        "size": "1024x1792",
        "quality": "high",
        "title": "第十章视觉实操工作坊指标评估图",
        "suggested_page": "docs/ch10-computer-vision/ch06-projects/03-hands-on-vision-workshop.md",
        "alt": "视觉实操工作坊指标评估图：classification accuracy、confusion matrix、box IoU、mask IoU 和 predictions.csv 共同解释模型表现。",
        "prompt": """
一张适合第十章视觉实操工作坊的评估指标图，主题是“视觉项目不能只看一张成功截图，要同时看分类、检测和分割指标”。
画面表现 prediction rows 进入 metric panel：classification accuracy、confusion matrix、box IoU、mask IoU、confidence、mean metrics。
展示 ground truth box 与 predicted box 计算 IoU，GT mask 与 predicted mask 计算 mask IoU，并写入 metrics.json 和 predictions.csv。
风格像评估仪表盘、表格和重叠区域示意图结合。
文字不是主体；标准术语保留英文，例如 classification accuracy、confusion matrix、box IoU、mask IoU、confidence、metrics.json、predictions.csv。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch10-workshop-failure-debug-map.png",
        "size": "1024x1792",
        "quality": "high",
        "title": "第十章视觉实操工作坊失败样本排查图",
        "suggested_page": "docs/ch10-computer-vision/ch06-projects/03-hands-on-vision-workshop.md",
        "alt": "视觉实操工作坊失败样本排查图：低置信度、低 box IoU、低 mask IoU、遮挡、小目标和低对比度样本进入 failure_cases.md。",
        "prompt": """
一张适合第十章视觉实操工作坊的失败样本排查图，主题是“视觉项目最有价值的证据往往来自失败样本”。
画面展示 failure_cases.md 收集 low confidence、low box IoU、low mask IoU、occluded、small_object、low_contrast、blurred 等样本。
旁边画出排查顺序：inspect original image、check preprocessing、check annotation、compare prediction visualization、choose fix action、rerun regression。
风格像视觉错误样本墙、侦查清单和项目复盘板结合。
文字不是主体；标准术语保留英文，例如 failure_cases.md、low confidence、box IoU、mask IoU、occluded、small_object、low_contrast、preprocessing、annotation、rerun regression。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch10-image-array-shape-channel-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "图像数组 Shape 与通道语义图",
        "suggested_page": "docs/ch10-computer-vision/ch01-cv-basics/01-image-fundamentals.md",
        "alt": "图像数组 Shape 与通道语义图：灰度图二维矩阵、RGB 图三维张量、height width channel、uint8 到 float normalization。",
        "prompt": """
一张适合计算机视觉入门的图像数组解释图，主题是“计算机看到的图像首先是数字网格和通道张量”。
画面表现 grayscale image as 2D matrix、RGB image as H x W x C tensor、channel split、uint8 0-255、float normalization 0-1。
风格像教学白板和像素网格可视化结合，清晰展示 shape 维度含义。
文字不是主体；标准术语保留英文，例如 grayscale、RGB、H x W x C、channel、uint8、normalization。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch10-opencv-bgr-coordinate-crop-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "OpenCV BGR、坐标与裁剪顺序图",
        "suggested_page": "docs/ch10-computer-vision/ch01-cv-basics/02-opencv-basics.md",
        "alt": "OpenCV BGR、坐标与裁剪顺序图：BGR 与 RGB 转换、图像数组先 y 后 x、crop 写作 y1:y2 x1:x2。",
        "prompt": """
一张适合 OpenCV 基础操作章节的坑点说明图，主题是“OpenCV 初学最容易混淆 BGR 和 y/x 裁剪顺序”。
画面表现 BGR vs RGB color order、image coordinate grid、array indexing [y1:y2, x1:x2]、resize、crop、draw rectangle。
风格像 OpenCV 工作台和坐标网格提示卡，直观、工程化。
文字不是主体；标准术语保留英文，例如 OpenCV、BGR、RGB、array indexing、y1:y2、x1:x2、crop、resize。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch10-image-processing-operation-decision-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "经典图像处理操作选择图",
        "suggested_page": "docs/ch10-computer-vision/ch01-cv-basics/03-image-processing.md",
        "alt": "经典图像处理操作选择图：去噪用 blur，高斯滤波，边缘检测，阈值化，形态学开闭运算形成像素规则流水线。",
        "prompt": """
一张适合图像处理技术章节的操作选择图，主题是“经典图像处理是一组目的明确的像素规则”。
画面表现 denoise -> blur/GaussianBlur、find changes -> Canny edges、separate foreground -> threshold、clean shapes -> morphology open/close、contour pipeline。
风格像图像处理实验台，展示同一张简化灰度图经过不同操作后的变化。
文字不是主体；标准术语保留英文，例如 GaussianBlur、Canny、threshold、morphology、open、close、contour pipeline。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch10-augmentation-invariance-risk-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "数据增强不变性与语义风险图",
        "suggested_page": "docs/ch10-computer-vision/ch02-classification/01-data-augmentation.md",
        "alt": "数据增强不变性与语义风险图：合理变换保持标签，过度旋转裁剪可能破坏语义，检测和分割要同步 box mask。",
        "prompt": """
一张适合图像数据增强章节的风险边界图，主题是“增强要模拟合理变化，而不能破坏标签语义”。
画面表现 original image、flip/crop/color jitter、label preserved、too strong rotation/crop、semantic risk、box/mask synchronized for detection and segmentation。
风格像图片增强样例墙和风险仪表盘结合，帮助新人判断合理增强。
文字不是主体；标准术语保留英文，例如 data augmentation、flip、crop、color jitter、label preserved、semantic risk、box、mask、Mixup。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch10-classification-architecture-evolution-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "图像分类架构演进与选择图",
        "suggested_page": "docs/ch10-computer-vision/ch02-classification/02-modern-architectures.md",
        "alt": "图像分类架构演进与选择图：VGG、ResNet、EfficientNet、ConvNeXt 分别解决深度、可训练性、效率和现代卷积路线。",
        "prompt": """
一张适合现代图像分类架构章节的演进图，主题是“架构名字背后是深度、稳定性和效率问题的演进”。
画面表现 VGG: deeper regular stack、ResNet: residual connection、EfficientNet: accuracy/efficiency balance、ConvNeXt: modernized convolution。
风格像模型家族时间线和工程选择图结合，突出为什么而不是只列名字。
文字不是主体；标准术语保留英文，例如 VGG、ResNet、residual connection、EfficientNet、accuracy/efficiency、ConvNeXt、baseline。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch10-classification-training-diagnosis-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "图像分类训练诊断矩阵图",
        "suggested_page": "docs/ch10-computer-vision/ch02-classification/03-training-tricks.md",
        "alt": "图像分类训练诊断矩阵图：数据问题、训练问题和评估问题通过 loss 曲线、混淆矩阵、错误样本和泄漏检查定位。",
        "prompt": """
一张适合图像分类训练技巧章节的诊断矩阵图，主题是“效果不好时先诊断，不要立刻换模型”。
画面表现 data issues、training issues、evaluation issues 三栏，以及 train/val loss curves、confusion matrix、class imbalance、data leakage、error samples。
风格像训练实验仪表盘和排障表结合，清楚指向下一步动作。
文字不是主体；标准术语保留英文，例如 train loss、val loss、confusion matrix、class imbalance、data leakage、error samples、learning rate。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch10-detection-output-iou-error-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "目标检测输出、IoU 与错误类型图",
        "suggested_page": "docs/ch10-computer-vision/ch03-detection/01-detection-overview.md",
        "alt": "目标检测输出、IoU 与错误类型图：class、box、score、IoU、false positive、false negative、localization error 和 duplicate boxes。",
        "prompt": """
一张适合目标检测概述章节的输出拆解图，主题是“检测同时回答是什么、在哪里和有多大把握”。
画面表现 object detection output: class、bounding box、confidence score、IoU overlap with ground truth，并分出 false positive、false negative、localization error、duplicate boxes。
风格像带标注框的街景教学图和评估表结合，适合新人理解检测质量。
文字不是主体；标准术语保留英文，例如 class、bounding box、confidence score、IoU、false positive、false negative、localization error、duplicate boxes。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch10-classic-detectors-shared-feature-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "R-CNN 家族共享特征演进图",
        "suggested_page": "docs/ch10-computer-vision/ch03-detection/02-classic-detectors.md",
        "alt": "R-CNN 家族共享特征演进图：R-CNN 每个候选框单独提特征，Fast R-CNN 共享整图特征，Faster R-CNN 学习 proposal。",
        "prompt": """
一张适合经典检测架构章节的演进图，主题是“R-CNN 家族是在不断减少重复计算”。
画面表现 R-CNN: proposal crops each run CNN、Fast R-CNN: shared feature map and ROI pooling、Faster R-CNN: Region Proposal Network + shared features。
风格像检测架构对比图，左到右展示速度和共享计算逐步增强。
文字不是主体；标准术语保留英文，例如 R-CNN、Fast R-CNN、Faster R-CNN、proposal、shared feature map、ROI pooling、RPN。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch10-yolo-threshold-nms-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "YOLO 候选框、阈值与 NMS 图",
        "suggested_page": "docs/ch10-computer-vision/ch03-detection/03-yolo-series.md",
        "alt": "YOLO 候选框、阈值与 NMS 图：模型输出多个候选框，经 score threshold 和 NMS 去重后得到最终检测框。",
        "prompt": """
一张适合 YOLO 系列章节的后处理流程图，主题是“YOLO 输出候选框，后处理决定最终画哪些框”。
画面表现 raw predictions with many boxes、score threshold filtering、IoU overlap、NMS suppression、final detections。
风格像实时检测控制台，展示同一目标周围多个重叠框被筛选。
文字不是主体；标准术语保留英文，例如 YOLO、raw predictions、score threshold、IoU、NMS、suppression、final detections。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch10-detection-practice-eval-buckets-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "检测项目评估与误检漏检分桶图",
        "suggested_page": "docs/ch10-computer-vision/ch03-detection/04-detection-practice.md",
        "alt": "检测项目评估与误检漏检分桶图：标注规范、baseline、IoU mAP、false positive、false negative、localization error 和下一步优化。",
        "prompt": """
一张适合检测实战章节的项目评估闭环图，主题是“检测项目先定义什么算对，再分析为什么错”。
画面表现 annotation rules、baseline model、IoU/mAP evaluation、false positive bucket、false negative bucket、localization error bucket、next action data/threshold/model。
风格像视觉项目评估看板和错误样本墙结合，强调可复盘。
文字不是主体；标准术语保留英文，例如 annotation rules、baseline、IoU、mAP、false positive、false negative、localization error、next action。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch10-semantic-segmentation-iou-boundary-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "语义分割 Mask、IoU 与边界误差图",
        "suggested_page": "docs/ch10-computer-vision/ch04-segmentation/01-semantic-segmentation.md",
        "alt": "语义分割 Mask、IoU 与边界误差图：原图、GT mask、prediction mask、per-class IoU、mIoU、boundary error 和 class imbalance。",
        "prompt": """
一张适合语义分割章节的像素级评估图，主题是“分割结果要看区域重叠和边界，不只是彩色 mask 好不好看”。
画面表现 original image、GT mask、prediction mask、per-class IoU、mIoU、boundary error、small class imbalance。
风格像原图/真值/预测三联图和指标卡片结合，突出边界细节。
文字不是主体；标准术语保留英文，例如 semantic segmentation、GT mask、prediction mask、per-class IoU、mIoU、boundary error、class imbalance。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch10-instance-segmentation-count-mask-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "实例分割个体拆分、计数与面积图",
        "suggested_page": "docs/ch10-computer-vision/ch04-segmentation/02-instance-segmentation.md",
        "alt": "实例分割个体拆分、计数与面积图：同类目标被拆成不同 instance id，每个 mask 可用于计数、面积和跟踪。",
        "prompt": """
一张适合实例分割章节的个体拆分图，主题是“实例分割不仅知道类别，还要分清同类里的每个个体”。
画面表现 semantic mask all persons same color vs instance masks with instance id 1/2/3，旁边展示 counting、area per instance、tracking handoff。
风格像分割 mask 教学图和实例统计面板结合，清楚展示相邻目标粘连风险。
文字不是主体；标准术语保留英文，例如 instance segmentation、instance id、mask、counting、area、tracking、semantic mask。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch10-segmentation-practice-failure-buckets-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "分割项目失败样本分桶图",
        "suggested_page": "docs/ch10-computer-vision/ch04-segmentation/03-segmentation-practice.md",
        "alt": "分割项目失败样本分桶图：边界错、小类别漏掉、类别混淆、mask 标注不一致和下一步优化动作。",
        "prompt": """
一张适合分割实战章节的失败样本分桶图，主题是“分割项目不能只看平均 IoU，要逐样本回看失败类型”。
画面表现 original/GT/pred triplets，failure buckets: boundary error、small class missed、class confusion、mask annotation inconsistency，并连接 data/loss/resolution/model next action。
风格像失败样本墙和项目复盘看板结合，务实清晰。
文字不是主体；标准术语保留英文，例如 GT、prediction、boundary error、small class missed、class confusion、mask annotation、IoU、next action。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch10-face-recognition-threshold-pipeline-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "人脸检测、对齐、Embedding 与阈值风险图",
        "suggested_page": "docs/ch10-computer-vision/ch05-advanced/01-face-detection.md",
        "alt": "人脸检测、对齐、Embedding 与阈值风险图：detect、align、embedding、cosine similarity、threshold、false accept 和 false reject。",
        "prompt": """
一张适合人脸检测与识别章节的系统流水线图，主题是“人脸识别是一条检测、对齐、表示和阈值决策流水线”。
画面表现 face detection、alignment、embedding vector、cosine similarity、threshold decision、false accept、false reject、privacy boundary。
风格像机场身份核验流程和向量相似度仪表盘结合，突出系统风险。
文字不是主体；标准术语保留英文，例如 face detection、alignment、embedding、cosine similarity、threshold、false accept、false reject、privacy。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch10-video-frame-tracking-temporal-window-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "视频分析抽帧、跟踪与时序窗口图",
        "suggested_page": "docs/ch10-computer-vision/ch05-advanced/02-video-analysis.md",
        "alt": "视频分析抽帧、跟踪与时序窗口图：frame sampling、single-frame detection、tracking id、temporal window 和 action/event recognition。",
        "prompt": """
一张适合视频分析章节的时间维度图，主题是“视频分析不只是逐帧看图，还要把目标和事件跨时间串起来”。
画面表现 frame sampling、single-frame detection、tracking id across frames、trajectory、temporal window、action/event recognition。
风格像视频时间轴和目标轨迹可视化结合，突出时间窗口。
文字不是主体；标准术语保留英文，例如 frame sampling、detection、tracking id、trajectory、temporal window、action recognition、event。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch10-ocr-layout-reading-order-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "OCR 检测、识别与阅读顺序恢复图",
        "suggested_page": "docs/ch10-computer-vision/ch05-advanced/03-ocr.md",
        "alt": "OCR 检测、识别与阅读顺序恢复图：text detection、text recognition、layout analysis、reading order、field extraction 和错误分桶。",
        "prompt": """
一张适合 OCR 文字识别章节的版面理解图，主题是“OCR 不只是识字，还要找区域、读文字、恢复结构和顺序”。
画面表现 document image、text detection boxes、text recognition、layout analysis、reading order recovery、field extraction、error buckets。
风格像票据/文档 AI 工作台，突出多栏、表格和字段结构。
文字不是主体；标准术语保留英文，例如 OCR、text detection、text recognition、layout analysis、reading order、field extraction、error buckets。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch10-3d-depth-disparity-pointcloud-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "3D 视觉深度、视差与点云直觉图",
        "suggested_page": "docs/ch10-computer-vision/ch05-advanced/04-3d-vision.md",
        "alt": "3D 视觉深度、视差与点云直觉图：stereo cameras、disparity、depth、pixel to 3D point、point cloud 和 camera geometry。",
        "prompt": """
一张适合 3D 视觉入门章节的空间直觉图，主题是“3D 视觉把平面像素和真实空间距离联系起来”。
画面表现 stereo cameras、disparity large means near、disparity small means far、depth map、pixel to 3D point、point cloud、camera geometry。
风格像相机几何教学图和点云可视化结合，清晰展示空间关系。
文字不是主体；标准术语保留英文，例如 stereo cameras、disparity、depth map、pixel to 3D point、point cloud、camera geometry。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch10-security-detection-alert-dedup-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "安防检测、规则、跟踪与告警去重图",
        "suggested_page": "docs/ch10-computer-vision/ch06-projects/01-security-detection.md",
        "alt": "安防检测、规则、跟踪与告警去重图：video stream、object detection、danger zone rule、track_id、alert dedup、human review 和 failure review。",
        "prompt": """
一张适合智能安防项目章节的系统闭环图，主题是“安防项目交付的是可信告警，不是单张图上的框”。
画面表现 video stream、object detection、danger zone rule、tracking id、alert deduplication、human review、audit log、false alarm/missed alarm review。
风格像监控系统控制台和项目交付闭环结合，强调告警体验。
文字不是主体；标准术语保留英文，例如 video stream、object detection、danger zone、track_id、alert deduplication、human review、audit log、false alarm。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch10-medical-imaging-risk-review-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "医学影像风险指标与人工复核闭环图",
        "suggested_page": "docs/ch10-computer-vision/ch06-projects/02-medical-imaging.md",
        "alt": "医学影像风险指标与人工复核闭环图：task boundary、annotation protocol、prediction mask、sensitivity、false negative rate、human review 和 failure analysis。",
        "prompt": """
一张适合医学影像项目章节的高风险评估闭环图，主题是“医学影像项目要讲清任务边界、风险指标和人工复核”。
画面表现 task boundary、annotation protocol、model prediction mask、Dice/IoU、sensitivity、false negative rate、human review、failure analysis、not replacing clinical judgment。
风格像临床辅助系统流程图和风险评估卡片结合，专业、克制、可信。
文字不是主体；标准术语保留英文，例如 task boundary、annotation protocol、Dice、IoU、sensitivity、false negative rate、human review、failure analysis。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch11-nlp-to-llm-backbone.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "NLP 到大模型技术主线图",
        "suggested_page": "docs/ch11-nlp/index.md",
        "alt": "NLP 到大模型技术主线图：文本预处理、文本表示、词向量和语言模型、文本分类、序列标注、Seq2Seq 注意力、预训练模型和 LLM 理解基础逐步连接。",
        "prompt": """
一张适合解释 NLP 和大模型关系的技术主线图，主题是“NLP 是理解大模型的重要来源”。
画面表现 text preprocessing、text representation、word embedding、language model、classification、sequence labeling、Seq2Seq/Attention、pretrained models、LLM foundation。
风格像技术历史路线和现代模型栈结合，突出概念延续。
文字不是主体；标准术语保留英文，例如 word embedding、language model、Seq2Seq、Attention、BERT、GPT、T5、LLM。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch11-study-guide-text-to-model-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "NLP 文本到模型学习指南图",
        "suggested_page": "docs/ch11-nlp/study-guide.md",
        "alt": "NLP 文本到模型学习指南图：文本预处理、表示、分类、抽取、生成和预训练模型组成文本任务主线。",
        "prompt": """
一张适合 NLP 学习指南的主线图，主题是“先理解文本如何进入模型，再理解任务和预训练”。
画面表现 raw text、preprocessing、representation、classification、extraction、generation、pretrained model，形成一条清楚学习路线。
风格像文本流水线和任务分流图结合，帮助新人不被模型名带乱。
文字不是主体；标准术语保留英文，例如 raw text、preprocessing、representation、classification、extraction、generation、pretrained model。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch11-text-basics-chapter-flow.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "文本基础章节学习顺序图",
        "suggested_page": "docs/ch11-nlp/ch01-text-basics/00-roadmap.md",
        "alt": "文本基础章节学习顺序图：NLP 地图、文本预处理、文本表示逐步连接，帮助学习者理解文本如何变成特征。",
        "prompt": """
一张适合文本基础导读页的章节学习顺序图，主题是“先把原始字符串变成可计算特征”。
画面表现 NLP map、text preprocessing、normalization、Token、BoW/TF-IDF、model input。
风格像文本清洗流水线和特征工程地图结合。
文字不是主体；标准术语保留英文，例如 NLP map、text preprocessing、normalization、Token、BoW、TF-IDF、model input。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch11-embeddings-chapter-flow.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "NLP 表示学习章节学习顺序图",
        "suggested_page": "docs/ch11-nlp/ch02-embeddings/00-roadmap.md",
        "alt": "NLP 表示学习章节学习顺序图：文本表示、词向量、上下文表示和语言模型逐步连接。",
        "prompt": """
一张适合 NLP 表示学习导读页的章节学习顺序图，主题是“从词频特征走向上下文语义表示”。
画面表现 text representation、word embedding、semantic space、contextual embedding、language model、LLM foundation。
风格像向量空间和文本上下文窗口结合，清楚展示表示升级。
文字不是主体；标准术语保留英文，例如 word embedding、semantic space、contextual embedding、language model、LLM。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch11-classification-chapter-flow.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "文本分类章节学习顺序图",
        "suggested_page": "docs/ch11-nlp/ch03-classification/00-roadmap.md",
        "alt": "文本分类章节学习顺序图：传统方法、深度学习方法和分类实战逐步连接。",
        "prompt": """
一张适合文本分类导读页的章节学习顺序图，主题是“把文本映射到清晰标签”。
画面表现 label definition、traditional methods、deep learning methods、train/evaluate、confusion matrix、error analysis。
风格像文本分类实验看板，突出标签边界和错例。
文字不是主体；标准术语保留英文，例如 label definition、traditional methods、deep learning、confusion matrix、error analysis。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch11-sequence-labeling-chapter-flow.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "序列标注章节学习顺序图",
        "suggested_page": "docs/ch11-nlp/ch04-sequence-labeling/00-roadmap.md",
        "alt": "序列标注章节学习顺序图：HMM 历史背景、NER、BiLSTM-CRF 和序列标注实践逐步连接。",
        "prompt": """
一张适合序列标注导读页的章节学习顺序图，主题是“不是给整句打标签，而是给每个 token 打标签”。
画面表现 HMM history、NER、BIO tags、BiLSTM-CRF、token-level labels、entity recovery、practice。
风格像句子标注带和实体抽取流程图结合。
文字不是主体；标准术语保留英文，例如 HMM、NER、BIO tags、BiLSTM-CRF、token-level labels、entity recovery。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch11-hmm-crf-sequence-history-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "HMM CRF 序列标注历史地图",
        "suggested_page": "docs/ch11-nlp/ch04-sequence-labeling/04-hmm-crf-history.md",
        "alt": "HMM CRF 序列标注历史地图：HMM、Viterbi、CRF、BiLSTM-CRF 和 BERT token classification 逐步连接。",
        "prompt": """
一张适合序列标注历史小节的简洁教学地图，主题是“HMM 到 BERT 的序列标注演进”。
画面只保留一条横向路线：HMM -> CRF -> BiLSTM-CRF -> BERT token classification；下方用一条短句子展示 BIO tags。
风格像清晰课程时间线和句子标签带结合，少元素、大图形、不要复杂背景。
文字不是主体；中文短标签为主，标准术语保留英文，例如 HMM、CRF、BiLSTM-CRF、BERT、BIO tags。不要整段英文说明、乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch11-seq2seq-chapter-flow.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Seq2Seq 与注意力章节学习顺序图",
        "suggested_page": "docs/ch11-nlp/ch05-seq2seq/00-roadmap.md",
        "alt": "Seq2Seq 与注意力章节学习顺序图：Encoder-Decoder、Attention、机器翻译和生成任务逐步连接。",
        "prompt": """
一张适合 Seq2Seq 导读页的章节学习顺序图，主题是“从输入序列生成输出序列”。
画面表现 Encoder-Decoder、context vector、Attention alignment、machine translation、summarization/generation。
风格像双序列对齐图和生成流水线结合，突出 Attention 解决信息瓶颈。
文字不是主体；标准术语保留英文，例如 Encoder-Decoder、context vector、Attention alignment、machine translation、generation。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch11-ctc-deep-speech-asr-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "CTC Deep Speech 语音识别对齐图",
        "suggested_page": "docs/ch11-nlp/ch05-seq2seq/04-ctc-deep-speech.md",
        "alt": "CTC Deep Speech 语音识别对齐图：长音频帧经过模型输出带 blank 和重复的路径，再折叠成最终文字。",
        "prompt": """
一张适合 CTC 与 Deep Speech 教学页的简洁序列对齐图，主题是“长音频怎样折叠成短文本”。
画面表现 audio waveform -> acoustic model -> CTC path with blank -> collapse -> transcript。
风格像清晰时间轴和折叠流程图，少元素、大箭头、帮助新人直观看懂 alignment。
文字不是主体；中文写概念提示，标准术语保留英文，例如 CTC、blank、collapse、audio frames、transcript、Deep Speech、ASR。不要整段英文说明、乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch11-pretrained-chapter-flow.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "预训练语言模型章节学习顺序图",
        "suggested_page": "docs/ch11-nlp/ch06-pretrained/00-roadmap.md",
        "alt": "预训练语言模型章节学习顺序图：预训练范式、BERT、GPT、T5 和 Transformers 库逐步连接。",
        "prompt": """
一张适合预训练语言模型导读页的章节学习顺序图，主题是“把通用语言能力迁移到具体任务”。
画面表现 pretraining paradigm、BERT、GPT、T5、Transformers library、fine-tuning/inference、task adaptation。
风格像模型家族地图和工具链路线图结合。
文字不是主体；标准术语保留英文，例如 pretraining、BERT、GPT、T5、Transformers、fine-tuning、inference、task adaptation。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch11-projects-delivery-loop.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "NLP 项目交付闭环图",
        "suggested_page": "docs/ch11-nlp/ch07-projects/00-roadmap.md",
        "alt": "NLP 项目交付闭环图：任务定义、标签或 schema、数据样例、模型或 Prompt、评估指标、错误文本、事实检查和项目 README 形成闭环。",
        "prompt": """
一张适合 NLP 综合项目导读页的项目交付闭环图，主题是“文本项目要讲清标签、字段、事实和评估”。
画面表现 task definition、label/schema、data samples、model or Prompt、metrics、error texts、fact check、README/portfolio。
风格像文本项目看板和评估闭环结合，强调任务边界。
文字不是主体；标准术语保留英文，例如 label、schema、Prompt、metrics、error texts、fact check、README、portfolio。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch11-workshop-text-to-artifacts-pipeline-map.png",
        "size": "1024x1792",
        "quality": "high",
        "title": "第十一章 NLP 实操工作坊端到端产物流水线图",
        "suggested_page": "docs/ch11-nlp/ch07-projects/05-hands-on-nlp-workshop.md",
        "alt": "NLP 实操工作坊端到端产物流水线图：原始文本、token、TF-IDF、分类、检索问答、摘要、抽取、指标和失败报告形成可复现闭环。",
        "prompt": """
一张适合第十一章自然语言处理实操工作坊的竖版流程图，主题是“从 raw text 到可复现 NLP 项目证据”。
画面从上到下展示 raw text、tokenization、TF-IDF vectors、classification、retrieval QA、summarization、information extraction、metrics、failure_cases.md、README。
强调先看图，再运行 Python 脚本，再检查 outputs 和 reports。
风格像文本工程流水线、课程实操路线图和项目证据看板结合，新手友好但工程感明确。
文字不是主体；标准术语保留英文，例如 raw text、tokenization、TF-IDF、classification、retrieval QA、summarization、information extraction、metrics、failure_cases.md、README。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch11-workshop-tfidf-classification-map.png",
        "size": "1024x1792",
        "quality": "high",
        "title": "第十一章 NLP 实操工作坊 TF-IDF 分类图",
        "suggested_page": "docs/ch11-nlp/ch07-projects/05-hands-on-nlp-workshop.md",
        "alt": "NLP 实操工作坊 TF-IDF 分类图：tokenize、IDF 权重、文本向量、标签质心、相似度、margin 和混淆矩阵解释分类 baseline。",
        "prompt": """
一张适合第十一章 NLP 实操工作坊的 TF-IDF 文本分类图，主题是“先用透明 baseline 看懂文本如何变成标签”。
画面展示 sentence 进入 tokenize，变成 tokens，再进入 TF-IDF weighting，形成 vector；多个训练样本形成 label centroid，测试文本通过 cosine similarity 选择 predicted label。
旁边展示 score、margin、confusion matrix、classification_predictions.csv。
风格像可视化实验台和文本分类仪表盘结合，适合新人理解代码执行顺序。
文字不是主体；标准术语保留英文，例如 tokenize、tokens、TF-IDF、vector、label centroid、cosine similarity、score、margin、confusion matrix、classification_predictions.csv。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch11-workshop-retrieval-summary-extraction-map.png",
        "size": "1024x1792",
        "quality": "high",
        "title": "第十一章 NLP 实操工作坊检索摘要抽取图",
        "suggested_page": "docs/ch11-nlp/ch07-projects/05-hands-on-nlp-workshop.md",
        "alt": "NLP 实操工作坊检索摘要抽取图：notes.jsonl 支持 retrieval QA，source text 支持 extractive summary，study logs 支持 schema extraction。",
        "prompt": """
一张适合第十一章 NLP 实操工作坊的三任务流程图，主题是“同一批文本能力如何分成 retrieval QA、summarization、information extraction 三种输出”。
画面分成三条竖向支线：notes.jsonl -> retrieve evidence -> answer/refuse；source text -> choose source sentences -> summary_outputs.md；study log -> regex/schema -> extraction_predictions.jsonl。
底部汇总到 metrics 和 failure_cases.md。
风格像三通道文本处理流水线、对比式教学图和项目交付看板结合。
文字不是主体；标准术语保留英文，例如 notes.jsonl、retrieve evidence、answer/refuse、source sentences、summary_outputs.md、regex、schema、extraction_predictions.jsonl、metrics、failure_cases.md。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch11-workshop-failure-debug-map.png",
        "size": "1024x1792",
        "quality": "high",
        "title": "第十一章 NLP 实操工作坊失败样本排查图",
        "suggested_page": "docs/ch11-nlp/ch07-projects/05-hands-on-nlp-workshop.md",
        "alt": "NLP 实操工作坊失败样本排查图：低 margin、错标签、弱证据、字段不匹配和无依据回答进入 failure_cases.md。",
        "prompt": """
一张适合第十一章 NLP 实操工作坊的失败样本排查图，主题是“文本项目最有价值的证据来自边界样本和失败样本”。
画面展示 failure_cases.md 收集 low margin、wrong label、weak evidence、unsupported answer、field mismatch、over cleaning、unclear schema 等问题。
旁边画出排查顺序：inspect raw text、check tokens、check label guide、check source evidence、check schema、adjust threshold、rerun regression。
风格像文本错误样本墙、排查清单和项目复盘板结合。
文字不是主体；标准术语保留英文，例如 failure_cases.md、low margin、wrong label、weak evidence、unsupported answer、field mismatch、raw text、tokens、label guide、schema、threshold、rerun regression。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch11-amr-semantic-graph-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "AMR 语义图理解地图",
        "suggested_page": "docs/ch11-nlp/ch07-projects/04-semantic-graphs-amr.md",
        "alt": "AMR 语义图理解地图：句子被解析成事件、角色、实体和关系图，连接信息抽取、知识图谱、RAG 和课件生成。",
        "prompt": """
一张适合语义图与 AMR 教学页的概念图，主题是“把句子变成结构化含义图”。
画面表现一句话进入 semantic parser，输出一个简洁 AMR-style graph：event、ARG0、ARG1、entity；再连接 information extraction、knowledge graph、RAG。
风格像文本到图结构的教学流程，结构清楚、少元素、有教育科技感。
文字不是主体；中文写概念提示，标准术语保留英文，例如 AMR、semantic graph、event、ARG0、ARG1、knowledge graph、RAG。不要整段英文说明、乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch11-nlp-task-landscape-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "NLP 任务全景图",
        "suggested_page": "docs/ch11-nlp/ch01-text-basics/01-nlp-overview.md",
        "alt": "NLP 任务全景图：raw text 经过 preprocessing、representation 后分流到 classification、extraction、generation、retrieval QA 和 evaluation。",
        "prompt": """
一张适合 NLP 概述章节的任务全景教学图，主题是“文本进入模型后会走向不同任务”。
画面表现 raw text 进入 preprocessing、tokenization、representation，再分流到 text classification、sequence labeling、information extraction、machine translation、summarization、retrieval QA，最后进入 evaluation 和 error analysis。
风格像课程总览地图和文本处理流水线结合，层次清楚、适合新人第一眼理解 NLP 范围。
文字不是主体；标准术语保留英文，例如 raw text、tokenization、representation、classification、extraction、generation、retrieval QA、evaluation。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch11-language-model-next-token-stack.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "语言模型 next token 预测图",
        "suggested_page": "docs/ch11-nlp/ch02-embeddings/03-language-models.md",
        "alt": "语言模型 next token 预测图：context tokens 进入 language model，输出 next token probability distribution，再通过 sampling 或 greedy decoding 生成下一个 token。",
        "prompt": """
一张适合语言模型基础章节的概率预测图，主题是“给定上下文，预测下一个 token”。
画面表现 context tokens、language model、next token probability distribution、top candidates、sampling/greedy decoding、generated token，并在旁边用很小区域对比 n-gram memory 和 neural generalization。
风格像概率分布仪表盘和文本生成流程图结合，强调从简单 next token 目标走向生成能力。
文字不是主体；标准术语保留英文，例如 context、next token、probability distribution、sampling、greedy decoding、n-gram、neural language model。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch11-traditional-classification-baseline-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "传统文本分类基线图",
        "suggested_page": "docs/ch11-nlp/ch03-classification/01-traditional-methods.md",
        "alt": "传统文本分类基线图：text cleaning、BoW/TF-IDF、linear classifier、prediction、confusion matrix 和 error analysis 组成快速可解释基线。",
        "prompt": """
一张适合传统文本分类章节的基线流程图，主题是“传统方法是快速、便宜、可解释的第一条文本分类基线”。
画面表现 text cleaning、BoW、TF-IDF、linear classifier、label probabilities、confusion matrix、error analysis、baseline report。
风格像实验工作台和分类看板结合，突出可解释、快速迭代和小数据场景。
文字不是主体；标准术语保留英文，例如 BoW、TF-IDF、linear classifier、baseline、confusion matrix、error analysis。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch11-neural-classification-embedding-pooling-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "神经文本分类结构图",
        "suggested_page": "docs/ch11-nlp/ch03-classification/02-deep-learning-methods.md",
        "alt": "神经文本分类结构图：token ids 进入 embedding，经过 pooling 得到 sentence vector，再由 classification head 输出 label probabilities。",
        "prompt": """
一张适合深度学习文本分类章节的结构图，主题是“从 token id 到句子向量，再到分类概率”。
画面表现 token ids、embedding matrix、token vectors、pooling、sentence vector、classification head、label probabilities，并用旁注显示 CNN/RNN/Transformer 都可以替换中间编码器。
风格像神经网络剖面图和文本序列流动图结合，适合新人理解 embedding 与 pooling 的作用。
文字不是主体；标准术语保留英文，例如 token ids、embedding、pooling、sentence vector、classification head、CNN、RNN、Transformer。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch11-bilstm-crf-label-path-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "BiLSTM CRF 标签路径解码图",
        "suggested_page": "docs/ch11-nlp/ch04-sequence-labeling/02-bilstm-crf.md",
        "alt": "BiLSTM CRF 标签路径解码图：tokens 经过 BiLSTM 得到 emission scores，CRF 加入 transition scores，Viterbi 选出最优 BIO path。",
        "prompt": """
一张适合 BiLSTM + CRF 章节的序列标注解码图，主题是“上下文表示 + 标签转移约束共同决定 BIO 路径”。
画面表现 tokens、embedding、BiLSTM、emission scores、transition scores、CRF layer、Viterbi decoding、best BIO path，并展示一个不合法路径被压低分数。
风格像句子标签带和路径搜索图结合，突出标签连续性和实体边界。
文字不是主体；标准术语保留英文，例如 BiLSTM、CRF、emission scores、transition scores、Viterbi、BIO path。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch11-ner-project-entity-eval-loop.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "NER 项目实体评估闭环图",
        "suggested_page": "docs/ch11-nlp/ch04-sequence-labeling/03-ner-practice.md",
        "alt": "NER 项目实体评估闭环图：label schema、annotation examples、BIO tags、entity recovery、entity-level precision recall F1 和 error buckets 组成闭环。",
        "prompt": """
一张适合 NER 实战章节的项目闭环图，主题是“NER 项目的关键是实体级评估和错例分桶”。
画面表现 label schema、annotation examples、BIO tags、model prediction、entity recovery、entity-level precision/recall/F1、boundary error、type error、missed entity、iteration。
风格像标注平台和项目评估看板结合，突出实体边界和类型。
文字不是主体；标准术语保留英文，例如 label schema、BIO tags、entity recovery、precision、recall、F1、boundary error、type error。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch11-seq2seq-encoder-decoder-bottleneck-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Seq2Seq 编码器解码器瓶颈图",
        "suggested_page": "docs/ch11-nlp/ch05-seq2seq/01-encoder-decoder.md",
        "alt": "Seq2Seq 编码器解码器瓶颈图：input sequence 进入 encoder 压缩成 context vector，decoder 逐步生成 output sequence，并显示 information bottleneck。",
        "prompt": """
一张适合 Seq2Seq 模型章节的教学图，主题是“输入序列被编码，再由解码器一步步生成输出序列”。
画面表现 input sequence、encoder、context vector、information bottleneck、decoder、output sequence、teacher forcing，并用一条支线提示 Attention 后续会缓解瓶颈。
风格像双语序列流动图和模型剖面图结合，简洁、有箭头、有生成时间步。
文字不是主体；标准术语保留英文，例如 Encoder、Decoder、context vector、information bottleneck、teacher forcing、Attention。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch11-machine-translation-error-analysis-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "机器翻译错误分析图",
        "suggested_page": "docs/ch11-nlp/ch05-seq2seq/03-machine-translation.md",
        "alt": "机器翻译错误分析图：parallel corpus、baseline output、omission、mistranslation、word order、terminology consistency 和 human evaluation。",
        "prompt": """
一张适合机器翻译实战章节的错误分析图，主题是“翻译项目要用错误类型驱动改进，而不是只看漂亮样例”。
画面表现 parallel corpus、baseline translation、reference translation、omission、mistranslation、word order issue、terminology consistency、BLEU/chrF、human evaluation、improvement plan。
风格像翻译对照表和质量评估仪表盘结合，突出错例分桶。
文字不是主体；标准术语保留英文，例如 parallel corpus、baseline、reference、omission、mistranslation、word order、BLEU、human evaluation。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch11-pretraining-transfer-finetune-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "预训练迁移微调关系图",
        "suggested_page": "docs/ch11-nlp/ch06-pretrained/01-pretrain-paradigm.md",
        "alt": "预训练迁移微调关系图：large unlabeled corpus 训练 foundation model，再迁移到 classification、NER、QA、summarization 等下游任务并 fine-tune 或 prompting。",
        "prompt": """
一张适合预训练范式章节的关系图，主题是“通用语言底座如何迁移到具体 NLP 任务”。
画面表现 large unlabeled corpus、self-supervised pretraining、foundation model、transfer learning、fine-tuning、prompting、downstream tasks，包括 classification、NER、QA、summarization。
风格像模型工厂和任务分发中心结合，突出共享底座和少量任务数据。
文字不是主体；标准术语保留英文，例如 pretraining、self-supervised、foundation model、transfer learning、fine-tuning、prompting、downstream tasks。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch11-t5-text-to-text-task-unification-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "T5 text-to-text 任务统一图",
        "suggested_page": "docs/ch11-nlp/ch06-pretrained/04-t5.md",
        "alt": "T5 text-to-text 任务统一图：classification、translation、summarization 和 QA 都被改写成 task prefix + input text -> output text。",
        "prompt": """
一张适合 T5 章节的任务统一图，主题是“把不同 NLP 任务都改写成 text-to-text”。
画面表现多个任务入口：classification、translation、summarization、QA，每个入口都变成 task prefix + input text，进入 T5 encoder-decoder，输出 output text。
风格像统一接口转换器和任务卡片矩阵，强调 task prefix 和同一套输入输出格式。
文字不是主体；标准术语保留英文，例如 T5、text-to-text、task prefix、input text、output text、classification、translation、summarization、QA。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch11-transformers-library-call-chain-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Transformers 库调用链图",
        "suggested_page": "docs/ch11-nlp/ch06-pretrained/05-transformers-library.md",
        "alt": "Transformers 库调用链图：Tokenizer、Config、Model、Task Head、Pipeline 和 outputs 组成从文本到预测结果的调用链。",
        "prompt": """
一张适合 Transformers 库实战章节的 API 调用链图，主题是“先分清对象职责，再使用具体模型类”。
画面表现 raw text、Tokenizer、input_ids/attention_mask、Config、Model backbone、Task Head、Pipeline、outputs，并用旁注展示 AutoTokenizer、AutoModel、AutoModelForSequenceClassification 的关系。
风格像工程调用链和模型模块剖面图结合，适合新人解除 API 混乱。
文字不是主体；标准术语保留英文，例如 Tokenizer、Config、Model、Task Head、Pipeline、AutoTokenizer、AutoModel、input_ids、attention_mask。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch11-qa-retrieval-answer-evaluation-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "问答系统检索回答评估闭环图",
        "suggested_page": "docs/ch11-nlp/ch07-projects/01-qa-system.md",
        "alt": "问答系统检索回答评估闭环图：query、retrieval、evidence、answer、refusal、evaluation set 和 error log 组成 QA 项目闭环。",
        "prompt": """
一张适合智能问答系统项目章节的系统闭环图，主题是“问答系统要先找到可靠依据，再生成或返回答案”。
画面表现 user query、retriever、knowledge base、top-k evidence、answer generation、refusal when no evidence、evaluation set、error log、iteration。
风格像 RAG/QA 系统架构图和项目评估看板结合，突出证据和拒答。
文字不是主体；标准术语保留英文，例如 query、retrieval、knowledge base、top-k evidence、answer、refusal、evaluation set、error log。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch11-summarization-extractive-generative-eval-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "文本摘要抽取生成评估图",
        "suggested_page": "docs/ch11-nlp/ch07-projects/02-text-summarization.md",
        "alt": "文本摘要抽取生成评估图：source document 分别走 extractive summary 和 generative summary，并用 coverage、faithfulness、length control、human review 评估。",
        "prompt": """
一张适合文本摘要系统项目章节的对比评估图，主题是“摘要项目要同时看压缩、覆盖和事实一致性”。
画面表现 source document、extractive summary、generative summary、coverage、faithfulness、length control、readability、human review、failure cases。
风格像文章压缩流程和人工评估表结合，突出事实不丢失。
文字不是主体；标准术语保留英文，例如 source document、extractive summary、generative summary、coverage、faithfulness、length control、human review。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch11-information-extraction-schema-pipeline-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "信息抽取 schema 流程图",
        "suggested_page": "docs/ch11-nlp/ch07-projects/03-information-extraction.md",
        "alt": "信息抽取 schema 流程图：raw text、schema design、rules、NER、relation extraction、JSON output、human review 和 downstream table。",
        "prompt": """
一张适合信息抽取项目章节的 schema 流程图，主题是“先定义字段和关系，再把文本稳定转成结构化数据”。
画面表现 raw text、schema design、rules/regex、NER、relation extraction、JSON output、human review、downstream table/RAG index。
风格像文档解析流水线和结构化数据表结合，强调字段边界、关系和复核。
文字不是主体；标准术语保留英文，例如 schema、rules、regex、NER、relation extraction、JSON output、human review、RAG index。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch12-multimodal-system-backbone.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "多模态系统主干图",
        "suggested_page": "docs/ch12-multimodal/index.md",
        "alt": "多模态系统主干图：文本、图像、语音和视频进入多模态系统，完成理解、生成、编辑和工作流自动化。",
        "prompt": """
一张适合解释多模态为什么重要的系统主干图，主题是“不同模态进入同一条理解和生成流程”。
画面表现 text、image、audio、video 汇入 multimodal system，再分向 understanding、generation、editing、workflow automation。
风格像多输入多输出的产品架构图，突出统一表示和工作流。
文字不是主体；标准术语保留英文，例如 text、image、audio、video、multimodal system、understanding、generation、workflow automation。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch12-study-guide-modal-workflow-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "多模态学习指南工作流图",
        "suggested_page": "docs/ch12-multimodal/study-guide.md",
        "alt": "多模态学习指南工作流图：不同模态编码为统一表示，模型完成理解或生成，再进入编辑、审核和产品工作流。",
        "prompt": """
一张适合多模态学习指南的系统学习图，主题是“不要追所有 Demo，先看清多模态工作流”。
画面表现 text/image/audio/video、unified representation、understanding、generation、editing、review、product workflow。
风格像产品流程图和创作控制台结合，帮助新人抓住主线。
文字不是主体；标准术语保留英文，例如 unified representation、understanding、generation、editing、review、product workflow。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch12-multimodal-chapter-flow.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "多模态基础章节学习顺序图",
        "suggested_page": "docs/ch12-multimodal/ch01-multimodal/00-roadmap.md",
        "alt": "多模态基础章节学习顺序图：多模态基础、视觉语言模型和多模态应用逐步连接。",
        "prompt": """
一张适合多模态基础导读页的章节学习顺序图，主题是“从图文对齐到视觉语言应用”。
画面表现 multimodal basics、vision-language models、alignment/fusion、image QA、document understanding、multimodal apps。
风格像图文对齐空间和应用地图结合。
文字不是主体；标准术语保留英文，例如 multimodal basics、vision-language models、alignment、fusion、image QA、document understanding。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch12-image-gen-chapter-flow.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "图像生成章节学习顺序图",
        "suggested_page": "docs/ch12-multimodal/ch02-image-gen/00-roadmap.md",
        "alt": "图像生成章节学习顺序图：扩散模型、Stable Diffusion、应用、微调和最新进展逐步连接。",
        "prompt": """
一张适合图像生成导读页的章节学习顺序图，主题是“从 Diffusion 原理到可控生成工作流”。
画面表现 Diffusion、Stable Diffusion、prompt/control、applications、LoRA/DreamBooth、latest progress、workflow review。
风格像生成模型组件图和创意流程图结合，突出可控、版本和审核。
文字不是主体；标准术语保留英文，例如 Diffusion、Stable Diffusion、prompt、ControlNet、LoRA、DreamBooth、workflow。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch12-video-gen-chapter-flow.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "视频语音数字人章节学习顺序图",
        "suggested_page": "docs/ch12-multimodal/ch03-video-gen/00-roadmap.md",
        "alt": "视频语音数字人章节学习顺序图：视频生成、TTS、数字人和时序内容工作流逐步连接。",
        "prompt": """
一张适合视频生成与数字人导读页的章节学习顺序图，主题是“时序内容不是一张图，而是脚本、镜头、声音和审核流水线”。
画面表现 video generation、storyboard、keyframes、TTS、digital human、subtitle、review/export。
风格像视频制作时间线和生成工作流结合，突出连续性和资产管理。
文字不是主体；标准术语保留英文，例如 video generation、storyboard、keyframes、TTS、digital human、subtitle、review、export。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch12-frontier-ethics-route-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "AIGC 前沿伦理合规路线图",
        "suggested_page": "docs/ch12-multimodal/ch04-frontier/00-roadmap.md",
        "alt": "AIGC 前沿伦理合规路线图：前沿趋势、AI 伦理、AI 监管、版权肖像、偏见虚假内容和使用边界逐步连接。",
        "prompt": """
一张适合 AIGC 前沿与伦理导读页的路线图，主题是“前沿变化很快，但风险边界必须从一开始设计”。
画面表现 frontier trends、AI ethics、AI regulations、copyright、portrait rights、bias、misinformation、usage boundary。
风格像技术雷达和合规检查清单结合，专业、清晰。
文字不是主体；标准术语保留英文，例如 frontier trends、AI ethics、AI regulations、copyright、bias、misinformation、usage boundary。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch12-multimodal-rag-agent-bridge.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "多模态 RAG 与 Agent 桥接图",
        "suggested_page": "docs/ch12-multimodal/index.md",
        "alt": "多模态 RAG 与 Agent 桥接图：截图或 PDF 经过视觉文档解析、结构化文本和图像片段、检索引用、多模态回答、人工编辑、内容审核和导出交付。",
        "prompt": """
一张适合多模态主线的系统桥接图，主题是“多模态能力如何扩展 RAG 和 Agent”。
用 6 个清晰大模块串联：screenshot/PDF、document parsing、retrieval、multimodal answer、Agent action、review/export。
风格像端到端系统架构图，背景干净，连线清楚，模块之间层次分明。
文字不是主体；标准术语保留英文，例如 screenshot、PDF、document parsing、retrieval、multimodal answer、Agent action、review/export。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch12-projects-delivery-loop.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "AIGC 创意平台项目交付闭环图",
        "suggested_page": "docs/ch12-multimodal/ch05-projects/00-roadmap.md",
        "alt": "AIGC 创意平台项目交付闭环图：主题输入、文案、配图提示词、分镜脚本、语音稿、生成结果、人工编辑、审核清单和导出交付形成闭环。",
        "prompt": """
一张适合 AIGC 综合项目导读页的项目交付闭环图，主题是“生成好看不等于可交付，创意平台需要编辑、审核和导出”。
画面表现 topic input、copywriting、image prompt、storyboard、voice script、generation result、human edit、review checklist、export delivery。
风格像创意产品看板和多模态工作流结合，明亮、专业。
文字不是主体；标准术语保留英文，例如 image prompt、storyboard、voice script、generation result、human edit、review checklist、export。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch12-workshop-creative-package-pipeline-map.png",
        "size": "1024x1792",
        "quality": "high",
        "title": "第十二章多模态创意包实操流水线图",
        "suggested_page": "docs/ch12-multimodal/ch05-projects/02-hands-on-multimodal-workshop.md",
        "alt": "多模态创意包实操流水线图：creative brief、Prompt 版本、SVG 资产、storyboard、asset manifest、safety review、export preview 和 failure cases 形成可复现闭环。",
        "prompt": """
一张适合第十二章 AIGC 与多模态实操工作坊的竖版流程图，主题是“从 creative brief 到可交付多模态内容包”。
画面从上到下展示 creative brief、prompt plan、SVG visual assets、storyboard、timeline、asset manifest、safety review、failure_cases.md、export_preview.html。
强调先看图，再运行 Python 脚本，再检查 prompts、assets、outputs 和 reports。
风格像创意产品流水线、课程实操路线图和资产管理看板结合，新手友好但工程感明确。
文字不是主体；标准术语保留英文，例如 creative brief、prompt plan、SVG assets、storyboard、timeline、asset manifest、safety review、failure_cases.md、export_preview.html。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch12-workshop-prompt-asset-version-map.png",
        "size": "1024x1792",
        "quality": "high",
        "title": "第十二章多模态 Prompt 到资产版本记录图",
        "suggested_page": "docs/ch12-multimodal/ch05-projects/02-hands-on-multimodal-workshop.md",
        "alt": "多模态 Prompt 到资产版本记录图：creative brief 拆成 scene prompt、negative prompt、size、style、asset id、source 和 license。",
        "prompt": """
一张适合第十二章多模态实操工作坊的 Prompt 与资产版本图，主题是“生成结果不是孤立文件，必须能追溯到 Prompt 和版本记录”。
画面展示 creative brief 拆成 scene_01、scene_02、scene_03，每个 scene 有 prompt、negative prompt、size、style、asset id、source、license、review status。
旁边展示 prompt_plan.json、prompt_versions.md、assets/scene_*.svg、asset_manifest.csv 之间的关系。
风格像版本控制时间线和创意资产管理系统结合，适合新人理解文件如何对应。
文字不是主体；标准术语保留英文，例如 creative brief、scene prompt、negative prompt、size、style、asset id、source、license、review status、prompt_plan.json、asset_manifest.csv。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch12-workshop-review-export-map.png",
        "size": "1024x1792",
        "quality": "high",
        "title": "第十二章多模态审核与导出流程图",
        "suggested_page": "docs/ch12-multimodal/ch05-projects/02-hands-on-multimodal-workshop.md",
        "alt": "多模态审核与导出流程图：asset manifest、source、license、portrait risk、contrast、safety review、storyboard、export preview 共同决定能否交付。",
        "prompt": """
一张适合第十二章 AIGC 实操工作坊的审核与导出图，主题是“能生成不等于能发布，review/export 是工作流的一部分”。
画面展示 assets 进入 review gate，检查 source、license、portrait risk、contrast、sensitive content、export limits，通过后进入 storyboard、timeline、content package、export_preview.html。
底部展示人工确认和 failed assets 回到 revision loop。
风格像内容审核闸门、导出流水线和产品发布 checklist 结合。
文字不是主体；标准术语保留英文，例如 review gate、source、license、portrait risk、contrast、sensitive content、export limits、storyboard、timeline、content package、export_preview.html、revision loop。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch12-workshop-failure-debug-map.png",
        "size": "1024x1792",
        "quality": "high",
        "title": "第十二章多模态失败样本排查图",
        "suggested_page": "docs/ch12-multimodal/ch05-projects/02-hands-on-multimodal-workshop.md",
        "alt": "多模态失败样本排查图：低对比度、缺少来源、授权不清、人像风险、Prompt 版本丢失和不可导出资产进入 failure_cases.md。",
        "prompt": """
一张适合第十二章多模态实操工作坊的失败样本排查图，主题是“AIGC 项目的失败不只是不好看，也可能是不可追溯、不可审核、不可导出”。
画面展示 failure_cases.md 收集 low contrast、missing source、unclear license、portrait risk、lost prompt version、unsupported export、incoherent storyboard 等问题。
旁边画出排查顺序：inspect brief、check prompt version、check asset manifest、check safety review、open export preview、choose fix action、rerun regression。
风格像多模态资产错误墙、审核清单和项目复盘板结合。
文字不是主体；标准术语保留英文，例如 failure_cases.md、low contrast、missing source、unclear license、portrait risk、prompt version、asset manifest、safety review、export preview、rerun regression。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch12-multimodal-app-engineering-loop.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "多模态应用工程链路图",
        "suggested_page": "docs/ch12-multimodal/ch01-multimodal/03-multimodal-apps.md",
        "alt": "多模态应用工程链路图：image/text/audio input、OCR、VLM、retrieval/tool call、fallback、privacy check 和 user feedback 组成产品闭环。",
        "prompt": """
一张适合多模态应用开发章节的工程链路图，主题是“多模态应用是输入处理、模型理解、工具调用和失败兜底的产品系统”。
画面表现 image/text/audio input、input quality check、OCR、VLM reasoning、retrieval/tool call、answer/action、fallback UI、privacy check、user feedback。
风格像产品系统架构图和应用工作流看板结合，突出工程链路和边界感。
文字不是主体；标准术语保留英文，例如 OCR、VLM、retrieval、tool call、fallback、privacy check、user feedback。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch12-sd-application-mode-selector-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Stable Diffusion 应用模式选择图",
        "suggested_page": "docs/ch12-multimodal/ch02-image-gen/03-sd-applications.md",
        "alt": "Stable Diffusion 应用模式选择图：text-to-image、img2img、inpainting、ControlNet、batch generation 和 human review 对应不同产品需求。",
        "prompt": """
一张适合 Stable Diffusion 应用章节的模式选择图，主题是“先判断用户需求，再选择生成或编辑模式”。
画面表现 user goal 分流到 text-to-image、img2img、inpainting、ControlNet/style control、batch generation、human review/export，并展示每种模式的输入输出差异。
风格像创意工具模式选择器和工作流地图结合，清晰、视觉化、适合新人做项目选型。
文字不是主体；标准术语保留英文，例如 text-to-image、img2img、inpainting、ControlNet、batch generation、human review、export。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch12-sd-finetuning-route-choice-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "图像生成微调路线选择图",
        "suggested_page": "docs/ch12-multimodal/ch02-image-gen/04-sd-finetuning.md",
        "alt": "图像生成微调路线选择图：Textual Inversion、LoRA、DreamBooth 分别适合概念触发词、可插拔风格和专属主体一致性。",
        "prompt": """
一张适合图像生成微调章节的路线选择图，主题是“不同微调方法在改不同层次的能力”。
画面表现 base Stable Diffusion 旁边分三条路线：Textual Inversion 学 concept token，LoRA 学 adapter/style，DreamBooth 学 subject identity；下方对比 data size、training cost、editability、identity consistency、overfitting risk。
风格像技术选型矩阵和模型插件示意图结合，专业但新人友好。
文字不是主体；标准术语保留英文，例如 Textual Inversion、LoRA、DreamBooth、concept token、adapter、subject identity、overfitting。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch12-image-generation-trend-radar-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "图像生成前沿趋势雷达图",
        "suggested_page": "docs/ch12-multimodal/ch02-image-gen/05-latest-progress.md",
        "alt": "图像生成前沿趋势雷达图：speed、control、editing、multimodal input、workflow integration、edge deployment 和 cost efficiency 组成趋势判断。",
        "prompt": """
一张适合图像生成最新进展章节的趋势雷达图，主题是“按方向追前沿，而不是只背模型名”。
画面表现 speed、quality、control、editing、multimodal input、workflow integration、edge deployment、cost efficiency 八个方向，中心是 image generation system，旁边有 trend priority / learning value 的判断卡。
风格像技术雷达和产品趋势地图结合，清晰、现代、适合课程扫读。
文字不是主体；标准术语保留英文，例如 speed、control、editing、multimodal input、workflow integration、edge deployment、cost efficiency。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch12-tts-text-to-speech-pipeline-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "TTS 文本到语音生成链路图",
        "suggested_page": "docs/ch12-multimodal/ch03-video-gen/02-tts.md",
        "alt": "TTS 文本到语音生成链路图：text normalization、phoneme、prosody、acoustic model、mel spectrogram、vocoder、speaker style 和 waveform。",
        "prompt": """
一张适合语音合成章节的文本到语音生成链路图，主题是“TTS 不只是读字，而是生成带音色、节奏和情感的声音”。
画面表现 raw text、text normalization、phoneme、prosody、speaker/style control、acoustic model、mel spectrogram、vocoder、waveform/audio，并标注 latency 和 quality tradeoff。
风格像音频信号流程图和语音产品控制台结合，突出波形和频谱视觉。
文字不是主体；标准术语保留英文，例如 text normalization、phoneme、prosody、mel spectrogram、vocoder、speaker style、waveform、latency。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch12-digital-human-sync-pipeline-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "数字人多模块同步图",
        "suggested_page": "docs/ch12-multimodal/ch03-video-gen/03-digital-human.md",
        "alt": "数字人多模块同步图：script、TTS、lip sync、face identity、gesture、background、review 和 export 需要时间轴同步。",
        "prompt": """
一张适合数字人技术章节的多模块同步图，主题是“数字人项目难在脚本、声音、口型、表情和身份一致性同步”。
画面表现 script、TTS voice、lip sync、face identity、expression/gesture、background scene、timeline alignment、quality review、export，并用时间轴展示误差累积。
风格像视频制作时间线和数字人系统架构结合，专业、直观、适合新人理解多模块管线。
文字不是主体；标准术语保留英文，例如 script、TTS、lip sync、face identity、gesture、timeline alignment、quality review、export。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch12-aigc-frontier-system-trend-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "AIGC 前沿趋势系统判断图",
        "suggested_page": "docs/ch12-multimodal/ch04-frontier/01-frontier-trends.md",
        "alt": "AIGC 前沿趋势系统判断图：model capability、cost efficiency、workflow integration、real-time generation、edge/local、governance 共同决定趋势价值。",
        "prompt": """
一张适合 AIGC 前沿趋势章节的系统判断图，主题是“趋势不是热词，而是能力、成本、产品和治理同时变化”。
画面表现 model capability、multimodal default、workflow generation、cost efficiency、real-time generation、edge/local deployment、system integration、governance boundary，中心是 product value。
风格像战略雷达和技术产品地图结合，适合新人从系统角度看前沿。
文字不是主体；标准术语保留英文，例如 model capability、workflow generation、cost efficiency、real-time generation、edge deployment、governance、product value。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch12-ai-ethics-safety-guardrail-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "AIGC 伦理安全风险护栏图",
        "suggested_page": "docs/ch12-multimodal/ch04-frontier/02-ai-ethics.md",
        "alt": "AIGC 伦理安全风险护栏图：bias、privacy、misinformation、abuse、overtrust 通过 policy、filter、human review、audit log 和 appeal 形成护栏。",
        "prompt": """
一张适合 AI 伦理与安全章节的风险护栏图，主题是“伦理问题要落到工程护栏和复核机制”。
画面表现 risk inputs：bias、privacy leakage、misinformation、abuse、overtrust；中间是 guardrails：policy, data minimization, output filter, human review, audit log, appeal mechanism；右侧是 safer product outcome。
风格像安全控制台和风险地图结合，克制、可信、清晰。
文字不是主体；标准术语保留英文，例如 bias、privacy leakage、misinformation、abuse、overtrust、policy、human review、audit log、appeal。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch12-ai-regulation-engineering-translation-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "AI 合规工程转译图",
        "suggested_page": "docs/ch12-multimodal/ch04-frontier/03-ai-regulations.md",
        "alt": "AI 合规工程转译图：privacy、risk classification、traceability、human oversight、content labeling 和 audit 被转译为系统配置。",
        "prompt": """
一张适合 AI 法规与合规章节的工程转译图，主题是“把法规语言翻译成系统配置和产品流程”。
画面左侧是 compliance requirements：privacy, risk classification, traceability, human oversight, content labeling, audit；中间是 engineering translation；右侧是 system controls：permission, logging, review queue, watermark/label, data retention, incident response。
风格像合规矩阵和系统架构图结合，专业、清晰、非法律建议感。
文字不是主体；标准术语保留英文，例如 privacy、risk classification、traceability、human oversight、content labeling、audit、logging、review queue、incident response。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
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
        "filename": "appendix-ai-15-stage-history-map-en.png",
        "size": "1024x1792",
        "quality": "high",
        "title": "AI 15-stage development history vertical map",
        "suggested_page": "docs/appendix/ai-milestones.md",
        "alt": "AI 15-stage development history vertical map: Symbolic AI, Expert Systems, ML, Deep Learning, Transformer, LLM, RAG, Agent, and Multimodal development stages.",
        "prompt": """
A vertical English course infographic, timeline plus comic panels, teaching beginners the 15-stage development history of AI.
Use a clean vertical timeline with 15 numbered stage cards. Each card has one big icon and a short label only:
1 Symbolic AI, 2 Expert Systems, 3 Statistical ML, 4 Feature Engineering, 5 Neural Networks, 6 Deep Learning, 7 CNN Vision, 8 RNN Sequence, 9 Attention, 10 Transformer, 11 Pretraining, 12 LLM, 13 RAG, 14 Agent, 15 Multimodal.
Show the flow as a relay: rules -> data -> representation -> scale -> language -> tools -> multimodal systems.
Make it beginner-friendly, readable on mobile, and organized as a teaching map rather than a decorative poster.
Use sparse large English labels only. Avoid dense years, tiny text, gibberish, fake paper covers, watermarks, and real brand logos.
Style: professional course illustration, vertical 9:16, timeline, light comic feeling, clear arrows, calm colors, enough whitespace.
""".strip(),
    },
    {
        "filename": "appendix-ai-15-stage-history-map.png",
        "size": "1024x1792",
        "quality": "high",
        "title": "AI 15 阶段发展史竖版地图",
        "suggested_page": "docs/appendix/ai-milestones.md",
        "alt": "AI 15 阶段发展史竖版地图：从符号主义、专家系统到机器学习、深度学习、Transformer、LLM、RAG、Agent 和多模态。",
        "prompt": """
一张竖版中文课程图解，用时间轴 + 漫画分镜感，面向新人讲清 AI 15 阶段发展史。
画面是一条清晰竖向时间轴，包含 15 张编号阶段卡。每张卡只放一个大图标和一个短标签：
1 符号主义、2 专家系统、3 统计机器学习、4 特征工程、5 神经网络、6 深度学习、7 CNN 视觉、8 RNN 序列、9 Attention、10 Transformer、11 预训练、12 LLM、13 RAG、14 Agent、15 多模态。
整体表达接力关系：规则 -> 数据 -> 表征 -> 规模 -> 语言 -> 工具 -> 多模态系统。
要像教学地图，不要装饰海报；移动端可读，层级清楚，箭头少而明确。
文字要少、字号大、清晰。技术词 CNN、RNN、Attention、Transformer、LLM、RAG、Agent 可保留英文。不要密集年份、不要小字乱码、不要真实论文封面、不要真实品牌 logo、不要水印。
风格专业、温和、新手友好，竖版 9:16，时间轴、轻漫画感、图标清楚，留白充足。
""".strip(),
    },
    {
        "filename": "appendix-ai-15-stage-history-map-ja.png",
        "size": "1024x1792",
        "quality": "high",
        "title": "AI 15段階発展史の縦型マップ",
        "suggested_page": "docs/appendix/ai-milestones.md",
        "alt": "AI 15段階発展史の縦型マップ：記号AI、エキスパートシステムから機械学習、深層学習、Transformer、LLM、RAG、Agent、マルチモーダルへ進む流れ。",
        "prompt": """
日本語の縦長コース図解。タイムライン + 漫画パネル風で、初心者向けに AI の 15 段階発展史を説明する。
中央に読みやすい縦型タイムラインを置き、15 枚の番号付きステージカードを並べる。各カードは大きなアイコンと短いラベルだけ：
1 記号AI、2 エキスパートシステム、3 統計的機械学習、4 特徴量設計、5 ニューラルネット、6 深層学習、7 CNN 画像、8 RNN 系列、9 Attention、10 Transformer、11 事前学習、12 LLM、13 RAG、14 Agent、15 マルチモーダル。
全体はリレーの流れとして見せる：ルール -> データ -> 表現 -> スケール -> 言語 -> ツール -> マルチモーダルシステム。
教材マップとして分かりやすくし、装飾ポスターにしない。モバイルで読めるように階層を整理し、矢印は少なく明確に。
文字は少なく、大きく、読みやすく。CNN、RNN、Attention、Transformer、LLM、RAG、Agent などの標準用語は英語表記でよい。細かい年表、文字化け、実在論文カバー、実在ブランドロゴ、透かしは禁止。
スタイルはプロフェッショナルでやさしい初心者向け教材図解、縦 9:16、タイムライン、軽い漫画感、明快なアイコン、十分な余白。
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
        "filename": "appendix-debug-mre-help-flow.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "卡点最小复现与求助流程图",
        "suggested_page": "docs/appendix/troubleshooting.md",
        "alt": "卡点最小复现与求助流程图：reproduce、environment、minimal input、error log、expected vs actual 和 clear question 组成求助闭环。",
        "prompt": """
一张适合学习卡点救援附录的行动流程图，主题是“把卡住变成可复现、可求助、可解决的问题”。
画面表现 reproduce problem、environment info、minimal input、error log、expected vs actual、clear question、helper feedback、fix and note，形成闭环。
风格像技术救援流程图和温暖学习支持卡片结合，降低焦虑、强调行动步骤。
文字不是主体；标准术语保留英文，例如 reproduce、environment、minimal input、error log、expected vs actual、clear question。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "appendix-hardware-local-cloud-api-cost-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "本地 云 API 三路线成本对比图",
        "suggested_page": "docs/appendix/hardware.md",
        "alt": "本地 云 API 三路线成本对比图：local laptop、cloud GPU 和 API-first 三条路线在 cost、setup、training、deployment 和 stability 上取舍不同。",
        "prompt": """
一张适合硬件与云资源附录的三路线对比图，主题是“先按任务选择本地、云 GPU 或 API 路线”。
画面表现 local laptop、cloud GPU、API-first 三条路线，分别对比 cost、setup complexity、training ability、app development、stability、when to choose。
风格像理性购买决策表和学习阶段路线图结合，明确表达“不要一开始就买显卡”的新人友好判断。
文字不是主体；标准术语保留英文，例如 local laptop、cloud GPU、API-first、cost、training、deployment、stability。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "appendix-job-portfolio-storyline-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "AI 求职作品集故事线图",
        "suggested_page": "docs/appendix/job-prep.md",
        "alt": "AI 求职作品集故事线图：target role、user problem、technical solution、metrics、failure analysis、README 和 interview story 串成作品集表达。",
        "prompt": """
一张适合 AI 求职准备附录的作品集故事线图，主题是“项目要讲成能被面试官理解的能力证据”。
画面表现 target role、user problem、technical solution、metrics/result、failure analysis、README/GitHub、interview story 七个节点，形成从项目到面试表达的故事线。
风格像作品集路线图和简历打磨看板结合，专业、清晰、鼓励新人聚焦 2 到 3 个项目。
文字不是主体；标准术语保留英文，例如 target role、user problem、technical solution、metrics、failure analysis、README、interview story。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "appendix-learning-paper-project-notes-loop.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "论文 项目 知识库复盘循环图",
        "suggested_page": "docs/appendix/continuous-learning.md",
        "alt": "论文 项目 知识库复盘循环图：paper reading、code experiment、project improvement、notes card 和 weekly review 形成持续学习闭环。",
        "prompt": """
一张适合持续学习方法论附录的学习闭环图，主题是“看过的内容要回到项目和知识库里沉淀”。
画面表现 paper reading、code experiment、project improvement、notes card、weekly review、next question 六个节点循环，旁边提示 avoid collection-only learning。
风格像长期学习飞轮和个人知识库工作流结合，稳定、温暖、有行动感。
文字不是主体；标准术语保留英文，例如 paper reading、code experiment、project improvement、notes card、weekly review、next question。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "appendix-resource-bottleneck-priority-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "资源选择卡点优先级图",
        "suggested_page": "docs/appendix/resources.md",
        "alt": "资源选择卡点优先级图：concept、code、math、engineering、project 和 communication 卡点分别对应不同资源补充路径。",
        "prompt": """
一张适合推荐学习资源附录的卡点优先级图，主题是“先识别卡点，再选择资源，而不是囤资源”。
画面表现 bottleneck diagnosis 分流到 concept、code、math、engineering、project、communication 六类卡点，每类连接一种资源类型：course note、docs、visual explanation、source code、case study、peer review，最后回到 project validation。
风格像资源导航图和问题分诊台结合，清晰、降低信息焦虑。
文字不是主体；标准术语保留英文，例如 bottleneck、concept、code、math、engineering、project validation、peer review。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "appendix-faq-confidence-reset-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "FAQ 焦虑重置与行动分流图",
        "suggested_page": "docs/appendix/faq.md",
        "alt": "FAQ 焦虑重置与行动分流图：math anxiety、GPU anxiety、time anxiety、project anxiety、paper anxiety 和 job anxiety 被拆成下一步行动。",
        "prompt": """
一张适合常见问题附录的新人焦虑重置图，主题是“把模糊焦虑拆成下一步行动”。
画面表现 math anxiety、GPU anxiety、time anxiety、project anxiety、paper anxiety、job anxiety 六个气泡，经过 confidence reset 和 action routing，落到 review chapter、run minimal demo、ask for help、build portfolio 等行动卡片。
风格温暖、明亮、支持感强，像学习咨询地图，不要说教。
文字不是主体；标准术语保留英文，例如 math anxiety、GPU anxiety、project anxiety、confidence reset、action routing、minimal demo、portfolio。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "appendix-quick-ref-debug-index-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "AI 项目速查排障索引图",
        "suggested_page": "docs/appendix/resource-quick-ref.md",
        "alt": "AI 项目速查排障索引图：environment、data、training、evaluation、RAG、Agent、Prompt 和 frontend 分流到不同速查检查项。",
        "prompt": """
一张适合学习资源速查附录的排障索引图，主题是“做项目时快速判断该查哪一块”。
画面表现中心 project stuck，分流到 environment、data、training、evaluation、RAG、Agent、Prompt、frontend 八个区域，每个区域连接 2 到 3 个 check items。
风格像工程速查地图和控制台索引，清晰、模块化、适合项目旁边快速查看。
文字不是主体；标准术语保留英文，例如 environment、data、training、evaluation、RAG、Agent、Prompt、frontend、check items。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "appendix-course-numbering-maintenance-check.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "课程维护命名一致性检查图",
        "suggested_page": "docs/appendix/course-numbering.md",
        "alt": "课程维护命名一致性检查图：display chapter、source directory、sidebar label、image filename 和 manifest entry 需要保持一致。",
        "prompt": """
一张适合课程编号约定附录的维护检查图，主题是“展示编号、源码目录、侧边栏和图片命名要一致”。
画面表现 display chapter number、source directory、sidebar label、image filename、manifest entry、build validation 六个检查点，形成 content maintenance checklist。
风格像文档工程质检看板，清楚、克制、适合维护者避免 chxx 和中文章节号混用。
文字不是主体；标准术语保留英文，例如 display chapter、source directory、sidebar label、image filename、manifest entry、build validation。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "appendix-image-production-pipeline-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "课程图片从缺口到生成发布流程图",
        "suggested_page": "docs/appendix/visual-enhancement-plan.md",
        "alt": "课程图片从缺口到生成发布流程图：content gap、visual intent、image2 generation、markdown insertion、manifest progress、validation 和 build 组成图片生产闭环。",
        "prompt": """
一张适合课程视觉增强规划附录的图片生产流程图，主题是“图片资产要从理解缺口出发，经过生成、引用、记录和校验闭环”。
画面表现 content gap scan、visual intent、image2 prompt、generate PNG、insert markdown、update manifest/progress、validate docs、build check、commit，形成课程图片生产流水线。
风格像内容生产控制台和发布管线结合，强调图片服务理解而不是装饰。
文字不是主体；标准术语保留英文，例如 content gap、visual intent、image2 prompt、manifest、progress、validate docs、build check、commit。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "elective-cpp-deployment-module-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "C++ 与模型部署模块学习地图",
        "suggested_page": "docs/electives/module-a/index.md",
        "alt": "C++ 与模型部署模块学习地图：C++ 基础、进阶资源管理、模型优化、推理引擎、边缘部署、服务化和部署项目逐步连接。",
        "prompt": """
一张适合 C++ 与模型部署选修模块首页的学习地图，主题是“从能跑模型到稳定部署模型”。
画面表现 C++ basics、RAII/ownership、model optimization、inference engine、edge deployment、model serving、deployment project 七个节点逐步连接。
风格像工程学习路线图，突出部署不是单点技术，而是性能、资源、硬件、服务和交付共同组成的链路。
文字不是主体；标准术语保留英文，例如 C++、RAII、ownership、ONNX、inference engine、edge、serving。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "elective-cpp-raii-ownership-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "C++ RAII 与所有权地图",
        "suggested_page": "docs/electives/module-a/02-cpp-advanced.md",
        "alt": "C++ RAII 与所有权地图：资源获取、对象生命周期、智能指针、移动语义和抽象接口共同支撑部署工程。",
        "prompt": """
一张适合 C++ 进阶课程的教学图，主题是“RAII 和所有权如何让部署代码更稳”。
用 5 个大模块串联：resource acquire、RAII guard、smart pointer、move semantics、inference backend。
风格像清晰的工程白板，展示资源从创建到自动释放的生命周期。
文字不是主体；标准术语保留英文，例如 RAII、unique_ptr、move semantics、backend。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "elective-edge-deployment-constraint-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "边缘部署约束决策图",
        "suggested_page": "docs/electives/module-a/05-edge-deployment.md",
        "alt": "边缘部署约束决策图：内存、功耗、延迟、离线能力、散热和运维共同决定模型适配方案。",
        "prompt": """
一张适合边缘设备部署课程的决策图，主题是“边缘部署不是把云端服务搬到小机器上”。
画面表现 memory budget、power budget、latency target、offline fallback、thermal limit、remote update 六个约束围绕一个 edge device。
突出模型压缩、推理引擎、缓存、降级策略和运维监控如何一起选择，风格像硬件方案白板。
文字不是主体；标准术语保留英文，例如 edge device、memory、power、latency、offline fallback、thermal、update。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "elective-deployment-project-delivery-loop.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "部署综合项目交付闭环图",
        "suggested_page": "docs/electives/module-a/07-projects.md",
        "alt": "部署综合项目交付闭环图：模型准备、推理执行、服务接口、指标统计、部署说明、监控和项目展示形成闭环。",
        "prompt": """
一张适合部署综合项目页的交付闭环图，主题是“部署项目展示的是稳定系统，而不是单次预测截图”。
画面表现 model export、optimization、inference runtime、API/batch interface、metrics、monitoring、README demo 七个环节形成闭环。
重点突出 latency、throughput、memory、version、rollback 和 failure cases 是作品集展示重点，风格专业、项目交付感强。
文字不是主体；标准术语保留英文，例如 model export、runtime、API、batch、latency、throughput、monitoring、README。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "elective-python-advanced-module-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Python 进阶专题模块地图",
        "suggested_page": "docs/electives/module-b/index.md",
        "alt": "Python 进阶专题模块地图：装饰器、生成器、asyncio 并发和元编程共同提升工程代码的复用性、流式处理和可维护性。",
        "prompt": """
一张适合 Python 进阶选修模块首页的学习地图，主题是“让工程代码更稳、更快、更好维护”。
画面表现 decorators、generators、asyncio concurrency、metaprogramming 四个工具模块连接到 logging/retry、streaming data、I/O tasks、registry/config driven code。
风格像 Python 工程工具箱，清爽、现代、适合新人理解专题用途。
文字不是主体；标准术语保留英文，例如 decorator、generator、asyncio、event loop、registry、config。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "elective-generator-stream-pipeline.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "生成器流式管道图",
        "suggested_page": "docs/electives/module-b/02-iterators-advanced.md",
        "alt": "生成器流式管道图：数据按需产出，经过读取、过滤、规范化和消费，不必一次性全部进入内存。",
        "prompt": """
一张适合 Python 迭代器与生成器课程的教学图，主题是“数据一边产生，一边消费”。
画面表现 source stream、yield、filter、normalize、batch、consumer 六个管道节点，旁边对比一次性 list loading 的内存压力。
重点突出 lazy evaluation、streaming、yield from、memory saving 和 pipeline composition，风格直观、工程化。
文字不是主体；标准术语保留英文，例如 generator、yield、yield from、lazy evaluation、streaming、pipeline。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "elective-metaprogramming-registry-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Python 元编程注册器地图",
        "suggested_page": "docs/electives/module-b/04-metaprogramming.md",
        "alt": "Python 元编程注册器地图：动态类、注册器、描述符和配置驱动代码帮助减少重复样板，但需要控制复杂度。",
        "prompt": """
一张适合 Python 元编程课程的教学图，主题是“用代码组织代码结构，但不要把它变成魔法”。
画面表现 dynamic class、registry、descriptor、config driven code 四个模块，中间连接到插件加载、模型加载器和字段校验。
同时表现一条风险提示：overuse、hard to debug、hidden behavior，需要适度使用。
文字不是主体；标准术语保留英文，例如 type、registry、descriptor、__set_name__、config、plugin。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "elective-classic-ml-module-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "经典 ML 补充算法模块地图",
        "suggested_page": "docs/electives/module-c/index.md",
        "alt": "经典 ML 补充算法模块地图：SVM、KNN、朴素贝叶斯和 LDA 适合中小数据、可解释 baseline 和模型选择补充。",
        "prompt": """
一张适合经典机器学习补充模块首页的学习地图，主题是“经典算法是判断工具箱，不是过时清单”。
画面表现 SVM、KNN、Naive Bayes、LDA 四个算法节点，分别连接到 margin boundary、neighbor voting、evidence update、supervised projection。
突出中小数据、强 baseline、可解释性、特征缩放和模型选择，风格像算法工具箱地图。
文字不是主体；标准术语保留英文，例如 SVM、KNN、Naive Bayes、LDA、baseline、scaling。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "elective-lda-projection-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "LDA 监督式投影直觉图",
        "suggested_page": "docs/electives/module-c/04-lda.md",
        "alt": "LDA 监督式投影直觉图：利用标签寻找投影方向，让类内更紧、类间更远，同时支持分类和降维。",
        "prompt": """
一张适合 LDA 课程的数学直觉图，主题是“类内更紧，类间更远的监督式投影”。
画面表现二维散点按类别分布，LDA 找到一条投影轴，投影后同类点聚拢、不同类中心拉开；旁边用 PCA 对比只看总体方差。
重点突出 within-class scatter、between-class scatter、supervised projection、classification and dimensionality reduction。
文字不是主体；标准术语保留英文，例如 LDA、PCA、projection、within-class、between-class。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "elective-optimization-tradeoff-dashboard.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "模型优化取舍仪表盘",
        "suggested_page": "docs/electives/module-a/03-model-optimization.md",
        "alt": "模型优化取舍仪表盘：latency、throughput、memory、accuracy、hardware fit 和 maintenance cost 共同决定量化、蒸馏、剪枝、融合和 batching 的选择。",
        "prompt": """
一张适合模型优化选修课的取舍仪表盘图，主题是“优化不是白赚，而是在指标之间做工程取舍”。
画面表现 latency、throughput、memory、accuracy、hardware fit、maintenance cost 六个仪表盘，中间连接 quantization、distillation、pruning、operator fusion、batching 五种优化手段。
风格像部署性能控制台和技术选型白板结合，突出瓶颈定位、指标对比和回归测试。
文字不是主体；标准术语保留英文，例如 latency、throughput、memory、accuracy、quantization、distillation、operator fusion、batching。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "elective-inference-engine-selection-matrix.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "推理引擎选型矩阵图",
        "suggested_page": "docs/electives/module-a/04-inference-engines.md",
        "alt": "推理引擎选型矩阵图：model format、target hardware、latency、throughput、deployment environment 和 maintenance 决定 ONNX Runtime、TensorRT、OpenVINO 的选择。",
        "prompt": """
一张适合推理引擎课程的选型矩阵图，主题是“推理引擎要匹配模型格式、硬件和运行目标”。
画面表现 ONNX Runtime、TensorRT、OpenVINO 三个候选工具与 model format、CPU/GPU/NPU、latency、throughput、deployment environment、maintenance complexity 的关系。
风格像工程决策矩阵和硬件适配地图结合，清晰、专业、帮助新人不迷信单一引擎。
文字不是主体；标准术语保留英文，例如 ONNX Runtime、TensorRT、OpenVINO、CPU、GPU、NPU、latency、throughput。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "elective-serving-metrics-version-routing-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "模型服务指标与版本路由图",
        "suggested_page": "docs/electives/module-a/06-model-serving.md",
        "alt": "模型服务指标与版本路由图：request queue、dynamic batching、P95/P99 latency、error rate、version routing、canary 和 rollback 组成服务运维闭环。",
        "prompt": """
一张适合模型服务化课程的运行指标与版本路由图，主题是“模型服务是长期运行系统，不是单次推理脚本”。
画面表现 request queue、dynamic batching、model executor、version routing、canary release、rollback、metrics dashboard，包括 P95/P99 latency、throughput、error rate、batch efficiency。
风格像线上服务运维控制台和架构图结合，突出可观测性、灰度和回滚。
文字不是主体；标准术语保留英文，例如 request queue、dynamic batching、P95、P99、version routing、canary、rollback、metrics。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "elective-decorator-crosscutting-layers.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "装饰器横切逻辑分层图",
        "suggested_page": "docs/electives/module-b/01-decorators-advanced.md",
        "alt": "装饰器横切逻辑分层图：logging、timing、retry、auth wrappers 包裹 original function，并通过 functools.wraps 保留函数身份。",
        "prompt": """
一张适合 Python 装饰器进阶课程的横切逻辑分层图，主题是“装饰器把通用工程逻辑包在原函数外层”。
画面表现 original function 被 logging wrapper、timing wrapper、retry wrapper、auth wrapper 逐层包裹，请求穿过 wrapper 后执行函数再返回；旁边强调 functools.wraps 保留 metadata。
风格像函数调用剖面图和工程中间件栈结合，清晰、适合新人理解为什么装饰器常见。
文字不是主体；标准术语保留英文，例如 wrapper、original function、logging、timing、retry、auth、functools.wraps、metadata。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "elective-asyncio-timeout-cancel-rate-limit-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "异步任务超时取消与限流图",
        "suggested_page": "docs/electives/module-b/03-concurrency.md",
        "alt": "异步任务超时取消与限流图：event loop、semaphore、timeout、cancellation、retry 和 rate limit 共同保护 LLM API、RAG 抓取和 Agent 工具调用。",
        "prompt": """
一张适合 Python asyncio 并发课程的任务控制图，主题是“并发要配合超时、取消和限流才适合生产系统”。
画面表现 event loop 调度多个 API tasks，通过 semaphore 控制并发上限，timeout 终止慢任务，cancellation 清理任务，retry 队列处理可恢复错误，rate limit 保护上游服务。
风格像异步运行时仪表盘和任务队列图结合，适合 AI 工程场景。
文字不是主体；标准术语保留英文，例如 event loop、semaphore、timeout、cancellation、retry、rate limit、LLM API、Agent tools。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "elective-svm-c-kernel-decision-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "SVM 参数 C 与 kernel 选择图",
        "suggested_page": "docs/electives/module-c/01-svm.md",
        "alt": "SVM 参数 C 与 kernel 选择图：C 控制 margin 与容错，kernel 控制 linear/RBF 非线性边界，StandardScaler 是常见前置步骤。",
        "prompt": """
一张适合 SVM 选修课的参数直觉图，主题是“C 和 kernel 改变分类边界的行为”。
画面左右对比 low C 与 high C 的 margin/容错差异，下方对比 linear kernel 和 RBF kernel 的边界形状；旁边标出 StandardScaler 前置步骤。
风格像机器学习决策白板，散点图清楚，边界线和 margin 大而易懂。
文字不是主体；标准术语保留英文，例如 C、margin、support vectors、linear kernel、RBF kernel、StandardScaler。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "elective-ai-security-threat-regression-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "AI 安全威胁建模与回归集图",
        "suggested_page": "docs/electives/module-d.md",
        "alt": "AI 安全威胁建模与回归集图：assets、attack surface、failure impact、red team cases、guardrail fix 和 regression suite 形成持续安全闭环。",
        "prompt": """
一张适合 AI 安全与红队课程的威胁建模图，主题是“红队测试要从资产和攻击面开始，并沉淀成回归集”。
画面表现 assets、attack surface、failure impact、red team cases、risk scoring、guardrail fix、regression suite、continuous evaluation 的闭环。
风格像安全作战室和评估看板结合，专业、清晰、强调系统链路而不是单个 jailbreak prompt。
文字不是主体；标准术语保留英文，例如 assets、attack surface、red team cases、risk scoring、guardrail、regression suite、continuous evaluation。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "elective-ai-frontend-state-machine-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "AI 前端状态机与体验闭环图",
        "suggested_page": "docs/electives/module-e.md",
        "alt": "AI 前端状态机与体验闭环图：idle、loading、streaming、success、error、retry、cancel 和 feedback 组成 AI 产品交互体验。",
        "prompt": """
一张适合 AI 前端基础课程的状态机图，主题是“AI 前端要处理不确定耗时和失败恢复”。
画面表现用户输入后进入 idle、loading、streaming、success、error、retry、cancel、feedback 等状态，连接到 fetch/API、loading skeleton、error banner、result card、history。
风格像前端交互状态图和产品界面流程图结合，清晰、现代、适合新人理解体验闭环。
文字不是主体；标准术语保留英文，例如 idle、loading、streaming、success、error、retry、cancel、feedback、fetch API。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "elective-ai-product-experiment-metrics-loop.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "AI 产品实验与指标闭环图",
        "suggested_page": "docs/electives/module-f.md",
        "alt": "AI 产品实验与指标闭环图：user problem、hypothesis、MVP、success metrics、risk boundary、feedback 和 iteration 决定 AI 功能是否继续投入。",
        "prompt": """
一张适合 AI 产品设计思维课程的实验闭环图，主题是“先验证用户问题和成功指标，再决定继续投入 AI 功能”。
画面表现 user problem、hypothesis、MVP、success metrics、risk boundary、cost estimate、user feedback、iteration decision，形成产品实验循环。
风格像产品策略看板和实验漏斗结合，专业、清晰、强调价值、成本、风险和体验。
文字不是主体；标准术语保留英文，例如 user problem、hypothesis、MVP、success metrics、risk boundary、feedback、iteration。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "elective-workshop-route-map.png",
        "size": "1024x1792",
        "quality": "high",
        "title": "选修模块实操路线图",
        "suggested_page": "docs/electives/hands-on-elective-workshop.md",
        "alt": "选修模块实操路线图：从选择 A-F 方向，到运行代码、生成证据、修复失败和整理作品集。",
        "prompt": """
一张适合选修模块总实操页的竖版课程路线图，主题是“选修不是额外阅读，而是把一个方向做成可交付证据”。
画面从上到下分成 5 个清晰步骤：choose track、run workshop、inspect artifacts、fix failure、portfolio evidence。
旁边用六张小卡片表示 Module A C++ deployment、Module B advanced Python、Module C classic ML、Module D safety、Module E frontend、Module F product。
风格像新人可以照着操作的漫画式流程课页，线条清楚，步骤编号大，强调先选方向再动手。
文字不是主体；标准术语保留英文，例如 Module A、deployment、pipeline、KNN、red team、dashboard、RICE、README。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "elective-workshop-evidence-pipeline.png",
        "size": "1024x1792",
        "quality": "high",
        "title": "选修实操证据包流水线图",
        "suggested_page": "docs/electives/hands-on-elective-workshop.md",
        "alt": "选修实操证据包流水线图：需求、模块任务、可运行代码、输出文件、评估报告和作品集 README 连接成证据链。",
        "prompt": """
一张适合选修模块实操课的竖版证据流水线图，主题是“每个选修知识点最后都要落到一个可检查文件”。
画面表现 requirement card 进入 module task registry，然后流向 runnable code、CSV/JSON/HTML/Markdown artifacts、readiness report、failure cases、portfolio README。
用小图标区分 CSV、JSON、HTML、Markdown 和 dashboard，突出证据不是截图，而是可以复跑、检查、提交的文件。
风格像工程发布流水线和学习笔记结合，清晰、分步骤、适合新人跟做。
文字不是主体；标准术语保留英文，例如 task registry、artifact、readiness score、failure cases、README、re-run。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "elective-workshop-code-execution-sequence.png",
        "size": "1024x1792",
        "quality": "high",
        "title": "选修实操代码执行顺序图",
        "suggested_page": "docs/electives/hands-on-elective-workshop.md",
        "alt": "选修实操代码执行顺序图：reset workspace、run module A-F、generate dashboard、build readiness report 和 print expected output 依次发生。",
        "prompt": """
一张适合选修模块实操代码讲解的竖版执行顺序图，主题是“main 函数怎样把 A-F 六个选修方向串成一次完整运行”。
画面从上到下展示 reset workspace、write module_tasks.json、run module A deployment、run module B pipeline、run module C KNN、run module D red team、run module F RICE、run module E dashboard、build readiness report、print output。
每一步旁边放一个简短输出文件或结果标签，让新人知道代码不是黑盒。
风格像代码执行时间线和漫画式调试图结合，步骤清楚、箭头明确。
文字不是主体；标准术语保留英文，例如 main()、reset_workspace、module-a、KNN、red team、RICE、dashboard、readiness_score.json。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "elective-workshop-debug-loop.png",
        "size": "1024x1792",
        "quality": "high",
        "title": "选修实操常见错误排查闭环图",
        "suggested_page": "docs/electives/hands-on-elective-workshop.md",
        "alt": "选修实操常见错误排查闭环图：找不到文件、输出为空、指标不达标、红队失败和浏览器打不开都进入定位、修复、复跑闭环。",
        "prompt": """
一张适合选修模块实操课的竖版错误排查图，主题是“卡住时先定位证据断在哪一层，再修复和复跑”。
画面表现 5 个常见故障卡片：file not found、empty output、low score、red-team failure、dashboard not opening。
每张故障卡片连接到定位动作：check path、print intermediate data、inspect CSV/JSON、add regression case、open HTML locally，然后回到 re-run workshop。
风格像故障排查漫画和工程 runbook，友好但专业，适合基础薄弱的学习者。
文字不是主体；标准术语保留英文，例如 file not found、empty output、low score、red-team failure、dashboard、re-run。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "elective-workshop-portfolio-pack.png",
        "size": "1024x1792",
        "quality": "high",
        "title": "选修实操作品集证据包图",
        "suggested_page": "docs/electives/hands-on-elective-workshop.md",
        "alt": "选修实操作品集证据包图：README、运行命令、指标、失败案例、截图、改进计划和下一步任务组成可展示作品。",
        "prompt": """
一张适合选修模块收尾练习的竖版作品集证据包图，主题是“完成选修后要留下别人能复现、能检查、能相信的材料”。
画面像一个整洁的项目文件夹，里面有 README、运行命令、指标表、失败案例、dashboard 截图、下一步任务、commit note 七张卡片。
强调作品集不是只说学过，而是展示代码、输出、评估、风险和下一步改进。
风格像项目交付清单和教学漫画结合，竖版、分步骤、清楚。
文字不是主体；标准术语保留英文，例如 README、metrics、failure cases、dashboard、commit。其他说明尽量用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "appendix-ai-main-relay-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "AI 主线接力总地图",
        "suggested_page": "docs/appendix/ai-milestones.md",
        "alt": "AI 主线接力总地图：概率信息论、早期神经网络、经典机器学习、深度学习复兴、Transformer 大模型、RAG Agent 和多模态 AIGC 接力演进。",
        "prompt": """
一张适合 AI 历史附录的主线总地图，主题是“AI 发展像一场接力赛”。
用 6 个大节点从左到右串联：probability、classic ML、deep learning、Transformer/LLM、RAG/Agent、multimodal AIGC。
风格像清晰的历史路线图，节点大、连线少、层次干净。
文字不是主体；标准术语保留英文。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "appendix-ai-history-comic-turning-points.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "AI 历史转折多格漫画",
        "suggested_page": "docs/appendix/ai-milestones.md",
        "alt": "AI 历史转折多格漫画：感知器希望、XOR 冷却、反向传播复兴、AlexNet 点火、Transformer 换轨和 Agent 工程化组成六格剧情。",
        "prompt": """
一张适合 AI 课程附录的六格漫画式历史图，主题是“AI 历史的六次转折”。
画面用 2 行 3 列六个大面板展示：Perceptron、XOR、Backprop、AlexNet、Transformer、Agent。
每格只放一个大图标和一个短标签：线性分界线、XOR 冰块、多层网络梯度箭头、GPU 训练火花、tokens 之间的 attention 光束、带工具和护栏的 Agent 控制台。
风格像简洁科普漫画，留白充足，线条清楚，不画真实人物，不做复杂背景。
文字不是主体；只保留少量大号英文标签 Perceptron、XOR、Backprop、AlexNet、Transformer、Agent。不要密集小字、不要乱码、不要真实品牌 logo、不要真实论文封面。
""".strip(),
    },
    {
        "filename": "appendix-ai-paper-problem-solution-impact-chain.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "AI 论文问题方法影响因果链图",
        "suggested_page": "docs/appendix/ai-milestones.md",
        "alt": "AI 论文问题方法影响因果链图：old bottleneck、new method、new capability 和 project impact 帮新人读懂论文为什么重要。",
        "prompt": """
一张适合 AI 重要论文学习方法的因果链图，主题是“不要先背论文名，先读懂它接住了哪个老问题”。
画面从左到右展示 old bottleneck、new method、new capability、project impact 四段链条，并用三条示例线贯穿：Backprop、Transformer、CLIP。
Backprop 示例连接多层网络训练困难到 gradient flow；Transformer 示例连接 RNN 并行瓶颈到 self-attention；CLIP 示例连接图文割裂到 shared embedding space。
风格像研究白板和课程路线图结合，箭头清楚，适合新人建立“问题驱动读论文”的方法。
文字不是主体；标准术语保留英文，例如 old bottleneck、new method、new capability、project impact、Backprop、Transformer、CLIP。其他说明用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "appendix-ai-project-lens-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "从项目视角读 AI 时间线地图",
        "suggested_page": "docs/appendix/ai-milestones.md",
        "alt": "从项目视角读 AI 时间线地图：数学基础、机器学习、深度学习、Transformer、RAG、Agent 和多模态分别对应课程项目能力。",
        "prompt": """
一张适合 AI 历史附录的项目视角地图，主题是“把论文历史翻译成项目能力”。
用 6 个大模块表现：math foundation、ML baseline、training loop、Transformer interface、RAG/Agent system、multimodal product。
风格像课程项目路线图，清晰、少字、适合新人快速理解。
文字不是主体；标准术语保留英文。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "appendix-classic-ml-branch-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "经典机器学习分支地图",
        "suggested_page": "docs/appendix/ai-milestones.md",
        "alt": "经典机器学习分支地图：贝叶斯、EM、SVM、树模型、集成学习和核方法共同建立稳定建模与评估主线。",
        "prompt": """
一张适合 AI 历史附录经典机器学习章节的分支地图，主题是“经典 ML 建立稳定建模习惯”。
用 5 个大分支表现：Bayes、EM、SVM、decision tree、ensemble，最后汇入 baseline and evaluation。
风格像算法谱系图，干净、专业、少字。
文字不是主体；标准术语保留英文。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "appendix-neural-network-waves-timeline.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "神经网络三次浪潮与两次低谷时间线",
        "suggested_page": "docs/appendix/ai-milestones.md",
        "alt": "神经网络三次浪潮与两次低谷时间线：Perceptron、XOR、Backprop、vanishing gradient、LSTM、RBM、AlexNet、ResNet 和 Transformer 串起神经网络复兴。",
        "prompt": """
一张适合 AI 历史附录的神经网络三次浪潮时间线，主题是“神经网络为什么会兴起、低谷、再复兴”。
画面从左到右分成三次浪潮和两次低谷：1958 Perceptron、1969 XOR limitation、1986 Backprop、1994 vanishing gradient、1997 LSTM、2006 RBM/DBN、2012 AlexNet/ImageNet、2015 ResNet、2017 Transformer。
用海浪或山谷视觉隐喻表现热潮与低谷：上升是新能力，低谷是瓶颈，下一次上升是新方法加数据/算力/工程条件成熟。
风格像技术史海报和学习地图结合，清晰、戏剧感适度、适合新人理解“不是一路顺风”。
文字不是主体；标准术语保留英文，例如 Perceptron、XOR、Backprop、vanishing gradient、LSTM、RBM、AlexNet、ResNet、Transformer。其他说明用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "appendix-nlp-llm-lineage-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "NLP 到大模型谱系地图",
        "suggested_page": "docs/appendix/ai-milestones.md",
        "alt": "NLP 到大模型谱系地图：HMM、Word2Vec、Seq2Seq、Attention、Transformer、BERT、GPT 和 GPT-3 逐步连接。",
        "prompt": """
一张适合 AI 历史附录现代 NLP 章节的谱系图，主题是“文本任务如何走向大模型”。
用 6 个大节点串联：HMM、Word2Vec、Seq2Seq、Attention、Transformer、BERT/GPT。
风格像简洁技术谱系图，突出 token、embedding、context 和 pretraining 的传承关系。
文字不是主体；标准术语保留英文。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "appendix-agent-system-lineage-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "对齐 Agent 与系统化主线地图",
        "suggested_page": "docs/appendix/ai-milestones.md",
        "alt": "对齐 Agent 与系统化主线地图：RLHF、tool use、RAG、Agent planning、evaluation 和 safety 共同把大模型接入系统。",
        "prompt": """
一张适合 AI 历史附录 Agent 系统章节的主线图，主题是“大模型从生成走向系统执行”。
用 5 个大模块组成闭环：alignment、tool use、RAG grounding、Agent planning、evaluation/safety。
风格像系统控制环，清晰、专业、少字。
文字不是主体；标准术语保留英文。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
""".strip(),
    },
    {
        "filename": "appendix-llm-to-agent-evolution-timeline.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "大模型到 Agent 工程化演化图",
        "suggested_page": "docs/appendix/ai-milestones.md",
        "alt": "大模型到 Agent 工程化演化图：pretraining、instruction tuning、RLHF、RAG、function calling、tool use、planning、evaluation 和 safety 逐步把模型接入系统。",
        "prompt": """
一张适合 AI 历史附录的大模型到 Agent 演化图，主题是“从会续写到能被约束地做事”。
画面从左到右只展示四张大卡片：pretraining、instruction tuning、RAG/tools、Agent system。
四张卡片分别用简单图标表示：token 训练流水线、偏好反馈、知识库加工具调用、带 trace 和 safety guardrail 的系统控制台。
重点表达从模型能力到工程系统的升级，构图干净、箭头少、节点大、留白充足。
文字不是主体；只保留少量大号英文标签 pretraining、instruction tuning、RAG/tools、Agent system、trace、safety。不要密集小字、不要乱码、不要真实品牌 logo。
""".strip(),
    },
    {
        "filename": "appendix-multimodal-aigc-lineage-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "多模态与 AIGC 谱系地图",
        "suggested_page": "docs/appendix/ai-milestones.md",
        "alt": "多模态与 AIGC 谱系地图：CNN、CLIP、Diffusion、Stable Diffusion、video generation 和 multimodal agents 共同扩展输入输出媒介。",
        "prompt": """
一张适合 AI 历史附录多模态章节的谱系地图，主题是“AI 从文字扩展到图像、语音、视频和创作工作流”。
用 6 个大节点串联：CNN vision、CLIP、Diffusion、Stable Diffusion、video generation、multimodal workflow。
风格像清晰的多模态路线图，强调理解、生成、编辑、审核和交付。
文字不是主体；标准术语保留英文。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
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
        "size": "1024x1792",
        "quality": "high",
        "title": "AI API 请求响应竖版漫画",
        "suggested_page": "docs/ch02-python/ch03-projects/04-ai-api-experience.md",
        "alt": "AI API 请求响应竖版漫画：用户输入通过 Python OpenAI SDK 调用 Responses API，模型返回 output_text，程序处理错误、成本和展示。",
        "prompt": """
制作一页 9:16 竖版中文科普漫画，主题：“一次 OpenAI Responses API 调用发生了什么”。
像真正的课程漫画页，适合 Python 新手，6 个清晰分镜，文字短句，气泡、代码卡片、小黑板和错误标签分散排版，不要把文字堆在一起。

第 1 格：学习者输入问题。气泡：“请用一句话介绍 Python。”标签：“用户输入”。
第 2 格：Python 程序创建 OpenAI 客户端。代码卡片只写关键行：client = OpenAI()。旁白：“SDK 会读取 OPENAI_API_KEY”。
第 3 格：请求卡片展示 Responses API：model、instructions、input、max_output_tokens。小黑板：“API = 程序之间的调用门”。
第 4 格：网络箭头把请求送到 AI 服务，旁边有锁和计时器。标签：“保护 Key”“设置超时”“记录日志”。
第 5 格：模型返回 response.output_text，并显示 usage：input tokens、output tokens、cost。旁白：“拿到文本，也要看成本”。
第 6 格：程序把结果展示给用户；旁边红色小卡片：“错误处理：超时、限流、Key 错误、网络失败”。箭头：“失败要提示、重试或退出”。

底部总结：“历史意义/工程意义：你不是在训练模型，而是在用 Python 安全地调用一个已经训练好的模型服务。”
所有文字必须是简体中文，OpenAI、Responses API、SDK、Token、output_text 可以保留英文术语。不要乱码，不要真实品牌 logo，不要水印。
""".strip(),
    },
    {
        "filename": "ch02-ai-api-request-response-en.png",
        "size": "1024x1792",
        "quality": "high",
        "title": "AI API request-response vertical comic",
        "suggested_page": "docs/ch02-python/ch03-projects/04-ai-api-experience.md",
        "alt": "Vertical comic explaining a Python OpenAI SDK call to the Responses API, from user input to output_text, usage, cost, and error handling.",
        "prompt": """
Create one 9:16 vertical English educational comic page titled: "One OpenAI Responses API Call, Step by Step".
Make it look like a real beginner-friendly course comic page, with 6 clean panels, short labels, speech bubbles, code cards, a mini blackboard, arrows, and small engineering warning tags. Do not crowd the page.

Panel 1: A learner types a prompt. Speech bubble: "Introduce Python in one sentence." Label: "User input".
Panel 2: A Python program creates the OpenAI client. Code card: client = OpenAI(). Caption: "The SDK reads OPENAI_API_KEY."
Panel 3: A request card for Responses API shows: model, instructions, input, max_output_tokens. Mini blackboard: "API = a doorway between programs."
Panel 4: A secure network arrow sends the request to the AI service. Add a lock and timer. Tags: "Protect the key", "Set timeout", "Log safely".
Panel 5: The model returns response.output_text and usage: input tokens, output tokens, cost. Caption: "Get text, then check usage."
Panel 6: The app shows the answer to the user. Red small card: "Handle errors: timeout, rate limit, bad key, network failure." Arrow: "Fail gracefully: explain, retry, or exit."

Bottom summary: "Engineering meaning: you are not training a model; you are safely calling a trained model service from Python."
All text must be clear English. Keep OpenAI, Responses API, SDK, Token, output_text as technical terms. No gibberish, no real logos, no watermark.
""".strip(),
    },
    {
        "filename": "ch02-ai-api-request-response-ja.png",
        "size": "1024x1792",
        "quality": "high",
        "title": "AI API リクエスト・レスポンス縦型マンガ",
        "suggested_page": "i18n/ja/docusaurus-plugin-content-docs/current/ch02-python/ch03-projects/04-ai-api-experience.md",
        "alt": "Python OpenAI SDK から Responses API を呼び、output_text、usage、コスト、エラー処理までを説明する日本語縦型マンガ。",
        "prompt": """
9:16 縦長の日本語教育マンガページを作成。タイトル：「OpenAI Responses API 呼び出しの流れ」。
Python 初心者向けの本物の講義マンガのように、6コマ構成、短いラベル、吹き出し、コードカード、小黒板、矢印、注意ラベルを使う。文字を詰め込まない。

1コマ目：学習者が質問を入力。吹き出し：「Python を一文で紹介して」。ラベル：「ユーザー入力」。
2コマ目：Python プログラムが OpenAI クライアントを作る。コードカード：client = OpenAI()。説明：「SDK が OPENAI_API_KEY を読む」。
3コマ目：Responses API のリクエストカード：model、instructions、input、max_output_tokens。小黒板：「API = プログラム同士の入り口」。
4コマ目：安全なネットワーク矢印で AI サービスへ送る。鍵とタイマーを描く。ラベル：「Key を守る」「timeout を設定」「安全にログ」。
5コマ目：モデルが response.output_text と usage を返す：input tokens、output tokens、cost。説明：「返答と使用量を確認」。
6コマ目：アプリが回答を表示。赤いカード：「エラー処理：timeout、rate limit、Key エラー、通信失敗」。矢印：「分かりやすく失敗する、再試行する、終了する」。

下部まとめ：「工学的な意味：モデルを訓練しているのではなく、Python から訓練済みモデルサービスを安全に呼び出している。」
文字は自然な日本語。OpenAI、Responses API、SDK、Token、output_text は技術用語として残す。文字化け、実在ロゴ、透かしは禁止。
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
]

IMAGE_JOBS.extend(
    [
    ]
)


IMAGE_JOBS.extend(
    [
    ]
)

P0_REMAKE_IMAGE_JOBS = [
    {
        "filename": "ch07-token-to-answer-lifecycle.png",
        "size": "1024x1792",
        "quality": "high",
        "title": "Token 到答案生命周期",
        "suggested_page": "docs/ch07-llm-principles/ch01-nlp-crash/00-roadmap.md",
        "alt": "一个 token 如何参与生成答案：输入 token、上下文窗口、Transformer 层、logits 概率、解码、校验。",
        "prompt": """
竖版 9:16 中文教学插画，像课堂分镜和模型工作台，不要白底圆角框。
标题写清楚：“一个 token 如何变成答案”。
画面中学习者看着大屏：左上是输入 token 卡片进入 context window，屏幕中间是多层 Transformer 像透明机器堆叠，右侧出现 logits 能量柱和概率候选，下方是 decoder 逐字选词，最后进入 answer check 小面板。
需要清楚编号和文字锚点：
① 输入 token：文字切成模型单位
② 上下文窗口：历史和证据一起占预算
③ Transformer 层：混合上下文信息
④ logits / 概率：给下一个 token 打分
⑤ 解码：选择并拼出答案
⑥ 校验：检查格式与证据
底部结论：“生成答案不是一次吐出整段，而是一步步选下一个 token。”
文字要大、自然、可读；保留 token、context window、Transformer、logits、decoder。不要密集小字、乱码、水印、真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch09-agent-execution-loop.png",
        "size": "1024x1792",
        "quality": "high",
        "title": "Agent 执行回合",
        "suggested_page": "docs/ch09-agent/index.md",
        "alt": "Agent 执行闭环：用户目标、状态记忆、计划、工具调用、观察结果、更新状态、停止条件、trace log。",
        "prompt": """
竖版 9:16 中文教学插画，必须像一页课堂漫画/控制室分镜，绝对不要白底、不要圆角框、不要纵向表单、不要普通流程图。
标题写清楚：“Agent 一轮怎么执行”。
画面是一间 Agent 控制室：学习者坐在中央大屏前监督一个小机器人 Agent。大屏不是列表，而是分成 6 个场景小分镜：
左上：用户把目标贴到任务板；左中：状态/记忆抽屉打开；中间：Agent 在白板上画下一步计划；右中：工具机械臂拿着参数去调用 API；右下：工具结果作为观察卡片回到桌面；底部：Agent 更新状态，旁边有停止条件信号灯和 trace log 记录本。
每个场景必须有编号和大号短标签，并配一句很短解释：
① 目标：要完成什么
② 记忆：已有信息
③ 计划：下一步动作
④ 工具：带参数执行
⑤ 观察：读回结果
⑥ 更新：写回状态
底部结论：“直到完成、失败或超限，trace log 保留每一步。”
文字清楚可读；保留 Agent、tool、trace log。不要密集小字、乱码、水印、真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch10-vision-pipeline-loop.png",
        "size": "1024x1792",
        "quality": "high",
        "title": "视觉流水线与失败复盘",
        "suggested_page": "docs/ch10-computer-vision/index.md",
        "alt": "视觉项目流水线：raw image、crop/preprocess、模型输出、box/mask/OCR、人工 review、失败复盘。",
        "prompt": """
竖版 9:16 中文教学插画，像电脑视觉实验台的分步骤操作，不要白底圆角框。
标题写清楚：“一张图片如何进入视觉系统”。
画面从上到下是一张真实街景/商品照片被处理：原图放入工作台，裁剪和预处理在屏幕上发生，模型给出分类、检测框、mask 或 OCR 结果，人工 review 用红笔圈出错误，最后把失败样本放进复盘板。
需要编号和短说明：
① Raw image：保留原始输入
② Crop / preprocess：尺寸、颜色、裁剪
③ Model output：模型给出预测
④ Box / mask / OCR：输出形态不同
⑤ Human review：检查错框、错字、漏检
⑥ Error review：把失败样本用于下一轮
底部结论：“视觉项目要保存原图、预测图和失败原因，不能只看一个分数。”
文字清楚可读；保留 raw image、crop、preprocess、box、mask、OCR。不要密集小字、乱码、水印、真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch11-text-to-task-pipeline.png",
        "size": "1024x1792",
        "quality": "high",
        "title": "文本到 NLP 任务",
        "suggested_page": "docs/ch11-nlp/index.md",
        "alt": "文本进入 NLP 系统：raw document、tokens、embedding、annotation、extraction、summary、retrieval QA、evaluation。",
        "prompt": """
竖版 9:16 中文教学插画，像“文本进工厂”的课堂工作台，不要白底流程框。
标题写清楚：“文本如何变成 NLP 任务输出”。
画面上方是一叠 raw document、聊天和日志纸张；中间学习者把文本切成 tokens，再放入 embedding 机器变成向量珠；下方分成几个实际产物：标注过的实体、抽取表格、摘要卡片、带来源的 QA 答案、评估记录。
需要编号和短说明：
① Raw document：先看文本来源
② Tokens：切成可处理单位
③ Embedding：变成数字表示
④ Annotation：标注标签和字段
⑤ Extraction：抽取结构化信息
⑥ Summary：压缩成摘要
⑦ Retrieval QA：回答必须带来源
⑧ Evaluation：用错例检查质量
底部结论：“先定义输出形态，再选择分类、抽取、摘要或问答。”
文字清楚可读；保留 raw document、tokens、embedding、QA。不要密集小字、乱码、水印、真实品牌 logo。
""".strip(),
    },
    {
        "filename": "ch12-multimodal-workflow-loop.png",
        "size": "1024x1792",
        "quality": "high",
        "title": "多模态工作流",
        "suggested_page": "docs/ch12-multimodal/index.md",
        "alt": "多模态工作流：text/PDF、image/screenshot、audio waveform、video frames 进入对齐与理解模块，输出 human review 和 report package。",
        "prompt": """
竖版 9:16 中文教学插画，像多模态项目控制台和分镜工作台，不要白底圆角框。
标题写清楚：“多模态资料如何合成一个结果”。
画面上方四种输入同时进入工作台：text/PDF 文件、image/screenshot、audio waveform、video frames；中间是对齐与理解模块，把时间线、画面区域、文本证据连起来；下方是 human review 审核区和 report package 交付包。
需要编号和短说明：
① Text / PDF：提供文字证据
② Image / screenshot：提供视觉区域
③ Audio waveform：提供声音线索
④ Video frames：提供时间变化
⑤ Align：把时间、画面、文本对齐
⑥ Understand：生成跨模态理解
⑦ Human review：人工确认风险
⑧ Report package：输出可交付证据包
底部结论：“多模态不是把文件堆一起，而是对齐、理解、审核后再交付。”
文字清楚可读；保留 Text/PDF、screenshot、waveform、video frames、Align。不要密集小字、乱码、水印、真实品牌 logo。
""".strip(),
    },
]


def remake_items(*pairs: tuple[str, str]) -> list[dict[str, str]]:
    return [{"label": label, "detail": detail} for label, detail in pairs]


def remake_callouts(
    boxes: list[tuple[float, float, float, float]],
    targets: list[tuple[float, float]],
) -> list[dict[str, Any]]:
    return [{"box": box, "target": target} for box, target in zip(boxes, targets)]


def remake_scene_prompt(scene: str) -> str:
    return f"""
Create a vertical 9:16 bitmap teaching illustration for an AI full-stack course.
The image must feel like a practical classroom comic or production workbench, not a white SVG diagram, not a slide, and not stacked rounded boxes.
Use a rich illustrated scene with people, screens, tools, data artifacts, arrows, and clear spatial flow. The scene must expose distinct visual regions for each concept so labels and arrows can be anchored to the right object.
Do not include readable text inside the generated artwork, because exact multilingual labels will be overlaid by the course script.
Leave some calm negative space near the top and side margins for anchored callouts. Do not reserve a large unrelated text block at the bottom.
Avoid tiny text, gibberish, watermarks, real brand logos, plain white backgrounds, and generic flowchart boxes.

Scene plan:
{scene}
""".strip()


def remake_comic_panel_prompt(panels: list[str]) -> str:
    panel_text = "\n".join(f"Panel {index}: {panel}" for index, panel in enumerate(panels, start=1))
    return f"""
Create a vertical 9:16 bitmap teaching comic page for an AI full-stack course.
Make it look like a real illustrated classroom comic, not SVG, not a whiteboard flowchart, not a slide, not stacked rounded UI cards.
Use exactly six clear comic panels arranged as a 2-column by 3-row page. Each panel must show one concrete action, with arrows or motion between panels.
Reserve a small calm caption strip near the bottom of each panel, but do not put readable text in the generated artwork. The course script will place exact multilingual captions.
Use rich visual storytelling: hands moving cards, machines processing data, screens, evidence sheets, gauges, and visible cause-and-effect. Avoid generic abstract boxes.
No tiny text, no gibberish, no watermarks, no real brand logos.

Panel plan:
{panel_text}
""".strip()


def remake_fully_generated_comic_prompt(
    *,
    title: str,
    subtitle: str,
    panels: list[tuple[str, str, str]],
    footer: str,
) -> str:
    panel_text = "\n".join(
        f"Panel {index}: visible caption exactly \"{label}\" and short line exactly \"{detail}\". Scene: {scene}"
        for index, (label, detail, scene) in enumerate(panels, start=1)
    )
    return f"""
Create one complete vertical 9:16 teaching image as a finished bitmap. Do not rely on any post-processing, captions, labels, overlays, SVG, markdown, or external text.
The image itself must contain all teaching text, drawn naturally as part of the comic page.
Style: high-quality illustrated classroom comic, six panels arranged 2 columns x 3 rows, rich visual storytelling, practical course illustration, not a whiteboard flowchart, not UI cards, not pasted text boxes.

Visible title at the top must be exactly:
{title}

Visible subtitle under the title must be exactly:
{subtitle}

Each panel must show the concept visually and include its own caption inside the panel, close to the visual action:
{panel_text}

Visible footer at the bottom must be exactly:
{footer}

Important: text must be large, clean, readable, and integrated with the panel design. The caption must sit near the object it explains. Avoid tiny text, gibberish, watermark, real brand logo, blank white background, and SVG-like diagram style.
""".strip()


def register_remake_job(
    *,
    filename: str,
    title: str,
    suggested_page: str,
    alt: str,
    scene: str,
    subtitle: str,
    items: list[dict[str, str]],
    footer: str,
    callouts: list[dict[str, Any]] | None = None,
    prompt: str | None = None,
    overlay_style: str | None = None,
    generated_only: bool = False,
    background_key: str | None = None,
) -> None:
    overlay: dict[str, Any] | None = None
    if not generated_only:
        overlay = {
            "title": title,
            "subtitle": subtitle,
            "items": items,
            "footer": footer,
        }
        if overlay_style:
            overlay["style"] = overlay_style
        if callouts:
            overlay["style"] = "callouts"
            overlay["callouts"] = callouts

    remake_data = {
        "filename": filename,
        "size": "1024x1792",
        "quality": "high",
        "title": title,
        "suggested_page": suggested_page,
        "alt": alt,
        "prompt": prompt or remake_scene_prompt(scene),
        "overlay": overlay,
    }
    if background_key:
        remake_data["background_key"] = background_key

    for job in [*IMAGE_JOBS, *P0_REMAKE_IMAGE_JOBS]:
        if str(job.get("filename")) == filename:
            job.update(remake_data)
            return
    P0_REMAKE_IMAGE_JOBS.append(remake_data)


TOKEN_LIFECYCLE_CALLOUTS = remake_callouts(
    boxes=[
        (0.04, 0.30, 0.33, 0.085),
        (0.04, 0.46, 0.33, 0.085),
        (0.40, 0.31, 0.33, 0.085),
        (0.63, 0.48, 0.33, 0.085),
        (0.07, 0.73, 0.34, 0.085),
        (0.54, 0.73, 0.39, 0.085),
    ],
    targets=[
        (0.55, 0.29),
        (0.32, 0.57),
        (0.55, 0.50),
        (0.88, 0.49),
        (0.63, 0.67),
        (0.85, 0.82),
    ],
)

TOKEN_LIFECYCLE_PANEL_PROMPT = remake_comic_panel_prompt(
    [
        "A hand feeds colorful prompt token cards from a text tray into the model entrance. The action should clearly show text becoming small token units.",
        "A transparent context window shelf holds recent conversation cards and evidence documents together, with a visible limited space or budget gauge.",
        "Token cards travel through stacked transformer machinery; attention-like light beams connect distant cards to show context mixing.",
        "A probability scoreboard shows several candidate next tokens as different-height glowing bars; one candidate is highlighted but not yet chosen.",
        "A decoder mechanism picks the highlighted token tile and attaches it to a growing answer ribbon, showing one-step-at-a-time generation.",
        "A reviewer desk checks the answer ribbon against evidence sheets and a format checklist, with pass/warning marks visible as icons only.",
    ]
)

TOKEN_LIFECYCLE_GENERATED_PROMPT_EN = remake_fully_generated_comic_prompt(
    title="How One Token Becomes an Answer",
    subtitle="Generation repeats one next-token decision at a time.",
    panels=[
        ("1 Prompt tokens", "Text becomes model units.", "A hand feeds colorful text fragments into a token tray; the fragments visibly become small token tiles."),
        ("2 Context window", "History and evidence share budget.", "Conversation cards and evidence sheets sit together inside a limited transparent context shelf with a budget gauge."),
        ("3 Transformer layers", "Context is mixed layer by layer.", "Token tiles move through stacked transformer machinery; light beams connect related tokens across layers."),
        ("4 Logits / probability", "Candidates receive scores.", "Several candidate next-token tiles stand beside glowing probability bars with one clearly highest bar."),
        ("5 Decoder", "One token is chosen and appended.", "A mechanical picker selects the highest-scored token tile and attaches it to a growing answer ribbon."),
        ("6 Answer check", "Format and evidence are reviewed.", "A reviewer compares the answer ribbon with evidence sheets and a checklist before approving."),
    ],
    footer="Think loop: score, choose, append, check.",
)

TOKEN_LIFECYCLE_GENERATED_PROMPT_ZH = remake_fully_generated_comic_prompt(
    title="一个 token 如何变成答案",
    subtitle="生成答案不是一次吐完整段，而是反复决定下一个 token。",
    panels=[
        ("1 输入 token", "文字先切成模型单位。", "一只手把彩色文本碎片送入 token 托盘，碎片变成一个个小 token 方块。"),
        ("2 上下文窗口", "历史和证据共同占预算。", "对话卡片和证据纸张一起放进透明的 context window 架子，旁边有可见预算刻度。"),
        ("3 Transformer 层", "逐层混合上下文信息。", "token 方块穿过多层透明机器，光线把相关 token 在不同层之间连接起来。"),
        ("4 logits / 概率", "给候选 token 打分。", "几个候选 token 站在概率柱旁边，其中一个柱子最高并发光。"),
        ("5 decoder", "选中一个并接到答案后面。", "机械手选中最高分 token，把它接到正在变长的答案纸带后面。"),
        ("6 答案校验", "检查格式、事实和证据。", "学习者把答案纸带与证据页和格式清单对照，确认后打勾。"),
    ],
    footer="记住循环：打分、选择、追加、检查。",
)

TOKEN_LIFECYCLE_GENERATED_PROMPT_JA = remake_fully_generated_comic_prompt(
    title="1つの token が答えになるまで",
    subtitle="回答は一気に出るのではなく、次の token を何度も選ぶ。",
    panels=[
        ("1 入力 token", "文章をモデル単位に分ける。", "手が色付きの文章片を token トレイへ入れ、文章片が小さな token タイルに変わる。"),
        ("2 context window", "履歴と根拠が同じ予算を使う。", "会話カードと根拠資料が透明な context window の棚に入り、横に予算ゲージが見える。"),
        ("3 Transformer 層", "文脈情報を段階的に混ぜる。", "token タイルが積み重なった transformer 機械を通り、関連 token が光の線で結ばれる。"),
        ("4 logits / 確率", "候補 token に点数を付ける。", "候補 token が確率バーの横に並び、最も高いバーが光っている。"),
        ("5 decoder", "1つ選んで回答へ追加する。", "機械のアームが最高点の token を選び、伸びていく回答リボンに追加する。"),
        ("6 回答チェック", "形式、事実、根拠を確認する。", "学習者が回答リボンを根拠資料とチェックリストで確認し、合格マークを付ける。"),
    ],
    footer="ループで考える：採点、選択、追加、確認。",
)


register_remake_job(
    filename="ch07-token-to-answer-lifecycle-en.png",
    title="How One Token Becomes an Answer",
    suggested_page="docs/ch07-llm-principles/index.md",
    alt="A token generation lifecycle: prompt tokens, context window, transformer layers, logits and probabilities, decoder, and answer check.",
    scene="A learner watches a model workstation. Prompt token cards enter a context window, pass through transparent transformer layers, produce glowing probability bars, then a decoder assembles the next answer token while a review panel checks evidence and format.",
    subtitle="Generation is a repeated next-token decision, not one instant paragraph.",
    items=remake_items(
        ("Prompt tokens", "Text is split into model units."),
        ("Context window", "History and evidence share budget."),
        ("Transformer layers", "Layers mix signals from the context."),
        ("Logits / probability", "Candidates receive next-token scores."),
        ("Decoder", "One token is chosen and appended."),
        ("Answer check", "Format and evidence are reviewed."),
    ),
    footer="Read token generation as a loop: score, choose, append, check.",
    prompt=TOKEN_LIFECYCLE_GENERATED_PROMPT_EN,
    generated_only=True,
)

register_remake_job(
    filename="ch07-token-to-answer-lifecycle.png",
    title="一个 token 如何变成答案",
    suggested_page="i18n/zh-Hans/docusaurus-plugin-content-docs/current/ch07-llm-principles/index.md",
    alt="Token 生成生命周期：输入 token、上下文窗口、Transformer 层、logits 概率、decoder 和答案校验。",
    scene="学习者站在模型工作台前。输入 token 卡片进入上下文窗口，穿过透明的 Transformer 层，右侧出现概率能量柱，下方 decoder 逐步拼出答案，旁边有证据和格式校验面板。",
    subtitle="生成答案不是一次吐完整段，而是反复决定下一个 token。",
    items=remake_items(
        ("输入 token", "文字先被切成模型单位。"),
        ("上下文窗口", "历史和证据共同占预算。"),
        ("Transformer 层", "逐层混合上下文信息。"),
        ("logits / 概率", "给候选 token 打分。"),
        ("decoder", "选中一个并接到答案后面。"),
        ("答案校验", "检查格式、事实和证据。"),
    ),
    footer="把生成看成循环：打分、选择、追加、检查。",
    prompt=TOKEN_LIFECYCLE_GENERATED_PROMPT_ZH,
    generated_only=True,
)

register_remake_job(
    filename="ch07-token-to-answer-lifecycle-ja.png",
    title="1つの token が答えになるまで",
    suggested_page="i18n/ja/docusaurus-plugin-content-docs/current/ch07-llm-principles/index.md",
    alt="Token 生成の流れ：入力 token、context window、Transformer 層、logits と確率、decoder、回答チェック。",
    scene="学習者がモデル実験台を見ている。入力 token カードが context window に入り、透明な Transformer 層を通り、右側に確率バーが光り、decoder が次の token を少しずつ足し、証拠と形式の確認パネルがある。",
    subtitle="回答は一気に出るのではなく、次の token を何度も選ぶ。",
    items=remake_items(
        ("入力 token", "文章をモデル単位に分ける。"),
        ("context window", "履歴と根拠が同じ予算を使う。"),
        ("Transformer 層", "文脈情報を段階的に混ぜる。"),
        ("logits / 確率", "候補 token に点数を付ける。"),
        ("decoder", "1つ選んで回答へ追加する。"),
        ("回答チェック", "形式、事実、根拠を確認する。"),
    ),
    footer="生成は「採点、選択、追加、確認」のループとして読む。",
    prompt=TOKEN_LIFECYCLE_GENERATED_PROMPT_JA,
    generated_only=True,
)

register_remake_job(
    filename="ch08-async-concurrency-semaphore-timeout-map-en.png",
    title="Controlled Concurrency: Semaphore + Timeout",
    suggested_page="docs/ch08-rag/ch04-engineering/01-async-programming.md",
    alt="Controlled async concurrency: queue tasks, pass through a semaphore gate, call APIs, cancel slow work with timeout, summarize errors, and return stable results.",
    scene="A backend dispatch room. On the left, uncontrolled requests crowd an API door with red alarms. In the main scene, tasks wait in a queue, pass through a semaphore gate, call an API with only a few workers, timeout slow jobs, and summarize results on a dashboard.",
    subtitle="Async is not unlimited parallelism; it is controlled pressure.",
    items=remake_items(
        ("Task queue", "Collect work before sending it."),
        ("Semaphore", "Limit work in flight."),
        ("API call", "Call upstream with bounded pressure."),
        ("Timeout", "Cancel tasks that hang too long."),
        ("Error summary", "Group failures by cause."),
        ("Stable result", "Return partial success clearly."),
    ),
    footer="Concurrency is useful only when the failure mode is also designed.",
)

register_remake_job(
    filename="ch08-async-concurrency-semaphore-timeout-map.png",
    title="并发请求如何不把服务打爆",
    suggested_page="i18n/zh-Hans/docusaurus-plugin-content-docs/current/ch08-rag/ch04-engineering/01-async-programming.md",
    alt="受控异步并发：任务排队、Semaphore 限流、API 调用、Timeout 取消慢任务、错误汇总和稳定结果。",
    scene="后端请求调度室。左侧小场景是无限并发冲向 API 门口并触发红色报警；主画面里任务先排队，通过 Semaphore 限流闸门，少量 worker 调用 API，慢任务被 Timeout 取消，结果汇总到看板。",
    subtitle="异步不是无限并发，而是把压力控制住。",
    items=remake_items(
        ("任务队列", "先收集任务，不直接冲上游。"),
        ("Semaphore", "限制同时进行的数量。"),
        ("API 调用", "用可控压力访问服务。"),
        ("Timeout", "慢任务及时取消。"),
        ("错误汇总", "按原因记录失败。"),
        ("稳定结果", "清楚返回成功与失败。"),
    ),
    footer="并发设计必须同时设计失败时怎么收口。",
)

register_remake_job(
    filename="ch08-async-concurrency-semaphore-timeout-map-ja.png",
    title="同時実行を Semaphore と Timeout で守る",
    suggested_page="i18n/ja/docusaurus-plugin-content-docs/current/ch08-rag/ch04-engineering/01-async-programming.md",
    alt="制御された非同期処理：タスクキュー、Semaphore、API 呼び出し、Timeout、エラー集計、安定した結果。",
    scene="バックエンドのリクエスト管制室。左側では無制限の同時実行が API の入口に殺到し赤い警報が出る。中央ではタスクがキューで待ち、Semaphore のゲートを通り、少数の worker が API を呼び、遅い処理は Timeout で止まり、結果がダッシュボードに集計される。",
    subtitle="async は無制限並列ではなく、圧力を制御する仕組み。",
    items=remake_items(
        ("タスクキュー", "仕事を先に並べる。"),
        ("Semaphore", "同時実行数を制限する。"),
        ("API 呼び出し", "上流へ安全な圧力で送る。"),
        ("Timeout", "遅すぎる処理を止める。"),
        ("エラー集計", "失敗原因をまとめる。"),
        ("安定した結果", "成功と失敗を明確に返す。"),
    ),
    footer="同時実行は、失敗時の閉じ方まで設計して初めて安定する。",
)

register_remake_job(
    filename="ch08-chunk-size-overlap-tradeoff-map-en.png",
    title="Chunk Size and Overlap Trade-off",
    suggested_page="docs/ch08-rag/ch01-rag/02-document-processing.md",
    alt="Chunking trade-off: large chunks keep context but add noise, small chunks improve precision but may split facts, overlap protects boundaries with extra cost.",
    scene="A RAG document workshop. A long policy document is cut with three tools: large chunks, tiny chunks, and overlapped chunks. Evidence cards show recall, precision, boundary facts, metadata, and token cost as physical gauges.",
    subtitle="Chunking is a retrieval design choice, not a file-splitting chore.",
    items=remake_items(
        ("Large chunks", "More context, more noise."),
        ("Small chunks", "Sharper matches, facts may split."),
        ("Overlap", "Protects boundary evidence."),
        ("Metadata", "Keeps source and section usable."),
        ("Cost", "More text means more tokens."),
        ("Audit", "Check citations before indexing."),
    ),
    footer="Pick chunk size by evidence quality, not by a magic number.",
)

register_remake_job(
    filename="ch08-chunk-size-overlap-tradeoff-map.png",
    title="Chunk 大小与 overlap 取舍",
    suggested_page="i18n/zh-Hans/docusaurus-plugin-content-docs/current/ch08-rag/ch01-rag/02-document-processing.md",
    alt="切块取舍：大 chunk 保留上下文但噪声多，小 chunk 更精确但可能切断事实，overlap 保护边界但增加成本。",
    scene="RAG 文档处理工作台。一份长政策文档被三种工具切分：大 chunk、小 chunk、带 overlap 的 chunk。旁边用证据卡和仪表展示 recall、precision、边界事实、metadata 与 token cost。",
    subtitle="切块不是机械分文件，而是在设计检索质量。",
    items=remake_items(
        ("大 chunk", "上下文多，但噪声也多。"),
        ("小 chunk", "匹配精确，但事实可能被切断。"),
        ("overlap", "保护跨边界的信息。"),
        ("metadata", "保留来源、章节和页码。"),
        ("成本", "文本越多，token 越多。"),
        ("审计", "入库前先检查引用是否可追。"),
    ),
    footer="按证据质量选 chunk，不要迷信固定数字。",
)

register_remake_job(
    filename="ch08-chunk-size-overlap-tradeoff-map-ja.png",
    title="Chunk サイズと overlap の取捨選択",
    suggested_page="i18n/ja/docusaurus-plugin-content-docs/current/ch08-rag/ch01-rag/02-document-processing.md",
    alt="Chunking の取捨選択：大きい chunk は文脈を保つがノイズが増え、小さい chunk は精度が上がるが事実を分断しやすく、overlap は境界を守るがコストが増える。",
    scene="RAG の文書処理ワークベンチ。長い規約文書を、大きい chunk、小さい chunk、overlap 付き chunk の三つの道具で切る。証拠カードとメーターが recall、precision、境界情報、metadata、token cost を示す。",
    subtitle="chunking は単なる分割ではなく、検索品質の設計。",
    items=remake_items(
        ("大きい chunk", "文脈は残るがノイズも増える。"),
        ("小さい chunk", "精密だが事実が切れやすい。"),
        ("overlap", "境界の情報を守る。"),
        ("metadata", "出典、章、ページを残す。"),
        ("コスト", "文字量が token 数を増やす。"),
        ("監査", "索引化前に引用を確認する。"),
    ),
    footer="固定値ではなく、証拠の見つかり方で chunk を決める。",
)

register_remake_job(
    filename="ch08-llm-api-robust-client-loop-map-en.png",
    title="Robust LLM API Client",
    suggested_page="docs/ch08-rag/ch04-engineering/02-llm-api-design.md",
    alt="A robust LLM API client handles request setup, retry, backoff, parsing, usage logging, request id, and explainable errors.",
    scene="A production API client control room. A request card enters a sender, recoverable failures go through retry and backoff timers, JSON and text responses are parsed, token usage and latency are logged, and final output returns as either normalized result or explainable error.",
    subtitle="Production clients must succeed cleanly and fail clearly.",
    items=remake_items(
        ("Request", "Build the provider call."),
        ("Retry", "Retry only recoverable failures."),
        ("Backoff", "Wait longer after repeated failure."),
        ("Parse", "Normalize JSON or text."),
        ("Usage log", "Track tokens, cost, latency."),
        ("Error return", "Return a useful failure reason."),
    ),
    footer="A stable client is part of the product, not just wrapper code.",
)

register_remake_job(
    filename="ch08-llm-api-robust-client-loop-map-ja.png",
    title="堅牢な LLM API クライアント",
    suggested_page="i18n/ja/docusaurus-plugin-content-docs/current/ch08-rag/ch04-engineering/02-llm-api-design.md",
    alt="堅牢な LLM API クライアント：request、retry、backoff、parse、usage log、request id、説明可能な error return。",
    scene="本番 API クライアントの管制室。リクエストカードが送信台に入り、復旧可能な失敗だけ retry と backoff タイマーを通り、JSON と text が parse され、token 使用量と latency が記録され、最後は正規化された結果か説明可能な error として返る。",
    subtitle="本番クライアントは、成功だけでなく失敗も読みやすく返す。",
    items=remake_items(
        ("Request", "provider への呼び出しを作る。"),
        ("Retry", "復旧可能な失敗だけ再試行。"),
        ("Backoff", "失敗後は間隔を広げる。"),
        ("Parse", "JSON / text を正規化する。"),
        ("Usage log", "tokens、費用、latency を記録。"),
        ("Error return", "失敗理由を分かる形で返す。"),
    ),
    footer="安定したクライアントは wrapper ではなく製品品質の一部。",
)

register_remake_job(
    filename="ch08-observability-logs-metrics-trace-map-en.png",
    title="Logs, Metrics, Trace: Read Together",
    suggested_page="docs/ch08-rag/ch04-engineering/03-logging-monitoring.md",
    alt="Observability for RAG systems: logs explain what happened, metrics show whether it is abnormal, traces show where the request went.",
    scene="A RAG incident review room. One bad answer is pinned at the top, and three investigation lenses appear: logs as event notes, metrics as changing gauges, and trace as a glowing path through retrieval, generation, and response.",
    subtitle="A bad answer needs all three signals, not one screenshot.",
    items=remake_items(
        ("Logs", "What happened in this request."),
        ("Metrics", "Whether the system is abnormal."),
        ("Trace", "Where the request traveled."),
        ("request_id", "Connect all evidence."),
        ("Alert", "Fire only on useful symptoms."),
        ("Fix", "Use evidence before guessing."),
    ),
    footer="Debug one request with logs + metrics + trace together.",
)

register_remake_job(
    filename="ch08-observability-logs-metrics-trace-map.png",
    title="Logs / Metrics / Trace 一起看",
    suggested_page="i18n/zh-Hans/docusaurus-plugin-content-docs/current/ch08-rag/ch04-engineering/03-logging-monitoring.md",
    alt="RAG 可观测性：logs 说明发生了什么，metrics 判断是否异常，trace 看到请求走过哪里。",
    scene="RAG 事故复盘室。一条坏回答被贴在上方，下面三种调查镜头同时打开：logs 是事件记录，metrics 是波动仪表，trace 是请求穿过 retrieval、generation、response 的发光路径。",
    subtitle="一个坏回答，不能只靠截图猜原因。",
    items=remake_items(
        ("logs", "这次请求发生了什么。"),
        ("metrics", "系统是否整体异常。"),
        ("trace", "请求经过了哪些环节。"),
        ("request_id", "把证据串起来。"),
        ("alert", "只对有用症状报警。"),
        ("fix", "先看证据，再改系统。"),
    ),
    footer="排查一次请求，要把 logs + metrics + trace 放在一起。",
)

register_remake_job(
    filename="ch08-observability-logs-metrics-trace-map-ja.png",
    title="Logs / Metrics / Trace を一緒に読む",
    suggested_page="i18n/ja/docusaurus-plugin-content-docs/current/ch08-rag/ch04-engineering/03-logging-monitoring.md",
    alt="RAG の観測性：logs は何が起きたか、metrics は異常かどうか、trace はリクエストの経路を示す。",
    scene="RAG 障害のレビュー室。悪い回答が上に貼られ、三つの調査レンズが開く。logs はイベントメモ、metrics は変化するメーター、trace は retrieval、generation、response を通る光の経路。",
    subtitle="悪い回答の原因は、一枚の画面だけでは分からない。",
    items=remake_items(
        ("logs", "このリクエストで何が起きたか。"),
        ("metrics", "全体が異常かどうか。"),
        ("trace", "どの経路を通ったか。"),
        ("request_id", "証拠をつなぐ ID。"),
        ("alert", "役に立つ症状だけ通知。"),
        ("fix", "推測より証拠から直す。"),
    ),
    footer="1つの要求を logs + metrics + trace で同時に見る。",
)

register_remake_job(
    filename="ch08-unified-api-provider-gateway-map-en.png",
    title="Why Use a Unified Provider Gateway",
    suggested_page="docs/ch08-rag/ch04-engineering/02-llm-api-design.md",
    alt="A provider gateway hides provider differences behind routing, adapters, fallback, normalized usage, and normalized errors.",
    scene="A split production architecture scene. The left side shows tangled direct provider wires from app code. The right side shows a clean gateway desk: business request enters gateway, routing chooses provider, adapter converts format, fallback handles failure, and usage/error logs return normalized.",
    subtitle="Keep provider differences out of business code.",
    items=remake_items(
        ("Direct wiring", "Every provider leaks into code."),
        ("Gateway", "Business uses one entry."),
        ("Routing", "Choose by cost and availability."),
        ("Adapter", "Convert provider formats."),
        ("Fallback", "Switch path on failure."),
        ("Usage log", "Normalize cost and latency."),
    ),
    footer="A gateway turns provider chaos into one stable contract.",
)

register_remake_job(
    filename="ch08-unified-api-provider-gateway-map-ja.png",
    title="統一 Provider Gateway が必要な理由",
    suggested_page="i18n/ja/docusaurus-plugin-content-docs/current/ch08-rag/ch04-engineering/02-llm-api-design.md",
    alt="Provider gateway は routing、adapter、fallback、usage log、normalized error で provider の違いを業務コードから隠す。",
    scene="本番アーキテクチャの左右比較。左は業務コードが複数 provider へ直接つながり、線が絡まっている。右は gateway の作業台で、request が入り、routing が provider を選び、adapter が形式変換し、fallback が失敗を受け、usage と error が正規化される。",
    subtitle="provider の違いを業務コードに漏らさない。",
    items=remake_items(
        ("直結", "provider ごとの差がコードに漏れる。"),
        ("Gateway", "業務側は入口を1つにする。"),
        ("Routing", "費用と可用性で選ぶ。"),
        ("Adapter", "provider 形式へ変換する。"),
        ("Fallback", "失敗時に経路を替える。"),
        ("Usage log", "費用と latency を統一する。"),
    ),
    footer="gateway は provider の混乱を安定した契約に変える。",
)

register_remake_job(
    filename="ch08-vector-record-metadata-filter-map-en.png",
    title="Vector Record + Metadata Filter",
    suggested_page="docs/ch08-rag/ch01-rag/03-vector-databases.md",
    alt="A vector record contains id, vector, text, metadata, and score; metadata filters narrow retrieval before top-k evidence is returned.",
    scene="A vector database inspection desk. A document chunk becomes a vector bead, text card, and metadata tag set. A filter gate accepts section, source, date, and page, then only matching records compete for top-k evidence.",
    subtitle="Vectors find similarity; metadata keeps retrieval controllable.",
    items=remake_items(
        ("id", "Stable record identity."),
        ("vector", "Numeric meaning representation."),
        ("text", "Evidence the reader can inspect."),
        ("metadata", "Source, section, page, date."),
        ("filter", "Narrow the search space."),
        ("top-k", "Return ranked evidence."),
    ),
    footer="Missing metadata makes filters weak and citations fragile.",
)

register_remake_job(
    filename="ch08-vector-record-metadata-filter-map.png",
    title="向量记录与 metadata 过滤",
    suggested_page="i18n/zh-Hans/docusaurus-plugin-content-docs/current/ch08-rag/ch01-rag/03-vector-databases.md",
    alt="向量记录包含 id、vector、text、metadata 和 score；metadata filter 先缩小范围，再返回 top-k 证据。",
    scene="向量数据库检查台。一个文档 chunk 变成向量珠、文本卡和 metadata 标签。过滤闸门读取 section、source、date、page，只让匹配记录进入 top-k 证据排序区。",
    subtitle="vector 负责相似度，metadata 负责可控检索。",
    items=remake_items(
        ("id", "稳定定位一条记录。"),
        ("vector", "文本含义的数字表示。"),
        ("text", "可以被人检查的证据。"),
        ("metadata", "来源、章节、页码、日期。"),
        ("filter", "先缩小搜索范围。"),
        ("top-k", "返回排序后的证据。"),
    ),
    footer="metadata 缺失，过滤会变弱，引用也会变脆。",
)

register_remake_job(
    filename="ch08-vector-record-metadata-filter-map-ja.png",
    title="Vector record と metadata filter",
    suggested_page="i18n/ja/docusaurus-plugin-content-docs/current/ch08-rag/ch01-rag/03-vector-databases.md",
    alt="vector record は id、vector、text、metadata、score を持ち、metadata filter が検索範囲を絞ってから top-k evidence を返す。",
    scene="ベクトルDBの点検机。文書 chunk が vector の粒、text カード、metadata タグへ変わる。filter ゲートが section、source、date、page を読み、一致する record だけが top-k evidence の順位付けに進む。",
    subtitle="vector は類似度、metadata は検索の制御を担当する。",
    items=remake_items(
        ("id", "record を安定して識別する。"),
        ("vector", "意味の数値表現。"),
        ("text", "人が確認できる証拠。"),
        ("metadata", "出典、章、ページ、日付。"),
        ("filter", "検索範囲を先に絞る。"),
        ("top-k", "順位付きの証拠を返す。"),
    ),
    footer="metadata が弱いと filter も引用も弱くなる。",
)

register_remake_job(
    filename="ch09-agent-boundary-map-en.png",
    title="Choose the Smallest Useful Agent Boundary",
    suggested_page="docs/ch09-agent/index.md",
    alt="Agent boundary choices: fixed workflow, RAG evidence desk, single function call, or autonomous agent loop.",
    scene="An automation design desk with four lanes. A simple checklist goes through fixed workflow, evidence lookup goes through a RAG desk, one action goes through a single function call, and only multi-step uncertain work enters an autonomous agent loop with a safety boundary.",
    subtitle="Do not use an agent when a simpler boundary works.",
    items=remake_items(
        ("Fixed workflow", "Known steps, repeatable path."),
        ("RAG desk", "Find evidence first."),
        ("Function call", "One tool action is enough."),
        ("Agent loop", "Multi-step uncertain work."),
        ("Boundary", "Limit tools, budget, and scope."),
        ("Human stop", "Ask before risky actions."),
    ),
    footer="Start with the smallest boundary that can solve the task.",
)

register_remake_job(
    filename="ch09-agent-boundary-map.png",
    title="选择最小可用 Agent 边界",
    suggested_page="i18n/zh-Hans/docusaurus-plugin-content-docs/current/ch09-agent/index.md",
    alt="Agent 边界选择：固定流程、RAG 证据台、单次 function call 或自主 Agent loop。",
    scene="自动化方案设计桌上有四条路线：固定清单走 fixed workflow，需要查证据走 RAG desk，单动作走 function call，只有多步且不确定的任务才进入带安全边界的 autonomous agent loop。",
    subtitle="能用简单边界解决，就不要一上来用 Agent。",
    items=remake_items(
        ("固定流程", "步骤已知、路径稳定。"),
        ("RAG 证据台", "先找证据再回答。"),
        ("function call", "一次工具动作就够。"),
        ("Agent loop", "多步探索才需要。"),
        ("边界", "限制工具、预算和范围。"),
        ("人工停机", "高风险动作先确认。"),
    ),
    footer="从能完成任务的最小边界开始设计。",
)

register_remake_job(
    filename="ch09-agent-boundary-map-ja.png",
    title="最小で役立つ Agent 境界を選ぶ",
    suggested_page="i18n/ja/docusaurus-plugin-content-docs/current/ch09-agent/index.md",
    alt="Agent 境界の選び方：固定 workflow、RAG evidence desk、single function call、autonomous agent loop。",
    scene="自動化設計の机に四つのレーンがある。既知の手順は fixed workflow、根拠探しは RAG desk、単発操作は function call、不確実な複数ステップだけが安全境界付きの autonomous agent loop に入る。",
    subtitle="単純な境界で足りるなら Agent にしない。",
    items=remake_items(
        ("固定 workflow", "手順が既知で再現できる。"),
        ("RAG desk", "先に根拠を探す。"),
        ("function call", "一回の道具操作で足りる。"),
        ("Agent loop", "不確実な複数ステップ用。"),
        ("境界", "tools、予算、範囲を制限。"),
        ("人の停止", "危険操作は確認する。"),
    ),
    footer="解ける最小の境界から設計する。",
)

AGENT_EXECUTION_CALLOUTS = remake_callouts(
    boxes=[
        (0.04, 0.26, 0.32, 0.085),
        (0.04, 0.43, 0.34, 0.085),
        (0.36, 0.28, 0.32, 0.085),
        (0.63, 0.40, 0.33, 0.085),
        (0.58, 0.58, 0.37, 0.085),
        (0.10, 0.72, 0.40, 0.085),
    ],
    targets=[
        (0.22, 0.38),
        (0.44, 0.40),
        (0.50, 0.46),
        (0.70, 0.50),
        (0.82, 0.51),
        (0.62, 0.74),
    ],
)

AGENT_EXECUTION_GENERATED_PROMPT_EN = remake_fully_generated_comic_prompt(
    title="One Agent Execution Round",
    subtitle="Each step must be visible, limited, and traceable.",
    panels=[
        ("1 Goal", "Define what success means.", "A user places a goal card onto a task board with a clear finish flag and scope boundary."),
        ("2 State / memory", "Read known facts first.", "The agent opens drawers of notes, prior observations, and remaining steps before acting."),
        ("3 Plan", "Choose the next safe action.", "The agent sketches one next step on a planning board with a small budget and stop condition meter."),
        ("4 Tool", "Call a tool with arguments.", "A tool arm sends a structured request card into an API/tool station; arguments are visible as tokens or parameter chips."),
        ("5 Observation", "Read the result or error.", "A result card returns from the tool station; the agent compares success, error, cost, and latency signals."),
        ("6 Update / stop", "Continue, retry, or finish.", "The agent writes the new state into a trace log and chooses continue, retry, human review, or done."),
    ],
    footer="Trace every round: goal, action, input, observation, result.",
)

AGENT_EXECUTION_GENERATED_PROMPT_ZH = remake_fully_generated_comic_prompt(
    title="Agent 一轮怎么执行",
    subtitle="每一步都要可见、受限、可追踪。",
    panels=[
        ("1 目标", "先定义成功标准。", "用户把目标卡贴到任务板上，旁边有完成旗帜和清楚的范围边界。"),
        ("2 状态 / 记忆", "先读取已知事实。", "Agent 打开记忆抽屉，查看笔记、观察结果和剩余步骤，再决定行动。"),
        ("3 计划", "只选择下一步安全动作。", "Agent 在计划白板上画出下一步，旁边有预算和停止条件仪表。"),
        ("4 工具", "带参数调用工具。", "工具机械臂把结构化请求卡送入 API / tool 站，参数像小芯片一样清楚可见。"),
        ("5 观察", "读取结果或错误。", "工具返回结果卡，Agent 对比成功、错误、成本和延迟信号。"),
        ("6 更新 / 停止", "继续、重试或完成。", "Agent 把新状态写进 trace log，并选择继续、重试、人工确认或完成。"),
    ],
    footer="每轮都留下 trace：目标、动作、输入、观察、结果。",
)

AGENT_EXECUTION_GENERATED_PROMPT_JA = remake_fully_generated_comic_prompt(
    title="Agent の1回の実行ラウンド",
    subtitle="各ステップは見える、制限される、追跡できる必要がある。",
    panels=[
        ("1 Goal", "成功条件を先に決める。", "ユーザーが goal カードをタスクボードに貼り、完了フラグと範囲の境界が見える。"),
        ("2 State / memory", "既知の事実を先に読む。", "Agent が memory 引き出しを開き、メモ、観察結果、残り手順を確認してから動く。"),
        ("3 Plan", "安全な次の一手を選ぶ。", "Agent が planning board に次の一手を書き、横に予算と停止条件メーターがある。"),
        ("4 Tool", "引数付きで tool を呼ぶ。", "tool アームが構造化 request カードを API / tool station に送り、引数チップが見える。"),
        ("5 Observation", "結果または error を読む。", "tool station から result カードが戻り、success、error、cost、latency を確認する。"),
        ("6 Update / stop", "続行、再試行、完了を選ぶ。", "Agent が新しい state を trace log に書き、続行、retry、人の確認、done を選ぶ。"),
    ],
    footer="各ラウンドに trace を残す：goal、action、input、observation、result。",
)

register_remake_job(
    filename="ch09-agent-execution-loop-en.png",
    title="One Agent Execution Round",
    suggested_page="docs/ch09-agent/index.md",
    alt="Agent execution loop: user goal, state and memory, plan next step, tool call, observe result, update state, stop condition, and trace log.",
    scene="An agent control room. A learner supervises a small agent on a large console. The console shows goal intake, memory drawers, a planning board, tool arms calling APIs, observation cards returning results, state update, stop lights, and a trace log notebook.",
    subtitle="An agent is useful only when each step can be inspected.",
    items=remake_items(
        ("Goal", "What should be completed."),
        ("State / memory", "What is already known."),
        ("Plan", "Choose the next action."),
        ("Tool", "Execute with arguments."),
        ("Observation", "Read result or error."),
        ("Update / stop", "Continue, retry, or finish."),
    ),
    footer="Trace every round: goal, action, input, observation, result.",
    prompt=AGENT_EXECUTION_GENERATED_PROMPT_EN,
    generated_only=True,
)

register_remake_job(
    filename="ch09-agent-execution-loop.png",
    title="Agent 一轮怎么执行",
    suggested_page="i18n/zh-Hans/docusaurus-plugin-content-docs/current/ch09-agent/index.md",
    alt="Agent 执行闭环：用户目标、状态记忆、计划下一步、工具调用、观察结果、更新状态、停止条件和 trace log。",
    scene="Agent 控制室。学习者监督一个小型 Agent，大屏上有目标输入、记忆抽屉、计划白板、调用 API 的工具机械臂、返回结果的观察卡、状态更新、停止信号灯和 trace log 记录本。",
    subtitle="Agent 不是黑箱自动跑，每一步都要能检查。",
    items=remake_items(
        ("目标", "要完成什么。"),
        ("状态 / 记忆", "已经知道什么。"),
        ("计划", "选择下一步动作。"),
        ("工具", "带参数执行。"),
        ("观察", "读取结果或错误。"),
        ("更新 / 停止", "继续、重试或完成。"),
    ),
    footer="每轮都要留下 trace：目标、动作、输入、观察、结果。",
    prompt=AGENT_EXECUTION_GENERATED_PROMPT_ZH,
    generated_only=True,
)

register_remake_job(
    filename="ch09-agent-execution-loop-ja.png",
    title="Agent の1回の実行ラウンド",
    suggested_page="i18n/ja/docusaurus-plugin-content-docs/current/ch09-agent/index.md",
    alt="Agent 実行ループ：goal、state / memory、plan、tool call、observation、state update、stop condition、trace log。",
    scene="Agent 管制室。学習者が小さな Agent を監督し、大きなコンソールには goal 受付、memory 引き出し、planning board、API を呼ぶ tool アーム、observation カード、state update、停止ランプ、trace log ノートが並ぶ。",
    subtitle="Agent はブラックボックスではなく、各ステップを検査できる必要がある。",
    items=remake_items(
        ("Goal", "何を完了するか。"),
        ("State / memory", "すでに分かっていること。"),
        ("Plan", "次の行動を選ぶ。"),
        ("Tool", "引数付きで実行する。"),
        ("Observation", "結果やエラーを読む。"),
        ("Update / stop", "続行、再試行、完了。"),
    ),
    footer="各ラウンドに trace を残す：goal、action、input、observation、result。",
    prompt=AGENT_EXECUTION_GENERATED_PROMPT_JA,
    generated_only=True,
)

register_remake_job(
    filename="ch10-vision-pipeline-loop-en.png",
    title="Vision Pipeline and Failure Review",
    suggested_page="docs/ch10-computer-vision/index.md",
    alt="Vision pipeline: raw image, crop and preprocess, model output, box, mask, OCR, human review, error review, and next dataset update.",
    scene="A computer vision lab. A street or product image enters a workstation, is cropped and preprocessed, then a model produces class, box, mask, and OCR outputs. A reviewer circles mistakes, saves original and prediction images, and adds failed samples to the next dataset board.",
    subtitle="Save the image, the prediction, and the reason for failure.",
    items=remake_items(
        ("Raw image", "Keep the original input."),
        ("Preprocess", "Resize, crop, normalize."),
        ("Model output", "Predict labels or regions."),
        ("Box / mask / OCR", "Different tasks return different shapes."),
        ("Human review", "Find misses and false positives."),
        ("Error review", "Feed failures into the next run."),
    ),
    footer="Do not trust one score without looking at failed examples.",
)

register_remake_job(
    filename="ch10-vision-pipeline-loop-ja.png",
    title="Vision pipeline と失敗レビュー",
    suggested_page="i18n/ja/docusaurus-plugin-content-docs/current/ch10-computer-vision/index.md",
    alt="Vision pipeline：raw image、crop / preprocess、model output、box、mask、OCR、human review、error review、次の dataset 更新。",
    scene="コンピュータビジョンの実験室。街や商品の画像がワークステーションに入り、crop と preprocess を通り、model が class、box、mask、OCR を出す。レビュアーが間違いを丸で示し、元画像と予測画像を保存し、失敗サンプルを次の dataset ボードへ入れる。",
    subtitle="画像、予測、失敗理由をセットで残す。",
    items=remake_items(
        ("Raw image", "元の入力を保存する。"),
        ("Preprocess", "resize、crop、normalize。"),
        ("Model output", "ラベルや領域を予測する。"),
        ("Box / mask / OCR", "タスクごとに出力形が違う。"),
        ("Human review", "漏れと誤検出を探す。"),
        ("Error review", "失敗例を次回に回す。"),
    ),
    footer="スコアだけでなく、失敗例を見て判断する。",
)

register_remake_job(
    filename="ch10-vision-task-granularity-ladder-en.png",
    title="Vision Task Output Granularity",
    suggested_page="docs/ch10-computer-vision/index.md",
    alt="Computer vision task granularity ladder: image classification, object detection boxes, segmentation masks, OCR text, and visual question answering.",
    scene="A visual AI output ladder. One image moves through increasingly detailed output stations: whole-image class label, object boxes, pixel masks, OCR text extraction, and visual question answering with evidence regions.",
    subtitle="The task should match the output shape you need.",
    items=remake_items(
        ("Classification", "One label for the whole image."),
        ("Detection", "Boxes around objects."),
        ("Segmentation", "Pixel-level masks."),
        ("OCR", "Text read from the image."),
        ("Visual QA", "Answer with visual evidence."),
        ("Metric", "Evaluate the right output shape."),
    ),
    footer="Choose the task by the deliverable, not by model fashion.",
)

register_remake_job(
    filename="ch10-vision-task-granularity-ladder.png",
    title="视觉任务输出粒度",
    suggested_page="i18n/zh-Hans/docusaurus-plugin-content-docs/current/ch10-computer-vision/index.md",
    alt="计算机视觉任务粒度：图像分类、目标检测框、分割 mask、OCR 文本和视觉问答。",
    scene="视觉 AI 输出阶梯。同一张图片依次通过更细的输出站：整图分类标签、目标框、像素级 mask、OCR 文字读取、带视觉证据的问答。",
    subtitle="先明确需要什么输出形态，再选任务。",
    items=remake_items(
        ("分类", "整张图一个标签。"),
        ("检测", "给目标画 bounding box。"),
        ("分割", "输出像素级 mask。"),
        ("OCR", "读出图片里的文字。"),
        ("视觉问答", "结合图像证据回答。"),
        ("指标", "按输出形态评估。"),
    ),
    footer="按交付物选任务，不要按模型名凑方案。",
)

register_remake_job(
    filename="ch10-vision-task-granularity-ladder-ja.png",
    title="Vision task の出力粒度",
    suggested_page="i18n/ja/docusaurus-plugin-content-docs/current/ch10-computer-vision/index.md",
    alt="コンピュータビジョンの出力粒度：image classification、object detection box、segmentation mask、OCR text、visual QA。",
    scene="視覚 AI の出力ラダー。一枚の画像が段階的に詳しい出力ステーションを通る。画像全体の class label、object box、pixel mask、OCR text、視覚根拠つきの question answering。",
    subtitle="必要な出力形を先に決めてからタスクを選ぶ。",
    items=remake_items(
        ("Classification", "画像全体に1つのラベル。"),
        ("Detection", "物体を box で囲む。"),
        ("Segmentation", "pixel-level mask を出す。"),
        ("OCR", "画像内の文字を読む。"),
        ("Visual QA", "画像根拠で答える。"),
        ("Metric", "出力形に合う指標で見る。"),
    ),
    footer="モデル名ではなく、納品物からタスクを選ぶ。",
)

register_remake_job(
    filename="ch11-nlp-task-output-map-en.png",
    title="NLP Task Output Map",
    suggested_page="docs/ch11-nlp/index.md",
    alt="NLP tasks by output shape: classification label, entity JSON, summary paragraph, QA answer with source, and comparison score.",
    scene="A text analysis workbench. The same raw document is routed into different output trays: category label, entity JSON, summary paragraph, QA answer with source card, and comparison score chart. A learner checks the target format before choosing the model.",
    subtitle="Look at the output shape first, then choose the task code.",
    items=remake_items(
        ("Classification", "Return a label."),
        ("NER / extraction", "Return structured fields."),
        ("Summary", "Return a shorter paragraph."),
        ("QA", "Return answer plus source."),
        ("Similarity", "Return a comparison score."),
        ("Evaluation", "Check errors by output type."),
    ),
    footer="NLP work starts by defining the output contract.",
)

register_remake_job(
    filename="ch11-nlp-task-output-map.png",
    title="NLP 任务输出图",
    suggested_page="i18n/zh-Hans/docusaurus-plugin-content-docs/current/ch11-nlp/index.md",
    alt="NLP 任务按输出形态区分：分类标签、实体 JSON、摘要段落、带来源的 QA 答案和相似度分数。",
    scene="文本分析工作台。同一份 raw document 被送到不同输出托盘：分类标签、实体 JSON、摘要段落、带来源卡片的 QA 答案、相似度分数图。学习者先检查目标格式，再选模型任务。",
    subtitle="先看输出形态，再决定任务代码。",
    items=remake_items(
        ("分类", "返回一个标签。"),
        ("实体 / 抽取", "返回结构化字段。"),
        ("摘要", "返回更短段落。"),
        ("QA", "返回答案和来源。"),
        ("相似度", "返回比较分数。"),
        ("评估", "按输出类型查错。"),
    ),
    footer="NLP 实战从定义输出契约开始。",
)

register_remake_job(
    filename="ch11-nlp-task-output-map-ja.png",
    title="NLP タスク出力マップ",
    suggested_page="i18n/ja/docusaurus-plugin-content-docs/current/ch11-nlp/index.md",
    alt="NLP タスクの出力形：classification label、entity JSON、summary paragraph、source 付き QA answer、comparison score。",
    scene="テキスト分析の作業台。同じ raw document が複数の出力トレイへ分岐する。category label、entity JSON、summary paragraph、source カード付き QA answer、comparison score chart。学習者は model を選ぶ前に target format を確認する。",
    subtitle="先に出力形を見てから、タスクコードを選ぶ。",
    items=remake_items(
        ("Classification", "ラベルを返す。"),
        ("NER / extraction", "構造化フィールドを返す。"),
        ("Summary", "短い段落を返す。"),
        ("QA", "回答と出典を返す。"),
        ("Similarity", "比較スコアを返す。"),
        ("Evaluation", "出力タイプ別に誤りを見る。"),
    ),
    footer="NLP 実装は出力契約を決めるところから始まる。",
)

register_remake_job(
    filename="ch11-text-to-task-pipeline-en.png",
    title="Text to NLP Task Pipeline",
    suggested_page="docs/ch11-nlp/index.md",
    alt="Text to NLP pipeline: raw document, tokens, embedding, annotation, extraction, summary, retrieval QA, and evaluation.",
    scene="A text factory classroom scene. Raw documents, chats, and logs enter a token cutter, pass through an embedding machine as vector beads, then branch into annotation, extraction table, summary card, retrieval QA with sources, and evaluation board.",
    subtitle="Define the desired output before choosing the NLP method.",
    items=remake_items(
        ("Raw document", "Know where text came from."),
        ("Tokens", "Split into processable units."),
        ("Embedding", "Turn meaning into numbers."),
        ("Annotation", "Mark labels and fields."),
        ("Extraction", "Create structured records."),
        ("Summary / QA", "Produce user-facing text."),
        ("Evaluation", "Use errors to improve."),
    ),
    footer="Different NLP tasks share the same input but not the same output.",
)

register_remake_job(
    filename="ch11-text-to-task-pipeline.png",
    title="文本如何变成 NLP 任务输出",
    suggested_page="i18n/zh-Hans/docusaurus-plugin-content-docs/current/ch11-nlp/index.md",
    alt="文本进入 NLP 系统：raw document、tokens、embedding、annotation、extraction、summary、retrieval QA 和 evaluation。",
    scene="文本工厂课堂场景。raw documents、聊天和日志进入 token 切分器，通过 embedding 机器变成向量珠，然后分支到 annotation、抽取表格、摘要卡、带来源的 retrieval QA 和 evaluation 看板。",
    subtitle="先定义想要的输出，再选择 NLP 方法。",
    items=remake_items(
        ("raw document", "先看文本来源。"),
        ("tokens", "切成可处理单位。"),
        ("embedding", "把语义变成数字。"),
        ("annotation", "标注标签和字段。"),
        ("extraction", "生成结构化记录。"),
        ("summary / QA", "生成面向用户的文本。"),
        ("evaluation", "用错例继续改进。"),
    ),
    footer="同一份文本可以进入不同任务，但输出契约不同。",
)

register_remake_job(
    filename="ch11-text-to-task-pipeline-ja.png",
    title="テキストから NLP タスク出力へ",
    suggested_page="i18n/ja/docusaurus-plugin-content-docs/current/ch11-nlp/index.md",
    alt="テキストが NLP システムへ入る流れ：raw document、tokens、embedding、annotation、extraction、summary、retrieval QA、evaluation。",
    scene="テキスト工場の授業風景。raw documents、チャット、ログが token カッターに入り、embedding machine で vector beads へ変わり、annotation、extraction table、summary card、source 付き retrieval QA、evaluation board に分岐する。",
    subtitle="欲しい出力を定義してから NLP 手法を選ぶ。",
    items=remake_items(
        ("raw document", "テキストの出所を確認する。"),
        ("tokens", "処理単位に分ける。"),
        ("embedding", "意味を数値に変える。"),
        ("annotation", "ラベルと項目を付ける。"),
        ("extraction", "構造化 record を作る。"),
        ("summary / QA", "ユーザー向け文章を出す。"),
        ("evaluation", "誤りから改善する。"),
    ),
    footer="同じ入力でも、NLP タスクごとに出力契約は違う。",
)

register_remake_job(
    filename="ch12-multimodal-workflow-loop-en.png",
    title="Multimodal Workflow Loop",
    suggested_page="docs/ch12-multimodal/index.md",
    alt="Multimodal workflow: text/PDF, image/screenshot, audio waveform, video frames, alignment, understanding, human review, and report package.",
    scene="A multimodal project console. Text/PDF files, image screenshots, audio waveforms, and video frames enter a shared alignment table. Time, region, and textual evidence are linked before a model produces understanding, a reviewer checks risk, and a report package is exported.",
    subtitle="Multiple file types become useful only after alignment and review.",
    items=remake_items(
        ("Text / PDF", "Written evidence."),
        ("Image", "Visual regions."),
        ("Audio", "Sound clues."),
        ("Video", "Time-changing frames."),
        ("Align", "Connect time, region, text."),
        ("Understand", "Generate cross-modal meaning."),
        ("Review / export", "Check risk and package evidence."),
    ),
    footer="Multimodal work is align, understand, review, then deliver.",
)

register_remake_job(
    filename="ch12-multimodal-workflow-loop.png",
    title="多模态资料如何合成一个结果",
    suggested_page="i18n/zh-Hans/docusaurus-plugin-content-docs/current/ch12-multimodal/index.md",
    alt="多模态工作流：text/PDF、image/screenshot、audio waveform、video frames、align、understand、human review 和 report package。",
    scene="多模态项目控制台。Text/PDF 文件、image screenshot、audio waveform、video frames 同时进入对齐工作台。系统把时间、画面区域和文本证据连起来，再生成理解，由人工审核风险，最后导出 report package。",
    subtitle="多种文件堆在一起还不够，必须先对齐再审核。",
    items=remake_items(
        ("Text / PDF", "提供文字证据。"),
        ("Image", "提供视觉区域。"),
        ("Audio", "提供声音线索。"),
        ("Video", "提供时间变化。"),
        ("Align", "连接时间、区域和文本。"),
        ("Understand", "形成跨模态理解。"),
        ("Review / export", "审核风险并打包证据。"),
    ),
    footer="多模态流程是：对齐、理解、审核、交付。",
)

register_remake_job(
    filename="ch12-multimodal-workflow-loop-ja.png",
    title="マルチモーダル資料を1つの結果にする",
    suggested_page="i18n/ja/docusaurus-plugin-content-docs/current/ch12-multimodal/index.md",
    alt="マルチモーダル workflow：text/PDF、image/screenshot、audio waveform、video frames、align、understand、human review、report package。",
    scene="マルチモーダル project console。Text/PDF、image screenshot、audio waveform、video frames が同じ alignment table に入り、時間、領域、テキスト根拠が結ばれる。model が理解を作り、reviewer が risk を確認し、report package が export される。",
    subtitle="複数のファイルは、そろえて確認して初めて使える。",
    items=remake_items(
        ("Text / PDF", "文字の根拠。"),
        ("Image", "視覚的な領域。"),
        ("Audio", "音の手がかり。"),
        ("Video", "時間変化する frame。"),
        ("Align", "時間、領域、文章をつなぐ。"),
        ("Understand", "横断的な意味を作る。"),
        ("Review / export", "リスク確認と証拠パッケージ。"),
    ),
    footer="マルチモーダルは align、understand、review、deliver の順で進める。",
)


SVG_REPLACEMENT_CALLOUTS_5 = remake_callouts(
    boxes=[
        (0.04, 0.24, 0.36, 0.085),
        (0.60, 0.33, 0.36, 0.085),
        (0.04, 0.47, 0.36, 0.085),
        (0.60, 0.61, 0.36, 0.085),
        (0.18, 0.76, 0.64, 0.085),
    ],
    targets=[
        (0.28, 0.30),
        (0.67, 0.40),
        (0.32, 0.55),
        (0.72, 0.63),
        (0.50, 0.78),
    ],
)

SVG_REPLACEMENT_CALLOUTS_6 = remake_callouts(
    boxes=[
        (0.04, 0.23, 0.35, 0.082),
        (0.61, 0.28, 0.35, 0.082),
        (0.04, 0.42, 0.35, 0.082),
        (0.61, 0.50, 0.35, 0.082),
        (0.04, 0.66, 0.35, 0.082),
        (0.61, 0.73, 0.35, 0.082),
    ],
    targets=[
        (0.30, 0.30),
        (0.69, 0.35),
        (0.33, 0.49),
        (0.68, 0.55),
        (0.32, 0.68),
        (0.70, 0.76),
    ],
)


def svg_replacement_prompt(scene: str) -> str:
    return f"""
Create a vertical 9:16 bitmap teaching illustration for an AI full-stack course.
This replaces an old SVG diagram. Redesign it as a real teaching image, not as SVG-style art.
The image must teach through the scene itself: visible workbench, cards, screens, hands moving data, arrows, before/after states, and concrete artifacts.
Do not make a white-background rounded-box flowchart, slide, UI card stack, or pure text poster.
Use readable, sparse teaching text that is physically attached to the visual objects it explains. Do not make text-only posters, dense paragraphs, or top/bottom blocks unrelated to the image.
Avoid tiny text, gibberish, watermarks, real brand logos, decorative-only art, and unrelated background detail.

Teaching scene:
{scene}
""".strip()


def svg_replacement_direct_prompt(
    scene: str,
    locale: str,
    data: dict[str, Any],
    shared_layout: str,
) -> str:
    language_name = {
        "en": "English",
        "zh": "Simplified Chinese",
        "ja": "Japanese",
    }[locale]
    allowed_terms = {
        "en": "English only, plus standard code symbols when required.",
        "zh": "自然简体中文为主；API、RAG、Agent、token、embedding、Pipeline、Git、PATH、grep、wc、map、apply、NumPy、Pandas、p-value 等必要技术词可以保留英文。",
        "ja": "自然な日本語を主に使う。API、RAG、Agent、token、embedding、Pipeline、Git、PATH、grep、wc、map、apply、NumPy、Pandas、p-value など必要な技術語は英語表記のままでよい。",
    }[locale]
    locale_guard = {
        "en": "Important: every explanatory word visible in the image must be natural English. Only code, commands, math symbols, and API names may use technical notation.",
        "zh": "重要：图片中所有解释性文字必须是自然简体中文。只有代码、命令、数学符号和 API/工具名可以保留英文技术写法，不要出现英文说明句或日文假字。",
        "ja": "重要：画像内の説明文は自然な日本語にする。code、command、数式、API 名だけは英語表記のままでよい。英語の説明文や中国語の文字列は入れない。",
    }[locale]
    item_lines = "\n".join(
        f"- Show label text exactly: \"{label}\". Put nearby short note exactly: \"{detail}\""
        for label, detail in data["items"]
    )
    return f"""
Create one complete vertical 9:16 final bitmap teaching image for an AI full-stack course.
Target language: {language_name}.
Visible text policy: {allowed_terms}
{locale_guard}

Language-set consistency: this image belongs to a Simplified Chinese / English / Japanese triplet. All three variants must share the same composition, visual metaphor, camera angle, panel order, object placement, color rhythm, and reading path. Only the localized visible text should change.
Shared composition for this triplet:
{shared_layout}

This replaces an old SVG. Do not imitate SVG style. Do not create a white-background rounded-box flowchart, slide deck, UI card stack, pure text poster, or decorative scene.
The image must work as a practical worked example: a learner should understand the actual input, the command/code/action, the output/result, and the rule or mistake avoided before reading the article.
Prefer a concrete tutorial board, debugging notebook, terminal/code screen, data table, math grid, A/B test chart, or before/after example. People, devices, arrows, and panels are allowed only when they make the example easier to learn.
Every visual region must answer one teaching question: What is the input? What operation runs? What changes? What output appears? What mistake is prevented?
Show real, simple artifacts from the teaching scene: short code snippets, tiny tables, arrays, terminal commands, result files, equations, chart markers, or error/fix states. Make them large enough to read on a phone.
All visible text must be sparse, large, clean, readable, and physically attached to the visual object it explains. Use the target language for explanations. Code, commands, symbols, and named APIs may stay exactly as technical notation.
Do not include any other language, pseudo text, gibberish, filler UI text, watermark, logo, unrelated small text, fantasy factory decoration, or cute scene that does not teach the worked example.

Visible title exactly:
"{data['title']}"

Visible subtitle exactly:
"{data['subtitle']}"

Required teaching labels and short explanations. Include all of them, readable, near the matching visual action:
{item_lines}

Only draw the label text and the short note text. Do not draw field names such as "label", "explanation", "caption", "标签", "解释", "ラベル", or "説明".

Visible footer exactly:
"{data['footer']}"

Teaching scene:
{scene}
""".strip()


def register_svg_replacement_group(
    *,
    slug: str,
    pages: dict[str, str],
    scene: str,
    chapter_context: str,
    shared_layout: str,
    variants: dict[str, dict[str, Any]],
    callouts: list[dict[str, Any]],
) -> None:
    suffix = {"zh": "", "en": "-en", "ja": "-ja"}
    for locale, data in variants.items():
        register_remake_job(
            filename=f"{slug}{suffix[locale]}.png",
            title=str(data["title"]),
            suggested_page=pages[locale],
            alt=str(data["alt"]),
            scene=f"{scene}\n\nNearby chapter context to honor:\n{chapter_context}",
            subtitle=str(data["subtitle"]),
            items=[{"label": label, "detail": detail} for label, detail in data["items"]],
            footer=str(data["footer"]),
            prompt=svg_replacement_direct_prompt(
                f"{scene}\n\nNearby chapter context to honor:\n{chapter_context}",
                locale,
                data,
                shared_layout,
            ),
            generated_only=True,
        )


SVG_REPLACEMENT_GROUPS: list[dict[str, Any]] = [
    {
        "slug": "ch01-terminal-pipe-redirection-path",
        "pages": {
            "en": "docs/ch01-tools/ch01-terminal/02-basic-operations.md",
            "zh": "i18n/zh-Hans/docusaurus-plugin-content-docs/current/ch01-tools/ch01-terminal/02-basic-operations.md",
            "ja": "i18n/ja/docusaurus-plugin-content-docs/current/ch01-tools/ch01-terminal/02-basic-operations.md",
        },
        "scene": "A practical terminal worked example based on the nearby lesson. Show a file list containing train.py, model.py, README.md, notes.txt. The first command ls -la | grep \".py\" should send the file list through a pipe and leave only train.py and model.py. Then show a wc line-counter tool receiving two .py lines and outputting the number 2. Important: do not write the option token -l anywhere in the artwork, because image models often confuse lowercase ell with the digit 1; instead label the wc tool with the natural-language note line count / 行数统计 / 行数カウント depending on locale. A redirection example should show python train.py > training_log.txt saving output to a log file, and echo \"A new line\" >> notes.txt appending. A small PATH panel should show echo $PATH and which python pointing to a command location. The learner should clearly see pipe, redirection, append, count, and PATH as the same route-for-text idea.",
        "chapter_context": "The image appears at the start of Part 3: Pipes and redirection. The text explains that a pipe takes the previous command output and uses it as the next command input. Nearby examples are ls -la | grep \".py\", history | grep \"git\", counting Python files with wc, ls -la > filelist.txt, echo \"A new line\" >> notes.txt, python train.py > training_log.txt, cat model.py piped into wc, grep -r \"TODO\" ./ piped into wc, then Part 4 introduces echo $PATH and which python. In the generated image, avoid drawing the ambiguous wc option token; teach the count result visually instead.",
        "shared_layout": "Use the same vertical terminal blackboard for all three languages: title band at top, terminal output list in the upper-left, pipe/filter path through the center, wc count result in a small middle panel, redirection and append examples in the lower-left, PATH lookup panel on the lower-right, and a footer question at the bottom. Keep the same chalk/terminal texture, arrows, colors, and object positions across zh/en/ja.",
        "callouts": SVG_REPLACEMENT_CALLOUTS_5,
        "variants": {
            "en": {
                "title": "Terminal Data Flow",
                "subtitle": "Read a command line as a route for text, files, and tools.",
                "items": [
                    ("Pipe |", "Output becomes the next input."),
                    ("Filter", "grep keeps matching lines."),
                    ("Count", "wc turns text into a number."),
                    ("Redirect >", "Save output as evidence."),
                    ("PATH", "The shell finds the command file."),
                ],
                "footer": "Ask: where does the text go next?",
                "alt": "Terminal data flow: pipe sends output to the next command, grep filters text, wc counts lines, redirection saves evidence, and PATH resolves commands.",
            },
            "zh": {
                "title": "终端数据流",
                "subtitle": "把命令行看成文本、文件和工具的路线。",
                "items": [
                    ("管道 |", "上一步输出变成下一步输入。"),
                    ("过滤", "grep 只保留匹配行。"),
                    ("计数", "wc 把文本变成数量。"),
                    ("重定向 >", "把输出保存成证据文件。"),
                    ("PATH", "shell 按路径找到命令程序。"),
                ],
                "footer": "先问：这段文本下一步流向哪里？",
                "alt": "终端数据流图：管道把输出交给下一条命令，grep 过滤文本，wc 计数，重定向保存证据，PATH 解析命令位置。",
            },
            "ja": {
                "title": "Terminal のデータ流",
                "subtitle": "コマンド行を、テキスト、ファイル、ツールの経路として読む。",
                "items": [
                    ("Pipe |", "前の出力が次の入力になる。"),
                    ("Filter", "grep が一致行だけ残す。"),
                    ("Count", "wc がテキストを数にする。"),
                    ("Redirect >", "出力を証拠ファイルに保存。"),
                    ("PATH", "shell がコマンド本体を探す。"),
                ],
                "footer": "まず考える：このテキストは次にどこへ流れる？",
                "alt": "Terminal のデータ流：pipe が出力を次のコマンドへ渡し、grep が絞り込み、wc が数え、redirect が保存し、PATH がコマンドを解決する。",
            },
        },
    },
    {
        "slug": "ch01-git-merge-conflict-resolution",
        "pages": {
            "en": "docs/ch01-tools/ch02-git/04-branches.md",
            "zh": "i18n/zh-Hans/docusaurus-plugin-content-docs/current/ch01-tools/ch02-git/04-branches.md",
            "ja": "i18n/ja/docusaurus-plugin-content-docs/current/ch01-tools/ch02-git/04-branches.md",
        },
        "scene": "A practical Git merge conflict worked example based on the nearby lesson. Show two branch timelines, alice/update-model and bob/update-model, both editing src/model.py. Alice changes SimpleCNN to 32 filters and Bob changes it to 64 filters with a 5x5 kernel. In the center, show a readable conflict zone with conflict markers around the conv1 and fc1 lines. Then show a clean resolved src/model.py where the learner intentionally chooses the final architecture, followed by terminal steps git add src/model.py and git commit. The learner should clearly see why Git stopped, what same-location edit caused the conflict, and what stage/commit does after repair.",
        "chapter_context": "The image appears in Merge Conflicts. The chapter creates alice/update-model and bob/update-model from main. Alice edits src/model.py to use nn.Conv2d(3, 32, 3, padding=1) and fc1 with 32 * 16 * 16. Bob edits the same location to nn.Conv2d(3, 64, 5, padding=2) and fc1 with 64 * 16 * 16. Merging Bob after Alice produces CONFLICT (content), then the learner opens src/model.py, resolves conflict markers, stages, and commits.",
        "shared_layout": "Use the same vertical debugging board for all three languages: two colored branch timelines enter from the top, a large conflicted src/model.py file sits in the center, a clean resolved file sits below it, terminal commands git add and git commit sit near the bottom, and a final merge commit stamp appears before the footer. Keep identical branch colors, file positions, arrows, and reading order across zh/en/ja.",
        "callouts": SVG_REPLACEMENT_CALLOUTS_6,
        "variants": {
            "en": {
                "title": "Merge Conflict Repair",
                "subtitle": "A conflict asks you to combine two edits, not guess.",
                "items": [
                    ("Two branches", "Both changed the same file."),
                    ("Same line", "Git cannot choose automatically."),
                    ("Conflict zone", "Compare both versions."),
                    ("Resolve", "Keep the correct final code."),
                    ("git add", "Mark the file as fixed."),
                    ("Commit", "Save the merge result."),
                ],
                "footer": "Resolve by intent, then stage and commit.",
                "alt": "Git merge conflict repair: two branches edit the same line, Git marks a conflict zone, the developer resolves it, runs git add, and commits the merge.",
            },
            "zh": {
                "title": "合并冲突修复现场",
                "subtitle": "冲突是在请你合并两份修改，不是让你猜。",
                "items": [
                    ("两条分支", "都改了同一个文件。"),
                    ("同一位置", "Git 无法自动选择。"),
                    ("冲突区", "对照两边版本。"),
                    ("修复", "留下正确的最终代码。"),
                    ("git add", "标记这个文件已解决。"),
                    ("Commit", "保存合并结果。"),
                ],
                "footer": "先按意图修复，再暂存并提交。",
                "alt": "Git 合并冲突修复图：两条分支修改同一行，Git 标出冲突区，开发者对照并修复，git add 后提交合并结果。",
            },
            "ja": {
                "title": "Merge conflict の直し方",
                "subtitle": "conflict は、2つの編集を人が統合する合図。",
                "items": [
                    ("2つの branch", "同じ file を変更した。"),
                    ("同じ位置", "Git は自動で選べない。"),
                    ("Conflict zone", "両方の版を見比べる。"),
                    ("Resolve", "正しい最終 code にする。"),
                    ("git add", "解決済みとして印を付ける。"),
                    ("Commit", "merge 結果を保存する。"),
                ],
                "footer": "意図で直し、stage して commit する。",
                "alt": "Git merge conflict の修復：2つの branch が同じ行を変更し、conflict zone を比較して直し、git add して merge commit する。",
            },
        },
    },
    {
        "slug": "ch02-string-index-slice",
        "pages": {
            "en": "docs/ch02-python/ch01-basics/02-data-types.md",
            "zh": "i18n/zh-Hans/docusaurus-plugin-content-docs/current/ch02-python/ch01-basics/02-data-types.md",
            "ja": "i18n/ja/docusaurus-plugin-content-docs/current/ch02-python/ch01-basics/02-data-types.md",
        },
        "scene": "A practical Python string indexing worked example based on the nearby lesson. Show text = \"Python\" as six large tiles: P y t h o n. Above the tiles, positive indexes 0 1 2 3 4 5; below them, negative indexes -6 -5 -4 -3 -2 -1. Show text[0] -> P, text[-1] -> n, text[0:3] -> Pyt with the stop boundary before h visibly excluded, text[2:5] -> tho, text[::2] -> Pto, and text[::-1] -> nohtyP. The learner should see that indexes point to characters, while slices use start:stop:step and the stop is not included.",
        "chapter_context": "The image appears in String Indexing and Slicing. The text uses text = \"Python\", shows positive indexes 0..5 and negative indexes -6..-1, then examples text[0], text[5], text[-1], text[-2], text[0:3] -> Pyt, text[2:5] -> tho, text[:3], text[3:], text[:], text[::2] -> Pto, and text[::-1] -> nohtyP. It stresses left-closed, right-open slicing.",
        "shared_layout": "Use the same vertical ruler-and-tiles composition for all three languages: title at top, six large Python character tiles across the upper-middle, positive index ruler above, negative index ruler below, slice brackets and excluded stop marker in the center, example outputs in small panels around the tiles, and the footer at the bottom. Keep tile colors, ruler positions, arrows, and examples in the same places across zh/en/ja.",
        "callouts": SVG_REPLACEMENT_CALLOUTS_5,
        "variants": {
            "en": {
                "title": "String Index and Slice",
                "subtitle": "Indexes point to characters; slices point between boundaries.",
                "items": [
                    ("Index 0", "Counting starts at the first character."),
                    ("-1", "Negative index starts from the end."),
                    ("Start", "Slice includes this boundary."),
                    ("Stop", "Slice excludes this boundary."),
                    ("text[0:3]", "Returns the selected characters."),
                ],
                "footer": "For slices, read the stop position as the first item left out.",
                "alt": "String index and slicing: Python character tiles show positive and negative indexes, a slice start boundary, an excluded stop boundary, and text[0:3].",
            },
            "zh": {
                "title": "字符串索引与切片",
                "subtitle": "索引指向字符，切片更像在边界之间截取。",
                "items": [
                    ("index 0", "从第一个字符开始数。"),
                    ("-1", "负索引从末尾往前数。"),
                    ("start", "切片包含这个边界。"),
                    ("stop", "切片不包含这个边界。"),
                    ("text[0:3]", "返回被选中的字符。"),
                ],
                "footer": "读切片时，把 stop 当成第一个不取的位置。",
                "alt": "字符串索引和切片图：Python 字符方块展示正向索引、负向索引、切片起点、被排除的终点和 text[0:3]。",
            },
            "ja": {
                "title": "文字列の index と slice",
                "subtitle": "index は文字を指し、slice は境界の間を切り出す。",
                "items": [
                    ("index 0", "最初の文字から数える。"),
                    ("-1", "負の index は末尾から数える。"),
                    ("start", "slice はここを含む。"),
                    ("stop", "slice はここを含まない。"),
                    ("text[0:3]", "選ばれた文字を返す。"),
                ],
                "footer": "slice の stop は、最初に取らない位置として読む。",
                "alt": "文字列 index と slice：Python の文字タイルに正の index、負の index、start 境界、除外される stop 境界、text[0:3] を示す。",
            },
        },
    },
    {
        "slug": "ch02-short-circuit-safety-check",
        "pages": {
            "en": "docs/ch02-python/ch01-basics/03-operators.md",
            "zh": "i18n/zh-Hans/docusaurus-plugin-content-docs/current/ch02-python/ch01-basics/03-operators.md",
            "ja": "i18n/ja/docusaurus-plugin-content-docs/current/ch02-python/ch01-basics/03-operators.md",
        },
        "scene": "A practical Python short-circuit worked example based on the nearby lesson. Show False and print(\"This line will not be executed\") where the print call is skipped because and already knows the result is False. Show True or print(\"This line will not be executed either\") where print is skipped because or already knows the result is True. Then show the real safety check data = [] and if len(data) > 0 and data[0] > 10: because the list is empty, len(data) > 0 evaluates to False, so this first gate must be red/closed, not green/open. The data[0] access is skipped and locked, and an IndexError hazard stays behind a closed gate. The learner should clearly see left-to-right evaluation, the False first check for an empty list, and why the guard len(data) > 0 must come before data[0].",
        "chapter_context": "The image appears in Short-circuit evaluation. The text says Python's and and or evaluate from left to right and may skip the second expression. Nearby examples are False and print(...), True or print(...), then a safety check with data = [] and if len(data) > 0 and data[0] > 10 to avoid accessing an empty list.",
        "shared_layout": "Use the same vertical two-lane safety-check board for all three languages: top lane shows and with a red stop after False, middle lane shows or with a green stop after True, lower lane shows data=[] hitting len(data)>0 as a red False gate, then a locked data[0] gate with the error hazard blocked. Never put a green checkmark on len(data)>0 when data=[]. Keep gate shapes, arrows, warning colors, and lane order identical across zh/en/ja.",
        "callouts": SVG_REPLACEMENT_CALLOUTS_5,
        "variants": {
            "en": {
                "title": "Short-circuit Safety Check",
                "subtitle": "Python evaluates boolean conditions from left to right.",
                "items": [
                    ("First check", "data=[] makes len(data) > 0 False."),
                    ("and", "False stops the rest."),
                    ("Protected access", "data[0] is skipped."),
                    ("or", "True can stop early too."),
                    ("Order matters", "Put the safety check first."),
                ],
                "footer": "Short-circuiting is a guardrail against unsafe access.",
                "alt": "Short-circuit evaluation as a safety gate: len(data) > 0 runs before data[0], and a false first condition stops unsafe access.",
            },
            "zh": {
                "title": "短路求值安全检查",
                "subtitle": "Python 从左到右计算布尔条件。",
                "items": [
                    ("先检查", "data=[] 时 len(data) > 0 为 False。"),
                    ("and", "左边为 False 就停止。"),
                    ("保护访问", "跳过 data[0]。"),
                    ("or", "左边为 True 也会提前停止。"),
                    ("顺序重要", "安全检查要放在前面。"),
                ],
                "footer": "短路求值是避免危险访问的护栏。",
                "alt": "短路求值安全检查图：len(data) > 0 在 data[0] 前执行，左侧 False 会阻止危险访问。",
            },
            "ja": {
                "title": "短絡評価の安全チェック",
                "subtitle": "Python は条件を左から右へ評価する。",
                "items": [
                    ("最初の確認", "data=[] なら len(data) > 0 は False。"),
                    ("and", "左が False なら残りは止まる。"),
                    ("安全な access", "data[0] を skip する。"),
                    ("or", "左が True なら早く止まる。"),
                    ("順序が重要", "安全チェックを先に置く。"),
                ],
                "footer": "短絡評価は危険な access を防ぐ guardrail。",
                "alt": "短絡評価の安全チェック：len(data) > 0 を data[0] より先に評価し、False なら危険な access を止める。",
            },
        },
    },
    {
        "slug": "ch02-mutable-default-trap",
        "pages": {
            "en": "docs/ch02-python/ch01-basics/07-functions.md",
            "zh": "i18n/zh-Hans/docusaurus-plugin-content-docs/current/ch02-python/ch01-basics/07-functions.md",
            "ja": "i18n/ja/docusaurus-plugin-content-docs/current/ch02-python/ch01-basics/07-functions.md",
        },
        "scene": "A practical Python mutable default worked example based on the nearby lesson. Show the wrong code def add_item(item, items=[]): items.append(item); return items. Visualize the default list being created once when the function is defined, then reused. Show print(add_item(\"a\")) -> ['a'] and print(add_item(\"b\")) -> ['a', 'b'] as the bug. On the fix side, show def add_item(item, items=None): if items is None: items = []; items.append(item); return items, with separate fresh lists for calls. The learner should clearly see definition-time default values and why None is safer.",
        "chapter_context": "The image appears inside a caution block called The Default Parameter Trap. The text states default parameter values are determined when the function is defined and warns not to use mutable objects such as lists or dictionaries as defaults. The exact bad function is add_item(item, items=[]), and the fix is add_item(item, items=None) with a new list created inside.",
        "shared_layout": "Use the same vertical bug-vs-fix split for all three languages: left side is the wrong shared default list path, right side is the safe items=None path, with code snippets at the top, call results in the middle, shared/fresh list visuals below, and a footer warning at the bottom. Keep the same red bug color on the left and green fix color on the right across zh/en/ja.",
        "callouts": SVG_REPLACEMENT_CALLOUTS_5,
        "variants": {
            "en": {
                "title": "Mutable Default Trap",
                "subtitle": "A default list is created once, then reused.",
                "items": [
                    ("Definition time", "items=[] is made once."),
                    ("Call 1", "The shared list gets a."),
                    ("Call 2", "The same list now has a and b."),
                    ("Bug", "State leaks between calls."),
                    ("Fix", "Use items=None, then create a list."),
                ],
                "footer": "Use None when the default should be fresh each time.",
                "alt": "Mutable default parameter trap: a default list is created once, reused across calls, accumulates values, and is fixed with items=None.",
            },
            "zh": {
                "title": "可变默认参数陷阱",
                "subtitle": "默认 list 只创建一次，后续调用会复用。",
                "items": [
                    ("定义时", "items=[] 只创建一次。"),
                    ("第 1 次调用", "共享 list 加入 a。"),
                    ("第 2 次调用", "同一个 list 又加入 b。"),
                    ("Bug", "调用之间泄漏状态。"),
                    ("修复", "用 items=None，再新建 list。"),
                ],
                "footer": "想要每次都是新对象，就不要把可变对象写进默认值。",
                "alt": "可变默认参数陷阱图：默认 list 在定义时只创建一次，多次调用共享并累积值，正确做法是 items=None 后再创建新 list。",
            },
            "ja": {
                "title": "mutable default の罠",
                "subtitle": "default の list は一度だけ作られ、後で再利用される。",
                "items": [
                    ("定義時", "items=[] が一度だけ作られる。"),
                    ("1回目", "共有 list に a が入る。"),
                    ("2回目", "同じ list に b も入る。"),
                    ("Bug", "呼び出し間で状態が漏れる。"),
                    ("修正", "items=None にして中で list を作る。"),
                ],
                "footer": "毎回新しい object が必要なら、mutable object を default にしない。",
                "alt": "mutable default parameter の罠：default list が一度だけ作られ、複数回の呼び出しで共有され、items=None で修正する。",
            },
        },
    },
    {
        "slug": "ch03-numpy-view-copy-trap",
        "pages": {
            "en": "docs/ch03-data-analysis/ch02-numpy/03-indexing-slicing.md",
            "zh": "i18n/zh-Hans/docusaurus-plugin-content-docs/current/ch03-data-analysis/ch02-numpy/03-indexing-slicing.md",
            "ja": "i18n/ja/docusaurus-plugin-content-docs/current/ch03-data-analysis/ch02-numpy/03-indexing-slicing.md",
        },
        "scene": "A practical NumPy view versus copy worked example based on the nearby lesson. Show arr = np.array([1, 2, 3, 4, 5]). On the view side, sub = arr[1:4] points to the same memory cells [2, 3, 4]; after sub[0] = 99, show sub as [99, 3, 4] and arr as [1, 99, 3, 4, 5]. On the copy side, sub = arr[1:4].copy(); after sub[0] = 99, show sub as [99, 3, 4] but arr still [1, 2, 3, 4, 5]. Add a small note that slicing returns a view, boolean/fancy indexing return copies, and .copy() isolates edits. The learner should clearly see shared memory versus independent data.",
        "chapter_context": "The image appears in View vs Copy. The chapter says NumPy slicing returns a view, not a copy. It uses arr = np.array([1, 2, 3, 4, 5]), sub = arr[1:4], sub[0] = 99, and shows arr changing to [1, 99, 3, 4, 5]. The copy example uses arr[1:4].copy() and shows the original array unchanged. A nearby table contrasts slicing, boolean indexing, fancy indexing, .copy(), and reshape.",
        "shared_layout": "Use the same vertical memory-board comparison for all three languages: original array cells across the top, left column shows view sharing the highlighted cells and changing arr, right column shows copy on a separate board and leaving arr unchanged, with a small operation table near the bottom. Keep memory-cell colors, left/right split, arrows, and edit sequence identical across zh/en/ja.",
        "callouts": SVG_REPLACEMENT_CALLOUTS_5,
        "variants": {
            "en": {
                "title": "NumPy View vs Copy",
                "subtitle": "A slice may share memory with the original array.",
                "items": [
                    ("Original array", "Cells live in one memory block."),
                    ("View", "Slice points to the same cells."),
                    ("Edit view", "The original changes too."),
                    ("copy()", "Creates independent cells."),
                    ("Check intent", "Use copy when edits must be isolated."),
                ],
                "footer": "Before editing a slice, ask whether it shares memory.",
                "alt": "NumPy view versus copy: a slice view shares the original memory and changes the array, while copy() creates independent cells.",
            },
            "zh": {
                "title": "NumPy view 与 copy",
                "subtitle": "切片可能和原数组共享同一块内存。",
                "items": [
                    ("原数组", "数据在同一块内存里。"),
                    ("view", "切片指向同一批格子。"),
                    ("改 view", "原数组也会一起变。"),
                    ("copy()", "复制出独立格子。"),
                    ("看意图", "需要隔离修改时用 copy。"),
                ],
                "footer": "修改切片前，先问它是否共享内存。",
                "alt": "NumPy view 与 copy 图：切片 view 与原数组共享内存，修改 view 会影响原数组；copy() 生成独立数据。",
            },
            "ja": {
                "title": "NumPy view と copy",
                "subtitle": "slice は元 array と memory を共有することがある。",
                "items": [
                    ("元 array", "同じ memory block にある。"),
                    ("view", "slice は同じ cell を指す。"),
                    ("view を編集", "元 array も変わる。"),
                    ("copy()", "独立した cell を作る。"),
                    ("意図を確認", "隔離したい編集では copy を使う。"),
                ],
                "footer": "slice を編集する前に、memory 共有を確認する。",
                "alt": "NumPy view と copy：slice view は元 array と memory を共有し、編集が元 array に反映される。copy() は独立した data を作る。",
            },
        },
    },
    {
        "slug": "ch03-pandas-transform-method-choice",
        "pages": {
            "en": "docs/ch03-data-analysis/ch03-pandas/05-data-transform.md",
            "zh": "i18n/zh-Hans/docusaurus-plugin-content-docs/current/ch03-data-analysis/ch03-pandas/05-data-transform.md",
            "ja": "i18n/ja/docusaurus-plugin-content-docs/current/ch03-data-analysis/ch03-pandas/05-data-transform.md",
        },
        "scene": "A practical Pandas transform choice worked example based on the nearby lesson. Show a small student/customer DataFrame using the chapter's kinds of columns: name, math, english, gender, department_code, city. For math, show apply(grade) turning scores into excellent/good/average/pass. For a row operation, show apply(axis=1) creating total = math + english or a description. For gender and department_code, show map translating M/F and department codes into names. For city, show replace changing BJ to Beijing and SH/GZ/SZ to city names. For continuous scores, show binning into high/medium/low. The learner should choose by question: one value translation -> map, dirty value swap -> replace, custom row/column logic -> apply, new derived column -> assign/binning.",
        "chapter_context": "The image appears near the start of a Pandas data transformation lesson. The text asks what each column should become and contrasts translating codes, calculating a result from several row columns, and splitting continuous numbers into levels. Nearby examples include df with name/math/english, apply(np.sqrt), apply(grade), lambda pass/fail, apply(axis=1) for total/description, map for gender M/F and department_code, replace for city codes, and later sorting/ranking.",
        "shared_layout": "Use the same vertical DataFrame decision board for all three languages: source table at the top, five method lanes below it for map, replace, apply, assign, and binning, each lane showing one column goal and a before/after mini-column. Keep the table position, lane order, color coding, and footer rule identical across zh/en/ja.",
        "callouts": SVG_REPLACEMENT_CALLOUTS_5,
        "variants": {
            "en": {
                "title": "Choose a Pandas Transform",
                "subtitle": "Start from what each column should become.",
                "items": [
                    ("map", "Replace known values with a dictionary."),
                    ("replace", "Clean fixed labels or codes."),
                    ("apply", "Use custom row or column logic."),
                    ("assign", "Create a new derived column."),
                    ("binning", "Turn numbers into buckets."),
                ],
                "footer": "Use the simplest transform that matches the column goal.",
                "alt": "Pandas transform choice: map for known value lookup, replace for fixed labels, apply for custom logic, assign for derived columns, and binning for buckets.",
            },
            "zh": {
                "title": "选择 Pandas 转换方法",
                "subtitle": "先问：这一列最终要变成什么？",
                "items": [
                    ("map", "用字典替换已知值。"),
                    ("replace", "清理固定标签或编码。"),
                    ("apply", "写自定义行列逻辑。"),
                    ("assign", "生成新的派生列。"),
                    ("分箱", "把数字变成区间标签。"),
                ],
                "footer": "选择最贴合列目标的最简单方法。",
                "alt": "Pandas 转换方法选择图：map 做字典映射，replace 清理固定标签，apply 处理自定义逻辑，assign 生成派生列，分箱生成区间标签。",
            },
            "ja": {
                "title": "Pandas 変換メソッドの選び方",
                "subtitle": "まず、この列を何に変えたいかを見る。",
                "items": [
                    ("map", "辞書で既知の値を置き換える。"),
                    ("replace", "固定ラベルや code を直す。"),
                    ("apply", "行や列の custom logic を使う。"),
                    ("assign", "派生列を作る。"),
                    ("binning", "数値を区間ラベルにする。"),
                ],
                "footer": "列の目的に合う一番単純な変換を選ぶ。",
                "alt": "Pandas 変換メソッド選択：map は辞書変換、replace は固定ラベル修正、apply は custom logic、assign は派生列、binning は区間化。",
            },
        },
    },
    {
        "slug": "ch03-pandas-resample-rolling-timeline",
        "pages": {
            "en": "docs/ch03-data-analysis/ch03-pandas/08-time-series.md",
            "zh": "i18n/zh-Hans/docusaurus-plugin-content-docs/current/ch03-data-analysis/ch03-pandas/08-time-series.md",
            "ja": "i18n/ja/docusaurus-plugin-content-docs/current/ch03-data-analysis/ch03-pandas/08-time-series.md",
        },
        "scene": "A practical Pandas time-series worked example based on the nearby lesson, but keep it mobile-readable and not dense. Use only 7 daily sales points: Mon 10, Tue 12, Wed 8, Thu 14, Fri 16, Sat 9, Sun 11. Top panel: three preparation steps only, pd.to_datetime(date), set date as index, choose resample or rolling. Left panel: resample('W').sum() wraps the seven days into one weekly bucket and outputs total = 80. Right panel: rolling(3).mean() shows a 3-day window sliding from Mon-Wed to Tue-Thu, smoothing nearby points while daily frequency stays daily. Use large labels, large arrows, and no tables longer than 7 rows. Do not include quick-reference tables, many dates, many code blocks, or tiny chart axis labels. The learner should clearly see frequency bucket versus moving window.",
        "chapter_context": "The image appears just before Resampling in a time-series lesson. The chapter first gives a reliable beginner sequence: convert to datetime, set date as index, then resampling and rolling. It explains resample changes time frequency, with examples daily -> monthly/weekly aggregation, and rolling for the average of recent days. For this generated image, simplify the data to 7 days so the concept is readable on mobile.",
        "shared_layout": "Use the same sparse vertical time-axis board for all three languages: title at top, three small preparation steps in a horizontal strip, a 7-day sales timeline in the center, resample bucket on the left showing all seven days compressed into one total, rolling window on the right showing a 3-day window sliding one day, and a one-line footer. Keep the same timeline, bucket shape, moving-window overlay, colors, and branch order across zh/en/ja. Avoid dense tables and tiny text.",
        "callouts": SVG_REPLACEMENT_CALLOUTS_5,
        "variants": {
            "en": {
                "title": "Resample vs Rolling",
                "subtitle": "Resample changes frequency; rolling slides a window.",
                "items": [
                    ("Datetime", "Convert dates first."),
                    ("Time index", "Put dates on the index."),
                    ("resample", "Group into new time buckets."),
                    ("rolling", "Slide a fixed-size window."),
                    ("Compare trend", "Use both to see signal."),
                ],
                "footer": "Buckets summarize periods; windows smooth nearby points.",
                "alt": "Pandas time series diagram: convert dates, set a time index, resample daily points into buckets, and use rolling windows to smooth trends.",
            },
            "zh": {
                "title": "resample 与 rolling 时间线",
                "subtitle": "resample 改频率，rolling 滑动窗口。",
                "items": [
                    ("datetime", "先把日期转成时间类型。"),
                    ("时间索引", "把日期放到 index 上。"),
                    ("resample", "按新时间桶聚合。"),
                    ("rolling", "用固定窗口滑动计算。"),
                    ("看趋势", "两者配合看信号。"),
                ],
                "footer": "时间桶总结一段时间，窗口平滑相邻点。",
                "alt": "Pandas 时间序列图：先转换 datetime 并设置时间索引，resample 把日数据聚合到时间桶，rolling 用滑动窗口观察趋势。",
            },
            "ja": {
                "title": "resample と rolling の時間線",
                "subtitle": "resample は頻度を変え、rolling は窓を滑らせる。",
                "items": [
                    ("datetime", "日付を先に時刻型へ変換。"),
                    ("time index", "日付を index に置く。"),
                    ("resample", "新しい時間 bucket で集計。"),
                    ("rolling", "固定幅の window を滑らせる。"),
                    ("trend", "両方で信号を見る。"),
                ],
                "footer": "bucket は期間を要約し、window は近くの点をならす。",
                "alt": "Pandas 時系列図：datetime 変換、time index、resample の時間 bucket、rolling window による trend smoothing。",
            },
        },
    },
    {
        "slug": "ch04-vector-norm-unit-vector",
        "pages": {
            "en": "docs/ch04-ai-math/ch01-linear-algebra/01-vectors.md",
            "zh": "i18n/zh-Hans/docusaurus-plugin-content-docs/current/ch04-ai-math/ch01-linear-algebra/01-vectors.md",
            "ja": "i18n/ja/docusaurus-plugin-content-docs/current/ch04-ai-math/ch01-linear-algebra/01-vectors.md",
        },
        "scene": "A practical vector norm worked example based on the nearby lesson. Use a dark chalkboard / dark coordinate-grid style, not a white worksheet, not a paper notebook, and not a pale slide. Show vector a = np.array([3, 4]) as an arrow from origin to point (3,4) on a dark coordinate grid. Draw the 3 horizontal and 4 vertical legs as the classic 3-4-5 triangle. Show two code paths: manual length = sqrt(a[0]^2 + a[1]^2) = 5.0 and np.linalg.norm(a) = 5.0. Then show unit_a = a / np.linalg.norm(a) -> [0.6, 0.8], with an arrow in the same direction but length 1. Add the AI intuition: compare direction after normalization, not raw size. The learner should clearly see length, unit vector, and direction-only comparison.",
        "chapter_context": "The image appears in Vector Length (Magnitude / Norm). The chapter uses a = np.array([3, 4]), manual calculation with np.sqrt(a[0]**2 + a[1]**2), np.linalg.norm(a), and unit_a = a / np.linalg.norm(a) -> [0.6, 0.8]. It notes the 3-4-5 triangle and explains that AI often compares vector direction rather than size after normalization.",
        "shared_layout": "Use the same dark vertical math grid for all three languages: black or deep navy background, glowing/chalk coordinate grid with [3,4] vector in the upper half, 3-4-5 triangle and norm formula beside it, normalization operation in the center, unit vector arrow in the lower half, and AI direction-comparison note near the footer. Keep the grid, arrow angles, triangle colors, formula placement, and dark visual style identical across zh/en/ja. Do not use a white background, beige paper, worksheet, slide, or rounded-box infographic style.",
        "callouts": SVG_REPLACEMENT_CALLOUTS_5,
        "variants": {
            "en": {
                "title": "Vector Norm and Unit Vector",
                "subtitle": "Norm measures length; normalization keeps direction.",
                "items": [
                    ("Vector [3, 4]", "A point and direction from origin."),
                    ("Right triangle", "3 and 4 create length 5."),
                    ("Norm", "||a|| = 5 measures size."),
                    ("Normalize", "Divide every component by 5."),
                    ("Unit vector", "Same direction, length 1."),
                ],
                "footer": "Use norm for size; use unit vector for direction only.",
                "alt": "Vector norm and unit vector: a 3-4-5 vector on a grid is normalized by dividing components by five, keeping direction with length one.",
            },
            "zh": {
                "title": "向量范数与单位向量",
                "subtitle": "范数测长度，归一化保留方向。",
                "items": [
                    ("向量 [3, 4]", "从原点出发的方向。"),
                    ("直角三角形", "3 和 4 得到长度 5。"),
                    ("范数", "||a|| = 5 表示大小。"),
                    ("归一化", "每个分量都除以 5。"),
                    ("单位向量", "方向相同，长度为 1。"),
                ],
                "footer": "要大小看范数，只要方向看单位向量。",
                "alt": "向量范数与单位向量图：网格上的 3-4-5 向量通过每个分量除以 5 归一化，方向不变、长度变为 1。",
            },
            "ja": {
                "title": "vector norm と unit vector",
                "subtitle": "norm は長さを測り、正規化は向きを残す。",
                "items": [
                    ("vector [3, 4]", "原点からの向き。"),
                    ("直角三角形", "3 と 4 から長さ 5。"),
                    ("norm", "||a|| = 5 が大きさ。"),
                    ("正規化", "各成分を 5 で割る。"),
                    ("unit vector", "同じ向きで長さ 1。"),
                ],
                "footer": "大きさは norm、向きだけなら unit vector。",
                "alt": "vector norm と unit vector：grid 上の 3-4-5 vector を成分ごとに 5 で割り、向きを保ったまま長さ 1 にする。",
            },
        },
    },
    {
        "slug": "ch04-matrix-multiplication-shape-rule",
        "pages": {
            "en": "docs/ch04-ai-math/ch01-linear-algebra/02-matrices.md",
            "zh": "i18n/zh-Hans/docusaurus-plugin-content-docs/current/ch04-ai-math/ch01-linear-algebra/02-matrices.md",
            "ja": "i18n/ja/docusaurus-plugin-content-docs/current/ch04-ai-math/ch01-linear-algebra/02-matrices.md",
        },
        "scene": "A practical matrix multiplication shape worked example based on the nearby lesson. Show A = [[1,2,3],[4,5,6]] with shape 2 x 3 and B = [[1,2],[3,4],[5,6]] with shape 3 x 2. Highlight the matching inner dimensions 3 and 3, then show C = A @ B with shape 2 x 2 and result [[22,28],[49,64]]. Include one readable cell calculation: C[0,0] = 1*1 + 2*3 + 3*5 = 22. Add a failed side example (2 x 3) @ (4 x 2) where inner 3 and 4 do not match. The learner should clearly see inner dimensions must match, outer dimensions remain, and A @ B order matters.",
        "chapter_context": "The image appears in Size rules for matrix multiplication. The chapter states columns of the left matrix must equal rows of the right matrix, and result shape is rows of left by columns of right. It uses A = [[1,2,3],[4,5,6]] shape 2x3, B = [[1,2],[3,4],[5,6]] shape 3x2, C = A @ B shape 2x2, output [[22,28],[49,64]], and warns matrix multiplication is not commutative.",
        "shared_layout": "Use the same vertical matrix-tile board for all three languages: A matrix on the left, B matrix on the right, matching inner dimensions highlighted where they meet, C result matrix below, one C[0,0] dot-product calculation in a callout, mismatch example in a small warning panel, and footer rule at bottom. Keep tile sizes, matrix positions, colors, and highlight order identical across zh/en/ja.",
        "callouts": SVG_REPLACEMENT_CALLOUTS_5,
        "variants": {
            "en": {
                "title": "Matrix Multiplication Shape Rule",
                "subtitle": "Inner dimensions must match; outer dimensions become the result.",
                "items": [
                    ("A: m x n", "Rows of samples, n features."),
                    ("B: n x p", "n inputs, p outputs."),
                    ("Inner n", "These two dimensions must match."),
                    ("C: m x p", "Outer dimensions remain."),
                    ("Mismatch", "If inner sizes differ, multiplication fails."),
                ],
                "footer": "Read A @ B as (m x n) @ (n x p) -> (m x p).",
                "alt": "Matrix multiplication shape rule: A is m by n, B is n by p, inner n dimensions must match, and result C is m by p.",
            },
            "zh": {
                "title": "矩阵乘法尺寸规则",
                "subtitle": "中间维度要接得上，外侧维度留下来。",
                "items": [
                    ("A: m x n", "m 行样本，n 个特征。"),
                    ("B: n x p", "n 个输入，p 个输出。"),
                    ("中间 n", "这两个维度必须相等。"),
                    ("C: m x p", "外侧维度组成结果。"),
                    ("不匹配", "中间维度不同就不能乘。"),
                ],
                "footer": "把 A @ B 读成 (m x n) @ (n x p) -> (m x p)。",
                "alt": "矩阵乘法尺寸规则图：A 是 m x n，B 是 n x p，中间 n 必须相等，结果 C 是 m x p。",
            },
            "ja": {
                "title": "行列積の shape rule",
                "subtitle": "内側の次元が一致し、外側の次元が結果になる。",
                "items": [
                    ("A: m x n", "m 行の sample、n features。"),
                    ("B: n x p", "n inputs、p outputs。"),
                    ("内側の n", "この2つが一致する必要がある。"),
                    ("C: m x p", "外側の次元が残る。"),
                    ("不一致", "内側が違うと掛けられない。"),
                ],
                "footer": "A @ B は (m x n) @ (n x p) -> (m x p) と読む。",
                "alt": "行列積の shape rule：A は m x n、B は n x p、内側の n が一致し、結果 C は m x p になる。",
            },
        },
    },
    {
        "slug": "ch04-pvalue-null-distribution",
        "pages": {
            "en": "docs/ch04-ai-math/ch02-probability/03-statistical-inference.md",
            "zh": "i18n/zh-Hans/docusaurus-plugin-content-docs/current/ch04-ai-math/ch02-probability/03-statistical-inference.md",
            "ja": "i18n/ja/docusaurus-plugin-content-docs/current/ch04-ai-math/ch02-probability/03-statistical-inference.md",
        },
        "scene": "A practical A/B test p-value worked example based on the nearby lesson. Show H0: A and B have no real click-through-rate difference. Use the chapter's product example: Group A blue button n=1000, true click-through rate 10%; Group B green button n=1000, true click-through rate 12%. Show observed difference as a marker in the right tail of a null distribution centered at 0. Shade the tail area as p-value, label small p-value like 0.01 as unusual under H0 and large p-value like 0.3 as common under H0. Add a decision check strip for sample size, experiment design, business impact, and many tests. The learner should clearly see p-value is a probability under H0, not proof of the alternative hypothesis.",
        "chapter_context": "The image appears in Intuition for p-values. The chapter defines p-value as the probability of a difference this large or larger by random fluctuation assuming no real difference. It contrasts p=0.01 and p=0.3, warns p-value does not prove the alternative hypothesis, and says real products must also check sample size, experiment design, business impact, and multiple tests. The following A/B example simulates blue button A at 10% and green button B at 12%, each n=1000.",
        "shared_layout": "Use the same vertical A/B-test analysis board for all three languages: H0 statement and A/B sample cards at the top, null distribution curve in the center, observed lift marker in the right tail, shaded p-value area, small p=0.01 vs p=0.3 interpretation strip, decision-check strip near the bottom, and footer warning at the bottom. Keep chart shape, marker position, colors, and panel order identical across zh/en/ja.",
        "callouts": SVG_REPLACEMENT_CALLOUTS_5,
        "variants": {
            "en": {
                "title": "p-value Under H0",
                "subtitle": "How unusual is this result if no real difference exists?",
                "items": [
                    ("H0", "Assume A and B are equal."),
                    ("Null distribution", "Simulate expected random differences."),
                    ("Observed result", "Place the real experiment on the curve."),
                    ("Tail area", "This area is the p-value."),
                    ("Decision check", "Also inspect sample size and impact."),
                ],
                "footer": "Small p-value means unusual under H0, not automatically true in business.",
                "alt": "p-value intuition: under H0, simulated no-difference experiments form a distribution, the observed result sits in a tail, and tail area is the p-value.",
            },
            "zh": {
                "title": "零假设下的 p-value",
                "subtitle": "如果真的没有差异，这个结果有多罕见？",
                "items": [
                    ("H0", "先假设 A 和 B 没差异。"),
                    ("零假设分布", "模拟随机波动会长什么样。"),
                    ("观测结果", "把真实实验放到曲线上。"),
                    ("尾部面积", "这块面积就是 p-value。"),
                    ("决策检查", "还要看样本量和业务影响。"),
                ],
                "footer": "p-value 小表示在 H0 下罕见，不等于业务上一定成立。",
                "alt": "p-value 直觉图：在零假设 H0 下模拟无差异实验形成分布，真实观测结果落在尾部，尾部面积就是 p-value。",
            },
            "ja": {
                "title": "H0 のもとでの p-value",
                "subtitle": "本当に差がないなら、この結果はどれほど珍しいか。",
                "items": [
                    ("H0", "A と B は同じと仮定する。"),
                    ("null distribution", "偶然の差の分布を見る。"),
                    ("観測結果", "実験結果を曲線上に置く。"),
                    ("tail area", "この面積が p-value。"),
                    ("判断確認", "sample size と impact も見る。"),
                ],
                "footer": "p-value が小さいとは、H0 のもとで珍しいという意味。",
                "alt": "p-value の直感：H0 のもとで無差の実験をシミュレーションして分布を作り、観測結果の尾部面積を p-value として見る。",
            },
        },
    },
]

for svg_group in SVG_REPLACEMENT_GROUPS:
    register_svg_replacement_group(**svg_group)

DIRECT_TRIPLET_GROUPS: list[dict[str, Any]] = [
    {
        "slug": "ch05-sklearn-workflow-result-map",
        "pages": {
            "en": "docs/ch05-machine-learning/ch01-ml-basics/02-sklearn-intro.md",
            "zh": "i18n/zh-Hans/docusaurus-plugin-content-docs/current/ch05-machine-learning/ch01-ml-basics/02-sklearn-intro.md",
            "ja": "i18n/ja/docusaurus-plugin-content-docs/current/ch05-machine-learning/ch01-ml-basics/02-sklearn-intro.md",
        },
        "scene": "A practical sklearn lab result interpretation board for the Iris workflow, not a code-copy poster. Do not draw long code blocks. Use only short code chips such as fit(), predict(), score(), dump(), load(), and X_test[:1]. Iris data has 150 rows and 4 feature columns. The split is test_size=0.25, so show train=112 and test=38; do not show any other counts. Three comparable pipelines run on the same split: logistic uses scaling plus a classifier, tree uses a tree model, knn uses scaling plus a neighbor classifier. A scoreboard shows logistic 0.921, tree 0.895, knn 0.921, and a tie-break note chooses logistic. Show a simple classification report summary, not a full confusion matrix: setosa support 12 with f1=1.00, versicolor support 13 with f1 about 0.89, virginica support 13 with f1 about 0.88, total accuracy about 0.92. Do not invent different support counts or large matrix numbers. Then the winning Pipeline is saved to iris_pipeline.joblib and reloaded to predict the first test sample. The learner should understand why terminal print output is evidence: compare models, choose a winner, inspect errors, save and reload.",
        "chapter_context": "This image is inserted after the expected output of the Scikit-learn follow-along script. Nearby code loads Iris, splits train/test, creates three sklearn Pipelines, prints accuracy for logistic, tree, and knn, prints the best model and classification report, saves iris_pipeline.joblib, reloads it, and confirms a reloaded prediction. The page teaches fit, transform, predict, score, Pipeline, fair comparison, and saved evidence.",
        "shared_layout": "Vertical 9:16. Top title and subtitle. Upper section shows Iris data split into train=112 and test=38, no other counts. Middle section has three parallel model lanes with the same input split and only short chips: fit(), predict(), score(); avoid long code blocks and avoid drawing a scaler inside the tree lane. Right or lower-middle section has a compact classification-report summary card with three rows, support 12/13/13, and total accuracy about 0.92; avoid a detailed confusion matrix. Bottom section shows save to iris_pipeline.joblib, reload, and first prediction check using X_test[:1]. Keep object positions, three lanes, scores, and reading order identical across zh/en/ja.",
        "variants": {
            "zh": {
                "title": "sklearn 实验结果怎么看",
                "subtitle": "同一份测试集比较模型，再把胜出 Pipeline 保存成证据。",
                "items": [
                    ("同一 split", "train/test 不变，比较才公平。"),
                    ("三条 Pipeline", "logistic、tree、knn 跑同一份输入。"),
                    ("tree lane", "树模型不需要缩放也能作为对比。"),
                    ("scoreboard", "0.921、0.895、0.921 先看测试分数。"),
                    ("report", "support=12/13/13，accuracy≈0.92。"),
                    ("best=logistic", "并列时选择更稳定、可解释的版本。"),
                    ("joblib reload", "保存后重新加载，预测仍一致。"),
                ],
                "footer": "完整实验不是只打印数字，而是留下可复查的选择证据。",
                "alt": "sklearn Iris 完整实验结果图：同一 train/test split 下比较 logistic、tree、knn，阅读 classification report，并保存重载 Pipeline。",
            },
            "en": {
                "title": "Reading sklearn Lab Results",
                "subtitle": "Compare models on one test set, then save the winning Pipeline as evidence.",
                "items": [
                    ("same split", "Fair comparison uses one train/test split."),
                    ("three Pipelines", "logistic, tree, and knn use the same input."),
                    ("tree lane", "The tree model can be compared without scaling."),
                    ("scoreboard", "0.921, 0.895, 0.921: start with test scores."),
                    ("report", "support=12/13/13, accuracy≈0.92."),
                    ("best=logistic", "Break ties with stability and interpretability."),
                    ("joblib reload", "Reloaded model gives the same prediction."),
                ],
                "footer": "A complete lab leaves reviewable evidence, not just printed numbers.",
                "alt": "sklearn Iris lab result map: compare logistic, tree, and knn on the same train/test split, read the classification report, then save and reload the Pipeline.",
            },
            "ja": {
                "title": "sklearn 実験結果の読み方",
                "subtitle": "同じテストデータで比較し、勝った Pipeline を証拠として保存する。",
                "items": [
                    ("同じ split", "train/test を固定すると公平に比べられる。"),
                    ("3つの Pipeline", "logistic、tree、knn が同じ入力を使う。"),
                    ("tree lane", "木モデルは scaling なしでも比較できる。"),
                    ("scoreboard", "0.921、0.895、0.921 はまず test score。"),
                    ("report", "support=12/13/13、accuracy≈0.92。"),
                    ("best=logistic", "同点なら安定性と説明しやすさで選ぶ。"),
                    ("joblib reload", "保存後に読み直しても予測が一致。"),
                ],
                "footer": "完全な実験は数字の print だけでなく、確認できる証拠を残す。",
                "alt": "sklearn Iris 実験結果図：同じ train/test split で logistic、tree、knn を比較し、classification report を読み、Pipeline を保存して再読み込みする。",
            },
        },
    },
    {
        "slug": "ch05-decision-tree-depth-pruning-result-map",
        "pages": {
            "en": "docs/ch05-machine-learning/ch02-supervised/03-decision-trees.md",
            "zh": "i18n/zh-Hans/docusaurus-plugin-content-docs/current/ch05-machine-learning/ch02-supervised/03-decision-trees.md",
            "ja": "i18n/ja/docusaurus-plugin-content-docs/current/ch05-machine-learning/ch02-supervised/03-decision-trees.md",
        },
        "scene": "A decision tree lab result board based on the Iris and diabetes script. Important: do not put any numeric text inside the tree nodes. Tree thumbnails are only visual icons with plain colored nodes and branches; no X1, no thresholds, no gini, no n counts, no fake class probabilities. Put all numbers only in the dedicated result cards. Use only these numeric results from the lesson: max_depth=1 train=0.670 test=0.658 leaves=2; max_depth=2 train=0.964 test=0.947 leaves=3; max_depth=3 train=0.982 test=0.974 leaves=5; max_depth=None train=0.982 test=0.974 leaves=5. For pruning, use exactly: ccp_alpha=0.0000 test=0.921 leaves=7; ccp_alpha=0.0067 test=0.921 leaves=5; ccp_alpha=0.2636 test=0.658 leaves=2. For regression tree, use exactly: max_depth=2 MAE=47.3 leaves=4; max_depth=4 MAE=44.4 leaves=14; max_depth=None MAE=48.7 leaves=25. Include a step-like prediction sketch with no extra numeric values. The teaching point is that depth and pruning control complexity; more leaves do not automatically improve test performance.",
        "chapter_context": "This image is inserted after the expected output of the decision_tree_lab.py script. Nearby code prints classification_depth_lab, tree_rules, feature_importance, pruning_lab, and regression_tree_lab. The text explains max_depth=1 is too simple, max_depth=3 asks useful follow-up questions, small pruning keeps the same score with fewer leaves, and too much pruning loses useful rules.",
        "shared_layout": "Vertical 9:16. Top title and subtitle. Upper half has three clean tree thumbnails labeled max_depth=1, max_depth=3, and too complex; the tree thumbnails must contain no internal text or numbers. Next to them place a four-row result card for depth 1/2/3/None using the exact train/test/leaves values from the scene. Middle has a pruning slider from low alpha to high alpha with exactly three cards: 0.0000 -> 7 leaves / test 0.921, 0.0067 -> 5 leaves / test 0.921, 0.2636 -> 2 leaves / test 0.658; tree icons inside these cards must also contain no internal text. Lower section has a regression tree step prediction sketch plus a three-row MAE card with exact values 47.3, 44.4, 48.7. Keep the same tree sizes, alpha positions, scores, colors, and reading path across zh/en/ja. Do not draw extra numeric curves.",
        "variants": {
            "zh": {
                "title": "决策树实验结果怎么读",
                "subtitle": "深度控制复杂度，剪枝用更少叶子保留泛化能力。",
                "items": [
                    ("max_depth=1", "问题太少，train 和 test 都低。"),
                    ("max_depth=3", "规则够用，test=0.974。"),
                    ("too complex", "叶子变多不等于更会预测。"),
                    ("ccp_alpha=0.0067", "5 个叶子保持 test=0.921。"),
                    ("ccp_alpha=0.2636", "剪太狠，test 降到 0.658。"),
                    ("regression tree", "MAE：47.3 -> 44.4 -> 48.7。"),
                ],
                "footer": "看树模型先问：规则是否够用，复杂度是否过头。",
                "alt": "决策树实验结果图：比较 max_depth、train/test 分数、ccp_alpha 剪枝叶子数，并展示回归树阶梯预测。",
            },
            "en": {
                "title": "Reading Decision Tree Results",
                "subtitle": "Depth controls complexity; pruning keeps generalization with fewer leaves.",
                "items": [
                    ("max_depth=1", "Too few questions; train and test are low."),
                    ("max_depth=3", "Useful rules; test=0.974."),
                    ("too complex", "More leaves do not mean better prediction."),
                    ("ccp_alpha=0.0067", "5 leaves keep test=0.921."),
                    ("ccp_alpha=0.2636", "Too much pruning drops test to 0.658."),
                    ("regression tree", "MAE: 47.3 -> 44.4 -> 48.7."),
                ],
                "footer": "For tree models, ask: enough rules, or too much complexity?",
                "alt": "Decision tree lab result map: compare max_depth train/test scores, ccp_alpha pruning leaf counts, and regression tree step predictions.",
            },
            "ja": {
                "title": "決定木の実験結果を読む",
                "subtitle": "深さで複雑さを制御し、枝刈りで少ない葉でも汎化を保つ。",
                "items": [
                    ("max_depth=1", "質問が少なすぎて train も test も低い。"),
                    ("max_depth=3", "有効な規則で test=0.974。"),
                    ("too complex", "葉が多いほど良い予測とは限らない。"),
                    ("ccp_alpha=0.0067", "5枚の葉で test=0.921 を維持。"),
                    ("ccp_alpha=0.2636", "刈りすぎると test は 0.658 へ低下。"),
                    ("regression tree", "MAE：47.3 -> 44.4 -> 48.7。"),
                ],
                "footer": "木モデルでは、規則が足りるか、複雑すぎないかを読む。",
                "alt": "決定木実験結果図：max_depth の train/test score、ccp_alpha 枝刈りの葉数、回帰木の階段状予測を比較する。",
            },
        },
    },
    {
        "slug": "ch05-clustering-result-interpretation-map",
        "pages": {
            "en": "docs/ch05-machine-learning/ch03-unsupervised/01-clustering.md",
            "zh": "i18n/zh-Hans/docusaurus-plugin-content-docs/current/ch05-machine-learning/ch03-unsupervised/01-clustering.md",
            "ja": "i18n/ja/docusaurus-plugin-content-docs/current/ch05-machine-learning/ch03-unsupervised/01-clustering.md",
        },
        "scene": "A clustering lab result visualization. Do not draw full code blocks, terminal logs, or invented metric tables. Show a blob dataset on the left with exactly three round blob groups and exactly three centroid markers, because the nearby lab uses centers=3. Show K candidates 2, 3, 4, 5, but only K=3 has a numeric callout: silhouette=0.869. If drawing inertia, show only a qualitative downward arrow or unlabeled falling line, because exact inertia values are not needed; do not write any inertia numbers. On the right, show a two-moons dataset where K-Means cuts the moons incorrectly with exactly ARI=0.475, while DBSCAN with exactly eps=0.25 follows the curved moons with exactly ARI=0.995 and one noise dot. Lower corner shows hierarchical clustering on exactly three round blob groups with exactly ARI=1.0 as a compact-cluster alternative. Do not invent extra silhouette, inertia, sample count, time, or iteration numbers.",
        "chapter_context": "This image is inserted after the expected output of clustering_lab.py. Nearby code prints K-Means inertia and silhouette for K=2..5, cluster centers, K-Means ARI on blobs, K-Means failure on make_moons, DBSCAN eps tuning, and agglomerative clustering ARI. The text explains K-Means fits round compact clusters, inertia always drops as K grows, silhouette helps choose K, and DBSCAN follows dense curved shapes.",
        "shared_layout": "Vertical 9:16. Top title and subtitle. Left half has exactly three round blob groups with three centroid markers, K-selection mini chart with only one numeric badge: K=3 silhouette=0.869, plus a qualitative inertia-down note with no inertia numbers. Right half has moon scatter comparison: K-Means wrong cut ARI=0.475 versus DBSCAN eps=0.25 ARI=0.995. Bottom has a small method-choice strip: K-Means for round groups, DBSCAN for curved/noisy groups, hierarchical ARI=1.0 for inspection over exactly three round blob groups. Keep chart shapes, colors, metric numbers, and panel order identical across zh/en/ja. No code blocks, no terminal tables, no extra metric numbers.",
        "variants": {
            "zh": {
                "title": "聚类结果不只看数字",
                "subtitle": "K、形状和业务含义要一起读，图比 print 更直观。",
                "items": [
                    ("K=3", "silhouette=0.869，圆形簇最清楚。"),
                    ("inertia", "K 变大一定下降，不能单独用。"),
                    ("centroids", "中心点代表每个 K-Means 簇。"),
                    ("K-Means on moons", "ARI=0.475，直线切错弯月形。"),
                    ("DBSCAN eps=0.25", "ARI=0.995，沿密度找到两条弯月。"),
                    ("hierarchical", "圆形簇上也能作为可检查替代。"),
                ],
                "footer": "聚类要同时检查分数、散点形状和可解释性。",
                "alt": "聚类实验结果图：K-Means 用 inertia 和 silhouette 选择 K=3，DBSCAN 在双月数据上比 K-Means 更符合形状。",
            },
            "en": {
                "title": "Read Clustering Beyond Numbers",
                "subtitle": "Read K, shape, and meaning together; plots make print output usable.",
                "items": [
                    ("K=3", "silhouette=0.869; round groups are clear."),
                    ("inertia", "Always falls as K grows, so never use it alone."),
                    ("centroids", "Each center represents one K-Means cluster."),
                    ("K-Means on moons", "ARI=0.475; a straight cut breaks curves."),
                    ("DBSCAN eps=0.25", "ARI=0.995; density follows two moons."),
                    ("hierarchical", "A checkable alternative for compact groups."),
                ],
                "footer": "Clustering needs scores, scatter shape, and interpretability together.",
                "alt": "Clustering lab result map: K-Means uses inertia and silhouette to select K=3, while DBSCAN fits two-moon data better than K-Means.",
            },
            "ja": {
                "title": "クラスタリング結果は数字だけで読まない",
                "subtitle": "K、形状、意味を一緒に見ると print の結果が使える。",
                "items": [
                    ("K=3", "silhouette=0.869、丸いクラスタが明確。"),
                    ("inertia", "K が増えると必ず下がるので単独では使わない。"),
                    ("centroids", "各中心点が K-Means のクラスタを代表。"),
                    ("K-Means on moons", "ARI=0.475、直線分割では曲線を壊す。"),
                    ("DBSCAN eps=0.25", "ARI=0.995、密度で2つの月を追う。"),
                    ("hierarchical", "コンパクトな群の確認用代替になる。"),
                ],
                "footer": "クラスタリングは score、散布図の形、説明可能性を合わせて読む。",
                "alt": "クラスタリング実験結果図：K-Means は inertia と silhouette で K=3 を選び、双月データでは DBSCAN が形に合う。",
            },
        },
    },
    {
        "slug": "ch05-pca-result-dashboard-map",
        "pages": {
            "en": "docs/ch05-machine-learning/ch03-unsupervised/02-dimensionality-reduction.md",
            "zh": "i18n/zh-Hans/docusaurus-plugin-content-docs/current/ch05-machine-learning/ch03-unsupervised/02-dimensionality-reduction.md",
            "ja": "i18n/ja/docusaurus-plugin-content-docs/current/ch05-machine-learning/ch03-unsupervised/02-dimensionality-reduction.md",
        },
        "scene": "A PCA lab result dashboard for the sklearn digits dataset. Show a tiny handwritten digit image made of 64 pixel features entering StandardScaler and PCA. A 2D scatter map shows digit groups compressed to PC1 and PC2, with a warning badge that two components keep only 21.6% variance. Next to it, a components table shows 10 components: variance 0.591 accuracy 0.858, 20 components: variance 0.791 accuracy 0.942, 40 components: variance 0.953 accuracy 0.960. A reconstruction strip shows 10 components blurry, 20 readable, 40 clearer, with MSE dropping 0.390 -> 0.199 -> 0.045. The learner should see that PCA choice depends on visualization, modeling, or compression.",
        "chapter_context": "This image is inserted after the expected output of pca_lab.py. Nearby code prints pca_2d_map, explained variance for two components, model accuracy for 10, 20, and 40 components, and reconstruction MSE. The text explains 2D PCA is good for plotting but keeps too little variance for serious classification, while more components improve accuracy and reconstruction at the cost of more dimensions.",
        "shared_layout": "Vertical 9:16. Top title and subtitle. Upper left shows 8x8 digit pixels becoming a 64-feature vector, then scaler, then PCA. Upper right shows a PC1/PC2 scatter with the 21.6% variance warning. Middle has a three-row components dashboard for 10, 20, 40 with variance and accuracy. Bottom has reconstruction before/after strip with MSE values dropping. Keep data values, digit example, panels, and reading path identical across zh/en/ja.",
        "variants": {
            "zh": {
                "title": "PCA 实验结果仪表盘",
                "subtitle": "同一段输出要分三种目标读：画图、建模、压缩。",
                "items": [
                    ("64 pixels", "每张数字图先是 64 个特征。"),
                    ("2D map", "PC1/PC2 方便观察，只保留 21.6%。"),
                    ("10 components", "variance=0.591，accuracy=0.858。"),
                    ("20 components", "variance=0.791，accuracy=0.942。"),
                    ("40 components", "variance=0.953，accuracy=0.960。"),
                    ("reconstruction", "MSE 从 0.390 降到 0.045。"),
                ],
                "footer": "PCA 不是越少越好，而是看当前目标需要保留什么。",
                "alt": "PCA 实验结果仪表盘：手写数字从 64 特征压到 2D，比较 10、20、40 个组件的方差、准确率和重建误差。",
            },
            "en": {
                "title": "PCA Lab Result Dashboard",
                "subtitle": "Read one output three ways: visualization, modeling, and compression.",
                "items": [
                    ("64 pixels", "Each digit image starts as 64 features."),
                    ("2D map", "PC1/PC2 helps plotting but keeps only 21.6%."),
                    ("10 components", "variance=0.591, accuracy=0.858."),
                    ("20 components", "variance=0.791, accuracy=0.942."),
                    ("40 components", "variance=0.953, accuracy=0.960."),
                    ("reconstruction", "MSE drops from 0.390 to 0.045."),
                ],
                "footer": "PCA is not about using fewer at all costs; choose for the goal.",
                "alt": "PCA lab result dashboard: handwritten digits compress from 64 features to 2D, then compare 10, 20, and 40 components by variance, accuracy, and reconstruction error.",
            },
            "ja": {
                "title": "PCA 実験結果ダッシュボード",
                "subtitle": "同じ出力を、可視化、モデリング、圧縮の3目的で読む。",
                "items": [
                    ("64 pixels", "数字画像はまず 64 個の特徴量。"),
                    ("2D map", "PC1/PC2 は可視化向きだが 21.6% だけ保持。"),
                    ("10 components", "variance=0.591、accuracy=0.858。"),
                    ("20 components", "variance=0.791、accuracy=0.942。"),
                    ("40 components", "variance=0.953、accuracy=0.960。"),
                    ("reconstruction", "MSE は 0.390 から 0.045 へ下がる。"),
                ],
                "footer": "PCA は少なければ良いのではなく、目的に合わせて選ぶ。",
                "alt": "PCA 実験結果ダッシュボード：手書き数字を64特徴から2Dへ圧縮し、10、20、40 components の分散、精度、再構成誤差を比較する。",
            },
        },
    },
    {
        "slug": "ch06-training-loop-loss-checkpoint-map",
        "pages": {
            "en": "docs/ch06-deep-learning/ch02-pytorch/05-training-loop.md",
            "zh": "i18n/zh-Hans/docusaurus-plugin-content-docs/current/ch06-deep-learning/ch02-pytorch/05-training-loop.md",
            "ja": "i18n/ja/docusaurus-plugin-content-docs/current/ch06-deep-learning/ch02-pytorch/05-training-loop.md",
        },
        "scene": "A PyTorch training loop lab result board. Do not draw long code blocks or invented metrics. Show synthetic regression data for y = 3*x1 + 2*x2 + 5, split into train=192 and val=48. Show the train loop as batch -> forward -> loss -> backward -> optimizer.step, and validation as eval + no_grad with no update. A loss curve should show train and val loss dropping from epoch 1 to 80, with exact callouts: epoch 1 val_loss=25.3358, epoch 20 val_loss=0.0856, epoch 60 val_loss=0.0760, epoch 80 val_loss=0.0776. Mark best_val=0.0734 as a checkpoint badge. Bottom shows three prediction checks: [1,2] -> 12.05 close to 12, [-1,0.5] -> 3.00 close to 3, [0,0] -> 4.98 close to 5.",
        "chapter_context": "This image is inserted after the expected output of the complete runnable PyTorch training loop. Nearby code trains a small regressor, prints train and validation loss every 20 epochs, keeps the best validation state_dict with copy.deepcopy, loads the best state, and prints three predictions close to the noiseless values 12, 3, and 5.",
        "shared_layout": "Vertical 9:16. Top title and subtitle. Upper section shows dataset formula and train/val split. Middle section has two lanes: training lane with gradient update and validation lane with eval/no_grad/no update. Center has a loss curve with only the exact epoch callouts from the scene and a best checkpoint marker. Bottom has three prediction cards comparing predicted value to true noiseless value. Same panel order, colors, and numbers across zh/en/ja.",
        "variants": {
            "zh": {
                "title": "PyTorch 训练循环结果怎么看",
                "subtitle": "看 loss 下降、验证集 checkpoint，以及预测是否接近真值。",
                "items": [
                    ("train=192 / val=48", "训练更新参数，验证只检查泛化。"),
                    ("train loop", "forward、loss、backward、optimizer.step。"),
                    ("eval + no_grad", "验证不记录梯度，也不更新权重。"),
                    ("best_val=0.0734", "保存验证集最好的 checkpoint。"),
                    ("epoch 80", "val_loss=0.0776，略高于最好值。"),
                    ("prediction check", "12.05≈12，3.00≈3，4.98≈5。"),
                ],
                "footer": "训练循环的证据不是最后一行，而是曲线、最好 checkpoint 和预测检查。",
                "alt": "PyTorch 训练循环结果图：训练和验证 loss 下降，best_val checkpoint 被保存，三个测试点预测接近真实线性值。",
            },
            "en": {
                "title": "Reading PyTorch Training Loop Results",
                "subtitle": "Read loss drop, validation checkpoint, and predictions near true values.",
                "items": [
                    ("train=192 / val=48", "Training updates weights; validation checks generalization."),
                    ("train loop", "forward, loss, backward, optimizer.step."),
                    ("eval + no_grad", "Validation records no gradients and makes no update."),
                    ("best_val=0.0734", "Save the checkpoint with best validation loss."),
                    ("epoch 80", "val_loss=0.0776, slightly above the best value."),
                    ("prediction check", "12.05≈12, 3.00≈3, 4.98≈5."),
                ],
                "footer": "Training evidence is the curve, best checkpoint, and prediction check.",
                "alt": "PyTorch training loop result map: train and validation loss drop, best_val checkpoint is saved, and three test predictions are close to true linear values.",
            },
            "ja": {
                "title": "PyTorch 学習ループ結果の読み方",
                "subtitle": "loss の低下、検証 checkpoint、真値に近い予測を読む。",
                "items": [
                    ("train=192 / val=48", "学習は重みを更新し、検証は汎化を確認。"),
                    ("train loop", "forward、loss、backward、optimizer.step。"),
                    ("eval + no_grad", "検証では勾配を記録せず、更新もしない。"),
                    ("best_val=0.0734", "検証 loss が最良の checkpoint を保存。"),
                    ("epoch 80", "val_loss=0.0776、最良値より少し高い。"),
                    ("prediction check", "12.05≈12、3.00≈3、4.98≈5。"),
                ],
                "footer": "学習の証拠は最後の行ではなく、曲線、best checkpoint、予測確認。",
                "alt": "PyTorch 学習ループ結果図：train と validation loss が下がり、best_val checkpoint を保存し、3つの予測が真の線形式に近い。",
            },
        },
    },
    {
        "slug": "ch06-cnn-four-class-result-map",
        "pages": {
            "en": "docs/ch06-deep-learning/ch03-cnn/05-image-classification-practice.md",
            "zh": "i18n/zh-Hans/docusaurus-plugin-content-docs/current/ch06-deep-learning/ch03-cnn/05-image-classification-practice.md",
            "ja": "i18n/ja/docusaurus-plugin-content-docs/current/ch06-deep-learning/ch03-cnn/05-image-classification-practice.md",
        },
        "scene": "A CNN four-class image classification lab result board. Do not draw long code blocks. Show four 16x16 grayscale pattern classes: vertical, horizontal, diagonal down, diagonal up. Show tensor shapes: train (384,1,16,16), val (96,1,16,16), features (4,32,1,1), logits (4,4). Show the training result: epoch 1 val_acc=0.188, epoch 20 val_acc=1.000, epoch 80 val_acc=1.000. Show a confusion matrix with rows=true columns=pred and exact diagonal counts 30, 22, 18, 26 and all off-diagonal cells zero. Bottom shows a sample card: true vertical, pred vertical, probs [1.0, 0.0, 0.0, 0.0].",
        "chapter_context": "This image is inserted after the expected output of the CNN practice script. Nearby code creates synthetic line-pattern images, keeps tensors in N,C,H,W format, trains a TinyCNNClassifier, prints shapes, validation loss and accuracy, prints a confusion matrix, and inspects a single sample prediction.",
        "shared_layout": "Vertical 9:16. Top title and subtitle. Upper section has four sample image tiles with labels. Middle left shows tensor shape flow into CNN feature compression and logits. Middle right shows training accuracy milestones. Lower section has the confusion matrix with exact diagonal counts and zero mistakes. Bottom has one sample prediction card with class probabilities. Same layout, colors, labels, and numbers across zh/en/ja.",
        "variants": {
            "zh": {
                "title": "CNN 四分类实验结果",
                "subtitle": "从样本图、张量形状、训练曲线到混淆矩阵一起读。",
                "items": [
                    ("4 classes", "vertical、horizontal、diag_down、diag_up。"),
                    ("N,C,H,W", "train=(384,1,16,16)，val=(96,1,16,16)。"),
                    ("features", "CNN 压缩到 (4,32,1,1)。"),
                    ("logits", "4 个样本，各有 4 个类别分数。"),
                    ("val_acc=1.000", "第 20 轮后验证集全对。"),
                    ("confusion matrix", "30、22、18、26 都在对角线上。"),
                ],
                "footer": "图像分类实验要同时看形状、学习进展和错误位置。",
                "alt": "CNN 四分类实验结果图：四类线条样本、NCHW 张量形状、验证准确率、对角线混淆矩阵和单样本概率。",
            },
            "en": {
                "title": "CNN Four-Class Lab Results",
                "subtitle": "Read sample images, tensor shapes, training progress, and confusion matrix together.",
                "items": [
                    ("4 classes", "vertical, horizontal, diag_down, diag_up."),
                    ("N,C,H,W", "train=(384,1,16,16), val=(96,1,16,16)."),
                    ("features", "CNN compresses to (4,32,1,1)."),
                    ("logits", "4 samples, 4 class scores each."),
                    ("val_acc=1.000", "Validation is perfect after epoch 20."),
                    ("confusion matrix", "30, 22, 18, 26 sit on the diagonal."),
                ],
                "footer": "Image classification evidence combines shapes, learning progress, and error locations.",
                "alt": "CNN four-class lab result map: four line-pattern classes, NCHW tensor shapes, validation accuracy, diagonal confusion matrix, and one sample probability card.",
            },
            "ja": {
                "title": "CNN 4分類実験の結果",
                "subtitle": "サンプル画像、tensor 形状、学習進捗、混同行列を一緒に読む。",
                "items": [
                    ("4 classes", "vertical、horizontal、diag_down、diag_up。"),
                    ("N,C,H,W", "train=(384,1,16,16)、val=(96,1,16,16)。"),
                    ("features", "CNN が (4,32,1,1) へ圧縮。"),
                    ("logits", "4サンプル、それぞれ4クラスの score。"),
                    ("val_acc=1.000", "epoch 20 以降は検証が全問正解。"),
                    ("confusion matrix", "30、22、18、26 が対角線上に並ぶ。"),
                ],
                "footer": "画像分類の証拠は、形状、学習進捗、誤り位置を合わせて読む。",
                "alt": "CNN 4分類実験結果図：4種類の線パターン、NCHW tensor 形状、検証精度、対角線の混同行列、単一サンプルの確率。",
            },
        },
    },
    {
        "slug": "ch06-lstm-forecast-result-curve-map",
        "pages": {
            "en": "docs/ch06-deep-learning/ch04-rnn/03-sequence-practice.md",
            "zh": "i18n/zh-Hans/docusaurus-plugin-content-docs/current/ch06-deep-learning/ch04-rnn/03-sequence-practice.md",
            "ja": "i18n/ja/docusaurus-plugin-content-docs/current/ch06-deep-learning/ch04-rnn/03-sequence-practice.md",
        },
        "scene": "An LSTM forecasting lab result board. Do not draw code blocks, terminal windows, file names, or extra tables. Show only these three learning evidences: (1) continuous series -> sliding windows -> time-order split, with X=(204,16,1), train=(163,16,1), val=(41,16,1); (2) a compact MSE comparison card: naive_val_mse=0.0115, LSTM val_mse=0.0030, LSTM beats baseline; (3) a true vs predicted validation line chart for exactly the first five points: pred [0.323, 0.261, 0.145, -0.025, -0.192], true [0.400, 0.213, 0.045, -0.076, -0.128]. Optionally show a small validation MSE curve with only these exact labels: epoch 1 val_mse=0.4633 and epoch 120 val_mse=0.0030. Critical accuracy rules: never write 0.115; never write -0.4633; never invent baseline.txt, saved files, commands, or extra numeric rows.",
        "chapter_context": "This image is inserted after the expected output of the LSTM forecasting lab. Nearby code creates sliding windows, splits in time order, compares naive last-value baseline, trains an LSTM, prints validation MSE, and prints first five predictions versus true values.",
        "shared_layout": "Vertical 9:16. Top title and subtitle. Upper third shows continuous series -> 16-step window -> time-order split. Middle third shows two large result cards, naive_val_mse=0.0115 versus LSTM val_mse=0.0030, plus a small falling MSE curve. Lower third shows the five-point true vs pred line chart with clear labels. Use large readable text, no dense tables, no fake terminal panels. Same layout, colors, numbers, and reading path across zh/en/ja.",
        "variants": {
            "zh": {
                "title": "LSTM 预测实验结果",
                "subtitle": "时间序列要看窗口、时间顺序、baseline 和预测曲线。",
                "items": [
                    ("windows", "X=(204,16,1)，每行看 16 个时间步。"),
                    ("time split", "train=163，val=41，不能随机泄漏未来。"),
                    ("naive baseline", "last value baseline MSE=0.0115。"),
                    ("LSTM val_mse", "从 0.4633 降到 0.0030。"),
                    ("first 5", "pred 和 true 方向接近。"),
                    ("beats baseline", "0.0030 明显低于 0.0115。"),
                ],
                "footer": "序列模型不能只看 MSE，还要画预测曲线检查滞后和形状。",
                "alt": "LSTM 时间序列预测结果图：滑动窗口、时间顺序切分、naive baseline、验证 MSE 曲线和前五个 true/pred 对比。",
            },
            "en": {
                "title": "LSTM Forecast Lab Results",
                "subtitle": "Read windows, time split, baseline, and prediction curve together.",
                "items": [
                    ("windows", "X=(204,16,1): each row sees 16 time steps."),
                    ("time split", "train=163, val=41; do not leak the future."),
                    ("naive baseline", "Last-value baseline MSE=0.0115."),
                    ("LSTM val_mse", "Drops from 0.4633 to 0.0030."),
                    ("first 5", "Pred and true move in similar directions."),
                    ("beats baseline", "0.0030 is far below 0.0115."),
                ],
                "footer": "For sequence models, plot predictions to check lag and shape, not only MSE.",
                "alt": "LSTM time-series forecast result map: sliding windows, time-order split, naive baseline, validation MSE curve, and first five true/pred comparisons.",
            },
            "ja": {
                "title": "LSTM 予測実験の結果",
                "subtitle": "window、時系列 split、baseline、予測曲線を一緒に読む。",
                "items": [
                    ("windows", "X=(204,16,1)、各行が16時点を見る。"),
                    ("time split", "train=163、val=41、未来を漏らさない。"),
                    ("naive baseline", "直前値 baseline の MSE=0.0115。"),
                    ("LSTM val_mse", "0.4633 から 0.0030 へ低下。"),
                    ("first 5", "pred と true の方向が近い。"),
                    ("beats baseline", "0.0030 は 0.0115 よりかなり低い。"),
                ],
                "footer": "系列モデルは MSE だけでなく、予測曲線で遅れと形を確認する。",
                "alt": "LSTM 時系列予測結果図：スライディング window、時系列 split、naive baseline、validation MSE 曲線、最初の5点の true/pred 比較。",
            },
        },
    },
    {
        "slug": "ch06-gan-1d-distribution-result-map",
        "pages": {
            "en": "docs/ch06-deep-learning/ch06-generative/01-gan.md",
            "zh": "i18n/zh-Hans/docusaurus-plugin-content-docs/current/ch06-deep-learning/ch06-generative/01-gan.md",
            "ja": "i18n/ja/docusaurus-plugin-content-docs/current/ch06-deep-learning/ch06-generative/01-gan.md",
        },
        "scene": "A tiny 1D GAN lab result board. Do not draw code blocks, terminal windows, or any loss_d/loss_g numeric values. Do not invent extra metrics. Use one consistent dark lab dashboard style for all languages. Show exactly two ideas. First: a simple two-step adversarial loop: Step 1 train D with real data centered near 2.0 and fake.detach(); Step 2 train G so D says real. Second: four separate fake distribution result cards, not one shared axis. Each card has a mini bell curve, a small marker saying where the fake_mean sits relative to real center 2.0, and the exact printed values: step 001 fake_mean=0.025 fake_std=0.117; step 100 fake_mean=1.093 fake_std=0.204; step 200 fake_mean=2.988 fake_std=0.291; step 300 fake_mean=1.384 fake_std=0.056. Mark step 200 as overshoot and step 300 as low diversity / possible mode collapse, not as the best final line.",
        "chapter_context": "This image is inserted after the expected output of the tiny 1D GAN lab. Nearby text says not to read the final line as best, because real data is centered near 2.0, G and D chase each other, fake_std becoming small is a low-diversity or mode-collapse warning, and GAN loss curves are hard to interpret without sample inspection.",
        "shared_layout": "Vertical 9:16. Top title and subtitle. Upper half shows only the two-step D/G loop with detach clearly placed on the D step. Middle has a real center reference card: real center = 2.0. Lower half is a 2x2 grid of result cards for step 001, 100, 200, and 300; each card includes mean, std, and a mini distribution. Bottom warning says step 300 has low diversity and the last line is not automatically best. Use large readable labels, no dense tables, no shared axis that can misplace the curves, no loss-number panels. Same dark dashboard layout, colors, values, and reading path across zh/en/ja.",
        "variants": {
            "zh": {
                "title": "GAN 1D 实验结果怎么看",
                "subtitle": "不要只看最后一行，要看分布是否靠近真实且保持多样性。",
                "items": [
                    ("real center", "真实分布中心在 2.0 附近。"),
                    ("step 001", "fake_mean=0.025，离真实还远。"),
                    ("step 200", "fake_mean=2.988，追过头了。"),
                    ("step 300", "fake_std=0.056，多样性太低。"),
                    ("detach", "训练 D 时 fake 不更新 G。"),
                    ("mode collapse", "看起来像真，但样本太单一。"),
                ],
                "footer": "GAN 的证据是样本分布、均值、方差和 loss 一起看。",
                "alt": "1D GAN 实验结果图：真实分布中心约 2.0，fake_mean 随训练追逐变化，fake_std 过低提示 mode collapse。",
            },
            "en": {
                "title": "Reading 1D GAN Lab Results",
                "subtitle": "Do not trust the last line alone; inspect real match and diversity.",
                "items": [
                    ("real center", "Real samples center near 2.0."),
                    ("step 001", "fake_mean=0.025, still far from real."),
                    ("step 200", "fake_mean=2.988, overshoots the target."),
                    ("step 300", "fake_std=0.056, diversity is too low."),
                    ("detach", "D training uses fake without updating G."),
                    ("mode collapse", "Looks plausible but becomes repetitive."),
                ],
                "footer": "GAN evidence combines sample distribution, mean, std, and losses.",
                "alt": "1D GAN lab result map: real distribution centers near 2.0, fake_mean changes as G and D chase each other, and low fake_std warns about mode collapse.",
            },
            "ja": {
                "title": "1D GAN 実験結果の読み方",
                "subtitle": "最後の行だけでなく、実分布との近さと多様性を見る。",
                "items": [
                    ("real center", "実データの中心は 2.0 付近。"),
                    ("step 001", "fake_mean=0.025、まだ実分布から遠い。"),
                    ("step 200", "fake_mean=2.988、目標を越えすぎ。"),
                    ("step 300", "fake_std=0.056、多様性が低すぎる。"),
                    ("detach", "D の学習では fake で G を更新しない。"),
                    ("mode collapse", "本物らしいが出力が単調になる。"),
                ],
                "footer": "GAN の証拠は sample 分布、mean、std、loss を合わせて読む。",
                "alt": "1D GAN 実験結果図：実分布は 2.0 付近、fake_mean は追いかけながら変化し、fake_std が低いと mode collapse の警告になる。",
            },
        },
    },
    {
        "slug": "ch06-vae-2d-latent-sample-result-map",
        "pages": {
            "en": "docs/ch06-deep-learning/ch06-generative/02-vae.md",
            "zh": "i18n/zh-Hans/docusaurus-plugin-content-docs/current/ch06-deep-learning/ch06-generative/02-vae.md",
            "ja": "i18n/ja/docusaurus-plugin-content-docs/current/ch06-deep-learning/ch06-generative/02-vae.md",
        },
        "scene": "A tiny 2D VAE lab result board. Do not draw code blocks, terminal windows, dense tables, or any extra epoch rows. Do not invent extra generated points. Show two input clusters centered near [1,0] and [-1,0]. Show VAE flow: encoder -> mu/logvar -> reparameterize z = mu + eps*std -> decoder. Show exactly three result cards with large labels: recon 0.5903 -> 0.0244; KL at epoch 200 = 0.7138; loss 0.5917 -> 0.0601. Critical accuracy rule: KL is 0.7138, not 0.0601. Show generated points from random z as five plotted dots near the two clusters; label only the two anchor points [1.075,-0.014] and [-0.997,-0.001], and list the other three smaller: [-1.118,-0.054], [0.553,0.041], [0.740,0.021]. Make clear these are decoded from random z, not copied from training data.",
        "chapter_context": "This image is inserted after the expected output of the tiny VAE 2D lab. Nearby text explains encoder, mu, logvar, reparameterization, reconstruction loss, KL regularization, generated_points, and the tradeoff between reconstruction and a smooth samplable latent space.",
        "shared_layout": "Vertical 9:16. Top title and subtitle. Upper section shows two 2D input clusters and VAE flow with mu/logvar and z sampling. Middle section has three large metric cards: recon, KL, loss, using only the exact summary values from the scene. Lower section plots five generated 2D dots near the two clusters, with two large anchor coordinate labels and a compact list for the other three points. Bottom says random z is decoded, not copied, and shows reconstruction versus KL tradeoff. Same dark dashboard layout, colors, values, and reading path across zh/en/ja.",
        "variants": {
            "zh": {
                "title": "VAE 2D 实验结果怎么看",
                "subtitle": "看重建误差下降、KL 受控，以及随机 z 是否解码成合理点。",
                "items": [
                    ("input clusters", "训练点围绕 [1,0] 和 [-1,0]。"),
                    ("mu / logvar", "encoder 输出潜在分布。"),
                    ("z sample", "z = mu + eps * std，梯度仍能流动。"),
                    ("recon", "0.5903 降到 0.0244。"),
                    ("KL", "不是越小越好，要受控。"),
                    ("generated_points", "随机 z 解码出接近数据区域的点。"),
                ],
                "footer": "VAE 学的是可采样的潜在空间，不只是复制输入。",
                "alt": "VAE 2D 实验结果图：两簇输入点经过 encoder、mu/logvar、reparameterize、decoder，重建误差下降，随机 z 生成合理点。",
            },
            "en": {
                "title": "Reading 2D VAE Lab Results",
                "subtitle": "Read reconstruction drop, controlled KL, and decoded random z samples.",
                "items": [
                    ("input clusters", "Training points sit near [1,0] and [-1,0]."),
                    ("mu / logvar", "Encoder outputs a latent distribution."),
                    ("z sample", "z = mu + eps * std keeps gradients flowing."),
                    ("recon", "Drops from 0.5903 to 0.0244."),
                    ("KL", "Not zero; controlled pressure is useful."),
                    ("generated_points", "Random z decodes near data regions."),
                ],
                "footer": "A VAE learns a samplable latent space, not just copied inputs.",
                "alt": "2D VAE lab result map: two input clusters pass through encoder, mu/logvar, reparameterization, and decoder; reconstruction drops and random z generates plausible points.",
            },
            "ja": {
                "title": "2D VAE 実験結果の読み方",
                "subtitle": "再構成誤差の低下、制御された KL、random z の decode を読む。",
                "items": [
                    ("input clusters", "学習点は [1,0] と [-1,0] 付近。"),
                    ("mu / logvar", "encoder が潜在分布を出す。"),
                    ("z sample", "z = mu + eps * std で勾配が流れる。"),
                    ("recon", "0.5903 から 0.0244 へ低下。"),
                    ("KL", "0 にすれば良いのではなく、制御する。"),
                    ("generated_points", "random z がデータ領域近くへ decode される。"),
                ],
                "footer": "VAE は入力コピーではなく、sample できる latent space を学ぶ。",
                "alt": "2D VAE 実験結果図：2つの入力クラスタが encoder、mu/logvar、reparameterization、decoder を通り、再構成誤差が下がり random z から妥当な点を生成する。",
            },
        },
    },
    {
        "slug": "ch07-tokenizer-embedding-lab-result-map",
        "pages": {
            "en": "docs/ch07-llm-principles/ch01-nlp-crash/05-tokenizer-embedding-lab.md",
            "zh": "i18n/zh-Hans/docusaurus-plugin-content-docs/current/ch07-llm-principles/ch01-nlp-crash/05-tokenizer-embedding-lab.md",
            "ja": "i18n/ja/docusaurus-plugin-content-docs/current/ch07-llm-principles/ch01-nlp-crash/05-tokenizer-embedding-lab.md",
        },
        "scene": "A tokenizer plus embedding lab result board. Do not draw code blocks, terminal windows, dense full print logs, or invented model dimensions such as 768. Show the actual runtime evidence from the lesson as a teaching workflow: raw text -> tokens -> input_ids -> attention_mask -> sentence vector -> cosine similarity. Use exactly three input texts and exact values. Text 1: 'please help reset password'; tokens [CLS], please, help, reset, password, [SEP]; input_ids [2,8,9,4,5,3]; attention_mask [1,1,1,1,1,1]; sentence_vec [0.260,0.307,0.662]. Text 2: 'reset password'; tokens [CLS], reset, password, [SEP], [PAD], [PAD]; input_ids [2,4,5,3,0,0]; attention_mask [1,1,1,1,0,0]; sentence_vec [0.110,0.190,0.935]. Text 3: 'refund order'; tokens [CLS], refund, order, [SEP], [PAD], [PAD]; input_ids [2,6,7,3,0,0]; attention_mask [1,1,1,1,0,0]; sentence_vec [0.825,0.750,0.125]. Critical rule: refund=6 and order=7, never swap them. Show an embedding lookup scene where IDs become 3-number vectors and special tokens are excluded from the simple average. Bottom similarity board must use exactly: text1 vs text2 = 0.949, text1 vs text3 = 0.607.",
        "chapter_context": "This image is inserted after the expected output of tokenizer_embedding_lab.py. Nearby code tokenizes three short texts, maps tokens to input_ids, creates padding and attention_mask, averages content-token embeddings while excluding special tokens, then computes cosine similarity. The page teaches how printed token lists, masks, vectors, and similarity scores connect into the first bridge from human text to model computation.",
        "shared_layout": "Vertical 9:16. Top title and subtitle. Upper section shows the main text 1 flowing through tokens, input_ids, mask, embedding lookup, and sentence vector with large readable chips. Middle section has two compact comparison cards for text 2 and text 3, including their exact tokens, input_ids, masks, and sentence vectors. Lower section has a semantic vector map or arrows: text 1 near text 2, text 3 farther away. Bottom scoreboard shows 0.949 versus 0.607 and a short note that cosine similarity compares vector direction. Same dark lab dashboard layout, colors, numbers, and reading path across zh/en/ja. No dense tables; no invented token IDs, vector values, or model dimensions.",
        "variants": {
            "zh": {
                "title": "Tokenizer + Embedding 输出怎么看",
                "subtitle": "从 token、ID、mask 到句向量和相似度，把 print 串成证据。",
                "items": [
                    ("raw text", "please help reset password"),
                    ("tokens", "[CLS]、please、help、reset、password、[SEP]。"),
                    ("input_ids", "[2,8,9,4,5,3] 是模型能读的编号。"),
                    ("attention_mask", "短句的 [PAD] 位置用 0 忽略。"),
                    ("sentence_vec", "内容词向量平均后得到句向量。"),
                    ("similarity", "0.949 比 0.607 更近。"),
                ],
                "footer": "文本先变成 ID，再变成向量；相似度比较的是向量方向。",
                "alt": "Tokenizer 与 Embedding 实验结果图：raw text 经过 tokens、input_ids、attention_mask、embedding 平均得到句向量，并比较 0.949 与 0.607 相似度。",
            },
            "en": {
                "title": "Reading Tokenizer + Embedding Output",
                "subtitle": "Connect token, ID, mask, sentence vector, and similarity prints as evidence.",
                "items": [
                    ("raw text", "please help reset password"),
                    ("tokens", "[CLS], please, help, reset, password, [SEP]."),
                    ("input_ids", "[2,8,9,4,5,3] are model-readable IDs."),
                    ("attention_mask", "PAD positions in shorter texts are ignored with 0."),
                    ("sentence_vec", "Average content-token vectors into a sentence vector."),
                    ("similarity", "0.949 is closer than 0.607."),
                ],
                "footer": "Text becomes IDs, IDs become vectors, and similarity compares vector direction.",
                "alt": "Tokenizer and embedding lab result map: raw text becomes tokens, input_ids, attention_mask, averaged embeddings, sentence vectors, then similarity scores 0.949 and 0.607.",
            },
            "ja": {
                "title": "Tokenizer + Embedding 出力の読み方",
                "subtitle": "token、ID、mask、文ベクトル、類似度の print を証拠としてつなぐ。",
                "items": [
                    ("raw text", "please help reset password"),
                    ("tokens", "[CLS]、please、help、reset、password、[SEP]。"),
                    ("input_ids", "[2,8,9,4,5,3] はモデルが読める ID。"),
                    ("attention_mask", "短い文の [PAD] 位置は 0 で無視する。"),
                    ("sentence_vec", "内容 token のベクトルを平均して文ベクトルにする。"),
                    ("similarity", "0.949 は 0.607 より近い。"),
                ],
                "footer": "文は ID になり、ID はベクトルになり、類似度は方向を比べる。",
                "alt": "Tokenizer と Embedding 実験結果図：raw text が tokens、input_ids、attention_mask、平均 embedding、文ベクトルになり、0.949 と 0.607 の類似度を比較する。",
            },
        },
    },
    {
        "slug": "ch07-llm-call-workbench-validation-trace",
        "pages": {
            "en": "docs/ch07-llm-principles/ch02-llm-overview/04-llm-call-workbench.md",
            "zh": "i18n/zh-Hans/docusaurus-plugin-content-docs/current/ch07-llm-principles/ch02-llm-overview/04-llm-call-workbench.md",
            "ja": "i18n/ja/docusaurus-plugin-content-docs/current/ch07-llm-principles/ch02-llm-overview/04-llm-call-workbench.md",
        },
        "scene": "An LLM call workbench runtime trace. Do not draw code blocks, terminal windows, or full JSON paragraphs. Show the actual expected output shape from the offline lab as an engineering loop. Top: token budget card with CONTEXT_LIMIT=4096, used input tokens estimate=36, max_output_tokens=600, remaining output room=3460, request model=gpt-5.5. Middle: payload parts as chips: model, instructions, input, text.format=json_object, max_output_tokens, temperature=0.3. Then attempt 1 card: model returns timeline JSON missing summary in era 0; validator result exactly era_0_missing_['summary']; status failed. Retry fix card: add 'Do not omit any required field.' and lower temperature to 0.1. Attempt 2 card: validation=valid and first era has period 1936-1950. Bottom teaching point: printing a nice answer is demo; parse, validate, and retry with a cause turns it into a workflow.",
        "chapter_context": "This image is inserted after the expected output shape of llm_call_workbench.py. Nearby code builds a payload, estimates token budget, uses a fake first model response missing the summary field, validates the JSON schema, changes the instructions and temperature for retry, and accepts the second response. The page teaches that LLM engineering is a controlled request-parse-validate-retry loop, not just prompt text.",
        "shared_layout": "Vertical 9:16. Top title and subtitle. Upper section shows token budget as a clear allocation bar: 36 used input, 600 output cap, 3460 remaining, within 4096. Middle section shows request payload chips and the validation gate. Lower section is a two-attempt timeline: attempt 1 failed because summary is missing, retry fixes the cause, attempt 2 is valid. Bottom has a reliability checklist: parse JSON, check schema, explain failure, retry with change. Same dark workbench layout, colors, numbers, validation reason, and reading path across zh/en/ja. No full JSON blocks; no invented token counts or model names.",
        "variants": {
            "zh": {
                "title": "LLM 调用结果怎么读",
                "subtitle": "一次可靠调用要经过预算、payload、解析、校验和有原因的重试。",
                "items": [
                    ("Token 预算", "4096 上限，输入约 36，输出上限 600，剩余 3460。"),
                    ("payload", "model、instructions、input、format、temperature。"),
                    ("attempt 1", "era 0 缺少 summary，校验失败。"),
                    ("validation", "era_0_missing_['summary']"),
                    ("retry fix", "加强必填字段说明，temperature 降到 0.1。"),
                    ("attempt 2", "validation=valid，first era 可被程序使用。"),
                ],
                "footer": "只打印答案是 demo；能解析、校验、定位失败并重试，才是工作流。",
                "alt": "LLM 调用工作台结果图：Token 预算、payload 字段、第一次 schema 校验失败、按原因重试、第二次 validation valid。",
            },
            "en": {
                "title": "Reading an LLM Call Trace",
                "subtitle": "A reliable call moves through budget, payload, parsing, validation, and cause-aware retry.",
                "items": [
                    ("token budget", "4096 limit, about 36 input, 600 output cap, 3460 remaining."),
                    ("payload", "model, instructions, input, format, temperature."),
                    ("attempt 1", "era 0 misses summary, so validation fails."),
                    ("validation", "era_0_missing_['summary']"),
                    ("retry fix", "Strengthen required fields and lower temperature to 0.1."),
                    ("attempt 2", "validation=valid; first era is usable by code."),
                ],
                "footer": "Printing an answer is a demo; parse, validate, diagnose, and retry to make a workflow.",
                "alt": "LLM call workbench result map: token budget, payload fields, first schema validation failure, cause-aware retry, and second validation valid.",
            },
            "ja": {
                "title": "LLM 呼び出し結果の読み方",
                "subtitle": "信頼できる呼び出しは、予算、payload、parse、検証、原因つき retry を通る。",
                "items": [
                    ("Token 予算", "上限 4096、入力約 36、出力上限 600、残り 3460。"),
                    ("payload", "model、instructions、input、format、temperature。"),
                    ("attempt 1", "era 0 に summary がなく、検証に失敗。"),
                    ("validation", "era_0_missing_['summary']"),
                    ("retry fix", "必須フィールドを強め、temperature を 0.1 に下げる。"),
                    ("attempt 2", "validation=valid、first era をコードで使える。"),
                ],
                "footer": "回答を表示するだけなら demo。parse、検証、原因特定、retry で workflow になる。",
                "alt": "LLM 呼び出しワークベンチ結果図：Token 予算、payload fields、1回目の schema 検証失敗、原因を直す retry、2回目の validation valid。",
            },
        },
    },
    {
        "slug": "ch02-functional-pipeline",
        "pages": {
            "en": "docs/ch02-python/ch02-advanced/04-functional.md",
            "zh": "i18n/zh-Hans/docusaurus-plugin-content-docs/current/ch02-python/ch02-advanced/04-functional.md",
            "ja": "i18n/ja/docusaurus-plugin-content-docs/current/ch02-python/ch02-advanced/04-functional.md",
        },
        "scene": "A Python functional programming worked example, shown as a real data-processing desk rather than a decorative pipe. The input is a small list of student records with messy names and score strings. Function cards are passed into map, filter, and sorted: map cleans the name and converts score to int, filter keeps passing students, sorted uses key=lambda s: s['score'] to order by score, and the output is a clean ranked list. Also show a small first-class function note: a function card can be assigned to a variable and passed as an argument.",
        "chapter_context": "This image appears at the start of the functional programming basics section. The nearby text says lambda, map, filter, sorted key, closures, and decorators appear in data processing, framework source code, and utility functions. It says beginners do not need to chase elegance; functional style is often used for batch transformation, filtering, sorting, and passing custom logic into frameworks. Later examples show map(lambda x: x ** 2), filter(lambda x: x % 2 == 0), sorted(..., key=lambda s: s['score']), and functions as first-class citizens.",
        "shared_layout": "Vertical 9:16. Top title and subtitle. Upper area shows input student records. Middle has three large worked stations in order: map, filter, sorted(key=lambda...). Each station receives a visible function card and shows before/after records, not just abstract icons. Bottom shows ranked output plus a small note that functions can be passed like data. Keep the same records, station order, colors, and reading path across zh/en/ja.",
        "variants": {
            "zh": {
                "title": "函数式数据流水线",
                "subtitle": "把小函数作为参数传进去，让数据按步骤变干净。",
                "items": [
                    ("input records", "姓名有空格，score 还是字符串。"),
                    ("function card", "函数可以像数据一样传给工具。"),
                    ("map()", "对每条记录做同样清洗。"),
                    ("filter()", "只保留 score >= 60 的学生。"),
                    ("sorted(key=...)", "用 lambda 指定按 score 排序。"),
                    ("output", "得到干净、筛选后、排好序的列表。"),
                ],
                "footer": "函数式写法的重点：转换、筛选、排序都由小函数控制。",
                "alt": "Python 函数式数据流水线：学生记录经过 map 清洗、filter 筛选及格、sorted key=lambda 按 score 排序，展示函数作为参数传入。",
            },
            "en": {
                "title": "Functional Data Pipeline",
                "subtitle": "Pass small functions into tools so data changes step by step.",
                "items": [
                    ("input records", "Names have spaces; score is still a string."),
                    ("function card", "A function can be passed like data."),
                    ("map()", "Clean every record the same way."),
                    ("filter()", "Keep only students with score >= 60."),
                    ("sorted(key=...)", "Use lambda to sort by score."),
                    ("output", "Get a clean, filtered, ranked list."),
                ],
                "footer": "Functional style means transform, filter, and sort with small functions.",
                "alt": "Python functional data pipeline: student records pass through map cleaning, filter for passing scores, sorted key=lambda by score, showing functions passed as arguments.",
            },
            "ja": {
                "title": "関数型データパイプライン",
                "subtitle": "小さな関数を引数として渡し、データを段階的に整える。",
                "items": [
                    ("input records", "名前に空白があり、score は文字列のまま。"),
                    ("function card", "関数はデータのように渡せる。"),
                    ("map()", "各レコードを同じ規則で整える。"),
                    ("filter()", "score >= 60 の学生だけ残す。"),
                    ("sorted(key=...)", "lambda で score 順を指定する。"),
                    ("output", "整形、抽出、並べ替え済みのリスト。"),
                ],
                "footer": "関数型の要点：変換、抽出、並べ替えを小さな関数で制御する。",
                "alt": "Python 関数型データパイプライン：学生 records を map で整形し、filter で合格点を残し、sorted key=lambda で score 順に並べる。",
            },
        },
    },
    {
        "slug": "ch03-numpy-overview-array-engine",
        "pages": {
            "en": "docs/ch03-data-analysis/ch02-numpy/01-overview.md",
            "zh": "i18n/zh-Hans/docusaurus-plugin-content-docs/current/ch03-data-analysis/ch02-numpy/01-overview.md",
            "ja": "i18n/ja/docusaurus-plugin-content-docs/current/ch03-data-analysis/ch02-numpy/01-overview.md",
        },
        "scene": "A NumPy lesson workbench. Put one ndarray cube engine in the center. On the left, a slow Python list loop tries to multiply numbers one by one. In the middle, ndarray stores the same numbers as a regular multi-dimensional block with shape and axis direction visible. On the right, the engine powers Pandas table analysis, charting, machine learning arrays, and deep learning tensors. Show that NumPy is the shared array layer under later libraries, not just a standalone tool.",
        "chapter_context": "This image appears under 'What Is NumPy?'. The surrounding text says NumPy is the core scientific computing library, an engine for Python data science, and that later Pandas, visualization, machine learning, deep learning, and scientific computing build on ndarray. The following section contrasts Python lists with NumPy vectorized operations.",
        "shared_layout": "Vertical 9:16. Top title, then a large central ndarray engine. Left side shows slow Python list loop; center shows shape/axis ndarray block; right side has four downstream application stations. Bottom footer summarizes that NumPy is the array engine under data and AI libraries. Keep composition, colors, engine position, and four stations identical across zh/en/ja.",
        "variants": {
            "zh": {
                "title": "NumPy 是数组引擎",
                "subtitle": "先理解 ndarray，后面的数据与 AI 库才有共同底座。",
                "items": [
                    ("Python list", "循环逐个处理，写法笨重。"),
                    ("ndarray", "规则多维数组，适合批量计算。"),
                    ("shape / axis", "描述数据形状和计算方向。"),
                    ("Pandas", "表格处理建立在数组能力上。"),
                    ("可视化", "图表读取数组和统计结果。"),
                    ("ML / DL", "模型训练大量使用矩阵和张量。"),
                ],
                "footer": "NumPy 不是孤立工具，而是 Python 数据科学的数组底座。",
                "alt": "NumPy 科学计算引擎图：ndarray 作为底层数组能力，支撑 Pandas、可视化、机器学习和深度学习。",
            },
            "en": {
                "title": "NumPy Is the Array Engine",
                "subtitle": "Understand ndarray first, then later data and AI libraries have a shared base.",
                "items": [
                    ("Python list", "Loop item by item; code grows heavy."),
                    ("ndarray", "Regular multi-dimensional arrays for batch math."),
                    ("shape / axis", "Describe data shape and compute direction."),
                    ("Pandas", "Table work builds on array power."),
                    ("Visualization", "Charts read arrays and statistics."),
                    ("ML / DL", "Training uses matrices and tensors everywhere."),
                ],
                "footer": "NumPy is not an isolated tool; it is the array base of Python data science.",
                "alt": "NumPy scientific computing engine diagram: ndarray is the array base supporting Pandas, visualization, machine learning, and deep learning.",
            },
            "ja": {
                "title": "NumPy は配列エンジン",
                "subtitle": "ndarray を理解すると、後のデータ分析と AI ライブラリがつながる。",
                "items": [
                    ("Python list", "1つずつ処理し、コードが重くなる。"),
                    ("ndarray", "規則的な多次元配列でまとめて計算。"),
                    ("shape / axis", "データ形状と計算方向を表す。"),
                    ("Pandas", "表処理は配列能力の上にある。"),
                    ("可視化", "グラフは配列と統計結果を読む。"),
                    ("ML / DL", "学習では行列と tensor を多用する。"),
                ],
                "footer": "NumPy は単独ツールではなく、Python データ科学の配列基盤。",
                "alt": "NumPy 科学計算エンジン図：ndarray が Pandas、可視化、機械学習、深層学習を支える配列基盤になる。",
            },
        },
    },
    {
        "slug": "ch03-pandas-roadmap",
        "pages": {
            "en": "docs/ch03-data-analysis/ch03-pandas/00-roadmap.md",
            "zh": "i18n/zh-Hans/docusaurus-plugin-content-docs/current/ch03-data-analysis/ch03-pandas/00-roadmap.md",
            "ja": "i18n/ja/docusaurus-plugin-content-docs/current/ch03-data-analysis/ch03-pandas/00-roadmap.md",
        },
        "scene": "A Pandas table workshop. Raw CSV, Excel, JSON, and SQL query results enter from the top. A single table moves through eight stations in order: read, inspect, select, clean, transform, group, merge, export. Each station visibly changes the table: missing values get marked, columns are filtered, dirty rows are fixed, a new feature column appears, grouped summary rows appear, another table joins by key, then charts/report/model outputs receive the final table.",
        "chapter_context": "This image appears at the start of the Pandas roadmap. The page tells learners to keep the one-line flow in mind: read -> inspect -> select -> clean -> transform -> group -> merge -> export. It says not to memorize APIs first, but to ask what table they have, what table they need, and which step changes one into the other.",
        "shared_layout": "Vertical 9:16. Top title and input files. Middle is a left-to-right and top-to-bottom table conveyor with eight numbered stations. Bottom shows final table feeding chart, report, and model outputs. Keep station order and artifacts identical across zh/en/ja.",
        "variants": {
            "zh": {
                "title": "Pandas 数据处理路线",
                "subtitle": "先看表从哪里来，再看每一步把表变成什么。",
                "items": [
                    ("read", "把 CSV / Excel / JSON 读成 DataFrame。"),
                    ("inspect", "先看列、类型、缺失值。"),
                    ("select", "取出当前问题需要的行和列。"),
                    ("clean", "修缺失、重复和异常。"),
                    ("transform", "派生新列，重编码字段。"),
                    ("group", "按问题做分组统计。"),
                    ("merge", "按 key 合并多张表。"),
                    ("export", "输出给图表、报告或模型。"),
                ],
                "footer": "学 Pandas 先记住表的变化顺序，再记 API。",
                "alt": "Pandas 数据处理路线图：读入数据、看结构、选择过滤、清洗、转换、聚合、合并和输出给图表或模型。",
            },
            "en": {
                "title": "Pandas Data Processing Roadmap",
                "subtitle": "First see where the table comes from, then what each step changes.",
                "items": [
                    ("read", "Load CSV / Excel / JSON into a DataFrame."),
                    ("inspect", "Check columns, types, and missing values."),
                    ("select", "Keep rows and columns for the question."),
                    ("clean", "Fix missing, duplicate, and abnormal data."),
                    ("transform", "Derive columns and recode fields."),
                    ("group", "Summarize by the question."),
                    ("merge", "Join tables by key."),
                    ("export", "Send results to charts, reports, or models."),
                ],
                "footer": "Learn the table-changing order before memorizing Pandas APIs.",
                "alt": "Pandas roadmap: read, inspect, select, clean, transform, group, merge, and export data for charts or models.",
            },
            "ja": {
                "title": "Pandas データ処理ロードマップ",
                "subtitle": "表がどこから来て、各ステップでどう変わるかを見る。",
                "items": [
                    ("read", "CSV / Excel / JSON を DataFrame に読む。"),
                    ("inspect", "列、型、欠損値を先に確認。"),
                    ("select", "問いに必要な行と列を残す。"),
                    ("clean", "欠損、重複、異常値を直す。"),
                    ("transform", "派生列を作り、値を変換する。"),
                    ("group", "問いに合わせて集計する。"),
                    ("merge", "key で複数表を結合する。"),
                    ("export", "グラフ、レポート、モデルへ渡す。"),
                ],
                "footer": "Pandas は API 暗記の前に、表の変化順を覚える。",
                "alt": "Pandas データ処理ロードマップ：read、inspect、select、clean、transform、group、merge、export。",
            },
        },
    },
    {
        "slug": "ch03-pandas-read-write-first-look",
        "pages": {
            "en": "docs/ch03-data-analysis/ch03-pandas/02-read-write.md",
            "zh": "i18n/zh-Hans/docusaurus-plugin-content-docs/current/ch03-data-analysis/ch03-pandas/02-read-write.md",
            "ja": "i18n/ja/docusaurus-plugin-content-docs/current/ch03-data-analysis/ch03-pandas/02-read-write.md",
        },
        "scene": "A beginner data import checkpoint. External files CSV, Excel, JSON, and SQL enter a Pandas workbench. The learner reads a file into a DataFrame, immediately checks the first rows, columns, dtypes, missing values, row count, and whether dates/numbers changed. A warning area shows the real danger: the file loads, but separators, encoding, headers, dates, or numbers are wrong. Then a clean export goes back to CSV or Excel.",
        "chapter_context": "The image appears under 'First, Build a Map'. The text says data read/write means first read it in, then confirm whether it was read correctly. It warns the scary part is not that a file will not move in, but that it moved in with changed format. The page teaches CSV, Excel, JSON, SQL, common parameters, and chunked reading.",
        "shared_layout": "Vertical 9:16. Top input file sources. Middle DataFrame import table. Right side first-check panel with head, columns, dtypes, missing, row count. Lower warning strip for silent format changes. Bottom export to clean file. Keep the checkpoint layout identical across zh/en/ja.",
        "variants": {
            "zh": {
                "title": "读入数据后，先检查",
                "subtitle": "文件能打开不等于读对了，格式可能已经变了。",
                "items": [
                    ("read", "CSV / Excel / JSON / SQL 进入 DataFrame。"),
                    ("head()", "先看前几行是否像原表。"),
                    ("columns", "确认列名没有错位。"),
                    ("dtypes", "检查日期、数字、文本类型。"),
                    ("missing", "确认缺失值是否被正确识别。"),
                    ("export", "检查通过后再输出干净文件。"),
                ],
                "footer": "读写数据的第一原则：读进来后马上验证。",
                "alt": "Pandas 读写数据首检图：读取 CSV、Excel、JSON 或 SQL 后先检查 head、columns、dtypes、缺失值和行数。",
            },
            "en": {
                "title": "After Reading Data, Check First",
                "subtitle": "A file can open and still be parsed incorrectly.",
                "items": [
                    ("read", "CSV / Excel / JSON / SQL becomes a DataFrame."),
                    ("head()", "Check whether first rows look right."),
                    ("columns", "Confirm headers did not shift."),
                    ("dtypes", "Check dates, numbers, and text."),
                    ("missing", "Confirm missing values are recognized."),
                    ("export", "Write a clean file after checks pass."),
                ],
                "footer": "First rule of data I/O: validate immediately after reading.",
                "alt": "Pandas read/write first-look workflow: load data, then check head, columns, dtypes, missing values, and row count.",
            },
            "ja": {
                "title": "読み込んだら、まず確認",
                "subtitle": "開けたファイルでも、形式が崩れていることがある。",
                "items": [
                    ("read", "CSV / Excel / JSON / SQL を DataFrame にする。"),
                    ("head()", "先頭行が元表らしいか見る。"),
                    ("columns", "列名がずれていないか確認。"),
                    ("dtypes", "日付、数値、文字列の型を見る。"),
                    ("missing", "欠損値が正しく認識されたか確認。"),
                    ("export", "確認後に clean file として出力。"),
                ],
                "footer": "データ入出力の第一原則：読み込んだ直後に検証する。",
                "alt": "Pandas 読み書きの初回確認：読み込み後に head、columns、dtypes、欠損値、行数を確認する。",
            },
        },
    },
    {
        "slug": "ch03-relational-database-foundation",
        "pages": {
            "en": "docs/ch03-data-analysis/ch05-database/01-relational-db.md",
            "zh": "i18n/zh-Hans/docusaurus-plugin-content-docs/current/ch03-data-analysis/ch05-database/01-relational-db.md",
            "ja": "i18n/ja/docusaurus-plugin-content-docs/current/ch03-data-analysis/ch05-database/01-relational-db.md",
        },
        "scene": "A lesson showing why a relational database is not just a bigger spreadsheet. On the left, one messy spreadsheet has repeated customer and product information, inconsistent edits, and no control over concurrent changes. In the center, the same data is split into Customers, Orders, and Products tables. Primary keys and foreign keys connect rows. On the right, database system features protect the data: constraints, indexes, transactions, permissions, and backup. A small Python/Pandas bridge reads query results for analysis.",
        "chapter_context": "This image appears at the start of the relational database lesson. The text tells beginners not to think a database is just a large CSV. It introduces database, table, row, column, primary key, foreign key, index, transaction, permissions, and reliability. The goal is to understand long-term data management and collaboration.",
        "shared_layout": "Vertical 9:16. Top title. Left panel shows messy spreadsheet problems. Middle panel shows three related tables connected by keys. Right panel shows system safeguards. Bottom shows a query result flowing to Python/Pandas analysis. Keep panels and arrows identical across zh/en/ja.",
        "variants": {
            "zh": {
                "title": "从表格到可靠数据库",
                "subtitle": "数据库不是更大的 CSV，而是长期管理数据的系统。",
                "items": [
                    ("混乱表格", "重复信息多，修改容易冲突。"),
                    ("表", "把客户、订单、商品分开保存。"),
                    ("主键", "稳定定位一行记录。"),
                    ("外键", "把相关表连接起来。"),
                    ("约束 / 事务", "减少脏数据和半完成修改。"),
                    ("索引 / 权限", "加速查询并控制访问。"),
                ],
                "footer": "关系型数据库用结构和规则保护长期数据。",
                "alt": "关系型数据库基础图：数据库、表、行、列、主键、外键、索引和权限共同支撑可靠数据管理。",
            },
            "en": {
                "title": "From Spreadsheet to Reliable Database",
                "subtitle": "A database is not a bigger CSV; it is a system for long-term data management.",
                "items": [
                    ("Messy sheet", "Repeated facts make edits conflict."),
                    ("Tables", "Store customers, orders, products separately."),
                    ("Primary key", "Locate one stable record."),
                    ("Foreign key", "Connect related tables."),
                    ("Constraint / transaction", "Reduce dirty and half-finished writes."),
                    ("Index / permission", "Speed queries and control access."),
                ],
                "footer": "Relational databases protect long-lived data with structure and rules.",
                "alt": "Relational database foundation: tables, rows, columns, primary keys, foreign keys, indexes, transactions, permissions, and backup.",
            },
            "ja": {
                "title": "表計算から信頼できるデータベースへ",
                "subtitle": "データベースは大きな CSV ではなく、長期データ管理の仕組み。",
                "items": [
                    ("乱れた表", "重複が多く、更新が衝突しやすい。"),
                    ("table", "顧客、注文、商品を分けて保存。"),
                    ("primary key", "1 行を安定して識別する。"),
                    ("foreign key", "関連する表をつなぐ。"),
                    ("constraint / transaction", "不正データと中途半端な更新を防ぐ。"),
                    ("index / permission", "検索を速くし、アクセスを制御。"),
                ],
                "footer": "関係データベースは構造とルールで長期データを守る。",
                "alt": "関係データベース基礎：table、row、column、primary key、foreign key、index、transaction、permission が信頼性を支える。",
            },
        },
    },
    {
        "slug": "ch03-database-design-erd-normalization",
        "pages": {
            "en": "docs/ch03-data-analysis/ch05-database/04-db-design.md",
            "zh": "i18n/zh-Hans/docusaurus-plugin-content-docs/current/ch03-data-analysis/ch05-database/04-db-design.md",
            "ja": "i18n/ja/docusaurus-plugin-content-docs/current/ch03-data-analysis/ch05-database/04-db-design.md",
        },
        "scene": "A database design review board. Start with one overloaded order table that repeats customer and product fields and has update conflicts. Then split it into Customers, Orders, OrderItems, and Products tables. Draw clear key connections. Show normalization as reducing duplication and conflict, not as abstract theory. Add an index card next to the query path for searching orders by customer/date/product. Include a final check list: less duplication, safer updates, query still fast.",
        "chapter_context": "This image appears in the database design lesson. The text says beginners often focus on formal normalization definitions, but the core is reducing duplication, reducing conflicts, and avoiding future maintenance problems. It teaches table design, relationships, normalization, and indexes.",
        "shared_layout": "Vertical 9:16. Top shows bad wide table and warning signs. Middle shows split tables with keys. Right side shows index speeding a query path. Bottom checklist compares before and after. Same panel order and colors across zh/en/ja.",
        "variants": {
            "zh": {
                "title": "先分表，再连表，再补索引",
                "subtitle": "范式的目的不是背概念，而是减少重复和维护事故。",
                "items": [
                    ("坏宽表", "客户和商品信息反复出现。"),
                    ("重复", "同一事实改多处，容易不一致。"),
                    ("分表", "按实体拆成客户、订单、商品。"),
                    ("主键 / 外键", "用稳定 ID 连接关系。"),
                    ("索引", "给常查路径加目录。"),
                    ("检查", "重复少、更新稳、查询快。"),
                ],
                "footer": "好的数据库设计，是为了以后少出错、好维护。",
                "alt": "数据库设计与范式图：实体拆表、主键外键、范式、索引和查询场景共同减少重复和维护风险。",
            },
            "en": {
                "title": "Split Tables, Link Them, Then Add Indexes",
                "subtitle": "Normalization is about fewer duplicates and fewer maintenance accidents.",
                "items": [
                    ("Bad wide table", "Customer and product facts repeat."),
                    ("Duplication", "One fact edited in many places can drift."),
                    ("Split tables", "Separate customers, orders, products."),
                    ("Primary / foreign key", "Link relations with stable IDs."),
                    ("Index", "Add a lookup path for common queries."),
                    ("Check", "Less duplicate, safer update, fast query."),
                ],
                "footer": "Good database design makes future changes safer and easier.",
                "alt": "Database design and normalization: split entities into tables, connect them with keys, then add indexes for query paths.",
            },
            "ja": {
                "title": "表を分け、つなぎ、索引を足す",
                "subtitle": "正規化は暗記ではなく、重複と保守事故を減らすため。",
                "items": [
                    ("悪い wide table", "顧客と商品情報が何度も出る。"),
                    ("重複", "同じ事実を複数箇所で更新する。"),
                    ("分割", "顧客、注文、商品に分ける。"),
                    ("primary / foreign key", "安定した ID で関係をつなぐ。"),
                    ("index", "よく使う検索に目次を付ける。"),
                    ("確認", "重複少、更新安全、検索高速。"),
                ],
                "footer": "よい DB 設計は、後の変更を安全で楽にする。",
                "alt": "データベース設計と正規化：entity を表に分け、key でつなぎ、index を加えて保守性を上げる。",
            },
        },
    },
    {
        "slug": "ch03-data-analysis-backbone",
        "pages": {
            "en": "docs/ch03-data-analysis/index.md",
            "zh": "i18n/zh-Hans/docusaurus-plugin-content-docs/current/ch03-data-analysis/index.md",
            "ja": "i18n/ja/docusaurus-plugin-content-docs/current/ch03-data-analysis/index.md",
        },
        "scene": "A data analysis loop for the chapter entry page. Raw files, logs, database tables, and survey forms enter a workbench. The loop has six large stages: read, inspect, clean, summarize, visualize, explain. Each stage leaves evidence: data profile, cleaning log, grouped summary, chart, conclusion note. At the end, the explanation either becomes a trustworthy report or loops back to inspect if the conclusion is weak.",
        "chapter_context": "This image appears in the Chapter 3 index under 'See The Data Analysis Loop'. The text explicitly says 'Read the picture first' and gives the loop read -> inspect -> clean -> summarize -> visualize -> explain. It warns not to draw charts first; first understand fields, units, missing values, duplicates, and sample sources.",
        "shared_layout": "Vertical 9:16. Use a circular loop around a central data workbench. Input sources at the top, six loop stations around the center, evidence artifacts beside each station, and a final report/loop-back choice at the bottom. Keep same stage order and object placement across zh/en/ja.",
        "variants": {
            "zh": {
                "title": "数据分析主线闭环",
                "subtitle": "不要先画图，先把数据读懂、查清、清洗干净。",
                "items": [
                    ("read", "读入文件、日志或数据库表。"),
                    ("inspect", "看字段、单位、缺失值。"),
                    ("clean", "记录并处理脏数据。"),
                    ("summarize", "分组统计回答问题。"),
                    ("visualize", "选择能解释问题的图。"),
                    ("explain", "写结论、限制和证据。"),
                ],
                "footer": "图表不是起点，可信结论才是终点。",
                "alt": "数据分析主线闭环：读取、检查、清洗、汇总、可视化和解释共同形成可信结论。",
            },
            "en": {
                "title": "The Data Analysis Loop",
                "subtitle": "Do not chart first; understand, inspect, and clean the data first.",
                "items": [
                    ("read", "Load files, logs, or database tables."),
                    ("inspect", "Check fields, units, missing values."),
                    ("clean", "Record and fix dirty data."),
                    ("summarize", "Group statistics for the question."),
                    ("visualize", "Choose charts that explain."),
                    ("explain", "Write conclusion, limits, evidence."),
                ],
                "footer": "Charts are not the starting point; trustworthy conclusions are the goal.",
                "alt": "Data analysis loop: read, inspect, clean, summarize, visualize, and explain to reach trustworthy conclusions.",
            },
            "ja": {
                "title": "データ分析のメインループ",
                "subtitle": "先にグラフを描かず、まずデータを理解し確認し、きれいにする。",
                "items": [
                    ("read", "ファイル、ログ、DB 表を読む。"),
                    ("inspect", "列、単位、欠損値を見る。"),
                    ("clean", "汚れたデータを記録して直す。"),
                    ("summarize", "問いに合わせて集計する。"),
                    ("visualize", "説明できるグラフを選ぶ。"),
                    ("explain", "結論、限界、証拠を書く。"),
                ],
                "footer": "グラフが出発点ではなく、信頼できる結論が目標。",
                "alt": "データ分析のメインループ：read、inspect、clean、summarize、visualize、explain で信頼できる結論へ進む。",
            },
        },
    },
    {
        "slug": "ch03-data-visualization",
        "pages": {
            "en": "docs/ch03-data-analysis/index.md",
            "zh": "i18n/zh-Hans/docusaurus-plugin-content-docs/current/ch03-data-analysis/index.md",
            "ja": "i18n/ja/docusaurus-plugin-content-docs/current/ch03-data-analysis/index.md",
        },
        "scene": "A chapter-opening data analysis workbench. A messy sales table with inconsistent dates, duplicate rows, missing values, and noisy notes enters from the top. The learner checks fields and units, cleans the table, groups sales by product and region, then chooses charts that support a short evidence-backed conclusion. Show the key idea that visualization is useful only after data is understood and cleaned.",
        "chapter_context": "This image appears at the top of Chapter 3. The nearby text says data, visualization, and analysis turn messy data into trustworthy conclusions with reproducible code and charts. A few lines later the chapter tells learners to read the data analysis loop first and warns not to draw charts before understanding fields, units, missing values, duplicates, and sample sources.",
        "shared_layout": "Vertical 9:16. Use the same practical analyst desk across zh/en/ja: messy source table at the top, inspection checklist, cleaning bench, grouped summary table, chart wall, and final report card. Keep the same objects, arrows, color rhythm, and reading path. The image must look like a teaching illustration, not a decorative dashboard.",
        "variants": {
            "zh": {
                "title": "数据分析与可视化",
                "subtitle": "从脏数据到可信结论，图表只是中间一步。",
                "items": [
                    ("原始数据", "日期、单位、缺失值可能混在一起。"),
                    ("检查", "先看字段、类型、来源和样本量。"),
                    ("清洗", "处理缺失、重复和异常。"),
                    ("汇总", "按问题分组统计。"),
                    ("可视化", "选择能解释问题的图。"),
                    ("结论", "把图表、限制和证据写进报告。"),
                ],
                "footer": "好图表来自可追踪的数据处理过程。",
                "alt": "数据分析与可视化主视觉：从原始脏数据开始，经过检查、清洗、汇总、可视化，最后形成带证据的分析结论。",
            },
            "en": {
                "title": "Data Analysis and Visualization",
                "subtitle": "From messy data to trustworthy conclusions; charts are only one step.",
                "items": [
                    ("Raw data", "Dates, units, missing values can be mixed."),
                    ("Inspect", "Check fields, types, source, and sample size."),
                    ("Clean", "Fix missing, duplicate, and abnormal values."),
                    ("Summarize", "Group statistics around the question."),
                    ("Visualize", "Choose charts that explain the question."),
                    ("Conclusion", "Write evidence, limits, and chart meaning."),
                ],
                "footer": "Good charts come from traceable data processing.",
                "alt": "Data analysis and visualization: raw messy data is inspected, cleaned, summarized, visualized, and turned into an evidence-backed conclusion.",
            },
            "ja": {
                "title": "データ分析と可視化",
                "subtitle": "乱れたデータから信頼できる結論へ。グラフは途中の一歩。",
                "items": [
                    ("生データ", "日付、単位、欠損値が混ざることがある。"),
                    ("確認", "列、型、出所、sample size を見る。"),
                    ("クリーニング", "欠損、重複、異常値を直す。"),
                    ("集計", "問いに合わせて分組して数える。"),
                    ("可視化", "問いを説明できるグラフを選ぶ。"),
                    ("結論", "証拠、限界、グラフの意味を書く。"),
                ],
                "footer": "よいグラフは、追跡できるデータ処理から生まれる。",
                "alt": "データ分析と可視化：乱れた生データを確認、クリーニング、集計、可視化し、証拠つきの結論へ進む。",
            },
        },
    },
    {
        "slug": "ch04-ai-math-backbone",
        "pages": {
            "en": "docs/ch04-ai-math/index.md",
            "zh": "i18n/zh-Hans/docusaurus-plugin-content-docs/current/ch04-ai-math/index.md",
            "ja": "i18n/ja/docusaurus-plugin-content-docs/current/ch04-ai-math/index.md",
        },
        "scene": "A model math loop in one runnable lab. Show a sample item being encoded as a vector and matrix, uncertainty shown as a probability gauge, loss shown as a wrongness meter, a gradient arrow pointing downhill, and parameters being updated into a better model. Include one tiny concrete loop: x vector, prediction probability, loss value, gradient direction, updated parameter.",
        "chapter_context": "This image appears under 'See The Model Math Loop'. The text says most AI math in this course supports one loop: represent data -> measure uncertainty -> measure loss -> update parameters. It explains vectors and matrices represent data, probability describes uncertainty, loss says how wrong the model is, and gradients say how to improve.",
        "shared_layout": "Vertical 9:16. Use one circular lab loop around a small model workbench. The four stations are representation, uncertainty, loss, and update. Put a small code/result strip beside the loop. Keep station order, arrows, and object placement identical across zh/en/ja.",
        "variants": {
            "zh": {
                "title": "AI 数学最小闭环",
                "subtitle": "数学不是公式墙，而是模型训练时反复执行的工具链。",
                "items": [
                    ("向量 / 矩阵", "把样本写成可计算的数字。"),
                    ("概率", "表达模型有多不确定。"),
                    ("loss", "把错误变成一个可优化数字。"),
                    ("gradient", "指出参数该往哪里改。"),
                    ("更新参数", "按小步改模型，再重复检查。"),
                ],
                "footer": "表示数据 -> 衡量不确定性 -> 衡量损失 -> 更新参数。",
                "alt": "AI 数学最小必要主线：向量和矩阵表示数据，概率描述不确定性，loss 衡量错误，gradient 指导参数更新。",
            },
            "en": {
                "title": "Minimum AI Math Loop",
                "subtitle": "Math is not a formula wall; it is the toolchain a model repeats during training.",
                "items": [
                    ("Vector / matrix", "Write samples as computable numbers."),
                    ("Probability", "Express model uncertainty."),
                    ("loss", "Turn wrongness into a number to optimize."),
                    ("gradient", "Point where parameters should move."),
                    ("Update parameters", "Take a small step, then check again."),
                ],
                "footer": "represent data -> measure uncertainty -> measure loss -> update parameters.",
                "alt": "AI math backbone: vectors and matrices represent data, probability describes uncertainty, loss measures error, and gradients update parameters.",
            },
            "ja": {
                "title": "AI 数学の最小ループ",
                "subtitle": "数学は公式の壁ではなく、学習で繰り返す道具の流れ。",
                "items": [
                    ("ベクトル / 行列", "サンプルを計算できる数値にする。"),
                    ("確率", "モデルの不確かさを表す。"),
                    ("loss", "間違いを最適化できる数にする。"),
                    ("gradient", "パラメータの動く方向を示す。"),
                    ("パラメータ更新", "小さく動かし、もう一度確認する。"),
                ],
                "footer": "データを表す -> 不確かさを測る -> 損失を測る -> パラメータを更新する。",
                "alt": "AI 数学の最小ループ：ベクトルと行列がデータを表し、確率が不確かさを表し、loss が誤差を測り、gradient が更新方向を示す。",
            },
        },
    },
    {
        "slug": "ch04-linear-algebra-chapter-flow",
        "pages": {
            "en": "docs/ch04-ai-math/ch01-linear-algebra/00-roadmap.md",
            "zh": "i18n/zh-Hans/docusaurus-plugin-content-docs/current/ch04-ai-math/ch01-linear-algebra/00-roadmap.md",
            "ja": "i18n/ja/docusaurus-plugin-content-docs/current/ch04-ai-math/ch01-linear-algebra/00-roadmap.md",
        },
        "scene": "A linear algebra chapter path inside an AI lab. A photo/sample is encoded as one vector, several vectors stack into a matrix, dot product compares two vectors, matrix multiplication applies many dot products at once, and eigen/PCA finds the strongest direction in a cloud of points. Show each concept doing a job in code, not as abstract decoration.",
        "chapter_context": "This image appears in the linear algebra roadmap. The table right below it says vector means one object written as numbers, matrix means many vectors stacked together or a transformation, dot product compares matching positions and adds them, matrix multiplication does many dot products at once, and eigenvalue/eigenvector are important directions for PCA intuition.",
        "shared_layout": "Vertical 9:16. Use a five-station learning path from top to bottom: vector, matrix, dot product, matrix multiplication, eigen/PCA. Each station has a tiny concrete artifact and a visible arrow to the next station. Keep the same station order and visual metaphor across zh/en/ja.",
        "variants": {
            "zh": {
                "title": "线性代数学习路径",
                "subtitle": "先看每个对象在 AI 代码里负责什么工作。",
                "items": [
                    ("vector", "一个样本写成一串数字。"),
                    ("matrix", "很多 vector 叠成一张数据表。"),
                    ("dot product", "对应位置相乘再相加，比较方向。"),
                    ("matrix multiply", "一次做很多个 dot product。"),
                    ("PCA 方向", "找出数据里最重要的方向。"),
                ],
                "footer": "线性代数把数据表示、比较和变换串起来。",
                "alt": "线性代数章节流程：vector、matrix、dot product、matrix multiplication 和 PCA 方向依次说明数据表示与变换。",
            },
            "en": {
                "title": "Linear Algebra Learning Path",
                "subtitle": "First see what each object does inside AI code.",
                "items": [
                    ("vector", "One sample written as numbers."),
                    ("matrix", "Many vectors stacked as a data table."),
                    ("dot product", "Multiply matching positions and add."),
                    ("matrix multiply", "Run many dot products at once."),
                    ("PCA direction", "Find the most important data direction."),
                ],
                "footer": "Linear algebra links data representation, comparison, and transformation.",
                "alt": "Linear algebra chapter flow: vector, matrix, dot product, matrix multiplication, and PCA direction explain representation and transformation.",
            },
            "ja": {
                "title": "線形代数の学習ルート",
                "subtitle": "各オブジェクトが AI のコードで何をするかを見る。",
                "items": [
                    ("vector", "1つのサンプルを数値列にする。"),
                    ("matrix", "多くの vector を積んだデータ表。"),
                    ("dot product", "対応位置を掛けて足し、方向を比べる。"),
                    ("matrix multiply", "多くの dot product を一度に行う。"),
                    ("PCA 方向", "データの重要な方向を探す。"),
                ],
                "footer": "線形代数は、データ表現、比較、変換をつなぐ。",
                "alt": "線形代数章の流れ：vector、matrix、dot product、matrix multiplication、PCA 方向で表現と変換を学ぶ。",
            },
        },
    },
    {
        "slug": "ch04-vector-space-high-level-map",
        "pages": {
            "en": "docs/ch04-ai-math/ch01-linear-algebra/04-vector-spaces.md",
            "zh": "i18n/zh-Hans/docusaurus-plugin-content-docs/current/ch04-ai-math/ch01-linear-algebra/04-vector-spaces.md",
            "ja": "i18n/ja/docusaurus-plugin-content-docs/current/ch04-ai-math/ch01-linear-algebra/04-vector-spaces.md",
        },
        "scene": "A high-level vector space classroom scene. Show basis vectors as coordinate rails, span as a reachable floor area, rank as the number of independent directions, orthogonal directions as clean non-overlapping axes, and SVD/PCA as a machine that rotates a messy point cloud to reveal strong directions. The learner should see this section as an organizing frame for previous vector, matrix, and eigen ideas.",
        "chapter_context": "This image appears in 'How does this section relate to the previous three?'. The text says vector spaces raise the viewpoint after learning vector representation, matrix transformation, and eigen directions. The nearby table decodes SVD, PCA, rank, basis, span, and orthogonal for beginners.",
        "shared_layout": "Vertical 9:16. Use one coordinate-room scene with a point cloud in the center. Arrange five teaching callouts around it: basis, span, rank, orthogonal, SVD/PCA. Show transformations as visible objects, not abstract boxes. Keep the same spatial layout across zh/en/ja.",
        "variants": {
            "zh": {
                "title": "向量空间高层地图",
                "subtitle": "把 vector、matrix、PCA 放到同一张空间图里理解。",
                "items": [
                    ("basis", "最少的一组坐标方向。"),
                    ("span", "这些方向组合能到达的范围。"),
                    ("rank", "真正独立的方向数量。"),
                    ("orthogonal", "互不重叠的信息方向。"),
                    ("SVD / PCA", "旋转数据，找主要方向。"),
                ],
                "footer": "向量空间帮你整理：能到哪里、方向是否独立、哪些方向最重要。",
                "alt": "向量空间高层理解图：basis、span、rank、orthogonal、SVD 和 PCA 共同解释数据空间中的方向与结构。",
            },
            "en": {
                "title": "Vector Space High-Level Map",
                "subtitle": "Put vectors, matrices, and PCA into one spatial view.",
                "items": [
                    ("basis", "The minimal coordinate directions."),
                    ("span", "The region reachable by mixing directions."),
                    ("rank", "How many directions are truly independent."),
                    ("orthogonal", "Information directions that do not overlap."),
                    ("SVD / PCA", "Rotate data to find strong directions."),
                ],
                "footer": "Vector space organizes reach, independence, and important directions.",
                "alt": "High-level vector space map: basis, span, rank, orthogonal directions, SVD, and PCA explain structure in data space.",
            },
            "ja": {
                "title": "ベクトル空間の高次マップ",
                "subtitle": "vector、matrix、PCA を 1 つの空間で見る。",
                "items": [
                    ("basis", "最小限の座標方向。"),
                    ("span", "方向を組み合わせて届く範囲。"),
                    ("rank", "本当に独立した方向の数。"),
                    ("orthogonal", "重ならない情報方向。"),
                    ("SVD / PCA", "データを回転し、強い方向を探す。"),
                ],
                "footer": "ベクトル空間は、到達範囲、独立性、重要方向を整理する。",
                "alt": "ベクトル空間の高次理解：basis、span、rank、orthogonal、SVD、PCA がデータ空間の方向と構造を説明する。",
            },
        },
    },
    {
        "slug": "ch04-probability-chapter-flow",
        "pages": {
            "en": "docs/ch04-ai-math/ch02-probability/00-roadmap.md",
            "zh": "i18n/zh-Hans/docusaurus-plugin-content-docs/current/ch04-ai-math/ch02-probability/00-roadmap.md",
            "ja": "i18n/ja/docusaurus-plugin-content-docs/current/ch04-ai-math/ch02-probability/00-roadmap.md",
        },
        "scene": "A probability and statistics learning path for AI models. Start with a single uncertain event, repeat it many times into a distribution shape, infer a conclusion from sample data, measure uncertainty with entropy, compare predicted probability distribution with target using cross-entropy, and compare two distributions using KL divergence. Make the path connect to model confidence and loss, not casino decoration.",
        "chapter_context": "This image appears in the probability roadmap. The text says probability and statistics explain model confidence, data variation, and why training uses loss values. The table asks first questions for probability, distribution, inference, entropy, cross-entropy, and KL divergence.",
        "shared_layout": "Vertical 9:16. Six stations from top to bottom: probability, distribution, inference, entropy, cross-entropy, KL divergence. Each station shows a tiny concrete probability artifact. Keep the same order, icon metaphors, and arrows across zh/en/ja.",
        "variants": {
            "zh": {
                "title": "概率统计学习路径",
                "subtitle": "理解模型置信度、数据波动和 loss 从哪里来。",
                "items": [
                    ("probability", "这件事有多可能发生。"),
                    ("distribution", "很多随机结果整体长什么样。"),
                    ("inference", "看见样本后能推出什么。"),
                    ("entropy", "结果有多不确定。"),
                    ("cross-entropy", "预测概率错得多远。"),
                    ("KL divergence", "两个分布差多少。"),
                ],
                "footer": "概率不是玄学，它把不确定性变成可计算对象。",
                "alt": "概率统计章节流程：probability、distribution、inference、entropy、cross-entropy 和 KL divergence 支撑模型置信度与损失。",
            },
            "en": {
                "title": "Probability and Statistics Path",
                "subtitle": "Understand confidence, data variation, and where loss comes from.",
                "items": [
                    ("probability", "How likely is this event?"),
                    ("distribution", "What shape do many random outcomes form?"),
                    ("inference", "What can we conclude from samples?"),
                    ("entropy", "How uncertain is the result?"),
                    ("cross-entropy", "How wrong are predicted probabilities?"),
                    ("KL divergence", "How different are two distributions?"),
                ],
                "footer": "Probability turns uncertainty into something computable.",
                "alt": "Probability and statistics chapter flow: probability, distribution, inference, entropy, cross-entropy, and KL divergence support confidence and loss.",
            },
            "ja": {
                "title": "確率統計の学習ルート",
                "subtitle": "confidence、データの揺れ、loss の理由を理解する。",
                "items": [
                    ("probability", "この事象はどれくらい起きそうか。"),
                    ("distribution", "多くの結果はどんな形になるか。"),
                    ("inference", "sample から何を結論できるか。"),
                    ("entropy", "結果はどれくらい不確かか。"),
                    ("cross-entropy", "予測確率はどれくらい外れたか。"),
                    ("KL divergence", "2つの分布はどれくらい違うか。"),
                ],
                "footer": "確率は不確かさを計算できる対象に変える。",
                "alt": "確率統計章の流れ：probability、distribution、inference、entropy、cross-entropy、KL divergence が confidence と loss を支える。",
            },
        },
    },
    {
        "slug": "ch04-distribution-random-world-map",
        "pages": {
            "en": "docs/ch04-ai-math/ch02-probability/02-distributions.md",
            "zh": "i18n/zh-Hans/docusaurus-plugin-content-docs/current/ch04-ai-math/ch02-probability/02-distributions.md",
            "ja": "i18n/ja/docusaurus-plugin-content-docs/current/ch04-ai-math/ch02-probability/02-distributions.md",
        },
        "scene": "A beginner distribution map built from repeated random phenomena. Show many repeated outcomes falling into bins, forming shapes: yes/no event becomes Bernoulli bars, many success counts become Binomial bars, measurement noise becomes a Normal bell, waiting/count events become Poisson bars. Add a small AI connection: probabilities feed confidence, loss, and evaluation.",
        "chapter_context": "This image appears under 'First, build a map'. The text says the point is not to memorize every distribution, but to know when a distribution appears, what it roughly looks like, and why it keeps appearing in AI. It defines a distribution as all possible values of a random variable and the probability of each value.",
        "shared_layout": "Vertical 9:16. Top shows a random phenomenon repeated many times and sorted into bins. Middle shows four large distribution shapes with simple example objects. Bottom connects distribution shapes to AI confidence, loss, and evaluation. Keep shapes and positions identical across zh/en/ja.",
        "variants": {
            "zh": {
                "title": "分布：随机现象的整体形状",
                "subtitle": "不要先背名字，先看重复很多次后会长成什么样。",
                "items": [
                    ("随机变量", "一次结果只是一个点。"),
                    ("重复实验", "很多次结果会形成形状。"),
                    ("Bernoulli", "只有成功 / 失败两种结果。"),
                    ("Binomial", "数很多次里成功了几次。"),
                    ("Normal", "测量误差常围绕中心波动。"),
                    ("Poisson", "单位时间内事件出现几次。"),
                ],
                "footer": "分布让模型能谈置信度、loss 和评估。",
                "alt": "概率分布随机现象地图：重复实验形成 Bernoulli、Binomial、Normal、Poisson 等分布形状，并连接到 AI 置信度与损失。",
            },
            "en": {
                "title": "Distribution: Shape of Randomness",
                "subtitle": "Do not memorize names first; see the shape after many repeats.",
                "items": [
                    ("Random variable", "One outcome is only one point."),
                    ("Repeated trials", "Many outcomes form a shape."),
                    ("Bernoulli", "Only success or failure."),
                    ("Binomial", "Count successes across many trials."),
                    ("Normal", "Measurement noise clusters near center."),
                    ("Poisson", "Count events in a time window."),
                ],
                "footer": "Distributions let models discuss confidence, loss, and evaluation.",
                "alt": "Probability distribution map: repeated random outcomes form Bernoulli, Binomial, Normal, and Poisson shapes connected to AI confidence and loss.",
            },
            "ja": {
                "title": "分布：ランダム現象の全体形",
                "subtitle": "名前を暗記する前に、何度も繰り返した形を見る。",
                "items": [
                    ("確率変数", "1回の結果は 1 つの点。"),
                    ("反復実験", "多くの結果が形を作る。"),
                    ("Bernoulli", "成功 / 失敗の 2 通り。"),
                    ("Binomial", "多くの試行で成功回数を数える。"),
                    ("Normal", "測定誤差は中心付近に集まる。"),
                    ("Poisson", "一定時間の event 数を数える。"),
                ],
                "footer": "分布があるから、confidence、loss、評価を扱える。",
                "alt": "確率分布のランダム現象マップ：反復実験が Bernoulli、Binomial、Normal、Poisson の形を作り、AI の confidence と loss につながる。",
            },
        },
    },
    {
        "slug": "ch04-gradient-descent-iteration-loop",
        "pages": {
            "en": "docs/ch04-ai-math/ch03-calculus/03-gradient-descent.md",
            "zh": "i18n/zh-Hans/docusaurus-plugin-content-docs/current/ch04-ai-math/ch03-calculus/03-gradient-descent.md",
            "ja": "i18n/ja/docusaurus-plugin-content-docs/current/ch04-ai-math/ch03-calculus/03-gradient-descent.md",
        },
        "scene": "A gradient descent teaching comic with a loss landscape and a model parameter notebook. A learner stands on a hill and checks the steepest downward direction, takes a small step controlled by learning rate, measures the new loss, and repeats until the slope is nearly flat. Also show a wrong path when learning rate is too large and jumps across the valley.",
        "chapter_context": "This image appears under 'First, build a map'. The text asks how to move parameters step by step to a better position once we know how a function changes. The next section uses the blindfolded downhill analogy: feel the steepest direction, take one step in the negative gradient direction, repeat until flat. It warns model training does not happen in one shot.",
        "shared_layout": "Vertical 9:16. Top title. Center large loss landscape with a dotted parameter path. Left side shows gradient direction check, right side shows learning-rate step size, bottom shows a small loss table and a warning path for too-large learning rate. Keep positions and path identical across zh/en/ja.",
        "variants": {
            "zh": {
                "title": "梯度下降迭代闭环",
                "subtitle": "训练不是一次算完，而是沿着 loss 下坡反复小步更新。",
                "items": [
                    ("计算 gradient", "找到 loss 上升最快的方向。"),
                    ("反方向走", "沿负梯度方向降低 loss。"),
                    ("learning rate", "决定每一步走多大。"),
                    ("重新计算 loss", "看这一步是否真的变好。"),
                    ("太大步", "可能越过谷底并震荡。"),
                    ("重复", "直到 slope 接近平。"),
                ],
                "footer": "参数更新 = 当前参数 - learning rate * gradient。",
                "alt": "梯度下降迭代闭环图：计算 gradient，沿负梯度方向按 learning rate 更新参数，重新计算 loss 并重复。",
            },
            "en": {
                "title": "Gradient Descent Loop",
                "subtitle": "Training is repeated small downhill updates on loss, not one calculation.",
                "items": [
                    ("Compute gradient", "Find the steepest upward direction."),
                    ("Move opposite", "Go against the gradient to lower loss."),
                    ("learning rate", "Controls how large each step is."),
                    ("Recompute loss", "Check whether the step improved."),
                    ("Too large step", "May jump over the valley and oscillate."),
                    ("Repeat", "Stop when the slope is nearly flat."),
                ],
                "footer": "parameter update = current parameter - learning rate * gradient.",
                "alt": "Gradient descent loop: compute gradient, move opposite it with a learning rate, recompute loss, and repeat until nearly flat.",
            },
            "ja": {
                "title": "勾配降下の反復ループ",
                "subtitle": "学習は一度で終わらず、loss の下り坂を小さく進む反復。",
                "items": [
                    ("gradient を計算", "loss が最も増える方向を探す。"),
                    ("反対へ進む", "負の勾配方向へ動いて loss を下げる。"),
                    ("learning rate", "1 歩の大きさを決める。"),
                    ("loss を再計算", "改善したか確認する。"),
                    ("大きすぎる一歩", "谷を飛び越えて振動する。"),
                    ("繰り返す", "slope がほぼ平らになるまで。"),
                ],
                "footer": "parameter update = current parameter - learning rate * gradient。",
                "alt": "勾配降下の反復ループ：gradient を計算し、learning rate で負の方向へ更新し、loss を再計算して繰り返す。",
            },
        },
    },
    {
        "slug": "ch05-machine-learning",
        "pages": {
            "en": "docs/ch05-machine-learning/index.md",
            "zh": "i18n/zh-Hans/docusaurus-plugin-content-docs/current/ch05-machine-learning/index.md",
            "ja": "i18n/ja/docusaurus-plugin-content-docs/current/ch05-machine-learning/index.md",
        },
        "scene": "A machine learning project workbench. A business question becomes a labeled dataset with features and target. The learner splits data into train and test, trains a simple baseline, trains an improved model, evaluates with metrics, inspects error samples, improves features with a Pipeline, and writes a short report. Show the idea that ML is an evaluable project loop, not just model names.",
        "chapter_context": "This image appears at the top of Chapter 5. The text says the chapter turns a data problem into a trainable, evaluable, improvable ML project. A few lines later it gives the reliable loop: define task -> split data -> train baseline -> evaluate -> inspect errors -> improve, and says to start with a baseline before chasing model names.",
        "shared_layout": "Vertical 9:16. Use the same project desk across zh/en/ja: problem card at top, labeled dataset, train/test split, baseline model, improved model, metric board, error sample tray, Pipeline feature station, and report card. Keep object placement, arrows, and colors identical across languages.",
        "variants": {
            "zh": {
                "title": "机器学习项目闭环",
                "subtitle": "先做可评估的 baseline，再谈模型改进。",
                "items": [
                    ("定义任务", "把业务问题写成可预测目标。"),
                    ("feature / label", "整理输入列和答案列。"),
                    ("划分数据", "训练集和测试集先分开。"),
                    ("baseline", "建立必须超过的简单基准。"),
                    ("evaluate", "用 metric 判断是否变好。"),
                    ("inspect errors", "看错样本，再决定怎么改。"),
                    ("Pipeline", "把预处理和模型封装，减少泄漏。"),
                ],
                "footer": "可靠 ML 工作是循环：定义、划分、训练、评估、看错、改进。",
                "alt": "机器学习主视觉：从定义任务、整理 feature 和 label、划分数据、训练 baseline，到评估、查看错误、用 Pipeline 改进并写报告。",
            },
            "en": {
                "title": "Machine Learning Project Loop",
                "subtitle": "Build an evaluable baseline before chasing model names.",
                "items": [
                    ("Define task", "Turn the business question into a target."),
                    ("feature / label", "Organize input columns and answers."),
                    ("Split data", "Separate train and test first."),
                    ("baseline", "Create the simple score to beat."),
                    ("evaluate", "Use metrics to judge improvement."),
                    ("inspect errors", "Study wrong samples before changing."),
                    ("Pipeline", "Package preprocessing and model to reduce leakage."),
                ],
                "footer": "Reliable ML loops through define, split, train, evaluate, inspect, improve.",
                "alt": "Machine learning project loop: define task, prepare features and labels, split data, train baseline, evaluate, inspect errors, improve with Pipeline, and report.",
            },
            "ja": {
                "title": "機械学習プロジェクトループ",
                "subtitle": "モデル名を追う前に、評価できる baseline を作る。",
                "items": [
                    ("タスク定義", "業務の問いを予測 target にする。"),
                    ("feature / label", "入力列と答え列を整理する。"),
                    ("データ分割", "train と test を先に分ける。"),
                    ("baseline", "超えるべき単純な基準を作る。"),
                    ("evaluate", "metric で改善を判断する。"),
                    ("inspect errors", "誤りサンプルを見てから直す。"),
                    ("Pipeline", "前処理とモデルをまとめ、leakage を減らす。"),
                ],
                "footer": "信頼できる ML は、定義、分割、学習、評価、誤り確認、改善のループ。",
                "alt": "機械学習プロジェクトループ：task 定義、feature と label、データ分割、baseline、評価、誤り確認、Pipeline 改善、レポート。",
            },
        },
    },
    {
        "slug": "ch05-hyperparameter-tuning-workflow",
        "pages": {
            "en": "docs/ch05-machine-learning/ch04-evaluation/04-hyperparameter-tuning.md",
            "zh": "i18n/zh-Hans/docusaurus-plugin-content-docs/current/ch05-machine-learning/ch04-evaluation/04-hyperparameter-tuning.md",
            "ja": "i18n/ja/docusaurus-plugin-content-docs/current/ch05-machine-learning/ch04-evaluation/04-hyperparameter-tuning.md",
        },
        "scene": "A safe hyperparameter tuning lab. Show data split first into train and final holdout. Inside train data, cross-validation folds feed GridSearchCV or RandomizedSearchCV over a search space of random forest settings such as n_estimators, max_depth, min_samples_leaf. The best setting is chosen by CV score, then checked once on final holdout. Add a red warning path where someone repeatedly checks the test set until it looks good, causing over-tuning.",
        "chapter_context": "This image appears in the hyperparameter tuning lesson. The tip says tuning is not trying settings until the test score looks good. A safe workflow searches on training folds, chooses by cross-validation, and checks once on a final holdout. The lesson covers parameter vs hyperparameter, GridSearchCV, RandomizedSearchCV, search space, budget, final holdout, and over-tuning.",
        "shared_layout": "Vertical 9:16. Top data split into train and final holdout. Middle CV search arena with search space knobs and fold scores. Right side best params card. Bottom final holdout check once, plus a red over-tuning warning path. Keep layout and arrows identical across zh/en/ja.",
        "variants": {
            "zh": {
                "title": "安全的超参数调优流程",
                "subtitle": "不要反复看测试集，用 CV 选择，final holdout 只检查一次。",
                "items": [
                    ("先划分", "train 用来搜索，final holdout 先封存。"),
                    ("search space", "列出允许尝试的候选设置。"),
                    ("GridSearchCV", "系统尝试小而规则的组合。"),
                    ("RandomizedSearchCV", "空间大时按预算抽样。"),
                    ("CV score", "用训练折平均分选择设置。"),
                    ("final holdout", "最后只用一次验证泛化。"),
                    ("过度调参", "反复看测试集会污染评估。"),
                ],
                "footer": "调参的目标是稳健泛化，不是把测试集刷漂亮。",
                "alt": "超参数调优验证流程图：先划分 final holdout，在训练折内用 GridSearchCV 或 RandomizedSearchCV 搜索，由 CV score 选择，再最终检查一次。",
            },
            "en": {
                "title": "Safe Hyperparameter Tuning Flow",
                "subtitle": "Do not keep peeking at the test set; choose by CV and check holdout once.",
                "items": [
                    ("Split first", "Use train for search; seal final holdout."),
                    ("search space", "List candidate settings allowed."),
                    ("GridSearchCV", "Try small regular combinations."),
                    ("RandomizedSearchCV", "Sample by budget when space grows."),
                    ("CV score", "Choose settings by fold average."),
                    ("final holdout", "Check generalization once at the end."),
                    ("Over-tuning", "Repeated test peeking contaminates evaluation."),
                ],
                "footer": "Tuning aims for robust generalization, not a prettier test score.",
                "alt": "Safe hyperparameter tuning workflow: split final holdout first, search train folds with GridSearchCV or RandomizedSearchCV, choose by CV score, and check once.",
            },
            "ja": {
                "title": "安全なハイパーパラメータ調整",
                "subtitle": "test を何度も見ず、CV で選び、final holdout は最後に 1 回だけ。",
                "items": [
                    ("先に分割", "train で探索し、final holdout は封印。"),
                    ("search space", "試してよい候補設定を並べる。"),
                    ("GridSearchCV", "小さな規則的組み合わせを試す。"),
                    ("RandomizedSearchCV", "空間が大きい時は予算内で抽出。"),
                    ("CV score", "fold 平均で設定を選ぶ。"),
                    ("final holdout", "最後に 1 回だけ汎化を確認。"),
                    ("過剰調整", "test の覗き見は評価を汚す。"),
                ],
                "footer": "調整の目的は、test score を飾ることではなく安定した汎化。",
                "alt": "ハイパーパラメータ調整の安全フロー：final holdout を先に分け、train fold 内で GridSearchCV または RandomizedSearchCV を使い、CV score で選ぶ。",
            },
        },
    },
    {
        "slug": "ch05-hands-on-portfolio-pack",
        "pages": {
            "en": "docs/ch05-machine-learning/ch06-projects/05-hands-on-ml-workshop.md",
            "zh": "i18n/zh-Hans/docusaurus-plugin-content-docs/current/ch05-machine-learning/ch06-projects/05-hands-on-ml-workshop.md",
            "ja": "i18n/ja/docusaurus-plugin-content-docs/current/ch05-machine-learning/ch06-projects/05-hands-on-ml-workshop.md",
        },
        "scene": "A machine learning portfolio evidence pack on a desk after a workshop. Show folders and artifacts: README, config.json, dataset source note, train/test split note, baseline score, model comparison table, confusion matrix or threshold curve, error sample buckets, leakage check, experiment notes, and next-step plan. Show that the project is credible because someone else can reproduce and inspect it.",
        "chapter_context": "This image appears under 'Turn This Into a Portfolio Project'. The text tells learners to replace the synthetic dataset with their own CSV, add config.json, add one more model but keep Dummy baseline, add a confusion matrix or threshold curve, explain the top 3 error patterns, add next improvements to README, then make one small iteration and rerun.",
        "shared_layout": "Vertical 9:16. Use a clean binder/evidence board with reusable artifacts, not a decorative poster. Top has project goal and dataset. Middle has reproducibility and model evidence. Bottom has error analysis, leakage check, next steps, and rerun note. Keep artifact positions and style identical across zh/en/ja.",
        "variants": {
            "zh": {
                "title": "机器学习作品集证据包",
                "subtitle": "别人能复现、检查、继续改，项目才可信。",
                "items": [
                    ("README", "说明任务、数据、运行方法和结论。"),
                    ("config.json", "固定 random seed、test size、模型列表。"),
                    ("baseline", "保留 Dummy baseline 作为对照。"),
                    ("model comparison", "记录每个模型的 metric。"),
                    ("error buckets", "解释前 3 类错误样本。"),
                    ("leakage check", "确认目标信息没有进训练。"),
                    ("next steps", "写清下一步只改什么。"),
                    ("rerun note", "改一个点后重新运行并记录。"),
                ],
                "footer": "作品集不是炫分数，而是交付可复现的证据。",
                "alt": "机器学习作品集证据包：README、config、baseline、模型对比、错误样本、泄漏检查、下一步计划和复跑记录组成可复现项目。",
            },
            "en": {
                "title": "Machine Learning Portfolio Evidence Pack",
                "subtitle": "A project is credible when others can reproduce, inspect, and improve it.",
                "items": [
                    ("README", "Explain task, data, run steps, and conclusion."),
                    ("config.json", "Fix random seed, test size, and model list."),
                    ("baseline", "Keep Dummy baseline as comparison."),
                    ("model comparison", "Record metrics for each model."),
                    ("error buckets", "Explain top 3 error patterns."),
                    ("leakage check", "Confirm target information did not leak."),
                    ("next steps", "State the next single change."),
                    ("rerun note", "Change one thing, rerun, and record."),
                ],
                "footer": "A portfolio is not score theater; it is reproducible evidence.",
                "alt": "Machine learning portfolio evidence pack: README, config, baseline, model comparison, error buckets, leakage check, next steps, and rerun note.",
            },
            "ja": {
                "title": "機械学習ポートフォリオ証拠パック",
                "subtitle": "他人が再現し、確認し、改善できるとプロジェクトは信頼できる。",
                "items": [
                    ("README", "タスク、データ、実行手順、結論を書く。"),
                    ("config.json", "random seed、test size、モデル一覧を固定。"),
                    ("baseline", "Dummy baseline を比較用に残す。"),
                    ("model comparison", "各モデルの metric を記録する。"),
                    ("error buckets", "上位 3 つの誤りパターンを説明。"),
                    ("leakage check", "target 情報が漏れていないか確認。"),
                    ("next steps", "次に 1 つだけ変える点を書く。"),
                    ("rerun note", "1 点変更して再実行し記録する。"),
                ],
                "footer": "ポートフォリオは点数自慢ではなく、再現できる証拠の提出。",
                "alt": "機械学習ポートフォリオ証拠パック：README、config、baseline、model comparison、error buckets、leakage check、next steps、rerun note。",
            },
        },
    },
    {
        "slug": "ch06-deep-learning",
        "pages": {
            "en": "docs/ch06-deep-learning/index.md",
            "zh": "i18n/zh-Hans/docusaurus-plugin-content-docs/current/ch06-deep-learning/index.md",
            "ja": "i18n/ja/docusaurus-plugin-content-docs/current/ch06-deep-learning/index.md",
        },
        "scene": "A deep learning training room that teaches the actual loop. A mini batch of images and labels enters a DataLoader tray, becomes tensors, moves through a small neural network, produces predictions, compares with labels on a loss meter, sends gradients backward, lets the optimizer update weights, then records loss curves and a checkpoint. Also show a small warning note that beginners should train a small model and read logs before chasing giant models.",
        "chapter_context": "This image appears at the top of Chapter 6. The surrounding text says the chapter teaches how a model learns from loss, gradients, and repeated training steps. It gives the reliable loop: batch data -> model forward -> loss -> backward gradients -> optimizer step -> curves. It also says do not chase big models first; train a small model, log the process, and explain why it improved or failed.",
        "shared_layout": "Vertical 9:16. Use one continuous training bench: data tray at top, tensor conversion, model forward path, loss meter, backward gradient arrows, optimizer wrench updating weights, then loss curve and checkpoint notebook at bottom. Keep the same camera angle, bench, arrows, color rhythm, and object positions across zh/en/ja.",
        "variants": {
            "zh": {
                "title": "深度学习训练闭环",
                "subtitle": "模型靠一批批数据、loss、gradient 和 optimizer 反复变好。",
                "items": [
                    ("batch data", "一次只送入一小批样本。"),
                    ("tensor", "把图片或文本变成可计算数组。"),
                    ("forward", "模型根据当前参数做预测。"),
                    ("loss", "比较预测和 label 的差距。"),
                    ("backward", "把误差信号传回每一层。"),
                    ("optimizer step", "按 gradient 更新参数。"),
                    ("curves / log", "用曲线判断训练是否变好。"),
                ],
                "footer": "先训练小模型、看日志、解释失败，再扩大规模。",
                "alt": "深度学习训练闭环：batch data 进入模型，forward 得到预测，loss 衡量误差，backward 计算 gradient，optimizer step 更新参数并记录曲线。",
            },
            "en": {
                "title": "Deep Learning Training Loop",
                "subtitle": "A model improves through batches, loss, gradients, and optimizer steps.",
                "items": [
                    ("batch data", "Feed a small group of samples at a time."),
                    ("tensor", "Turn images or text into computable arrays."),
                    ("forward", "Predict with the current parameters."),
                    ("loss", "Compare prediction with the label."),
                    ("backward", "Send error signals through layers."),
                    ("optimizer step", "Update parameters by gradients."),
                    ("curves / log", "Use curves to judge training progress."),
                ],
                "footer": "Train small, read logs, explain failures, then scale up.",
                "alt": "Deep learning training loop: batch data enters the model, forward predicts, loss measures error, backward computes gradients, optimizer step updates weights, and curves are logged.",
            },
            "ja": {
                "title": "深層学習の訓練ループ",
                "subtitle": "モデルは batch、loss、gradient、optimizer step の反復で良くなる。",
                "items": [
                    ("batch data", "少量のサンプルを一度に入れる。"),
                    ("tensor", "画像や文章を計算できる配列にする。"),
                    ("forward", "現在のパラメータで予測する。"),
                    ("loss", "予測と label の差を測る。"),
                    ("backward", "誤差信号を各層へ戻す。"),
                    ("optimizer step", "gradient でパラメータを更新する。"),
                    ("curves / log", "曲線で学習の進み方を確認する。"),
                ],
                "footer": "まず小さく訓練し、log を読み、失敗を説明してから広げる。",
                "alt": "深層学習の訓練ループ：batch data、forward、loss、backward、optimizer step、curves と log。",
            },
        },
    },
    {
        "slug": "ch07-llm-principles",
        "pages": {
            "en": "docs/ch07-llm-principles/index.md",
            "zh": "i18n/zh-Hans/docusaurus-plugin-content-docs/current/ch07-llm-principles/index.md",
            "ja": "i18n/ja/docusaurus-plugin-content-docs/current/ch07-llm-principles/index.md",
        },
        "scene": "A worked LLM request path. A user question enters a text gate, is split into tokens, embedded as vectors, placed in a context window, processed by Transformer attention heads, then the model predicts one next token at a time until an answer forms. Around the path, show practical control levers: prompt, structured output, RAG evidence, fine-tuning, tools, and evaluation. The scene should teach how a chat answer becomes stable enough for an application.",
        "chapter_context": "This image appears at the top of Chapter 7. The text asks what path user text takes through an LLM and how to make results stable for an app. It explains text -> tokens -> vectors -> Transformer predicts next token from context, then control the result with prompt, structured output, RAG, fine-tuning, tools, and evaluation.",
        "shared_layout": "Vertical 9:16. Top user question, middle token-to-vector-to-context-to-Transformer path, lower next-token output strip, side levers for prompt, structured output, RAG, fine-tuning, tools, and eval. Keep the same route, panels, and lever positions across zh/en/ja.",
        "variants": {
            "zh": {
                "title": "从文字到答案的 LLM 路径",
                "subtitle": "先看 token 和 context，再理解为什么需要控制输出。",
                "items": [
                    ("token", "把文字切成模型能处理的小片段。"),
                    ("embedding", "把 token 放进向量空间。"),
                    ("context window", "模型只看窗口里的信息。"),
                    ("Transformer", "用 attention 读取上下文关系。"),
                    ("next token", "一步步预测下一个 token。"),
                    ("control layer", "用 Prompt、RAG、工具和评估让结果稳定。"),
                ],
                "footer": "应用里的 LLM 不是魔法回答，而是受控的生成流程。",
                "alt": "LLM 原理路径图：文字切成 token，转成 embedding，进入 context window，经 Transformer attention 逐步预测 next token，并用 Prompt、RAG、工具和评估控制输出。",
            },
            "en": {
                "title": "LLM Path from Text to Answer",
                "subtitle": "Understand tokens and context before controlling output.",
                "items": [
                    ("token", "Split text into model-sized pieces."),
                    ("embedding", "Place tokens in vector space."),
                    ("context window", "The model sees only what fits inside."),
                    ("Transformer", "Use attention to read relationships."),
                    ("next token", "Predict the next token step by step."),
                    ("control layer", "Use prompt, RAG, tools, and eval to stabilize output."),
                ],
                "footer": "An app LLM is not magic; it is a controlled generation process.",
                "alt": "LLM principle path: text becomes tokens, embeddings enter a context window, Transformer attention predicts next tokens, and prompt, RAG, tools, and eval control output.",
            },
            "ja": {
                "title": "文字から回答までの LLM 経路",
                "subtitle": "token と context を見てから、出力制御を理解する。",
                "items": [
                    ("token", "文章をモデル用の小片に分ける。"),
                    ("embedding", "token をベクトル空間へ置く。"),
                    ("context window", "モデルは窓に入る情報だけを見る。"),
                    ("Transformer", "attention で文脈関係を読む。"),
                    ("next token", "次の token を一歩ずつ予測する。"),
                    ("control layer", "Prompt、RAG、tool、eval で出力を安定させる。"),
                ],
                "footer": "アプリ内の LLM は魔法ではなく、制御された生成プロセス。",
                "alt": "LLM の経路：文字を token にし、embedding と context window を通し、Transformer attention が next token を予測し、Prompt、RAG、tool、eval で制御する。",
            },
        },
    },
    {
        "slug": "ch08-rag-engineering",
        "pages": {
            "en": "docs/ch08-rag/index.md",
            "zh": "i18n/zh-Hans/docusaurus-plugin-content-docs/current/ch08-rag/index.md",
            "ja": "i18n/ja/docusaurus-plugin-content-docs/current/ch08-rag/index.md",
        },
        "scene": "A RAG engineering room where the system reads before answering. Source manuals, PDFs, notes, and tickets enter a knowledge preparation station. Chunks with metadata go into a vector index. A user question searches the index, retrieves evidence cards, and sends only the relevant context to the LLM. The final answer must show citations, no-answer when evidence is missing, and logs for evaluation and improvement.",
        "chapter_context": "This image appears at the top of Chapter 8. The chapter says RAG connects documents, retrieves evidence, answers with citations, logs failures, and improves with an eval set. It frames RAG as read before answering, with knowledge preparation, retrieval, generation, application, and operations layers.",
        "shared_layout": "Vertical 9:16. Top source documents, middle-left chunk and metadata preparation, middle vector index, right user question and retrieval, lower LLM answer with citations, bottom eval/log improvement loop. Keep all stations and arrow order identical across zh/en/ja.",
        "variants": {
            "zh": {
                "title": "RAG：先查证据再回答",
                "subtitle": "把文档变成可检索证据，再让 LLM 带引用回答。",
                "items": [
                    ("documents", "PDF、手册、工单先进入知识层。"),
                    ("chunks + metadata", "切块并保留来源、页码和权限。"),
                    ("vector index", "用 embedding 支持相似度检索。"),
                    ("top-k evidence", "只取与问题相关的证据片段。"),
                    ("answer + citations", "回答必须能指回来源。"),
                    ("eval / logs", "用失败样本继续改检索和生成。"),
                ],
                "footer": "RAG 的核心不是更会编，而是回答前先读到证据。",
                "alt": "RAG 工程闭环：文档切块加 metadata，进入 vector index，按问题检索 top-k evidence，再让 LLM 带 citations 回答并记录 eval/logs。",
            },
            "en": {
                "title": "RAG: Read Evidence Before Answering",
                "subtitle": "Turn documents into retrievable evidence, then answer with citations.",
                "items": [
                    ("documents", "PDFs, manuals, and tickets enter knowledge prep."),
                    ("chunks + metadata", "Split text and keep source, page, permission."),
                    ("vector index", "Use embeddings for similarity search."),
                    ("top-k evidence", "Fetch only question-relevant snippets."),
                    ("answer + citations", "Every claim can point back to sources."),
                    ("eval / logs", "Use failures to improve retrieval and generation."),
                ],
                "footer": "RAG is not better guessing; it is reading evidence before answering.",
                "alt": "RAG engineering loop: documents become chunks with metadata, enter a vector index, retrieve top-k evidence for a question, answer with citations, and improve through eval logs.",
            },
            "ja": {
                "title": "RAG：根拠を読んでから答える",
                "subtitle": "文書を検索できる証拠にし、LLM が citation 付きで答える。",
                "items": [
                    ("documents", "PDF、マニュアル、チケットを知識層へ入れる。"),
                    ("chunks + metadata", "分割し、出典、ページ、権限を残す。"),
                    ("vector index", "embedding で類似検索を行う。"),
                    ("top-k evidence", "質問に関係する断片だけを取る。"),
                    ("answer + citations", "回答は出典へ戻れるようにする。"),
                    ("eval / logs", "失敗例で検索と生成を改善する。"),
                ],
                "footer": "RAG の核心は推測ではなく、回答前に証拠を読むこと。",
                "alt": "RAG エンジニアリング：documents を chunks と metadata にし、vector index で top-k evidence を検索し、citations 付き回答と eval logs で改善する。",
            },
        },
    },
    {
        "slug": "ch08-rag-app-loop",
        "pages": {
            "en": "docs/ch08-rag/index.md",
            "zh": "i18n/zh-Hans/docusaurus-plugin-content-docs/current/ch08-rag/index.md",
            "ja": "i18n/ja/docusaurus-plugin-content-docs/current/ch08-rag/index.md",
        },
        "scene": "A layered RAG application loop. Show five layers as a practical service, not a flowchart: knowledge layer parses and chunks documents, retrieval layer embeds and ranks query results, generation layer assembles prompt and citations, application layer exposes CLI/API/chat, and operations layer tracks evaluation, logs, token cost, latency, and failure cases. Feedback arrows from operations improve chunking, retrieval, and prompts.",
        "chapter_context": "This image appears near the start of Chapter 8 where the application loop is listed: Knowledge parses documents, cleans and chunks them with metadata; Retrieval handles query, top-k, scores, and source IDs; Generation handles prompt, answer, citations, and no-answer; Application includes CLI/API/chat; Operations includes eval, logs, token cost, latency, and failures.",
        "shared_layout": "Vertical 9:16. Five stacked service layers with real artifacts in each layer, plus feedback arrows from bottom operations back to knowledge, retrieval, and generation. Keep layer order, icons, and arrows identical across zh/en/ja.",
        "variants": {
            "zh": {
                "title": "RAG 应用五层闭环",
                "subtitle": "知识、检索、生成、应用、运维一起决定回答质量。",
                "items": [
                    ("知识层", "清洗文档、切 chunk、保存 metadata。"),
                    ("检索层", "query、top-k、score、source id。"),
                    ("生成层", "组装 prompt、回答、citation、no-answer。"),
                    ("应用层", "把能力接到 CLI、API 或聊天界面。"),
                    ("运维层", "跟踪 eval、log、cost、latency、failure。"),
                    ("反馈改进", "用错误样本回改切块、检索和提示词。"),
                ],
                "footer": "RAG 不是单次问答，而是可观测、可迭代的应用系统。",
                "alt": "RAG 应用五层闭环：知识层、检索层、生成层、应用层和运维层通过 eval、log、cost、latency 与 failure 反馈持续改进。",
            },
            "en": {
                "title": "Five-Layer RAG Application Loop",
                "subtitle": "Knowledge, retrieval, generation, app, and ops together shape answer quality.",
                "items": [
                    ("Knowledge", "Clean docs, split chunks, keep metadata."),
                    ("Retrieval", "Query, top-k, score, and source id."),
                    ("Generation", "Assemble prompt, answer, citation, no-answer."),
                    ("Application", "Expose CLI, API, or chat UI."),
                    ("Operations", "Track eval, log, cost, latency, failure."),
                    ("Feedback", "Use errors to improve chunks, retrieval, prompts."),
                ],
                "footer": "RAG is an observable, iterative application system, not one chat call.",
                "alt": "Five-layer RAG app loop: knowledge, retrieval, generation, application, and operations use eval, logs, cost, latency, and failures to improve the system.",
            },
            "ja": {
                "title": "RAG アプリの五層ループ",
                "subtitle": "知識、検索、生成、アプリ、運用が回答品質を決める。",
                "items": [
                    ("知識層", "文書を整え、chunk に分け、metadata を残す。"),
                    ("検索層", "query、top-k、score、source id。"),
                    ("生成層", "prompt、回答、citation、no-answer を作る。"),
                    ("アプリ層", "CLI、API、chat UI へつなぐ。"),
                    ("運用層", "eval、log、cost、latency、failure を見る。"),
                    ("改善フィードバック", "失敗例で chunk、検索、prompt を直す。"),
                ],
                "footer": "RAG は一回の会話ではなく、観測して改善するアプリシステム。",
                "alt": "RAG アプリ五層ループ：知識層、検索層、生成層、アプリ層、運用層が eval、log、cost、latency、failure で改善する。",
            },
        },
    },
    {
        "slug": "ch08-rag-data-to-answer-pipeline",
        "pages": {
            "en": "docs/ch08-rag/ch01-rag/00-roadmap.md",
            "zh": "i18n/zh-Hans/docusaurus-plugin-content-docs/current/ch08-rag/ch01-rag/00-roadmap.md",
            "ja": "i18n/ja/docusaurus-plugin-content-docs/current/ch08-rag/ch01-rag/00-roadmap.md",
        },
        "scene": "A document-to-answer conveyor. Raw documents are loaded, split into chunks, stamped with metadata, embedded into vectors, stored in an index, then a user query is embedded, retrieves candidates, reranks them, assembles a context pack, produces an answer, attaches citations, and sends the result to an evaluation checklist. Show a small evidence card traveling beside the answer so learners see source traceability.",
        "chapter_context": "This roadmap image teaches the core RAG loop: load documents, split chunks, add metadata, embedding, retrieve, rerank, assemble context, answer, cite sources, evaluate. The nearby text asks learners to build the smallest retrieval check before trusting the final answer.",
        "shared_layout": "Vertical 9:16. Use a document conveyor from top to bottom with ten compact stations, but make each station a concrete artifact rather than a rounded box. Keep the same order and source-trace evidence card across zh/en/ja.",
        "variants": {
            "zh": {
                "title": "从资料到带证据的回答",
                "subtitle": "先确认资料能被检到，再让模型生成答案。",
                "items": [
                    ("load docs", "读取原始文档。"),
                    ("split chunks", "切成可检索片段。"),
                    ("metadata", "保留来源、页码、权限。"),
                    ("embedding index", "把片段放入向量索引。"),
                    ("retrieve + rerank", "先召回，再重排最相关证据。"),
                    ("answer + cite", "用 context 回答并引用来源。"),
                    ("evaluate", "检查答案是否被证据支持。"),
                ],
                "footer": "RAG 的最小验证：正确片段是否真的进入 context？",
                "alt": "RAG 从资料到回答的流水线：load docs、split chunks、metadata、embedding index、retrieve、rerank、context、answer、citations 和 evaluate。",
            },
            "en": {
                "title": "From Source Material to Evidence-Backed Answer",
                "subtitle": "Check retrieval before trusting generation.",
                "items": [
                    ("load docs", "Read raw source documents."),
                    ("split chunks", "Create searchable text pieces."),
                    ("metadata", "Keep source, page, and permission."),
                    ("embedding index", "Store chunks as vectors."),
                    ("retrieve + rerank", "Recall candidates, then reorder evidence."),
                    ("answer + cite", "Answer from context with source links."),
                    ("evaluate", "Check whether evidence supports the answer."),
                ],
                "footer": "Minimal RAG check: did the right chunk reach the context?",
                "alt": "RAG data-to-answer pipeline: load docs, split chunks, add metadata, build embedding index, retrieve, rerank, assemble context, answer, cite, and evaluate.",
            },
            "ja": {
                "title": "資料から根拠付き回答へ",
                "subtitle": "生成を信じる前に、検索できているか確認する。",
                "items": [
                    ("load docs", "元文書を読み込む。"),
                    ("split chunks", "検索できる断片に分ける。"),
                    ("metadata", "出典、ページ、権限を残す。"),
                    ("embedding index", "chunk をベクトル索引へ入れる。"),
                    ("retrieve + rerank", "候補を取り、根拠順に並べる。"),
                    ("answer + cite", "context から答え、出典を付ける。"),
                    ("evaluate", "回答が根拠で支えられるか確認。"),
                ],
                "footer": "最小 RAG 確認：正しい chunk が context に入ったか？",
                "alt": "RAG の資料から回答まで：load docs、split chunks、metadata、embedding index、retrieve、rerank、context、answer、cite、evaluate。",
            },
        },
    },
    {
        "slug": "ch08-rerank-query-rewrite-funnel-map",
        "pages": {
            "en": "docs/ch08-rag/ch02-advanced/retrieval-strategies.md",
            "zh": "i18n/zh-Hans/docusaurus-plugin-content-docs/current/ch08-rag/ch02-advanced/retrieval-strategies.md",
            "ja": "i18n/ja/docusaurus-plugin-content-docs/current/ch08-rag/ch02-advanced/retrieval-strategies.md",
        },
        "scene": "A retrieval funnel that contrasts query rewrite and rerank. A vague user question enters a rewrite desk before retrieval, where synonyms and domain terms are added to make it searchable. The rewritten query retrieves a broad candidate pile from the index. After retrieval, a rerank judge reorders candidates by evidence quality and pushes the best snippets into the final context pack. Show a side-by-side note: rewrite changes the input query; rerank changes the order of retrieved candidates.",
        "chapter_context": "This image appears where the lesson explains that query rewrite happens before retrieval and makes the user question easier to search, while rerank happens after rough recall and reorders candidates. The text says one changes the input, the other changes ranking.",
        "shared_layout": "Vertical 9:16. Top vague question, upper rewrite desk before the index, middle broad retrieval candidate pile, lower rerank judge, bottom final context pack. Keep the before/after positions and comparison strip identical across zh/en/ja.",
        "variants": {
            "zh": {
                "title": "Query Rewrite 与 Rerank 的位置",
                "subtitle": "一个改查询，一个改候选排序，发生在不同阶段。",
                "items": [
                    ("原始问题", "用户说法可能口语、含糊、缺术语。"),
                    ("query rewrite", "检索前补关键词和同义表达。"),
                    ("rough recall", "先从索引拿到一批候选片段。"),
                    ("rerank", "检索后按证据质量重新排序。"),
                    ("final context", "只把最有用证据交给 LLM。"),
                    ("区别", "rewrite 改输入，rerank 改顺序。"),
                ],
                "footer": "先让问题更好搜，再让结果更会排。",
                "alt": "Query rewrite 与 rerank 对比：query rewrite 在检索前改写问题，rough recall 取候选，rerank 在检索后重排候选，再形成 final context。",
            },
            "en": {
                "title": "Where Query Rewrite and Rerank Happen",
                "subtitle": "One changes the query; the other changes candidate order.",
                "items": [
                    ("raw question", "User wording may be vague or missing terms."),
                    ("query rewrite", "Add keywords and synonyms before search."),
                    ("rough recall", "Fetch a broad set of candidate chunks."),
                    ("rerank", "Reorder candidates by evidence quality."),
                    ("final context", "Send only useful evidence to the LLM."),
                    ("difference", "Rewrite changes input; rerank changes order."),
                ],
                "footer": "First make the question easier to search, then rank results better.",
                "alt": "Query rewrite and rerank funnel: rewrite modifies the question before retrieval, rough recall fetches candidates, rerank reorders them, and final context goes to the LLM.",
            },
            "ja": {
                "title": "Query Rewrite と Rerank の位置",
                "subtitle": "一方は検索文を変え、もう一方は候補の順序を変える。",
                "items": [
                    ("元の質問", "ユーザーの表現は曖昧で術語が足りない。"),
                    ("query rewrite", "検索前にキーワードや同義語を補う。"),
                    ("rough recall", "索引から広めに候補 chunk を取る。"),
                    ("rerank", "検索後に根拠品質で並べ替える。"),
                    ("final context", "有用な証拠だけを LLM に渡す。"),
                    ("違い", "rewrite は入力を変え、rerank は順序を変える。"),
                ],
                "footer": "まず検索しやすくし、次に結果を良い順へ並べる。",
                "alt": "Query rewrite と rerank の比較：rewrite は検索前に質問を変え、rough recall で候補を取り、rerank は検索後に候補を並べ替える。",
            },
        },
    },
    {
        "slug": "ch08-faithfulness-citation-check-map",
        "pages": {
            "en": "docs/ch08-rag/ch02-advanced/rag-evaluation.md",
            "zh": "i18n/zh-Hans/docusaurus-plugin-content-docs/current/ch08-rag/ch02-advanced/rag-evaluation.md",
            "ja": "i18n/ja/docusaurus-plugin-content-docs/current/ch08-rag/ch02-advanced/rag-evaluation.md",
        },
        "scene": "A citation and faithfulness inspection desk. An answer card is split into three claim strips. Each claim has a colored thread to an evidence snippet with source id and page number. Supported claims get green checks; one unsupported claim gets a red warning and is moved to a fix tray. A reviewer also checks whether the citation points to the exact page or source region, not merely a similar document.",
        "chapter_context": "This image appears in the RAG evaluation lesson. The text says to split an answer into key conclusions and link each one to evidence, marking supported and unsupported claims. It teaches faithfulness and citation checking as more reliable than trusting fluent answers.",
        "shared_layout": "Vertical 9:16. Top answer card, middle three claim strips, right evidence snippets with source ids and pages, red unsupported claim path to fix tray, bottom faithfulness checklist. Keep claim-to-evidence thread positions identical across zh/en/ja.",
        "variants": {
            "zh": {
                "title": "逐条检查回答是否有证据",
                "subtitle": "流畅不等于可信，每个结论都要能连回来源。",
                "items": [
                    ("claim 1", "把答案拆成可检查结论。"),
                    ("evidence", "找到支持该结论的原文片段。"),
                    ("citation", "记录 source id 和页码。"),
                    ("supported", "证据能直接支撑就通过。"),
                    ("unsupported", "找不到证据就标红并修改。"),
                    ("faithfulness", "检查回答是否忠实于检索内容。"),
                ],
                "footer": "RAG 评估要看证据链，不只看答案像不像。",
                "alt": "RAG faithful citation 检查图：把回答拆成 claim，逐条连接 evidence 和 citation，区分 supported 与 unsupported，并检查 faithfulness。",
            },
            "en": {
                "title": "Check Each Answer Claim Against Evidence",
                "subtitle": "Fluent is not trustworthy; every claim must point back to a source.",
                "items": [
                    ("claim 1", "Split the answer into checkable claims."),
                    ("evidence", "Find the source snippet that supports it."),
                    ("citation", "Record source id and page."),
                    ("supported", "Pass only when evidence directly supports it."),
                    ("unsupported", "Mark red and revise when evidence is missing."),
                    ("faithfulness", "Check whether answer follows retrieved content."),
                ],
                "footer": "RAG evaluation follows evidence chains, not just fluent wording.",
                "alt": "RAG faithfulness and citation check: split answer into claims, connect each to evidence and citation, mark supported or unsupported, and verify faithfulness.",
            },
            "ja": {
                "title": "回答の各主張を証拠で確認",
                "subtitle": "流暢でも信頼できるとは限らない。各主張を出典へ戻す。",
                "items": [
                    ("claim 1", "回答を確認できる主張に分ける。"),
                    ("evidence", "支える原文断片を探す。"),
                    ("citation", "source id とページを記録する。"),
                    ("supported", "証拠が直接支える時だけ通す。"),
                    ("unsupported", "証拠がなければ赤で直す。"),
                    ("faithfulness", "回答が検索内容に忠実か確認する。"),
                ],
                "footer": "RAG 評価は流暢さではなく、証拠のつながりを見る。",
                "alt": "RAG の faithfulness と citation 確認：回答を claim に分け、evidence と citation へ結び、supported と unsupported を判定する。",
            },
        },
    },
    {
        "slug": "ch08-enterprise-kb-permission-citation-map",
        "pages": {
            "en": "docs/ch08-rag/ch03-production/enterprise-kb.md",
            "zh": "i18n/zh-Hans/docusaurus-plugin-content-docs/current/ch08-rag/ch03-production/enterprise-kb.md",
            "ja": "i18n/ja/docusaurus-plugin-content-docs/current/ch08-rag/ch03-production/enterprise-kb.md",
        },
        "scene": "An enterprise knowledge base gate. A user badge with role and department reaches a permission filter before retrieval. Allowed documents pass into retrieval and rerank; blocked confidential documents stay behind a red gate even if semantically similar. The answer panel shows citations and an audit log. Make it clear that permission filtering must happen before retrieval/generation, not after the answer leaks information.",
        "chapter_context": "This image appears in the enterprise knowledge base lesson. The text says enterprise KB needs retrieval, citations, and permissions. It warns not to rely only on semantic relevance. First filter candidates by user permission, then retrieve and rerank, then answer with source citations, otherwise the system may leak internal docs or become untraceable.",
        "shared_layout": "Vertical 9:16. Top user badge, then a permission gate before the vector index. Left allowed docs, right blocked confidential docs behind red gate. Middle retrieval/rerank from allowed set only. Bottom answer with citations and audit log. Keep gate placement and blocked path identical across zh/en/ja.",
        "variants": {
            "zh": {
                "title": "企业知识库先过权限门",
                "subtitle": "相似不代表可见，检索前必须按用户权限过滤。",
                "items": [
                    ("user role", "先读取部门、角色和访问范围。"),
                    ("permission filter", "只让可见文档进入检索。"),
                    ("blocked docs", "相似但无权限的资料不能泄漏。"),
                    ("retrieve / rerank", "只在允许集合内排序证据。"),
                    ("citations", "回答要标明来源。"),
                    ("audit log", "记录谁问了什么、用了哪些来源。"),
                ],
                "footer": "企业 RAG 的底线：先控权限，再谈答案质量。",
                "alt": "企业知识库权限与引用图：user role 先经过 permission filter，blocked docs 被挡住，allowed docs 才进入 retrieve/rerank，回答带 citations 和 audit log。",
            },
            "en": {
                "title": "Enterprise KB Starts with Permission",
                "subtitle": "Similar does not mean visible; filter by access before retrieval.",
                "items": [
                    ("user role", "Read department, role, and access scope."),
                    ("permission filter", "Only visible documents enter retrieval."),
                    ("blocked docs", "Similar but forbidden sources must not leak."),
                    ("retrieve / rerank", "Rank evidence only inside allowed set."),
                    ("citations", "Answer with source references."),
                    ("audit log", "Record who asked and which sources were used."),
                ],
                "footer": "Enterprise RAG controls access before optimizing answer quality.",
                "alt": "Enterprise knowledge base permission and citation map: user role passes permission filter, blocked docs stay hidden, allowed docs enter retrieve/rerank, and answers include citations and audit logs.",
            },
            "ja": {
                "title": "企業 KB は権限確認から始める",
                "subtitle": "似ていても見えてよいとは限らない。検索前に権限で絞る。",
                "items": [
                    ("user role", "部署、役割、アクセス範囲を読む。"),
                    ("permission filter", "見える文書だけを検索へ渡す。"),
                    ("blocked docs", "似ていても権限外の資料は漏らさない。"),
                    ("retrieve / rerank", "許可集合の中だけで証拠を並べる。"),
                    ("citations", "回答に出典を付ける。"),
                    ("audit log", "誰が何を聞き、どの出典を使ったか記録。"),
                ],
                "footer": "企業 RAG は回答品質の前にアクセス制御を守る。",
                "alt": "企業 KB の権限と引用：user role を permission filter に通し、blocked docs を防ぎ、allowed docs だけで retrieve/rerank し、citations と audit log を残す。",
            },
        },
    },
    {
        "slug": "ch08-rag-layer-failure-debug-map",
        "pages": {
            "en": "docs/ch08-rag/ch01-rag/rag-basics.md",
            "zh": "i18n/zh-Hans/docusaurus-plugin-content-docs/current/ch08-rag/ch01-rag/rag-basics.md",
            "ja": "i18n/ja/docusaurus-plugin-content-docs/current/ch08-rag/ch01-rag/rag-basics.md",
        },
        "scene": "A RAG debugging bench with five checkpoints: document parsing, chunking/metadata, embedding index, retrieval top-k, context assembly, and generation. A wrong answer report arrives at the top. The learner follows three diagnostic questions: was the right text chunked, did it enter top-k, did it reach final context? Only after these checks does the learner inspect the generation prompt.",
        "chapter_context": "This image appears in the RAG basics lesson. The text says a RAG failure is not always the model. First check whether the right information was chunked, whether it entered top-k retrieval, whether it was included in final context, and only then suspect generation.",
        "shared_layout": "Vertical 9:16. Top wrong-answer report, then a diagnostic path through parse/chunk/index/top-k/context/generation. Three large question stamps appear beside the relevant stations. Keep the checkpoint order and stamp positions identical across zh/en/ja.",
        "variants": {
            "zh": {
                "title": "RAG 错答先分层排查",
                "subtitle": "不要一上来怪模型，先看证据有没有走到 context。",
                "items": [
                    ("parse", "原文是否被正确读取？"),
                    ("chunk + metadata", "正确片段是否被切出来？"),
                    ("top-k", "检索时有没有召回它？"),
                    ("context", "它是否进入最终提示词？"),
                    ("generation", "证据在场时再检查生成。"),
                    ("fix log", "记录失败层，再针对性修改。"),
                ],
                "footer": "RAG 调试三问：切到了吗？检到了吗？进 context 了吗？",
                "alt": "RAG 分层故障排查：从 parse、chunk、metadata、top-k、context 到 generation，先确认正确证据是否被切分、召回并进入 context。",
            },
            "en": {
                "title": "Debug RAG Failures by Layer",
                "subtitle": "Do not blame the model first; check whether evidence reached context.",
                "items": [
                    ("parse", "Was the source read correctly?"),
                    ("chunk + metadata", "Was the right text chunk created?"),
                    ("top-k", "Did retrieval recall it?"),
                    ("context", "Did it enter the final prompt?"),
                    ("generation", "Check generation after evidence is present."),
                    ("fix log", "Record the failing layer before changing."),
                ],
                "footer": "RAG debug questions: chunked? retrieved? included in context?",
                "alt": "RAG layer failure debugging: inspect parse, chunk, metadata, top-k, context, and generation to see whether the right evidence was chunked, retrieved, and included.",
            },
            "ja": {
                "title": "RAG の誤答は層ごとに調べる",
                "subtitle": "すぐモデルを疑わず、証拠が context まで届いたか見る。",
                "items": [
                    ("parse", "元文書を正しく読めたか？"),
                    ("chunk + metadata", "正しい断片が作られたか？"),
                    ("top-k", "検索で召回されたか？"),
                    ("context", "最終 prompt に入ったか？"),
                    ("generation", "証拠がある時に生成を確認する。"),
                    ("fix log", "失敗した層を記録して直す。"),
                ],
                "footer": "RAG 調査の三問：chunk 化？検索？context 入り？",
                "alt": "RAG 障害の層別調査：parse、chunk、metadata、top-k、context、generation を見て、正しい証拠が届いたか確認する。",
            },
        },
    },
    {
        "slug": "ch09-agent-systems",
        "pages": {
            "en": "docs/ch09-ai-agents/index.md",
            "zh": "i18n/zh-Hans/docusaurus-plugin-content-docs/current/ch09-ai-agents/index.md",
            "ja": "i18n/ja/docusaurus-plugin-content-docs/current/ch09-ai-agents/index.md",
        },
        "scene": "An agent execution loop as a controlled work desk. A goal card starts the loop. The agent reads current state, writes a plan, selects one tool, sends a tool call, reads the observation, updates memory, writes a trace entry, checks stop and safety conditions, then either continues or returns a final result. Show that an agent is not just a chatbot with tools; it is a controlled loop with state and trace.",
        "chapter_context": "This image appears at the top of Chapter 9. The text says an agent acts toward a goal: plan next step, call a tool, read observation, adjust, stop safely, and leave a trace. It defines goal, state, plan, tool, observation, memory, and trace.",
        "shared_layout": "Vertical 9:16. Top goal card. Center circular but concrete desk loop: state, plan, tool call, observation, memory. Bottom trace ledger and stop/safety gate. Keep loop order, ledger, and gate identical across zh/en/ja.",
        "variants": {
            "zh": {
                "title": "Agent 是受控执行循环",
                "subtitle": "目标、状态、计划、工具、观察和轨迹缺一不可。",
                "items": [
                    ("goal", "明确本轮要完成什么。"),
                    ("state", "读取当前信息和约束。"),
                    ("plan", "决定下一步动作。"),
                    ("tool call", "按参数调用外部工具。"),
                    ("observation", "读取工具返回结果。"),
                    ("memory / trace", "保存关键事实和执行记录。"),
                    ("stop check", "满足条件才结束或交付。"),
                ],
                "footer": "可靠 Agent 不是会聊天，而是能安全循环执行。",
                "alt": "Agent 系统执行循环：goal、state、plan、tool call、observation、memory、trace 和 stop check 共同组成受控执行过程。",
            },
            "en": {
                "title": "An Agent Is a Controlled Execution Loop",
                "subtitle": "Goal, state, plan, tool, observation, and trace all matter.",
                "items": [
                    ("goal", "Clarify what this run must finish."),
                    ("state", "Read current facts and constraints."),
                    ("plan", "Choose the next action."),
                    ("tool call", "Call an external tool with parameters."),
                    ("observation", "Read the tool result."),
                    ("memory / trace", "Save facts and execution records."),
                    ("stop check", "Finish only when conditions are met."),
                ],
                "footer": "A reliable agent is not chatty; it safely loops through work.",
                "alt": "Agent system loop: goal, state, plan, tool call, observation, memory, trace, and stop check form controlled execution.",
            },
            "ja": {
                "title": "Agent は制御された実行ループ",
                "subtitle": "goal、state、plan、tool、observation、trace がそろって動く。",
                "items": [
                    ("goal", "この実行で終えることを明確にする。"),
                    ("state", "現在の事実と制約を読む。"),
                    ("plan", "次の行動を決める。"),
                    ("tool call", "引数を付けて外部ツールを呼ぶ。"),
                    ("observation", "ツール結果を読む。"),
                    ("memory / trace", "重要事実と実行記録を残す。"),
                    ("stop check", "条件を満たした時だけ終了する。"),
                ],
                "footer": "信頼できる Agent は雑談ではなく、安全に作業を回す仕組み。",
                "alt": "Agent システムの実行ループ：goal、state、plan、tool call、observation、memory、trace、stop check。",
            },
        },
    },
    {
        "slug": "ch09-agent-cost-routing-cache-budget-map",
        "pages": {
            "en": "docs/ch09-ai-agents/ch03-production/cost-control.md",
            "zh": "i18n/zh-Hans/docusaurus-plugin-content-docs/current/ch09-ai-agents/ch03-production/cost-control.md",
            "ja": "i18n/ja/docusaurus-plugin-content-docs/current/ch09-ai-agents/ch03-production/cost-control.md",
        },
        "scene": "An agent cost control dashboard that follows one task chain. A user request goes through a cache check, a small-model router, context trimming, tool budget meter, retry guard, and only then a large-model path for hard steps. Show before/after cost receipts: without routing, every step uses the expensive model and tool calls pile up; with routing and cache, only necessary steps spend more.",
        "chapter_context": "This image appears in the agent cost lesson. The text says cost expands from a single model call to a task-chain bill. It highlights model routing, context length, tool calls, cache hits, failed retries, and budget limits. It suggests small-model routing or filtering, then large model only for complex parts.",
        "shared_layout": "Vertical 9:16. Top user request. Left red expensive path without controls. Right controlled path with cache, router, context trim, tool budget, retry guard, large model only for hard step. Bottom before/after cost receipts. Keep paths and meters identical across zh/en/ja.",
        "variants": {
            "zh": {
                "title": "Agent 成本控制路线",
                "subtitle": "成本来自整条任务链，不只是一次模型调用。",
                "items": [
                    ("cache hit", "重复问题优先复用结果。"),
                    ("model routing", "简单步骤交给小模型。"),
                    ("context trim", "只保留必要上下文。"),
                    ("tool budget", "限制搜索和 API 调用次数。"),
                    ("retry guard", "失败重试要有上限。"),
                    ("large model", "只用于复杂判断或最终合成。"),
                    ("cost receipt", "按步骤记录 token、tool 和 retry。"),
                ],
                "footer": "先设计路由、缓存和预算，再让 Agent 自动跑。",
                "alt": "Agent 成本控制图：cache hit、model routing、context trim、tool budget、retry guard 和 large model 路径共同降低任务链成本。",
            },
            "en": {
                "title": "Agent Cost Control Route",
                "subtitle": "Cost comes from the whole task chain, not one model call.",
                "items": [
                    ("cache hit", "Reuse results for repeated questions."),
                    ("model routing", "Send simple steps to a small model."),
                    ("context trim", "Keep only necessary context."),
                    ("tool budget", "Limit search and API calls."),
                    ("retry guard", "Cap failed retries."),
                    ("large model", "Use only for hard reasoning or final synthesis."),
                    ("cost receipt", "Record token, tool, and retry cost by step."),
                ],
                "footer": "Design routing, cache, and budget before letting agents run.",
                "alt": "Agent cost control route: cache hit, model routing, context trim, tool budget, retry guard, and selective large model use reduce task-chain cost.",
            },
            "ja": {
                "title": "Agent のコスト制御ルート",
                "subtitle": "費用は一回の model call ではなく、タスク列全体から生まれる。",
                "items": [
                    ("cache hit", "同じ質問は結果を再利用する。"),
                    ("model routing", "簡単な手順は小さいモデルへ渡す。"),
                    ("context trim", "必要な文脈だけを残す。"),
                    ("tool budget", "検索と API 呼び出しを制限する。"),
                    ("retry guard", "失敗リトライに上限を置く。"),
                    ("large model", "難しい判断や最終統合だけに使う。"),
                    ("cost receipt", "token、tool、retry の費用を手順別に記録。"),
                ],
                "footer": "Agent を走らせる前に、routing、cache、budget を設計する。",
                "alt": "Agent コスト制御：cache hit、model routing、context trim、tool budget、retry guard、大きなモデルの限定利用でタスク列費用を下げる。",
            },
        },
    },
    {
        "slug": "ch09-mcp-server-tool-contract-map",
        "pages": {
            "en": "docs/ch09-ai-agents/ch04-mcp/mcp-server.md",
            "zh": "i18n/zh-Hans/docusaurus-plugin-content-docs/current/ch09-ai-agents/ch04-mcp/mcp-server.md",
            "ja": "i18n/ja/docusaurus-plugin-content-docs/current/ch09-ai-agents/ch04-mcp/mcp-server.md",
        },
        "scene": "An MCP server gatekeeper teaching scene. The client first asks list_tools and receives tool names with schemas. Then a call_tool request with parameters reaches the server. The server validates arguments, executes the real backend action, standardizes the returned result, and turns errors into structured messages the client can understand. Show a rejected request with a missing required parameter and a successful request returning a clean result.",
        "chapter_context": "This image appears in the MCP server lesson. The text says the server is a gatekeeper: it exposes list_tools, validates call_tool parameters, executes logic, standardizes returned results, and turns errors into client-understandable structures.",
        "shared_layout": "Vertical 9:16. Top client-server handshake. Middle list_tools schema card. Lower call_tool path through validation, execution, standardized result. Side red rejected missing-parameter path. Keep the gate and two paths identical across zh/en/ja.",
        "variants": {
            "zh": {
                "title": "MCP Server 是工具契约门卫",
                "subtitle": "先公开工具 schema，再验证调用、执行逻辑、统一返回。",
                "items": [
                    ("list_tools", "告诉客户端有哪些工具和参数。"),
                    ("schema", "定义必填字段和类型。"),
                    ("call_tool", "客户端带参数发起调用。"),
                    ("validate", "缺字段或类型错就拒绝。"),
                    ("execute", "通过后执行真实后端逻辑。"),
                    ("standard result", "把结果或错误整理成统一结构。"),
                ],
                "footer": "MCP 的价值是让工具调用可描述、可验证、可追踪。",
                "alt": "MCP Server 工具契约图：list_tools 暴露 schema，call_tool 带参数进入 validate，通过后 execute，返回 standard result，错误结构化。",
            },
            "en": {
                "title": "MCP Server Is the Tool Contract Gate",
                "subtitle": "Expose schemas, validate calls, run logic, and standardize returns.",
                "items": [
                    ("list_tools", "Tell the client available tools and arguments."),
                    ("schema", "Define required fields and types."),
                    ("call_tool", "Client sends a request with parameters."),
                    ("validate", "Reject missing fields or wrong types."),
                    ("execute", "Run backend logic after validation."),
                    ("standard result", "Return results or errors in one structure."),
                ],
                "footer": "MCP makes tool calls describable, validatable, and traceable.",
                "alt": "MCP server tool contract: list_tools exposes schema, call_tool sends parameters, validation rejects bad requests, execution runs backend logic, and standard result structures output.",
            },
            "ja": {
                "title": "MCP Server は tool 契約の門番",
                "subtitle": "schema を公開し、呼び出しを検証し、実行し、戻り値をそろえる。",
                "items": [
                    ("list_tools", "利用できる tool と引数を client へ示す。"),
                    ("schema", "必須項目と型を定義する。"),
                    ("call_tool", "client が引数付きで呼び出す。"),
                    ("validate", "不足項目や型違いを拒否する。"),
                    ("execute", "検証後に backend logic を実行する。"),
                    ("standard result", "結果や error を同じ構造で返す。"),
                ],
                "footer": "MCP は tool 呼び出しを記述可能、検証可能、追跡可能にする。",
                "alt": "MCP Server の tool 契約：list_tools が schema を公開し、call_tool を validate し、execute 後に standard result と error を返す。",
            },
        },
    },
    {
        "slug": "ch10-classification-architecture-evolution-map",
        "pages": {
            "en": "docs/ch10-computer-vision/ch02-classification/02-modern-architectures.md",
            "zh": "i18n/zh-Hans/docusaurus-plugin-content-docs/current/ch10-computer-vision/ch02-classification/02-modern-architectures.md",
            "ja": "i18n/ja/docusaurus-plugin-content-docs/current/ch10-computer-vision/ch02-classification/02-modern-architectures.md",
        },
        "scene": "A practical image classification architecture evolution board, not a leaderboard. Four model stations handle the same image classification task: VGG as the classic depth starting point, ResNet adding skip connections to make deep training stable, EfficientNet balancing accuracy and compute, and ConvNeXt modernizing convolution design. At the bottom, a beginner chooses ResNet as the safe first project baseline, EfficientNet when resources matter, and the VGG->ResNet->ConvNeXt path when studying architecture evolution.",
        "chapter_context": "This image appears beside a practical model selection table. The text says VGG is a classic teaching starting point, ResNet is the reliable engineering baseline, EfficientNet is valuable when caring about more value for the same resources, and ConvNeXt shows how convolutional families can be modernized. The tip says do not read the diagram as a leaderboard; read it as a problem evolution map.",
        "shared_layout": "Vertical 9:16. Top shared input image and task. Middle four architecture stations in evolution order: VGG, ResNet, EfficientNet, ConvNeXt. Bottom decision strip for first project, efficiency, and evolution study. Keep station order, icons, and decision strip identical across zh/en/ja.",
        "variants": {
            "zh": {
                "title": "分类架构演进不是排行榜",
                "subtitle": "看每个模型解决了什么问题，再决定先用谁。",
                "items": [
                    ("VGG", "用堆叠卷积理解深度。"),
                    ("ResNet", "skip connection 让深层训练更稳。"),
                    ("EfficientNet", "在性能和计算量之间做平衡。"),
                    ("ConvNeXt", "把卷积架构现代化。"),
                    ("first project", "初学项目优先选 ResNet baseline。"),
                    ("selection", "按数据量、算力和学习目标选择。"),
                ],
                "footer": "读架构图时问：它解决了哪个训练或资源问题？",
                "alt": "图像分类架构演进图：VGG、ResNet、EfficientNet、ConvNeXt 按问题演进排列，并说明初学项目先选 ResNet baseline。",
            },
            "en": {
                "title": "Classification Architectures Are Not a Leaderboard",
                "subtitle": "Ask what problem each model solved before choosing one.",
                "items": [
                    ("VGG", "Use stacked convolutions to understand depth."),
                    ("ResNet", "Skip connections make deep training stable."),
                    ("EfficientNet", "Balance performance and compute."),
                    ("ConvNeXt", "Modernize the convolutional path."),
                    ("first project", "Start with a ResNet baseline."),
                    ("selection", "Choose by data size, compute, and learning goal."),
                ],
                "footer": "When reading architectures, ask which training or resource problem they solve.",
                "alt": "Image classification architecture evolution map: VGG, ResNet, EfficientNet, and ConvNeXt are shown as problem evolution choices, with ResNet as a beginner baseline.",
            },
            "ja": {
                "title": "分類アーキテクチャは順位表ではない",
                "subtitle": "各モデルが解いた問題を見てから選ぶ。",
                "items": [
                    ("VGG", "積み重ねた畳み込みで深さを学ぶ。"),
                    ("ResNet", "skip connection で深い学習を安定させる。"),
                    ("EfficientNet", "性能と計算量のバランスを取る。"),
                    ("ConvNeXt", "畳み込みの流れを現代化する。"),
                    ("first project", "最初は ResNet baseline から始める。"),
                    ("selection", "データ量、計算資源、学習目的で選ぶ。"),
                ],
                "footer": "アーキテクチャを見る時は、どの訓練問題や資源問題を解くかを問う。",
                "alt": "画像分類アーキテクチャの進化：VGG、ResNet、EfficientNet、ConvNeXt を問題解決の流れとして示し、初学者は ResNet baseline から始める。",
            },
        },
    },
    {
        "slug": "ch10-classification-training-diagnosis-map",
        "pages": {
            "en": "docs/ch10-computer-vision/ch02-classification/03-training-tricks.md",
            "zh": "i18n/zh-Hans/docusaurus-plugin-content-docs/current/ch10-computer-vision/ch02-classification/03-training-tricks.md",
            "ja": "i18n/ja/docusaurus-plugin-content-docs/current/ch10-computer-vision/ch02-classification/03-training-tricks.md",
        },
        "scene": "An image classification training clinic. A model has poor validation performance. The clinic board first checks data: split quality, label-preserving augmentation, class imbalance, duplicate leakage. Then it checks training: loss curves for underfitting, overfitting, noisy training, learning rate, batch size. Finally it checks evaluation: confusion matrix and wrong image samples. Show that learners should inspect evidence before changing the backbone.",
        "chapter_context": "This image appears after a table for telling overfitting from underfitting. The text says augmentation should preserve label semantics, random augmentation should apply to train not validation, and poor classification should be diagnosed through loss curves, validation leakage, class imbalance, and error samples before rushing to change the model.",
        "shared_layout": "Vertical 9:16. Top symptom card: poor validation. Middle three clinic lanes: data, training, evaluation. Each lane has concrete artifacts: split folders, augmentation examples, loss curves, confusion matrix, wrong-image tray. Bottom decision note says change data/training first, backbone only after evidence. Keep lanes and artifacts identical across zh/en/ja.",
        "variants": {
            "zh": {
                "title": "分类训练问题诊断",
                "subtitle": "性能差时先查数据、曲线和错例，不要急着换模型。",
                "items": [
                    ("data split", "确认 train / val 没有重复泄漏。"),
                    ("augmentation", "增强不能改变 label 语义。"),
                    ("loss curves", "区分欠拟合、过拟合和震荡。"),
                    ("learning rate", "曲线剧烈波动时先调小。"),
                    ("confusion matrix", "看哪些类别被系统性混淆。"),
                    ("wrong samples", "用错例决定下一步修改。"),
                ],
                "footer": "先诊断证据，再决定调数据、训练或 backbone。",
                "alt": "图像分类训练诊断图：检查 data split、augmentation、loss curves、learning rate、confusion matrix 和 wrong samples，再决定是否换 backbone。",
            },
            "en": {
                "title": "Classification Training Diagnosis",
                "subtitle": "When performance is poor, inspect data, curves, and errors before changing models.",
                "items": [
                    ("data split", "Check train / val duplicates and leakage."),
                    ("augmentation", "Transforms must preserve label meaning."),
                    ("loss curves", "Separate underfitting, overfitting, and noise."),
                    ("learning rate", "Lower it when curves fluctuate wildly."),
                    ("confusion matrix", "Find classes that are confused systematically."),
                    ("wrong samples", "Let errors decide the next change."),
                ],
                "footer": "Diagnose evidence before tuning data, training, or backbone.",
                "alt": "Image classification training diagnosis: check data split, augmentation, loss curves, learning rate, confusion matrix, and wrong samples before changing backbone.",
            },
            "ja": {
                "title": "分類訓練の問題診断",
                "subtitle": "性能が悪い時は、モデル変更前にデータ、曲線、誤例を見る。",
                "items": [
                    ("data split", "train / val の重複と leakage を確認。"),
                    ("augmentation", "変換は label の意味を保つ。"),
                    ("loss curves", "未学習、過学習、揺れを分ける。"),
                    ("learning rate", "曲線が大きく揺れるなら下げる。"),
                    ("confusion matrix", "混同される class を見つける。"),
                    ("wrong samples", "誤例から次の修正を決める。"),
                ],
                "footer": "証拠を診断してから、データ、訓練、backbone を調整する。",
                "alt": "画像分類の訓練診断：data split、augmentation、loss curves、learning rate、confusion matrix、wrong samples を確認してから backbone を変える。",
            },
        },
    },
    {
        "slug": "ch11-seq2seq-chapter-flow",
        "pages": {
            "en": "docs/ch11-nlp/ch05-seq2seq/00-roadmap.md",
            "zh": "i18n/zh-Hans/docusaurus-plugin-content-docs/current/ch11-nlp/ch05-seq2seq/00-roadmap.md",
            "ja": "i18n/ja/docusaurus-plugin-content-docs/current/ch11-nlp/ch05-seq2seq/00-roadmap.md",
        },
        "scene": "A Seq2Seq generation bridge. An input sentence enters an encoder that creates a compact representation. A decoder then generates output tokens one by one. Attention beams connect each output step back to useful input tokens. Around the bridge, show common tasks: translation, summarization, rewriting, dialogue, and correction. Also show a bottleneck warning when the decoder cannot look back enough.",
        "chapter_context": "This image appears in the Seq2Seq roadmap. The text says Seq2Seq handles tasks where both input and output are sequences: translation, summarization, rewriting, dialogue, and error correction. It says generation happens step by step, and attention helps the decoder look back at useful input positions.",
        "shared_layout": "Vertical 9:16. Top input sequence. Middle encoder-decoder bridge with attention lines from decoder steps back to input tokens. Side task badges for translation, summary, rewrite, dialogue, correction. Bottom bottleneck warning and modern LLM bridge note. Keep token positions and attention lines identical across zh/en/ja.",
        "variants": {
            "zh": {
                "title": "Seq2Seq：输入序列到输出序列",
                "subtitle": "编码输入，逐步解码输出，attention 帮助回看关键信息。",
                "items": [
                    ("input sequence", "原文可以和输出长度不同。"),
                    ("encoder", "把输入读成上下文表示。"),
                    ("decoder", "一次生成一个输出 token。"),
                    ("attention", "每一步回看相关输入位置。"),
                    ("tasks", "翻译、摘要、改写、对话、纠错。"),
                    ("bottleneck", "只靠一个向量会丢信息。"),
                ],
                "footer": "现代生成模型仍保留逐步生成和注意力回看的核心直觉。",
                "alt": "Seq2Seq 与 attention 路线图：input sequence 进入 encoder，decoder 逐步生成 output token，attention 回看输入，用于翻译、摘要、改写、对话和纠错。",
            },
            "en": {
                "title": "Seq2Seq: Input Sequence to Output Sequence",
                "subtitle": "Encode input, decode output step by step, and use attention to look back.",
                "items": [
                    ("input sequence", "Source and output can have different lengths."),
                    ("encoder", "Read input into a context representation."),
                    ("decoder", "Generate one output token at a time."),
                    ("attention", "Look back to useful input positions."),
                    ("tasks", "Translation, summary, rewrite, dialogue, correction."),
                    ("bottleneck", "One vector alone can lose information."),
                ],
                "footer": "Modern generation still relies on stepwise output and attention intuition.",
                "alt": "Seq2Seq and attention roadmap: input sequence enters encoder, decoder generates output tokens step by step, attention looks back to input for translation, summarization, rewriting, dialogue, and correction.",
            },
            "ja": {
                "title": "Seq2Seq：入力列から出力列へ",
                "subtitle": "入力を encode し、出力を一歩ずつ decode し、attention で見返す。",
                "items": [
                    ("input sequence", "入力と出力の長さは違ってよい。"),
                    ("encoder", "入力を文脈表現として読む。"),
                    ("decoder", "出力 token を一つずつ生成する。"),
                    ("attention", "各ステップで有用な入力位置を見る。"),
                    ("tasks", "翻訳、要約、書き換え、対話、訂正。"),
                    ("bottleneck", "1つのベクトルだけでは情報が落ちる。"),
                ],
                "footer": "現代の生成にも、逐次生成と attention の直感が残っている。",
                "alt": "Seq2Seq と attention：input sequence を encoder が読み、decoder が output token を逐次生成し、attention が入力位置を見返す。",
            },
        },
    },
    {
        "slug": "ch12-multimodal-aigc",
        "pages": {
            "en": "docs/ch12-multimodal/index.md",
            "zh": "i18n/zh-Hans/docusaurus-plugin-content-docs/current/ch12-multimodal/index.md",
            "ja": "i18n/ja/docusaurus-plugin-content-docs/current/ch12-multimodal/index.md",
        },
        "scene": "A multimodal AIGC workbench for a real product workflow. Inputs include PDF pages, screenshots, charts, audio waveform, video frames, and text notes. The system turns them into structured records with source, page, region, timestamp, and license fields. These records connect to RAG, Agent, and generation tools. A reviewer checks risk and quality before the final export goes to a report, app, or creative asset package.",
        "chapter_context": "This image appears at the top of Chapter 12. The text says AI is no longer only text: images, PDFs, audio, video, screenshots, charts, and generated assets enter the same workflow. It says do not chase demos; turn non-text input into structured records, connect RAG/Agent, generate or edit assets, review risks, and export usable results.",
        "shared_layout": "Vertical 9:16. Top multimodal inputs. Middle structured record table with source/page/region/time/license. Right connected modules RAG, Agent, generation. Bottom review gate and export outputs. Keep input set, record table, modules, and gate identical across zh/en/ja.",
        "variants": {
            "zh": {
                "title": "多模态 AIGC 工作台",
                "subtitle": "把图片、PDF、音频、视频先变成可追踪记录，再接入系统。",
                "items": [
                    ("inputs", "PDF、截图、图表、音频、视频、文本。"),
                    ("structured records", "记录来源、页码、区域、时间和许可。"),
                    ("RAG", "把非文本证据接入检索回答。"),
                    ("Agent", "让工具按步骤处理多种媒体。"),
                    ("generate / edit", "生成或修改可交付素材。"),
                    ("review gate", "检查版权、隐私、幻觉和质量。"),
                    ("export", "输出报告、应用或创意内容包。"),
                ],
                "footer": "多模态项目先追踪证据，再生成结果。",
                "alt": "多模态 AIGC 工作流：PDF、截图、图表、音频、视频和文本变成 structured records，再连接 RAG、Agent、生成编辑、审核和导出。",
            },
            "en": {
                "title": "Multimodal AIGC Workbench",
                "subtitle": "Turn media into traceable records before connecting it to systems.",
                "items": [
                    ("inputs", "PDF, screenshot, chart, audio, video, text."),
                    ("structured records", "Track source, page, region, time, license."),
                    ("RAG", "Connect non-text evidence to retrieval answers."),
                    ("Agent", "Let tools process media step by step."),
                    ("generate / edit", "Create or revise deliverable assets."),
                    ("review gate", "Check copyright, privacy, hallucination, quality."),
                    ("export", "Ship reports, apps, or creative packages."),
                ],
                "footer": "Multimodal projects track evidence first, then generate results.",
                "alt": "Multimodal AIGC workflow: PDF, screenshot, chart, audio, video, and text become structured records connected to RAG, Agent, generation, review, and export.",
            },
            "ja": {
                "title": "マルチモーダル AIGC ワークベンチ",
                "subtitle": "メディアを追跡可能な記録にしてから、システムへつなぐ。",
                "items": [
                    ("inputs", "PDF、スクリーンショット、図表、音声、動画、テキスト。"),
                    ("structured records", "出典、ページ、領域、時刻、license を記録。"),
                    ("RAG", "非テキスト証拠を検索回答へつなぐ。"),
                    ("Agent", "tool が手順ごとに媒体を処理する。"),
                    ("generate / edit", "納品できる素材を生成・修正する。"),
                    ("review gate", "著作権、privacy、hallucination、品質を見る。"),
                    ("export", "レポート、アプリ、creative package へ出力。"),
                ],
                "footer": "マルチモーダルは、まず証拠を追跡し、それから生成する。",
                "alt": "マルチモーダル AIGC：PDF、スクリーンショット、図表、音声、動画、テキストを structured records にし、RAG、Agent、生成、review、export へつなぐ。",
            },
        },
    },
    {
        "slug": "ch12-multimodal-workflow-loop",
        "pages": {
            "en": "docs/ch12-multimodal/index.md",
            "zh": "i18n/zh-Hans/docusaurus-plugin-content-docs/current/ch12-multimodal/index.md",
            "ja": "i18n/ja/docusaurus-plugin-content-docs/current/ch12-multimodal/index.md",
        },
        "scene": "A multimodal workflow loop with five concrete stations. Input station records source, license, and version. Parse and align station performs OCR, layout analysis, visual labels, and transcript alignment with page, region, and time references. Understand and generate station summarizes, retrieves, or creates assets. Edit and review station compares versions and rejected outputs. Export and integrate station produces README, files, API payload, or app screen with limitations noted.",
        "chapter_context": "This image appears near the start of Chapter 12. The workflow table lists Input with source/license/version, Parse and align with OCR/layout/visual/transcript plus page/region/time refs, Understand/generate, Edit/review, and Export/integrate. The text stresses preserving evidence and review through the workflow.",
        "shared_layout": "Vertical 9:16. Five stations in a loop around one media project board: input, parse/align, understand/generate, edit/review, export/integrate. A source-evidence ribbon travels through all stations. Keep station order and ribbon identical across zh/en/ja.",
        "variants": {
            "zh": {
                "title": "多模态项目工作流",
                "subtitle": "每一步都要保留来源、版本和证据位置。",
                "items": [
                    ("Input", "记录 source、license、version。"),
                    ("Parse / align", "OCR、layout、visual、transcript 对齐。"),
                    ("refs", "保留 page、region、time。"),
                    ("Understand / generate", "总结、检索、生成或编辑。"),
                    ("Review", "比较版本，标记拒绝输出。"),
                    ("Export", "交付 README、文件、API 或应用界面。"),
                ],
                "footer": "多模态交付不是一次生成，而是有证据的循环。",
                "alt": "多模态工作流闭环：Input 记录 source/license/version，Parse align 做 OCR/layout/visual/transcript，保留 refs，再 Understand generate、Review 和 Export。",
            },
            "en": {
                "title": "Multimodal Project Workflow",
                "subtitle": "Keep source, version, and evidence location at every step.",
                "items": [
                    ("Input", "Record source, license, version."),
                    ("Parse / align", "Run OCR, layout, visual, transcript alignment."),
                    ("refs", "Keep page, region, and time."),
                    ("Understand / generate", "Summarize, retrieve, generate, or edit."),
                    ("Review", "Compare versions and rejected outputs."),
                    ("Export", "Ship README, files, API, or app UI."),
                ],
                "footer": "Multimodal delivery is an evidence loop, not a one-shot generation.",
                "alt": "Multimodal workflow loop: Input records source/license/version, Parse align runs OCR/layout/visual/transcript, refs preserve page/region/time, then understand generate, review, and export.",
            },
            "ja": {
                "title": "マルチモーダルプロジェクトの流れ",
                "subtitle": "各ステップで出典、version、証拠位置を残す。",
                "items": [
                    ("Input", "source、license、version を記録。"),
                    ("Parse / align", "OCR、layout、visual、transcript を合わせる。"),
                    ("refs", "page、region、time を残す。"),
                    ("Understand / generate", "要約、検索、生成、編集を行う。"),
                    ("Review", "version と却下出力を比較する。"),
                    ("Export", "README、file、API、app UI として出す。"),
                ],
                "footer": "マルチモーダル納品は一回生成ではなく、証拠付きのループ。",
                "alt": "マルチモーダル workflow：Input で source/license/version、Parse align で OCR/layout/visual/transcript、refs で page/region/time を残し、Understand generate、Review、Export へ進む。",
            },
        },
    },
    {
        "slug": "diffusion-noise-denoise",
        "pages": {
            "en": "docs/ch12-multimodal/ch02-image-gen/01-diffusion-models.md",
            "zh": "i18n/zh-Hans/docusaurus-plugin-content-docs/current/ch12-multimodal/ch02-image-gen/01-diffusion-models.md",
            "ja": "i18n/ja/docusaurus-plugin-content-docs/current/ch12-multimodal/ch02-image-gen/01-diffusion-models.md",
        },
        "scene": "A diffusion model teaching strip with two mirrored paths. The forward path starts from a clean training image and adds noise step by step until it becomes nearly pure noise. The training panel learns to predict the noise added at each step. The reverse path starts from random noise and denoises step by step, guided by a prompt, until a new image appears. Add two small comparison cards: GAN tries one-shot generation, VAE learns a latent space; diffusion learns a gradual cleanup path.",
        "chapter_context": "This image appears in the diffusion models lesson. The text contrasts GAN one-shot generation, VAE latent distribution, and diffusion: first corrupt a real sample step by step, then learn to clean step by step. The learner should see why denoising is the core intuition.",
        "shared_layout": "Vertical 9:16. Top forward noising path left-to-right, middle training learns predicted noise, lower reverse denoising path right-to-left, bottom tiny GAN/VAE comparison cards. Keep the clean image, noise sequence, reverse sequence, and comparison cards identical across zh/en/ja.",
        "variants": {
            "zh": {
                "title": "Diffusion：先加噪，再学会去噪",
                "subtitle": "模型不是一次画完，而是从噪声一步步清理出图像。",
                "items": [
                    ("forward noising", "真实图像逐步加入 noise。"),
                    ("training target", "学习每一步加了多少噪声。"),
                    ("random noise", "生成时从纯噪声开始。"),
                    ("reverse denoising", "一步步预测并去掉噪声。"),
                    ("prompt guidance", "文本提示引导清理方向。"),
                    ("GAN / VAE", "对比：一次生成或学习 latent space。"),
                ],
                "footer": "理解扩散模型：训练学会还原，生成反向清理。",
                "alt": "扩散模型加噪去噪图：forward noising 把真实图像逐步变成 noise，training target 学预测噪声，reverse denoising 从 random noise 生成图像。",
            },
            "en": {
                "title": "Diffusion: Add Noise, Then Learn to Remove It",
                "subtitle": "The model does not draw once; it cleans an image out of noise step by step.",
                "items": [
                    ("forward noising", "Add noise to a real image step by step."),
                    ("training target", "Learn how much noise was added."),
                    ("random noise", "Generation starts from pure noise."),
                    ("reverse denoising", "Predict and remove noise gradually."),
                    ("prompt guidance", "Text steers the cleanup direction."),
                    ("GAN / VAE", "Contrast: one-shot generation or latent space."),
                ],
                "footer": "Diffusion trains restoration, then generates by reversing cleanup.",
                "alt": "Diffusion noise and denoise diagram: forward noising turns a real image into noise, training predicts noise, and reverse denoising starts from random noise to generate an image.",
            },
            "ja": {
                "title": "Diffusion：ノイズを足し、取り除くことを学ぶ",
                "subtitle": "一度で描くのではなく、noise から少しずつ画像を整える。",
                "items": [
                    ("forward noising", "本物の画像へ段階的に noise を足す。"),
                    ("training target", "各ステップの noise 量を学ぶ。"),
                    ("random noise", "生成は純粋な noise から始める。"),
                    ("reverse denoising", "noise を予測し、少しずつ取り除く。"),
                    ("prompt guidance", "テキストが整える方向を導く。"),
                    ("GAN / VAE", "比較：一度生成、または latent space 学習。"),
                ],
                "footer": "Diffusion は復元を学び、逆向きの cleanup で生成する。",
                "alt": "Diffusion のノイズ付与と除去：forward noising で本物画像を noise にし、training target で noise を学び、reverse denoising で random noise から画像を作る。",
            },
        },
    },
]

for direct_group in DIRECT_TRIPLET_GROUPS:
    register_svg_replacement_group(
        slug=direct_group["slug"],
        pages=direct_group["pages"],
        scene=direct_group["scene"],
        chapter_context=direct_group["chapter_context"],
        shared_layout=direct_group["shared_layout"],
        variants=direct_group["variants"],
        callouts=[],
    )

EXPERIMENT_RESULT_GROUPS: list[dict[str, Any]] = [
    {
        "slug": "ch05-linear-regression-lab-result-map",
        "pages": {
            "en": "docs/ch05-machine-learning/ch02-supervised/01-linear-regression.md",
            "zh": "i18n/zh-Hans/docusaurus-plugin-content-docs/current/ch05-machine-learning/ch02-supervised/01-linear-regression.md",
            "ja": "i18n/ja/docusaurus-plugin-content-docs/current/ch05-machine-learning/ch02-supervised/01-linear-regression.md",
        },
        "scene": "A linear regression lab result board based on the expected output of ch05_linear_regression_lab.py. Do not draw full code blocks or terminal logs. Show a synthetic house-price dataset with three feature dials: area, rooms, age. Show baseline average prediction as a flat weak line with baseline_rmse=123.23. Show the learned linear model as a fitted line/plane with linear_rmse=11.68, linear_mae=8.59, linear_r2=0.991. Show the learned equation with intercept=30.54 and coefficients [2.85, 17.97, -1.72]. Show one prediction card: first_prediction=457.07 and first_residual=30.0, with residual drawn as a vertical error bar from predicted to true. Show polynomial Ridge as a comparison card with ridge_poly_rmse=13.8, labeled worse than simple linear in this run. Do not invent extra feature values or extra metrics.",
        "chapter_context": "The image is inserted after the expected output of the linear regression lab. Nearby text explains baseline RMSE, learned coefficients, first residual, polynomial Ridge, and why the simpler linear model is safer in this synthetic run.",
        "shared_layout": "Vertical 9:16. Top title and subtitle. Upper section shows input features flowing into the prediction formula. Middle has three result panels: baseline flat prediction, linear fit, and polynomial Ridge comparison. Lower section has the first residual error-bar card. Bottom summarizes model choice: features matter, residuals show misses, and lower validation/test error wins. Keep values, panel order, colors, and reading path identical across zh/en/ja.",
        "variants": {
            "zh": {
                "title": "线性回归实验结果怎么看",
                "subtitle": "把 RMSE、系数、残差和模型选择连成一条证据链。",
                "items": [
                    ("baseline_rmse", "123.23：只猜平均值，误差很大。"),
                    ("linear_rmse", "11.68：特征明显解释了价格。"),
                    ("coefficients", "[2.85, 17.97, -1.72] 读方向和量级。"),
                    ("first_residual", "30.0：真实值比预测高约 30。"),
                    ("ridge_poly_rmse", "13.8：更复杂不一定更好。"),
                    ("choose simpler", "本次 simple linear 更稳。"),
                ],
                "footer": "回归实验要看基线、误差大小、系数含义和残差。",
                "alt": "线性回归实验结果图：baseline RMSE、linear RMSE、系数、first residual 和 polynomial Ridge 对比。",
            },
            "en": {
                "title": "Reading Linear Regression Results",
                "subtitle": "Connect RMSE, coefficients, residuals, and model choice as evidence.",
                "items": [
                    ("baseline_rmse", "123.23: guessing the mean is weak."),
                    ("linear_rmse", "11.68: features explain price well."),
                    ("coefficients", "[2.85, 17.97, -1.72] show direction and size."),
                    ("first_residual", "30.0: true value is about 30 higher."),
                    ("ridge_poly_rmse", "13.8: more complex is not always better."),
                    ("choose simpler", "Simple linear is safer in this run."),
                ],
                "footer": "Regression evidence starts with baseline, error size, coefficient meaning, and residuals.",
                "alt": "Linear regression lab result map: baseline RMSE, linear RMSE, coefficients, first residual, and polynomial Ridge comparison.",
            },
            "ja": {
                "title": "線形回帰の実験結果を読む",
                "subtitle": "RMSE、係数、残差、モデル選択を証拠としてつなぐ。",
                "items": [
                    ("baseline_rmse", "123.23：平均だけの予測は弱い。"),
                    ("linear_rmse", "11.68：特徴量が価格をよく説明する。"),
                    ("coefficients", "[2.85, 17.97, -1.72] で向きと大きさを見る。"),
                    ("first_residual", "30.0：実値が予測より約 30 高い。"),
                    ("ridge_poly_rmse", "13.8：複雑なら良いとは限らない。"),
                    ("choose simpler", "この実行では simple linear が安全。"),
                ],
                "footer": "回帰実験は baseline、誤差、係数の意味、残差から読む。",
                "alt": "線形回帰実験結果図：baseline RMSE、linear RMSE、係数、first residual、polynomial Ridge を比較する。",
            },
        },
    },
    {
        "slug": "ch05-logistic-threshold-lab-result-map",
        "pages": {
            "en": "docs/ch05-machine-learning/ch02-supervised/02-logistic-regression.md",
            "zh": "i18n/zh-Hans/docusaurus-plugin-content-docs/current/ch05-machine-learning/ch02-supervised/02-logistic-regression.md",
            "ja": "i18n/ja/docusaurus-plugin-content-docs/current/ch05-machine-learning/ch02-supervised/02-logistic-regression.md",
        },
        "scene": "A logistic regression lab result board. Show probabilities flowing into three threshold gates 0.3, 0.5, and 0.7. Use only these exact metric rows: threshold 0.3 accuracy=0.979 precision=0.968 recall=1.000 fp=3 fn=0; threshold 0.5 accuracy=0.986 precision=0.989 recall=0.989 fp=1 fn=1; threshold 0.7 accuracy=0.951 precision=0.988 recall=0.933 fp=1 fn=6. Show top coefficient chips: worst texture=-1.250, radius error=-1.070, worst symmetry=-0.957. Show regularization C comparison as a small bar: C=0.1 coef_norm=1.77 acc=0.979; C=1.0 coef_norm=3.52 acc=0.986; C=10.0 coef_norm=8.38 acc=0.972. Show a compact multiclass probability card with setosa 0.98, versicolor 0.62, and accuracy=0.921. Critical accuracy rule: do not draw a confusion matrix, do not draw TN or TP counts, and do not invent any sample counts. Show only fp and fn from the expected output. Do not draw dense full tables or extra metrics.",
        "chapter_context": "The image is inserted after the expected output of logistic_lab.py. Nearby text explains score -> probability -> threshold, false positives versus false negatives, top standardized coefficients, regularization strength C, and multi-class probability output.",
        "shared_layout": "Vertical 9:16. Top title and subtitle. Upper section shows score to probability to threshold gates. Middle section has three large threshold cards with accuracy, precision, recall, fp, and fn only; no confusion matrix and no TN/TP/sample-count cells. Lower left shows top coefficient evidence. Lower right shows C regularization strength and a compact multiclass probability card. Keep values and panel order identical across zh/en/ja.",
        "variants": {
            "zh": {
                "title": "逻辑回归输出怎么读",
                "subtitle": "同一个概率模型，换 threshold 就会改变 FP 和 FN。",
                "items": [
                    ("threshold=0.3", "recall=1.000，FN=0，但 FP=3。"),
                    ("threshold=0.5", "acc=0.986，FP=1，FN=1。"),
                    ("threshold=0.7", "precision 高，但 FN 增到 6。"),
                    ("top coefficients", "看标准化后最有影响的特征。"),
                    ("C strength", "C 越大，coef_norm 越大。"),
                    ("multiclass", "每一类都有概率，不只一个标签。"),
                ],
                "footer": "分类结果要按业务成本选择 threshold，而不是只看 accuracy。",
                "alt": "逻辑回归实验结果图：三个 threshold 的 precision、recall、FP/FN 对比，系数、C 正则化和多分类概率输出。",
            },
            "en": {
                "title": "Reading Logistic Regression Output",
                "subtitle": "One probability model changes FP and FN when the threshold changes.",
                "items": [
                    ("threshold=0.3", "recall=1.000, FN=0, but FP=3."),
                    ("threshold=0.5", "acc=0.986, FP=1, FN=1."),
                    ("threshold=0.7", "High precision, but FN rises to 6."),
                    ("top coefficients", "Read influential standardized features."),
                    ("C strength", "Larger C makes coef_norm larger."),
                    ("multiclass", "Each class has a probability, not just a label."),
                ],
                "footer": "Choose classification thresholds by business cost, not accuracy alone.",
                "alt": "Logistic regression lab result map: threshold precision, recall, FP/FN tradeoffs, coefficients, C regularization, and multiclass probabilities.",
            },
            "ja": {
                "title": "ロジスティック回帰の出力を読む",
                "subtitle": "同じ確率モデルでも threshold で FP と FN が変わる。",
                "items": [
                    ("threshold=0.3", "recall=1.000、FN=0、ただし FP=3。"),
                    ("threshold=0.5", "acc=0.986、FP=1、FN=1。"),
                    ("threshold=0.7", "precision は高いが FN は 6 へ増える。"),
                    ("top coefficients", "標準化後に効く特徴量を見る。"),
                    ("C strength", "C が大きいほど coef_norm が大きい。"),
                    ("multiclass", "ラベルだけでなく各クラス確率を見る。"),
                ],
                "footer": "分類 threshold は accuracy だけでなく業務コストで選ぶ。",
                "alt": "ロジスティック回帰実験結果図：threshold ごとの precision、recall、FP/FN、係数、C 正則化、多クラス確率を読む。",
            },
        },
    },
    {
        "slug": "ch05-svm-kernel-scaling-result-map",
        "pages": {
            "en": "docs/ch05-machine-learning/ch02-supervised/05-svm.md",
            "zh": "i18n/zh-Hans/docusaurus-plugin-content-docs/current/ch05-machine-learning/ch02-supervised/05-svm.md",
            "ja": "i18n/ja/docusaurus-plugin-content-docs/current/ch05-machine-learning/ch02-supervised/05-svm.md",
        },
        "scene": "An SVM lab result board for make_moons. Show a curved two-moons scatter. Compare a linear boundary with accuracy=0.920 and support_vectors=125 versus an RBF curved boundary with accuracy=0.950 and support_vectors=98. Show a scaling check with the y-axis stretched 100x: without_scaling=0.880 versus with_scaling=0.950, making the distance metric visibly distorted before StandardScaler. Show a compact C/gamma board with these exact highlights: C=0.1 gamma=1.0 accuracy=0.960 support_vectors=173; C=10.0 gamma=1.0 accuracy=0.920 support_vectors=57. Teaching point: RBF can fit curved boundaries, scaling is part of the model, and fewer support vectors does not automatically mean better test score. Do not invent extra rows or numbers.",
        "chapter_context": "The image is inserted after the expected output of svm_lab.py. Nearby text explains kernel comparison, why scaling is not optional, and how C and gamma control margin and local boundary complexity.",
        "shared_layout": "Vertical 9:16. Top title and subtitle. Upper section compares linear and RBF boundaries on the same moons. Middle section shows the scaling failure/fix with before/after axes. Lower section shows two C/gamma result cards and a support-vector count gauge. Keep values and visual order identical across zh/en/ja.",
        "variants": {
            "zh": {
                "title": "SVM 实验结果怎么看",
                "subtitle": "边界形状、特征尺度和 support vectors 要一起读。",
                "items": [
                    ("linear kernel", "直线边界，accuracy=0.920。"),
                    ("RBF kernel", "弯曲边界，accuracy=0.950。"),
                    ("without scaling", "尺度扭曲距离，分数降到 0.880。"),
                    ("with scaling", "StandardScaler 后回到 0.950。"),
                    ("C/gamma", "边界越局部，越可能过拟合。"),
                    ("support_vectors", "数量变少不等于测试更好。"),
                ],
                "footer": "SVM 是距离敏感模型，缩放和核函数都是实验的一部分。",
                "alt": "SVM 实验结果图：linear 与 RBF 核在 make_moons 上对比，展示特征缩放前后分数和 C/gamma 的影响。",
            },
            "en": {
                "title": "Reading SVM Lab Results",
                "subtitle": "Read boundary shape, feature scale, and support vectors together.",
                "items": [
                    ("linear kernel", "Straight boundary, accuracy=0.920."),
                    ("RBF kernel", "Curved boundary, accuracy=0.950."),
                    ("without scaling", "Scale distorts distance; score drops to 0.880."),
                    ("with scaling", "StandardScaler brings it back to 0.950."),
                    ("C/gamma", "More local boundaries can overfit."),
                    ("support_vectors", "Fewer vectors do not guarantee better test score."),
                ],
                "footer": "SVM is distance-sensitive; scaling and kernel choice are part of the experiment.",
                "alt": "SVM lab result map: compare linear and RBF kernels on make_moons, feature scaling effects, and C/gamma support-vector behavior.",
            },
            "ja": {
                "title": "SVM 実験結果の読み方",
                "subtitle": "境界の形、特徴量スケール、support vectors を一緒に読む。",
                "items": [
                    ("linear kernel", "直線境界で accuracy=0.920。"),
                    ("RBF kernel", "曲線境界で accuracy=0.950。"),
                    ("without scaling", "尺度で距離が歪み、score は 0.880。"),
                    ("with scaling", "StandardScaler 後は 0.950。"),
                    ("C/gamma", "局所的な境界ほど過学習しやすい。"),
                    ("support_vectors", "数が少ないほど良いとは限らない。"),
                ],
                "footer": "SVM は距離に敏感なので、scaling と kernel も実験対象。",
                "alt": "SVM 実験結果図：make_moons で linear と RBF kernel、特徴量 scaling、C/gamma と support vectors を比較する。",
            },
        },
    },
    {
        "slug": "ch05-metrics-threshold-regression-result-map",
        "pages": {
            "en": "docs/ch05-machine-learning/ch04-evaluation/01-metrics.md",
            "zh": "i18n/zh-Hans/docusaurus-plugin-content-docs/current/ch05-machine-learning/ch04-evaluation/01-metrics.md",
            "ja": "i18n/ja/docusaurus-plugin-content-docs/current/ch05-machine-learning/ch04-evaluation/01-metrics.md",
        },
        "scene": "An evaluation metrics lab result board. Do not draw terminal commands, command flags, sample counts, class distribution counts, confusion matrices, TN/TP counts, raw test size, row/index tables, or file paths. The page output does not provide those counts or command flags, so they must not appear. Use only the exact values listed here. Baseline card must explicitly show three numeric lines: accuracy=0.917, precision=0.000, recall=0.000, plus a visual warning that accuracy can look high while recall is zero. Threshold cards must contain exactly these rows and no other numbers: threshold 0.2 precision=0.462 recall=0.720 fp=21 fn=7; threshold 0.5 precision=0.833 recall=0.400 fp=2 fn=15; threshold 0.8 precision=1.000 recall=0.080 fp=0 fn=23. Show ranking metrics as two gauges only: roc_auc=0.889 and average_precision=0.660. Show regression comparison only: mean_baseline mae=65.5 rmse=74.9 r2=-0.014 versus linear mae=41.5 rmse=53.4 r2=0.485. Critical accuracy rule: do not add totals such as 1000, 917, 83, 300, 25, 275, or any other inferred counts.",
        "chapter_context": "The image is inserted after the expected output of metrics_lab.py. Nearby text explains the accuracy trap, threshold tuning, ROC AUC versus average precision, and regression metrics MAE/RMSE/R2.",
        "shared_layout": "Vertical 9:16. Top title and subtitle. Upper section has one baseline trap card with the three exact numeric metric lines and an icon showing missed positives, but no counts. Middle section has three threshold cards with only precision, recall, fp, fn, and a qualitative score-distribution sketch with no axis numbers except threshold labels 0.2, 0.5, 0.8. Lower section has ranking metric gauges and a regression baseline-versus-linear comparison. Keep values and panel order identical across zh/en/ja. No terminal commands, no command flags, no confusion matrix, no class distribution, no totals, no TN/TP/sample-count cells.",
        "variants": {
            "zh": {
                "title": "评估指标实验怎么读",
                "subtitle": "高 accuracy 可能什么正例都没找到，必须看错误类型。",
                "items": [
                    ("accuracy trap", "0.917 但 recall=0.000。"),
                    ("threshold=0.2", "recall 高，FP=21。"),
                    ("threshold=0.5", "precision 高，FN=15。"),
                    ("threshold=0.8", "FP=0，但 FN=23。"),
                    ("ranking metrics", "ROC AUC=0.889，AP=0.660。"),
                    ("regression", "linear 的 RMSE 从 74.9 降到 53.4。"),
                ],
                "footer": "指标不是装饰，它决定你愿意支付哪种错误成本。",
                "alt": "评估指标实验结果图：不平衡分类 accuracy trap、threshold 的 FP/FN 权衡、ROC AUC/AP 和回归指标对比。",
            },
            "en": {
                "title": "Reading Evaluation Metrics",
                "subtitle": "High accuracy can miss every positive; always inspect error types.",
                "items": [
                    ("accuracy trap", "0.917 but recall=0.000."),
                    ("threshold=0.2", "High recall, FP=21."),
                    ("threshold=0.5", "Higher precision, FN=15."),
                    ("threshold=0.8", "FP=0, but FN=23."),
                    ("ranking metrics", "ROC AUC=0.889, AP=0.660."),
                    ("regression", "Linear drops RMSE from 74.9 to 53.4."),
                ],
                "footer": "Metrics are not decoration; they decide which error cost you accept.",
                "alt": "Evaluation metrics lab result map: imbalanced accuracy trap, threshold FP/FN tradeoffs, ROC AUC/AP, and regression metric comparison.",
            },
            "ja": {
                "title": "評価指標の実験結果を読む",
                "subtitle": "高い accuracy でも positive を全部見逃すことがある。",
                "items": [
                    ("accuracy trap", "0.917 でも recall=0.000。"),
                    ("threshold=0.2", "recall は高いが FP=21。"),
                    ("threshold=0.5", "precision は高いが FN=15。"),
                    ("threshold=0.8", "FP=0 だが FN=23。"),
                    ("ranking metrics", "ROC AUC=0.889、AP=0.660。"),
                    ("regression", "linear で RMSE が 74.9 から 53.4 へ。"),
                ],
                "footer": "指標は飾りではなく、受け入れる error cost を決める。",
                "alt": "評価指標実験結果図：不均衡分類の accuracy trap、threshold ごとの FP/FN、ROC AUC/AP、回帰指標を比較する。",
            },
        },
    },
    {
        "slug": "ch05-hyperparameter-search-result-map",
        "pages": {
            "en": "docs/ch05-machine-learning/ch04-evaluation/04-hyperparameter-tuning.md",
            "zh": "i18n/zh-Hans/docusaurus-plugin-content-docs/current/ch05-machine-learning/ch04-evaluation/04-hyperparameter-tuning.md",
            "ja": "i18n/ja/docusaurus-plugin-content-docs/current/ch05-machine-learning/ch04-evaluation/04-hyperparameter-tuning.md",
        },
        "scene": "A hyperparameter tuning lab result board. Show train data split into CV folds and a locked final holdout. For GridSearchCV, show the winning parameter set exactly: max_depth=5, min_samples_leaf=3, n_estimators=80, best_cv_f1=0.968. Show final holdout metrics exactly: accuracy=0.956, recall=0.972, f1=0.966. For RandomizedSearchCV, show best params exactly: n_estimators=100, min_samples_leaf=1, max_features=log2, max_depth=8, best_cv_f1=0.972. Show top grid results only as three tied bars, each labeled score=0.968, and do not write any parameter details for those tied bars. Teaching point: tiny CV differences and ties should not be over-read. Do not draw the final holdout as used during search. Do not invent extra scores, parameter rows, trial numbers, scatter axes, or alternative best settings.",
        "chapter_context": "The image is inserted after the expected output of tuning_lab.py. Nearby text explains GridSearchCV, RandomizedSearchCV, search space, cross-validation score, final holdout, and over-tuning risk.",
        "shared_layout": "Vertical 9:16. Top title and subtitle. Upper section shows CV folds searching only inside training data and final holdout locked away. Middle section has GridSearchCV winner plus final holdout evaluation. Lower section has RandomizedSearchCV winner and top-three tied grid bars that show only three score=0.968 bars with no parameter text. Keep parameter values and panel order identical across zh/en/ja. Avoid dense tables and invented axes.",
        "variants": {
            "zh": {
                "title": "调参实验结果怎么看",
                "subtitle": "搜索在训练折里完成，final holdout 只在最后用一次。",
                "items": [
                    ("GridSearchCV", "best_cv_f1=0.968。"),
                    ("best params", "depth=5, leaf=3, trees=80。"),
                    ("final holdout", "accuracy=0.956, recall=0.972, f1=0.966。"),
                    ("RandomizedSearchCV", "8 次试验找到 best_cv_f1=0.972。"),
                    ("top ties", "多个组合同为 0.968，不要过度解读。"),
                    ("safe workflow", "测试集不参与搜索。"),
                ],
                "footer": "调参的目标是稳健选择，不是把测试集调到好看。",
                "alt": "超参数调参实验结果图：GridSearchCV、RandomizedSearchCV、CV 分数、final holdout 指标和 top grid ties。",
            },
            "en": {
                "title": "Reading Tuning Lab Results",
                "subtitle": "Search inside training folds; use the final holdout only once.",
                "items": [
                    ("GridSearchCV", "best_cv_f1=0.968."),
                    ("best params", "depth=5, leaf=3, trees=80."),
                    ("final holdout", "accuracy=0.956, recall=0.972, f1=0.966."),
                    ("RandomizedSearchCV", "8 trials find best_cv_f1=0.972."),
                    ("top ties", "Several combos tie at 0.968; do not over-read."),
                    ("safe workflow", "The test set is not part of search."),
                ],
                "footer": "Tuning is for robust selection, not making the test score look good.",
                "alt": "Hyperparameter tuning lab result map: GridSearchCV, RandomizedSearchCV, CV score, final holdout metrics, and top grid ties.",
            },
            "ja": {
                "title": "調整実験の結果を読む",
                "subtitle": "探索は training fold 内で行い、final holdout は最後に一度だけ使う。",
                "items": [
                    ("GridSearchCV", "best_cv_f1=0.968。"),
                    ("best params", "depth=5、leaf=3、trees=80。"),
                    ("final holdout", "accuracy=0.956, recall=0.972, f1=0.966。"),
                    ("RandomizedSearchCV", "8 試行で best_cv_f1=0.972。"),
                    ("top ties", "複数設定が 0.968、読みすぎない。"),
                    ("safe workflow", "test set は探索に使わない。"),
                ],
                "footer": "調整は test score を飾るためでなく、安定した選択のため。",
                "alt": "ハイパーパラメータ調整実験結果図：GridSearchCV、RandomizedSearchCV、CV score、final holdout 指標、top grid ties を読む。",
            },
        },
    },
    {
        "slug": "ch05-ensemble-comparison-result-map",
        "pages": {
            "en": "docs/ch05-machine-learning/ch02-supervised/04-ensemble-learning.md",
            "zh": "i18n/zh-Hans/docusaurus-plugin-content-docs/current/ch05-machine-learning/ch02-supervised/04-ensemble-learning.md",
            "ja": "i18n/ja/docusaurus-plugin-content-docs/current/ch05-machine-learning/ch02-supervised/04-ensemble-learning.md",
        },
        "scene": "An ensemble learning comparison lab result board based on ch05_ensemble_lab.py. The code block above the image already lists accuracy, so this teaching image must focus on the F1 ranking, leakage-safe workflow, and feature evidence. Do not write the word accuracy anywhere. Do not draw a dense table, terminal log, confusion matrix, ROC curve, sample counts, vote counts, axis ticks on the four model cards, or invented model names. Show exactly four test result cards as large readable F1 score chips: single_tree f1=0.956 as the baseline tree; random_forest f1=0.967 as many trees voting; gradient_boost f1=0.956 as sequential error correction that does not beat the baseline in this run; stacking_cv f1=0.989 as the winner. Mark stacking_cv as the winner only after showing a visible cv=5 / out-of-fold guard to prevent leakage. Show Random Forest top feature importance bars exactly: worst perimeter=0.146, worst area=0.140, worst concave points=0.109. Teaching point: compare every ensemble against the baseline, read top features as evidence, and trust stacking only when its meta-model uses out-of-fold predictions.",
        "chapter_context": "The image is inserted after the expected output of ch05_ensemble_lab.py. Nearby text explains single-tree baseline, Random Forest stability, Boosting validation control, leakage-safe stacking, and feature importance.",
        "shared_layout": "Vertical 9:16. Top title and subtitle. Upper section shows the four model F1 result cards on the same holdout test, with a visual winner marker on stacking_cv. Use plain F1 score chips for the four cards, not vertical bars or axis scales, and do not show accuracy values. Middle section shows the three ensemble mechanisms as concrete scenes: forest vote without invented vote totals, boosting error repair without invented sample counts, and stacking with cv=5 out-of-fold prediction tickets flowing to a meta-model. Lower section shows Random Forest top feature importance bars and a caution note that stacking must be leakage-safe. Keep values, model order, and visual rhythm identical across zh/en/ja. Use clear educational illustration, not SVG-style boxes or pure text.",
        "variants": {
            "zh": {
                "title": "集成学习实验结果怎么看",
                "subtitle": "先和单棵树 baseline 比，再看稳定性、泄漏风险和特征证据。",
                "items": [
                    ("single_tree", "baseline：f1=0.956。"),
                    ("random_forest", "投票更稳：f1=0.967。"),
                    ("gradient_boost", "本次未超过 baseline：f1=0.956。"),
                    ("stacking_cv", "cv=5 防泄漏后胜出：f1=0.989。"),
                    ("top features", "worst perimeter=0.146 是最强证据。"),
                    ("decision rule", "最高分还要通过泄漏检查。"),
                ],
                "footer": "Stacking 赢分数可以看，但先确认它用的是 out-of-fold 预测。",
                "alt": "集成学习实验结果图：单棵树、随机森林、梯度提升、CV Stacking 的 accuracy/F1 对比，以及随机森林前三个重要特征。",
            },
            "en": {
                "title": "Reading Ensemble Lab Results",
                "subtitle": "Compare against the single-tree baseline, then check stability, leakage, and feature evidence.",
                "items": [
                    ("single_tree", "baseline: f1=0.956."),
                    ("random_forest", "Voting is steadier: f1=0.967."),
                    ("gradient_boost", "This run ties the baseline: f1=0.956."),
                    ("stacking_cv", "cv=5 protects stacking: f1=0.989."),
                    ("top features", "worst perimeter=0.146 is strongest evidence."),
                    ("decision rule", "Highest score still needs a leakage check."),
                ],
                "footer": "Stacking can win, but only trust it when the meta-model sees out-of-fold predictions.",
                "alt": "Ensemble learning lab result map: compare single tree, Random Forest, Gradient Boosting, and CV Stacking F1 scores, plus top Random Forest features.",
            },
            "ja": {
                "title": "アンサンブル実験結果を読む",
                "subtitle": "単一木 baseline と比べ、安定性、リーク、特徴量証拠を確認する。",
                "items": [
                    ("single_tree", "baseline：f1=0.956。"),
                    ("random_forest", "投票で安定：f1=0.967。"),
                    ("gradient_boost", "この実行では baseline と同等：f1=0.956。"),
                    ("stacking_cv", "cv=5 でリークを防ぎ勝つ：f1=0.989。"),
                    ("top features", "worst perimeter=0.146 が最も強い証拠。"),
                    ("decision rule", "最高スコアでもリーク確認が必要。"),
                ],
                "footer": "Stacking は勝てるが、out-of-fold 予測を使う場合だけ信頼する。",
                "alt": "アンサンブル学習実験結果図：単一木、Random Forest、Gradient Boosting、CV Stacking の F1 と Random Forest の重要特徴量を比較する。",
            },
        },
    },
    {
        "slug": "ch05-anomaly-contamination-result-map",
        "pages": {
            "en": "docs/ch05-machine-learning/ch03-unsupervised/03-anomaly-detection.md",
            "zh": "i18n/zh-Hans/docusaurus-plugin-content-docs/current/ch05-machine-learning/ch03-unsupervised/03-anomaly-detection.md",
            "ja": "i18n/ja/docusaurus-plugin-content-docs/current/ch05-machine-learning/ch03-unsupervised/03-anomaly-detection.md",
        },
        "scene": "An anomaly detection lab result board based on anomaly_lab.py. The image must teach the alert trade-off, not simply copy a terminal. Do not draw a confusion matrix, TN/TP cells, sample totals, raw class distribution, precision/recall numeric tables, invented counts, invented score values, or extra contamination ticks. Show a synthetic two-cluster scatter with a contamination dial controlling how many points become alerts; the dial may show only these three numeric tick labels: 0.03, 0.06, 0.12. Show exactly three Isolation Forest result cards with only these exact values: contamination=0.03 flagged=12 f1=0.667 fp=0 fn=12; contamination=0.06 flagged=23 f1=0.809 fp=4 fn=5; contamination=0.12 flagged=46 f1=0.629 fp=24 fn=2. Show score inspection as exactly one suspicious sample card, not a ranked list and not a scale: score=-0.747, true_anomaly=True, with a caption that lower score means more abnormal. Show a small LOF comparison badge with exactly flagged=23 and f1=0.851, labeled local density. Teaching point: lower contamination misses more anomalies, higher contamination creates more false positives, and the best setting depends on review cost.",
        "chapter_context": "The image is inserted after the expected output of anomaly_lab.py. Nearby text explains contamination, alert thresholds, score inspection, Isolation Forest, LOF, false positives, false negatives, and review capacity.",
        "shared_layout": "Vertical 9:16. Top title and subtitle. Upper section shows normal clusters and anomalies with a contamination dial that has only 0.03, 0.06, and 0.12 as numeric ticks. Middle section has three alert-setting cards for 0.03, 0.06, and 0.12, with flagged, F1, FP, and FN only. Lower section shows exactly one score inspection card with score=-0.747 and true_anomaly=True, plus LOF local-density comparison. Keep all values, card order, and visual structure identical across zh/en/ja. Use concrete alert operations imagery, not SVG-style boxes or pure text.",
        "variants": {
            "zh": {
                "title": "异常检测实验结果怎么看",
                "subtitle": "contamination 像告警旋钮：报警越多，误报和漏报一起变化。",
                "items": [
                    ("contamination=0.03", "flagged=12，FP=0，FN=12，F1=0.667。"),
                    ("contamination=0.06", "flagged=23，FP=4，FN=5，F1=0.809。"),
                    ("contamination=0.12", "flagged=46，FP=24，FN=2，F1=0.629。"),
                    ("score queue", "score=-0.747，true_anomaly=True。"),
                    ("LOF", "局部密度：flagged=23，F1=0.851。"),
                    ("trade-off", "复核成本决定旋钮位置。"),
                ],
                "footer": "异常检测不是只求最高分，而是在漏报和误报之间设计告警流程。",
                "alt": "异常检测实验结果图：contamination 0.03、0.06、0.12 的 flagged、FP、FN、F1 对比，score inspection 和 LOF 对比。",
            },
            "en": {
                "title": "Reading Anomaly Lab Results",
                "subtitle": "contamination acts like an alert dial: more alerts change both misses and false alarms.",
                "items": [
                    ("contamination=0.03", "flagged=12, FP=0, FN=12, F1=0.667."),
                    ("contamination=0.06", "flagged=23, FP=4, FN=5, F1=0.809."),
                    ("contamination=0.12", "flagged=46, FP=24, FN=2, F1=0.629."),
                    ("score queue", "score=-0.747, true_anomaly=True."),
                    ("LOF", "Local density: flagged=23, F1=0.851."),
                    ("trade-off", "Review cost decides where to set the dial."),
                ],
                "footer": "Anomaly detection is not just the highest score; it is an alert workflow between misses and false alarms.",
                "alt": "Anomaly detection lab result map: compare contamination 0.03, 0.06, and 0.12 flagged, FP, FN, F1, score inspection, and LOF comparison.",
            },
            "ja": {
                "title": "異常検知の実験結果を読む",
                "subtitle": "contamination はアラートのつまみ。増やすほど見逃しと誤検知が変わる。",
                "items": [
                    ("contamination=0.03", "flagged=12、FP=0、FN=12、F1=0.667。"),
                    ("contamination=0.06", "flagged=23、FP=4、FN=5、F1=0.809。"),
                    ("contamination=0.12", "flagged=46、FP=24、FN=2、F1=0.629。"),
                    ("score queue", "score=-0.747、true_anomaly=True。"),
                    ("LOF", "局所密度：flagged=23、F1=0.851。"),
                    ("trade-off", "レビューコストでつまみ位置を決める。"),
                ],
                "footer": "異常検知は最高スコアだけでなく、見逃しと誤検知の間でアラート運用を設計する。",
                "alt": "異常検知実験結果図：contamination 0.03、0.06、0.12 の flagged、FP、FN、F1、score inspection、LOF を比較する。",
            },
        },
    },
    {
        "slug": "ch05-cross-validation-result-map",
        "pages": {
            "en": "docs/ch05-machine-learning/ch04-evaluation/02-cross-validation.md",
            "zh": "i18n/zh-Hans/docusaurus-plugin-content-docs/current/ch05-machine-learning/ch04-evaluation/02-cross-validation.md",
            "ja": "i18n/ja/docusaurus-plugin-content-docs/current/ch05-machine-learning/ch04-evaluation/02-cross-validation.md",
        },
        "scene": "A cross-validation lab result board based on cv_lab.py. The image must teach why one train-test split is a noisy snapshot and why 5-fold CV gives a mean plus variation. Do not draw a terminal log, confusion matrix, invented sample counts, or extra fold numbers. Show the single-split variance as five seed cards with exactly these accuracy values: seed=1 0.965, seed=2 0.972, seed=3 0.986, seed=4 0.972, seed=5 0.979. Show 5-fold CV as five fold cards with exactly these accuracy values: fold=1 0.974, fold=2 0.947, fold=3 0.965, fold=4 0.991, fold=5 0.991. Show the summary card exactly: accuracy=0.974+/-0.017, precision=0.968, recall=0.992, f1=0.979. Show a safe Pipeline lane where StandardScaler is fit inside each training fold, not before CV. Teaching point: a single split can look lucky or unlucky; CV reports average performance and fold-to-fold spread while keeping preprocessing leakage-safe.",
        "chapter_context": "The image is inserted after the expected output of cv_lab.py. Nearby text explains single split variance, StratifiedKFold, cross_validate, multi-metric summary, and leakage-safe Pipeline.",
        "shared_layout": "Vertical 9:16. Top title and subtitle. Upper section compares five single-split seed cards as different snapshots. Middle section shows five CV fold cards rotating through validation folds. Lower section has the mean-plus-variation summary and a leakage-safe Pipeline strip. Keep all values, order, and visual rhythm identical across zh/en/ja. Use concrete evaluation lab imagery, not SVG-style boxes or pure text.",
        "variants": {
            "zh": {
                "title": "交叉验证实验结果怎么看",
                "subtitle": "单次切分只是一个快照，K-Fold 看平均表现和波动。",
                "items": [
                    ("single split", "seed 分数从 0.965 到 0.986。"),
                    ("fold scores", "5 个 fold 分别检查不同验证集。"),
                    ("mean", "accuracy=0.974+/-0.017。"),
                    ("metrics", "precision=0.968，recall=0.992，f1=0.979。"),
                    ("pipeline", "StandardScaler 在每个训练 fold 内 fit。"),
                    ("decision", "看均值，也看波动。"),
                ],
                "footer": "交叉验证不是多跑几次求好看，而是估计模型在不同数据切片上的稳定性。",
                "alt": "交叉验证实验结果图：单次切分五个 seed accuracy、五折 CV accuracy、summary 平均和标准差，以及防泄漏 Pipeline。",
            },
            "en": {
                "title": "Reading Cross-Validation Results",
                "subtitle": "A single split is one snapshot; K-Fold reads mean performance and spread.",
                "items": [
                    ("single split", "Seed scores range from 0.965 to 0.986."),
                    ("fold scores", "Five folds test different validation slices."),
                    ("mean", "accuracy=0.974+/-0.017."),
                    ("metrics", "precision=0.968, recall=0.992, f1=0.979."),
                    ("pipeline", "StandardScaler fits inside each training fold."),
                    ("decision", "Read both the mean and the variation."),
                ],
                "footer": "Cross-validation is not rerunning until it looks good; it estimates stability across data slices.",
                "alt": "Cross-validation lab result map: single split seed accuracies, five-fold CV accuracies, mean and standard deviation summary, and leakage-safe Pipeline.",
            },
            "ja": {
                "title": "クロスバリデーション結果を読む",
                "subtitle": "単一分割は一つのスナップショット。K-Fold は平均とばらつきを見る。",
                "items": [
                    ("single split", "seed スコアは 0.965 から 0.986。"),
                    ("fold scores", "5 つの fold が別々の検証部分を見る。"),
                    ("mean", "accuracy=0.974+/-0.017。"),
                    ("metrics", "precision=0.968、recall=0.992、f1=0.979。"),
                    ("pipeline", "StandardScaler は各 training fold 内で fit。"),
                    ("decision", "平均だけでなく、ばらつきも読む。"),
                ],
                "footer": "クロスバリデーションは良い数字を探す再実行ではなく、データ切片ごとの安定性を見積もる方法。",
                "alt": "クロスバリデーション実験結果図：単一分割の seed accuracy、5-fold CV accuracy、平均と標準偏差、漏洩を防ぐ Pipeline を読む。",
            },
        },
    },
    {
        "slug": "ch05-bias-variance-result-map",
        "pages": {
            "en": "docs/ch05-machine-learning/ch04-evaluation/03-bias-variance.md",
            "zh": "i18n/zh-Hans/docusaurus-plugin-content-docs/current/ch05-machine-learning/ch04-evaluation/03-bias-variance.md",
            "ja": "i18n/ja/docusaurus-plugin-content-docs/current/ch05-machine-learning/ch04-evaluation/03-bias-variance.md",
        },
        "scene": "A bias-variance lab result board based on bias_variance_lab.py. The image must teach diagnosis from train score, test score, gap, and learning curve. Do not draw a dense terminal log, invented metrics, invented tree split thresholds, leaf sample counts, class probabilities, or tiny node text inside trees. Tree drawings should show only visual depth/branching and leaf count labels from the expected output, not split rules. Show complexity as four tree-depth cards with exactly these values: max_depth=1 train=0.923 test=0.923 gap=-0.001 leaves=2; max_depth=3 train=0.977 test=0.944 gap=0.032 leaves=7; max_depth=5 train=0.995 test=0.937 gap=0.058 leaves=15; max_depth=None train=1.000 test=0.923 gap=0.077 leaves=18. Visually label depth=1 as high bias or too simple, depth=3 as practical middle, and depth=None as high variance or overfit. Show the learning curve as two endpoint cards and one trend line: train_size=91 train=0.989 cv=0.847 gap=0.142; train_size=455 train=0.974 cv=0.919 gap=0.055. Teaching point: as complexity rises, training accuracy can improve while generalization gets worse; as more data is used, the gap shrinks.",
        "chapter_context": "The image is inserted after the expected output of bias_variance_lab.py. Nearby text explains underfitting, overfitting, train-test gap, model complexity, learning curves, and practical fixes.",
        "shared_layout": "Vertical 9:16. Top title and subtitle. Upper section shows four model-complexity tree cards from shallow to deep with train/test/gap/leaves. Tree sketches must not include split thresholds, sample counts, probabilities, or tiny node text. Middle section has a diagnosis path: too simple, practical middle, memorizes training details. Lower section shows the learning curve endpoints and a shrinking-gap arrow. Keep all values, order, and visual structure identical across zh/en/ja. Use concrete tree and curve imagery, not SVG-style boxes or pure text.",
        "variants": {
            "zh": {
                "title": "偏差方差实验结果怎么看",
                "subtitle": "看 train、test 和 gap，判断模型是太简单还是太会记训练集。",
                "items": [
                    ("depth=1", "train=test=0.923，gap≈0：太简单。"),
                    ("depth=3", "test=0.944，gap=0.032：中间更稳。"),
                    ("depth=5", "train=0.995，gap=0.058：开始过拟合。"),
                    ("depth=None", "train=1.000，test=0.923：记住训练细节。"),
                    ("learning curve", "数据从 91 到 455，gap 从 0.142 降到 0.055。"),
                    ("fix", "高方差优先简化模型或增加数据。"),
                ],
                "footer": "诊断不是背术语，而是看分数形状：训练、验证和 gap 一起读。",
                "alt": "偏差方差实验结果图：不同 max_depth 的 train/test/gap/leaves 对比，以及学习曲线 gap 随训练数据增加而缩小。",
            },
            "en": {
                "title": "Reading Bias-Variance Results",
                "subtitle": "Read train, test, and gap to tell whether the model is too simple or memorizing.",
                "items": [
                    ("depth=1", "train=test=0.923, gap≈0: too simple."),
                    ("depth=3", "test=0.944, gap=0.032: safer middle."),
                    ("depth=5", "train=0.995, gap=0.058: overfitting begins."),
                    ("depth=None", "train=1.000, test=0.923: memorizes training details."),
                    ("learning curve", "Data 91 to 455 shrinks gap from 0.142 to 0.055."),
                    ("fix", "For high variance, simplify the model or add data."),
                ],
                "footer": "Diagnosis is not memorizing terms; read the score shape: train, validation, and gap together.",
                "alt": "Bias-variance lab result map: compare train/test/gap/leaves across max_depth values and show learning-curve gap shrinking with more data.",
            },
            "ja": {
                "title": "バイアス・バリアンス結果を読む",
                "subtitle": "train、test、gap を見て、単純すぎるか訓練を覚えすぎか判断する。",
                "items": [
                    ("depth=1", "train=test=0.923、gap≈0：単純すぎる。"),
                    ("depth=3", "test=0.944、gap=0.032：中間が安定。"),
                    ("depth=5", "train=0.995、gap=0.058：過学習が始まる。"),
                    ("depth=None", "train=1.000、test=0.923：訓練詳細を記憶。"),
                    ("learning curve", "データ 91 から 455 で gap は 0.142 から 0.055。"),
                    ("fix", "high variance ならモデルを簡単にするかデータを増やす。"),
                ],
                "footer": "診断は用語暗記ではなく、train、validation、gap の形を一緒に読むこと。",
                "alt": "バイアス・バリアンス実験結果図：max_depth ごとの train/test/gap/leaves と、データ増加で学習曲線 gap が縮む様子を読む。",
            },
        },
    },
]

for experiment_group in EXPERIMENT_RESULT_GROUPS:
    register_svg_replacement_group(
        slug=experiment_group["slug"],
        pages=experiment_group["pages"],
        scene=experiment_group["scene"],
        chapter_context=experiment_group["chapter_context"],
        shared_layout=experiment_group["shared_layout"],
        variants=experiment_group["variants"],
        callouts=[],
    )

existing_filenames = {str(job.get("filename")) for job in IMAGE_JOBS}
IMAGE_JOBS.extend(job for job in P0_REMAKE_IMAGE_JOBS if job["filename"] not in existing_filenames)

IMAGE_JOB_PROMPT_OVERRIDES = {
    "ch08-async-concurrency-semaphore-timeout-map.png": """
竖版 9:16 中文教学漫画，像后端请求调度室，不要白底流程框。
标题写清楚：“并发请求如何不把服务打爆”。
画面左侧对比小分镜是“无限并发崩溃”：请求人群冲向 API 门口，仪表变红；右侧主画面是受控并发：多个请求在队列排队，通过 Semaphore 限流闸门，少量请求进入 API 调用区，慢请求被 timeout 取消，结果和错误汇总到报告板。
需要编号和短说明：
① 请求队列：先排队，不一拥而上
② Semaphore：限制同时进行数量
③ API 调用：受控访问上游
④ Timeout：慢任务及时取消
⑤ Error summary：集中记录失败原因
⑥ Stable result：吞吐稳定、错误可解释
底部结论：“并发不是越多越好，受控并发才能稳定。”
文字清楚可读；保留 Semaphore、API、Timeout。不要密集小字、乱码、水印、真实品牌 logo。
""".strip(),
    "ch08-llm-api-robust-client-loop-map.png": """
竖版 9:16 中文教学插画，像稳健 API 客户端作战台，不要白底圆角框。
标题写清楚：“稳健 LLM API 客户端”。
画面中学习者在控制台处理一次模型请求：请求卡片进入发送台；失败案例区显示网络错误、限流、坏 JSON；系统用 retry/backoff 等待重试；成功后进入 parse/normalize 区；旁边记录 usage、latency、request_id；最后统一返回结果或可解释错误。
需要编号和短说明：
① Request：构造请求
② Retry：只重试可恢复错误
③ Backoff：失败后拉开间隔
④ Parse：解析 JSON / text
⑤ Usage log：记录 tokens 与费用
⑥ Error return：失败也要可解释
底部结论：“生产客户端要能成功，也要能优雅失败。”
文字清楚可读；保留 LLM API、retry、backoff、usage、request_id。不要密集小字、乱码、水印、真实品牌 logo。
""".strip(),
    "ch08-unified-api-provider-gateway-map.png": """
竖版 9:16 中文教学对比插画，不要白底流程框。
标题写清楚：“为什么需要统一 Provider 网关”。
画面左侧是混乱小分镜：业务代码直接接多家 provider，接口、错误、usage 格式各不相同，线缆缠在一起。右侧是统一网关工作台：业务请求先进入 gateway，再由 routing 选择 provider，经 adapter 转成对应格式；失败时 fallback；最后输出 normalized error 和 usage log。
需要编号和短说明：
① 混乱直连：每家接口都不同
② Gateway：业务只接一个入口
③ Routing：按模型、成本、可用性选择
④ Adapter：适配不同 provider
⑤ Fallback：失败时换路
⑥ Usage log：统一记录成本与延迟
底部结论：“网关把 provider 差异挡在业务代码外面。”
文字清楚可读；保留 provider、gateway、routing、adapter、fallback、usage。不要密集小字、乱码、水印、真实品牌 logo。
""".strip(),
    "ch11-seq2seq-chapter-flow.png": """
竖版 9:16 中文教学插画，主题标题必须清楚写在画面上方：“Seq2Seq 学习路线”。
画面像课堂分镜/工作台，不要白底流程框：一位学习者看着屏幕，左侧是一串输入词卡，经过 Encoder 工作台压成上下文胶囊，右侧 Decoder 像打字机一样逐步吐出输出词卡；旁边有 Attention 聚光灯照回输入词，底部有翻译与摘要两个小成果样本。
必须有少量大字短标签，中文自然可读：① 输入序列、② Encoder、③ 上下文、④ Decoder、⑤ Attention、⑥ 输出序列。
加一句短说明：“先读完整输入，再一步步生成输出。”
文字要大、少、清晰；允许少量英文术语 Encoder、Decoder、Attention。不要密密麻麻的小字，不要乱码，不要真实品牌 logo，不要白底圆角框流程图。
""".strip(),
    "ch11-seq2seq-encoder-decoder-bottleneck-map.png": """
竖版 9:16 中文教学插画，主题标题：“Encoder-Decoder 信息瓶颈”。
画面是学习者在实验台上观察翻译模型：左上是输入句子词卡排队进入 Encoder 机器，中间只有一个狭窄的“上下文瓶颈”玻璃瓶，里面塞满压缩信息，右侧 Decoder 逐步生成目标词卡。长句子有部分信息从瓶口溢出，用醒目的警示标记表示丢失风险；远处有 Attention 聚光灯作为下一节的解决线索。
必须有清楚短标签：① 输入、② Encoder、③ 瓶颈、④ Decoder、⑤ 逐步生成、⑥ Attention 缓解。
加一句短说明：“只靠一个向量，长句信息容易挤丢。”
文字少而大，中文自然；保留 Encoder、Decoder、Attention。不要白底流程图、不要圆角框堆叠、不要乱码小字。
""".strip(),
    "seq2seq-attention-alignment.png": """
竖版 9:16 中文教学插画，主题标题：“Attention 如何对齐词语”。
画面像翻译课堂的屏幕分镜：上方是输入词卡队列，下方是 Decoder 正在生成目标词；每生成一步，一个可见聚光灯照向输入中的相关词卡，形成柔和热力带。学习者用笔圈出“当前输出看哪里”。右侧有小型对齐热力板，但不要做成表格幻灯片。
必须有清楚短标签：① 输入词、② 当前输出、③ 注意力权重、④ 对齐位置、⑤ 动态查看。
加一句短说明：“每一步生成时，模型重新看最相关的输入位置。”
文字少、大、可读；术语 Attention 可保留英文。不要白底框图、不要密集小字、不要乱码。
""".strip(),
    "ch11-machine-translation-error-analysis-map.png": """
竖版 9:16 中文教学插画，主题标题：“机器翻译错例复盘”。
画面是翻译项目评审桌：左侧是源句与参考译文纸张，右侧是模型译文被老师和学习者用彩笔圈出问题；下方把错误样本放入几个实物托盘：漏译、错译、语序、术语、风格。旁边有一个简洁评分仪表，表示机器指标和人工复核要一起看。
必须有清楚短标签：① 源句、② 参考译文、③ 模型译文、④ 漏译、⑤ 错译、⑥ 术语不一致、⑦ 人工复核。
加一句短说明：“不要只看顺眼样例，要按错误类型改系统。”
文字少、大、自然；允许 BLEU / chrF 小标签。不要白底流程框、不要乱码、不要真实品牌 logo。
""".strip(),
    "ch11-ctc-deep-speech-asr-map.png": """
竖版 9:16 中文教学插画，主题标题：“CTC：长音频到短文本”。
画面像语音识别实验台：上方是一条长音频波形和很多音频帧，小机器人/模型逐帧给出字符候选；中间出现带 blank 的长路径，像一串可折叠纸带；下方学习者把重复字符和 blank 折叠掉，得到短文本结果。突出“帧很多，字更少”的直觉。
必须有清楚短标签：① 音频帧、② 声学模型、③ CTC 路径、④ blank、⑤ 折叠、⑥ 最终文本。
加一句短说明：“先允许长路径，再折叠成真正 transcript。”
文字少、大、可读；保留 CTC、blank、transcript。不要白底流程图、不要圆角框模板、不要乱码小字。
""".strip(),
}

for job in IMAGE_JOBS:
    override_prompt = IMAGE_JOB_PROMPT_OVERRIDES.get(str(job.get("filename")))
    if override_prompt and not job.get("overlay"):
        job["prompt"] = override_prompt


def convert_remaining_overlays_to_generated_images() -> None:
    for job in IMAGE_JOBS:
        # SVG replacement jobs intentionally generate a no-text background once,
        # then apply exact zh/en/ja labels locally. Do not convert them into
        # fully generated text images.
        if job.get("background_key"):
            continue
        overlay = job.get("overlay")
        if not overlay:
            continue
        items = list(overlay.get("items", []))
        if not items:
            job["overlay"] = None
            continue

        panel_lines = []
        for index, item in enumerate(items[:6], start=1):
            label = str(item.get("label", "")).strip()
            detail = str(item.get("detail", "")).strip()
            panel_lines.append(
                f"Panel {index}: visible caption exactly \"{index} {label}\" and short line exactly \"{detail}\". "
                f"Show this concept as a concrete action or before/after scene directly inside the panel."
            )

        title = str(overlay.get("title", job.get("title", ""))).strip()
        subtitle = str(overlay.get("subtitle", "")).strip()
        footer = str(overlay.get("footer", "")).strip()
        job["prompt"] = f"""
Create one complete vertical 9:16 teaching image as a finished bitmap for an AI full-stack course. Do not rely on post-processing, captions, labels, overlays, SVG, markdown, or external text.
The image itself must contain all teaching text, drawn naturally as part of the illustration.
Style: high-quality classroom comic or practical workflow illustration, not a whiteboard flowchart, not UI cards, not pasted text boxes, not an SVG-style diagram.
Visible title at the top must be exactly: {title}
Visible subtitle under the title must be exactly: {subtitle}

Choose the clearest teaching composition for the topic: comic panels for lifecycles, side-by-side comparison for trade-offs, layered architecture for systems, annotated workbench for data structures, or incident board for debugging. Do not force every topic into six panels. Each text label must sit inside or beside the visual action it explains:
{chr(10).join(panel_lines)}

Visible footer at the bottom must be exactly: {footer}
Make the relationship between visual action and teaching text obvious. Text must be readable, short, and integrated with the scene. Avoid tiny text, gibberish, watermarks, real brand logos, blank white background, and generic rounded flowchart boxes.
""".strip()
        job["overlay"] = None


convert_remaining_overlays_to_generated_images()


HOMEPAGE_HISTORY_COMIC_FILENAMES = {
    f"homepage-ai-history-comic-{index:02d}-{slug}.png"
    for index, slug in [
        (1, "turing"),
        (2, "dartmouth"),
        (3, "perceptron"),
        (4, "expert-systems"),
        (5, "backprop"),
        (6, "lenet"),
        (7, "statistical-ml"),
        (8, "imagenet-alexnet"),
        (9, "resnet"),
        (10, "rnn-lstm"),
        (11, "attention"),
        (12, "transformer"),
        (13, "bert-gpt"),
        (14, "rlhf-chatgpt"),
        (15, "rag-agent"),
    ]
}

for job in IMAGE_JOBS:
    if not job.get("allow_landscape"):
        job["size"] = DEFAULT_COURSE_IMAGE_SIZE
        job["quality"] = DEFAULT_COURSE_IMAGE_QUALITY


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate course images and save them under static/img/course.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned image jobs without calling the API.")
    parser.add_argument("--only", nargs="*", help="Generate only selected filenames, such as ai-fullstack-hero.png.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for generated images.")
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR), help="Directory for generated image reports.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Image model name.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenAI-compatible base URL.")
    parser.add_argument("--request-timeout", type=int, default=DEFAULT_REQUEST_TIMEOUT, help="HTTP request timeout in seconds.")
    parser.add_argument("--retries", type=int, default=DEFAULT_IMAGE_RETRIES, help="Retry count for transient image API errors.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing valid PNG files.")
    parser.add_argument("--continue-on-error", action="store_true", help="Keep generating remaining images and write an error report.")
    parser.add_argument("--ensure-placeholders", action="store_true", help="Create local preview PNG files for all planned images.")
    parser.add_argument("--http-fallback", action="store_true", help="Use the built-in HTTP client instead of the openai Python package.")
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


def write_manifest(report_dir: Path, jobs: list[dict[str, Any]]) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    manifest = report_dir / "manifest.md"
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


def job_dimensions(job: dict[str, Any]) -> tuple[int, int]:
    size = str(job.get("size", "1536x1024"))
    if size == "default":
        return 1536, 1024
    try:
        width_text, height_text = size.lower().split("x", 1)
        return int(width_text), int(height_text)
    except (AttributeError, TypeError, ValueError):
        return 1536, 1024


def write_placeholder(output_dir: Path, job: dict[str, Any], overwrite: bool = False) -> None:
    output_path = output_dir / job["filename"]
    if output_path.exists() and not overwrite:
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

    width, height = job_dimensions(job)
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

    margin_x = max(56, width // 14)
    margin_y = max(72, height // 14)
    card = (margin_x, margin_y, width - margin_x, height - margin_y)
    draw.rounded_rectangle(card, radius=max(36, width // 32), outline=accent, width=6)

    panel_gap = max(24, width // 28)
    panel_w = (card[2] - card[0] - panel_gap * 3) // 2
    panel_h = max(150, (card[3] - card[1] - panel_gap * 5) // 4)
    top_panel_y = card[1] + panel_gap
    left_panel = (card[0] + panel_gap, top_panel_y, card[0] + panel_gap + panel_w, top_panel_y + panel_h)
    right_panel = (left_panel[2] + panel_gap, top_panel_y, left_panel[2] + panel_gap + panel_w, top_panel_y + panel_h)
    draw.rounded_rectangle(left_panel, radius=28, fill="#0f172a", outline="#334155", width=3)
    draw.rounded_rectangle(right_panel, radius=28, fill="#0f172a", outline="#334155", width=3)

    text_box_y = card[1] + panel_gap * 2 + panel_h
    text_box = (
        card[0] + panel_gap,
        text_box_y,
        card[2] - panel_gap,
        min(card[3] - panel_gap * 2, text_box_y + max(190, height // 6)),
    )
    draw.rounded_rectangle(text_box, radius=36, fill="#020617", outline=warm, width=5)

    icon_y = (left_panel[1] + left_panel[3]) // 2
    icon_r = max(28, min(panel_w, panel_h) // 5)
    icon_positions = [
        (left_panel[0] + panel_w // 4, icon_y, icon_r, accent),
        (left_panel[0] + panel_w // 2, icon_y, icon_r, warm),
        (left_panel[0] + panel_w * 3 // 4, icon_y, icon_r, "#60a5fa"),
    ]
    for x, y, r, color in icon_positions:
        draw.ellipse([(x - r, y - r), (x + r, y + r)], fill=color)
    arrow_y = icon_y
    draw.line([(left_panel[2] + panel_gap // 4, arrow_y), (right_panel[0] - panel_gap // 4, arrow_y)], fill="#e5e7eb", width=8)
    draw.polygon(
        [
            (right_panel[0] - panel_gap // 4, arrow_y),
            (right_panel[0] - panel_gap // 4 - 30, arrow_y - 20),
            (right_panel[0] - panel_gap // 4 - 30, arrow_y + 20),
        ],
        fill="#e5e7eb",
    )

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

    title_font = load_font(max(34, width // 23))
    body_font = load_font(max(24, width // 38))
    label_font = load_font(max(20, width // 48))

    draw.text((text_box[0] + 28, text_box[1] + 28), job["title"], font=title_font, fill="#f8fafc")

    alt_lines = textwrap.wrap(job["alt"], width=max(24, width // 34))
    y = text_box[1] + 92
    for line in alt_lines[:3]:
        draw.text((text_box[0] + 32, y), line, font=body_font, fill="#cbd5e1")
        y += max(34, width // 28)

    preview_label = str(job.get("preview_label", "Preview Asset"))
    draw.text((card[2] - max(260, width // 4), card[3] - max(48, height // 36)), preview_label, font=label_font, fill="#94a3b8")
    image.save(output_path, format="PNG")
    set_user_readable_permissions(output_path)


def ensure_placeholders(output_dir: Path) -> None:
    ensure_output_dir(output_dir)
    for job in IMAGE_JOBS:
        write_placeholder(output_dir, job)


def ensure_selected_placeholders(output_dir: Path, jobs: list[dict[str, Any]], overwrite: bool = False) -> None:
    ensure_output_dir(output_dir)
    for job in jobs:
        write_placeholder(output_dir, job, overwrite=overwrite)


def set_user_readable_permissions(output_path: Path) -> None:
    output_path.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)


def load_overlay_font(size: int, *, bold: bool = False):
    from PIL import ImageFont

    font_paths = [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    ]
    for font_path in font_paths:
        if Path(font_path).exists():
            try:
                return ImageFont.truetype(font_path, size=size, index=1 if bold else 0)
            except (OSError, TypeError):
                try:
                    return ImageFont.truetype(font_path, size=size)
                except OSError:
                    continue
    return ImageFont.load_default()


def wrap_overlay_text(draw: Any, text: str, font: Any, max_width: int) -> list[str]:
    if not text:
        return []

    if " " in text:
        lines: list[str] = []
        current = ""
        for token in text.split(" "):
            candidate = token if not current else f"{current} {token}"
            bbox = draw.textbbox((0, 0), candidate, font=font)
            if bbox[2] - bbox[0] <= max_width:
                current = candidate
                continue
            if current:
                lines.append(current)
            token_bbox = draw.textbbox((0, 0), token, font=font)
            if token_bbox[2] - token_bbox[0] <= max_width:
                current = token
            else:
                split_lines = wrap_overlay_text(draw, token.replace(" ", ""), font, max_width)
                lines.extend(split_lines[:-1])
                current = split_lines[-1] if split_lines else ""
        if current:
            lines.append(current)
        return lines

    lines: list[str] = []
    current = ""
    for char in text:
        candidate = f"{current}{char}"
        bbox = draw.textbbox((0, 0), candidate, font=font)
        if bbox[2] - bbox[0] <= max_width or not current:
            current = candidate
            continue
        lines.append(current)
        current = char
    if current:
        lines.append(current)
    return lines


def apply_callout_overlay(image: Any, job: dict[str, Any], overlay: dict[str, Any]) -> Any:
    from PIL import ImageDraw

    width, height = image.size
    draw = ImageDraw.Draw(image, "RGBA")

    title = str(overlay.get("title", job.get("title", "")))
    subtitle = str(overlay.get("subtitle", ""))
    items = list(overlay.get("items", []))
    callouts = list(overlay.get("callouts", []))
    footer = str(overlay.get("footer", ""))

    margin = max(36, width // 22)
    title_font = load_overlay_font(max(40, width // 18), bold=True)
    subtitle_font = load_overlay_font(max(22, width // 40))
    label_font = load_overlay_font(max(20, width // 47), bold=True)
    detail_font = load_overlay_font(max(17, width // 57))
    footer_font = load_overlay_font(max(24, width // 39), bold=True)

    top_height = max(150, int(height * 0.125))
    footer_height = max(74, int(height * 0.055)) if footer else 0

    draw.rectangle((0, 0, width, top_height), fill=(3, 7, 18, 224))
    draw.rectangle((0, top_height - 4, width, top_height + 2), fill=(56, 189, 248, 220))

    y = 24
    for line in wrap_overlay_text(draw, title, title_font, width - margin * 2)[:2]:
        draw.text((margin, y), line, font=title_font, fill=(248, 250, 252, 255))
        y += title_font.size + 6
    for line in wrap_overlay_text(draw, subtitle, subtitle_font, width - margin * 2)[:2]:
        draw.text((margin, y + 2), line, font=subtitle_font, fill=(203, 213, 225, 255))
        y += subtitle_font.size + 4

    palette = [
        (96, 165, 250, 255),
        (52, 211, 153, 255),
        (251, 191, 36, 255),
        (196, 181, 253, 255),
        (248, 113, 113, 255),
        (45, 212, 191, 255),
        (244, 114, 182, 255),
        (129, 140, 248, 255),
    ]

    for index, item in enumerate(items):
        if index >= len(callouts):
            break
        callout = callouts[index]
        box = callout.get("box", (0.05, 0.2 + index * 0.1, 0.35, 0.09))
        target = callout.get("target", (0.5, 0.5))
        x = int(float(box[0]) * width)
        y = int(float(box[1]) * height)
        box_w = int(float(box[2]) * width)
        box_h = int(float(box[3]) * height)
        target_x = int(float(target[0]) * width)
        target_y = int(float(target[1]) * height)

        label = str(item.get("label", ""))
        detail = str(item.get("detail", ""))
        color = palette[index % len(palette)]
        radius = max(18, width // 58)
        badge_size = max(30, width // 33)

        center_x = x + box_w // 2
        center_y = y + box_h // 2
        start_x = x if target_x < center_x else x + box_w
        start_y = min(max(target_y, y + 10), y + box_h - 10)
        draw.line((start_x, start_y, target_x, target_y), fill=(*color[:3], 230), width=max(4, width // 180))
        draw.ellipse(
            (
                target_x - badge_size // 3,
                target_y - badge_size // 3,
                target_x + badge_size // 3,
                target_y + badge_size // 3,
            ),
            fill=(*color[:3], 238),
            outline=(255, 255, 255, 230),
            width=2,
        )

        draw.rounded_rectangle(
            (x, y, x + box_w, y + box_h),
            radius=radius,
            fill=(3, 7, 18, 218),
            outline=(*color[:3], 246),
            width=3,
        )
        draw.ellipse((x + 12, y + 13, x + 12 + badge_size, y + 13 + badge_size), fill=color)
        number_text = str(index + 1)
        number_bbox = draw.textbbox((0, 0), number_text, font=detail_font)
        draw.text(
            (
                x + 12 + badge_size / 2 - (number_bbox[2] - number_bbox[0]) / 2,
                y + 13 + badge_size / 2 - (number_bbox[3] - number_bbox[1]) / 2 - 1,
            ),
            number_text,
            font=detail_font,
            fill=(3, 7, 18, 255),
        )

        text_x = x + 24 + badge_size
        max_text_width = box_w - badge_size - 38
        label_lines = wrap_overlay_text(draw, label, label_font, max_text_width)
        detail_lines = wrap_overlay_text(draw, detail, detail_font, max_text_width)
        text_y = y + 12
        for line in label_lines[:2]:
            draw.text((text_x, text_y), line, font=label_font, fill=(248, 250, 252, 255))
            text_y += label_font.size + 2
        for line in detail_lines[:2]:
            draw.text((text_x, text_y + 2), line, font=detail_font, fill=(203, 213, 225, 255))
            text_y += detail_font.size + 2

    if footer:
        footer_top = height - footer_height
        draw.rectangle((0, footer_top, width, height), fill=(3, 7, 18, 225))
        draw.rectangle((margin, footer_top + 12, width - margin, footer_top + 16), fill=(251, 191, 36, 235))
        footer_y = footer_top + 25
        for line in wrap_overlay_text(draw, footer, footer_font, width - margin * 2)[:2]:
            draw.text((margin, footer_y), line, font=footer_font, fill=(254, 249, 195, 255))
            footer_y += footer_font.size + 4

    return image


def apply_comic_panel_overlay(image: Any, job: dict[str, Any], overlay: dict[str, Any]) -> Any:
    from PIL import ImageDraw

    width, height = image.size
    draw = ImageDraw.Draw(image, "RGBA")

    title = str(overlay.get("title", job.get("title", "")))
    subtitle = str(overlay.get("subtitle", ""))
    items = list(overlay.get("items", []))
    footer = str(overlay.get("footer", ""))

    margin = max(32, width // 24)
    gap = max(14, width // 70)
    title_font = load_overlay_font(max(38, width // 19), bold=True)
    subtitle_font = load_overlay_font(max(21, width // 43))
    label_font = load_overlay_font(max(18, width // 54), bold=True)
    detail_font = load_overlay_font(max(15, width // 66))
    footer_font = load_overlay_font(max(22, width // 45), bold=True)

    top_height = max(132, int(height * 0.12))
    footer_height = max(68, int(height * 0.052)) if footer else 0
    panel_top = top_height + gap
    panel_bottom = height - footer_height - gap
    panel_area_height = panel_bottom - panel_top
    panel_w = (width - margin * 2 - gap) // 2
    panel_h = (panel_area_height - gap * 2) // 3

    draw.rectangle((0, 0, width, top_height), fill=(3, 7, 18, 226))
    draw.rectangle((0, top_height - 4, width, top_height + 2), fill=(56, 189, 248, 220))
    y = 18
    for line in wrap_overlay_text(draw, title, title_font, width - margin * 2)[:2]:
        draw.text((margin, y), line, font=title_font, fill=(248, 250, 252, 255))
        y += title_font.size + 4
    for line in wrap_overlay_text(draw, subtitle, subtitle_font, width - margin * 2)[:2]:
        draw.text((margin, y + 2), line, font=subtitle_font, fill=(203, 213, 225, 255))
        y += subtitle_font.size + 4

    palette = [
        (96, 165, 250, 255),
        (52, 211, 153, 255),
        (251, 191, 36, 255),
        (196, 181, 253, 255),
        (248, 113, 113, 255),
        (45, 212, 191, 255),
    ]

    for index, item in enumerate(items[:6]):
        row = index // 2
        col = index % 2
        x = margin + col * (panel_w + gap)
        y = panel_top + row * (panel_h + gap)
        rect = (x, y, x + panel_w, y + panel_h)
        color = palette[index % len(palette)]
        draw.rectangle(rect, outline=(248, 250, 252, 230), width=max(3, width // 260))

        label = str(item.get("label", ""))
        detail = str(item.get("detail", ""))
        caption_h = max(72, int(panel_h * 0.24))
        caption_top = y + panel_h - caption_h
        draw.rectangle((x, caption_top, x + panel_w, y + panel_h), fill=(3, 7, 18, 218))
        draw.rectangle((x, caption_top, x + panel_w, caption_top + 4), fill=(*color[:3], 235))

        badge = max(25, width // 38)
        badge_x = x + 13
        badge_y = caption_top + 14
        draw.ellipse((badge_x, badge_y, badge_x + badge, badge_y + badge), fill=color)
        number_text = str(index + 1)
        number_bbox = draw.textbbox((0, 0), number_text, font=detail_font)
        draw.text(
            (
                badge_x + badge / 2 - (number_bbox[2] - number_bbox[0]) / 2,
                badge_y + badge / 2 - (number_bbox[3] - number_bbox[1]) / 2 - 1,
            ),
            number_text,
            font=detail_font,
            fill=(3, 7, 18, 255),
        )

        text_x = badge_x + badge + 10
        max_text_width = panel_w - (text_x - x) - 12
        text_y = caption_top + 9
        for line in wrap_overlay_text(draw, label, label_font, max_text_width)[:2]:
            draw.text((text_x, text_y), line, font=label_font, fill=(248, 250, 252, 255))
            text_y += label_font.size + 1
        for line in wrap_overlay_text(draw, detail, detail_font, max_text_width)[:2]:
            draw.text((text_x, text_y + 1), line, font=detail_font, fill=(203, 213, 225, 255))
            text_y += detail_font.size + 1

    if footer:
        footer_top = height - footer_height
        draw.rectangle((0, footer_top, width, height), fill=(3, 7, 18, 226))
        draw.rectangle((margin, footer_top + 10, width - margin, footer_top + 14), fill=(251, 191, 36, 235))
        footer_y = footer_top + 23
        for line in wrap_overlay_text(draw, footer, footer_font, width - margin * 2)[:2]:
            draw.text((margin, footer_y), line, font=footer_font, fill=(254, 249, 195, 255))
            footer_y += footer_font.size + 4

    return image


def apply_text_overlay(output_path: Path, job: dict[str, Any]) -> None:
    overlay = job.get("overlay")
    if not overlay:
        return

    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print(f"Pillow is not installed; skipping text overlay for {job['filename']}", flush=True)
        return

    image = Image.open(output_path).convert("RGBA")
    if overlay.get("style") == "comic_panels":
        image = apply_comic_panel_overlay(image, job, overlay)
        image.convert("RGB").save(output_path, format="PNG")
        return
    if overlay.get("style") == "callouts":
        image = apply_callout_overlay(image, job, overlay)
        image.convert("RGB").save(output_path, format="PNG")
        return

    width, height = image.size
    draw = ImageDraw.Draw(image, "RGBA")

    title = str(overlay.get("title", job.get("title", "")))
    subtitle = str(overlay.get("subtitle", ""))
    items = list(overlay.get("items", []))
    footer = str(overlay.get("footer", ""))

    margin = max(44, width // 18)
    title_font = load_overlay_font(max(42, width // 17), bold=True)
    subtitle_font = load_overlay_font(max(25, width // 34))
    label_font = load_overlay_font(max(25, width // 37), bold=True)
    detail_font = load_overlay_font(max(22, width // 45))
    footer_font = load_overlay_font(max(27, width // 34), bold=True)

    top_height = max(190, int(height * 0.15))
    bottom_top = int(height * 0.66)
    footer_height = max(88, int(height * 0.06)) if footer else 0

    draw.rectangle((0, 0, width, top_height), fill=(4, 10, 28, 226))
    draw.rectangle((0, bottom_top, width, height), fill=(4, 10, 28, 232))
    draw.rectangle((0, top_height - 5, width, top_height + 2), fill=(56, 189, 248, 205))
    draw.rectangle((0, bottom_top - 5, width, bottom_top + 2), fill=(52, 211, 153, 205))

    title_lines = wrap_overlay_text(draw, title, title_font, width - margin * 2)
    y = 34
    for line in title_lines[:2]:
        draw.text((margin, y), line, font=title_font, fill=(248, 250, 252, 255))
        y += title_font.size + 8
    for line in wrap_overlay_text(draw, subtitle, subtitle_font, width - margin * 2)[:2]:
        draw.text((margin, y + 4), line, font=subtitle_font, fill=(203, 213, 225, 255))
        y += subtitle_font.size + 7

    if items:
        columns = 2 if len(items) > 4 else 1
        rows = (len(items) + columns - 1) // columns
        available_height = height - bottom_top - footer_height - 44
        row_height = max(86, available_height // max(rows, 1))
        column_width = (width - margin * 2 - (28 if columns == 2 else 0)) // columns
        palette = [
            (96, 165, 250, 255),
            (52, 211, 153, 255),
            (251, 191, 36, 255),
            (196, 181, 253, 255),
            (248, 113, 113, 255),
            (45, 212, 191, 255),
            (244, 114, 182, 255),
            (129, 140, 248, 255),
        ]

        for index, item in enumerate(items):
            label = str(item.get("label", ""))
            detail = str(item.get("detail", ""))
            col = index % columns
            row = index // columns
            x = margin + col * (column_width + 28)
            y = bottom_top + 28 + row * row_height
            color = palette[index % len(palette)]
            marker = max(30, width // 32)
            draw.ellipse((x, y + 4, x + marker, y + 4 + marker), fill=color)
            number_text = str(index + 1)
            number_bbox = draw.textbbox((0, 0), number_text, font=detail_font)
            draw.text(
                (
                    x + marker / 2 - (number_bbox[2] - number_bbox[0]) / 2,
                    y + 4 + marker / 2 - (number_bbox[3] - number_bbox[1]) / 2 - 1,
                ),
                number_text,
                font=detail_font,
                fill=(3, 7, 18, 255),
            )
            text_x = x + marker + 16
            max_text_width = column_width - marker - 18
            label_lines = wrap_overlay_text(draw, label, label_font, max_text_width)
            draw.text((text_x, y), label_lines[0] if label_lines else label, font=label_font, fill=(248, 250, 252, 255))
            detail_y = y + label_font.size + 6
            for detail_line in wrap_overlay_text(draw, detail, detail_font, max_text_width)[:2]:
                draw.text((text_x, detail_y), detail_line, font=detail_font, fill=(203, 213, 225, 255))
                detail_y += detail_font.size + 4

    if footer:
        footer_top = height - footer_height
        draw.rectangle((0, footer_top, width, height), fill=(2, 6, 23, 238))
        draw.rectangle((margin, footer_top + 16, width - margin, footer_top + 20), fill=(251, 191, 36, 230))
        footer_lines = wrap_overlay_text(draw, footer, footer_font, width - margin * 2)
        footer_y = footer_top + 30
        for line in footer_lines[:2]:
            draw.text((margin, footer_y), line, font=footer_font, fill=(254, 249, 195, 255))
            footer_y += footer_font.size + 6

    image.convert("RGB").save(output_path, format="PNG")


def write_generation_errors(report_dir: Path, errors: list[dict[str, str]]) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    error_report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "errors": errors,
    }
    (report_dir / "generation-errors.json").write_text(
        json.dumps(error_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def generate_image_with_http(
    api_key: str,
    base_url: str,
    model: str,
    job: dict[str, Any],
    retries: int,
    request_timeout: int,
) -> bytes:
    """Generate one image through an OpenAI-compatible HTTP endpoint."""
    endpoint = f"{base_url.rstrip('/')}/images/generations"
    payload = {
        "model": model,
        "prompt": job["prompt"],
    }
    if job.get("size") != "default":
        payload["size"] = job["size"]
    if job.get("quality") != "default":
        payload["quality"] = job["quality"]
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
            with urllib.request.urlopen(request, timeout=request_timeout) as response:
                response_data = json.loads(response.read().decode("utf-8"))
                break
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace").strip()
            error_detail = f": {error_body[:500]}" if error_body else ""
            if exc.code in {408, 429, 500, 502, 503, 504, 524} and attempt < retries:
                wait_seconds = 8 * (attempt + 1)
                print(
                    f"Image API returned HTTP {exc.code} for {job['filename']}; "
                    f"retrying in {wait_seconds}s...{error_detail}",
                    flush=True,
                )
                time.sleep(wait_seconds)
                continue
            raise RuntimeError(
                f"Image API request failed with HTTP {exc.code} for {job['filename']}{error_detail}"
            ) from exc
        except (urllib.error.URLError, TimeoutError, ConnectionResetError, http.client.HTTPException) as exc:
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
    report_dir = Path(args.report_dir)
    jobs = selected_jobs(args.only)

    print(f"model: {args.model}", flush=True)
    print(f"base_url: {args.base_url}", flush=True)
    print(f"output_dir: {output_dir}", flush=True)
    print(f"report_dir: {report_dir}", flush=True)
    print(f"request_timeout: {args.request_timeout}s", flush=True)
    print(f"retries: {args.retries}", flush=True)
    print(f"jobs: {len(jobs)}", flush=True)

    if args.ensure_placeholders:
        ensure_selected_placeholders(output_dir, jobs, overwrite=args.overwrite)
        write_manifest(report_dir, IMAGE_JOBS)
        print(f"Placeholders ensured under {output_dir}", flush=True)
        return

    if args.dry_run:
        for job in jobs:
            print(f"DRY RUN: {job['filename']} ({job['size']}, {job['quality']}) - {job['title']}", flush=True)
        return

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set. Set it in your local shell before generating images.")

    ensure_output_dir(output_dir)
    write_manifest(report_dir, jobs)

    client = None
    if args.http_fallback:
        print("Using the built-in HTTP fallback.", flush=True)
    else:
        try:
            from openai import OpenAI
        except ImportError as exc:
            print("The Python package `openai` is not installed; using the built-in HTTP fallback.", flush=True)
        else:
            client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=args.base_url, timeout=args.request_timeout)

    errors: list[dict[str, str]] = []
    generated_backgrounds: dict[str, bytes] = {}
    for job in jobs:
        output_path = output_dir / job["filename"]
        if output_path.exists() and output_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") and not args.overwrite:
            print(f"Skipping existing valid PNG: {job['filename']} (use --overwrite to regenerate)", flush=True)
            continue
        print(f"Generating {job['filename']}...", flush=True)
        try:
            background_key = str(job.get("background_key") or "")
            if background_key and background_key in generated_backgrounds:
                output_path.write_bytes(generated_backgrounds[background_key])
            else:
                if client:
                    result = client.images.generate(
                        model=args.model,
                        prompt=job["prompt"],
                        **({} if job.get("size") == "default" else {"size": job["size"]}),
                        **({} if job.get("quality") == "default" else {"quality": job["quality"]}),
                    )
                    image_base64 = result.data[0].b64_json
                    image_bytes = base64.b64decode(image_base64)
                else:
                    image_bytes = generate_image_with_http(
                        api_key=os.environ["OPENAI_API_KEY"],
                        base_url=args.base_url,
                        model=args.model,
                        job=job,
                        retries=args.retries,
                        request_timeout=args.request_timeout,
                    )
                if background_key:
                    generated_backgrounds[background_key] = image_bytes
                output_path.write_bytes(image_bytes)
            apply_text_overlay(output_path, job)
            set_user_readable_permissions(output_path)
            print(f"Saved {output_path}", flush=True)
        except Exception as exc:
            errors.append({"filename": job["filename"], "error": str(exc)})
            write_generation_errors(report_dir, errors)
            if args.continue_on_error:
                print(f"Failed {job['filename']}: {exc}", flush=True)
                continue
            raise

    write_generation_errors(report_dir, errors)
    if errors:
        raise SystemExit(f"Image generation finished with {len(errors)} error(s). See {report_dir / 'generation-errors.json'}.")


if __name__ == "__main__":
    main()
