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
        "filename": "stage0-tools-foundation.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "开发者工具基础主视觉",
        "suggested_page": "docs/stage0/index.md",
        "alt": "开发者工具基础主视觉：终端、Git、编辑器和 Python 环境组成稳定工作台。",
        "prompt": """
一张适合中文技术课程的阶段主视觉，主题是“开发者工具基础”。
画面表现一个整洁的开发工作台：终端窗口、Git 分支图、代码编辑器、Python 环境、文件夹和检查清单，整体感觉新手友好、清爽、可靠。
适合放在课程阶段首页，画面留出一定呼吸空间，不要生成具体文字，不要出现真实品牌 logo。
""".strip(),
    },
    {
        "filename": "stage1-python-foundation.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Python 编程基础主视觉",
        "suggested_page": "docs/stage1/index.md",
        "alt": "Python 编程基础主视觉：变量、函数、数据结构和小项目逐步组成程序。",
        "prompt": """
一张适合中文编程课程的阶段主视觉，主题是“Python 编程基础”。
画面表现变量、函数、列表、字典、文件读写和命令行小工具像积木一样组成一个可运行程序，风格现代、轻松、清晰。
适合新手学习页面，不要生成具体文字，不要出现真实品牌 logo。
""".strip(),
    },
    {
        "filename": "stage2-data-visualization.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "数据分析与可视化主视觉",
        "suggested_page": "docs/stage2/index.md",
        "alt": "数据分析与可视化主视觉：表格数据经过清洗、聚合和图表表达形成分析报告。",
        "prompt": """
一张适合数据分析课程的阶段主视觉，主题是“数据分析与可视化”。
画面表现一张凌乱数据表逐步被清洗、聚合、转成折线图、柱状图和分析报告，风格清晰、亲切、现代，适合新手建立数据工作流直觉。
不要生成具体文字，不要出现真实品牌 logo。
""".strip(),
    },
    {
        "filename": "stage3-ai-math.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "AI 数学基础主视觉",
        "suggested_page": "docs/stage3/index.md",
        "alt": "AI 数学基础主视觉：向量、矩阵、概率分布和梯度下降连接到模型训练。",
        "prompt": """
一张适合 AI 数学入门课程的阶段主视觉，主题是“AI 数学最小必要基础”。
画面把向量箭头、矩阵网格、概率分布曲线、梯度下降路径和小模型训练连接在一起，像一张温和的学习地图。
风格清晰、具体、帮助新手降低公式恐惧，不要生成具体文字，不要出现真实品牌 logo。
""".strip(),
    },
    {
        "filename": "stage3-linear-algebra-roadmap.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "线性代数学习地图",
        "suggested_page": "docs/stage3/ch01-linear-algebra/00-roadmap.md",
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
        "suggested_page": "docs/stage3/ch01-linear-algebra/03-eigenvalues.md",
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
        "suggested_page": "docs/stage3/ch01-linear-algebra/04-vector-spaces.md",
        "alt": "向量空间基向量张成示意图：基向量通过线性组合覆盖可到达空间。",
        "prompt": """
一张适合线性代数入门课程的教学图，主题是“基向量如何张成一个向量空间”。
画面表现两个不同颜色的基向量从原点出发，通过平移网格和线性组合覆盖整个二维平面；同时用一个目标点展示“沿着两个基向量走几步就能到达”。
风格像新手友好的数学白板，几何直觉强，空间感清楚。
不要出现真实品牌 logo，不要生成复杂公式，不要生成难以阅读的小字。
""".strip(),
    },
    {
        "filename": "stage3-probability-roadmap.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "概率与统计学习地图",
        "suggested_page": "docs/stage3/ch02-probability/00-roadmap.md",
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
        "suggested_page": "docs/stage3/ch02-probability/02-distributions.md",
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
        "suggested_page": "docs/stage3/ch02-probability/03-statistical-inference.md",
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
        "filename": "stage3-calculus-roadmap.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "微积分与优化学习地图",
        "suggested_page": "docs/stage3/ch03-calculus/00-roadmap.md",
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
        "suggested_page": "docs/stage3/ch03-calculus/01-derivatives.md",
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
        "suggested_page": "docs/stage3/ch03-calculus/02-partial-derivatives-gradient.md",
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
        "suggested_page": "docs/stage3/ch03-calculus/04-chain-rule-backprop.md",
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
        "suggested_page": "docs/stage3/study-guide.md",
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
        "suggested_page": "docs/stage3/task-list.md",
        "alt": "数学最小通关任务单图：向量相似度、概率分布和梯度下降三个小实验组成最低通关。",
        "prompt": """
一张适合 AI 数学阶段任务清单的插图，主题是“三个最小实验通关数学基础”。
画面表现三个并列小任务：画向量并比较相似度、采样随机数并观察分布、让一个点沿损失曲面下降。每个任务像可完成的实验卡片，带有轻量 check 标记和代码工作台氛围。
风格新手友好、实战感强、让学习者愿意动手。
不要出现真实品牌 logo，不要生成密集小字，不要生成乱码文字。
""".strip(),
    },
    {
        "filename": "stage4-machine-learning.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "机器学习主视觉",
        "suggested_page": "docs/stage4/index.md",
        "alt": "机器学习主视觉：数据、baseline、评估、特征工程和模型复盘组成建模闭环。",
        "prompt": """
一张适合机器学习课程的阶段主视觉，主题是“机器学习入门到实战”。
画面表现数据表进入建模流水线，经过训练集测试集、baseline、评估指标、错误分析和特征工程，最终形成一份模型报告。
风格专业但新手友好，不要生成具体文字，不要出现真实品牌 logo。
""".strip(),
    },
    {
        "filename": "stage5-deep-learning.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "深度学习与 Transformer 主视觉",
        "suggested_page": "docs/stage5/index.md",
        "alt": "深度学习与 Transformer 主视觉：张量、训练循环、CNN、RNN、Attention 和 Transformer 串成学习路径。",
        "prompt": """
一张适合深度学习课程的阶段主视觉，主题是“深度学习与 Transformer 基础”。
画面表现张量数据进入神经网络训练循环，再连接到 CNN、RNN、Attention 和 Transformer 模块，像打开模型发动机舱。
风格现代科技感但不过度炫目，适合课程首页，不要生成具体文字，不要出现真实品牌 logo。
""".strip(),
    },
    {
        "filename": "stage6-computer-vision.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "计算机视觉主视觉",
        "suggested_page": "docs/stage6/index.md",
        "alt": "计算机视觉主视觉：图像分类、目标检测、分割、OCR 和医学影像组成视觉任务地图。",
        "prompt": """
一张适合计算机视觉课程的阶段主视觉，主题是“计算机视觉”。
画面表现同一张图片被模型逐步理解：分类标签、目标框、分割 mask、OCR 文本框和医学影像热力图，突出从粗到细的视觉任务层级。
风格清晰、工程感、适合新手课程，不要生成具体文字，不要出现真实品牌 logo。
""".strip(),
    },
    {
        "filename": "stage7-nlp.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "自然语言处理主视觉",
        "suggested_page": "docs/stage7/index.md",
        "alt": "自然语言处理主视觉：文本经过分词、词向量、分类、序列标注和预训练模型处理。",
        "prompt": """
一张适合自然语言处理课程的阶段主视觉，主题是“NLP 自然语言处理”。
画面表现一段文本被分词、转成向量、进行情感分类、实体标注、摘要和问答，最后连接到 BERT/GPT 风格的预训练模型抽象模块。
风格温和、清晰、有学习地图感，不要生成具体文字，不要出现真实品牌 logo。
""".strip(),
    },
    {
        "filename": "stage8a-llm-principles.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "大模型原理主视觉",
        "suggested_page": "docs/stage8a/index.md",
        "alt": "大模型原理主视觉：Token、Embedding、Transformer、预训练、Prompt、微调和对齐组成能力链路。",
        "prompt": """
一张适合大模型课程的阶段主视觉，主题是“大模型原理、Prompt 与微调”。
画面表现文本被切成 Token，进入 Embedding 和 Transformer 层，经过预训练、Prompt、微调和对齐，形成可控的大模型能力。
风格现代、抽象但易懂，不要生成具体文字，不要出现真实品牌 logo。
""".strip(),
    },
    {
        "filename": "stage8b-rag-engineering.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "LLM 应用与 RAG 主视觉",
        "suggested_page": "docs/stage8b/index.md",
        "alt": "LLM 应用与 RAG 主视觉：文档处理、向量库、检索、Prompt、API 和日志组成知识库系统。",
        "prompt": """
一张适合 LLM 应用工程课程的阶段主视觉，主题是“LLM 应用开发与 RAG”。
画面表现 PDF、Word、PPT 和网页资料进入文档处理流水线，写入向量数据库，再通过检索、Prompt、API、日志和评估生成带来源答案。
风格工程化、清晰、适合新手理解系统结构，不要生成具体文字，不要出现真实品牌 logo。
""".strip(),
    },
    {
        "filename": "stage9-agent-systems.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "AI Agent 系统主视觉",
        "suggested_page": "docs/stage9/index.md",
        "alt": "AI Agent 系统主视觉：目标、计划、工具、记忆、观察和评估组成智能体执行闭环。",
        "prompt": """
一张适合 AI Agent 课程的阶段主视觉，主题是“AI Agent 与智能体系统”。
画面表现一个 AI 助手围绕目标拆解计划、调用工具、读取资料、更新记忆、记录 trace、做安全检查并输出结果。
风格清晰、系统设计感、适合技术课程，不要生成具体文字，不要出现真实品牌 logo。
""".strip(),
    },
    {
        "filename": "stage10-multimodal-aigc.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "AIGC 与多模态主视觉",
        "suggested_page": "docs/stage10/index.md",
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
        "suggested_page": "docs/stage0/ch02-git/01-git-basics.md",
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
        "suggested_page": "docs/stage2/ch03-pandas/01-core-structures.md",
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
        "suggested_page": "docs/stage2/ch04-visualization/04-best-practices.md",
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
        "suggested_page": "docs/stage3/ch03-calculus/03-gradient-descent.md",
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
        "suggested_page": "docs/stage4/ch01-ml-basics/01-what-is-ml.md",
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
        "suggested_page": "docs/stage4/ch04-evaluation/01-metrics.md",
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
        "suggested_page": "docs/stage5/ch02-pytorch/05-training-loop.md",
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
        "suggested_page": "docs/stage5/ch03-cnn/01-convolution-basics.md",
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
        "suggested_page": "docs/stage5/ch05-transformer/01-attention-mechanism.md",
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
        "suggested_page": "docs/stage6/ch03-detection/01-detection-overview.md",
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
        "suggested_page": "docs/stage6/ch04-segmentation/01-semantic-segmentation.md",
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
        "suggested_page": "docs/stage7/ch04-sequence-labeling/01-ner-overview.md",
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
        "suggested_page": "docs/stage7/ch06-pretrained/00-roadmap.md",
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
        "suggested_page": "docs/stage8a/ch05-prompt/01-prompt-basics.md",
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
        "suggested_page": "docs/stage8a/ch06-finetuning/02-lora-qlora.md",
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
        "suggested_page": "docs/stage8b/ch01-rag/01-rag-basics.md",
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
        "suggested_page": "docs/stage8b/ch05-projects/04-courseware-assistant.md",
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
        "suggested_page": "docs/stage9/ch03-tools/08-multi-tool-practice.md",
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
        "suggested_page": "docs/stage9/ch08-eval-safety/04-guardrails.md",
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
        "suggested_page": "docs/stage10/ch02-image-gen/01-diffusion-models.md",
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
        "suggested_page": "docs/stage2/ch04-visualization/01-matplotlib.md",
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
        "suggested_page": "docs/stage2/ch04-visualization/02-seaborn.md",
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
        "suggested_page": "docs/stage2/ch05-database/02-sql-basics.md",
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
        "suggested_page": "docs/stage2/ch06-projects/01-eda-project.md",
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
        "suggested_page": "docs/stage3/ch01-linear-algebra/01-vectors.md",
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
        "suggested_page": "docs/stage3/ch01-linear-algebra/02-matrices.md",
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
        "suggested_page": "docs/stage3/ch02-probability/01-probability-basics.md",
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
        "suggested_page": "docs/stage3/ch02-probability/04-information-theory.md",
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
        "suggested_page": "docs/stage4/ch01-ml-basics/02-sklearn-intro.md",
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
        "suggested_page": "docs/stage4/ch02-supervised/01-linear-regression.md",
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
        "suggested_page": "docs/stage4/ch02-supervised/02-logistic-regression.md",
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
        "suggested_page": "docs/stage4/ch02-supervised/03-decision-trees.md",
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
        "suggested_page": "docs/stage4/ch02-supervised/04-ensemble-learning.md",
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
        "suggested_page": "docs/stage4/ch03-unsupervised/01-clustering.md",
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
        "suggested_page": "docs/stage4/ch03-unsupervised/02-dimensionality-reduction.md",
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
        "suggested_page": "docs/stage5/ch01-nn-basics/02-forward-backward.md",
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
        "suggested_page": "docs/stage5/ch03-cnn/03-classic-architectures.md",
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
        "suggested_page": "docs/stage5/ch04-rnn/02-lstm-gru.md",
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
        "suggested_page": "docs/stage5/ch05-transformer/02-transformer-architecture.md",
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
        "suggested_page": "docs/stage7/ch02-embeddings/01-word-embedding.md",
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
        "suggested_page": "docs/stage7/ch06-pretrained/02-bert.md",
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
        "suggested_page": "docs/stage7/ch06-pretrained/03-gpt-series.md",
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
        "suggested_page": "docs/stage8a/ch07-alignment/02-rlhf.md",
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
        "suggested_page": "docs/stage8b/ch01-rag/07-rag-evaluation.md",
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
        "suggested_page": "docs/stage9/ch04-memory/01-memory-overview.md",
        "alt": "Agent 记忆系统分层图：短期上下文、长期记忆、情景记忆和程序记忆共同支持持续任务。",
        "prompt": """
一张适合 AI Agent 课程的系统结构图，主题是“Agent 记忆系统”。
画面表现短期上下文、长期知识库、用户偏好、情景记录、程序经验和检索机制如何共同支持 Agent 执行长期任务。
风格清晰、像智能体大脑剖面图，不要出现真实品牌 logo，不要生成难以阅读的小字。
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
        except urllib.error.URLError as exc:
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
