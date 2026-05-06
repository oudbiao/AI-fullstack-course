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
        "filename": "homepage-ai-history-comic-01-turing.png",
        "size": "1024x1792",
        "quality": "high",
        "title": "首页 AI 发展史漫画 01 图灵与 AI 梦想的起点",
        "suggested_page": "docs/index.md",
        "alt": "图灵与 AI 梦想的起点漫画：图灵机和图灵测试把机器智能变成可讨论、可测试的问题。",
        "prompt": """
制作一页 9:16 竖版中文科普故事漫画，标题：“图灵与 AI 梦想的起点 1936-1950”。
像真正漫画书内页，不是海报；暖色旧纸张质感，半写实人物，竖向 6 格分镜，格子之间有漫画边框。

第 1 格：1936，Alan Turing 在书桌前写论文，桌上有纸带和机械读写头。气泡：“机器能不能按规则思考？”
第 2 格：小黑板解释图灵机：纸带、读写头、状态、规则箭头。短句：“一步一步执行规则。”
第 3 格：图灵机纸带动起来，0 和 1 被读写头扫描。旁白：“复杂计算，可以拆成简单动作。”
第 4 格：1950，图灵测试：评委隔着屏幕和人类、机器对话。气泡：“我分不清谁是机器。”
第 5 格：成功点标签：“机器智能变成可测试问题。”问题标签：“会聊天不等于真理解。”
第 6 格：远处出现未来机器人和问号，箭头写：“下一阶段：AI 正式成为研究领域。”

底部历史意义：“AI 的核心问题被正式提出：机器能否表现出智能。”
所有文字必须是简体中文，短句、清晰、融入气泡、小黑板、旁白框和标签牌。不要乱码，不要英文水印，不要真实 logo。
""".strip(),
    },
    {
        "filename": "homepage-ai-history-comic-02-dartmouth.png",
        "size": "1536x1024",
        "quality": "high",
        "title": "首页 AI 发展史漫画 02 达特茅斯会议",
        "suggested_page": "docs/index.md",
        "alt": "达特茅斯会议 AI 正式诞生漫画：科学家围绕黑板提出 Artificial Intelligence，符号主义规则推理兴起。",
        "prompt": """
制作一页 9:16 竖版中文科普故事漫画，标题：“1956 达特茅斯会议：AI 正式诞生”。
像真正漫画书内页，1950 年代学术会议氛围，暖色旧纸张质感，竖向 6 格分镜。

第 1 格：科学家围在达特茅斯会议桌前，黑板写“人工智能”。姓名牌：约翰·麦卡锡、马文·明斯基、香农、纽厄尔、西蒙。
第 2 格：麦卡锡指向黑板。气泡：“给这个梦想一个名字：人工智能。”
第 3 格：机器用“如果 A 那么 B”的规则卡片证明数学题。旁白：“早期 AI 擅长逻辑推理。”
第 4 格：流程图展示符号主义：规则 → 推理 → 结论。短句：“把知识写成规则。”
第 5 格：成功点标签：“AI 成为正式研究领域。”问题标签：“人类智能很难全写成规则。”
第 6 格：规则卡片堆成高塔，箭头写：“下一阶段：研究者开始寻找会学习的机器。”

底部历史意义：“人类第一次把制造智能机器作为正式科学目标。”
所有文字必须是简体中文，短句、清晰、融入气泡、小黑板、旁白框和标签牌。不要乱码，不要英文水印，不要真实 logo。
""".strip(),
    },
    {
        "filename": "homepage-ai-history-comic-03-perceptron.png",
        "size": "1536x1024",
        "quality": "high",
        "title": "首页 AI 发展史漫画 03 感知机与第一次低谷",
        "suggested_page": "docs/index.md",
        "alt": "感知机兴奋与神经网络第一次低谷漫画：感知机通过权重分类，XOR 暴露单层网络无法线性分割。",
        "prompt": """
制作一页 9:16 竖版中文科普故事漫画，标题：“感知机的兴奋与低谷 1957-1969”。
像真正漫画书内页，复古实验室风格，竖向 6 格分镜，人物和技术解释都要清楚。

第 1 格：弗兰克·罗森布拉特展示感知机机器。气泡：“机器可以从数据里调权重！”
第 2 格：小黑板画人工神经元：输入 x1、x2，乘权重 w1、w2，相加后输出。短句：“超过阈值就输出 1。”
第 3 格：感知机把两类点用一条直线分开。旁白：“线性可分时很好用。”
第 4 格：马文·明斯基和西摩尔·帕普特在黑板画 XOR 四点图。气泡：“一条直线分不开。”
第 5 格：成功点标签：“机器能用数据调整权重。”问题标签：“单层网络表达能力太弱。”
第 6 格：神经网络实验室灯光变暗，箭头写：“下一阶段：规则系统重新成为主角。”

底部历史意义：“神经网络第一次出现希望，也第一次被现实打击。”
所有文字必须是简体中文，短句、清晰、融入气泡、小黑板、旁白框和标签牌。不要乱码，不要英文水印，不要真实 logo。
""".strip(),
    },
    {
        "filename": "homepage-ai-history-comic-04-expert-systems.png",
        "size": "1536x1024",
        "quality": "high",
        "title": "首页 AI 发展史漫画 04 专家系统时代",
        "suggested_page": "docs/index.md",
        "alt": "专家系统时代漫画：专家把 IF-THEN 规则写入知识库，窄领域有效但规则堆积导致难维护。",
        "prompt": """
制作一页 9:16 竖版中文科普故事漫画，标题：“专家系统时代：规则的辉煌与崩塌 1970s-1980s”。
像真正漫画书内页，复古医院、实验室和企业机房氛围，竖向 6 格分镜。

第 1 格：爱德华·费根鲍姆、布鲁斯·布坎南向医生和化学专家请教。旁白：“把专家经验搬进机器。”
第 2 格：专家把规则卡片塞进知识库机器。卡片写：“如果 发烧+感染 → 考虑诊断。”
第 3 格：推理机沿箭头查规则。小黑板：“知识库 + 推理机 = 专家系统。”
第 4 格：机器在医学、化学、企业配置中给出建议。气泡：“窄领域真的有用！”
第 5 格：成功点标签：“规则系统能解决专业任务。”问题标签：“规则太多、冲突、难维护。”
第 6 格：规则卡片堆成山，机器冒烟，箭头写：“下一阶段：让机器自己从数据学习。”

底部历史意义：“AI 证明了窄领域规则系统有价值，也证明了人工写规则难以扩展。”
所有文字必须是简体中文，短句、清晰、融入气泡、小黑板、旁白框和标签牌。不要乱码，不要英文水印，不要真实 logo。
""".strip(),
    },
    {
        "filename": "homepage-ai-history-comic-05-backprop.png",
        "size": "1536x1024",
        "quality": "high",
        "title": "首页 AI 发展史漫画 05 反向传播",
        "suggested_page": "docs/index.md",
        "alt": "反向传播重新点火漫画：错误信号从输出层传回前面每层，指导多层神经网络调整参数。",
        "prompt": """
制作一页 9:16 竖版中文科普故事漫画，标题：“反向传播：神经网络重新点火 1986”。
像真正漫画书内页，黑板课堂加实验室氛围，竖向 6 格分镜。

第 1 格：杰弗里·辛顿、鲁梅尔哈特、威廉姆斯站在多层网络黑板前。气泡：“多层网络该怎么训练？”
第 2 格：前向传播：输入经过多层，输出答错。旁白：“先看预测错了多少。”
第 3 格：错误信号像红色水流，从输出层往回流。短句：“把错误传回每一层。”
第 4 格：每层旋钮轻轻调整权重。小黑板：“参数 = 参数 - 学习率 × 梯度。”
第 5 格：成功点标签：“多层网络知道每层该怎么改。”问题标签：“当时数据少、算力弱。”
第 6 格：网络重新亮起，但远处有更深网络大山。箭头：“下一阶段：视觉任务先取得实用突破。”

底部历史意义：“反向传播为后来深度学习打下训练基础。”
所有文字必须是简体中文，公式可以保留符号，短句清晰，融入气泡、小黑板、旁白框和标签牌。不要乱码，不要英文水印，不要真实 logo。
""".strip(),
    },
    {
        "filename": "homepage-ai-history-comic-06-lenet.png",
        "size": "1536x1024",
        "quality": "high",
        "title": "首页 AI 发展史漫画 06 LeNet 与 CNN 实用成功",
        "suggested_page": "docs/index.md",
        "alt": "LeNet 与 CNN 实用成功漫画：卷积滤镜扫描手写数字，逐层学习边缘、笔画和数字类别。",
        "prompt": """
制作一页 9:16 竖版中文科普故事漫画，标题：“LeNet 与 CNN 的实用成功 1989-1998”。
像真正漫画书内页，1990 年代实验室和邮政/支票识别场景，竖向 6 格分镜。

第 1 格：杨立昆和团队面对一堆手写数字、支票和邮政编码。气泡：“让机器认出手写数字。”
第 2 格：小滤镜像放大镜在数字图片上滑动。旁白：“卷积：局部扫描。”
第 3 格：网络逐层看到线条、笔画、数字形状。短句：“低层看边缘，高层看数字。”
第 4 格：机器输出：“这是 8。”流水线开始自动识别。
第 5 格：成功点标签：“神经网络第一次在工业场景证明价值。”问题标签：“任务仍较窄。”
第 6 格：自然图片里的猫狗、街景出现，箭头：“下一阶段：更大的数据和更强算力。”

底部历史意义：“CNN 展示了自动学习视觉特征的力量。”
所有文字必须是简体中文，短句、清晰、融入气泡、小黑板、旁白框和标签牌。不要乱码，不要英文水印，不要真实 logo。
""".strip(),
    },
    {
        "filename": "homepage-ai-history-comic-07-statistical-ml.png",
        "size": "1536x1024",
        "quality": "high",
        "title": "首页 AI 发展史漫画 07 统计机器学习时代",
        "suggested_page": "docs/index.md",
        "alt": "统计机器学习时代漫画：数据取代手写规则，工程师仍需提取特征卡片喂给分类器。",
        "prompt": """
制作一页 9:16 竖版中文科普故事漫画，标题：“统计机器学习时代：数据取代规则 1990s-2000s”。
像真正漫画书内页，数据实验室和互联网早期业务场景，竖向 6 格分镜。

第 1 格：瓦普尼克、布雷曼、弗里德曼站在算法白板旁。旁白：“不再只写规则，开始从数据学规律。”
第 2 格：表格数据、用户点击、文本词频进入模型流水线。短句：“数据越多，规律越清楚。”
第 3 格：工程师把“颜色、边缘、词频、点击次数”做成特征卡片。气泡：“先帮模型整理信息。”
第 4 格：SVM、随机森林、Boosting 像分类器机器输出预测。
第 5 格：成功点标签：“表格、搜索、广告、分类任务表现强。”问题标签：“太依赖人工特征工程。”
第 6 格：图片、语音、语言堆成大山，箭头：“下一阶段：让神经网络自己学特征。”

底部历史意义：“AI 从手写规则转向数据学习，但仍没摆脱人工特征设计。”
所有文字必须是简体中文，术语 SVM、Boosting 可保留，短句清晰，融入气泡、小黑板、旁白框和标签牌。不要乱码，不要英文水印，不要真实 logo。
""".strip(),
    },
    {
        "filename": "homepage-ai-history-comic-08-imagenet-alexnet.png",
        "size": "1536x1024",
        "quality": "high",
        "title": "首页 AI 发展史漫画 08 ImageNet 与 AlexNet",
        "suggested_page": "docs/index.md",
        "alt": "ImageNet 与 AlexNet 深度学习爆发漫画：大数据、GPU 和 CNN 让模型从原始图片自动学习特征。",
        "prompt": """
制作一页 9:16 竖版中文科普故事漫画，标题：“ImageNet 与 AlexNet：深度学习爆发 2009-2012”。
像真正漫画书内页，现代实验室和竞赛现场，竖向 6 格分镜。

第 1 格：李飞飞建造巨大的图片图书馆。旁白：“ImageNet：给机器看海量图片。”
第 2 格：Alex Krizhevsky、Ilya Sutskever、Geoffrey Hinton 在 GPU 机器旁训练 AlexNet。
第 3 格：CNN 层级机器逐层学习：边缘 → 眼睛耳朵 → 猫狗类别。
第 4 格：比赛排行榜上 AlexNet 大幅领先传统方法。气泡：“原始图片也能自动学特征！”
第 5 格：成功点标签：“大数据 + GPU + CNN 引爆深度学习。”问题标签：“模型更大，算力需求更高。”
第 6 格：更深网络像高楼延伸，箭头：“下一阶段：怎样训练非常深的网络？”

底部历史意义：“2012 年成为深度学习真正爆发的转折点。”
所有文字必须是简体中文，ImageNet、AlexNet、GPU、CNN 可保留，短句清晰，融入气泡、小黑板、旁白框和标签牌。不要乱码，不要英文水印，不要真实 logo。
""".strip(),
    },
    {
        "filename": "homepage-ai-history-comic-09-resnet.png",
        "size": "1536x1024",
        "quality": "high",
        "title": "首页 AI 发展史漫画 09 ResNet 残差连接",
        "suggested_page": "docs/index.md",
        "alt": "ResNet 残差连接漫画：信息高速公路把 X 直接送到后面，网络学习修正量 F(X)。",
        "prompt": """
制作一页 9:16 竖版中文科普故事漫画，标题：“ResNet：为什么要把 X 加回来 2015”。
像真正漫画书内页，工程结构感强，竖向 6 格分镜。

第 1 格：何恺明、张祥雨、任少卿、孙剑站在很深的神经网络高楼前。气泡：“网络越深，反而更难训练？”
第 2 格：普通信号爬楼梯越爬越弱。旁白：“深层网络会退化，梯度也难传播。”
第 3 格：一条信息高速公路绕过复杂层，把 X 直接送到后面。
第 4 格：小黑板写：“输出 = X + F(X)”。短句：“先保留原信息，再学习修正量。”
第 5 格：成功点标签：“50 层、101 层、152 层网络可训练。”问题标签：“模型更大，计算更贵。”
第 6 格：残差桥连接到未来 Transformer，箭头：“下一阶段：序列和语言模型继续发展。”

底部历史意义：“残差连接成为深度学习基础组件，也被 Transformer 继承。”
所有文字必须是简体中文，公式可保留，短句清晰，融入气泡、小黑板、旁白框和标签牌。不要乱码，不要英文水印，不要真实 logo。
""".strip(),
    },
    {
        "filename": "homepage-ai-history-comic-10-rnn-lstm.png",
        "size": "1536x1024",
        "quality": "high",
        "title": "首页 AI 发展史漫画 10 RNN 与 LSTM",
        "suggested_page": "docs/index.md",
        "alt": "RNN 与 LSTM 序列模型漫画：RNN 一个词一个词传递 hidden state，LSTM 用门控制记住和忘掉。",
        "prompt": """
制作一页 9:16 竖版中文科普故事漫画，标题：“RNN 与 LSTM：语言序列的早期主力 1997-2014”。
像真正漫画书内页，语言序列和时间线清楚，竖向 6 格分镜。

第 1 格：Hochreiter 和 Schmidhuber 展示 LSTM 记忆门。旁白：“文字、语音、时间序列都要按顺序理解。”
第 2 格：RNN 像接力队，一个词一个词传递“记忆接力棒”。短句：“历史信息存在 hidden state。”
第 3 格：长句子传到后面，接力棒变暗。气泡：“太远的信息容易忘。”
第 4 格：LSTM 加上三个门：记住、忘掉、输出。小黑板：“门 = 信息开关。”
第 5 格：成功点标签：“能处理文本、语音和翻译序列。”问题标签：“按顺序计算，难并行，长距离仍难。”
第 6 格：一束聚光灯照向句子重点，箭头：“下一阶段：Attention 直接看重点。”

底部历史意义：“RNN/LSTM 让机器能处理序列，也暴露了长距离依赖和并行瓶颈。”
所有文字必须是简体中文，RNN、LSTM、hidden state 可保留，短句清晰，融入气泡、小黑板、旁白框和标签牌。不要乱码，不要英文水印，不要真实 logo。
""".strip(),
    },
    {
        "filename": "homepage-ai-history-comic-11-attention.png",
        "size": "1536x1024",
        "quality": "high",
        "title": "首页 AI 发展史漫画 11 Attention",
        "suggested_page": "docs/index.md",
        "alt": "Attention 机器翻译漫画：生成当前词时用聚光灯关注输入句子中最相关的位置。",
        "prompt": """
制作一页 9:16 竖版中文科普故事漫画，标题：“Attention：别死记，直接看重点 2014”。
像真正漫画书内页，机器翻译课堂和聚光灯比喻，竖向 6 格分镜。

第 1 格：Bahdanau、Cho、Bengio 在翻译黑板旁。气泡：“整句话压成一个向量，信息太挤。”
第 2 格：旧 Seq2Seq 把长句塞进小瓶子，瓶子快爆。旁白：“信息瓶颈。”
第 3 格：翻译机器人生成“人工”时，聚光灯照向英文 artificial。
第 4 格：权重条像调光器：重点词 70%，次要词 20%。短句：“按权重看重点。”
第 5 格：成功点标签：“生成每个词时能直接看相关位置。”问题标签：“早期仍常依赖 RNN，难并行。”
第 6 格：RNN 链条被剪开，箭头：“下一阶段：只用 Attention 的 Transformer。”

底部历史意义：“Attention 让模型从记忆整句转向按需查看重点。”
所有文字必须是简体中文，Attention、Seq2Seq 可保留，短句清晰，融入气泡、小黑板、旁白框和标签牌。不要乱码，不要英文水印，不要真实 logo。
""".strip(),
    },
    {
        "filename": "homepage-ai-history-comic-12-transformer.png",
        "size": "1536x1024",
        "quality": "high",
        "title": "首页 AI 发展史漫画 12 Transformer",
        "suggested_page": "docs/index.md",
        "alt": "Transformer 自注意力漫画：所有 token 围坐会议桌，通过 Q/K/V 和 multi-head attention 直接交流。",
        "prompt": """
制作一页 9:16 竖版中文科普故事漫画，标题：“Transformer：Attention Is All You Need 2017”。
像真正漫画书内页，现代科技感但保持漫画分镜，竖向 6 格分镜。

第 1 格：Vaswani、Shazeer、Parmar、Uszkoreit、Jones、Gomez、Kaiser、Polosukhin 作为作者团队站在论文黑板前。
第 2 格：所有 token 围成会议桌，每个词都能直接看其他词。气泡：“不用一个个排队传话。”
第 3 格：Q/K/V 三张卡片：Q“我想找什么”，K“我有什么标签”，V“我提供的信息”。
第 4 格：多个专家头像同时观察句子：语法、指代、语义。短句：“多头注意力 = 多个视角。”
第 5 格：成功点标签：“并行训练，更好处理长距离依赖。”问题标签：“注意力计算量随长度变大。”
第 6 格：Transformer 变成大语言模型地基，箭头：“下一阶段：预训练基础模型。”

底部历史意义：“Transformer 成为现代大语言模型的基础架构。”
所有文字必须是简体中文，Q/K/V、token、Transformer 可保留，短句清晰，融入气泡、小黑板、旁白框和标签牌。不要乱码，不要英文水印，不要真实 logo。
""".strip(),
    },
    {
        "filename": "homepage-ai-history-comic-13-bert-gpt.png",
        "size": "1536x1024",
        "quality": "high",
        "title": "首页 AI 发展史漫画 13 BERT 与 GPT",
        "suggested_page": "docs/index.md",
        "alt": "BERT 与 GPT 预训练漫画：BERT 遮词再猜，GPT 根据前文预测下一个 token，预训练基础模型时代开始。",
        "prompt": """
制作一页 9:16 竖版中文科普故事漫画，标题：“BERT 与 GPT：预训练时代开始 2018-2020”。
像真正漫画书内页，左右路线对比清楚，竖向 6 格分镜。

第 1 格：Jacob Devlin 和 Alec Radford 站在 Transformer 地基前。旁白：“先用海量文本训练通用模型。”
第 2 格：BERT 像阅读理解学生，遮住一个词再猜。短句：“看左右上下文，做填空题。”
第 3 格：GPT 像写作机器人，从左到右续写。短句：“根据前文预测下一个 token。”
第 4 格：GPT-3 看到几个示例就模仿任务。气泡：“给我例子，我就接着做。”
第 5 格：成功点标签：“理解和生成任务都能靠预训练迁移。”问题标签：“会幻觉，事实不可靠。”
第 6 格：续写机器人走向对话教室，箭头：“下一阶段：让模型更听指令。”

底部历史意义：“AI 从为每个任务训练模型，转向大规模预训练基础模型。”
所有文字必须是简体中文，BERT、GPT、GPT-3、token 可保留，短句清晰，融入气泡、小黑板、旁白框和标签牌。不要乱码，不要英文水印，不要真实 logo。
""".strip(),
    },
    {
        "filename": "homepage-ai-history-comic-14-rlhf-chatgpt.png",
        "size": "1536x1024",
        "quality": "high",
        "title": "首页 AI 发展史漫画 14 SFT RLHF 与 ChatGPT",
        "suggested_page": "docs/index.md",
        "alt": "SFT RLHF 与 ChatGPT 漫画：续写模型经过指令训练和人类反馈，变成更会对话的助手。",
        "prompt": """
制作一页 9:16 竖版中文科普故事漫画，标题：“SFT、RLHF 与 ChatGPT：续写器变成助手 2022”。
像真正漫画书内页，训练教室和对话产品场景，竖向 6 格分镜。

第 1 格：只会续写的机器人不停写长文本。气泡：“我只会接着写。”
第 2 格：人类教师给“用户问题-助手回答”示例。小黑板：“SFT：看标准答案练习。”
第 3 格：人类评审比较多个回答，给出偏好排名。短句：“RLHF：学习人类更喜欢哪种回答。”
第 4 格：机器人学会对话，开始听指令、分步骤回答。
第 5 格：成功点标签：“普通用户第一次大规模体验 AI 对话。”问题标签：“仍会幻觉、过度自信。”
第 6 格：助手旁边出现知识库、工具箱和安全警示，箭头：“下一阶段：连接资料和工具完成真实任务。”

底部历史意义：“大语言模型从文本续写器变成大众可用的对话助手。”
所有文字必须是简体中文，SFT、RLHF、ChatGPT 可保留，短句清晰，融入气泡、小黑板、旁白框和标签牌。不要乱码，不要英文水印，不要真实 logo。
""".strip(),
    },
    {
        "filename": "homepage-ai-history-comic-15-rag-agent.png",
        "size": "1536x1024",
        "quality": "high",
        "title": "首页 AI 发展史漫画 15 RAG 工具调用与 Agent",
        "suggested_page": "docs/index.md",
        "alt": "RAG 工具调用与 Agent 漫画：AI 通过检索资料、调用工具、规划执行和检查结果走向真实任务。",
        "prompt": """
制作一页 9:16 竖版中文科普故事漫画，标题：“RAG、工具调用和 Agent：AI 走向真实任务 2023 至今”。
像真正漫画书内页，现代 AI 工作台、知识库和工具箱场景，竖向 6 格分镜。

第 1 格：用户提出真实任务：“帮我写一份课程课件。”AI 助手拿到目标。
第 2 格：RAG 流程图：问题 → 检索资料 → 放进上下文 → 基于资料回答。旁白：“先查资料，再回答。”
第 3 格：AI 调用工具：搜索、代码、计算器、文档生成器。短句：“不会的事，交给工具做。”
第 4 格：Agent 任务面板：目标 → 计划 → 调工具 → 观察结果 → 继续行动。
第 5 格：成功点标签：“AI 从回答问题走向完成任务。”问题标签：“检索错、工具错、提示注入、安全风险。”
第 6 格：安全护栏、权限确认、人工复核保护流程，箭头：“下一阶段：更可靠的长期自治系统。”

底部历史意义：“AI 从会说话，走向会查资料、会用工具、会执行任务。”
所有文字必须是简体中文，RAG、Agent 可保留，短句清晰，融入气泡、小黑板、旁白框和标签牌。不要乱码，不要英文水印，不要真实 logo。
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
画面像一个打开的项目文件夹和作品集页面，文件夹卡片包括 README、run command、baseline metrics、model comparison、threshold review、error samples、leakage check、next steps。
强调证据不是装饰，而是项目交付的一部分。
风格专业、干净、有作品集质感，适合放在实操课程末尾。
文字不是主体；标准术语和文件名保留英文。中文只用短标签。不要乱码小字或真实品牌 logo。
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
        "size": "1536x1024",
        "quality": "medium",
        "title": "PyTorch 自动求导梯度生命周期图",
        "suggested_page": "docs/ch06-deep-learning/ch02-pytorch/02-autograd.md",
        "alt": "PyTorch 自动求导梯度生命周期图：forward 生成 loss，backward 把梯度写入 grad，optimizer.step 更新参数，zero_grad 清空旧梯度。",
        "prompt": """
一张解释 PyTorch autograd 的循环机制图，主题是“梯度从产生、使用到清空的生命周期”。
画面按一轮训练循环展示：forward 计算 loss，autograd 记录 graph，loss.backward() 把 gradient 写进 .grad，optimizer.step() 更新 parameter，optimizer.zero_grad() 清空旧梯度；旁边标出“PyTorch 默认累计梯度”这个警示。
风格像透明训练机器剖面图，强调 .grad 存储和清零动作。
文字不是主体；中文短标签为主，API 和标准术语保留英文，例如 autograd、loss.backward()、.grad、optimizer.step()、zero_grad()。不要整张图全英文，不要乱码小字或真实品牌 logo。
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
        "filename": "ch09-learning-quest-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "Agent 学习闯关地图",
        "suggested_page": "docs/ch09-agent/index.md",
        "alt": "Agent 学习闯关地图：目标理解、任务拆解、计划生成、工具调用、观察结果、记忆更新、自我检查和完成任务逐步连接。",
        "prompt": """
一张适合 AI Agent 阶段首页的学习闯关地图，主题是“让 AI 从回答问题升级成执行任务”。
画面表现目标理解、任务拆解、plan、tool call、observation、memory update、self-check、task done 逐步连接，像一条可追踪的执行路线。
风格像课程冒险地图和工程 trace 面板结合，清晰、现代、适合新人。
文字不是主体；标准术语保留英文，例如 Agent、plan、tool call、observation、memory、trace、self-check。其他说明可用少量中文短标签。不要整张图全英文，不要乱码小字或真实品牌 logo。
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
        "filename": "ch10-learning-quest-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "计算机视觉学习闯关地图",
        "suggested_page": "docs/ch10-computer-vision/index.md",
        "alt": "计算机视觉学习闯关地图：像素颜色、OpenCV 预处理、图像分类、目标检测、图像分割、OCR 视频和视觉综合项目逐步连接。",
        "prompt": """
一张适合计算机视觉首页的学习闯关地图，主题是“模型如何从像素看见世界”。
画面表现 pixels/RGB、OpenCV preprocessing、image classification、object detection、segmentation、OCR/video、CV project 逐步连接。
风格像视觉任务路线图和工程流水线结合，清晰、新人友好。
文字不是主体；标准术语保留英文，例如 pixels、RGB、OpenCV、classification、object detection、segmentation、OCR、video。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
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
        "filename": "ch11-learning-quest-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "NLP 学习闯关地图",
        "suggested_page": "docs/ch11-nlp/index.md",
        "alt": "NLP 学习闯关地图：文本清洗、分词与 Token、词向量、文本分类、序列标注、Seq2Seq、预训练模型和 LLM 理解基础逐步连接。",
        "prompt": """
一张适合 NLP 首页的学习闯关地图，主题是“把人类语言变成模型能计算的表示”。
画面表现 text cleaning、Token、Embedding、text classification、sequence labeling、Seq2Seq、pretrained models、LLM foundation 逐步连接。
风格像文本处理流水线和学习路线图结合，清晰、新人友好。
文字不是主体；标准术语保留英文，例如 Token、Embedding、text classification、sequence labeling、Seq2Seq、pretrained models、LLM。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
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
        "filename": "ch12-learning-quest-map.png",
        "size": "1536x1024",
        "quality": "medium",
        "title": "多模态与 AIGC 学习闯关地图",
        "suggested_page": "docs/ch12-multimodal/index.md",
        "alt": "多模态与 AIGC 学习闯关地图：文本理解、图文对齐、图像生成、语音视频、多模态问答、内容审核和创意工作流逐步连接。",
        "prompt": """
一张适合多模态与 AIGC 首页的学习闯关地图，主题是“把 AI 从文字世界带到真实世界输入输出”。
画面表现 text understanding、vision-language alignment、image generation、voice/video、multimodal QA、content safety、creative workflow。
风格像创意产品路线图和多模态系统图结合，明亮、专业。
文字不是主体；标准术语保留英文，例如 vision-language alignment、image generation、voice、video、multimodal QA、content safety。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
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
画面像一个整洁的项目文件夹，里面有 README、run command、metrics table、failure cases、dashboard screenshot、next action、commit note 七张卡片。
强调作品集不是只说学过，而是展示代码、输出、评估、风险和下一步改进。
风格像项目交付清单和教学漫画结合，竖版、分步骤、清楚。
文字不是主体；标准术语保留英文，例如 README、run command、metrics、failure cases、dashboard、next action、commit。其他说明可用少量中文短标签。不要乱码小字或真实品牌 logo。
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

IMAGE_JOBS.extend(
    [
        {
            "filename": "homepage-ai-history-comic-en-01-turing.png",
            "size": "1024x1792",
            "quality": "high",
            "title": "English homepage AI history comic 01 Turing and the AI dream",
            "suggested_page": "i18n/en/docusaurus-plugin-content-docs/current/index.md",
            "alt": "Comic about Turing, the Turing machine, and the Turing Test as the starting point of the AI dream.",
            "prompt": """
Create a 9:16 vertical English science comic page titled: "Turing and the AI Dream 1936-1950".
Make it feel like a real comic-book page, not a poster: warm vintage paper, semi-realistic historical characters, six clear panels with comic borders.

Panel 1: 1936, Alan Turing writes a paper at his desk, with a paper tape and mechanical read/write head nearby. Speech bubble: "Can a machine follow rules to think?"
Panel 2: Blackboard explains a Turing machine: tape, read/write head, state, rule arrows. Short caption: "Step by step rule execution."
Panel 3: The machine scans 0s and 1s on a moving tape. Narration: "Complex computation can be broken into simple actions."
Panel 4: 1950 Turing Test: a judge chats through screens with a human and a machine. Speech bubble: "I cannot tell which one is the machine."
Panel 5: Tags: "Breakthrough: machine intelligence became testable." and "Limitation: conversation is not the same as understanding."
Panel 6: A future robot appears with a question mark. Arrow label: "Next: AI becomes a formal research field."

Bottom historical meaning: "AI's central question was formally raised: can machines behave intelligently?"
All text must be clear English, short, and integrated into speech bubbles, blackboards, captions, and labels. No gibberish, no watermark, no real logo.
""".strip(),
        },
        {
            "filename": "homepage-ai-history-comic-en-02-dartmouth.png",
            "size": "1024x1792",
            "quality": "high",
            "title": "English homepage AI history comic 02 Dartmouth Conference",
            "suggested_page": "i18n/en/docusaurus-plugin-content-docs/current/index.md",
            "alt": "Comic about the 1956 Dartmouth Conference and the birth of artificial intelligence as a research field.",
            "prompt": """
Create a 9:16 vertical English science comic page titled: "The Dartmouth Conference: AI Is Born 1956".
Make it a real comic-book page with a 1950s academic meeting atmosphere, warm vintage paper, and six clear panels.

Panel 1: Scientists gather around a Dartmouth meeting table. Name cards: John McCarthy, Marvin Minsky, Claude Shannon, Allen Newell, Herbert Simon.
Panel 2: McCarthy points at a blackboard. Speech bubble: "Let's give this dream a name: Artificial Intelligence."
Panel 3: A machine proves a small math theorem using "IF A THEN B" rule cards. Narration: "Early AI was strong at logical reasoning."
Panel 4: Flowchart: rules -> reasoning -> conclusion. Caption: "Write knowledge as rules."
Panel 5: Tags: "Breakthrough: AI became a formal research field." and "Limitation: human intelligence is hard to write as rules."
Panel 6: Rule cards pile into a tower. Arrow: "Next: researchers look for machines that can learn."

Bottom historical meaning: "For the first time, building intelligent machines became a formal scientific goal."
All text must be clear English, short, and integrated into bubbles, blackboards, captions, and labels. No gibberish, no watermark, no real logo.
""".strip(),
        },
        {
            "filename": "homepage-ai-history-comic-en-03-perceptron.png",
            "size": "1024x1792",
            "quality": "high",
            "title": "English homepage AI history comic 03 Perceptron and first neural winter",
            "suggested_page": "i18n/en/docusaurus-plugin-content-docs/current/index.md",
            "alt": "Comic about the perceptron boom, XOR limitation, and the first neural network downturn.",
            "prompt": """
Create a 9:16 vertical English science comic page titled: "Perceptron Boom and First Winter 1957-1969".
Use a real comic-book layout: retro lab style, six panels, clear technical visuals.

Panel 1: Frank Rosenblatt presents an early perceptron machine. Speech bubble: "A machine can adjust weights from data!"
Panel 2: Blackboard shows an artificial neuron: inputs x1, x2, weights w1, w2, sum, threshold, output. Caption: "If the score passes the threshold, output 1."
Panel 3: A perceptron separates two classes with one straight line. Narration: "Works well when data is linearly separable."
Panel 4: Marvin Minsky and Seymour Papert draw the XOR four-point problem on a blackboard. Speech bubble: "One line cannot separate this."
Panel 5: Tags: "Breakthrough: learning from data by changing weights." and "Limitation: a single layer is too weak."
Panel 6: Neural network lab lights dim. Arrow: "Next: rule-based expert systems take the spotlight."

Bottom historical meaning: "Neural networks showed promise, then met their first hard reality check."
All text must be clear English, short, and integrated into bubbles, blackboards, captions, and labels. No gibberish, no watermark, no real logo.
""".strip(),
        },
        {
            "filename": "homepage-ai-history-comic-en-04-expert-systems.png",
            "size": "1024x1792",
            "quality": "high",
            "title": "English homepage AI history comic 04 Expert systems",
            "suggested_page": "i18n/en/docusaurus-plugin-content-docs/current/index.md",
            "alt": "Comic about expert systems, knowledge bases, rule engines, and maintenance limits.",
            "prompt": """
Create a 9:16 vertical English science comic page titled: "Expert Systems: Rules Shine and Crack 1970s-1980s".
Use a real comic-book page style with vintage hospitals, laboratories, and office machines, six panels.

Panel 1: Edward Feigenbaum and Bruce Buchanan interview doctors and chemists. Narration: "Move expert knowledge into machines."
Panel 2: Experts feed rule cards into a knowledge-base machine. Rule card: "IF fever + infection THEN consider diagnosis."
Panel 3: Inference engine follows arrows through a rule cabinet. Blackboard: "Knowledge base + inference engine = expert system."
Panel 4: The system gives useful advice in medicine, chemistry, and business configuration. Speech bubble: "It works in narrow domains!"
Panel 5: Tags: "Breakthrough: rule systems solved specialist tasks." and "Limitation: rules conflict, pile up, and become hard to maintain."
Panel 6: Rule cards become a mountain; the machine smokes. Arrow: "Next: let machines learn from data."

Bottom historical meaning: "AI proved narrow-domain rule systems were useful, but hand-written rules could not scale to the real world."
All text must be clear English, short, and integrated into bubbles, blackboards, captions, and labels. No gibberish, no watermark, no real logo.
""".strip(),
        },
        {
            "filename": "homepage-ai-history-comic-en-05-backprop.png",
            "size": "1024x1792",
            "quality": "high",
            "title": "English homepage AI history comic 05 Backpropagation",
            "suggested_page": "i18n/en/docusaurus-plugin-content-docs/current/index.md",
            "alt": "Comic about backpropagation sending error signals backward through a multilayer neural network.",
            "prompt": """
Create a 9:16 vertical English science comic page titled: "Backpropagation Reignites Neural Networks 1986".
Use a real comic-book layout with classroom blackboards and lab equipment, six panels.

Panel 1: Geoffrey Hinton, David Rumelhart, and Ronald Williams stand beside a multilayer network blackboard. Speech bubble: "How do we train many layers?"
Panel 2: Forward pass: input flows through layers and produces a wrong answer. Narration: "First measure how wrong the prediction is."
Panel 3: Red error signal flows backward from output to earlier layers. Caption: "Send the error back."
Panel 4: Each layer adjusts small weight knobs. Blackboard formula: "parameter = parameter - learning rate x gradient."
Panel 5: Tags: "Breakthrough: each layer knows how to change." and "Limitation: data and compute were still too small."
Panel 6: The network lights up, but a mountain of deeper networks waits ahead. Arrow: "Next: vision tasks show practical success."

Bottom historical meaning: "Backpropagation became the training foundation for later deep learning."
All text must be clear English, with formulas allowed, integrated into bubbles, blackboards, captions, and labels. No gibberish, no watermark, no real logo.
""".strip(),
        },
        {
            "filename": "homepage-ai-history-comic-en-06-lenet.png",
            "size": "1024x1792",
            "quality": "high",
            "title": "English homepage AI history comic 06 LeNet and CNN",
            "suggested_page": "i18n/en/docusaurus-plugin-content-docs/current/index.md",
            "alt": "Comic about LeNet, convolution filters, handwritten digit recognition, and practical CNN success.",
            "prompt": """
Create a 9:16 vertical English science comic page titled: "LeNet and the First Practical CNN Success 1989-1998".
Use a real comic-book page style with a 1990s lab and postal/check recognition scenes, six panels.

Panel 1: Yann LeCun and teammates examine handwritten digits, checks, and postal codes. Speech bubble: "Can a machine read handwriting?"
Panel 2: A small filter slides like a magnifying glass over a digit image. Caption: "Convolution: scan locally."
Panel 3: Layers reveal edges, strokes, and digit shapes. Caption: "Low layers see edges; high layers see numbers."
Panel 4: The machine outputs: "This is 8." An industrial workflow starts using it.
Panel 5: Tags: "Breakthrough: neural networks worked in real industry." and "Limitation: tasks were still narrow."
Panel 6: Cats, dogs, and street scenes appear ahead. Arrow: "Next: larger data and stronger compute."

Bottom historical meaning: "CNNs showed the power of automatically learning visual features."
All text must be clear English, short, and integrated into bubbles, blackboards, captions, and labels. No gibberish, no watermark, no real logo.
""".strip(),
        },
        {
            "filename": "homepage-ai-history-comic-en-07-statistical-ml.png",
            "size": "1024x1792",
            "quality": "high",
            "title": "English homepage AI history comic 07 Statistical machine learning",
            "suggested_page": "i18n/en/docusaurus-plugin-content-docs/current/index.md",
            "alt": "Comic about statistical machine learning, feature engineering, SVM, random forests, and boosting.",
            "prompt": """
Create a 9:16 vertical English science comic page titled: "Statistical Machine Learning: Data Replaces Rules 1990s-2000s".
Use a real comic-book page style with data labs and early internet business scenes, six panels.

Panel 1: Vladimir Vapnik, Leo Breiman, and Jerome Friedman stand beside an algorithm whiteboard. Narration: "Instead of writing rules, learn patterns from data."
Panel 2: Tables, clicks, text counts, and labels flow into a model pipeline. Caption: "More data reveals patterns."
Panel 3: Engineers create feature cards: color, edge, word count, click count. Speech bubble: "First, organize information for the model."
Panel 4: SVM, random forest, and boosting machines output predictions.
Panel 5: Tags: "Breakthrough: strong results for tables, search, ads, and classification." and "Limitation: too much manual feature engineering."
Panel 6: Images, speech, and language become a mountain. Arrow: "Next: neural networks learn features themselves."

Bottom historical meaning: "AI shifted from hand-written rules to data learning, but still depended on human-designed features."
All text must be clear English; SVM and Boosting may remain as terms. No gibberish, no watermark, no real logo.
""".strip(),
        },
        {
            "filename": "homepage-ai-history-comic-en-08-imagenet-alexnet.png",
            "size": "1024x1792",
            "quality": "high",
            "title": "English homepage AI history comic 08 ImageNet and AlexNet",
            "suggested_page": "i18n/en/docusaurus-plugin-content-docs/current/index.md",
            "alt": "Comic about ImageNet, AlexNet, GPU training, CNN feature learning, and the deep learning explosion.",
            "prompt": """
Create a 9:16 vertical English science comic page titled: "ImageNet and AlexNet: Deep Learning Explodes 2009-2012".
Use a real comic-book page style with a modern lab, image library, and competition scene, six panels.

Panel 1: Fei-Fei Li builds a huge image library labeled ImageNet. Narration: "Let machines see millions of images."
Panel 2: Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton train AlexNet beside glowing GPU machines.
Panel 3: CNN layers learn features: edge -> parts -> cats and dogs.
Panel 4: A competition leaderboard shows AlexNet far ahead of older methods. Speech bubble: "Raw images can teach features!"
Panel 5: Tags: "Breakthrough: big data + GPU + CNN launched deep learning." and "Limitation: bigger models need much more compute."
Panel 6: A deeper neural network tower rises. Arrow: "Next: how do we train very deep networks?"

Bottom historical meaning: "2012 became the turning point when deep learning truly broke out."
All text must be clear English; ImageNet, AlexNet, GPU, and CNN may remain. No gibberish, no watermark, no real logo.
""".strip(),
        },
        {
            "filename": "homepage-ai-history-comic-en-09-resnet.png",
            "size": "1024x1792",
            "quality": "high",
            "title": "English homepage AI history comic 09 ResNet",
            "suggested_page": "i18n/en/docusaurus-plugin-content-docs/current/index.md",
            "alt": "Comic about ResNet residual connections and why adding X back helps train deep networks.",
            "prompt": """
Create a 9:16 vertical English science comic page titled: "ResNet: Why Add X Back? 2015".
Use a real comic-book page style with strong engineering structure, six panels.

Panel 1: Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun stand before a very deep neural-network tower. Speech bubble: "Why does a deeper network become harder to train?"
Panel 2: A normal signal climbs stairs and becomes weak. Narration: "Deep networks can degrade; gradients struggle to travel."
Panel 3: An information highway bypasses complex layers and carries X forward.
Panel 4: Blackboard: "output = X + F(X)". Caption: "Keep the original signal; learn the correction."
Panel 5: Tags: "Breakthrough: 50, 101, and 152 layers became trainable." and "Limitation: larger models cost more compute."
Panel 6: A residual bridge connects to a future Transformer. Arrow: "Next: sequence and language models evolve."

Bottom historical meaning: "Residual connections became a core deep learning component and later influenced Transformer design."
All text must be clear English, formulas allowed. No gibberish, no watermark, no real logo.
""".strip(),
        },
        {
            "filename": "homepage-ai-history-comic-en-10-rnn-lstm.png",
            "size": "1024x1792",
            "quality": "high",
            "title": "English homepage AI history comic 10 RNN and LSTM",
            "suggested_page": "i18n/en/docusaurus-plugin-content-docs/current/index.md",
            "alt": "Comic about RNNs, LSTMs, hidden state, memory gates, and long-distance dependency limits.",
            "prompt": """
Create a 9:16 vertical English science comic page titled: "RNN and LSTM: Early Workhorses for Sequences 1997-2014".
Use a real comic-book page style with language sequences and timelines, six panels.

Panel 1: Hochreiter and Schmidhuber present LSTM memory gates. Narration: "Text, speech, and time series need order."
Panel 2: RNN is drawn as a relay team passing a memory baton word by word. Caption: "History lives in the hidden state."
Panel 3: In a long sentence, the baton fades. Speech bubble: "Faraway information gets weak."
Panel 4: LSTM adds three gates: remember, forget, output. Blackboard: "Gates are information switches."
Panel 5: Tags: "Breakthrough: sequences for text, speech, and translation." and "Limitation: sequential, slow, and still hard for long dependencies."
Panel 6: A spotlight shines on important words. Arrow: "Next: Attention looks directly at what matters."

Bottom historical meaning: "RNN/LSTM made sequence modeling practical, while exposing the bottlenecks of long memory and parallel training."
All text must be clear English; RNN, LSTM, hidden state may remain. No gibberish, no watermark, no real logo.
""".strip(),
        },
        {
            "filename": "homepage-ai-history-comic-en-11-attention.png",
            "size": "1024x1792",
            "quality": "high",
            "title": "English homepage AI history comic 11 Attention",
            "suggested_page": "i18n/en/docusaurus-plugin-content-docs/current/index.md",
            "alt": "Comic about attention for machine translation, information bottlenecks, and weighted focus.",
            "prompt": """
Create a 9:16 vertical English science comic page titled: "Attention: Look at What Matters 2014".
Use a real comic-book page style with a machine translation classroom and spotlight metaphor, six panels.

Panel 1: Bahdanau, Cho, and Bengio stand beside a translation blackboard. Speech bubble: "One vector for a whole sentence is too crowded."
Panel 2: Old Seq2Seq squeezes a long sentence into a tiny bottle that is about to burst. Narration: "Information bottleneck."
Panel 3: A translation robot generates a word while a spotlight shines on the matching source word.
Panel 4: Weight bars act like dimmer switches: important word 70%, other word 20%. Caption: "Focus by weight."
Panel 5: Tags: "Breakthrough: each output word can look at relevant input positions." and "Limitation: early attention still often used RNNs, so parallelism was limited."
Panel 6: An RNN chain is cut away. Arrow: "Next: Transformer uses only Attention."

Bottom historical meaning: "Attention moved models from memorizing a whole sentence to looking up what matters when needed."
All text must be clear English; Attention and Seq2Seq may remain. No gibberish, no watermark, no real logo.
""".strip(),
        },
        {
            "filename": "homepage-ai-history-comic-en-12-transformer.png",
            "size": "1024x1792",
            "quality": "high",
            "title": "English homepage AI history comic 12 Transformer",
            "suggested_page": "i18n/en/docusaurus-plugin-content-docs/current/index.md",
            "alt": "Comic about Transformer self-attention, QKV, multi-head attention, parallel training, and LLM foundations.",
            "prompt": """
Create a 9:16 vertical English science comic page titled: "Transformer: Attention Is All You Need 2017".
Use a real comic-book page style with modern tech visuals and six clear panels.

Panel 1: The Transformer author team stands before a paper blackboard. Name tags: Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin.
Panel 2: Tokens sit around a meeting table and connect directly to each other. Speech bubble: "No need to pass messages one by one."
Panel 3: Q/K/V cards: Q = "what I am looking for", K = "what label I have", V = "the information I provide".
Panel 4: Multiple expert heads observe a sentence: grammar, reference, meaning. Caption: "Multi-head attention = many viewpoints."
Panel 5: Tags: "Breakthrough: parallel training and better long-range dependencies." and "Limitation: attention cost grows with sequence length."
Panel 6: Transformer becomes the foundation under a city of large language models. Arrow: "Next: pretrained foundation models."

Bottom historical meaning: "Transformer became the core architecture behind modern large language models."
All text must be clear English; Q/K/V, token, Transformer may remain. No gibberish, no watermark, no real logo.
""".strip(),
        },
        {
            "filename": "homepage-ai-history-comic-en-13-bert-gpt.png",
            "size": "1024x1792",
            "quality": "high",
            "title": "English homepage AI history comic 13 BERT and GPT",
            "suggested_page": "i18n/en/docusaurus-plugin-content-docs/current/index.md",
            "alt": "Comic about BERT, GPT, masked language modeling, next-token prediction, and pretraining.",
            "prompt": """
Create a 9:16 vertical English science comic page titled: "BERT and GPT: The Pretraining Era Begins 2018-2020".
Use a real comic-book page style with two model routes side by side, six panels.

Panel 1: Jacob Devlin and Alec Radford stand before a Transformer foundation. Narration: "Train a general model on massive text first."
Panel 2: BERT is a reading-comprehension student filling a masked word. Caption: "Look both left and right; solve a fill-in-the-blank."
Panel 3: GPT is a writing robot continuing text from left to right. Caption: "Predict the next token from previous context."
Panel 4: GPT-3 sees a few examples and imitates the task. Speech bubble: "Give me examples, and I continue the pattern."
Panel 5: Tags: "Breakthrough: pretraining transfers to many tasks." and "Limitation: hallucinations and unreliable facts."
Panel 6: A text-completing robot walks into an instruction classroom. Arrow: "Next: make models follow instructions."

Bottom historical meaning: "AI moved from training one model per task to large pretrained foundation models."
All text must be clear English; BERT, GPT, GPT-3, token may remain. No gibberish, no watermark, no real logo.
""".strip(),
        },
        {
            "filename": "homepage-ai-history-comic-en-14-rlhf-chatgpt.png",
            "size": "1024x1792",
            "quality": "high",
            "title": "English homepage AI history comic 14 SFT RLHF and ChatGPT",
            "suggested_page": "i18n/en/docusaurus-plugin-content-docs/current/index.md",
            "alt": "Comic about SFT, RLHF, ChatGPT, and turning a text completer into an assistant.",
            "prompt": """
Create a 9:16 vertical English science comic page titled: "SFT, RLHF, and ChatGPT: From Completer to Assistant 2022".
Use a real comic-book page style with a training classroom and chat product scene, six panels.

Panel 1: A text-completing robot writes endless messy text. Speech bubble: "I only continue text."
Panel 2: A human teacher gives user-question and assistant-answer examples. Blackboard: "SFT: practice with good examples."
Panel 3: Human reviewers compare several answers and rank preferences. Caption: "RLHF: learn what humans prefer."
Panel 4: The robot becomes a conversational assistant that follows instructions and answers step by step.
Panel 5: Tags: "Breakthrough: everyday users experience AI conversation." and "Limitation: still hallucinates and can sound overconfident."
Panel 6: Knowledge base, tool box, and safety warning appear beside the assistant. Arrow: "Next: connect tools and knowledge to finish real tasks."

Bottom historical meaning: "Large language models moved from text completion to usable conversational assistants."
All text must be clear English; SFT, RLHF, ChatGPT may remain. No gibberish, no watermark, no real logo.
""".strip(),
        },
        {
            "filename": "homepage-ai-history-comic-en-15-rag-agent.png",
            "size": "1024x1792",
            "quality": "high",
            "title": "English homepage AI history comic 15 RAG tools and Agents",
            "suggested_page": "i18n/en/docusaurus-plugin-content-docs/current/index.md",
            "alt": "Comic about RAG, tool calling, AI Agents, planning, execution, safety, and real tasks.",
            "prompt": """
Create a 9:16 vertical English science comic page titled: "RAG, Tool Calling, and Agents: AI Tackles Real Tasks 2023-Present".
Use a real comic-book page style with a modern AI workspace, knowledge base, and tool box, six panels.

Panel 1: A user asks: "Help me create a course handout." The AI assistant receives the goal.
Panel 2: RAG flow diagram: question -> retrieve documents -> put into context -> answer from sources. Narration: "First retrieve, then answer."
Panel 3: The AI calls tools: search, code, calculator, document generator. Caption: "If the model cannot do it, use a tool."
Panel 4: Agent task board: goal -> plan -> call tools -> observe result -> continue.
Panel 5: Tags: "Breakthrough: AI moves from answering questions to completing tasks." and "Risks: bad retrieval, tool errors, prompt injection, permissions."
Panel 6: Safety guardrails, permission checks, and human review protect the workflow. Arrow: "Next: more reliable long-running autonomous systems."

Bottom historical meaning: "AI is moving from talking to retrieving, using tools, and executing real work."
All text must be clear English; RAG and Agent may remain. No gibberish, no watermark, no real logo.
""".strip(),
        },
    ]
)


IMAGE_JOBS.extend(
    [
        {
            "filename": "homepage-ai-history-comic-ja-01-turing.png",
            "size": "1024x1792",
            "quality": "high",
            "title": "日本語ホーム AI 史マンガ 01 チューリングと AI の夢",
            "suggested_page": "i18n/ja/docusaurus-plugin-content-docs/current/index.md",
            "alt": "チューリング機械とチューリングテストが AI の出発点になったことを説明する日本語マンガ。",
            "prompt": """
9:16 縦長の日本語科学マンガページを作成。タイトル：「チューリングと AI の夢 1936-1950」。
本物の漫画本の 1 ページのように、温かい古い紙の質感、半写実の歴史人物、読みやすい 6 コマ構成、漫画枠を明確にする。

1コマ目：1936年、Alan Turing が机で論文を書き、紙テープと読み書きヘッドがある。吹き出し：「機械は規則で考えられる？」
2コマ目：小黒板で Turing machine を説明：テープ、ヘッド、状態、規則の矢印。短い説明：「一歩ずつ規則を実行」。
3コマ目：0 と 1 の紙テープを機械が読み書きする。ナレーション：「複雑な計算も小さな動作に分けられる」。
4コマ目：1950年の Turing Test。審査員が画面越しに人間と機械へ質問。吹き出し：「どちらが機械か分からない」。
5コマ目：ラベル「成功した点：機械の知能をテスト可能にした」。ラベル「課題：会話できても理解とは限らない」。
6コマ目：未来のロボットと疑問符。矢印：「次へ：AI が正式な研究分野になる」。

下部に一文：「この段階の歴史的意義：AI の核心問題『機械は知的に振る舞えるか』が正式に問われた」。
文字はすべて自然な日本語。短文で、吹き出し、小黒板、キャプション、ラベルに自然に入れる。文字を一か所に詰めない。文字化け、透かし、実在ロゴは禁止。
""".strip(),
        },
        {
            "filename": "homepage-ai-history-comic-ja-02-dartmouth.png",
            "size": "1024x1792",
            "quality": "high",
            "title": "日本語ホーム AI 史マンガ 02 ダートマス会議",
            "suggested_page": "i18n/ja/docusaurus-plugin-content-docs/current/index.md",
            "alt": "1956年のダートマス会議で人工知能が正式な研究分野になったことを説明する日本語マンガ。",
            "prompt": """
9:16 縦長の日本語科学マンガページを作成。タイトル：「ダートマス会議：AI 誕生 1956」。
1950年代の学術会議の空気、温かい紙質、半写実人物、6コマの本格マンガ構成。

1コマ目：科学者が会議机を囲む。名札：John McCarthy、Marvin Minsky、Claude Shannon、Allen Newell、Herbert Simon。
2コマ目：McCarthy が黒板を指し、黒板に「Artificial Intelligence」。吹き出し：「この夢に名前を付けよう」。
3コマ目：機械が「IF A THEN B」の規則カードで数学定理を証明。説明：「初期 AI は論理推論が得意」。
4コマ目：フローチャート「規則 → 推論 → 結論」。小黒板：「知識を規則として書く」。
5コマ目：ラベル「成功した点：AI が正式な研究分野になった」。ラベル「課題：人間の知能を全部規則にするのは難しい」。
6コマ目：規則カードが高く積み上がる。矢印：「次へ：学習する機械を探す」。

下部に一文：「この段階の歴史的意義：知的な機械を作ることが科学の正式な目標になった」。
文字は自然な日本語で短く、吹き出し、黒板、ラベルに分散。文字化け、透かし、実在ロゴは禁止。
""".strip(),
        },
        {
            "filename": "homepage-ai-history-comic-ja-03-perceptron.png",
            "size": "1024x1792",
            "quality": "high",
            "title": "日本語ホーム AI 史マンガ 03 パーセプトロンと最初の冬",
            "suggested_page": "i18n/ja/docusaurus-plugin-content-docs/current/index.md",
            "alt": "パーセプトロン、XOR問題、ニューラルネットワーク最初の低迷を説明する日本語マンガ。",
            "prompt": """
9:16 縦長の日本語科学マンガページを作成。タイトル：「パーセプトロンの熱狂と最初の冬 1957-1969」。
レトロな研究所、半写実人物、6コマの本格マンガページ。

1コマ目：Frank Rosenblatt が Perceptron 機械を紹介。吹き出し：「データから重みを変えられる！」
2コマ目：小黒板に人工ニューロン：x1、x2、w1、w2、合計、しきい値、出力。説明：「点数がしきい値を超えたら 1」。
3コマ目：1本の直線で2種類の点を分ける図。説明：「線形に分けられる問題は得意」。
4コマ目：Marvin Minsky と Seymour Papert が XOR の4点を黒板に描く。吹き出し：「1本の線では分けられない」。
5コマ目：ラベル「成功した点：データで重みを学習」。ラベル「課題：単層では表現力が弱い」。
6コマ目：研究所のニューラルネットの明かりが暗くなる。矢印：「次へ：ルール型 AI が主役へ」。

下部に一文：「この段階の歴史的意義：ニューラルネットは希望を見せ、同時に限界も見せた」。
文字は自然な日本語で短く、技術図と会話に分散。文字化け、透かし、実在ロゴは禁止。
""".strip(),
        },
        {
            "filename": "homepage-ai-history-comic-ja-04-expert-systems.png",
            "size": "1024x1792",
            "quality": "high",
            "title": "日本語ホーム AI 史マンガ 04 エキスパートシステム",
            "suggested_page": "i18n/ja/docusaurus-plugin-content-docs/current/index.md",
            "alt": "エキスパートシステム、知識ベース、規則推論、保守限界を説明する日本語マンガ。",
            "prompt": """
9:16 縦長の日本語科学マンガページを作成。タイトル：「エキスパートシステム：規則の栄光と限界 1970s-1980s」。
病院、化学研究室、企業システムが出るレトロな 6コマ漫画。

1コマ目：Edward Feigenbaum と Bruce Buchanan が医師や化学者から知識を聞く。説明：「専門家の知識を機械へ」。
2コマ目：医師が規則カードを機械へ入れる。カード：「IF 発熱 + 感染 THEN 診断候補」。
3コマ目：知識ベースと推論エンジンの図。黒板：「知識ベース + 推論 = Expert System」。
4コマ目：医学、化学、企業設定で役立つ場面。吹き出し：「狭い分野なら強い！」。
5コマ目：ラベル「成功した点：専門領域で実用化」。ラベル「課題：規則が衝突し、保守が大変」。
6コマ目：規則カードが山になり機械が煙を出す。矢印：「次へ：データから学ぶ方法へ」。

下部に一文：「この段階の歴史的意義：狭い領域では AI が役立つ一方、手書き規則は世界全体に広がらなかった」。
文字は自然な日本語で短く、分散配置。文字化け、透かし、実在ロゴは禁止。
""".strip(),
        },
        {
            "filename": "homepage-ai-history-comic-ja-05-backprop.png",
            "size": "1024x1792",
            "quality": "high",
            "title": "日本語ホーム AI 史マンガ 05 誤差逆伝播",
            "suggested_page": "i18n/ja/docusaurus-plugin-content-docs/current/index.md",
            "alt": "誤差逆伝播で多層ニューラルネットの重みを調整する仕組みを説明する日本語マンガ。",
            "prompt": """
9:16 縦長の日本語科学マンガページを作成。タイトル：「誤差逆伝播：神経ネットが再点火 1986」。
教室の黒板と研究所を組み合わせた 6コマの本格科学マンガ。

1コマ目：Geoffrey Hinton、David Rumelhart、Ronald Williams が多層ネットの黒板前に立つ。吹き出し：「多層をどう訓練する？」
2コマ目：入力が前へ流れ、間違った予測を出す。説明：「まず誤差を測る」。
3コマ目：赤い誤差信号が出力層から左へ戻る。キャプション：「誤差を後ろへ伝える」。
4コマ目：各層が重みのつまみを少し回す。黒板：「パラメータ = パラメータ - 学習率 x 勾配」。
5コマ目：ラベル「成功した点：各層の直し方が分かった」。ラベル「課題：当時はデータも計算力も不足」。
6コマ目：ネットは光るが、遠くに深いネットの山。矢印：「次へ：画像認識で実用成功へ」。

下部に一文：「この段階の歴史的意義：誤差逆伝播は後の深層学習の訓練基盤になった」。
文字は自然な日本語。式は読みやすく、文字化け、透かし、実在ロゴは禁止。
""".strip(),
        },
        {
            "filename": "homepage-ai-history-comic-ja-06-lenet.png",
            "size": "1024x1792",
            "quality": "high",
            "title": "日本語ホーム AI 史マンガ 06 LeNet と CNN",
            "suggested_page": "i18n/ja/docusaurus-plugin-content-docs/current/index.md",
            "alt": "LeNet、畳み込み、手書き数字認識、CNNの実用成功を説明する日本語マンガ。",
            "prompt": """
9:16 縦長の日本語科学マンガページを作成。タイトル：「LeNet と CNN の実用成功 1989-1998」。
1990年代の研究室、郵便番号と小切手認識、6コマの本格マンガ。

1コマ目：Yann LeCun、Léon Bottou、Yoshua Bengio、Patrick Haffner が手書き数字を見る。吹き出し：「機械に手書きを読ませたい」。
2コマ目：小さなフィルタが虫眼鏡のように数字画像を滑る。説明：「畳み込み：近くを少しずつ見る」。
3コマ目：層ごとに「線 → 筆画 → 数字の形」が見える。キャプション：「低層は線、高層は数字」。
4コマ目：機械が出力：「これは 8」。郵便や小切手処理に使われる。
5コマ目：ラベル「成功した点：実産業でニューラルネットが働いた」。ラベル「課題：まだ狭いタスク中心」。
6コマ目：猫、犬、街の写真が遠くに見える。矢印：「次へ：大規模データと GPU へ」。

下部に一文：「この段階の歴史的意義：CNN は画像特徴を自動で学ぶ力を示した」。
文字は自然な日本語。CNN、LeNet は用語として残す。文字化け、透かし、実在ロゴは禁止。
""".strip(),
        },
        {
            "filename": "homepage-ai-history-comic-ja-07-statistical-ml.png",
            "size": "1024x1792",
            "quality": "high",
            "title": "日本語ホーム AI 史マンガ 07 統計的機械学習",
            "suggested_page": "i18n/ja/docusaurus-plugin-content-docs/current/index.md",
            "alt": "統計的機械学習、SVM、ランダムフォレスト、Boosting、特徴量設計を説明する日本語マンガ。",
            "prompt": """
9:16 縦長の日本語科学マンガページを作成。タイトル：「統計的機械学習：データが規則に代わる 1990s-2000s」。
データ研究室と初期インターネット企業の雰囲気、6コマ漫画。

1コマ目：Vladimir Vapnik、Leo Breiman、Jerome Friedman がアルゴリズム黒板の前に立つ。説明：「規則を書く代わりに、データからパターンを学ぶ」。
2コマ目：表データ、クリック、文章の単語数、ラベルがモデルへ流れる。キャプション：「データが多いほど規則性が見える」。
3コマ目：エンジニアが特徴量カードを作る：「色」「エッジ」「単語数」「クリック数」。吹き出し：「まず情報を整理する」。
4コマ目：SVM、Random Forest、Boosting の小型マシンが予測を出す。
5コマ目：ラベル「成功した点：表データ、検索、広告、分類に強い」。ラベル「課題：人手の特徴量設計に依存」。
6コマ目：画像、音声、言語の山。矢印：「次へ：特徴もモデルが学ぶ」。

下部に一文：「この段階の歴史的意義：AI は手書き規則からデータ学習へ移ったが、人手の特徴設計は残った」。
文字は自然な日本語。SVM、Boosting は用語として残す。文字化け、透かし、実在ロゴは禁止。
""".strip(),
        },
        {
            "filename": "homepage-ai-history-comic-ja-08-imagenet-alexnet.png",
            "size": "1024x1792",
            "quality": "high",
            "title": "日本語ホーム AI 史マンガ 08 ImageNet と AlexNet",
            "suggested_page": "i18n/ja/docusaurus-plugin-content-docs/current/index.md",
            "alt": "ImageNet、AlexNet、GPU、CNN、大規模画像認識による深層学習の爆発を説明する日本語マンガ。",
            "prompt": """
9:16 縦長の日本語科学マンガページを作成。タイトル：「ImageNet と AlexNet：深層学習の爆発 2009-2012」。
現代的な研究室、巨大画像図書館、コンペ会場を含む 6コマ漫画。

1コマ目：Fei-Fei Li が巨大な画像図書館 ImageNet を作る。説明：「機械に大量の画像を見せる」。
2コマ目：Alex Krizhevsky、Ilya Sutskever、Geoffrey Hinton が GPU マシンで AlexNet を訓練。
3コマ目：CNN の層が「エッジ → 部品 → 猫や犬」を学ぶ図。
4コマ目：コンペ順位表で AlexNet が古い方法を大きく上回る。吹き出し：「生画像から特徴を学べる！」。
5コマ目：ラベル「成功した点：大データ + GPU + CNN が深層学習を加速」。ラベル「課題：計算資源が大量に必要」。
6コマ目：さらに深いネットの塔。矢印：「次へ：深すぎるネットをどう訓練する？」。

下部に一文：「この段階の歴史的意義：2012年は深層学習が本格的に爆発した転換点になった」。
文字は自然な日本語。ImageNet、AlexNet、GPU、CNN は用語として残す。文字化け、透かし、実在ロゴは禁止。
""".strip(),
        },
        {
            "filename": "homepage-ai-history-comic-ja-09-resnet.png",
            "size": "1024x1792",
            "quality": "high",
            "title": "日本語ホーム AI 史マンガ 09 ResNet",
            "suggested_page": "i18n/ja/docusaurus-plugin-content-docs/current/index.md",
            "alt": "ResNetの残差接続と X を足し戻す意味を説明する日本語マンガ。",
            "prompt": """
9:16 縦長の日本語科学マンガページを作成。タイトル：「ResNet：なぜ X を足し戻すのか 2015」。
工学的で見やすい 6コマの科学マンガ。

1コマ目：Kaiming He、Xiangyu Zhang、Shaoqing Ren、Jian Sun が深いネットの塔を見上げる。吹き出し：「深くすると逆に訓練しにくい？」。
2コマ目：普通の信号が階段を登るうちに弱くなる。説明：「深い層では劣化や勾配の問題が起きる」。
3コマ目：情報高速道路が複雑な層をまたいで X を先へ運ぶ。
4コマ目：黒板：「出力 = X + F(X)」。説明：「元の情報を残し、修正量だけ学ぶ」。
5コマ目：ラベル「成功した点：50層、101層、152層が訓練可能に」。ラベル「課題：大きなモデルは計算コストが高い」。
6コマ目：残差の橋が未来の Transformer へつながる。矢印：「次へ：系列と言語モデルへ」。

下部に一文：「この段階の歴史的意義：残差接続は深層学習の基本部品となり、Transformer にも受け継がれた」。
文字は自然な日本語。式は明確に。文字化け、透かし、実在ロゴは禁止。
""".strip(),
        },
        {
            "filename": "homepage-ai-history-comic-ja-10-rnn-lstm.png",
            "size": "1024x1792",
            "quality": "high",
            "title": "日本語ホーム AI 史マンガ 10 RNN と LSTM",
            "suggested_page": "i18n/ja/docusaurus-plugin-content-docs/current/index.md",
            "alt": "RNN、LSTM、hidden state、記憶ゲート、長距離依存の限界を説明する日本語マンガ。",
            "prompt": """
9:16 縦長の日本語科学マンガページを作成。タイトル：「RNN と LSTM：系列モデルの主役 1997-2014」。
言語の流れと時間軸を見せる 6コマ漫画。

1コマ目：Sepp Hochreiter と Jürgen Schmidhuber が LSTM の記憶ゲートを示す。説明：「文章、音声、時系列には順番がある」。
2コマ目：RNN をリレー隊として描く。単語ごとに記憶バトンを渡す。キャプション：「履歴は hidden state に入る」。
3コマ目：長い文ではバトンが薄くなる。吹き出し：「遠い情報が弱くなる」。
4コマ目：LSTM が 3つの門「覚える」「忘れる」「出す」を持つ。黒板：「ゲートは情報のスイッチ」。
5コマ目：ラベル「成功した点：文章、音声、翻訳を扱えた」。ラベル「課題：順番処理で遅く、長距離依存が難しい」。
6コマ目：重要語にスポットライトが当たる。矢印：「次へ：Attention が必要な場所を見る」。

下部に一文：「この段階の歴史的意義：RNN/LSTM は系列処理を実用化し、長い記憶と並列化の壁も明らかにした」。
文字は自然な日本語。RNN、LSTM、hidden state は用語として残す。文字化け、透かし、実在ロゴは禁止。
""".strip(),
        },
        {
            "filename": "homepage-ai-history-comic-ja-11-attention.png",
            "size": "1024x1792",
            "quality": "high",
            "title": "日本語ホーム AI 史マンガ 11 Attention",
            "suggested_page": "i18n/ja/docusaurus-plugin-content-docs/current/index.md",
            "alt": "Attention、翻訳、情報ボトルネック、重み付き注目を説明する日本語マンガ。",
            "prompt": """
9:16 縦長の日本語科学マンガページを作成。タイトル：「Attention：丸暗記せず重点を見る 2014」。
翻訳教室とスポットライトの比喩を使う 6コマ漫画。

1コマ目：Dzmitry Bahdanau、Kyunghyun Cho、Yoshua Bengio が翻訳黒板の横に立つ。吹き出し：「1つのベクトルに全文を詰めるのは苦しい」。
2コマ目：古い Seq2Seq が長文を小さな瓶に押し込む。説明：「情報ボトルネック」。
3コマ目：翻訳ロボットが日本語単語を出す時、元の英単語へスポットライト。
4コマ目：重みバー：「重要語 70%」「別の語 20%」。説明：「重みで見る場所を決める」。
5コマ目：ラベル「成功した点：出力ごとに関連する入力位置を見られる」。ラベル「課題：初期 Attention は RNN 併用が多く、並列化が弱い」。
6コマ目：RNN の鎖が外れ始める。矢印：「次へ：Transformer は Attention だけで進む」。

下部に一文：「この段階の歴史的意義：モデルは全文を丸暗記するのではなく、必要な時に重点を見るようになった」。
文字は自然な日本語。Attention、Seq2Seq は用語として残す。文字化け、透かし、実在ロゴは禁止。
""".strip(),
        },
        {
            "filename": "homepage-ai-history-comic-ja-12-transformer.png",
            "size": "1024x1792",
            "quality": "high",
            "title": "日本語ホーム AI 史マンガ 12 Transformer",
            "suggested_page": "i18n/ja/docusaurus-plugin-content-docs/current/index.md",
            "alt": "Transformer、self-attention、QKV、multi-head attention、並列訓練を説明する日本語マンガ。",
            "prompt": """
9:16 縦長の日本語科学マンガページを作成。タイトル：「Transformer：Attention Is All You Need 2017」。
現代的な技術感のある 6コマ漫画。文字は短く、図で説明する。

1コマ目：著者チームが論文黒板の前に立つ。名札：Vaswani、Shazeer、Parmar、Uszkoreit、Jones、Gomez、Kaiser、Polosukhin。
2コマ目：単語 token が会議テーブルを囲み、全員が直接つながる。吹き出し：「順番に伝言しなくていい」。
3コマ目：Q/K/V カード。Q「何を探す？」 K「どんな札を持つ？」 V「渡す情報」。
4コマ目：複数の専門家ヘッドが文を見る：「文法」「指示語」「意味」。説明：「Multi-Head = 複数の見方」。
5コマ目：ラベル「成功した点：並列訓練と長距離依存に強い」。ラベル「課題：長い文では計算量が増える」。
6コマ目：Transformer が大規模言語モデルの街の土台になる。矢印：「次へ：大規模事前学習モデルへ」。

下部に一文：「この段階の歴史的意義：Transformer は現代の大規模言語モデルの中核構造になった」。
文字は自然な日本語。Q/K/V、token、Transformer は用語として残す。文字化け、透かし、実在ロゴは禁止。
""".strip(),
        },
        {
            "filename": "homepage-ai-history-comic-ja-13-bert-gpt.png",
            "size": "1024x1792",
            "quality": "high",
            "title": "日本語ホーム AI 史マンガ 13 BERT と GPT",
            "suggested_page": "i18n/ja/docusaurus-plugin-content-docs/current/index.md",
            "alt": "BERT、GPT、穴埋め学習、次トークン予測、事前学習時代を説明する日本語マンガ。",
            "prompt": """
9:16 縦長の日本語科学マンガページを作成。タイトル：「BERT と GPT：事前学習時代の始まり 2018-2020」。
2つのモデルの違いを横並びで見せる 6コマ漫画。

1コマ目：Jacob Devlin と Alec Radford が Transformer の土台の前に立つ。説明：「まず巨大な文章で汎用モデルを訓練」。
2コマ目：BERT は読解学生。穴あき単語を左右の文脈から当てる。キャプション：「両側を見て穴埋め」。
3コマ目：GPT は文章を書くロボット。左から右へ続きを書く。キャプション：「前文から次の token を予測」。
4コマ目：GPT-3 が少数例を見てタスクをまねる。吹き出し：「例を見れば型を続けられる」。
5コマ目：ラベル「成功した点：事前学習が多くのタスクへ転移」。ラベル「課題：幻覚と事実の不安定さ」。
6コマ目：文章補完ロボットが指示訓練の教室へ入る。矢印：「次へ：指示に従うモデルへ」。

下部に一文：「この段階の歴史的意義：AI はタスクごとの個別モデルから、大規模な基盤モデルへ移った」。
文字は自然な日本語。BERT、GPT、GPT-3、token は用語として残す。文字化け、透かし、実在ロゴは禁止。
""".strip(),
        },
        {
            "filename": "homepage-ai-history-comic-ja-14-rlhf-chatgpt.png",
            "size": "1024x1792",
            "quality": "high",
            "title": "日本語ホーム AI 史マンガ 14 SFT RLHF と ChatGPT",
            "suggested_page": "i18n/ja/docusaurus-plugin-content-docs/current/index.md",
            "alt": "SFT、RLHF、ChatGPT、文章補完モデルが対話助手になる流れを説明する日本語マンガ。",
            "prompt": """
9:16 縦長の日本語科学マンガページを作成。タイトル：「SFT、RLHF、ChatGPT：補完器から助手へ 2022」。
訓練教室とチャット画面を組み合わせた 6コマ漫画。

1コマ目：文章補完ロボットが長い文章を勝手に続ける。吹き出し：「私は続きを書くだけ」。
2コマ目：人間教師が「ユーザー質問 → 助手回答」の例を渡す。黒板：「SFT：良い回答例で練習」。
3コマ目：人間評価者が複数回答を比べて順位をつける。説明：「RLHF：人が好む回答を学ぶ」。
4コマ目：ロボットが対話助手になり、指示に沿って段階的に答える。
5コマ目：ラベル「成功した点：一般ユーザーが AI 対話を体験」。ラベル「課題：幻覚、過信、事実誤り」。
6コマ目：知識ベース、道具箱、安全標識が助手の横に出る。矢印：「次へ：知識と道具につなぐ」。

下部に一文：「この段階の歴史的意義：大規模言語モデルは文章補完から、使える会話助手へ近づいた」。
文字は自然な日本語。SFT、RLHF、ChatGPT は用語として残す。文字化け、透かし、実在ロゴは禁止。
""".strip(),
        },
        {
            "filename": "homepage-ai-history-comic-ja-15-rag-agent.png",
            "size": "1024x1792",
            "quality": "high",
            "title": "日本語ホーム AI 史マンガ 15 RAG と Agent",
            "suggested_page": "i18n/ja/docusaurus-plugin-content-docs/current/index.md",
            "alt": "RAG、ツール呼び出し、AI Agent、計画、実行、安全性を説明する日本語マンガ。",
            "prompt": """
9:16 縦長の日本語科学マンガページを作成。タイトル：「RAG、ツール呼び出し、Agent：実タスクへ 2023-現在」。
現代的な AI 作業場、知識ベース、道具箱、タスクボードを描く 6コマ漫画。

1コマ目：ユーザーが AI 助手へ依頼：「講義資料を作って」。AI が目標を受け取る。
2コマ目：RAG の流れを図解：「質問 → 資料検索 → 文脈に入れる → 根拠つき回答」。説明：「先に探し、資料に基づいて答える」。
3コマ目：AI が道具を呼ぶ：検索、コード、計算機、文書生成。キャプション：「できないことは道具を使う」。
4コマ目：Agent のタスクボード：「目標 → 計画 → 道具実行 → 観察 → 続行」。
5コマ目：ラベル「成功した点：質問回答から作業完了へ」。ラベル「課題：検索ミス、道具エラー、プロンプト注入、権限リスク」。
6コマ目：安全ガード、権限チェック、人間レビューがワークフローを守る。矢印：「次へ：より信頼できる長時間自律システムへ」。

下部に一文：「この段階の歴史的意義：AI は話すだけでなく、検索し、道具を使い、実際の仕事を進め始めた」。
文字は自然な日本語。RAG、Agent は用語として残す。文字化け、透かし、実在ロゴは禁止。
""".strip(),
        },
    ]
)


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


def write_generation_errors(report_dir: Path, errors: list[dict[str, str]]) -> None:
    if not errors:
        return
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
    try:
        from openai import OpenAI
    except ImportError as exc:
        print("The Python package `openai` is not installed; using the built-in HTTP fallback.", flush=True)
    else:
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=args.base_url, timeout=args.request_timeout)

    errors: list[dict[str, str]] = []
    for job in jobs:
        output_path = output_dir / job["filename"]
        if output_path.exists() and output_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") and not args.overwrite:
            print(f"Skipping existing valid PNG: {job['filename']} (use --overwrite to regenerate)", flush=True)
            continue
        print(f"Generating {job['filename']}...", flush=True)
        try:
            if client:
                result = client.images.generate(
                    model=args.model,
                    prompt=job["prompt"],
                    **({} if job.get("size") == "default" else {"size": job["size"]}),
                    **({} if job.get("quality") == "default" else {"quality": job["quality"]}),
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
                        retries=args.retries,
                        request_timeout=args.request_timeout,
                    )
                )
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
