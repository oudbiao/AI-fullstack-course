
# 🔧 环境准备

> **目标：** 确认你的电脑满足要求，了解需要安装的工具  
> **说明：** 这里只做概览。每个工具的详细安装步骤在第零阶段的对应章节里有手把手教程。

---

## 硬件要求

### 前四个阶段（Python + 数据分析 + 数学 + ML）

任何能正常使用的电脑都行：

| 配置项 | 最低要求 | 推荐配置 |
|-------|---------|---------|
| CPU | 任意双核 | 4 核以上 |
| 内存 | 4GB | 8GB 以上 |
| 硬盘 | 20GB 可用空间 | SSD，50GB 可用 |
| GPU | **不需要** | 不需要 |
| 操作系统 | Windows 10/11、macOS 10.15+、Ubuntu 20.04+ | 均可 |

:::tip
如果你的电脑很老，也不用担心。前四个阶段的所有代码都可以在 [Google Colab](https://colab.research.google.com) 上运行，只需要一个浏览器。
:::

### 第五阶段开始（深度学习）

从第五阶段开始训练神经网络，需要 GPU：

| 方案 | 说明 | 费用 | 推荐度 |
|------|------|------|:---:|
| **Google Colab** | 免费 T4 GPU，零配置 | 免费（Pro 版 $10/月） | ⭐⭐⭐⭐⭐ |
| **AutoDL** | 国内云 GPU，按小时计费 | 约 2-3 元/小时 | ⭐⭐⭐⭐ |
| **本地 NVIDIA GPU** | 显存 ≥ 8GB | 一次性投入 | ⭐⭐⭐ |

:::info 不用提前买 GPU
学完前四个阶段大约需要 4-6 个月。到了第五阶段再考虑 GPU 的事。课程在进入第五阶段前有详细的[硬件准备指南](/stage5/hardware-prep)。
:::

---

## 软件清单

以下是整个课程需要用到的软件，分阶段列出。**现在只需要安装前两项**，其他的到了对应阶段再装。

### 现在就需要安装的（第零阶段教你怎么装）

| 软件 | 是什么 | 为什么需要 |
|------|-------|----------|
| **Python 3.10+** | 编程语言 | 所有代码都用 Python 写 |
| **VS Code** | 代码编辑器 | 写代码、调试、查看文件 |
| **Git** | 版本管理工具 | 管理代码、上传 GitHub |
| **Miniconda** | Python 环境管理 | 创建隔离的虚拟环境，避免包冲突 |

### 第一阶段需要的

| 软件/库 | 用途 |
|---------|------|
| `requests` | 发送 HTTP 请求（爬虫、API 调用） |
| `beautifulsoup4` | 解析 HTML（爬虫） |
| `fastapi` + `uvicorn` | Web API 开发 |

### 第二阶段需要的

| 软件/库 | 用途 |
|---------|------|
| **Jupyter Notebook** | 交互式编程环境（数据分析标配） |
| `numpy` | 科学计算 |
| `pandas` | 数据处理 |
| `matplotlib` + `seaborn` | 数据可视化 |

### 第五阶段需要的

| 软件/库 | 用途 |
|---------|------|
| `torch`（PyTorch） | 深度学习框架 |
| `torchvision` | 图像相关工具 |
| CUDA Toolkit（本地 GPU 用户） | GPU 加速 |

### 后续阶段按需安装的

| 软件/库 | 阶段 | 用途 |
|---------|------|------|
| `transformers` | 第七/八A | HuggingFace 预训练模型 |
| `langchain` | 第八B/九 | 大模型应用开发框架 |
| `docker` | 第八B | 容器化部署 |
| `chromadb` / `faiss` | 第八B | 向量数据库 |
| `openai` / `anthropic` | 第八B | 大模型 API 调用 |

---

## Python 版本选择

**推荐 Python 3.11**。原因：

- 3.11 比 3.10 快 10-60%
- 目前所有主流 AI 库都兼容 3.11
- 3.12/3.13 太新，部分库可能还没适配

:::warning 不要用 Python 3.8 或更低版本
很多新版本的 AI 库已经不再支持 3.8/3.9。如果你电脑上已经有旧版 Python，不需要卸载，用 Miniconda 创建一个新的 3.11 环境就行（第零阶段会教你怎么做）。
:::

---

## 操作系统相关说明

### Windows 用户

- 推荐安装 **Windows Terminal**（Windows 11 自带，Windows 10 去微软商店下载）
- 命令行推荐用 **PowerShell** 或 **Git Bash**
- 如果遇到 Python 包安装问题，优先考虑用 Miniconda

### macOS 用户

- 推荐安装 **Homebrew** 包管理器
- macOS 自带 Python 2，不要用它。通过 Miniconda 安装 Python 3.11
- Apple Silicon（M1/M2/M3）的 PyTorch 支持已经很好，可以用 MPS 加速

### Linux（Ubuntu）用户

- 大部分 AI 工具对 Linux 支持最好
- 推荐 Ubuntu 22.04 LTS
- NVIDIA GPU 驱动安装可能需要一些额外步骤（第零阶段会覆盖）

---

## 网络环境

有些资源需要科学上网：

| 资源 | 是否需要科学上网 | 替代方案 |
|------|:---:|---------|
| Google Colab | 需要 | AutoDL、本地 Jupyter |
| GitHub | 部分地区需要 | Gitee 作为镜像 |
| HuggingFace | 部分地区需要 | HuggingFace 镜像站 |
| PyPI（pip 源） | 不需要，但国外源慢 | 使用清华/阿里镜像 |
| OpenAI API | 需要 | 国内大模型 API（通义千问、DeepSeek） |

### 配置 pip 国内镜像（推荐）

如果你在国内，pip 安装包会很慢。运行以下命令一劳永逸地配好清华镜像：

```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

配好后所有 `pip install` 都会从清华镜像下载，速度飞快。

---



:::tip 遇到环境问题不要怕
配环境是每个开发者都要经历的"痛苦"。如果你卡住了：
1. 把错误信息完整地复制下来
2. 粘贴到 Google 搜索（英文搜索效果更好）
3. 99% 的环境问题都有人遇到过，Stack Overflow 上一定有答案
4. 实在搞不定，先用 Google Colab 继续学习，环境问题以后再解决
:::