---
title: "Python 环境管理"
sidebar_position: 1
description: "用 Miniconda 管理 Python 版本和虚拟环境，从根源上避免包冲突"
---

# Python 环境管理

## 学习目标

- 理解为什么需要虚拟环境（通过一个真实的翻车案例）
- 安装和配置 Miniconda
- 掌握虚拟环境的创建、激活、切换、删除
- 理解 conda 和 pip 的区别，知道什么时候用哪个
- 学会导出和导入环境配置
- 能独立排查常见环境问题

---

## 为什么需要虚拟环境？

### 一个真实的翻车场景

小明在做两个 AI 项目：

- **项目 A**（图像分类）：需要 `torch==1.13`，因为用了一个只兼容 1.13 的旧库
- **项目 B**（大模型应用）：需要 `torch==2.1`，因为用了最新的 Flash Attention

如果他把两个项目的依赖都装在同一个 Python 里：

```bash
pip install torch==1.13    # 项目 A 能跑了
pip install torch==2.1     # 项目 B 能跑了，但 torch 被升级到 2.1
# 这时候回去跑项目 A —— 报错了！因为 torch 已经变成 2.1 了
```

这就是**包版本冲突**——一个 Python 环境只能装一个版本的同名包。

### 虚拟环境怎么解决这个问题？

虚拟环境 = 一个**独立的、隔离的 Python 安装**。每个项目一个环境，互不干扰：

```
项目 A 的环境：Python 3.10 + torch 1.13 + ...
项目 B 的环境：Python 3.11 + torch 2.1 + ...
```

切换项目时，切换环境就行。两边互不影响。

### 类比

把虚拟环境想象成手机上的**多用户/工作空间**。每个用户有自己独立安装的 App，互不干扰。你可以在"工作用户"里装钉钉，在"个人用户"里装游戏，它们之间完全隔离。

---

## 安装 Miniconda

### 为什么选 Miniconda？

| 工具 | 说明 | 推荐度 |
|------|------|:---:|
| **Miniconda** | 轻量级，只装核心组件，按需安装其他包 | ⭐⭐⭐⭐⭐ |
| Anaconda | 全家桶，预装 250+ 个包，占用 3GB+ | ⭐⭐⭐ |
| venv + pip | Python 自带，轻量但功能较少 | ⭐⭐⭐ |

Miniconda 是最佳选择：够轻量、能管理 Python 版本、能创建虚拟环境、AI 社区广泛使用。

### macOS 安装

```bash
# 下载安装脚本（Apple Silicon / M1/M2/M3）
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh

# Intel Mac 用这个
# curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh

# 运行安装
bash Miniconda3-latest-MacOSX-arm64.sh
```

安装过程中：
- 阅读协议：按 `q` 跳过，输入 `yes` 同意
- 安装路径：直接回车用默认路径
- 是否初始化：输入 `yes`

安装完成后，**关闭终端再重新打开**。

### Ubuntu/Linux 安装

```bash
# 下载
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# 安装
bash Miniconda3-latest-Linux-x86_64.sh

# 按提示操作，最后选 yes 初始化
```

关闭终端再重新打开。

### Windows 安装

1. 下载安装包：[Miniconda Windows 安装程序](https://docs.conda.io/en/latest/miniconda.html)
2. 双击运行，一路 Next
3. **勾选** "Add Miniconda3 to my PATH environment variable"（方便在 PowerShell 中使用）
4. 安装完成后重启 PowerShell

### 验证安装

```bash
conda --version
# 输出类似: conda 24.x.x

python --version
# 输出类似: Python 3.12.x
```

看到版本号就说明安装成功了。

### 配置国内镜像（国内用户强烈推荐）

默认的 conda 源在国外，下载速度很慢。配置清华镜像：

```bash
# 添加清华镜像
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch
conda config --set show_channel_urls yes
```

同时配置 pip 的清华镜像（如果之前没配过）：

```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## 虚拟环境操作

### 创建环境

```bash
# 创建一个名叫 ai-basic 的环境，Python 版本 3.11
conda create -n ai-basic python=3.11

# conda 会列出要安装的包，输入 y 确认
```

`-n ai-basic` 是环境的名字，你可以随便起，建议用项目名或用途命名。

### 激活环境

```bash
conda activate ai-basic
```

激活后，你的终端提示符前面会出现环境名：

```
(ai-basic) zhangsan@MacBook ~ $
```

这表示你现在在 `ai-basic` 环境里。在这个环境里安装的任何包，都只属于这个环境。

### 在环境里安装包

```bash
# 确认当前环境
conda info --envs
# 带 * 号的就是当前激活的环境

# 用 pip 安装包（推荐大部分情况使用 pip）
pip install numpy pandas matplotlib

# 用 conda 安装包（某些特殊的包用 conda 更好）
conda install scipy

# 查看当前环境安装了哪些包
pip list
# 或
conda list
```

### 案例：为不同项目创建不同环境

```bash
# 项目 A：传统机器学习
conda create -n ml-project python=3.11
conda activate ml-project
pip install numpy pandas scikit-learn matplotlib seaborn jupyter

# 项目 B：深度学习
conda create -n dl-project python=3.11
conda activate dl-project
pip install torch torchvision numpy matplotlib tensorboard

# 项目 C：大模型应用
conda create -n llm-project python=3.11
conda activate llm-project
pip install openai langchain chromadb fastapi
```

三个项目，三个独立的环境，互不干扰。

### 切换环境

```bash
# 切换到 ml-project 环境
conda activate ml-project

# 切换到 dl-project 环境
conda activate dl-project

# 退出当前环境（回到 base 环境）
conda deactivate
```

### 查看所有环境

```bash
conda env list
# 或
conda info --envs
```

输出：

```
# conda environments:
#
base                     /Users/zhangsan/miniconda3
ai-basic                 /Users/zhangsan/miniconda3/envs/ai-basic
ml-project            *  /Users/zhangsan/miniconda3/envs/ml-project
dl-project               /Users/zhangsan/miniconda3/envs/dl-project
llm-project              /Users/zhangsan/miniconda3/envs/llm-project
```

`*` 号表示当前激活的环境。

### 删除环境

```bash
# 删除一个不再需要的环境
conda env remove -n ai-basic

# 确认已删除
conda env list
```

---

## conda install vs pip install

这是新手最常问的问题。简单的原则：

| 情况 | 用什么 | 原因 |
|------|-------|------|
| 大部分 Python 包 | `pip install` | pip 的包最全，更新最快 |
| CUDA 相关的包 | `conda install` | conda 能自动处理 CUDA 依赖 |
| 系统级的库（如 MKL） | `conda install` | pip 装不了系统级的库 |
| 不确定用哪个 | 先试 `pip install` | pip 更通用 |

:::warning 一个重要原则
在同一个环境里，**尽量不要混用** conda install 和 pip install 安装同一个包。如果你用 pip 装了 numpy，就不要再用 conda 装一次 numpy。混用可能导致版本混乱。

推荐做法：在 conda 环境里，优先用 pip 安装所有 Python 包。
:::

---

## 导出和导入环境

### 场景：分享你的项目环境

你做完一个项目，想让同事（或未来的自己）能快速搭建相同的环境。

#### 方式一：pip freeze（最常用）

```bash
# 导出当前环境的所有包到 requirements.txt
pip freeze > requirements.txt
```

`requirements.txt` 长这样：

```
numpy==1.26.4
pandas==2.2.0
scikit-learn==1.4.0
matplotlib==3.8.2
torch==2.1.2
```

别人拿到后，一行命令恢复：

```bash
# 创建新环境
conda create -n restored-env python=3.11
conda activate restored-env

# 安装所有依赖
pip install -r requirements.txt
```

#### 方式二：conda env export

```bash
# 导出完整环境（包括 conda 和 pip 安装的包）
conda env export > environment.yml
```

恢复：

```bash
conda env create -f environment.yml
```

#### 该用哪种？

| 文件 | 适合场景 | 优点 | 缺点 |
|------|---------|------|------|
| `requirements.txt` | 大部分项目 | 简单、通用、跨平台 | 不含 Python 版本信息 |
| `environment.yml` | 包含 conda 特殊包的项目 | 完整、含 Python 版本 | 可能有平台差异 |

**建议：** 每个项目里都放一个 `requirements.txt`，这是 Python 社区的标准做法。

---

## 常见问题排查

### 问题1：`conda activate` 不生效

```
CommandNotFoundError: Your shell has not been properly configured to use 'conda activate'.
```

解决：

```bash
# 初始化 conda（根据你的 shell 选择）
conda init zsh     # macOS 默认
conda init bash    # Linux 默认

# 然后重启终端
```

### 问题2：`command not found: python`

安装了 Miniconda 但输入 `python` 报找不到。

```bash
# 检查 conda 环境是否激活
conda activate base

# 如果还不行，检查 PATH
which python
echo $PATH
```

### 问题3：包安装超时

```
pip install torch
# 卡住很久或报 timeout
```

解决：确认配置了国内镜像，或者手动指定：

```bash
pip install torch -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 问题4：版本冲突

```
ERROR: pip's dependency resolver found conflicts
```

解决思路：

```bash
# 方案1：创建一个全新的环境，逐个安装
conda create -n fresh python=3.11
conda activate fresh
pip install 包A
pip install 包B  # 如果冲突，会提示哪里冲突

# 方案2：把冲突的包降到兼容的版本
pip install "包A>=1.0,<2.0"
```

### 问题5：装了包但 import 报错

```python
import torch
# ModuleNotFoundError: No module named 'torch'
```

最常见原因：你装包的环境和运行代码的环境不是同一个。

```bash
# 检查当前环境
conda info --envs   # 看哪个带 *

# 检查包装在了哪个环境里
conda activate 你以为装了torch的环境
pip list | grep torch

# 确认 Python 路径
which python
# 应该指向你的 conda 环境目录
```

---

## 实操练习：搭建你的第一个学习环境

```bash
# 1. 创建一个专门用于本课程的环境
conda create -n ai-course python=3.11
conda activate ai-course

# 2. 安装第一阶段需要的基础包
pip install requests beautifulsoup4 fastapi uvicorn

# 3. 安装第二阶段需要的数据分析包
pip install numpy pandas matplotlib seaborn jupyter

# 4. 验证安装
python -c "
import numpy as np
import pandas as pd
print(f'NumPy 版本: {np.__version__}')
print(f'Pandas 版本: {pd.__version__}')
print('✅ 环境搭建成功！')
"

# 5. 导出环境配置
pip freeze > requirements.txt
cat requirements.txt

# 6. 查看环境列表
conda env list
```

如果最后看到 `✅ 环境搭建成功！`，你的 Python 环境就准备好了。

---

## 命令速查

| 命令 | 用途 |
|------|------|
| `conda create -n 名字 python=3.11` | 创建新环境 |
| `conda activate 名字` | 激活环境 |
| `conda deactivate` | 退出当前环境 |
| `conda env list` | 列出所有环境 |
| `conda env remove -n 名字` | 删除环境 |
| `pip install 包名` | 安装 Python 包 |
| `pip list` | 查看已安装的包 |
| `pip freeze > requirements.txt` | 导出依赖列表 |
| `pip install -r requirements.txt` | 从文件安装依赖 |
