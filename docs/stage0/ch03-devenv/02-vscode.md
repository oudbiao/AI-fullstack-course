---
title: "VS Code 配置"
sidebar_position: 2
description: "把 VS Code 配置成趁手的 AI 开发工具"
---

# VS Code 配置

## 学习目标

- 安装 VS Code 并完成中文化
- 安装 Python 开发必备扩展
- 学会用 VS Code 内置终端
- 掌握 10 个最常用的快捷键
- 了解 AI 辅助编程工具

---

## 为什么选 VS Code？

| 编辑器 | 优点 | 缺点 |
|-------|------|------|
| **VS Code** | 免费、轻量、扩展丰富、AI 支持好 | 大项目可能不如 PyCharm 智能 |
| PyCharm | Python 支持最强、重构方便 | 社区版免费但功能少，专业版收费 |
| Vim/NeoVim | 极快、极客 | 学习曲线陡峭 |

VS Code 是目前全球使用最多的代码编辑器，Python 和 AI 开发的支持非常好。对新手来说是最佳选择。

---

## 安装 VS Code

### macOS

```bash
# 用 Homebrew 安装（推荐）
brew install --cask visual-studio-code

# 或者从官网下载：https://code.visualstudio.com
```

安装完成后，配置命令行启动：

1. 打开 VS Code
2. 按 `Cmd + Shift + P`，输入 "shell command"
3. 选择 **Shell Command: Install 'code' command in PATH**

之后你就可以在终端里用 `code` 命令打开文件和文件夹了：

```bash
code .                  # 用 VS Code 打开当前文件夹
code ~/projects         # 打开指定文件夹
code hello.py           # 打开指定文件
```

### Windows

```powershell
# 用 winget 安装
winget install Microsoft.VisualStudioCode

# 或从官网下载：https://code.visualstudio.com
```

安装时**勾选** "Add to PATH"，这样就能在终端用 `code` 命令了。

### Ubuntu

```bash
# 方法1：用 snap（推荐）
sudo snap install code --classic

# 方法2：用 apt
sudo apt install software-properties-common apt-transport-https wget
wget -q https://packages.microsoft.com/keys/microsoft.asc -O- | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://packages.microsoft.com/repos/vscode stable main"
sudo apt update
sudo apt install code
```

---

## 中文设置

1. 打开 VS Code
2. 按 `Ctrl + Shift + X`（macOS 是 `Cmd + Shift + X`）打开扩展面板
3. 搜索 **Chinese (Simplified)**
4. 点击 **Install** 安装
5. 重启 VS Code，界面变成中文

---

## 安装必备扩展

打开扩展面板（左侧栏的方块图标，或按 `Ctrl/Cmd + Shift + X`），搜索并安装以下扩展：

### 必装扩展

| 扩展名 | 作用 | 搜索关键词 |
|-------|------|----------|
| **Python** | Python 语法支持、调试、运行 | `ms-python.python` |
| **Pylance** | Python 智能提示、类型检查 | `ms-python.vscode-pylance` |
| **Jupyter** | 在 VS Code 里运行 Notebook | `ms-toolsai.jupyter` |
| **GitLens** | 增强 Git 功能，看谁改了哪一行 | `eamodio.gitlens` |

### 推荐扩展

| 扩展名 | 作用 |
|-------|------|
| **autoDocstring** | 自动生成 Python 函数文档字符串 |
| **indent-rainbow** | 用颜色区分缩进层级 |
| **Error Lens** | 把错误信息直接显示在代码行末 |
| **Material Icon Theme** | 更好看的文件图标 |

---

## 配置 Python 解释器

安装完 Python 扩展后，需要告诉 VS Code 用哪个 Python 环境：

1. 按 `Ctrl/Cmd + Shift + P` 打开命令面板
2. 输入 **Python: Select Interpreter**
3. 选择你之前创建的 conda 环境（比如 `ai-course`）

你应该能看到类似这样的选项列表：

```
Python 3.11.7 ('ai-course')    ~/miniconda3/envs/ai-course/bin/python
Python 3.12.1 ('base')         ~/miniconda3/bin/python
```

选择 `ai-course` 那个。

:::tip 自动检测
VS Code 的 Python 扩展会自动检测你系统上所有的 Python 环境（包括 conda 和 venv 环境）。如果看不到你想要的环境，试试先在终端里 `conda activate` 那个环境，然后在终端里输入 `code .` 打开 VS Code。
:::

---

## 使用内置终端

VS Code 内置了终端，你不需要再单独开一个终端窗口。

### 打开终端

```
快捷键：Ctrl + `（键盘左上角，ESC 下面的那个键）
```

或者从菜单：**终端 → 新建终端**

### 案例：在 VS Code 里完成完整的开发流程

```bash
# 1. 在终端里激活环境
conda activate ai-course

# 2. 创建项目文件夹
mkdir my-first-project
cd my-first-project

# 3. 用 VS Code 打开这个文件夹（会在新窗口中打开）
code .
```

然后在 VS Code 中：

1. 在左侧的文件浏览器中，点击新建文件图标，创建 `hello.py`
2. 写入代码：

```python
name = input("你叫什么名字？")
print(f"你好，{name}！欢迎来到 AI 世界 🤖")
```

3. 点击右上角的 **▶ 运行** 按钮（或按 `Ctrl/Cmd + Shift + P` → "Run Python File"）
4. 看终端里的输出

### 终端技巧

- **多终端**：点终端面板右上角的 `+` 号可以开多个终端
- **分屏**：可以左右分屏，一边写代码一边看终端
- **终端类型**：可以选择 bash、zsh、PowerShell 等不同的 shell

---

## 最常用的快捷键

不需要背，先记住前 5 个，其他用到了再查。

### 基础操作

| 操作 | Windows/Linux | macOS |
|------|:---:|:---:|
| 命令面板（最重要！） | `Ctrl + Shift + P` | `Cmd + Shift + P` |
| 快速打开文件 | `Ctrl + P` | `Cmd + P` |
| 打开/关闭终端 | `` Ctrl + ` `` | `` Ctrl + ` `` |
| 保存 | `Ctrl + S` | `Cmd + S` |
| 撤销 | `Ctrl + Z` | `Cmd + Z` |

### 编辑代码

| 操作 | Windows/Linux | macOS |
|------|:---:|:---:|
| 复制当前行 | `Shift + Alt + ↓` | `Shift + Option + ↓` |
| 移动当前行 | `Alt + ↑/↓` | `Option + ↑/↓` |
| 删除当前行 | `Ctrl + Shift + K` | `Cmd + Shift + K` |
| 多光标编辑 | `Alt + 点击` | `Option + 点击` |
| 代码注释 | `Ctrl + /` | `Cmd + /` |
| 代码格式化 | `Shift + Alt + F` | `Shift + Option + F` |

### 搜索与导航

| 操作 | Windows/Linux | macOS |
|------|:---:|:---:|
| 全局搜索 | `Ctrl + Shift + F` | `Cmd + Shift + F` |
| 文件内搜索 | `Ctrl + F` | `Cmd + F` |
| 查找替换 | `Ctrl + H` | `Cmd + Option + F` |
| 跳转到指定行 | `Ctrl + G` | `Ctrl + G` |

### 案例：多光标编辑的威力

假设你需要把 5 个变量名从 `data1`、`data2`... 改成 `dataset1`、`dataset2`...：

```python
data1 = load("file1.csv")
data2 = load("file2.csv")
data3 = load("file3.csv")
data4 = load("file4.csv")
data5 = load("file5.csv")
```

操作：
1. 选中第一个 `data`
2. 按 `Ctrl/Cmd + D` 连续按 5 次，依次选中所有 `data`
3. 输入 `dataset`

5 个位置同时被修改，2 秒搞定。

---

## AI 辅助编程工具

现在有不少 AI 工具可以在 VS Code 里帮你写代码。作为 AI 课程的学习者，值得了解一下：

### GitHub Copilot

- 在你打字的时候自动补全代码
- 按 `Tab` 接受建议
- 学生可以免费使用（通过 GitHub Student Pack）
- 扩展搜索：`GitHub.copilot`

### Codeium

- 免费的 AI 代码补全工具
- 功能类似 Copilot，对个人用户完全免费
- 扩展搜索：`Codeium.codeium`

### 使用建议

:::warning 对学习者的建议
在学习阶段，**不要过度依赖 AI 代码补全**。它就像计算器——你还没学会心算就开始用计算器，数学永远学不好。

建议：
- 前两个阶段（Python 基础）：**关掉** AI 补全，自己写
- 第三阶段之后：可以开启 AI 补全，但要**理解**它生成的每一行代码
- 做项目时：可以自由使用，提高效率
:::

---

## 推荐的 VS Code 设置

按 `Ctrl/Cmd + ,` 打开设置，搜索并修改以下选项：

| 设置项 | 建议值 | 原因 |
|-------|-------|------|
| Auto Save | `afterDelay` | 自动保存，再也不怕忘记 Ctrl+S |
| Font Size | `14` 或 `15` | 代码字体稍大点，看着不累 |
| Tab Size | `4` | Python 标准缩进 |
| Word Wrap | `on` | 长行自动换行 |
| Minimap | `off` | 关掉右侧小地图，省屏幕空间 |

或者直接编辑 `settings.json`（`Ctrl/Cmd + Shift + P` → "Open Settings JSON"）：

```json
{
    "files.autoSave": "afterDelay",
    "editor.fontSize": 14,
    "editor.tabSize": 4,
    "editor.wordWrap": "on",
    "editor.minimap.enabled": false,
    "python.terminal.activateEnvironment": true,
    "editor.formatOnSave": true,
    "python.formatting.provider": "black"
}
```

---

## 实操练习

1. **安装 VS Code** 和必备扩展（Python、Pylance、Jupyter、GitLens）
2. **创建一个项目**并用 VS Code 打开：

```bash
mkdir vscode-practice && cd vscode-practice && code .
```

3. **新建 `practice.py`**，写入以下代码：

```python
# 练习 VS Code 快捷键
fruits = ["apple", "banana", "cherry", "date", "elderberry"]

for i, fruit in enumerate(fruits):
    print(f"{i + 1}. {fruit}")

# 计算水果名字的平均长度
avg_len = sum(len(f) for f in fruits) / len(fruits)
print(f"\n平均名字长度: {avg_len:.1f} 个字符")
```

4. **运行代码**（点右上角的 ▶ 按钮）
5. **尝试快捷键**：
   - 用 `Ctrl/Cmd + /` 注释掉最后两行
   - 用 `Alt + ↑/↓` 移动一行代码
   - 用 `Ctrl/Cmd + D` 多选一个单词
   - 用 `Ctrl/Cmd + Shift + F` 全局搜索 "fruit"
