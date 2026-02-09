---
title: "包管理器"
sidebar_position: 3
description: "用包管理器安装系统软件和开发工具"
---

# 包管理器

## 学习目标

- 理解什么是包管理器，为什么需要它
- 根据你的操作系统，学会使用对应的包管理器
- 用包管理器安装几个 AI 开发需要的基础工具

---

## 什么是包管理器？

你用手机的时候，想装一个 App，会打开 App Store 或应用商店，搜索、点击安装。

**包管理器就是电脑上的"应用商店"，不过用命令行操作。** 它帮你做三件事：

1. **安装软件**——一行命令搞定，不需要去网站下载安装包
2. **更新软件**——一行命令更新所有软件到最新版
3. **管理依赖**——自动处理"装 A 必须先有 B"的依赖关系

不同操作系统有不同的包管理器。找到你的系统，跟着做就行。

---

## macOS：Homebrew

[Homebrew](https://brew.sh) 是 macOS 上最流行的包管理器，几乎每个开发者都会装。

### 安装 Homebrew

打开终端，粘贴运行：

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

安装过程可能需要几分钟。如果提示需要密码，输入你的电脑开机密码（输入时不会显示字符，正常现象）。

安装完成后，验证一下：

```bash
brew --version
# 输出类似: Homebrew 4.x.x
```

:::info 国内用户
如果下载很慢，可以搜索 "Homebrew 清华镜像" 或 "Homebrew 中科大镜像" 使用国内加速源。
:::

### 常用命令

```bash
# 搜索软件
brew search git

# 安装软件
brew install git
brew install wget
brew install tree

# 查看已安装的软件
brew list

# 更新所有软件
brew update      # 更新 Homebrew 自身
brew upgrade     # 更新所有已安装的软件

# 卸载软件
brew uninstall wget

# 查看软件信息
brew info git
```

### 安装 AI 开发基础工具

```bash
# Git（版本管理，下一章会详细学）
brew install git

# tree（以树状结构显示目录，看项目结构很方便）
brew install tree

# wget（下载文件的工具）
brew install wget
```

安装完 tree 之后试一下：

```bash
cd ~/ai-study
tree
```

输出类似：

```
.
└── stage0
    └── terminal-practice
        ├── data.csv
        ├── hello.py
        ├── notes.txt
        └── notes_backup.txt
```

比 `ls` 更直观地看到整个目录结构。

---

## Ubuntu/Debian Linux：apt

`apt` 是 Ubuntu 和 Debian 系列 Linux 自带的包管理器，不需要额外安装。

### 常用命令

```bash
# 更新软件源信息（安装前建议先执行）
sudo apt update

# 安装软件
sudo apt install git
sudo apt install tree
sudo apt install wget
sudo apt install curl

# 搜索软件
apt search nodejs

# 查看已安装的软件
apt list --installed

# 更新所有软件
sudo apt update && sudo apt upgrade

# 卸载软件
sudo apt remove wget
```

:::info 关于 sudo
`sudo` 的意思是"用管理员权限执行"。安装系统级软件需要管理员权限，所以 `apt install` 前面要加 `sudo`，会要求你输入密码。
:::

### 安装 AI 开发基础工具

```bash
sudo apt update
sudo apt install -y git tree wget curl build-essential
```

`-y` 表示自动确认，不需要手动输入 "Y"。`build-essential` 包含了编译工具，有些 Python 库安装时需要用到。

---

## Windows：winget 和 Scoop

Windows 有两个主要的命令行包管理器。

### 方案一：winget（推荐，Windows 自带）

Windows 10 (1709+) 和 Windows 11 自带 `winget`。打开 PowerShell 试试：

```powershell
winget --version
```

如果有输出，说明已经可以用了。

```powershell
# 搜索软件
winget search vscode

# 安装软件
winget install Git.Git
winget install Microsoft.VisualStudioCode
winget install Python.Python.3.11

# 更新所有软件
winget upgrade --all

# 查看已安装的软件
winget list
```

### 方案二：Scoop（更贴近 Linux 的体验）

如果你喜欢更"开发者友好"的工具，可以安装 [Scoop](https://scoop.sh)：

```powershell
# 安装 Scoop（在 PowerShell 中运行）
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
irm get.scoop.sh | iex
```

```powershell
# 使用方式
scoop install git
scoop install python
scoop install tree

# 更新
scoop update *
```

### 安装 AI 开发基础工具（winget）

```powershell
winget install Git.Git
winget install Python.Python.3.11
```

:::tip Windows 用户的额外建议
强烈推荐安装 **Windows Terminal**（微软商店搜索即可），它比自带的 PowerShell 窗口好用很多——支持多标签页、更好的字体渲染、更方便的复制粘贴。
:::

---

## 包管理器 vs pip/conda

你可能会困惑：后面还会学到 `pip` 和 `conda`，它们不也是包管理器吗？有什么区别？

| 工具 | 管理什么 | 类比 |
|------|---------|------|
| **brew / apt / winget** | 操作系统级的软件（Git、Python、Node.js、Docker） | 手机应用商店 |
| **pip** | Python 库（numpy、pandas、torch） | Python 专属的应用商店 |
| **conda** | Python 环境 + Python 库 + 部分系统库 | 更强大的 Python 应用商店 |

简单说：

- 装 Git、Docker、系统工具 → 用 **brew / apt / winget**
- 装 Python 库 → 用 **pip** 或 **conda**
- 管理 Python 版本和虚拟环境 → 用 **conda**

这三者各司其职，不冲突。

---

## 实操练习

根据你的操作系统，完成以下练习：

### macOS 用户

```bash
# 1. 安装 Homebrew（如果还没装）
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 2. 安装 tree 和 wget
brew install tree wget

# 3. 用 tree 查看你之前创建的 ai-study 目录结构
tree ~/ai-study

# 4. 用 wget 下载一个文件试试
wget https://raw.githubusercontent.com/datasets/iris/master/data/iris.csv
cat iris.csv | head -5
```

### Ubuntu 用户

```bash
# 1. 更新软件源
sudo apt update

# 2. 安装 tree 和 wget
sudo apt install -y tree wget

# 3. 用 tree 查看目录
tree ~/ai-study

# 4. 下载测试文件
wget https://raw.githubusercontent.com/datasets/iris/master/data/iris.csv
head -5 iris.csv
```

### Windows 用户

```powershell
# 1. 确认 winget 可用
winget --version

# 2. 安装 Git（后续章节需要）
winget install Git.Git

# 3. 验证安装
git --version
```

---

## 本章自检

完成以下检查，确认你掌握了终端基础：

- [ ] 能打开终端并知道自己在哪个目录
- [ ] 能用 `cd`、`ls`、`mkdir`、`touch`、`cp`、`mv`、`rm` 完成基本文件操作
- [ ] 理解绝对路径和相对路径的区别
- [ ] 能用管道 `|` 组合两个命令
- [ ] 能用 `>` 或 `>>` 把输出保存到文件
- [ ] 能用你的包管理器安装一个软件
- [ ] 知道 `echo $PATH` 是什么意思

:::tip 全部打勾了？
你已经掌握了终端和命令行的核心技能。接下来我们学 Git——开发者的另一个必备工具。
:::
