---
title: "终端基础操作"
sidebar_position: 2
description: "掌握核心命令、路径概念、管道与环境变量"
---

# 终端基础操作

## 学习目标

- 掌握 10+ 个核心命令，覆盖日常 90% 的操作
- 理解绝对路径和相对路径
- 学会使用管道和重定向
- 理解环境变量的概念

---

## 打开终端

首先，找到并打开你的终端：

| 操作系统 | 怎么打开 |
|---------|---------|
| **Windows** | 搜索 "PowerShell" 或 "Windows Terminal"，点击打开 |
| **macOS** | `Command + 空格` 搜索 "Terminal"，回车打开 |
| **Linux** | `Ctrl + Alt + T` |

你会看到一个窗口，里面有一个闪烁的光标，等着你输入命令。这就是终端。

:::info Windows 用户的选择
Windows 有多种终端选项。推荐使用 **Windows Terminal**（可从微软商店免费安装），然后在里面选择 PowerShell 标签页。本教程的命令以 macOS/Linux 为主，Windows 下绝大部分命令相同，少数不同的地方会特别标注。
:::

---

## 第一部分：路径——你在哪里？

命令行没有图形界面，你需要用文字告诉计算机"我要操作哪个文件夹里的东西"。这就是**路径**。

### 你现在在哪？

```bash
pwd
```

`pwd` = **P**rint **W**orking **D**irectory（打印当前工作目录）

输出可能是这样的：

```
/Users/zhangsan          # macOS
/home/zhangsan           # Linux
C:\Users\zhangsan        # Windows PowerShell
```

这就是你当前所在的文件夹，叫做**工作目录**。

### 绝对路径 vs 相对路径

```
/Users/zhangsan/projects/ai-course/data/train.csv
```

这是一个**绝对路径**——从根目录 `/` 开始，完整地描述了文件的位置。就像现实中的完整地址："中国北京市海淀区中关村大街1号"。

```
data/train.csv
```

这是一个**相对路径**——相对于你当前所在的文件夹。如果你当前在 `/Users/zhangsan/projects/ai-course/`，那么 `data/train.csv` 就等于上面那个绝对路径。就像说"隔壁楼2层"。

### 路径中的特殊符号

| 符号 | 含义 | 例子 |
|------|------|------|
| `/` | 根目录（所有文件的起点） | `cd /` |
| `~` | 当前用户的主目录（Home） | `cd ~` 等于 `cd /Users/zhangsan` |
| `.` | 当前目录 | `./run.py` 表示当前目录下的 run.py |
| `..` | 上一级目录 | `cd ..` 回到上一层 |

一个练习帮你理解：

```bash
# 假设你在 /Users/zhangsan/projects/ai-course

pwd                    # 输出: /Users/zhangsan/projects/ai-course
cd ..                  # 回到上一级
pwd                    # 输出: /Users/zhangsan/projects
cd ~                   # 回到 Home 目录
pwd                    # 输出: /Users/zhangsan
cd ~/projects/ai-course  # 用绝对路径回去
pwd                    # 输出: /Users/zhangsan/projects/ai-course
```

---

## 第二部分：核心命令

以下命令是你每天都会用到的。先跟着敲一遍，不需要背，用多了自然就记住了。

### 导航命令

#### `cd` — 切换目录

```bash
cd projects        # 进入 projects 文件夹
cd ..              # 回到上一级
cd ~               # 回到 Home 目录
cd ~/Desktop       # 去桌面
cd -               # 回到上一次所在的目录（很实用！）
```

#### `ls` — 列出文件

```bash
ls                 # 列出当前目录下的文件和文件夹
ls -l              # 详细列表（显示大小、日期、权限）
ls -a              # 显示隐藏文件（以 . 开头的文件）
ls -la             # 两者组合
ls projects/       # 列出 projects 文件夹里的内容
```

:::note Windows PowerShell
在 PowerShell 中，`ls` 同样可用（它是 `Get-ChildItem` 的别名）。`ls -la` 不行，用 `ls -Force` 显示隐藏文件。
:::

### 文件和文件夹操作

#### `mkdir` — 创建文件夹

```bash
mkdir my-project               # 创建一个文件夹
mkdir -p a/b/c                 # 一次性创建多层嵌套的文件夹
```

#### `touch` — 创建空文件

```bash
touch hello.py                 # 创建一个空的 Python 文件
touch README.md                # 创建一个空的 Markdown 文件
```

:::note Windows
PowerShell 没有 `touch`，用 `New-Item hello.py` 代替。
:::

#### `cp` — 复制

```bash
cp file.txt file_backup.txt          # 复制文件
cp file.txt ~/Desktop/               # 复制到桌面
cp -r my-folder/ my-folder-backup/   # 复制整个文件夹（-r 表示递归）
```

#### `mv` — 移动 / 重命名

```bash
mv old_name.py new_name.py       # 重命名文件
mv file.txt ~/Desktop/           # 移动到桌面
mv project/ ~/projects/          # 移动文件夹
```

#### `rm` — 删除

```bash
rm file.txt                  # 删除文件
rm -r my-folder/             # 删除文件夹及其所有内容
```

:::warning 命令行删除没有回收站
`rm` 删除的文件不会进回收站，直接就没了。操作前请确认你删对了东西。养成习惯：删除前先用 `ls` 看一眼。
:::

### 查看文件内容

```bash
cat file.txt          # 显示整个文件内容（适合小文件）
head file.txt         # 显示文件前 10 行
head -20 file.txt     # 显示前 20 行
tail file.txt         # 显示文件最后 10 行
tail -f log.txt       # 实时跟踪文件更新（看日志很有用）
```

### 搜索

```bash
grep "error" log.txt              # 在文件中搜索包含 "error" 的行
grep -r "import torch" ./         # 在当前目录下所有文件中搜索
grep -n "def train" model.py      # 搜索并显示行号
```

`grep` 是你未来 debug 的好帮手——在几十个文件里快速找到某个函数或变量在哪里被用到。

### 其他实用命令

```bash
clear              # 清屏（或按 Ctrl + L）
history            # 查看你之前执行过的所有命令
which python       # 查看 python 命令的路径（排查环境问题常用）
echo "hello"       # 输出一段文字
```

---

## 第三部分：管道与重定向

这两个概念是命令行真正强大的地方。

### 管道 `|`

管道的意思是：把前一个命令的输出，作为后一个命令的输入。

```bash
# 列出所有文件，从中找到 .py 文件
ls -la | grep ".py"

# 查看历史命令中用过的 git 命令
history | grep "git"

# 统计当前目录下有多少个 Python 文件
ls *.py | wc -l
```

你可以把管道想象成工厂流水线：一个工序的产出是下一个工序的原料。

### 重定向 `>` 和 `>>`

把命令的输出保存到文件里，而不是显示在屏幕上：

```bash
# 把 ls 的结果保存到 filelist.txt（覆盖写入）
ls -la > filelist.txt

# 把结果追加到文件末尾（不覆盖）
echo "新的一行" >> notes.txt

# 把 Python 脚本的输出保存到文件
python train.py > training_log.txt
```

`>` 是覆盖，`>>` 是追加。实战中经常用来保存训练日志。

### 组合使用

```bash
# 运行脚本，把正常输出和错误输出都保存到日志文件
python train.py > log.txt 2>&1

# 统计一个 Python 文件有多少行代码
cat model.py | wc -l

# 找到所有包含 "TODO" 的文件，并统计数量
grep -r "TODO" ./ | wc -l
```

---

## 第四部分：环境变量

环境变量是存储在系统中的一些"全局配置"，很多程序会读取它们来决定自己的行为。

### 查看环境变量

```bash
# 查看所有环境变量
env

# 查看某一个环境变量的值
echo $PATH
echo $HOME
```

### 最重要的环境变量：PATH

`PATH` 决定了你在终端里输入一个命令时，系统去哪些目录里找这个命令。

```bash
echo $PATH
# 输出类似: /usr/local/bin:/usr/bin:/bin:/Users/zhangsan/miniconda3/bin
```

这些路径用 `:` 分隔。当你输入 `python` 时，系统会依次在这些目录里找 `python` 这个文件，找到第一个就执行。

如果你遇到 `command not found`（命令找不到），通常就是因为这个程序没在 `PATH` 的任何目录里。

### 设置环境变量

```bash
# 临时设置（只在当前终端窗口有效）
export MY_API_KEY="sk-abc123"
echo $MY_API_KEY    # 输出: sk-abc123

# 验证：关闭终端重新打开，MY_API_KEY 就没了
```

```bash
# 永久设置（写入配置文件）
# macOS/Linux 用 zsh：
echo 'export MY_API_KEY="sk-abc123"' >> ~/.zshrc
source ~/.zshrc    # 立即生效

# 如果用 bash：
echo 'export MY_API_KEY="sk-abc123"' >> ~/.bashrc
source ~/.bashrc
```

:::info 为什么需要了解环境变量？
在后续的学习中，你会经常用环境变量来存储 API Key（比如 OpenAI 的密钥）。这样做比把密钥写在代码里安全得多：

```python
import os
api_key = os.environ.get("OPENAI_API_KEY")
```
:::

---

## 实操练习

打开终端，依次完成以下操作：

```bash
# 1. 确认你在哪
pwd

# 2. 去到 Home 目录
cd ~

# 3. 创建一个学习项目文件夹
mkdir -p ai-study/stage0/terminal-practice

# 4. 进入这个文件夹
cd ai-study/stage0/terminal-practice

# 5. 创建几个文件
touch hello.py notes.txt data.csv

# 6. 查看创建的文件
ls -la

# 7. 往文件里写点东西
echo "print('Hello, AI!')" > hello.py
echo "第一天学习笔记" > notes.txt

# 8. 查看文件内容
cat hello.py
cat notes.txt

# 9. 复制 notes.txt 做个备份
cp notes.txt notes_backup.txt

# 10. 确认备份成功
ls

# 11. 给 notes.txt 追加内容
echo "学了 cd, ls, mkdir, touch, cp, cat 命令" >> notes.txt
cat notes.txt

# 12. 搜索包含 "AI" 的文件
grep -r "AI" ./

# 13. 回到上一级目录
cd ..
pwd
```

如果所有步骤都成功了，恭喜你——你已经掌握了命令行最核心的操作。

---

## 常用命令速查表

| 命令 | 用途 | 常用参数 |
|------|------|---------|
| `pwd` | 显示当前目录 | |
| `cd` | 切换目录 | `..` 上一级，`~` Home，`-` 上一次 |
| `ls` | 列出文件 | `-l` 详细，`-a` 隐藏文件 |
| `mkdir` | 创建文件夹 | `-p` 创建多层 |
| `touch` | 创建空文件 | |
| `cp` | 复制 | `-r` 复制文件夹 |
| `mv` | 移动/重命名 | |
| `rm` | 删除 | `-r` 删除文件夹 |
| `cat` | 查看文件 | |
| `head` / `tail` | 查看开头/结尾 | `-n 数字` 指定行数 |
| `grep` | 搜索文本 | `-r` 递归，`-n` 行号 |
| `echo` | 输出文字 | |
| `clear` | 清屏 | |
| `history` | 历史命令 | |
| `which` | 查看命令路径 | |

:::tip 记不住怎么办？
这张表不需要背。用多了自然就记住了。初期可以打印出来贴在屏幕旁边，或者保存到手机里随时查。大部分命令输入 `命令 --help` 就能看到用法说明。
:::
