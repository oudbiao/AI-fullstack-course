---
title: "Git 核心操作"
sidebar_position: 2
description: "掌握日常开发中最常用的 Git 操作"
---

# Git 核心操作

## 学习目标

- 熟练使用 `git add`、`git commit`、`git status`、`git log`
- 学会用 `git diff` 查看修改内容
- 会编写 `.gitignore` 文件
- 掌握几种常用的撤销操作

---

## 准备工作

我们用一个模拟的 AI 项目来练习所有操作。先创建项目：

```bash
mkdir ai-image-classifier
cd ai-image-classifier
git init

# 创建基本的项目结构
mkdir data models src
touch src/train.py src/model.py src/utils.py
touch README.md requirements.txt
```

---

## 查看状态：git status

`git status` 是你用得最多的命令，它告诉你当前仓库的状态：哪些文件被修改了？哪些在暂存区？哪些还没被 Git 跟踪？

```bash
git status
```

输出：

```
On branch main

No commits yet

Untracked files:
  (use "git add <file>..." to include in what will be committed)
        README.md
        requirements.txt
        src/

nothing added to commit but untracked files present
```

**Untracked files（未跟踪的文件）**：Git 看到了这些文件，但还没有管理它们。你需要用 `git add` 告诉 Git "请开始跟踪这些文件"。

---

## 添加到暂存区：git add

```bash
# 添加单个文件
git add README.md

# 添加多个文件
git add src/train.py src/model.py

# 添加整个文件夹
git add src/

# 添加所有文件（最常用）
git add .

# 添加所有修改过的已跟踪文件（不包括新文件）
git add -u
```

### 实操案例

先给文件写点内容：

```bash
echo "# AI 图像分类器" > README.md
echo "torch>=2.0" > requirements.txt

cat > src/model.py << 'EOF'
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 16 * 16, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 16 * 16 * 16)
        x = self.fc1(x)
        return x
EOF
```

现在添加所有文件并查看状态：

```bash
git add .
git status
```

输出：

```
On branch main

No commits yet

Changes to be committed:
  (use "git rm --cached <file>..." to unstage)
        new file:   README.md
        new file:   requirements.txt
        new file:   src/model.py
        new file:   src/train.py
        new file:   src/utils.py
```

文件变成了绿色的 **"Changes to be committed"**，说明它们已经在暂存区里，准备好被提交了。

---

## 提交：git commit

```bash
git commit -m "初始化项目：添加模型定义和项目结构"
```

输出：

```
[main (root-commit) a1b2c3d] 初始化项目：添加模型定义和项目结构
 5 files changed, 18 insertions(+)
 create mode 100644 README.md
 create mode 100644 requirements.txt
 create mode 100644 src/model.py
 create mode 100644 src/train.py
 create mode 100644 src/utils.py
```

### 提交信息怎么写？

提交信息应该简洁、清晰，让人一看就知道这次改了什么。

**好的提交信息：**

```bash
git commit -m "添加 CNN 模型定义"
git commit -m "修复训练循环中学习率未更新的 bug"
git commit -m "添加数据增强：随机翻转和颜色抖动"
git commit -m "更新 README：添加安装说明"
```

**不好的提交信息：**

```bash
git commit -m "update"           # 改了什么？
git commit -m "fix"              # 修了什么？
git commit -m "aaa"              # ？？？
git commit -m "改了一些东西"       # 等于没说
```

:::tip 一个实用的原则
提交信息回答这个问题：**"这次提交做了什么？"** 用动词开头（添加、修复、更新、删除、重构），说清楚对象。
:::

---

## 查看修改：git diff

`git diff` 告诉你"上次提交之后，你改了什么"。

### 案例：修改模型代码

```bash
# 给 model.py 添加一个新层
cat > src/model.py << 'EOF'
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)  # 新增的卷积层
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 8 * 8, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))  # 新增
        x = x.view(-1, 32 * 8 * 8)
        x = self.fc1(x)
        return x
EOF
```

现在查看变化：

```bash
git diff
```

输出会用红色和绿色高亮显示：

```diff
--- a/src/model.py
+++ b/src/model.py
@@ -1,4 +1,5 @@
+import torch
 import torch.nn as nn

 class SimpleCNN(nn.Module):
     def __init__(self):
         super().__init__()
         self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
+        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)  # 新增的卷积层
         self.pool = nn.MaxPool2d(2, 2)
-        self.fc1 = nn.Linear(16 * 16 * 16, 10)
+        self.fc1 = nn.Linear(32 * 8 * 8, 10)
```

- **红色（`-` 开头）**：被删除的行
- **绿色（`+` 开头）**：被新增的行

现在提交这次修改：

```bash
git add src/model.py
git commit -m "增加第二层卷积层，提升模型能力"
```

### diff 的几种用法

```bash
git diff                    # 查看工作目录中未暂存的修改
git diff --staged           # 查看暂存区中的修改（已 add 但未 commit）
git diff HEAD~1             # 查看最近一次提交改了什么
git diff abc1234 def5678    # 比较两次提交之间的差异
```

---

## 查看历史：git log

```bash
git log
```

输出：

```
commit def5678... (HEAD -> main)
Author: Zhang San <zhangsan@example.com>
Date:   Mon Feb 9 10:30:00 2026

    增加第二层卷积层，提升模型能力

commit a1b2c3d...
Author: Zhang San <zhangsan@example.com>
Date:   Mon Feb 9 10:00:00 2026

    初始化项目：添加模型定义和项目结构
```

### 更简洁的历史查看

```bash
# 一行一条记录（最常用）
git log --oneline
# 输出:
# def5678 增加第二层卷积层，提升模型能力
# a1b2c3d 初始化项目：添加模型定义和项目结构

# 带文件变更统计
git log --oneline --stat

# 图形化显示分支（有分支时很有用）
git log --oneline --graph --all
```

---

## .gitignore：告诉 Git 忽略哪些文件

有些文件不应该被 Git 管理：

- 数据文件（几个 GB 的训练数据）
- 模型权重文件（几百 MB）
- 虚拟环境文件夹
- 系统生成的临时文件
- API 密钥、密码等敏感信息

创建 `.gitignore` 文件来告诉 Git 忽略它们：

```bash
cat > .gitignore << 'EOF'
# Python 缓存
__pycache__/
*.pyc
*.pyo

# 虚拟环境
venv/
.venv/
env/

# Jupyter Notebook 检查点
.ipynb_checkpoints/

# 数据文件（太大了，不放进 Git）
data/*.csv
data/*.json
data/*.zip
*.h5
*.hdf5

# 模型权重文件
models/*.pt
models/*.pth
models/*.onnx
*.bin

# 环境变量文件（包含 API 密钥等敏感信息）
.env
.env.local

# IDE 配置
.vscode/
.idea/

# 操作系统文件
.DS_Store
Thumbs.db

# 日志
logs/
*.log

# 分发/打包
dist/
build/
*.egg-info/
EOF
```

```bash
git add .gitignore
git commit -m "添加 .gitignore：忽略缓存、数据、模型权重和敏感文件"
```

### 验证 .gitignore 生效

```bash
# 创建一些应该被忽略的文件
mkdir -p __pycache__
touch __pycache__/model.cpython-311.pyc
touch .env
echo "OPENAI_API_KEY=sk-secret123" > .env

# 查看状态——这些文件不会出现
git status
# 输出: nothing to commit, working tree clean
```

`.env` 和 `__pycache__/` 都被忽略了，不会被提交到 Git 里。你的 API 密钥是安全的。

:::warning 已经被跟踪的文件
`.gitignore` 只对**还没有被 Git 跟踪**的文件有效。如果你先提交了一个文件、后来才把它加入 `.gitignore`，它不会自动被忽略。需要先手动取消跟踪：

```bash
git rm --cached .env
git commit -m "从 Git 中移除 .env 文件"
```
:::

---

## 撤销操作：后悔药

Git 提供了几种不同的"后悔药"，根据你后悔到哪一步来选择。

### 场景1：改了文件但还没 add，想恢复原样

```bash
# 你改了 src/utils.py，但改得不满意，想恢复到上次提交的状态
git restore src/utils.py

# 恢复所有文件
git restore .
```

:::warning
`git restore` 会**丢弃你的修改**，无法恢复。确认你真的不要这些修改后再执行。
:::

### 场景2：已经 add 了，想从暂存区撤回（但保留修改）

```bash
# 你 add 了 model.py，但还不想提交它
git restore --staged src/model.py

# 文件会从暂存区退回到工作目录，你的修改还在
```

### 场景3：已经 commit 了，想修改提交信息

```bash
# 刚提交完发现消息写错了
git commit --amend -m "修正后的提交信息"
```

### 场景4：已经 commit 了，想撤销整个提交

```bash
# 撤销最近一次提交，但保留文件修改（回到 add 之前）
git reset HEAD~1

# 撤销最近一次提交，并且丢弃所有修改（彻底回退，慎用）
git reset --hard HEAD~1
```

### 案例：完整的后悔流程

```bash
# 假设你不小心把 API 密钥提交了
echo "API_KEY=sk-secret" > config.py
git add .
git commit -m "添加配置文件"

# 糟糕！密钥不应该提交。撤销这次提交
git reset HEAD~1

# 文件还在，但提交被撤销了。现在把它加入 .gitignore
echo "config.py" >> .gitignore
git add .gitignore
git commit -m "更新 .gitignore：忽略 config.py"
```

### 撤销操作速查

| 我想撤销什么 | 命令 | 文件修改还在吗 |
|------------|------|:-----------:|
| 工作目录的修改（还没 add） | `git restore 文件名` | ❌ 丢弃 |
| 暂存区的文件（已 add 未 commit） | `git restore --staged 文件名` | ✅ 保留 |
| 最近的提交信息写错了 | `git commit --amend -m "新信息"` | ✅ 保留 |
| 最近的提交（保留修改） | `git reset HEAD~1` | ✅ 保留 |
| 最近的提交（丢弃修改） | `git reset --hard HEAD~1` | ❌ 丢弃 |

---

## 实操练习：模拟一次完整的开发过程

```bash
# 1. 写训练脚本
cat > src/train.py << 'EOF'
import torch
from model import SimpleCNN

model = SimpleCNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
print("模型参数量:", sum(p.numel() for p in model.parameters()))
EOF

git add src/train.py
git commit -m "添加基础训练脚本"

# 2. 写工具函数
cat > src/utils.py << 'EOF'
def accuracy(predictions, labels):
    """计算准确率"""
    correct = (predictions.argmax(dim=1) == labels).sum().item()
    return correct / len(labels)
EOF

git add src/utils.py
git commit -m "添加准确率计算工具函数"

# 3. 更新 README
cat > README.md << 'EOF'
# AI 图像分类器

一个使用 CNN 的简单图像分类项目。

## 安装

```bash
pip install -r requirements.txt
```

## 使用

```bash
python src/train.py
```
EOF

git add README.md
git commit -m "更新 README：添加安装和使用说明"

# 4. 查看完整历史
git log --oneline
```

你应该看到类似这样的 5 条提交记录：

```
f6g7h8i 更新 README：添加安装和使用说明
d4e5f6g 添加准确率计算工具函数
b2c3d4e 添加基础训练脚本
9a0b1c2 添加 .gitignore：忽略缓存、数据、模型权重和敏感文件
a1b2c3d 初始化项目：添加模型定义和项目结构
```

每一条都是一个可以回退到的存档点。

---

## 小结

| 命令 | 用途 | 使用频率 |
|------|------|:------:|
| `git status` | 查看当前状态 | ⭐⭐⭐⭐⭐ |
| `git add .` | 暂存所有修改 | ⭐⭐⭐⭐⭐ |
| `git commit -m "消息"` | 提交 | ⭐⭐⭐⭐⭐ |
| `git log --oneline` | 查看历史 | ⭐⭐⭐⭐ |
| `git diff` | 查看修改内容 | ⭐⭐⭐⭐ |
| `git restore` | 撤销工作目录修改 | ⭐⭐⭐ |
| `git reset` | 撤销提交 | ⭐⭐ |
