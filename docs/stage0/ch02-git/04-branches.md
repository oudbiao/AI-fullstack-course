---
title: "分支与协作"
sidebar_position: 4
description: "用分支安全地开发新功能，了解 Pull Request 流程"
---

# 分支与协作

## 学习目标

- 理解分支的概念和使用场景
- 掌握创建、切换、合并分支的操作
- 了解 Pull Request 的协作流程
- 学会解决简单的合并冲突

:::info 本节定位
分支在个人项目中用得不多，但在团队协作和开源贡献中是必备技能。作为入门，你只需要理解概念、会基本操作就行。等到实际需要时再回来深入。
:::

---

## 什么是分支？

### 用装修来类比

想象你住在一间公寓里（`main` 分支 = 你正在住的家）。你想尝试一种新的装修风格，但不确定效果好不好。

你有两个选择：

1. **直接在家里改**——如果改坏了，你就没得住了
2. **先租一间一模一样的公寓（新分支），在那边尝试**——如果好看就搬过来，不好看就退租

分支就是选项 2。你在新分支上随便改，改好了合并回 `main`，改坏了直接删掉，`main` 完全不受影响。

### 在代码中的实际场景

```
你正在做一个 AI 图像分类项目，main 分支上是能正常运行的代码。

现在你想尝试：
  - 把模型从 CNN 换成 Vision Transformer
  - 不确定效果会不会更好
  - 改动很大，可能需要好几天

如果直接在 main 上改：
  ❌ 改到一半代码跑不了了
  ❌ 老板突然让你修个 bug，但 main 已经被你改乱了
  ❌ 最后发现 ViT 效果不好，想回去——已经改了 50 个文件

如果用分支：
  ✅ 在 feature/vit 分支上慢慢改
  ✅ 老板让修 bug？切回 main，修完推上去，再切回来继续
  ✅ 发现 ViT 不行？删掉分支，main 毫发无损
```

---

## 分支基本操作

### 查看分支

```bash
# 查看本地分支（当前分支前面有 * 号）
git branch
# 输出:
# * main

# 查看所有分支（包括远程）
git branch -a
```

### 创建并切换分支

```bash
# 创建一个新分支
git branch feature/data-augmentation

# 切换到新分支
git checkout feature/data-augmentation

# 或者一步到位：创建并切换（更常用）
git checkout -b feature/data-augmentation
```

:::tip 分支命名惯例
常见的命名方式：
- `feature/xxx` — 新功能（如 `feature/add-resnet`）
- `fix/xxx` — 修复 bug（如 `fix/training-crash`）
- `experiment/xxx` — 实验性尝试（如 `experiment/try-vit`）
:::

### 案例：在分支上开发新功能

让我们实际操作一下。继续使用之前的 `ai-image-classifier` 项目：

```bash
cd ai-image-classifier

# 确认当前在 main 分支
git branch
# * main

# 创建并切换到新分支：添加数据增强功能
git checkout -b feature/data-augmentation
```

现在你在新分支上了。开始写代码：

```bash
# 创建数据增强模块
cat > src/augmentation.py << 'EOF'
import torchvision.transforms as T

def get_train_transforms():
    """训练数据的增强策略"""
    return T.Compose([
        T.RandomHorizontalFlip(p=0.5),        # 50% 概率水平翻转
        T.RandomRotation(degrees=15),          # 随机旋转 ±15 度
        T.ColorJitter(                         # 颜色抖动
            brightness=0.2,
            contrast=0.2,
            saturation=0.2
        ),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

def get_test_transforms():
    """测试数据只做标准化，不做增强"""
    return T.Compose([
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
EOF

# 更新 train.py，使用数据增强
cat >> src/train.py << 'EOF'

# 新增：使用数据增强
from augmentation import get_train_transforms, get_test_transforms
train_transform = get_train_transforms()
test_transform = get_test_transforms()
print("数据增强策略已加载")
EOF

# 提交到当前分支
git add .
git commit -m "feat: 添加数据增强模块（随机翻转、旋转、颜色抖动）"
```

现在查看两个分支的状态：

```bash
# 查看当前分支的历史
git log --oneline -3
# 输出:
# aaa1111 feat: 添加数据增强模块（随机翻转、旋转、颜色抖动）
# bbb2222 完善 README：添加项目说明和使用方法
# ccc3333 添加 .gitignore

# 切回 main 看看
git checkout main

# main 上没有 augmentation.py！
ls src/
# model.py  train.py  utils.py  （没有 augmentation.py）

# 切回 feature 分支
git checkout feature/data-augmentation
ls src/
# augmentation.py  model.py  train.py  utils.py  （有了！）
```

这就是分支的魔力——两条时间线互不影响。

---

## 合并分支

当你在分支上的功能开发完成、测试通过后，就可以把它合并回 `main`。

```bash
# 第一步：切回 main 分支
git checkout main

# 第二步：把 feature 分支合并到 main
git merge feature/data-augmentation
```

输出：

```
Updating bbb2222..aaa1111
Fast-forward
 src/augmentation.py | 25 +++++++++++++++++++++++++
 src/train.py        |  5 +++++
 2 files changed, 30 insertions(+)
 create mode 100644 src/augmentation.py
```

现在 `main` 分支也有数据增强代码了：

```bash
ls src/
# augmentation.py  model.py  train.py  utils.py  ✅
```

### 合并后的清理

```bash
# 功能分支已经合并，可以删掉了（保持仓库整洁）
git branch -d feature/data-augmentation

# 查看分支——只剩 main
git branch
# * main
```

---

## 合并冲突

### 什么时候会冲突？

当两个分支修改了**同一个文件的同一个位置**，Git 不知道该保留哪个版本，就会产生冲突。

### 案例：制造一个冲突并解决它

```bash
# 从 main 创建两个分支，模拟两个人同时工作
git checkout -b alice/update-model
cat > src/model.py << 'EOF'
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)  # Alice: 改成 32 个滤波器
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 16 * 16, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 32 * 16 * 16)
        return self.fc1(x)
EOF
git add . && git commit -m "alice: 增加滤波器数量到 32"

# 切回 main，创建 bob 的分支
git checkout main
git checkout -b bob/update-model
cat > src/model.py << 'EOF'
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 5, padding=2)  # Bob: 改成 64 个滤波器，5x5 卷积核
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 16 * 16, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 64 * 16 * 16)
        return self.fc1(x)
EOF
git add . && git commit -m "bob: 改用 64 个滤波器和 5x5 卷积核"
```

现在合并 Alice 的修改：

```bash
git checkout main
git merge alice/update-model    # ✅ 成功，无冲突
```

再合并 Bob 的修改：

```bash
git merge bob/update-model
# 输出:
# CONFLICT (content): Merge conflict in src/model.py
# Automatic merge failed; fix conflicts and then commit the result.
```

**冲突了！** 因为 Alice 和 Bob 都修改了 `model.py` 的同一行。

### 解决冲突

打开 `src/model.py`，你会看到 Git 标记出了冲突的位置：

```python
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
<<<<<<< HEAD
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)  # Alice: 改成 32 个滤波器
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 16 * 16, 10)
=======
        self.conv1 = nn.Conv2d(3, 64, 5, padding=2)  # Bob: 改成 64 个滤波器，5x5 卷积核
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 16 * 16, 10)
>>>>>>> bob/update-model
```

- `<<<<<<< HEAD` 到 `=======` 之间是**当前分支**（main，包含了 Alice 的修改）的版本
- `=======` 到 `>>>>>>> bob/update-model` 之间是**要合并进来的分支**（Bob）的版本

**你需要手动决定最终要保留什么。** 比如我们决定采用 Bob 的方案：

```python
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 5, padding=2)  # 采用 Bob 的方案
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 16 * 16, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 64 * 16 * 16)
        return self.fc1(x)
```

把 `<<<<<<<`、`=======`、`>>>>>>>` 标记全部删掉，只留下你想要的代码。然后：

```bash
git add src/model.py
git commit -m "merge: 合并 Alice 和 Bob 的修改，采用 Bob 的 64 滤波器方案"
```

冲突解决了。

:::tip VS Code 的冲突解决
VS Code 遇到冲突时会高亮显示，并给你几个按钮：
- **Accept Current Change**（保留当前分支的版本）
- **Accept Incoming Change**（保留要合并进来的版本）
- **Accept Both Changes**（两个都保留）

点一下就行，比手动编辑方便得多。
:::

```bash
# 清理分支
git branch -d alice/update-model
git branch -d bob/update-model
```

---

## Pull Request（了解即可）

在团队协作中，你通常不会直接往 `main` 分支合并。而是通过 **Pull Request（PR）** 让别人先审查你的代码，确认没问题后再合并。

### Pull Request 的流程

```
1. 你创建一个 feature 分支，写代码
2. push 到 GitHub
3. 在 GitHub 上创建 Pull Request
4. 同事审查你的代码，提出修改建议
5. 你根据建议修改，push 新的提交
6. 同事点击 "Approve"（通过）
7. 代码被合并到 main 分支
```

### 实际操作

```bash
# 1. 创建分支并写代码
git checkout -b feature/add-evaluation
echo "def evaluate(model, dataloader): pass" > src/evaluate.py
git add . && git commit -m "添加模型评估模块"

# 2. 推送分支到 GitHub
git push -u origin feature/add-evaluation
```

然后打开 GitHub，你会看到一个提示：

> feature/add-evaluation had recent pushes — **Compare & pull request**

点击这个按钮，填写 PR 的标题和描述，点击 **Create pull request** 就完成了。

对于个人项目，你可以自己审查后直接在 GitHub 页面上点 **Merge pull request** 合并。

---

## 本章自检

完成以下检查，确认你掌握了 Git 基础：

- [ ] 能从头创建一个 Git 仓库
- [ ] 能用 `add` → `commit` 提交代码
- [ ] 能用 `git diff` 查看修改了什么
- [ ] 会写 `.gitignore` 文件
- [ ] 能把代码推送到 GitHub
- [ ] 能用 `git clone` 下载别人的项目
- [ ] 理解分支的概念，能创建和合并分支
- [ ] 遇到合并冲突不慌，知道怎么解决

:::tip 全部打勾了？
恭喜你完成了 Git 的学习！这些技能会贯穿你整个 AI 学习之旅。接下来我们来配置 Python 开发环境。
:::
