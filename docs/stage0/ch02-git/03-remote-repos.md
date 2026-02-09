---
title: "远程仓库"
sidebar_position: 3
description: "将代码推送到 GitHub，学会远程协作"
---

# 远程仓库

## 学习目标

- 在 GitHub 上创建仓库
- 配置 SSH 连接（再也不用输密码）
- 掌握 `git push`、`git pull`、`git clone`
- 写一个好的 README.md

---

## 为什么需要远程仓库？

到目前为止，你的 Git 记录都只存在于你自己的电脑上。如果电脑硬盘坏了，所有代码和历史就全没了。

**远程仓库**就是把你的代码存一份到云端（通常是 GitHub）。它有三个核心好处：

1. **备份**——电脑坏了也不怕，代码在云端
2. **协作**——多人可以往同一个仓库提交代码
3. **展示**——你的 GitHub 就是你的代码作品集，求职面试会看

---

## 注册 GitHub

1. 打开 [github.com](https://github.com)
2. 点击 **Sign up**，用邮箱注册
3. 用户名建议用英文，简洁好记（比如 `zhangsan-dev`），这会出现在你的项目链接里

:::info 国内用户
如果 GitHub 访问慢，可以同时注册一个 [Gitee](https://gitee.com)（码云）作为备用。操作方式几乎一样。但建议主力用 GitHub——它是全球最大的开源平台，求职时更有价值。
:::

---

## 配置 SSH 连接

每次 push 代码到 GitHub 都需要验证身份。SSH 是最方便的方式——配置一次，之后再也不用输密码。

### 第一步：生成 SSH 密钥

```bash
ssh-keygen -t ed25519 -C "你的邮箱@example.com"
```

会提示你几个问题，全部按回车（使用默认值）就行：

```
Enter file in which to save the key (/Users/你的用户名/.ssh/id_ed25519): [回车]
Enter passphrase (empty for no passphrase): [回车]
Enter same passphrase again: [回车]
```

### 第二步：复制公钥

```bash
# macOS
cat ~/.ssh/id_ed25519.pub | pbcopy

# Linux
cat ~/.ssh/id_ed25519.pub
# 然后手动复制输出的内容

# Windows PowerShell
Get-Content ~/.ssh/id_ed25519.pub | Set-Clipboard
```

输出类似这样（这是公钥，可以安全分享）：

```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAI... 你的邮箱@example.com
```

### 第三步：添加到 GitHub

1. 打开 [github.com/settings/keys](https://github.com/settings/keys)
2. 点击 **New SSH key**
3. Title 填 "My Laptop"（或任意名字，方便你认识是哪台电脑）
4. Key 栏粘贴你刚才复制的公钥
5. 点击 **Add SSH key**

### 第四步：验证连接

```bash
ssh -T git@github.com
```

如果看到：

```
Hi zhangsan! You've successfully authenticated, but GitHub does not provide shell access.
```

就说明配置成功了！

:::tip SSH 密钥的原理（选读）
SSH 密钥是一对"钥匙"：
- **私钥**（`id_ed25519`）存在你电脑上，绝对不能给任何人
- **公钥**（`id_ed25519.pub`）放在 GitHub 上

每次你 push 代码，GitHub 用公钥验证"这个人确实拥有对应的私钥"，验证通过就允许操作。这比输密码既安全又方便。
:::

---

## 创建远程仓库并推送

### 案例：把我们之前的 AI 项目推送到 GitHub

#### 方式一：先在 GitHub 上创建仓库，再关联本地项目

**第一步：在 GitHub 上创建仓库**

1. 打开 [github.com/new](https://github.com/new)
2. Repository name 填 `ai-image-classifier`
3. Description 填 "一个使用 CNN 的简单图像分类项目"
4. 选择 **Public**（公开，让别人也能看到你的作品）
5. **不要**勾选 "Add a README file"（我们本地已经有了）
6. 点击 **Create repository**

**第二步：把本地仓库关联到 GitHub**

GitHub 会显示一段命令，我们需要的是"push an existing repository"那段：

```bash
cd ai-image-classifier

# 关联远程仓库（把 zhangsan 换成你的 GitHub 用户名）
git remote add origin git@github.com:zhangsan/ai-image-classifier.git

# 把本地代码推送到 GitHub
git push -u origin main
```

`git remote add origin` 的意思是：给远程仓库起一个名字叫 `origin`（这是约定俗成的名字），地址是后面那个 URL。

`-u origin main` 的意思是：把本地的 `main` 分支和远程的 `main` 分支关联起来。以后只需要 `git push` 就行，不用再写完整命令。

**第三步：验证**

刷新 GitHub 页面，你应该能看到你的代码、提交历史、README 都在上面了。

#### 方式二：先 clone 空仓库，再往里加文件

如果你还没有本地代码，可以反过来操作：

```bash
# 从 GitHub 克隆一个空仓库（或别人的项目）
git clone git@github.com:zhangsan/my-new-project.git
cd my-new-project

# 在里面写代码...
echo "print('hello')" > main.py

# 提交并推送
git add .
git commit -m "添加主程序"
git push
```

---

## 日常推送和拉取

关联好远程仓库之后，日常操作就很简单了：

### git push：把本地新提交推送到远程

```bash
# 写了新代码
echo "新功能" >> src/utils.py
git add .
git commit -m "添加数据预处理函数"

# 推送到 GitHub
git push
```

### git pull：把远程的更新拉取到本地

```bash
# 假设你在另一台电脑上（或同事）修改了代码并推送到了 GitHub
# 你需要把最新的代码拉下来
git pull
```

### 实际工作中的节奏

```bash
# 每天开始工作前：先拉最新代码
git pull

# 写代码、做修改...

# 完成一个功能后：提交并推送
git add .
git commit -m "完成数据增强模块"
git push

# 继续写代码...

# 又完成一个功能
git add .
git commit -m "添加训练日志记录功能"
git push
```

---

## git clone：下载别人的项目

这可能是你最先会用到的 Git 操作——从 GitHub 上下载一个开源项目：

```bash
# 克隆一个 AI 相关的开源项目
git clone git@github.com:ultralytics/yolov5.git
cd yolov5
ls
```

`git clone` 做了三件事：
1. 创建一个和项目同名的文件夹
2. 把所有代码和完整的历史记录下载下来
3. 自动配置好远程仓库关联

### 克隆后的常用操作

```bash
# 查看这个项目的提交历史
git log --oneline -10    # 看最近 10 条

# 查看有哪些分支
git branch -a

# 查看远程仓库地址
git remote -v
```

---

## 写好 README.md

每个 GitHub 项目的首页会自动展示 `README.md` 的内容。一个好的 README 是你作品集的门面。

### AI 项目 README 模板

```markdown
# 项目名称

一句话介绍这个项目做了什么。

## 📋 项目简介

用 2-3 句话详细说明项目背景、解决的问题、使用的方法。

## ✨ 主要特性

- 特性1：XXX
- 特性2：XXX
- 特性3：XXX

## 🛠️ 技术栈

- Python 3.11
- PyTorch 2.0
- 其他用到的库

## 🚀 快速开始

### 环境安装

​```bash
git clone git@github.com:yourname/project.git
cd project
pip install -r requirements.txt
​```

### 运行

​```bash
python src/train.py
​```

## 📊 实验结果

| 模型 | 准确率 | 训练时间 |
|------|:-----:|:------:|
| SimpleCNN | 85.2% | 10 min |
| ResNet18 | 92.7% | 30 min |

## 📁 项目结构

​```
project/
├── data/              # 数据文件
├── models/            # 训练好的模型
├── src/
│   ├── model.py       # 模型定义
│   ├── train.py       # 训练脚本
│   └── utils.py       # 工具函数
├── requirements.txt
└── README.md
​```

## 📄 License

MIT
```

### 案例：给我们的项目更新 README

```bash
# 用上面的模板写一个 README（内容简化版）
cat > README.md << 'READMEEOF'
# AI 图像分类器

使用 CNN 对 CIFAR-10 数据集进行图像分类的入门项目。

## 技术栈

- Python 3.11
- PyTorch 2.0

## 快速开始

```bash
git clone git@github.com:zhangsan/ai-image-classifier.git
cd ai-image-classifier
pip install -r requirements.txt
python src/train.py
```

## 项目结构

```
ai-image-classifier/
├── data/              # 数据文件（git忽略）
├── models/            # 模型权重（git忽略）
├── src/
│   ├── model.py       # CNN 模型定义
│   ├── train.py       # 训练脚本
│   └── utils.py       # 工具函数
├── .gitignore
├── requirements.txt
└── README.md
```
READMEEOF

git add README.md
git commit -m "完善 README：添加项目说明和使用方法"
git push
```

---

## 常见问题

### push 被拒绝（rejected）

```
! [rejected]        main -> main (fetch first)
```

这意味着远程仓库有你本地没有的提交（可能是你在另一台电脑上改的，或者同事推送了新代码）。解决方法：

```bash
git pull          # 先拉取远程的更新
git push          # 然后再推送
```

### clone 很慢

国内 clone GitHub 项目可能很慢。几个解决方案：

```bash
# 方案1：只克隆最新版本（不要完整历史），大幅加速
git clone --depth 1 git@github.com:xxx/yyy.git

# 方案2：使用镜像加速
# 将 github.com 替换为镜像站点（具体镜像地址请搜索最新可用的）
```

### push 到了错误的仓库

```bash
# 查看当前关联的远程仓库
git remote -v

# 修改远程仓库地址
git remote set-url origin git@github.com:正确的用户名/正确的仓库名.git
```

---

## 小结

| 命令 | 用途 | 什么时候用 |
|------|------|----------|
| `git remote add origin URL` | 关联远程仓库 | 新项目第一次推送前 |
| `git push` | 推送本地提交到远程 | 完成功能后 |
| `git pull` | 拉取远程更新到本地 | 开始工作前 |
| `git clone URL` | 下载远程仓库到本地 | 第一次获取项目 |

日常节奏：**pull → 写代码 → add → commit → push**。就这么简单。
