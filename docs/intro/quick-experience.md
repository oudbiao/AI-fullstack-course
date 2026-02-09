

# 🚀 30分钟 AI 快速体验

> **目标：** 在正式学习前，先亲手玩一下 AI，感受它能做什么  
> **时间：** 30 分钟 ～ 1 小时  
> **需要准备：** 一个 Google 账号（用来打开 Colab），其他什么都不用装

## 为什么要先体验？

很多人学编程，一上来就啃语法、背概念，学了两周还不知道自己在干什么，然后就放弃了。

我们反过来——**先玩，再学。** 你先看看 AI 能干什么，觉得"这也行？"，然后带着好奇心开始系统学习。

这 30 分钟里你会体验三件事：

1. **让 AI 识别图片** —— 你给它一张照片，它告诉你照片里是什么
2. **和 AI 对话** —— 你说一句话，它接着往下编
3. **让 AI 画画** —— 你用文字描述一个画面，它给你生成出来

:::tip 不需要任何编程基础
下面的代码你现在不需要理解，只需要复制粘贴运行就行。等你学完第一阶段（Python 基础），回头再看这些代码，就会觉得非常简单。
:::

---

## 体验1：让 AI 看懂图片（10分钟）

### 第一步：打开 Google Colab

1. 在浏览器中打开 [Google Colab](https://colab.research.google.com)
2. 点击左上角的 **「新建笔记本」**
3. 你会看到一个类似记事本的界面，里面有一个输入框（叫做"代码单元格"）

### 第二步：安装需要的库

在代码单元格里粘贴下面的代码，然后按 `Shift + Enter` 运行：

```python
!pip install transformers torch pillow requests -q
```

等大约 1 分钟，看到输出不报红色错误就行。

### 第三步：运行图像识别

点击左上角的 **「+ 代码」** 新建一个单元格，粘贴以下代码并运行：

```python
from transformers import pipeline
from PIL import Image
import requests

# 加载一个图像分类模型（第一次运行要下载模型，稍等一下）
classifier = pipeline("image-classification", model="google/vit-base-patch16-224")

# 用一张网上的狗狗图片来测试
url = "https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/YellowLabradorLooking_new.jpg/1200px-YellowLabradorLooking_new.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# 让 AI 识别这张图片
results = classifier(image)

# 看看结果
print("🤖 AI 认为这张图片是：")
for r in results[:3]:
    print(f"  {r['label']:30s} 置信度: {r['score']:.1%}")
```

### 你应该看到类似这样的输出

```
🤖 AI 认为这张图片是：
  Labrador retriever              置信度: 89.3%
  golden retriever                置信度: 6.2%
  kuvasz                          置信度: 1.1%
```

> 🎉 **想一想：** 你没有教 AI 什么是拉布拉多，也没有给它标注图片，它怎么就认出来了？因为这个模型已经在 1400 万张图片上"学习"过了。这种"先大规模学习、再识别新事物"的过程，就是**深度学习**的核心思想——也是本课程要教你的东西。

### 试试换成你自己的图片

把 `url` 换成任何网上图片的链接，看看 AI 能不能认出来：

```python
url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"
image = Image.open(requests.get(url, stream=True).raw)
results = classifier(image)

print("🤖 AI 认为这张图片是：")
for r in results[:3]:
    print(f"  {r['label']:30s} 置信度: {r['score']:.1%}")
```

---

## 体验2：和 AI 对话（10分钟）

### 方式A：在 Colab 里运行一个小模型（免费）

新建一个代码单元格，粘贴运行：

```python
from transformers import pipeline

# 加载一个文本生成模型
generator = pipeline("text-generation", model="gpt2")

# 给一个开头，让 AI 接着写
prompt = "The future of artificial intelligence is"
result = generator(prompt, max_length=80, num_return_sequences=1)

print("📝 你给的开头：", prompt)
print()
print("🤖 AI 接着写：")
print(result[0]['generated_text'])
```

:::info 关于这个模型
GPT-2 是 2019 年的模型，比现在的 ChatGPT 弱很多，生成的内容可能不太通顺。但它帮你理解了一个关键原理——AI 写文章其实就是不断预测"下一个最可能出现的词是什么"。ChatGPT 也是这个原理，只不过模型大了几百倍、训练数据多了几百倍。
:::

### 方式B：直接体验最新大模型（推荐）

如果你想感受最强大的 AI 对话能力，打开以下任意一个（都免费）：

| 产品 | 网址 | 特点 |
|------|------|------|
| **ChatGPT** | [chat.openai.com](https://chat.openai.com) | 全球最知名，英文能力最强 |
| **Claude** | [claude.ai](https://claude.ai) | 长文本理解强，中文也不错 |
| **通义千问** | [tongyi.aliyun.com](https://tongyi.aliyun.com) | 阿里出品，国内直接访问 |
| **Kimi** | [kimi.moonshot.cn](https://kimi.moonshot.cn) | 支持超长上下文 |
| **DeepSeek** | [chat.deepseek.com](https://chat.deepseek.com) | 开源模型，性价比高 |

试着问它一个有挑战性的问题：

```
请用 Python 写一个函数，计算斐波那契数列的第 n 项。
要求：
1. 用递归实现一个版本
2. 用动态规划实现一个版本
3. 对比两种方式的效率差异
```

> 🎉 **想一想：** AI 不仅能聊天，还能写代码、翻译、总结文档、分析数据……在学完本课程后，你将能自己开发这样的 AI 应用，甚至构建能自主调用工具、自己做决策的 AI Agent。

---

## 体验3：让 AI 画画（10分钟）

### 操作步骤（无需写代码）

1. 打开以下任意一个 AI 绘画工具：
   - [Hugging Face Spaces 上的 Stable Diffusion](https://huggingface.co/spaces/stabilityai/stable-diffusion)
   - [LiblibAI](https://www.liblib.art/)（国内可直接访问）
   - 或搜索 "AI 在线绘画" 找其他工具

2. 在输入框中输入英文描述词（叫做 **Prompt**）：

```
a cute robot reading a book in a cozy library, digital art, warm lighting
```

3. 点击 **Generate**，等待 10-30 秒

4. AI 会为你生成一张全新的图片——这张图片世界上从未存在过，是 AI "想象"出来的

### 更多 Prompt 可以试试

```
a futuristic city at sunset, cyberpunk style, neon lights, rain
```

```
an astronaut riding a horse on the moon, oil painting style
```

```
a traditional Chinese ink painting of mountains and rivers, misty, elegant
```

:::tip Prompt 的奥秘
你会发现，描述词写得越具体，生成的图片质量越高。这种用文字控制 AI 输出的技巧叫做 **Prompt Engineering（提示词工程）**，是本课程第八A阶段的重要内容，也是当前 AI 行业最实用的技能之一。
:::

> 🎉 **想一想：** 这就是 AIGC（AI Generated Content，AI 生成内容）。你只需要用文字描述想要的画面，AI 就能"画"出来。本课程的第十阶段会教你这背后的扩散模型原理，以及如何微调模型生成你想要的风格。

---

## ✅ 体验完成！回顾一下

恭喜你完成了 AI 快速体验！花 30 分钟，你已经亲手感受了 AI 的三大核心能力：

| 你体验了什么 | 背后的技术 | 课程中哪里学 |
|------------|----------|------------|
| 图像识别 | 卷积神经网络 + 预训练模型 | 第五阶段（深度学习）+ 第六阶段（CV） |
| 文本对话 | 大语言模型 + Transformer 架构 | 第八A阶段（大模型原理） |
| 图片生成 | 扩散模型（Diffusion Model） | 第十阶段（AIGC） |

:::note 术语不用记
CNN、Transformer、Diffusion……这些词现在看着完全陌生没关系。等你一步步学过来，每一个都会变得清清楚楚。现在你只需要记住一件事——**这些你都能学会**。
:::

