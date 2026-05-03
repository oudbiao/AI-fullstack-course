---
sidebar_position: 0
title: "AI 全栈学习教程 —— 人工智能发展史漫画书"
description: "AI 全栈学习教程首页，用 15 页中文科普漫画讲清人工智能发展史：图灵测试、达特茅斯会议、感知机、专家系统、反向传播、CNN、机器学习、ImageNet、Transformer、GPT、RAG 与 AI Agent。"
keywords: [AI全栈学习教程, 人工智能发展史, AI历史漫画, 人工智能学习, 机器学习入门, 深度学习, Transformer, GPT, RAG, AI Agent, 零基础学AI, 自学课程]
---

# AI 全栈学习教程：人工智能发展史漫画书

欢迎来到这套 AI 全栈学习教程。首页先不急着列课程目录，我们先用一组漫画把人工智能从“机器能否思考”的问题，讲到今天的 RAG、工具调用和 Agent。

这 15 页漫画按“问题 -> 解决方案 -> 新问题”的方式展开。你不需要先背论文名，只要跟着故事看：每一代 AI 为什么出现，它解决了什么，又留下了什么新问题，下一代技术为什么必须登场。

## 怎么读这组漫画

| 读图线索 | 你要关注什么 |
|---|---|
| 历史时间 | 这一页发生在 AI 历史的哪个阶段 |
| 代表人物 | 谁提出了关键想法或推动了关键系统 |
| 成功点 | 这个阶段第一次把什么做成了 |
| 遇到的问题 | 为什么它还不够，为什么下一阶段会出现 |
| 技术比喻 | 用小黑板、机器、箭头、分镜理解核心技术 |

---

## 第 1 页：图灵与 AI 梦想的起点（1936–1950）

![图灵与 AI 梦想的起点漫画](/img/course/homepage-ai-history-comic-01-turing.png)

这一页讲的是 AI 的原始问题：机器能不能表现出智能。图灵把“机器思考”从哲学争论，变成一个可以通过对话测试讨论的问题。

---

## 第 2 页：达特茅斯会议，AI 正式诞生（1956）

![达特茅斯会议 AI 正式诞生漫画](/img/course/homepage-ai-history-comic-02-dartmouth.png)

这一页讲的是 AI 作为学科的命名时刻。研究者第一次把“制造智能机器”当成正式科学目标，也开启了符号主义和逻辑推理的早期乐观时代。

---

## 第 3 页：感知机的兴奋与第一次低谷（1957–1969）

![感知机兴奋与神经网络第一次低谷漫画](/img/course/homepage-ai-history-comic-03-perceptron.png)

这一页讲的是神经网络第一次从希望走向受挫：感知机能从数据里调整权重，但单层结构无法解决 XOR 这类非线性问题。

---

## 第 4 页：专家系统时代，规则的辉煌与崩塌（1970s–1980s）

![专家系统时代漫画](/img/course/homepage-ai-history-comic-04-expert-systems.png)

这一页讲的是规则系统的黄金时代。专家系统在窄领域很有用，但规则越写越多、越写越难维护，说明真实世界很难全部靠人工规则覆盖。

---

## 第 5 页：反向传播，神经网络重新点火（1986）

![反向传播重新点火漫画](/img/course/homepage-ai-history-comic-05-backprop.png)

这一页讲的是多层神经网络重新被点亮。反向传播让错误从输出层传回前面每一层，告诉每个参数应该怎么微调。

---

## 第 6 页：手写数字识别与 CNN 的第一次实用成功（1989–1998）

![LeNet 与 CNN 实用成功漫画](/img/course/homepage-ai-history-comic-06-lenet.png)

这一页讲的是 CNN 第一次在真实工业场景里证明价值。小滤镜在图片上滑动，逐层识别边缘、笔画和数字，为后来的视觉革命埋下伏笔。

---

## 第 7 页：统计机器学习时代，数据取代手写规则（1990s–2000s）

![统计机器学习时代漫画](/img/course/homepage-ai-history-comic-07-statistical-ml.png)

这一页讲的是 AI 从“写规则”转向“从数据学习”。SVM、决策树、随机森林和 Boosting 让表格数据、分类、搜索、广告等任务变得更可靠。

---

## 第 8 页：ImageNet 与 AlexNet，深度学习爆发（2009–2012）

![ImageNet 与 AlexNet 深度学习爆发漫画](/img/course/homepage-ai-history-comic-08-imagenet-alexnet.png)

这一页讲的是深度学习真正爆发的转折点。大规模数据、GPU 和 CNN 聚在一起，让模型能从原始图片中自动学习特征。

---

## 第 9 页：ResNet，为什么要把 X 加回来（2015）

![ResNet 残差连接漫画](/img/course/homepage-ai-history-comic-09-resnet.png)

这一页讲的是深层网络为什么需要“捷径”。ResNet 让信息可以绕过复杂层，把原始输入 X 加回来，使更深的网络变得可训练。

---

## 第 10 页：RNN 与 LSTM，语言序列的早期主力（1997–2014）

![RNN 与 LSTM 序列模型漫画](/img/course/homepage-ai-history-comic-10-rnn-lstm.png)

这一页讲的是机器如何一个词一个词读句子。RNN/LSTM 曾是语言、语音和时间序列的主力，但顺序计算慢，长距离依赖也容易变弱。

---

## 第 11 页：Attention，别死记，直接看重点（2014）

![Attention 机器翻译漫画](/img/course/homepage-ai-history-comic-11-attention.png)

这一页讲的是注意力机制的直觉：生成当前词时，不再把整句压成一个向量，而是直接看输入句子里最相关的位置。

---

## 第 12 页：Transformer，Attention Is All You Need（2017）

![Transformer 自注意力漫画](/img/course/homepage-ai-history-comic-12-transformer.png)

这一页讲的是现代大模型的架构底座。Transformer 去掉 RNN，用 self-attention 让 token 之间直接交流，并支持更高效的并行训练。

---

## 第 13 页：BERT 与 GPT，预训练时代开始（2018–2020）

![BERT 与 GPT 预训练漫画](/img/course/homepage-ai-history-comic-13-bert-gpt.png)

这一页讲的是大规模预训练的开始。BERT 像阅读理解学生，GPT 像写作机器人，它们让 AI 从“每个任务单独训练”走向基础模型。

---

## 第 14 页：SFT、RLHF 与 ChatGPT，续写器变成助手（2022）

![SFT RLHF 与 ChatGPT 漫画](/img/course/homepage-ai-history-comic-14-rlhf-chatgpt.png)

这一页讲的是大语言模型如何从续写器变成助手。SFT 教模型按示例回答，RLHF 让模型学习人类更喜欢哪种回答。

---

## 第 15 页：RAG、工具调用和 Agent，AI 走向真实任务（2023 至今）

![RAG 工具调用与 Agent 漫画](/img/course/homepage-ai-history-comic-15-rag-agent.png)

这一页讲的是今天的 AI 应用正在往哪里走：模型不只回答问题，还会查资料、调工具、写计划、观察结果，并在安全边界内完成任务。

---

## 看完漫画后，继续学习

如果你是第一次学习 AI，建议按下面顺序继续：

| 目标 | 下一步 |
|---|---|
| 想从零开始做 AI 应用 | 进入 [开发者工具基础](/ch01-tools)，先把环境、命令行和 Git 跑通 |
| 想理解模型为什么能学习 | 从 [AI 数学最小必要基础](/ch04-ai-math) 和 [机器学习入门到实战](/ch05-machine-learning) 开始 |
| 想做大模型应用 | 重点学习 [大模型原理、Prompt 与微调](/ch07-llm-principles)、[LLM 应用开发与 RAG](/ch08-rag) 和 [AI Agent 与智能体系统](/ch09-agent) |
| 想继续看完整时间线 | 查看 [AI 重要论文与算法时间线](/appendix/ai-milestones) |

真正学 AI，不是背下所有历史节点，而是理解每次突破背后的同一个节奏：旧问题暴露出来，新的解决方案出现，然后新的边界继续推动下一轮创新。
