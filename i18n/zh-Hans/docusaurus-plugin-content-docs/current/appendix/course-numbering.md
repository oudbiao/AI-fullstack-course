---
title: "课程编号约定"
description: "说明源码目录 ch01-tools、ch02-python 等编号与网页展示第 1～12 章之间的对应关系，避免维护课程时混淆。"
keywords: [课程编号, 章节目录, 章节编号, 课程维护]
---

# 课程编号约定

![展示章节号与源码目录对应地图](/img/course/appendix-course-numbering-map.png)

![课程维护命名一致性检查图](/img/course/appendix-course-numbering-maintenance-check.png)

:::tip 读图提示
维护课程时，网页标题、sidebar 顺序、源码目录和图片命名要互相对齐。读图时把“展示编号”和“文件路径”分开看，就不容易再出现 chxx 与中文章节号混用的问题。
:::

课程网页面向学习者时，统一使用第 1～12 章的展示编号。源码目录也已经和展示章节号对齐：`ch01-*` 对应第 1 章，`ch02-*` 对应第 2 章，依此类推。

目录名后半段用于说明主题，例如 `ch05-machine-learning` 表示第 5 章机器学习，`ch09-agent` 表示第 9 章 AI Agent。侧边栏里的“主线 1～4”只是学习分组，不作为文件目录层级。

## 对应关系

| 源码目录 | 网页展示章节 | 课程名称 |
|---|---|---|
| `docs/ch01-tools` | 第 1 章 | 开发者工具基础 |
| `docs/ch02-python` | 第 2 章 | Python 编程基础 |
| `docs/ch03-data-analysis` | 第 3 章 | 数据分析与可视化 |
| `docs/ch04-ai-math` | 第 4 章 | AI 数学最小必要基础 |
| `docs/ch05-machine-learning` | 第 5 章 | 机器学习入门到实战 |
| `docs/ch06-deep-learning` | 第 6 章 | 深度学习与 Transformer 基础 |
| `docs/ch07-llm-principles` | 第 7 章 | 大模型原理、Prompt 与微调 |
| `docs/ch08-rag` | 第 8 章 | LLM 应用开发与 RAG |
| `docs/ch09-agent` | 第 9 章 | AI Agent 与智能体系统 |
| `docs/ch10-computer-vision` | 第 10 章 | 计算机视觉 |
| `docs/ch11-nlp` | 第 11 章 | 自然语言处理 |
| `docs/ch12-multimodal` | 第 12 章 | AIGC 与多模态 |

## 写作规则

在页面标题、导读、任务单、附录说明、图片进度记录中，优先使用网页展示章节号，例如“第 5 章机器学习”。

在引用文件路径、代码脚本、图片文件名、内部链接时，使用 `ch05-machine-learning` 这类源码目录名。

不再新增旧式阶段目录或带字母后缀的阶段目录。新增章节、图片和脚本配置时，应优先沿用 `ch01-*` 到 `ch12-*` 的编号体系。

如果一句话里必须同时出现两者，推荐写法是：

```text
第 5 章机器学习（目录 docs/ch05-machine-learning）
```
