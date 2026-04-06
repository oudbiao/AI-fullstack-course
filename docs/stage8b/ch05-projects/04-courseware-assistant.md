---
title: "5.5 项目：知识库驱动的课件生成助手"
sidebar_position: 24
description: "围绕 PDF/Word/PPT 资料库、RAG 检索、外部资料补充、例题抽取和模板化 Word 生成，建立一个真正像产品的课件助手项目闭环。"
keywords: [courseware assistant, knowledge base, word generation, ppt parsing, document parsing, rag project]
---

# 项目：知识库驱动的课件生成助手

:::tip 本节定位
这个项目比普通知识库问答更进一步。  
它不是只回答问题，而是要真正产出：

- 一份符合格式要求的 Word 课件

所以它特别适合训练这些系统能力一起工作：

- 文档解析
- 知识检索
- 例题抽取
- 结构化输出
- 模板化文档生成
:::

## 学习目标

- 学会把“主题 -> 查资料 -> 抽例题 -> 生成课件”组织成完整流程
- 学会定义一个知识库驱动课件系统的最小项目边界
- 学会把内部知识库和外部资料补充分开设计
- 学会把这个项目做成一个有产品感的作品级系统

---

## 先建立一张地图

这个项目最适合按“知识入库 -> 检索 -> 结构化生成 -> 模板导出”来理解：

```mermaid
flowchart LR
    A["PDF / Word / PPT 资料"] --> B["文档解析与知识抽取"]
    B --> C["知识库 / 检索"]
    D["外部资料检索"] --> C
    C --> E["生成课件结构"]
    E --> F["套 Word 模板导出"]
```

所以这个项目真正想解决的是：

- 用户只给一个主题时，系统怎样自动去找资料、找例题、再按模板写出来

## 一、项目题目怎么收窄？

一个最稳的起点通常是：

> **做一个“知识库驱动的数学课件助手”，用户输入主题，系统自动生成一份包含知识点、例题和练习的 Word 初稿。**

为什么这个范围合适？

- 主题清楚
- 资料形态清楚
- 例题和知识点都能从文档里抽
- Word 输出目标明确

不建议一开始就做成：

- 所有学科通用
- 自动生成 PPT + Word + 讲稿 + 配音

这样很容易把项目主线冲散。

## 二、一个更适合新人的总类比

你可以把这个系统理解成：

- 一个会先翻资料、再整理提纲、最后替你起草课件的备课助理

它不是直接凭空写，而是：

1. 先查内部资料
2. 必要时再补外部资料
3. 再从资料里挑知识点和例题
4. 最后按固定格式写成课件

这个类比很重要，因为它会帮新人避免把项目想成：

- “直接让模型写一篇 Word”

## 三、最小系统闭环长什么样？

1. 文档入库
2. 解析正文、标题和例题
3. 用户输入主题
4. 系统检索内部知识块
5. 必要时补外部资料
6. 生成结构化课件对象
7. 套模板导出 Word

只要这 7 步跑顺，这个项目就已经非常像真正产品了。

## 四、先跑一个最小工作流示例

```python
knowledge_base = [
    {"topic": "折扣应用题", "content_type": "concept", "text": "折扣 = 原价 × 折扣率"},
    {"topic": "折扣应用题", "content_type": "example", "text": "商品原价 100 元，打 8 折后价格是多少？"},
    {"topic": "折扣应用题", "content_type": "exercise", "text": "一件衣服原价 80 元，打 7 折后是多少元？"},
]


def retrieve_internal(topic):
    return [item for item in knowledge_base if item["topic"] == topic]


def retrieve_external(topic):
    # 这里只做一个最小模拟
    return [{"topic": topic, "content_type": "note", "text": f"外部资料补充：{topic} 的常见教学误区。"}]


def build_courseware(topic):
    internal = retrieve_internal(topic)
    external = retrieve_external(topic)
    all_items = internal + external
    return {
        "title": topic,
        "concepts": [x["text"] for x in all_items if x["content_type"] == "concept"],
        "examples": [x["text"] for x in all_items if x["content_type"] == "example"],
        "exercises": [x["text"] for x in all_items if x["content_type"] == "exercise"],
        "notes": [x["text"] for x in all_items if x["content_type"] == "note"],
    }


print(build_courseware("折扣应用题"))
```

### 4.1 这个例子最关键的价值是什么？

它说明这个系统真正有价值的地方，不是只会：

- 查

而是能把查到的内容重新组织成：

- 课件需要的栏目结构

## 五、这个项目最需要哪些能力？

按系统分层看，核心能力是：

### 5.1 文档解析

- PDF / DOCX / PPTX 读取
- 扫描件 OCR
- 标题层级和例题识别

对应课程：
- [文档解析与知识抽取](../ch03-app-dev/07-document-parsing.md)
- [文档处理](../ch01-rag/02-document-processing.md)
- [OCR 文字识别](../../stage6/ch05-advanced/03-ocr.md)

### 5.2 知识库与检索

- 切块
- 元数据
- 主题检索
- 例题召回

对应课程：
- [RAG 基础](../ch01-rag/01-rag-basics.md)
- [向量数据库](../ch01-rag/03-vector-databases.md)
- [检索策略](../ch01-rag/04-retrieval-strategies.md)

### 5.3 结构化输出与模板生成

- 先生成大纲
- 再生成知识点 / 例题 / 练习
- 再套模板导出 Word

对应课程：
- [Prompt 基础](../../stage8a/ch05-prompt/01-prompt-basics.md)
- [结构化输出](../../stage8a/ch05-prompt/03-structured-output.md)
- [模板化文档生成（Word / PPT）](../ch03-app-dev/08-template-doc-generation.md)

### 5.4 工具调用与工作流

- 内部知识库检索
- 外部资料补充
- 模板渲染
- 导出文件

对应课程：
- [函数调用实践](../ch03-app-dev/03-function-calling.md)
- [对话系统与多轮管理](../ch03-app-dev/05-dialog-system.md)
- [Plan-and-Execute](../../stage9/ch02-reasoning/04-plan-and-execute.md)

## 六、这个项目最该怎么评估？

最值得先看的不是“写出来像不像”，而是：

1. 检索内容对不对
2. 例题抽得对不对
3. 结构有没有符合模板
4. 引用和来源能不能回溯

你可以先把评估拆成：

| 维度 | 更像在看什么 |
|---|---|
| 检索质量 | 主题资料和例题有没有找对 |
| 结构正确性 | 标题、知识点、例题、练习有没有放对位置 |
| 来源可追溯性 | 每一段内容能不能回溯到文档来源 |
| 模板符合度 | 最终 Word 是否符合格式规范 |

## 七、一个新人可直接照抄的推进顺序

第一次做这个项目时，更稳的顺序通常是：

1. 先只做内部知识库
2. 先不加外部资料
3. 先生成结构化 JSON
4. 再把 JSON 套到 Word 模板里
5. 最后再补外部检索、工具编排和更复杂的 Agent 逻辑

这样会比一上来就做“全自动备课 Agent”更容易把系统做稳。

## 八、如果把它做成作品集，最值得展示什么？

最值得展示的通常不是：

- “我能生成 Word”

而是：

1. 原始资料长什么样
2. 解析后的知识块长什么样
3. 用户输入主题后检索到了哪些内容
4. 最终课件结构是怎么长出来的
5. Word 模板导出后的结果长什么样

这样别人会更容易看出：

- 你做的是一个知识驱动内容生成系统
- 不只是让模型写了一篇文章

## 这节最该带走什么

- 这个项目最核心的不是“文档输出”，而是“文档知识 -> 结构化课件”的整条链
- 文档解析、RAG、结构化输出、模板渲染缺一块，系统都不稳
- 如果你想做这类系统，先把工作流版做稳，再考虑 Agent 化会更现实
