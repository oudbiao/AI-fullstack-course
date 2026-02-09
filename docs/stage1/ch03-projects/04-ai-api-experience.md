---
title: "1.4 项目：AI API 快速体验"
sidebar_position: 4
description: "通过调用 AI API 体验人工智能的能力"
---

# 项目：AI API 快速体验

## 项目目标

- 了解什么是 AI API，以及它和 AI 模型的关系
- 学会调用 OpenAI 等主流 AI API
- 体验对话生成、文本分析等 AI 能力
- 构建一个简单的 AI 聊天机器人

---

## 什么是 AI API？

在上一个项目中，你学会了自己写 API。而 AI API 就是**别人训练好的 AI 模型，打包成 API 供你调用**。

```
传统方式：你自己训练模型（需要大量数据、GPU、时间）
API 方式：直接调用别人的模型（只需要一个 API Key，几行代码）
```

就像你不需要自己种麦子才能吃面包一样——用 AI API，你可以**直接使用世界上最先进的 AI 能力**。

### 常见的 AI API 服务

| 服务 | 提供方 | 主要能力 |
|------|--------|---------|
| OpenAI API | OpenAI | 对话、文本生成、代码生成 |
| Claude API | Anthropic | 对话、文档分析、推理 |
| 通义千问 API | 阿里云 | 对话、文本理解 |
| 文心一言 API | 百度 | 对话、知识问答 |
| 智谱 API | 智谱AI | 对话、代码生成 |

---

## 第一步：获取 API Key

要使用 AI API，你需要先注册并获取一个 **API Key**——这就像一把钥匙，证明你有权限调用这个 API。

:::info 选择一个 API 服务
以下教程以 OpenAI API 为例。如果你在国内访问有困难，可以选择：
- **智谱 AI**（bigmodel.cn）—— 国内免费额度较大
- **通义千问**（dashscope.aliyun.com）—— 阿里云旗下

这些国产 API 的调用方式和 OpenAI 非常相似，只需要改一下地址和 Key 即可。
:::

### 获取 OpenAI API Key

1. 访问 [platform.openai.com](https://platform.openai.com)
2. 注册/登录账号
3. 进入 API Keys 页面
4. 点击 "Create new secret key"
5. 复制并**安全保存** Key（只会显示一次）

### 安装 OpenAI SDK

```bash
pip install openai
```

### 设置 API Key

```bash
# 方法 1：设置环境变量（推荐）
export OPENAI_API_KEY="sk-你的key"

# 方法 2：在代码中设置（不推荐提交到 Git）
```

:::caution API Key 安全
**永远不要**把 API Key 写在代码里并提交到 GitHub！这相当于把你的密码公开了。正确做法：
1. 使用环境变量
2. 使用 `.env` 文件 + `.gitignore`
:::

---

## 第二步：第一次调用 AI API

```python
from openai import OpenAI

# 创建客户端（自动从环境变量读取 OPENAI_API_KEY）
client = OpenAI()

# 发送一个简单的对话请求
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": "你好！请用一句话介绍 Python 语言。"}
    ]
)

# 获取 AI 的回复
reply = response.choices[0].message.content
print(f"AI 说: {reply}")
```

输出类似：

```
AI 说: Python 是一门简洁优雅、功能强大的高级编程语言，广泛应用于数据分析、人工智能、Web 开发等众多领域。
```

### 理解请求结构

```python
messages = [
    {"role": "system", "content": "你是一个 Python 编程助手。"},  # 系统提示（设定 AI 的角色）
    {"role": "user", "content": "什么是列表推导式？"},           # 用户消息
    {"role": "assistant", "content": "列表推导式是..."},         # AI 之前的回复
    {"role": "user", "content": "能给个例子吗？"},               # 用户的追问
]
```

| 角色 | 含义 |
|------|------|
| `system` | 设定 AI 的行为和角色（可选） |
| `user` | 用户发送的消息 |
| `assistant` | AI 之前的回复（用于多轮对话） |

---

## 第三步：构建交互式聊天机器人

```python
"""
AI 聊天机器人
使用 OpenAI API 实现多轮对话
"""

from openai import OpenAI

def create_chatbot(system_prompt: str = "你是一个友好的 AI 助手。"):
    """创建一个聊天机器人"""
    client = OpenAI()
    messages = [{"role": "system", "content": system_prompt}]

    print("=" * 50)
    print("  AI 聊天机器人")
    print("  输入 'quit' 退出，输入 'clear' 清除对话历史")
    print("=" * 50)

    while True:
        user_input = input("\n你: ").strip()

        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("再见！")
            break
        if user_input.lower() == "clear":
            messages = [{"role": "system", "content": system_prompt}]
            print("🧹 对话历史已清除")
            continue

        # 添加用户消息
        messages.append({"role": "user", "content": user_input})

        try:
            # 调用 API
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.7,    # 控制创造性（0=保守, 2=疯狂）
                max_tokens=1000,    # 最大回复长度
            )

            # 获取回复
            reply = response.choices[0].message.content
            print(f"\nAI: {reply}")

            # 把 AI 的回复也加入历史（实现多轮对话）
            messages.append({"role": "assistant", "content": reply})

            # 显示 Token 使用量
            usage = response.usage
            print(f"\n  [Token 使用: 输入={usage.prompt_tokens}, "
                  f"输出={usage.completion_tokens}, "
                  f"总计={usage.total_tokens}]")

        except Exception as e:
            print(f"\n❌ 调用失败: {e}")
            messages.pop()  # 移除失败的用户消息

if __name__ == "__main__":
    create_chatbot("你是一个专业的 Python 编程导师，用简洁通俗的语言回答问题。")
```

---

## 第四步：实用 AI 工具

### 工具 1：AI 代码审查助手

```python
def review_code(code: str) -> str:
    """让 AI 审查你的代码"""
    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "你是一个资深 Python 代码审查专家。"
                          "请审查用户的代码，指出问题并给出改进建议。"
                          "用中文回复，格式清晰。"
            },
            {
                "role": "user",
                "content": f"请审查以下代码:\n\n```python\n{code}\n```"
            }
        ],
        temperature=0.3  # 代码审查用低温度，更严谨
    )

    return response.choices[0].message.content

# 测试
code = """
def calc(l):
    s = 0
    for i in range(len(l)):
        s = s + l[i]
    return s / len(l)
"""

print(review_code(code))
```

### 工具 2：AI 文本摘要工具

```python
def summarize(text: str, max_sentences: int = 3) -> str:
    """让 AI 生成文本摘要"""
    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": f"请用不超过 {max_sentences} 句话概括以下文本的核心内容。用中文回复。"
            },
            {"role": "user", "content": text}
        ],
        temperature=0.3
    )

    return response.choices[0].message.content

# 使用
long_text = """
Python 是一种广泛使用的高级编程语言，由 Guido van Rossum 创建并于 1991 年首次发布。
Python 的设计理念强调代码可读性和简洁性，使用显著的空白缩进来定义代码块。
它支持多种编程范式，包括过程化、面向对象和函数式编程。
Python 拥有一个大型标准库，并且有丰富的第三方库生态系统。
在人工智能、机器学习、数据科学、Web 开发等领域，Python 都是最受欢迎的编程语言之一。
"""

print(summarize(long_text))
```

### 工具 3：AI 翻译工具

```python
def translate(text: str, target_lang: str = "英文") -> str:
    """AI 翻译"""
    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": f"你是一个专业翻译。请将用户输入的文本翻译成{target_lang}。"
                           "只返回翻译结果，不要添加任何解释。"
            },
            {"role": "user", "content": text}
        ],
        temperature=0.3
    )

    return response.choices[0].message.content

print(translate("人工智能正在改变世界"))
print(translate("Hello, how are you?", "中文"))
```

---

## 第五步：使用国产 AI API（替代方案）

如果你使用国产 AI API，代码结构几乎一样，只需要修改 API 地址和 Key。

### 智谱 AI（GLM 模型）

```bash
pip install zhipuai
```

```python
from zhipuai import ZhipuAI

client = ZhipuAI(api_key="your_api_key")

response = client.chat.completions.create(
    model="glm-4-flash",
    messages=[
        {"role": "user", "content": "你好！请介绍一下 Python"}
    ]
)

print(response.choices[0].message.content)
```

### 通用的 OpenAI 兼容接口

很多国产 API 都兼容 OpenAI 的接口格式，只需要修改 `base_url`：

```python
from openai import OpenAI

# 使用不同的 API 服务
client = OpenAI(
    api_key="your_api_key",
    base_url="https://api.your-provider.com/v1"  # 替换为对应服务的地址
)

# 之后的代码完全一样！
response = client.chat.completions.create(
    model="model_name",
    messages=[{"role": "user", "content": "你好"}]
)
```

---

## 理解 AI API 的关键参数

| 参数 | 含义 | 建议值 |
|------|------|--------|
| `model` | 使用哪个模型 | `gpt-3.5-turbo`（便宜），`gpt-4`（更强） |
| `temperature` | 创造性/随机性 | 0.0-0.3（事实性），0.7-1.0（创造性） |
| `max_tokens` | 最大输出长度 | 根据需要设置 |
| `messages` | 对话历史 | 包含完整的对话上下文 |
| `stream` | 是否流式输出 | `True` 实现打字机效果 |

### Token 和费用

AI API 按 **Token** 计费。Token 大约等于一个词或几个汉字。

```python
# 查看 Token 使用量
usage = response.usage
print(f"输入 Token: {usage.prompt_tokens}")
print(f"输出 Token: {usage.completion_tokens}")
print(f"总 Token: {usage.total_tokens}")
```

:::tip 控制成本
- 使用 `gpt-3.5-turbo` 而不是 `gpt-4`（便宜 10-20 倍）
- 控制 `max_tokens`，避免不必要的长回复
- 优化 system prompt，减少输入 Token
- 定期清理对话历史，避免累积太多 Token
:::

---

## 扩展挑战

### 挑战 1：流式输出

实现打字机效果（AI 回复一个字一个字出现）：

```python
# 提示：使用 stream=True 参数
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=messages,
    stream=True  # 开启流式输出
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### 挑战 2：结合 FastAPI

把 AI 聊天功能包装成 API，让其他人通过 HTTP 请求来使用你的 AI 机器人。

### 挑战 3：角色扮演

创建不同角色的 AI 助手（Python 导师、英语老师、面试官），让用户选择。

### 挑战 4：本地知识库

让 AI 先读取一个本地文件（比如你的笔记），然后基于文件内容回答问题。

---

## 项目自查清单

- [ ] 成功获取了 API Key 并安全存储
- [ ] 能成功调用 AI API 并获得回复
- [ ] 实现了多轮对话功能
- [ ] 构建了至少一个实用工具（代码审查/摘要/翻译）
- [ ] 有异常处理（网络错误、API 错误）
- [ ] API Key 没有硬编码在代码里

---

## 阶段总结

恭喜你完成了第一阶段的所有学习！回顾一下你掌握的技能：

| 章节 | 掌握的技能 |
|------|-----------|
| Python 基础 | 变量、数据类型、运算符、流程控制、数据结构、函数、模块 |
| Python 进阶 | 面向对象、异常处理、文件操作、函数式编程、生成器、类型注解 |
| 实战项目 | 命令行工具、网络爬虫、Web API、AI API 调用 |

你已经具备了：
- **编程思维**：能把问题拆解为代码逻辑
- **工程能力**：能写出结构清晰、有错误处理的程序
- **实战经验**：完成了 4 个真实项目

:::tip 下一步
第二阶段将进入**数据分析与可视化**——用 NumPy、Pandas、Matplotlib 处理和展示数据。这是 AI 工程师的核心技能，因为 AI 的第一步就是理解数据。带着你在第一阶段打下的 Python 基础，继续前进吧！
:::
