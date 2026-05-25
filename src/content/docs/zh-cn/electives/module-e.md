---
title: "E.E Web 前端快速入门"
description: "搭建最小 AI 前端闭环：输入、加载、成功、空输入、错误和重试。"
sidebar:
  order: 5
head:
  - tag: meta
    attrs:
      name: keywords
      content: "frontend, HTML, CSS, JavaScript, fetch, UI, AI product"
---
AI 功能需要用户能操作的界面。最小可用前端要清楚展示输入、加载、成功、空输入、错误和重试状态。

## 先看交互栈

![AI 前端交互栈图](/img/course/elective-ai-frontend-stack.webp)

![AI 前端状态机与体验闭环图](/img/course/elective-ai-frontend-state-machine-map.webp)

把每次模型调用都当成不确定请求：可能慢、失败、返回部分内容，或者需要重试。

## 运行最小静态 AI 页面

保存为 `ai-demo.html`，然后用浏览器打开：

```html
<!doctype html>
<html lang="zh-Hans">
<meta charset="utf-8" />
<title>AI 演示</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 720px; margin: 40px auto; }
  textarea, button, pre { width: 100%; box-sizing: border-box; margin-top: 12px; }
  textarea { min-height: 100px; padding: 12px; }
  button { padding: 10px; }
  pre { min-height: 80px; padding: 12px; background: #f3f4f6; white-space: pre-wrap; }
</style>
<h1>AI 功能演示</h1>
<textarea id="prompt" placeholder="输入一个问题"></textarea>
<button id="run">运行</button>
<p id="status">idle</p>
<pre id="result">结果会显示在这里</pre>
<script>
  const promptEl = document.querySelector("#prompt");
  const statusEl = document.querySelector("#status");
  const resultEl = document.querySelector("#result");

  document.querySelector("#run").addEventListener("click", async () => {
    const text = promptEl.value.trim();
    if (!text) {
      statusEl.textContent = "empty";
      resultEl.textContent = "请先输入提示词。";
      return;
    }

    try {
      statusEl.textContent = "loading";
      resultEl.textContent = "请稍候...";
      await new Promise((resolve) => setTimeout(resolve, 500));
      if (text.toLowerCase().includes("fail")) {
        throw new Error("simulated backend error");
      }
      statusEl.textContent = "success";
      resultEl.textContent = "模拟回答：" + text;
    } catch (error) {
      statusEl.textContent = "error";
      resultEl.textContent = error.message + "。请重试。";
    }
  });
</script>
</html>
```

预期浏览器行为：

- 空输入：状态变成 `empty`。
- 普通输入：状态从 `loading` 变成 `success`。
- 输入包含 `fail`：状态变成 `error`。

## 有意识地测试 UI 状态

不要只点一次成功路径。一个有用的 AI UI 测试要记录每条路径的状态变化和可见提示：

| 测试输入 | 预期状态 | 应留下的证据 |
|---|---|---|
| 空输入框 | `empty` | 截图或说明：界面要求先输入 |
| `summarize this note` | `loading` 后进入 `success` | 前后截图或事件 trace |
| `please fail` | `error` | 展示恢复提示的截图 |

这就是前端版的模型评测。本页里的模型调用是假的，但用户体验契约是真的：每个不确定调用都应该有进度、结果和恢复路径。

## 之后替换成真实调用

后端准备好后，把模拟等待替换成 `fetch`：

```javascript
const response = await fetch("/api/chat", {
  method: "POST",
  headers: {"Content-Type": "application/json"},
  body: JSON.stringify({prompt: text}),
});
const data = await response.json();
resultEl.textContent = data.answer;
```

## 通过标准

一个 AI 页面能清楚处理输入、加载、成功、空输入、错误和重试状态，且不会让用户迷惑，就算通过本选修。

## 留下的证据

学完这一页，至少保留这张证据卡：

```text
前端界面：聊天、仪表盘、编辑器、审阅面板或工作流 UI
状态模型：加载、流式传输、成功、空、错误、重试和复审状态
工件：UI 草图、组件行为、事件追踪或截图
失败检查：隐藏延迟、遗漏错误状态、引用不清晰或复核控制薄弱
期望产出：带状态和证据展示的 AI 前端交互说明
```

<details>
<summary>检查思路与讲解</summary>

一个合格答案应覆盖至少三条路径：空输入、普通输入、模拟失败。证据最好是一个小表格或截图，能看出状态、提示和重试行为。

重点是让用户清楚知道发生了什么，而不是只把一个结果框摆出来。

</details>
