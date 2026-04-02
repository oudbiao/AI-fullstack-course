---
title: "1.1 Web 前端基础速成"
sidebar_position: 5
description: "面向 AI 产品场景快速补齐前端核心直觉：页面结构、状态更新、表单交互、异步请求和最小可用界面。"
keywords: [frontend, HTML, CSS, JavaScript, fetch, UI, AI product]
---

# Web 前端基础速成

:::tip 本节定位
很多 AI 功能其实不是做不出来，而是“没有好界面承接”。  
用户真正接触到的，通常不是模型本身，而是：

- 输入框
- 历史消息
- 加载状态
- 错误提示

所以这节课不打算把前端讲成完整学科，而是优先补一条最实用主线：

> **怎样做出一个能让 AI 功能可交互、可理解、可迭代的最小前端。**
:::

## 学习目标

- 理解前端在 AI 产品落地里的基本作用
- 掌握 HTML / CSS / JS 的最小协作方式
- 理解异步请求和状态更新的最小模式
- 通过一个可运行静态页面例子建立界面组织直觉

---

## 一、AI 产品为什么离不开前端？

### 1.1 因为“模型能力”必须通过交互被看见

再好的模型，如果没有一个清楚的界面承接，  
用户也会觉得：

- 反馈慢
- 不知道系统在干嘛
- 不知道结果是否可信

### 1.2 前端最核心的价值

不是把页面做花哨，  
而是把下面这些状态表达清楚：

- 当前输入是什么
- 请求有没有发出去
- 系统在加载还是出错
- 输出结果该怎么看

### 1.3 一个类比

模型像引擎，前端像仪表盘和驾驶舱。  
没有驾驶舱，再强的引擎用户也很难顺畅使用。

---

## 二、先建立最小前端心智模型

### 2.1 HTML 负责结构

例如：

- 输入框
- 按钮
- 结果区域

### 2.2 CSS 负责外观

例如：

- 间距
- 排版
- 状态样式

### 2.3 JavaScript 负责行为

例如：

- 监听点击
- 发请求
- 更新页面内容

只要这三层关系理顺，  
大多数入门 AI 页面你都能开始看懂。

---

## 三、先跑一个最小 AI 页面

下面这个例子会做一件非常典型的事：

- 输入问题
- 点击按钮
- 显示“加载中”
- 再把结果渲染到页面

它用的是纯前端模拟，不依赖后端也能直接在浏览器里打开。

```html
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>AI Demo</title>
  <style>
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 0;
      background: linear-gradient(180deg, #f8fafc, #eef2ff);
      color: #0f172a;
    }
    .container {
      max-width: 720px;
      margin: 48px auto;
      padding: 24px;
      background: white;
      border-radius: 16px;
      box-shadow: 0 16px 40px rgba(15, 23, 42, 0.08);
    }
    h1 {
      margin-top: 0;
      font-size: 28px;
    }
    textarea {
      width: 100%;
      min-height: 120px;
      padding: 12px;
      border: 1px solid #cbd5e1;
      border-radius: 12px;
      font-size: 16px;
      box-sizing: border-box;
      resize: vertical;
    }
    button {
      margin-top: 12px;
      padding: 10px 16px;
      border: none;
      border-radius: 999px;
      background: #0f172a;
      color: white;
      cursor: pointer;
      font-size: 15px;
    }
    .status {
      margin-top: 16px;
      color: #475569;
      font-size: 14px;
    }
    .result {
      margin-top: 20px;
      padding: 16px;
      border-radius: 12px;
      background: #f8fafc;
      border: 1px solid #e2e8f0;
      line-height: 1.6;
      min-height: 72px;
    }
  </style>
</head>
<body>
  <div class="container">
    <h1>AI 功能最小页面</h1>
    <textarea id="prompt" placeholder="输入一个问题，例如：退款规则是什么？"></textarea>
    <button id="runBtn">提交</button>
    <div id="status" class="status">等待输入</div>
    <div id="result" class="result">结果会显示在这里</div>
  </div>

  <script>
    const promptInput = document.getElementById("prompt");
    const runBtn = document.getElementById("runBtn");
    const statusEl = document.getElementById("status");
    const resultEl = document.getElementById("result");

    function fakeAiResponse(prompt) {
      return new Promise((resolve) => {
        setTimeout(() => {
          resolve("模拟回答：" + prompt + "。这里可以替换成真实接口返回。");
        }, 700);
      });
    }

    runBtn.addEventListener("click", async () => {
      const prompt = promptInput.value.trim();
      if (!prompt) {
        statusEl.textContent = "请先输入内容";
        return;
      }

      statusEl.textContent = "请求处理中...";
      resultEl.textContent = "加载中...";
      runBtn.disabled = true;

      try {
        const answer = await fakeAiResponse(prompt);
        statusEl.textContent = "请求完成";
        resultEl.textContent = answer;
      } catch (error) {
        statusEl.textContent = "请求失败";
        resultEl.textContent = "系统出错，请稍后再试。";
      } finally {
        runBtn.disabled = false;
      }
    });
  </script>
</body>
</html>
```

### 3.1 这个例子最值得学的是什么？

它把前端最小闭环表达得很完整：

1. 获取输入
2. 触发动作
3. 展示加载状态
4. 更新结果区域

### 3.2 为什么“加载中”这么重要？

很多 AI 请求不是瞬时返回。  
如果前端没有任何反馈，用户会怀疑：

- 是不是卡住了
- 是不是没点到

所以状态反馈本身就是体验的一部分。

### 3.3 为什么这里先用 `fakeAiResponse`？

因为它能让你先把前端流程理解清楚。  
等后面有真实后端接口时，只要把这部分替换成 `fetch` 即可。

---

## 四、从本地模拟到真实接口通常怎么走？

### 4.1 用 `fetch` 发请求

最常见的下一步会是：

```javascript
const response = await fetch("/api/chat", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ prompt }),
});
const data = await response.json();
```

### 4.2 这时前端更需要关注什么？

- 请求是否成功
- 返回是否为空
- 错误如何展示
- 用户是否能继续操作

### 4.3 前端不只是“展示结果”

它还在承担：

- 用户输入校验
- 交互节奏设计
- 异常提示

---

## 五、AI 前端最容易踩的坑

### 5.1 误区一：先做很复杂的 UI，再接后端

更稳的顺序通常是：

- 先最小可用
- 再逐步增强

### 5.2 误区二：没有加载态和错误态

这会让体验很差，  
而且问题难以定位。

### 5.3 误区三：把前端当成“装饰层”

对 AI 产品来说，前端其实承担了非常重要的：

- 心智引导
- 反馈表达
- 信任建立

---

## 小结

这节最重要的是建立一个非常实用的前端观：

> **AI 前端的第一目标不是炫，而是把输入、处理中、结果和错误这四种状态表达清楚，让模型能力能被用户顺畅使用。**

只要这层理解稳了，后面你再学 React 或做更复杂产品，也会更自然。

---

## 练习

1. 给示例加一个“清空输入框”按钮。
2. 把 `fakeAiResponse` 改成随机返回成功或失败，练习错误态展示。
3. 想一想：为什么 AI 产品里“加载反馈”比很多普通静态页面更重要？
4. 你会如何把这个最小页面扩成聊天界面？
