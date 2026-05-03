---
title: "1.1 Web Front-End Basics in Fast Track"
sidebar_position: 5
description: "Quickly build the core front-end intuition needed for AI product scenarios: page structure, state updates, form interaction, asynchronous requests, and a minimally usable interface."
keywords: [frontend, HTML, CSS, JavaScript, fetch, UI, AI product]
---

# Web Front-End Basics in Fast Track

![AI front-end interaction stack diagram](/img/course/elective-ai-frontend-stack-en.png)

![AI front-end state machine and experience loop diagram](/img/course/elective-ai-frontend-state-machine-map-en.png)

:::tip Reading the diagram
An AI front end is not just a results display page. It needs to handle states such as idle, loading, streaming, success, error, retry, and cancel. When reading the diagram, think of model calls as an interaction flow with “uncertain latency.”
:::

:::tip Section focus
Many AI features are not impossible to build — they just lack a good interface to sit on top of.
What users actually interact with is usually not the model itself, but:

- input boxes
- message history
- loading states
- error messages

So this lesson is not meant to turn front-end development into a full discipline. Instead, it focuses on one very practical main line:

> **How to build the smallest front end that makes AI features interactive, understandable, and easy to iterate on.**
:::

## Learning Objectives

- Understand the basic role of the front end in AI product implementation
- Master the minimal collaboration pattern of HTML / CSS / JS
- Understand the minimal pattern for asynchronous requests and state updates
- Build interface intuition through a runnable static page example

---

## 1. Why AI products can’t do without the front end?

### 1.1 Because “model capability” must be made visible through interaction

No matter how good the model is, if there is no clear interface to present it,
users will still feel that:

- it is slow to respond
- they don’t know what the system is doing
- they don’t know whether the result is trustworthy

### 1.2 The core value of the front end

It is not to make the page flashy,
but to express these states clearly:

- what the current input is
- whether the request has been sent
- whether the system is loading or has failed
- how the output should be read

### 1.3 An analogy

The model is like the engine, and the front end is like the dashboard and cockpit.
Without the cockpit, even a very powerful engine is hard for users to operate smoothly.

---

## 2. First build a minimal front-end mental model

### 2.1 HTML handles structure

For example:

- input box
- button
- result area

### 2.2 CSS handles appearance

For example:

- spacing
- layout
- state styles

### 2.3 JavaScript handles behavior

For example:

- listening to clicks
- sending requests
- updating page content

As long as you understand how these three layers work together,
you can start making sense of most beginner-friendly AI pages.

---

## 3. Run a minimal AI page first

The example below does one very typical thing:

- enter a question
- click a button
- show “loading”
- then render the result on the page

It uses pure front-end simulation, so it can be opened directly in the browser without relying on a back end.

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
    <h1>Minimal Page for an AI Feature</h1>
    <textarea id="prompt" placeholder="Enter a question, for example: What is the refund policy?"></textarea>
    <button id="runBtn">Submit</button>
    <div id="status" class="status">Waiting for input</div>
    <div id="result" class="result">The result will appear here</div>
  </div>

  <script>
    const promptInput = document.getElementById("prompt");
    const runBtn = document.getElementById("runBtn");
    const statusEl = document.getElementById("status");
    const resultEl = document.getElementById("result");

    function fakeAiResponse(prompt) {
      return new Promise((resolve) => {
        setTimeout(() => {
          resolve("Simulated answer: " + prompt + ". This can later be replaced with a real API response.");
        }, 700);
      });
    }

    runBtn.addEventListener("click", async () => {
      const prompt = promptInput.value.trim();
      if (!prompt) {
        statusEl.textContent = "Please enter some content first";
        return;
      }

      statusEl.textContent = "Request in progress...";
      resultEl.textContent = "Loading...";
      runBtn.disabled = true;

      try {
        const answer = await fakeAiResponse(prompt);
        statusEl.textContent = "Request completed";
        resultEl.textContent = answer;
      } catch (error) {
        statusEl.textContent = "Request failed";
        resultEl.textContent = "Something went wrong. Please try again later.";
      } finally {
        runBtn.disabled = false;
      }
    });
  </script>
</body>
</html>
```

### 3.1 What is the most valuable thing to learn from this example?

It expresses the smallest front-end loop very clearly:

1. get input
2. trigger an action
3. show loading state
4. update the result area

### 3.2 Why is “loading” so important?

Many AI requests do not return instantly.
If the front end gives no feedback at all, users may wonder:

- did it freeze?
- did I fail to click it?

So state feedback itself is part of the experience.

### 3.3 Why use `fakeAiResponse` here first?

Because it helps you understand the front-end flow first.
When you later have a real back-end API, you only need to replace this part with `fetch`.

---

## 4. How do you usually move from local simulation to a real API?

### 4.1 Use `fetch` to send requests

The most common next step is:

```javascript
const response = await fetch("/api/chat", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ prompt }),
});
const data = await response.json();
```

### 4.2 What should the front end pay more attention to at this point?

- whether the request succeeds
- whether the response is empty
- how errors should be displayed
- whether the user can continue operating the page

### 4.3 The front end is not just for “showing results”

It also takes care of:

- input validation
- interaction pacing
- exception prompts

---

## 5. The most common pitfalls in AI front ends

### 5.1 Mistake 1: Building a very complex UI before connecting the back end

A more stable order is usually:

- start with the minimum usable version
- then improve it step by step

### 5.2 Mistake 2: No loading state or error state

This makes the experience poor,
and makes problems harder to locate.

### 5.3 Mistake 3: Treating the front end as a “decorative layer”

For AI products, the front end actually carries very important responsibilities:

- guiding the user’s mental model
- expressing feedback
- building trust

---

## Summary

The most important thing in this lesson is to build a very practical front-end perspective:

> **The first goal of an AI front end is not to look dazzling, but to clearly express the four states of input, processing, result, and error, so that the model’s capabilities can be used smoothly by users.**

Once this understanding is solid, learning React later or building more complex products will feel much more natural.

---

## Exercises

1. Add a “clear input box” button to the example.
2. Change `fakeAiResponse` so it randomly returns success or failure, and practice showing the error state.
3. Think about this: why is “loading feedback” more important in AI products than in many ordinary static pages?
4. How would you extend this minimal page into a chat interface?
