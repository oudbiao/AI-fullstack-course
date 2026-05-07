---
title: "E.E Web Front-End Basics in Fast Track"
sidebar_position: 5
description: "A concise hands-on guide to the minimum front-end loop needed for AI products: input, loading, result, error, and retry states."
keywords: [frontend, HTML, CSS, JavaScript, fetch, UI, AI product]
---

# E.E Web Front-End Basics in Fast Track

An AI feature needs a surface users can operate. The smallest useful front end shows input, loading, success, error, and retry states clearly.

## See the Interaction Stack First

![AI front-end interaction stack diagram](/img/course/elective-ai-frontend-stack-en.png)

![AI front-end state machine and experience loop diagram](/img/course/elective-ai-frontend-state-machine-map-en.png)

Think of every model call as an uncertain request: it may be slow, fail, stream partial output, or need retry.

## Run the Smallest Static AI Page

Save this as `ai-demo.html` and open it in a browser:

```html
<!doctype html>
<html lang="en">
<meta charset="utf-8" />
<title>AI Demo</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 720px; margin: 40px auto; }
  textarea, button, pre { width: 100%; box-sizing: border-box; margin-top: 12px; }
  textarea { min-height: 100px; padding: 12px; }
  button { padding: 10px; }
  pre { min-height: 80px; padding: 12px; background: #f3f4f6; white-space: pre-wrap; }
</style>
<h1>AI Feature Demo</h1>
<textarea id="prompt" placeholder="Ask a question"></textarea>
<button id="run">Run</button>
<p id="status">idle</p>
<pre id="result">result appears here</pre>
<script>
  const promptEl = document.querySelector("#prompt");
  const statusEl = document.querySelector("#status");
  const resultEl = document.querySelector("#result");

  document.querySelector("#run").addEventListener("click", async () => {
    statusEl.textContent = "loading";
    resultEl.textContent = "Please wait...";
    await new Promise((resolve) => setTimeout(resolve, 500));
    statusEl.textContent = "success";
    resultEl.textContent = "Simulated answer: " + promptEl.value.trim();
  });
</script>
</html>
```

Expected browser behavior: after you type a question and click `Run`, the status changes from `idle` to `loading` to `success`, and the result area updates.

## Replace the Fake Call Later

When your backend exists, replace the simulated wait with `fetch`:

```javascript
const response = await fetch("/api/chat", {
  method: "POST",
  headers: {"Content-Type": "application/json"},
  body: JSON.stringify({prompt: promptEl.value}),
});
const data = await response.json();
resultEl.textContent = data.answer;
```

## Pass Check

You pass this elective when your page clearly handles input, loading, success, empty input, and error states for one AI feature.
