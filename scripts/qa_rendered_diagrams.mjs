import http from "node:http";
import { spawn } from "node:child_process";

const urls = process.argv.slice(2);
const chromePath = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";

if (urls.length === 0) {
  console.error("Usage: node scripts/qa_rendered_diagrams.mjs <url> [url...]");
  process.exit(1);
}

function requestJson(url, method = "GET") {
  return new Promise((resolve, reject) => {
    const request = http
      .request(url, { method }, (response) => {
        let body = "";
        response.setEncoding("utf8");
        response.on("data", (chunk) => {
          body += chunk;
        });
        response.on("end", () => {
          try {
            resolve(JSON.parse(body));
          } catch (error) {
            reject(new Error(`Could not parse JSON from ${url}: ${error.message}`));
          }
        });
      })
      .on("error", reject);

    request.end();
  });
}

function websocketRequest(socket, method, params) {
  const id = ++websocketRequest.id;
  socket.send(JSON.stringify({ id, method, params }));

  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      websocketRequest.pending.delete(id);
      reject(new Error(`Chrome protocol request timed out: ${method}`));
    }, 10000);

    websocketRequest.pending.set(id, { resolve, reject, timeout });
  });
}
websocketRequest.id = 0;
websocketRequest.pending = new Map();

function connectWebsocket(webSocketDebuggerUrl) {
  return new Promise((resolve, reject) => {
    const socket = new WebSocket(webSocketDebuggerUrl);

    socket.addEventListener("open", () => resolve(socket), { once: true });
    socket.addEventListener(
      "error",
      () => reject(new Error(`Could not connect to ${webSocketDebuggerUrl}`)),
      { once: true },
    );
    socket.addEventListener("message", (event) => {
      const message = JSON.parse(event.data);
      if (!message.id) return;

      const pending = websocketRequest.pending.get(message.id);
      if (!pending) return;

      clearTimeout(pending.timeout);
      websocketRequest.pending.delete(message.id);
      if (message.error) {
        pending.reject(new Error(message.error.message));
      } else {
        pending.resolve(message.result);
      }
    });
  });
}

async function waitForDevTools(port) {
  const endpoint = `http://127.0.0.1:${port}/json/version`;
  const startedAt = Date.now();

  while (Date.now() - startedAt < 8000) {
    try {
      await requestJson(endpoint);
      return;
    } catch {
      await new Promise((resolve) => setTimeout(resolve, 100));
    }
  }

  throw new Error("Chrome DevTools endpoint did not become ready");
}

async function evaluateRenderedPage(url, index) {
  const port = 9222 + index;
  const chrome = spawn(
    chromePath,
    [
      "--headless=new",
      "--disable-gpu",
      "--no-first-run",
      "--no-default-browser-check",
      `--remote-debugging-port=${port}`,
      `--user-data-dir=/tmp/ai-roads-render-qa-${Date.now()}-${index}`,
      "about:blank",
    ],
    { stdio: ["ignore", "ignore", "pipe"] },
  );

  let stderr = "";
  chrome.stderr.on("data", (chunk) => {
    stderr += chunk;
  });

  try {
    await waitForDevTools(port);
    const newTab = await requestJson(`http://127.0.0.1:${port}/json/new?${encodeURIComponent(url)}`, "PUT");
    const socket = await connectWebsocket(newTab.webSocketDebuggerUrl);

    try {
      await websocketRequest(socket, "Page.enable");
      await websocketRequest(socket, "Runtime.enable");
      await new Promise((resolve) => setTimeout(resolve, 500));

      const expression = `
        (async () => {
          const startedAt = Date.now();
          while (Date.now() - startedAt < 9000) {
            const figures = document.querySelectorAll(".course-diagram-figure").length;
            const remainingCode = document.querySelectorAll('pre[data-language="mermaid"], pre[data-language="course-map"]').length;
            const completed = document.querySelectorAll(
              ".course-diagram-figure svg, .course-diagram-figure .course-map, .course-diagram-figure--error",
            ).length;
            if (figures > 0 && completed === figures && remainingCode === 0) break;
            await new Promise((resolve) => setTimeout(resolve, 100));
          }

          const figures = [...document.querySelectorAll(".course-diagram-figure")];
          const courseMaps = [...document.querySelectorAll(".course-diagram-figure .course-map")];
          const result = {
            url: location.href,
            figures: figures.length,
            svgCount: document.querySelectorAll(".course-diagram-figure svg").length,
            courseMapCount: courseMaps.length,
            remainingDiagramCodeBlocks: document.querySelectorAll('pre[data-language="mermaid"], pre[data-language="course-map"]').length,
            errors: document.querySelectorAll(".course-diagram-figure--error").length,
            clippedNodes: [],
            overflowingCourseMapItems: [],
          };

          for (const element of document.querySelectorAll(".course-map__root, .course-map__branch-title, .course-map__node")) {
            if (element.scrollWidth > element.clientWidth + 1 || element.scrollHeight > element.clientHeight + 1) {
              result.overflowingCourseMapItems.push({
                text: element.textContent.trim().replace(/\\s+/g, " ").slice(0, 120),
                className: element.className,
                scrollWidth: element.scrollWidth,
                clientWidth: element.clientWidth,
                scrollHeight: element.scrollHeight,
                clientHeight: element.clientHeight,
              });
            }
          }

          for (const figure of figures) {
            const svg = figure.querySelector("svg");
            if (!svg) continue;

            for (const node of svg.querySelectorAll("g.node")) {
              const shape = node.querySelector("rect, polygon, circle, ellipse");
              const label = node.querySelector(".label, .nodeLabel, text");
              if (!shape || !label) continue;

              const shapeBox = shape.getBoundingClientRect();
              const labelBox = label.getBoundingClientRect();
              if (shapeBox.width === 0 || labelBox.width === 0) continue;

              const isClipped =
                labelBox.top < shapeBox.top - 1 ||
                labelBox.left < shapeBox.left - 1 ||
                labelBox.bottom > shapeBox.bottom + 1 ||
                labelBox.right > shapeBox.right + 1;

              if (isClipped) {
                result.clippedNodes.push({
                  text: label.textContent.trim().replace(/\\s+/g, " ").slice(0, 120),
                  label: {
                    width: Math.round(labelBox.width),
                    height: Math.round(labelBox.height),
                  },
                  shape: {
                    width: Math.round(shapeBox.width),
                    height: Math.round(shapeBox.height),
                  },
                });
              }
            }
          }

          return result;
        })()
      `;
      const evaluation = await websocketRequest(socket, "Runtime.evaluate", {
        awaitPromise: true,
        expression,
        returnByValue: true,
      });

      return evaluation.result.value;
    } finally {
      socket.close();
    }
  } finally {
    chrome.kill("SIGTERM");
    if (stderr.includes("Trace/BPT trap")) {
      console.error(stderr);
    }
  }
}

const results = [];
for (const [index, url] of urls.entries()) {
  results.push(await evaluateRenderedPage(url, index));
}

let failed = false;
for (const result of results) {
  if (
    result.figures === 0 ||
    result.svgCount + result.courseMapCount !== result.figures ||
    result.remainingDiagramCodeBlocks !== 0 ||
    result.errors !== 0 ||
    result.clippedNodes.length > 0 ||
    result.overflowingCourseMapItems.length > 0
  ) {
    failed = true;
  }
}

console.log(JSON.stringify(results, null, 2));
if (failed) {
  process.exitCode = 1;
}
