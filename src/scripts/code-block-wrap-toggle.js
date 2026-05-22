const storageKey = "ai-roads-code-wrap";
const enhancedAttribute = "data-code-wrap-enhanced";
const skippedLanguages = new Set(["mermaid", "course-map"]);

function canUseStorage() {
  try {
    window.localStorage.getItem(storageKey);
    return true;
  } catch {
    return false;
  }
}

function readPreference() {
  if (!canUseStorage()) return false;
  return window.localStorage.getItem(storageKey) === "true";
}

function savePreference(isWrapped) {
  if (!canUseStorage()) return;
  window.localStorage.setItem(storageKey, String(isWrapped));
}

function applyWrapState(isWrapped) {
  document.documentElement.dataset.codeWrap = isWrapped ? "true" : "false";

  document.querySelectorAll(".expressive-code figure pre[data-language]").forEach((pre) => {
    if (!skippedLanguages.has(pre.dataset.language)) {
      pre.classList.toggle("wrap", isWrapped);
    }
  });

  document.querySelectorAll(".course-code-wrap-toggle").forEach((button) => {
    button.setAttribute("aria-pressed", String(isWrapped));
    button.setAttribute(
      "aria-label",
      isWrapped ? "Keep long code lines on one line" : "Wrap long code lines",
    );
    button.title = isWrapped ? "切换为横向滚动" : "切换为自动换行";
  });
}

function iconMarkup() {
  return `
    <svg aria-hidden="true" viewBox="0 0 24 24" focusable="false">
      <path d="M4 6h16" />
      <path d="M4 11h10" />
      <path d="M4 16h9" />
      <path d="M15 11h1.5a3 3 0 0 1 0 6H15" />
      <path d="M17 14l-3 3 3 3" />
    </svg>
  `;
}

function codeFigureCanWrap(figure) {
  const pre = figure.querySelector("pre[data-language]");
  if (!pre || !pre.querySelector("code")) return false;
  return !skippedLanguages.has(pre.dataset.language);
}

function createToggleButton() {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "course-code-wrap-toggle";
  button.innerHTML = iconMarkup();
  button.addEventListener("click", (event) => {
    const nextState = document.documentElement.dataset.codeWrap !== "true";
    savePreference(nextState);
    applyWrapState(nextState);

    if (event.detail > 0) {
      button.blur();
    }
  });

  return button;
}

function enhanceCodeBlocks() {
  document.querySelectorAll(".expressive-code figure").forEach((figure) => {
    if (figure.hasAttribute(enhancedAttribute) || !codeFigureCanWrap(figure)) return;

    const toolArea = figure.querySelector(".copy");
    const copyButton = toolArea?.querySelector("button[data-code]");
    if (!toolArea || !copyButton) return;

    const toggle = createToggleButton();
    toolArea.insertBefore(toggle, copyButton);
    figure.setAttribute(enhancedAttribute, "true");
  });

  applyWrapState(readPreference());
}

function watchCodeBlocks() {
  const observer = new MutationObserver((mutations) => {
    if (
      mutations.some((mutation) =>
        Array.from(mutation.addedNodes).some(
          (node) => node instanceof Element && (node.matches(".expressive-code") || node.querySelector(".expressive-code")),
        ),
      )
    ) {
      enhanceCodeBlocks();
    }
  });

  observer.observe(document.body, { childList: true, subtree: true });
}

function init() {
  enhanceCodeBlocks();
  watchCodeBlocks();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init, { once: true });
} else {
  init();
}

document.addEventListener("astro:page-load", enhanceCodeBlocks);
