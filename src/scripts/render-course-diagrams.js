const codeLineSeparator = "\u007f";
const figureSelector = ".expressive-code figure";
const mermaidLanguageSelector = 'pre[data-language="mermaid"]';
const courseMapLanguageSelector = 'pre[data-language="course-map"]';

let mermaidModule;
let renderCounter = 0;
let isWatchingTheme = false;

const customDiagramRenderers = {
  "course-map": renderCourseMap,
};

const sharedTheme = {
  background: "transparent",
  fontFamily:
    '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "PingFang SC", "Microsoft YaHei", sans-serif',
  fontSize: "16px",
};

const themePalettes = {
  light: {
    ...sharedTheme,
    primaryColor: "#eef6ff",
    primaryTextColor: "#0f172a",
    primaryBorderColor: "#2563eb",
    secondaryColor: "#e6fffb",
    tertiaryColor: "#f7fafc",
    lineColor: "#2563eb",
    textColor: "#1e293b",
    mainBkg: "#eef6ff",
    secondBkg: "#e6fffb",
    nodeBorder: "#2563eb",
    clusterBkg: "#f8fafc",
    clusterBorder: "#cbd5e1",
    edgeLabelBackground: "#ffffff",
    titleColor: "#0f172a",
    noteBkgColor: "#fff7ed",
    noteTextColor: "#0f172a",
    noteBorderColor: "#f59e0b",
  },
  dark: {
    ...sharedTheme,
    primaryColor: "#14223a",
    primaryTextColor: "#f8fafc",
    primaryBorderColor: "#60a5fa",
    secondaryColor: "#123432",
    tertiaryColor: "#172033",
    lineColor: "#7dd3fc",
    textColor: "#e2e8f0",
    mainBkg: "#14223a",
    secondBkg: "#123432",
    nodeBorder: "#60a5fa",
    clusterBkg: "#101827",
    clusterBorder: "#334155",
    edgeLabelBackground: "#08111f",
    titleColor: "#f8fafc",
    noteBkgColor: "#2d230f",
    noteTextColor: "#f8fafc",
    noteBorderColor: "#fbbf24",
  },
};

function currentThemeName() {
  return document.documentElement.dataset.theme === "light" ? "light" : "dark";
}

function mermaidConfig() {
  const themeVariables = themePalettes[currentThemeName()];

  return {
    startOnLoad: false,
    securityLevel: "strict",
    theme: "base",
    themeVariables,
    htmlLabels: false,
    flowchart: {
      curve: "basis",
      diagramPadding: 16,
      htmlLabels: false,
      nodeSpacing: 44,
      rankSpacing: 58,
      useMaxWidth: true,
      wrappingWidth: 220,
    },
    xyChart: {
      width: 640,
      height: 380,
    },
  };
}

async function getMermaid() {
  if (!mermaidModule) {
    mermaidModule = (await import("mermaid")).default;
  }

  mermaidModule.initialize(mermaidConfig());
  return mermaidModule;
}

function sourceFromCodeBlock(figure) {
  const copyButton = figure.querySelector(".copy button[data-code]");
  const source =
    copyButton?.getAttribute("data-code") ??
    figure.querySelector(`${mermaidLanguageSelector}, ${courseMapLanguageSelector}`)?.textContent ??
    "";

  return source.replaceAll(codeLineSeparator, "\n").trim();
}

function languageFromCodeBlock(figure) {
  return (
    figure.querySelector(mermaidLanguageSelector)?.dataset.language ??
    figure.querySelector(courseMapLanguageSelector)?.dataset.language ??
    ""
  );
}

function firstDiagramLine(source) {
  return (
    source
      .split(/\r?\n/)
      .map((line) => line.trim())
      .find((line) => line && !line.startsWith("%%")) ?? ""
  );
}

function diagramType(source, language) {
  if (language === "course-map") return "course-map";
  return firstDiagramLine(source).split(/\s+/)[0] || "";
}

function makeDiagramFigure(source, language) {
  const figure = document.createElement("figure");
  figure.className = "course-diagram-figure";
  figure.dataset.diagramSource = source;
  figure.dataset.diagramLanguage = language;

  const surface = document.createElement("div");
  surface.className = "course-diagram-figure__surface";
  surface.setAttribute("role", "img");

  const firstLine = source.split(/\r?\n/).find((line) => line.trim())?.trim();
  if (firstLine) {
    surface.setAttribute(
      "aria-label",
      language === "course-map" ? `Course summary map: ${firstLine}` : `Mermaid diagram: ${firstLine}`,
    );
  }

  figure.append(surface);

  return { figure, surface };
}

function preserveExpressiveCodeAssets(wrapper) {
  const assets = wrapper.querySelectorAll('link[rel="stylesheet"][href*="/_astro/ec."], script[type="module"][src*="/_astro/ec."]');

  for (const asset of assets) {
    const alreadyPresent =
      asset.tagName === "LINK"
        ? [...document.head.querySelectorAll('link[rel="stylesheet"]')].some((link) => link.href === asset.href)
        : [...document.head.querySelectorAll('script[type="module"]')].some((script) => script.src === asset.src);

    if (alreadyPresent) {
      asset.remove();
    } else {
      document.head.append(asset);
    }
  }
}

function showRenderError(surface, source, error) {
  const figure = surface.closest(".course-diagram-figure");
  figure?.classList.add("course-diagram-figure--error");
  surface.replaceChildren();

  const fallback = document.createElement("pre");
  const code = document.createElement("code");
  code.textContent = source;
  fallback.append(code);
  surface.append(fallback);

  console.error("[AI Roads] Diagram render failed", error);
}

function stripWrappingQuotes(label) {
  const trimmed = label.trim();
  if (trimmed.length >= 2 && trimmed[0] === trimmed.at(-1) && ['"', "'"].includes(trimmed[0])) {
    return trimmed.slice(1, -1).trim();
  }

  return trimmed;
}

function plainCourseMapLabel(rawLabel) {
  const label = rawLabel
    .trim()
    .replace(/^root\s*/i, "")
    .replace(/^[-*]\s*/, "")
    .replace(/^(\(\(|\(\[|\[\[|\(|\[|\{\{)/, "")
    .replace(/(\)\)|\]\]|\)|\]|\}\})$/, "")
    .replaceAll("<br/>", " ")
    .replaceAll("<br>", " ")
    .replace(/\s+/g, " ")
    .trim();

  return stripWrappingQuotes(label);
}

function parseCourseMap(source) {
  const lines = source.split(/\r?\n/);
  const nodes = [];
  const stack = [];

  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("%%") || trimmed === "mindmap" || trimmed === "course-map") continue;
    if (/^::/.test(trimmed)) continue;

    const indent = line.match(/^\s*/)?.[0].replace(/\t/g, "    ").length ?? 0;
    const node = {
      children: [],
      label: plainCourseMapLabel(trimmed),
      level: 0,
    };

    if (!node.label) continue;

    while (stack.length && indent <= stack.at(-1).indent) {
      stack.pop();
    }

    const parent = stack.at(-1)?.node;
    if (parent) {
      node.level = parent.level + 1;
      parent.children.push(node);
    }

    nodes.push(node);
    stack.push({ indent, node });
  }

  return nodes[0] ?? null;
}

function createCourseMapItem(node) {
  const item = document.createElement("li");
  item.className = "course-map__item";

  const label = document.createElement("span");
  label.className = "course-map__node";
  label.textContent = node.label;
  item.append(label);

  if (node.children.length > 0) {
    const children = document.createElement("ul");
    children.className = "course-map__children";
    for (const child of node.children) {
      children.append(createCourseMapItem(child));
    }
    item.append(children);
  }

  return item;
}

function renderCourseMap(surface, source) {
  const root = parseCourseMap(source);
  if (!root) {
    throw new Error("Course map source did not contain a root node.");
  }

  const rootNode = document.createElement("div");
  rootNode.className = "course-map";

  const rootLabel = document.createElement("div");
  rootLabel.className = "course-map__root";
  rootLabel.textContent = root.label;
  rootNode.append(rootLabel);

  const branches = document.createElement("div");
  branches.className = "course-map__branches";

  for (const child of root.children) {
    const branch = document.createElement("section");
    branch.className = "course-map__branch";

    const branchTitle = document.createElement("h3");
    branchTitle.className = "course-map__branch-title";
    branchTitle.textContent = child.label;
    branch.append(branchTitle);

    if (child.children.length > 0) {
      const children = document.createElement("ul");
      children.className = "course-map__children";
      for (const grandchild of child.children) {
        children.append(createCourseMapItem(grandchild));
      }
      branch.append(children);
    }

    branches.append(branch);
  }

  rootNode.append(branches);
  surface.replaceChildren(rootNode);
}

async function renderDiagram(surface, source, language) {
  try {
    const customRenderer = customDiagramRenderers[diagramType(source, language)];
    if (customRenderer) {
      customRenderer(surface, source);
      surface.closest(".course-diagram-figure")?.classList.remove("course-diagram-figure--error");
      return;
    }

    await document.fonts?.ready;
    const mermaid = await getMermaid();
    const id = `ai-roads-mermaid-${Date.now().toString(36)}-${renderCounter++}`;
    const { svg, bindFunctions } = await mermaid.render(id, source);
    surface.innerHTML = svg;
    bindFunctions?.(surface);
    surface.closest(".course-diagram-figure")?.classList.remove("course-diagram-figure--error");
  } catch (error) {
    showRenderError(surface, source, error);
  }
}

async function upgradeCodeBlocks() {
  const targets = [...document.querySelectorAll(figureSelector)]
    .filter((figure) => figure.querySelector(`${mermaidLanguageSelector}, ${courseMapLanguageSelector}`))
    .map((figure) => ({
      language: languageFromCodeBlock(figure),
      wrapper: figure.closest(".expressive-code") ?? figure,
      source: sourceFromCodeBlock(figure),
    }))
    .filter(({ source }) => source.length > 0);

  if (targets.length === 0) return;

  await Promise.all(
    targets.map(async ({ language, wrapper, source }) => {
      const { figure, surface } = makeDiagramFigure(source, language);
      preserveExpressiveCodeAssets(wrapper);
      wrapper.replaceWith(figure);
      await renderDiagram(surface, source, language);
    }),
  );
}

async function rerenderExistingFigures() {
  const figures = [...document.querySelectorAll(".course-diagram-figure")];
  if (figures.length === 0) return;

  await Promise.all(
    figures.map(async (figure) => {
      const source = figure.dataset.diagramSource;
      const language = figure.dataset.diagramLanguage;
      const surface = figure.querySelector(".course-diagram-figure__surface");
      if (!source || !surface) return;
      await renderDiagram(surface, source, language);
    }),
  );
}

function watchThemeChanges() {
  if (isWatchingTheme) return;
  isWatchingTheme = true;

  let timeoutId;
  const observer = new MutationObserver(() => {
    window.clearTimeout(timeoutId);
    timeoutId = window.setTimeout(() => {
      rerenderExistingFigures();
    }, 80);
  });

  observer.observe(document.documentElement, {
    attributes: true,
    attributeFilter: ["data-theme"],
  });
}

function start() {
  upgradeCodeBlocks().then(watchThemeChanges);
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", start, { once: true });
} else {
  start();
}
