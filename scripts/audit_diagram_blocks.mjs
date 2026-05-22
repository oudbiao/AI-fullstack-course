import fs from "node:fs/promises";
import path from "node:path";
import { JSDOM } from "jsdom";

const docsRoot = path.resolve("src/content/docs");
const courseMapLanguage = "course-map";
const supportedDiagramTypes = new Set([
  "architecture-beta",
  "block-beta",
  "classDiagram",
  "classDiagram-v2",
  "erDiagram",
  "flowchart",
  "graph",
  "gitGraph",
  "gantt",
  "journey",
  "packet-beta",
  "pie",
  "quadrantChart",
  "requirementDiagram",
  "sankey-beta",
  "sequenceDiagram",
  "stateDiagram",
  "stateDiagram-v2",
  "timeline",
  "xychart-beta",
]);
const deprecatedMermaidTypes = new Set(["mindmap"]);

const directDiagramFenceLanguages = new Set([...supportedDiagramTypes, ...deprecatedMermaidTypes]);

function createDom() {
  const dom = new JSDOM("<!doctype html><html><body></body></html>", {
    pretendToBeVisual: true,
  });

  globalThis.window = dom.window;
  globalThis.document = dom.window.document;
  globalThis.Element = dom.window.Element;
  globalThis.HTMLElement = dom.window.HTMLElement;
  globalThis.SVGElement = dom.window.SVGElement;
  globalThis.MutationObserver = dom.window.MutationObserver;
  Object.defineProperty(globalThis, "navigator", {
    value: dom.window.navigator,
    configurable: true,
  });
}

async function findMarkdownFiles(directory) {
  const files = [];

  async function walk(currentDirectory) {
    const entries = await fs.readdir(currentDirectory, { withFileTypes: true });
    await Promise.all(
      entries.map(async (entry) => {
        const absolutePath = path.join(currentDirectory, entry.name);
        if (entry.isDirectory()) {
          await walk(absolutePath);
          return;
        }
        if (/\.mdx?$/.test(entry.name)) {
          files.push(absolutePath);
        }
      }),
    );
  }

  await walk(directory);
  return files.sort();
}

function lineNumberForIndex(source, index) {
  return source.slice(0, index).split("\n").length;
}

function firstContentLine(source) {
  return (
    source
      .split(/\r?\n/)
      .map((line) => line.trim())
      .find((line) => line && !line.startsWith("%%")) ?? ""
  );
}

function diagramTypeFor(source) {
  const firstLine = firstContentLine(source);
  return firstLine.split(/\s+/)[0] || "(empty)";
}

function diagramBlocksFromFile(filePath, source) {
  const blocks = [];
  const fenceExpression = /(^|\n)(`{3,}|~{3,})([^\n]*)\n([\s\S]*?)\n\2(?=\n|$)/g;

  for (const match of source.matchAll(fenceExpression)) {
    const language = match[3].trim().split(/\s+/)[0];
    const line = lineNumberForIndex(source, match.index);

    if (directDiagramFenceLanguages.has(language)) {
      blocks.push({
        filePath,
        line,
        language,
        source: match[4].trim(),
        directDiagramFence: true,
      });
      continue;
    }

    if (language !== "mermaid" && language !== courseMapLanguage) continue;

    blocks.push({
      filePath,
      line,
      language,
      source: match[4].trim(),
      directDiagramFence: false,
    });
  }

  return blocks;
}

function printGroupedIssues(title, issues) {
  if (issues.length === 0) return;

  console.error(`\n${title}`);
  for (const issue of issues) {
    const relativePath = path.relative(process.cwd(), issue.filePath);
    console.error(`- ${relativePath}:${issue.line} ${issue.message}`);
  }
}

createDom();
const { default: mermaid } = await import("mermaid");
mermaid.initialize({
  startOnLoad: false,
  securityLevel: "strict",
  theme: "base",
});

const markdownFiles = await findMarkdownFiles(docsRoot);
const blocks = [];

for (const filePath of markdownFiles) {
  const source = await fs.readFile(filePath, "utf8");
  blocks.push(...diagramBlocksFromFile(filePath, source));
}

const directFenceIssues = [];
const courseMapIssues = [];
const deprecatedMermaidIssues = [];
const unsupportedTypeIssues = [];
const parseIssues = [];
const typeCounts = new Map();

for (const block of blocks) {
  if (block.directDiagramFence) {
    directFenceIssues.push({
      ...block,
      message: `uses \`\`\`${block.language}; use \`\`\`mermaid and put ${block.language} on the first diagram line.`,
    });
    continue;
  }

  if (block.language === courseMapLanguage) {
    const firstLine = firstContentLine(block.source);
    typeCounts.set(courseMapLanguage, (typeCounts.get(courseMapLanguage) ?? 0) + 1);
    if (!/^root\b/i.test(firstLine)) {
      courseMapIssues.push({
        ...block,
        message: `must start with a root(...) line; found "${firstLine || "(empty)"}".`,
      });
    }
    continue;
  }

  const diagramType = diagramTypeFor(block.source);
  typeCounts.set(diagramType, (typeCounts.get(diagramType) ?? 0) + 1);

  if (!supportedDiagramTypes.has(diagramType)) {
    if (deprecatedMermaidTypes.has(diagramType)) {
      deprecatedMermaidIssues.push({
        ...block,
        message: `uses Mermaid "${diagramType}"; use \`\`\`course-map for lesson summary cards instead.`,
      });
      continue;
    }

    unsupportedTypeIssues.push({
      ...block,
      message: `starts with unsupported Mermaid diagram type "${diagramType}".`,
    });
    continue;
  }

  try {
    await mermaid.parse(block.source);
  } catch (error) {
    parseIssues.push({
      ...block,
      message: `does not parse as Mermaid: ${error.message.split("\n")[0]}`,
    });
  }
}

printGroupedIssues("Direct diagram code fences found", directFenceIssues);
printGroupedIssues("Invalid course-map blocks found", courseMapIssues);
printGroupedIssues("Deprecated Mermaid diagram types found", deprecatedMermaidIssues);
printGroupedIssues("Unsupported Mermaid diagram types found", unsupportedTypeIssues);
printGroupedIssues("Mermaid parse failures found", parseIssues);

const summary = [...typeCounts.entries()]
  .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
  .map(([type, count]) => `${type}=${count}`)
  .join(", ");

if (
  directFenceIssues.length ||
  courseMapIssues.length ||
  deprecatedMermaidIssues.length ||
  unsupportedTypeIssues.length ||
  parseIssues.length
) {
  process.exitCode = 1;
}

console.log(
  `Checked ${blocks.length} diagram code fences in ${markdownFiles.length} Markdown files. Types: ${summary || "none"}.`,
);
