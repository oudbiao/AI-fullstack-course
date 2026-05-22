import fs from "node:fs";
import path from "node:path";

const DOCS_DIR = path.join(process.cwd(), "src", "content", "docs");
const LOCALES = new Map([
  ["zh-cn", "/zh-cn"],
  ["ja", "/ja"],
]);
const INTERNAL_ROOTS = [
  "intro",
  "ch01-tools",
  "ch02-python",
  "ch03-data-analysis",
  "ch04-ai-math",
  "ch05-machine-learning",
  "ch06-deep-learning",
  "ch07-llm-principles",
  "ch08-rag",
  "ch09-agent",
  "ch10-computer-vision",
  "ch11-nlp",
  "ch12-multimodal",
  "appendix",
  "electives",
];
const ASIDE_TYPE_MAP = {
  info: "note",
  note: "note",
  tip: "tip",
  warning: "caution",
  caution: "caution",
  danger: "danger",
};

function walk(dir) {
  const entries = fs.readdirSync(dir, { withFileTypes: true });
  const files = [];
  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      files.push(...walk(fullPath));
    } else if (/\.mdx?$/.test(entry.name)) {
      files.push(fullPath);
    }
  }
  return files;
}

function readFrontmatter(text) {
  if (!text.startsWith("---\n")) return null;
  const end = text.indexOf("\n---", 4);
  if (end === -1) return null;
  return {
    raw: text.slice(4, end),
    body: text.slice(end + 4),
  };
}

function stringifyFrontmatter(raw) {
  const lines = raw.split(/\r?\n/);
  const next = [];
  let sidebarOrder = null;
  let keywords = null;

  for (const line of lines) {
    if (line.startsWith("sidebar_position:")) {
      const value = Number(line.slice("sidebar_position:".length).trim());
      if (Number.isFinite(value)) sidebarOrder = value;
      continue;
    }

    if (line.startsWith("keywords:")) {
      const value = line.slice("keywords:".length).trim();
      if (value) keywords = value;
      continue;
    }

    if (line.trim()) next.push(line);
  }

  if (sidebarOrder !== null) {
    next.push("sidebar:");
    next.push(`  order: ${sidebarOrder}`);
  }

  if (keywords) {
    const content = keywords
      .replace(/^\[/, "")
      .replace(/\]$/, "")
      .split(",")
      .map((item) => item.trim().replace(/^['"]|['"]$/g, ""))
      .filter(Boolean)
      .join(", ");
    if (content) {
      next.push("head:");
      next.push("  - tag: meta");
      next.push("    attrs:");
      next.push("      name: keywords");
      next.push(`      content: "${content.replace(/\\/g, "\\\\").replace(/"/g, '\\"')}"`);
    }
  }

  return next.join("\n");
}

function transformAsides(text) {
  return text
    .replace(/^:{4,}(info|note|tip|warning|caution|danger)\s*([^\n]*)$/gm, (_match, type, title) => {
      const mapped = ASIDE_TYPE_MAP[type] ?? "note";
      const cleanTitle = title.trim();
      return cleanTitle ? `:::${mapped}[${cleanTitle}]` : `:::${mapped}`;
    })
    .replace(/^:{4,}\s*$/gm, ":::");
}

function transformLocalizedLinks(text, localePrefix) {
  text = text.replace(/\]\(\/zh-Hans(?=[/#)"])/g, "](/zh-cn");
  if (!localePrefix) return text;
  const roots = INTERNAL_ROOTS.join("|");
  const linkRe = new RegExp(`\\]\\(/(${roots})(?=[/#)"])`, "g");
  return text.replace(linkRe, `](${localePrefix}/$1`);
}

function transformHtmlCompat(text) {
  return text.replace(/\bclassName=/g, "class=");
}

function localePrefixFor(filePath) {
  const relative = path.relative(DOCS_DIR, filePath).split(path.sep);
  return LOCALES.get(relative[0]) ?? "";
}

let changed = 0;
for (const filePath of walk(DOCS_DIR)) {
  const original = fs.readFileSync(filePath, "utf8");
  const parsed = readFrontmatter(original);
  if (!parsed) continue;

  let body = parsed.body;
  body = transformHtmlCompat(body);
  body = transformAsides(body);
  body = transformLocalizedLinks(body, localePrefixFor(filePath));

  const next = `---\n${stringifyFrontmatter(parsed.raw)}\n---${body}`;
  if (next !== original) {
    fs.writeFileSync(filePath, next);
    changed += 1;
  }
}

console.log(`migrate_starlight_content: updated ${changed} files`);
