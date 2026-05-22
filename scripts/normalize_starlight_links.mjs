#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const projectRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const docsRoot = path.join(projectRoot, "src", "content", "docs");
const localeDirs = new Set(["zh-cn", "ja"]);

function walkMarkdownFiles(dir, files = []) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      walkMarkdownFiles(fullPath, files);
    } else if (entry.name.endsWith(".md") || entry.name.endsWith(".mdx")) {
      files.push(fullPath);
    }
  }
  return files;
}

function splitHref(href) {
  const splitAt = href.search(/[?#]/);
  if (splitAt === -1) {
    return { pathPart: href, suffix: "" };
  }
  return { pathPart: href.slice(0, splitAt), suffix: href.slice(splitAt) };
}

function routeForMarkdownFile(filePath) {
  const rel = path.relative(docsRoot, filePath).split(path.sep).join("/");
  const parts = rel.replace(/\.mdx?$/, "").split("/");
  const localePrefix = localeDirs.has(parts[0]) ? `/${parts.shift()}` : "";
  let route = parts.join("/");

  if (route === "index") {
    route = "";
  } else if (route.endsWith("/index")) {
    route = route.slice(0, -"index".length);
  } else {
    route = `${route}/`;
  }

  return normalizeRoutePath(`${localePrefix}/${route}`);
}

function docsBaseRoute(filePath) {
  const rel = path.relative(docsRoot, filePath).split(path.sep).join("/");
  const parts = rel.split("/");
  const localePrefix = localeDirs.has(parts[0]) ? `/${parts.shift()}` : "";
  parts.pop();
  const sourceDir = parts.join("/");
  return `${localePrefix}/${sourceDir ? `${sourceDir}/` : ""}`;
}

function normalizeRoutePath(routePath) {
  let normalized = routePath.replace(/\.mdx?$/, "");

  if (normalized === "/index" || normalized === "index") {
    normalized = "/";
  } else if (normalized.endsWith("/index")) {
    normalized = normalized.slice(0, -"index".length);
  } else if (!path.posix.extname(normalized) && !normalized.endsWith("/")) {
    normalized += "/";
  }

  if (!normalized.startsWith("/")) {
    normalized = `/${normalized}`;
  }

  return normalized;
}

function normalizeMarkdownHref(href, baseRoute, knownRoutes) {
  if (/^(?:[a-z][a-z0-9+.-]*:|\/\/|#|\?)/i.test(href)) {
    return href;
  }

  const { pathPart, suffix } = splitHref(href);
  if (!pathPart) {
    return href;
  }

  const resolvedPath = pathPart.startsWith("/")
    ? path.posix.normalize(pathPart)
    : path.posix.normalize(path.posix.join(baseRoute, pathPart));
  const routePath = normalizeRoutePath(resolvedPath);

  if (!knownRoutes.has(routePath)) {
    return href;
  }

  return routePath + suffix;
}

function normalizeText(text, baseRoute, knownRoutes) {
  let inFence = false;
  const lines = text.split("\n");

  return lines
    .map((line) => {
      if (/^\s*(```|~~~)/.test(line)) {
        inFence = !inFence;
        return line;
      }
      if (inFence) {
        return line;
      }

      return line.replace(/(!?)\]\(([^)\s]+(?:[?#][^)]*)?)\)/g, (match, imagePrefix, href) => {
        if (imagePrefix) {
          return match;
        }
        return `](${normalizeMarkdownHref(href, baseRoute, knownRoutes)})`;
      });
    })
    .join("\n");
}

let changedFiles = 0;
let changedLinks = 0;
const markdownFiles = walkMarkdownFiles(docsRoot);
const knownRoutes = new Set(markdownFiles.map(routeForMarkdownFile));

for (const file of markdownFiles) {
  const before = fs.readFileSync(file, "utf8");
  const baseRoute = docsBaseRoute(file);
  const after = normalizeText(before, baseRoute, knownRoutes);
  if (after !== before) {
    changedFiles += 1;
    const beforeLinks = before.match(/(?<!!)\]\([^)\s]+(?:[?#][^)]*)?\)/g) ?? [];
    const afterLinks = after.match(/(?<!!)\]\([^)\s]+(?:[?#][^)]*)?\)/g) ?? [];
    changedLinks += beforeLinks.filter((link, index) => link !== afterLinks[index]).length;
    fs.writeFileSync(file, after);
  }
}

console.log(`normalize_starlight_links: updated ${changedLinks} link(s) in ${changedFiles} file(s)`);
