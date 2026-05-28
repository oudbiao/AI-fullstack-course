#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";

const root = path.resolve(process.argv[2] || "dist");
const siteTitle = "AI Roads";
const localeConfig = {
  en: {
    titleLabel: "",
    minDescriptionLength: 90,
    maxDescriptionLength: 160,
    suffix: (title) =>
      `The lesson connects ${title} to AI engineering with runnable examples, practice checks, and portfolio evidence.`,
    fallback: "Use it for step-by-step practice, review notes, and a checkable learning record.",
  },
  "zh-cn": {
    titleLabel: "简体中文",
    minDescriptionLength: 80,
    maxDescriptionLength: 160,
    suffix: (title) =>
      `本页围绕「${title}」展开，结合可运行示例、练习检查、常见错误提醒和作品集证据，帮助你把知识用于真实 AI 项目。`,
    fallback: "适合按步骤跟做、复盘并保存可检查的学习记录。",
  },
  ja: {
    titleLabel: "日本語",
    minDescriptionLength: 80,
    maxDescriptionLength: 160,
    suffix: (title) =>
      `このページは「${title}」を軸に、実行例、練習チェック、よくある失敗、ポートフォリオ証拠で AI 実践につなげます。`,
    fallback: "手順に沿って復習し、確認できる学習記録を残せます。",
  },
};

function walkHtmlFiles(dir, files = []) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      walkHtmlFiles(fullPath, files);
    } else if (entry.name.endsWith(".html")) {
      files.push(fullPath);
    }
  }
  return files;
}

function toPosix(relativePath) {
  return relativePath.split(path.sep).join("/");
}

function routeFromHtmlFile(filePath) {
  const rel = toPosix(path.relative(root, filePath));
  if (rel === "index.html") return "/";
  if (rel.endsWith("/index.html")) return `/${rel.slice(0, -"index.html".length)}`;
  return `/${rel}`;
}

function localeForRoute(route) {
  if (route === "/zh-cn/" || route.startsWith("/zh-cn/")) return "zh-cn";
  if (route === "/ja/" || route.startsWith("/ja/")) return "ja";
  return "en";
}

function isVerificationRoute(route) {
  return (
    /^\/google[a-z0-9-]+\.html$/i.test(route) ||
    /^\/baidu_verify_[a-z0-9-]+\.html$/i.test(route)
  );
}

function decodeHtml(value) {
  return value
    .replace(/&amp;/g, "&")
    .replace(/&quot;/g, "\"")
    .replace(/&#39;/g, "'")
    .replace(/&#x27;/gi, "'")
    .replace(/&#x2F;/gi, "/")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">");
}

function escapeAttr(value) {
  return value
    .replace(/&/g, "&amp;")
    .replace(/"/g, "&quot;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function escapeText(value) {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function normalizeWhitespace(value) {
  return value.replace(/\s+/g, " ").trim();
}

function textLength(value) {
  return [...value].length;
}

function titleWithoutSite(rawTitle) {
  return normalizeWhitespace(decodeHtml(rawTitle))
    .replace(/\s+\|\s+简体中文\s+\|\s+AI Roads$/i, "")
    .replace(/\s+\|\s+日本語\s+\|\s+AI Roads$/i, "")
    .replace(/\s+\|\s+AI Roads$/i, "");
}

function pageTitle(rawTitle, locale) {
  const baseTitle = titleWithoutSite(rawTitle) || siteTitle;
  const label = localeConfig[locale].titleLabel;
  if (!label) return baseTitle === siteTitle ? siteTitle : `${baseTitle} | ${siteTitle}`;
  return baseTitle === siteTitle ? `${siteTitle} | ${label}` : `${baseTitle} | ${label} | ${siteTitle}`;
}

function openGraphTitle(rawTitle, locale) {
  const baseTitle = titleWithoutSite(rawTitle) || siteTitle;
  const label = localeConfig[locale].titleLabel;
  if (!label) return baseTitle;
  return baseTitle === siteTitle ? `${siteTitle} · ${label}` : `${baseTitle} · ${label}`;
}

function trimDescription(value, maxLength) {
  const decoded = decodeHtml(value);
  if (textLength(decoded) <= maxLength) return value;
  const chars = [...decoded];
  let trimmed = chars.slice(0, maxLength - 1).join("").trimEnd();
  trimmed = trimmed.replace(/[,:;，、；：。.\s]+$/u, "");
  return `${trimmed}…`;
}

function normalizeDescription(rawDescription, rawTitle, locale) {
  const config = localeConfig[locale];
  const baseTitle = titleWithoutSite(rawTitle) || siteTitle;
  let description = normalizeWhitespace(decodeHtml(rawDescription));
  if (!description) {
    description = baseTitle;
  }

  if (textLength(description) < config.minDescriptionLength) {
    description = normalizeWhitespace(`${description} ${config.suffix(baseTitle)}`);
  }
  if (textLength(description) < config.minDescriptionLength) {
    description = normalizeWhitespace(`${description} ${config.fallback}`);
  }
  if (textLength(description) < config.minDescriptionLength) {
    description = normalizeWhitespace(`${description} ${baseTitle}`);
  }

  return trimDescription(description, config.maxDescriptionLength);
}

function getTitle(html) {
  return html.match(/<title>([\s\S]*?)<\/title>/i)?.[1] ?? "";
}

function getMetaContent(html, attrName, attrValue) {
  const pattern = new RegExp(
    `<meta\\b(?=[^>]*\\b${attrName}=(["'])${attrValue}\\1)[^>]*\\bcontent=(["'])(.*?)\\2[^>]*>`,
    "is",
  );
  return html.match(pattern)?.[3] ?? "";
}

function replaceTitle(html, title) {
  return html.replace(/<title>[\s\S]*?<\/title>/i, `<title>${escapeText(title)}</title>`);
}

function replaceMetaContent(html, attrName, attrValue, content) {
  const pattern = new RegExp(
    `(<meta\\b(?=[^>]*\\b${attrName}=(["'])${attrValue}\\2)[^>]*\\bcontent=(["']))(.*?)(\\3[^>]*>)`,
    "is",
  );
  return html.replace(pattern, `$1${escapeAttr(content)}$5`);
}

if (!fs.existsSync(root)) {
  console.log(`normalize_seo_metadata: skipped, ${root} does not exist`);
  process.exit(0);
}

const summary = {
  checked: 0,
  changed: 0,
  titleUpdates: 0,
  descriptionUpdates: 0,
};

for (const file of walkHtmlFiles(root)) {
  const route = routeFromHtmlFile(file);
  if (route === "/404.html" || isVerificationRoute(route)) continue;

  const originalHtml = fs.readFileSync(file, "utf8");
  let html = originalHtml;
  const locale = localeForRoute(route);
  const rawTitle = getTitle(html);
  const rawDescription = getMetaContent(html, "name", "description");
  if (!rawTitle || !rawDescription) continue;

  const normalizedTitle = pageTitle(rawTitle, locale);
  const normalizedOgTitle = openGraphTitle(rawTitle, locale);
  const normalizedDescription = normalizeDescription(rawDescription, rawTitle, locale);

  summary.checked += 1;

  if (decodeHtml(rawTitle).trim() !== normalizedTitle) {
    html = replaceTitle(html, normalizedTitle);
    html = replaceMetaContent(html, "property", "og:title", normalizedOgTitle);
    summary.titleUpdates += 1;
  }

  if (decodeHtml(rawDescription).trim() !== normalizedDescription) {
    html = replaceMetaContent(html, "name", "description", normalizedDescription);
    html = replaceMetaContent(html, "property", "og:description", normalizedDescription);
    summary.descriptionUpdates += 1;
  }

  if (html !== originalHtml) {
    fs.writeFileSync(file, html);
    summary.changed += 1;
  }
}

console.log(`normalize_seo_metadata: ${JSON.stringify(summary)}`);
