#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const projectRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const distRoot = path.resolve(projectRoot, process.argv[2] ?? "dist");
const siteUrl = "https://airoads.org";
const oldDomainPattern = /learning\.airoads\.org/;
const docusaurusPattern = /docusaurus|__docusaurus/i;

function walkFiles(dir, predicate, files = []) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      walkFiles(fullPath, predicate, files);
    } else if (!predicate || predicate(fullPath)) {
      files.push(fullPath);
    }
  }
  return files;
}

function toPosix(relativePath) {
  return relativePath.split(path.sep).join("/");
}

function routeFromHtmlFile(filePath) {
  const rel = toPosix(path.relative(distRoot, filePath));
  if (rel === "index.html") return "/";
  if (rel.endsWith("/index.html")) return `/${rel.slice(0, -"index.html".length)}`;
  return `/${rel}`;
}

function isVerificationHtml(route) {
  return (
    /^\/google[a-z0-9-]+\.html$/i.test(route) ||
    /^\/baidu_verify_[a-z0-9-]+\.html$/i.test(route)
  );
}

function isLegacyZhHansRoute(route) {
  return route === "/zh-Hans/" || route.startsWith("/zh-Hans/");
}

function stripGeneratedCopyPayloads(html) {
  return html
    .replace(/\sdata-code=(?:"[^"]*"|'[^']*')/gs, "")
    .replace(/<figure\b[^>]*\bclass=(["'])[^"']*\bnot-content\b[^"']*\1[^>]*>[\s\S]*?<\/figure>/gi, "");
}

function getTags(html, tagName) {
  const pattern = new RegExp(`<${tagName}\\b[^>]*>`, "gi");
  return [...html.matchAll(pattern)].map((match) => match[0]);
}

function getElementBlocks(html, tagName) {
  const pattern = new RegExp(`<${tagName}\\b[^>]*>[\\s\\S]*?<\\/${tagName}>`, "gi");
  return [...html.matchAll(pattern)].map((match) => match[0]);
}

function getAttrs(html, attrName) {
  const pattern = new RegExp(`\\b${attrName}=(["'])(.*?)\\1`, "gis");
  return [...html.matchAll(pattern)].map((match) => match[2]);
}

function decodeBasicEntities(value) {
  return value
    .replace(/&amp;/g, "&")
    .replace(/&#x2F;/gi, "/")
    .replace(/&#x3A;/gi, ":")
    .replace(/&#x22;/gi, "\"")
    .replace(/&#39;/g, "'");
}

function isExternalReference(value) {
  return /^(?:[a-z][a-z0-9+.-]*:)?\/\//i.test(value) ||
    /^(?:mailto|tel|javascript|data|blob):/i.test(value);
}

function normalizeReference(value, currentRoute) {
  const decoded = decodeBasicEntities(value.trim());
  if (!decoded || decoded.startsWith("#") || decoded.startsWith("?") || isExternalReference(decoded)) {
    if (decoded.startsWith(`${siteUrl}/`)) {
      const url = new URL(decoded);
      return `${url.pathname}${url.search}${url.hash}`;
    }
    return null;
  }

  const withoutHash = decoded.split("#")[0];
  const withoutQuery = withoutHash.split("?")[0];
  if (!withoutQuery) return null;

  if (withoutQuery.startsWith("/")) return withoutQuery;

  const base = currentRoute.endsWith("/") ? currentRoute : path.posix.dirname(currentRoute) + "/";
  return path.posix.normalize(path.posix.join(base, withoutQuery));
}

function localReferenceExists(referencePath) {
  const safePath = decodeURIComponent(referencePath);
  const normalized = safePath.replace(/^\/+/, "");
  const candidates = [];

  if (!normalized || normalized.endsWith("/")) {
    candidates.push(path.join(distRoot, normalized, "index.html"));
  } else if (path.extname(normalized)) {
    candidates.push(path.join(distRoot, normalized));
  } else {
    candidates.push(path.join(distRoot, normalized));
    candidates.push(path.join(distRoot, normalized, "index.html"));
    candidates.push(path.join(distRoot, `${normalized}.html`));
  }

  return candidates.some((candidate) => fs.existsSync(candidate));
}

function assertNormalPage(route, sanitizedHtml, issues) {
  const titles = getElementBlocks(sanitizedHtml, "title");
  if (titles.length !== 1) {
    issues.push(`${route}: expected 1 title, found ${titles.length}`);
  }

  const h1s = getElementBlocks(sanitizedHtml, "h1");
  if (h1s.length !== 1) {
    issues.push(`${route}: expected 1 visible h1, found ${h1s.length}`);
  }

  const canonicals = [...sanitizedHtml.matchAll(/<link\b[^>]*\brel=(["'])canonical\1[^>]*>/gi)]
    .map((match) => match[0]);
  if (canonicals.length !== 1) {
    issues.push(`${route}: expected 1 canonical, found ${canonicals.length}`);
  } else {
    const href = getAttrs(canonicals[0], "href")[0] ?? "";
    if (!href.startsWith(siteUrl)) {
      issues.push(`${route}: canonical is not on ${siteUrl}: ${href}`);
    }
  }

  if (route !== "/404.html") {
    const alternates = [...sanitizedHtml.matchAll(/<link\b[^>]*\brel=(["'])alternate\1[^>]*>/gi)]
      .map((match) => match[0]);
    if (alternates.length < 3) {
      issues.push(`${route}: expected locale alternates, found ${alternates.length}`);
    }
    for (const alternate of alternates) {
      const href = getAttrs(alternate, "href")[0] ?? "";
      if (href.includes("/zh-Hans")) {
        issues.push(`${route}: alternate still points to legacy zh-Hans URL: ${href}`);
      }
      if (!href.startsWith(siteUrl)) {
        issues.push(`${route}: alternate is not on ${siteUrl}: ${href}`);
      }
    }
  }
}

function assertLegacyRedirect(route, sanitizedHtml, issues) {
  const expectedTarget = route.replace(/^\/zh-Hans/, "/zh-cn");
  const canonicals = [...sanitizedHtml.matchAll(/<link\b[^>]*\brel=(["'])canonical\1[^>]*>/gi)]
    .map((match) => match[0]);
  const canonicalHref = canonicals.length === 1 ? getAttrs(canonicals[0], "href")[0] ?? "" : "";
  const expectedCanonical = `${siteUrl}${expectedTarget}`;

  if (!sanitizedHtml.includes("noindex")) {
    issues.push(`${route}: legacy redirect is missing noindex`);
  }
  if (!sanitizedHtml.includes("http-equiv=\"refresh\"") && !sanitizedHtml.includes("http-equiv='refresh'")) {
    issues.push(`${route}: legacy redirect is missing meta refresh`);
  }
  if (!sanitizedHtml.includes(`href="${expectedTarget}"`)) {
    issues.push(`${route}: legacy redirect body link does not point to ${expectedTarget}`);
  }
  if (canonicalHref !== expectedCanonical) {
    issues.push(`${route}: legacy canonical expected ${expectedCanonical}, found ${canonicalHref || "(missing)"}`);
  }
}

function auditHtmlFile(filePath, knownHtmlFiles, issues, summary) {
  const route = routeFromHtmlFile(filePath);
  const rawHtml = fs.readFileSync(filePath, "utf8");
  const sanitizedHtml = stripGeneratedCopyPayloads(rawHtml);

  summary.htmlFiles += 1;
  if (rawHtml.includes("\0")) {
    issues.push(`${route}: contains NUL byte`);
  }
  if (oldDomainPattern.test(rawHtml)) {
    issues.push(`${route}: contains old learning.airoads.org domain`);
  }
  if (docusaurusPattern.test(rawHtml)) {
    issues.push(`${route}: contains Docusaurus residue`);
  }

  if (isVerificationHtml(route)) {
    summary.verificationFiles += 1;
    return;
  }

  if (isLegacyZhHansRoute(route)) {
    summary.legacyRedirectPages += 1;
    assertLegacyRedirect(route, sanitizedHtml, issues);
  } else {
    assertNormalPage(route, sanitizedHtml, issues);
  }

  for (const imgTag of getTags(sanitizedHtml, "img")) {
    const alt = getAttrs(imgTag, "alt");
    if (alt.length === 0 || alt[0].trim() === "") {
      issues.push(`${route}: image missing non-empty alt text: ${imgTag.slice(0, 140)}`);
    }
  }

  const refs = [
    ...getAttrs(sanitizedHtml, "src"),
    ...getAttrs(sanitizedHtml, "href"),
  ];
  for (const ref of refs) {
    const localRef = normalizeReference(ref, route);
    if (!localRef) continue;
    if (!localReferenceExists(localRef)) {
      issues.push(`${route}: broken local reference ${ref}`);
    }
  }
}

function auditSitemaps(issues, summary) {
  const sitemapFiles = walkFiles(distRoot, (file) => path.basename(file).startsWith("sitemap") && file.endsWith(".xml"));
  summary.sitemapFiles = sitemapFiles.length;
  for (const file of sitemapFiles) {
    const rel = toPosix(path.relative(distRoot, file));
    const content = fs.readFileSync(file, "utf8");
    if (content.includes("/zh-Hans")) {
      issues.push(`${rel}: sitemap includes legacy zh-Hans URL`);
    }
    if (oldDomainPattern.test(content)) {
      issues.push(`${rel}: sitemap includes old learning.airoads.org domain`);
    }
    if (!content.includes(siteUrl)) {
      issues.push(`${rel}: sitemap does not include ${siteUrl}`);
    }
  }
}

function assertCoreMetadata(issues) {
  const homepage = path.join(distRoot, "index.html");
  if (!fs.existsSync(homepage)) {
    issues.push("/: missing homepage");
    return;
  }
  const html = fs.readFileSync(homepage, "utf8");
  const required = [
    "AI Roads",
    `${siteUrl}/img/social-card.png`,
    "/img/favicon.svg",
    "application/ld+json",
  ];
  for (const text of required) {
    if (!html.includes(text)) {
      issues.push(`/: missing expected metadata ${text}`);
    }
  }
}

if (!fs.existsSync(distRoot)) {
  throw new Error(`Missing build output directory: ${distRoot}`);
}

const htmlFiles = walkFiles(distRoot, (file) => file.endsWith(".html"));
const knownHtmlFiles = new Set(htmlFiles.map((file) => toPosix(path.relative(distRoot, file))));
const issues = [];
const summary = {
  htmlFiles: 0,
  legacyRedirectPages: 0,
  verificationFiles: 0,
  sitemapFiles: 0,
};

for (const file of htmlFiles) {
  auditHtmlFile(file, knownHtmlFiles, issues, summary);
}
auditSitemaps(issues, summary);
assertCoreMetadata(issues);

if (issues.length > 0) {
  console.error(JSON.stringify({ ...summary, issues: issues.slice(0, 80), issueCount: issues.length }, null, 2));
  process.exit(1);
}

console.log(JSON.stringify({ ...summary, status: "passed" }, null, 2));
