#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const projectRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const distRoot = path.resolve(projectRoot, process.argv[2] ?? "dist");
const siteUrl = "https://airoads.org";
const indexNowKey = "a4e8d4b6c0f1424c910f2ad7360b8e5f";
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

function getMetaTagsByAttr(html, attrName, value) {
  return getTags(html, "meta").filter((tag) =>
    getAttrs(tag, attrName).some((attrValue) => attrValue.toLowerCase() === value.toLowerCase()),
  );
}

function getMetaContentByAttr(html, attrName, value) {
  const tags = getMetaTagsByAttr(html, attrName, value);
  return tags.map((tag) => getAttrs(tag, "content")[0] ?? "");
}

function getJsonLdPayloads(html) {
  const pattern =
    /<script\b[^>]*\btype=(["'])application\/ld\+json\1[^>]*>([\s\S]*?)<\/script>/gi;
  return [...html.matchAll(pattern)].map((match) => match[2].trim());
}

function schemaTypesFromPayload(payload, route, issues) {
  try {
    const parsed = JSON.parse(payload);
    const nodes = Array.isArray(parsed["@graph"]) ? parsed["@graph"] : [parsed];
    return nodes.flatMap((node) => {
      const type = node?.["@type"];
      return Array.isArray(type) ? type : type ? [type] : [];
    });
  } catch (error) {
    issues.push(`${route}: invalid JSON-LD payload: ${error.message}`);
    return [];
  }
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

function htmlRouteExists(route, knownHtmlFiles) {
  let normalized = decodeURIComponent(route).replace(/^\/+/, "");
  const candidates = [];

  if (!normalized || normalized.endsWith("/")) {
    candidates.push(`${normalized}index.html`);
  } else if (path.posix.extname(normalized)) {
    candidates.push(normalized);
  } else {
    candidates.push(`${normalized}/index.html`);
    candidates.push(`${normalized}.html`);
  }

  return candidates.some((candidate) => knownHtmlFiles.has(candidate));
}

function explicitLocaleForRoute(route) {
  if (route === "/zh-cn/" || route.startsWith("/zh-cn/")) return "zh-cn";
  if (route === "/ja/" || route.startsWith("/ja/")) return "ja";
  if (route === "/zh-Hans/" || route.startsWith("/zh-Hans/")) return "zh-Hans";
  return "en";
}

function isPageRoute(route, knownHtmlFiles) {
  return htmlRouteExists(route, knownHtmlFiles);
}

function assertLocalizedReferencesStayLocalized(route, sanitizedHtml, knownHtmlFiles, issues) {
  const currentLocale = explicitLocaleForRoute(route);
  if (!["zh-cn", "ja"].includes(currentLocale) || isLegacyZhHansRoute(route)) return;

  for (const anchorTag of getTags(sanitizedHtml, "a")) {
    const href = getAttrs(anchorTag, "href")[0];
    if (!href) continue;

    const localRef = normalizeReference(href, route);
    if (!localRef || !isPageRoute(localRef, knownHtmlFiles)) continue;

    const targetLocale = explicitLocaleForRoute(localRef);
    if (targetLocale !== "en") continue;

    const localizedRoute = localRef === "/" ? `/${currentLocale}/` : `/${currentLocale}${localRef}`;
    if (htmlRouteExists(localizedRoute, knownHtmlFiles)) {
      issues.push(
        `${route}: localized page links to English route ${href}; use ${localizedRoute} instead`,
      );
    }
  }
}

function assertNormalPage(route, sanitizedHtml, issues) {
  if (!/<html\b[^>]*\blang=(["'])[^"']+\1/i.test(sanitizedHtml)) {
    issues.push(`${route}: missing html lang attribute`);
  }

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

  if (route === "/404.html") {
    const content = getMetaContentByAttr(sanitizedHtml, "name", "robots").join(",").toLowerCase();
    if (!content.includes("noindex")) {
      issues.push(`${route}: 404 page should include noindex`);
    }
    return;
  }

  const descriptions = getMetaTagsByAttr(sanitizedHtml, "name", "description");
  if (descriptions.length !== 1) {
    issues.push(`${route}: expected 1 meta description, found ${descriptions.length}`);
  } else if (!(getAttrs(descriptions[0], "content")[0] ?? "").trim()) {
    issues.push(`${route}: meta description is empty`);
  }

  const robots = getMetaTagsByAttr(sanitizedHtml, "name", "robots");
  if (robots.length !== 1) {
    issues.push(`${route}: expected 1 robots meta tag, found ${robots.length}`);
  } else {
    const content = (getAttrs(robots[0], "content")[0] ?? "").toLowerCase();
    for (const directive of ["index", "follow", "max-image-preview:large"]) {
      if (!content.includes(directive)) {
        issues.push(`${route}: robots meta tag is missing ${directive}`);
      }
    }
  }

  const canonicalHref = canonicals.length === 1 ? getAttrs(canonicals[0], "href")[0] ?? "" : "";
  const ogUrl = getMetaContentByAttr(sanitizedHtml, "property", "og:url")[0] ?? "";
  if (ogUrl && ogUrl !== canonicalHref) {
    issues.push(`${route}: og:url does not match canonical: ${ogUrl} !== ${canonicalHref}`);
  }

  const requiredOpenGraph = ["og:title", "og:description", "og:url", "og:image", "og:site_name", "og:type"];
  for (const property of requiredOpenGraph) {
    const values = getMetaContentByAttr(sanitizedHtml, "property", property).filter(Boolean);
    if (values.length === 0) {
      issues.push(`${route}: missing Open Graph metadata ${property}`);
    }
  }

  const requiredTwitter = ["twitter:card", "twitter:image", "twitter:image:alt"];
  for (const name of requiredTwitter) {
    const values = getMetaContentByAttr(sanitizedHtml, "name", name).filter(Boolean);
    if (values.length === 0) {
      issues.push(`${route}: missing Twitter metadata ${name}`);
    }
  }

  const jsonLdPayloads = getJsonLdPayloads(sanitizedHtml);
  if (jsonLdPayloads.length === 0) {
    issues.push(`${route}: missing JSON-LD structured data`);
  }
  for (const payload of jsonLdPayloads) {
    schemaTypesFromPayload(payload, route, issues);
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
  if (rawHtml.includes("course-terminal-output")) {
    issues.push(`${route}: contains deprecated custom terminal output block`);
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
    assertLocalizedReferencesStayLocalized(route, sanitizedHtml, knownHtmlFiles, issues);
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
  const sitemapIndex = path.join(distRoot, "sitemap-index.xml");
  const sitemapZero = path.join(distRoot, "sitemap-0.xml");
  if (!fs.existsSync(sitemapIndex)) {
    issues.push("sitemap-index.xml: missing sitemap index");
  }
  if (!fs.existsSync(sitemapZero)) {
    issues.push("sitemap-0.xml: missing primary sitemap");
  }

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

  if (fs.existsSync(sitemapIndex)) {
    const content = fs.readFileSync(sitemapIndex, "utf8");
    if (!/<sitemapindex\b/i.test(content)) {
      issues.push("sitemap-index.xml: expected sitemapindex root");
    }
    if (!content.includes(`${siteUrl}/sitemap-0.xml`)) {
      issues.push("sitemap-index.xml: missing sitemap-0.xml location");
    }
  }

  if (fs.existsSync(sitemapZero)) {
    const content = fs.readFileSync(sitemapZero, "utf8");
    for (const expectedUrl of [`${siteUrl}/`, `${siteUrl}/zh-cn/`, `${siteUrl}/ja/`]) {
      if (!content.includes(`<loc>${expectedUrl}</loc>`)) {
        issues.push(`sitemap-0.xml: missing homepage URL ${expectedUrl}`);
      }
    }
  }
}

function assertSeoAssets(issues, summary) {
  const requiredFiles = [
    "robots.txt",
    "BingSiteAuth.xml",
    "googlebe050d9d769c46f0.html",
    `${indexNowKey}.txt`,
  ];
  summary.seoAssetFiles = 0;

  for (const relativeFile of requiredFiles) {
    const file = path.join(distRoot, relativeFile);
    if (!fs.existsSync(file)) {
      issues.push(`${relativeFile}: missing SEO asset`);
    } else {
      summary.seoAssetFiles += 1;
    }
  }

  const robotsFile = path.join(distRoot, "robots.txt");
  if (fs.existsSync(robotsFile)) {
    const robots = fs.readFileSync(robotsFile, "utf8");
    if (!/^User-agent:\s*\*/im.test(robots)) {
      issues.push("robots.txt: missing wildcard user-agent");
    }
    if (!/^Allow:\s*\/\s*$/im.test(robots)) {
      issues.push("robots.txt: missing Allow: /");
    }
    if (!/^Sitemap:\s*https:\/\/airoads\.org\/sitemap-index\.xml\s*$/im.test(robots)) {
      issues.push("robots.txt: sitemap does not point to https://airoads.org/sitemap-index.xml");
    }
    if (/^Sitemap:\s*https:\/\/airoads\.org\/sitemap\.xml\s*$/im.test(robots)) {
      issues.push("robots.txt: still points to missing sitemap.xml");
    }
  }

  const indexNowFile = path.join(distRoot, `${indexNowKey}.txt`);
  if (fs.existsSync(indexNowFile)) {
    const content = fs.readFileSync(indexNowFile, "utf8").trim();
    if (content !== indexNowKey) {
      issues.push(`${indexNowKey}.txt: IndexNow key file content mismatch`);
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

  const schemaTypes = new Set(
    getJsonLdPayloads(html).flatMap((payload) => schemaTypesFromPayload(payload, "/", issues)),
  );
  for (const type of ["Organization", "WebSite", "Course"]) {
    if (!schemaTypes.has(type)) {
      issues.push(`/: missing ${type} in JSON-LD graph`);
    }
  }
}

function htmlFileForRoute(route) {
  const normalized = route.replace(/^\/+/, "");
  if (!normalized || route.endsWith("/")) {
    return path.join(distRoot, normalized, "index.html");
  }
  return path.join(distRoot, normalized);
}

function snippetAround(html, text, radius = 1600) {
  const index = html.indexOf(text);
  if (index === -1) return null;
  return html.slice(Math.max(0, index - radius), Math.min(html.length, index + text.length + radius));
}

function assertTextRenderedInside({ route, text, requiredClass, forbiddenClass }, issues) {
  const file = htmlFileForRoute(route);
  if (!fs.existsSync(file)) {
    issues.push(`${route}: missing page for course presentation semantic check`);
    return;
  }

  const html = fs.readFileSync(file, "utf8");
  const snippet = snippetAround(html, text);
  if (!snippet) {
    issues.push(`${route}: missing text for course presentation semantic check: ${text}`);
    return;
  }

  if (requiredClass && !snippet.includes(requiredClass)) {
    issues.push(`${route}: "${text}" should render inside ${requiredClass}`);
  }
  if (forbiddenClass && snippet.includes(forbiddenClass)) {
    issues.push(`${route}: "${text}" should not render inside ${forbiddenClass}`);
  }
}

function assertCoursePresentationSemantics(issues) {
  const checks = [
    {
      route: "/zh-cn/ch05-machine-learning/ch01-ml-basics/02-sklearn-intro/",
      text: "机器学习问题",
      requiredClass: "course-evidence-card",
      forbiddenClass: "is-terminal",
    },
    {
      route: "/zh-cn/ch05-machine-learning/ch01-ml-basics/02-sklearn-intro/",
      text: "在全部数据上 fit scaler",
      requiredClass: "course-flow-line",
      forbiddenClass: "is-terminal",
    },
    {
      route: "/zh-cn/ch05-machine-learning/ch01-ml-basics/02-sklearn-intro/",
      text: "只在训练集 fit scaler",
      requiredClass: "course-flow-line",
      forbiddenClass: "is-terminal",
    },
    {
      route: "/zh-cn/ch05-machine-learning/ch01-ml-basics/05-sklearn-matplotlib-workshop/",
      text: "Sample 0: predicted=class_0",
      requiredClass: "is-terminal",
    },
    {
      route: "/zh-cn/ch05-machine-learning/ch01-ml-basics/05-sklearn-matplotlib-workshop/",
      text: "X shape: (178, 13)",
      requiredClass: "is-terminal",
    },
  ];

  for (const check of checks) {
    assertTextRenderedInside(check, issues);
  }
}

function assertLocalizedHomepageNavigation(issues) {
  const checks = [
    {
      route: "/zh-cn/",
      links: [
        'href="/zh-cn/intro/quick-experience/"',
        'href="/zh-cn/intro/learning-path/"',
        'href="/zh-cn/ch01-tools/"',
      ],
    },
    {
      route: "/ja/",
      links: [
        'href="/ja/intro/quick-experience/"',
        'href="/ja/intro/learning-path/"',
        'href="/ja/ch01-tools/"',
      ],
    },
  ];

  for (const check of checks) {
    const file = htmlFileForRoute(check.route);
    if (!fs.existsSync(file)) {
      issues.push(`${check.route}: missing localized homepage`);
      continue;
    }

    const html = fs.readFileSync(file, "utf8");
    if (/<a\b[^>]*\bhref=(["'])\.\.?\//i.test(html)) {
      issues.push(`${check.route}: localized homepage still has relative internal anchor links`);
    }
    for (const link of check.links) {
      if (!html.includes(link)) {
        issues.push(`${check.route}: homepage is missing localized navigation link ${link}`);
      }
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
  seoAssetFiles: 0,
};

for (const file of htmlFiles) {
  auditHtmlFile(file, knownHtmlFiles, issues, summary);
}
auditSitemaps(issues, summary);
assertSeoAssets(issues, summary);
assertCoreMetadata(issues);
assertCoursePresentationSemantics(issues);
assertLocalizedHomepageNavigation(issues);

if (issues.length > 0) {
  console.error(JSON.stringify({ ...summary, issues: issues.slice(0, 80), issueCount: issues.length }, null, 2));
  process.exit(1);
}

console.log(JSON.stringify({ ...summary, status: "passed" }, null, 2));
