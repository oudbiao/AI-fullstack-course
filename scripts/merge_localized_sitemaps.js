#!/usr/bin/env node

const fs = require("fs");
const path = require("path");

const buildRoot = path.resolve(process.argv[2] || "build");
const rootSitemap = path.join(buildRoot, "sitemap.xml");
const localeSitemaps = ["zh-Hans", "ja"].map((locale) =>
  path.join(buildRoot, locale, "sitemap.xml"),
);

const namespaces =
  'xmlns="http://www.sitemaps.org/schemas/sitemap/0.9" ' +
  'xmlns:news="http://www.google.com/schemas/sitemap-news/0.9" ' +
  'xmlns:xhtml="http://www.w3.org/1999/xhtml" ' +
  'xmlns:image="http://www.google.com/schemas/sitemap-image/1.1" ' +
  'xmlns:video="http://www.google.com/schemas/sitemap-video/1.1"';

function readUrls(file) {
  if (!fs.existsSync(file)) {
    return [];
  }
  const xml = fs.readFileSync(file, "utf8");
  return [...xml.matchAll(/<url>[\s\S]*?<\/url>/g)].map((match) => match[0]);
}

if (!fs.existsSync(rootSitemap)) {
  console.log(`merge_localized_sitemaps: skipped, ${rootSitemap} does not exist`);
  process.exit(0);
}

const seen = new Set();
const urls = [];

for (const file of [rootSitemap, ...localeSitemaps]) {
  for (const url of readUrls(file)) {
    const loc = url.match(/<loc>(.*?)<\/loc>/)?.[1] || url;
    if (seen.has(loc)) {
      continue;
    }
    seen.add(loc);
    urls.push(url);
  }
}

const merged = `<?xml version="1.0" encoding="UTF-8"?>\n<urlset ${namespaces}>\n${urls.join("\n")}\n</urlset>\n`;
fs.writeFileSync(rootSitemap, merged);

console.log(
  `merge_localized_sitemaps: merged ${urls.length} URLs into ${path.relative(process.cwd(), rootSitemap)}`,
);
