#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";

const root = path.resolve(process.argv[2] || "dist");
const sitemapIndex = path.join(root, "sitemap-index.xml");
const sitemapParts = fs.existsSync(root)
  ? fs.readdirSync(root).filter((file) => /^sitemap-\d+\.xml$/.test(file)).sort()
  : [];
const sitemapAlias = path.join(root, "sitemap.xml");

if (!fs.existsSync(root)) {
  console.log(`create_sitemap_alias: skipped, ${root} does not exist`);
  process.exit(0);
}

if (!fs.existsSync(sitemapIndex)) {
  throw new Error(`create_sitemap_alias: missing ${path.relative(process.cwd(), sitemapIndex)}`);
}

if (sitemapParts.length === 1) {
  fs.copyFileSync(path.join(root, sitemapParts[0]), sitemapAlias);
  console.log(`create_sitemap_alias: wrote sitemap.xml from ${sitemapParts[0]}`);
} else {
  fs.copyFileSync(sitemapIndex, sitemapAlias);
  console.log("create_sitemap_alias: wrote sitemap.xml from sitemap-index.xml");
}
