#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";

const root = path.resolve(process.argv[2] || "dist");
const legacyPathPattern = /<url>\s*<loc>https?:\/\/[^<]+\/zh-Hans(?:\/[^<]*)?<\/loc>[\s\S]*?<\/url>\s*/g;

let removed = 0;
let filesChanged = 0;

if (!fs.existsSync(root)) {
  console.log(`filter_sitemap_redirects: skipped, ${root} does not exist`);
  process.exit(0);
}

for (const entry of fs.readdirSync(root, { withFileTypes: true })) {
  if (!entry.isFile() || !/^sitemap-\d+\.xml$/.test(entry.name)) {
    continue;
  }

  const filePath = path.join(root, entry.name);
  const original = fs.readFileSync(filePath, "utf8");
  const updated = original.replace(legacyPathPattern, () => {
    removed += 1;
    return "";
  });

  if (updated !== original) {
    fs.writeFileSync(filePath, updated);
    filesChanged += 1;
  }
}

console.log(
  `filter_sitemap_redirects: removed ${removed} legacy zh-Hans URL(s) from ${filesChanged} sitemap file(s)`,
);
