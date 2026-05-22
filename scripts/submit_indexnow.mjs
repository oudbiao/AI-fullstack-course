#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const projectRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const args = process.argv.slice(2);
const dryRun = args.includes("--dry-run");
const distArg = args.find((arg) => !arg.startsWith("--")) ?? "dist";
const distRoot = path.resolve(projectRoot, distArg);
const siteUrl = "https://airoads.org";
const host = "airoads.org";
const indexNowKey = "a4e8d4b6c0f1424c910f2ad7360b8e5f";
const keyLocation = `${siteUrl}/${indexNowKey}.txt`;

function readSitemapUrls() {
  if (!fs.existsSync(distRoot)) {
    throw new Error(`Missing build output directory: ${distRoot}`);
  }

  const sitemapFiles = fs
    .readdirSync(distRoot, { withFileTypes: true })
    .filter((entry) => entry.isFile() && /^sitemap-\d+\.xml$/.test(entry.name))
    .map((entry) => path.join(distRoot, entry.name))
    .sort();

  if (sitemapFiles.length === 0) {
    throw new Error(`No sitemap-N.xml files found in ${distRoot}; run npm run build first.`);
  }

  const urls = [];
  for (const sitemapFile of sitemapFiles) {
    const xml = fs.readFileSync(sitemapFile, "utf8");
    for (const match of xml.matchAll(/<loc>(https:\/\/airoads\.org\/[^<]*)<\/loc>/g)) {
      urls.push(match[1]);
    }
  }

  return [...new Set(urls)].sort();
}

function assertKeyFile() {
  const keyFile = path.join(distRoot, `${indexNowKey}.txt`);
  if (!fs.existsSync(keyFile)) {
    throw new Error(`Missing IndexNow key file in dist: ${keyFile}; run npm run build first.`);
  }

  const content = fs.readFileSync(keyFile, "utf8").trim();
  if (content !== indexNowKey) {
    throw new Error(`IndexNow key file content mismatch: ${keyFile}`);
  }
}

const urlList = readSitemapUrls();
assertKeyFile();

const payload = {
  host,
  key: indexNowKey,
  keyLocation,
  urlList,
};

if (dryRun) {
  console.log(
    JSON.stringify(
      {
        endpoint: "https://api.indexnow.org/indexnow",
        host,
        keyLocation,
        urlCount: urlList.length,
        sampleUrls: urlList.slice(0, 5),
      },
      null,
      2,
    ),
  );
  process.exit(0);
}

const response = await fetch("https://api.indexnow.org/indexnow", {
  method: "POST",
  headers: {
    "content-type": "application/json; charset=utf-8",
  },
  body: JSON.stringify(payload),
});

if (!response.ok) {
  const body = await response.text();
  throw new Error(`IndexNow submission failed with ${response.status}: ${body}`);
}

console.log(JSON.stringify({ status: "submitted", urlCount: urlList.length }, null, 2));
