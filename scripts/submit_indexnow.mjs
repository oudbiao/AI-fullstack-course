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
const endpoint = process.env.INDEXNOW_ENDPOINT ?? "https://www.bing.com/indexnow";
const maxUrlsPerRequest = 10000;

function getArgValue(name) {
  const inline = args.find((arg) => arg.startsWith(`${name}=`));
  if (inline) {
    return inline.slice(name.length + 1);
  }

  const index = args.indexOf(name);
  if (index !== -1) {
    return args[index + 1];
  }

  return undefined;
}

const sitemapUrl = getArgValue("--sitemap-url") ?? process.env.INDEXNOW_SITEMAP_URL;

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

async function fetchText(url) {
  const response = await fetch(url, {
    headers: {
      accept: "application/xml,text/xml,text/plain,*/*",
      "user-agent": "AI-fullstack-course IndexNow notifier",
    },
  });

  if (!response.ok) {
    throw new Error(`Failed to fetch ${url}: ${response.status} ${response.statusText}`);
  }

  return response.text();
}

function extractLocations(xml) {
  return [...xml.matchAll(/<loc>\s*([^<\s]+)\s*<\/loc>/g)].map((match) => match[1].trim());
}

async function readOnlineSitemapUrls(url, visited = new Set()) {
  if (visited.has(url)) {
    return [];
  }
  visited.add(url);

  const xml = await fetchText(url);
  const locations = extractLocations(xml);

  if (/<sitemapindex\b/i.test(xml)) {
    const nestedUrls = [];
    for (const location of locations) {
      if (!location.startsWith(`${siteUrl}/`) || !location.endsWith(".xml")) {
        continue;
      }
      nestedUrls.push(...(await readOnlineSitemapUrls(location, visited)));
    }
    return [...new Set(nestedUrls)].sort();
  }

  return [...new Set(locations.filter((location) => location.startsWith(`${siteUrl}/`)))].sort();
}

async function assertOnlineKeyFile() {
  const content = (await fetchText(keyLocation)).trim();
  if (content !== indexNowKey) {
    throw new Error(`IndexNow key URL content mismatch: ${keyLocation}`);
  }
}

function chunk(items, size) {
  const chunks = [];
  for (let index = 0; index < items.length; index += size) {
    chunks.push(items.slice(index, index + size));
  }
  return chunks;
}

let urlList;
if (sitemapUrl) {
  urlList = await readOnlineSitemapUrls(sitemapUrl);
  await assertOnlineKeyFile();
} else {
  urlList = readSitemapUrls();
  assertKeyFile();
}

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
        endpoint,
        host,
        keyLocation,
        source: sitemapUrl ? "online-sitemap" : "dist",
        sitemapUrl: sitemapUrl ?? null,
        urlCount: urlList.length,
        sampleUrls: urlList.slice(0, 5),
      },
      null,
      2,
    ),
  );
  process.exit(0);
}

let submitted = 0;
for (const urlChunk of chunk(urlList, maxUrlsPerRequest)) {
  const response = await fetch(endpoint, {
    method: "POST",
    headers: {
      "content-type": "application/json; charset=utf-8",
    },
    body: JSON.stringify({ ...payload, urlList: urlChunk }),
  });

  if (!response.ok) {
    const body = await response.text();
    throw new Error(`IndexNow submission failed with ${response.status}: ${body}`);
  }

  submitted += urlChunk.length;
}

console.log(JSON.stringify({ status: "submitted", urlCount: submitted }, null, 2));
