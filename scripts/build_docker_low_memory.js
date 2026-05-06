#!/usr/bin/env node

const { spawnSync } = require("child_process");
const fs = require("fs");
const path = require("path");

const projectRoot = path.resolve(__dirname, "..");
const locales = ["en", "zh-Hans", "ja"];
const finalBuildDir = path.join(projectRoot, "build");
const tempRoot = path.join(projectRoot, ".tmp-docker-build");

function run(command, args, extraEnv = {}) {
  const result = spawnSync(command, args, {
    cwd: projectRoot,
    stdio: "inherit",
    env: {
      ...process.env,
      DOCUSAURUS_SSR_CONCURRENCY: "1",
      NODE_OPTIONS: [
        process.env.NODE_OPTIONS || "",
        "--max-old-space-size=1024",
        "--expose-gc",
      ]
        .filter(Boolean)
        .join(" "),
      ...extraEnv,
    },
  });

  if (result.status !== 0) {
    process.exit(result.status || 1);
  }
}

function removeDirectory(relativePath) {
  fs.rmSync(path.join(projectRoot, relativePath), {
    recursive: true,
    force: true,
  });
}

function copyDirectoryContents(sourceDir, targetDir) {
  if (!fs.existsSync(sourceDir)) {
    return;
  }
  fs.mkdirSync(targetDir, { recursive: true });
  for (const entry of fs.readdirSync(sourceDir, { withFileTypes: true })) {
    const sourcePath = path.join(sourceDir, entry.name);
    const targetPath = path.join(targetDir, entry.name);
    if (entry.isDirectory()) {
      fs.cpSync(sourcePath, targetPath, { recursive: true });
    } else if (entry.isFile()) {
      fs.copyFileSync(sourcePath, targetPath);
    }
  }
}

function forceGarbageCollection() {
  if (global.gc) {
    global.gc();
  }
}

removeDirectory("build");
removeDirectory(".docusaurus");
removeDirectory(".tmp-docker-build");
fs.mkdirSync(tempRoot, { recursive: true });

for (const locale of locales) {
  console.log(`\n[build:docker] Building locale: ${locale}`);
  const localeOutDir = path.join(tempRoot, locale);
  removeDirectory(path.relative(projectRoot, localeOutDir));
  run("npx", ["docusaurus", "build", "--locale", locale, "--out-dir", localeOutDir, "--no-minify"]);
  copyDirectoryContents(localeOutDir, finalBuildDir);
  removeDirectory(path.relative(projectRoot, localeOutDir));
  removeDirectory(".docusaurus");
  forceGarbageCollection();
}

run("node", ["scripts/strip_build_null_bytes.js", "build"]);
run("node", ["scripts/merge_localized_sitemaps.js", "build"]);
removeDirectory(".tmp-docker-build");
