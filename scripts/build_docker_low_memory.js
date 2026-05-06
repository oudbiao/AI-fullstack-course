#!/usr/bin/env node

const { spawnSync } = require("child_process");
const fs = require("fs");
const path = require("path");

const projectRoot = path.resolve(__dirname, "..");
const locales = ["en", "zh-Hans", "ja"];
const finalBuildDir = path.join(projectRoot, "build");
const dockerBuildOldSpace = process.env.DOCKER_BUILD_NODE_OLD_SPACE || "1536";

function getNodeOptions() {
  const baseOptions = (process.env.NODE_OPTIONS || "")
    .split(/\s+/)
    .filter(
      (option) =>
        option &&
        !option.startsWith("--max-old-space-size=") &&
        option !== "--expose-gc",
    );

  // Disable minification in the build command and keep SSR concurrency low.
  // This keeps Docker builds lighter while letting Docusaurus generate correct
  // locale paths and language-switch links in one i18n-aware build.
  return [...baseOptions, `--max-old-space-size=${dockerBuildOldSpace}`, "--expose-gc"].join(
    " ",
  );
}

function run(command, args, extraEnv = {}) {
  const result = spawnSync(command, args, {
    cwd: projectRoot,
    stdio: "inherit",
    env: {
      ...process.env,
      DOCUSAURUS_SSR_CONCURRENCY: "1",
      NODE_OPTIONS: getNodeOptions(),
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

function assertFileContains(relativePath, expectedText) {
  const filePath = path.join(projectRoot, relativePath);
  if (!fs.existsSync(filePath)) {
    throw new Error(`Missing expected build file: ${relativePath}`);
  }

  const content = fs.readFileSync(filePath, "utf8");
  if (!content.includes(expectedText)) {
    throw new Error(`Expected ${relativePath} to contain: ${expectedText}`);
  }
}

function validateLocaleOutput() {
  const checks = [
    ["build/index.html", 'name="docusaurus_locale" content="en"'],
    ["build/index.html", 'href="/zh-Hans/"'],
    ["build/index.html", 'href="/ja/"'],
    ["build/zh-Hans/index.html", 'name="docusaurus_locale" content="zh-Hans"'],
    ["build/ja/index.html", 'name="docusaurus_locale" content="ja"'],
  ];

  for (const [relativePath, expectedText] of checks) {
    assertFileContains(relativePath, expectedText);
  }

  console.log("[build:docker] Locale output validation passed");
}

function forceGarbageCollection() {
  if (global.gc) {
    global.gc();
  }
}

removeDirectory("build");
removeDirectory(".docusaurus");
removeDirectory(".tmp-docker-build");

console.log(`\n[build:docker] Building locales: ${locales.join(", ")}`);
const localeArgs = locales.flatMap((locale) => ["--locale", locale]);
run("npx", [
  "docusaurus",
  "build",
  ...localeArgs,
  "--out-dir",
  finalBuildDir,
  "--no-minify",
]);
removeDirectory(".docusaurus");
forceGarbageCollection();

run("node", ["scripts/strip_build_null_bytes.js", "build"]);
run("node", ["scripts/merge_localized_sitemaps.js", "build"]);
validateLocaleOutput();
