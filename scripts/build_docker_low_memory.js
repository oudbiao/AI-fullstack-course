#!/usr/bin/env node

const { spawnSync } = require("child_process");
const fs = require("fs");
const path = require("path");

const projectRoot = path.resolve(__dirname, "..");
const locales = ["en", "zh-Hans", "ja"];
const finalBuildDir = path.join(projectRoot, "build");
const docusaurusBin = path.join(
  projectRoot,
  "node_modules",
  ".bin",
  process.platform === "win32" ? "docusaurus.cmd" : "docusaurus",
);
const dockerBuildOldSpace = process.env.DOCKER_BUILD_NODE_OLD_SPACE || "1536";
const localizedStaticLocales = locales.filter((locale) => locale !== "en");
const textExtensions = new Set([
  ".css",
  ".html",
  ".js",
  ".json",
  ".map",
  ".txt",
  ".xml",
]);

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
      DOCUSAURUS_DISABLE_LAST_UPDATE: "true",
      CI: "true",
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

function walkFiles(directory, callback) {
  if (!fs.existsSync(directory)) {
    return;
  }

  for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
    const entryPath = path.join(directory, entry.name);
    if (entry.isDirectory()) {
      walkFiles(entryPath, callback);
    } else if (entry.isFile()) {
      callback(entryPath);
    }
  }
}

function rewriteLocalizedImageUrls() {
  let rewrittenFiles = 0;
  let replacementCount = 0;

  for (const locale of localizedStaticLocales) {
    const localeBuildDir = path.join(finalBuildDir, locale);
    const from = `/${locale}/img/`;
    const to = "/img/";

    walkFiles(localeBuildDir, (filePath) => {
      if (!textExtensions.has(path.extname(filePath))) {
        return;
      }

      const original = fs.readFileSync(filePath, "utf8");
      if (!original.includes(from)) {
        return;
      }

      const updated = original.split(from).join(to);
      fs.writeFileSync(filePath, updated);
      rewrittenFiles += 1;
      replacementCount += original.split(from).length - 1;
    });

    fs.rmSync(path.join(localeBuildDir, "img"), {
      recursive: true,
      force: true,
    });
  }

  console.log(
    `[build:docker] Shared static images: rewrote ${replacementCount} URL(s) in ${rewrittenFiles} file(s) and removed localized img copies`,
  );
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

  for (const locale of localizedStaticLocales) {
    const localizedImgDir = path.join(finalBuildDir, locale, "img");
    if (fs.existsSync(localizedImgDir)) {
      throw new Error(`Localized static image directory should have been removed: ${path.relative(projectRoot, localizedImgDir)}`);
    }
  }

  console.log("[build:docker] Locale output validation passed");
}

function forceGarbageCollection() {
  if (global.gc) {
    global.gc();
  }
}

if (!fs.existsSync(docusaurusBin)) {
  throw new Error(`Missing local Docusaurus binary: ${path.relative(projectRoot, docusaurusBin)}`);
}

removeDirectory("build");
removeDirectory(".docusaurus");
removeDirectory(".tmp-docker-build");

console.log(`\n[build:docker] Building locales: ${locales.join(", ")}`);
const localeArgs = locales.flatMap((locale) => ["--locale", locale]);
run(docusaurusBin, [
  "build",
  ...localeArgs,
  "--out-dir",
  finalBuildDir,
  "--no-minify",
]);
rewriteLocalizedImageUrls();
removeDirectory(".docusaurus");
forceGarbageCollection();

run("node", ["scripts/strip_build_null_bytes.js", "build"]);
run("node", ["scripts/merge_localized_sitemaps.js", "build"]);
validateLocaleOutput();
