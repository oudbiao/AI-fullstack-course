#!/usr/bin/env node

const { spawnSync } = require("child_process");
const fs = require("fs");
const path = require("path");

const projectRoot = path.resolve(__dirname, "..");
const finalBuildDir = path.join(projectRoot, "dist");
const astroBin = path.join(
  projectRoot,
  "node_modules",
  ".bin",
  process.platform === "win32" ? "astro.cmd" : "astro",
);
const dockerBuildOldSpace = process.env.DOCKER_BUILD_NODE_OLD_SPACE || "1536";

function getNodeOptions() {
  const baseOptions = (process.env.NODE_OPTIONS || "")
    .split(/\s+/)
    .filter((option) => option && !option.startsWith("--max-old-space-size="));

  return [...baseOptions, `--max-old-space-size=${dockerBuildOldSpace}`].join(" ");
}

function run(command, args, extraEnv = {}) {
  const result = spawnSync(command, args, {
    cwd: projectRoot,
    stdio: "inherit",
    env: {
      ...process.env,
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

function assertFileNotContains(relativePath, unexpectedText) {
  const filePath = path.join(projectRoot, relativePath);
  if (!fs.existsSync(filePath)) {
    return;
  }

  const content = fs.readFileSync(filePath, "utf8");
  if (content.includes(unexpectedText)) {
    throw new Error(`Expected ${relativePath} not to contain: ${unexpectedText}`);
  }
}

function validateOutput() {
  const checks = [
    ["dist/index.html", 'lang="en-US"'],
    ["dist/index.html", 'value="/zh-cn/"'],
    ["dist/index.html", 'value="/ja/"'],
    ["dist/zh-cn/index.html", 'lang="zh-CN"'],
    ["dist/ja/index.html", 'lang="ja-JP"'],
    ["dist/zh-Hans/index.html", 'href="/zh-cn/"'],
    ["dist/sitemap-index.xml", "sitemap-0.xml"],
    ["dist/sitemap.xml", "sitemap-0.xml"],
  ];

  for (const [relativePath, expectedText] of checks) {
    assertFileContains(relativePath, expectedText);
  }

  assertFileNotContains("dist/sitemap-0.xml", "/zh-Hans");

  console.log("[build:docker] Astro Starlight output validation passed");
}

if (!fs.existsSync(astroBin)) {
  throw new Error(`Missing local Astro binary: ${path.relative(projectRoot, astroBin)}`);
}

removeDirectory("dist");
removeDirectory(".astro");

console.log("\n[build:docker] Building Astro Starlight site");
run("npm", ["run", "build"]);

if (!fs.existsSync(finalBuildDir)) {
  throw new Error("Astro build did not create dist/");
}

validateOutput();
