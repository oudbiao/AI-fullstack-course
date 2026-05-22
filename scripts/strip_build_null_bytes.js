#!/usr/bin/env node

const fs = require("fs");
const path = require("path");

const root = path.resolve(process.argv[2] || "dist");
let checked = 0;
let cleaned = 0;
let removed = 0;

function walk(dir) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      walk(fullPath);
    } else if (entry.isFile() && fullPath.endsWith(".html")) {
      checked += 1;
      const source = fs.readFileSync(fullPath);
      if (!source.includes(0)) {
        continue;
      }
      const output = Buffer.allocUnsafe(source.length);
      let offset = 0;
      for (const byte of source) {
        if (byte === 0) {
          removed += 1;
          continue;
        }
        output[offset] = byte;
        offset += 1;
      }
      fs.writeFileSync(fullPath, output.subarray(0, offset));
      cleaned += 1;
    }
  }
}

if (!fs.existsSync(root)) {
  console.log(`strip_build_null_bytes: skipped, ${root} does not exist`);
  process.exit(0);
}

walk(root);
console.log(
  `strip_build_null_bytes: checked ${checked} HTML files, cleaned ${cleaned}, removed ${removed} NUL bytes`,
);
