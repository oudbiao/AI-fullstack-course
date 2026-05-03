#!/usr/bin/env node

const fs = require("fs");
const http = require("http");
const path = require("path");

function readOption(name, fallback) {
  const prefix = `--${name}=`;
  const inlineValue = process.argv.find((arg) => arg.startsWith(prefix));
  if (inlineValue) {
    return inlineValue.slice(prefix.length);
  }

  const optionIndex = process.argv.indexOf(`--${name}`);
  if (optionIndex !== -1 && process.argv[optionIndex + 1]) {
    return process.argv[optionIndex + 1];
  }

  return fallback;
}

const root = path.resolve(readOption("root", process.env.STATIC_ROOT || "build"));
const host = readOption("host", process.env.HOST || "127.0.0.1");
const port = Number(readOption("port", process.env.PORT || 3000));

const contentTypes = {
  ".html": "text/html; charset=utf-8",
  ".js": "application/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".xml": "application/xml; charset=utf-8",
  ".txt": "text/plain; charset=utf-8",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".gif": "image/gif",
  ".svg": "image/svg+xml",
  ".webp": "image/webp",
  ".ico": "image/x-icon",
  ".woff": "font/woff",
  ".woff2": "font/woff2",
};

function safePath(urlPath) {
  let decoded;
  try {
    decoded = decodeURIComponent(urlPath.split("?")[0]);
  } catch {
    return null;
  }
  const cleanPath = decoded.endsWith("/") ? `${decoded}index.html` : decoded;
  const filePath = path.join(root, cleanPath);
  return filePath.startsWith(root) ? filePath : null;
}

function resolvePrettyUrl(filePath) {
  if (!filePath.startsWith(root)) {
    return null;
  }

  if (fs.existsSync(filePath) && fs.statSync(filePath).isFile()) {
    return filePath;
  }

  if (fs.existsSync(filePath) && fs.statSync(filePath).isDirectory()) {
    const indexFile = path.join(filePath, "index.html");
    return fs.existsSync(indexFile) ? indexFile : null;
  }

  const indexFile = path.join(filePath, "index.html");
  if (fs.existsSync(indexFile)) {
    return indexFile;
  }

  const htmlFile = `${filePath}.html`;
  if (fs.existsSync(htmlFile)) {
    return htmlFile;
  }

  return null;
}

function sendFile(req, res, filePath) {
  const extension = path.extname(filePath);
  res.writeHead(200, {
    "content-type": contentTypes[extension] || "application/octet-stream",
  });
  if (req.method === "HEAD") {
    res.end();
    return;
  }

  const stream = fs.createReadStream(filePath);
  stream.on("error", () => {
    if (!res.headersSent) {
      res.writeHead(500);
    }
    res.end("Internal server error");
  });
  stream.pipe(res);
}

if (!fs.existsSync(root)) {
  console.error(`Build folder not found: ${root}`);
  console.error("Run `npm run build` before starting the static server.");
  process.exit(1);
}

const server = http.createServer((req, res) => {
  const filePath = safePath(req.url || "/");
  if (!filePath) {
    res.writeHead(403);
    res.end("Forbidden");
    return;
  }

  const resolvedPath = resolvePrettyUrl(filePath);
  if (resolvedPath) {
    sendFile(req, res, resolvedPath);
    return;
  }

  const fallback = path.join(root, "404.html");
  if (fs.existsSync(fallback)) {
    res.writeHead(404, { "content-type": contentTypes[".html"] });
    fs.createReadStream(fallback).pipe(res);
    return;
  }

  res.writeHead(404);
  res.end("Not found");
});

server.listen(port, host, () => {
  console.log(`Serving ${root} at http://${host}:${port}`);
});
