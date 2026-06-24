import { createReadStream } from "node:fs";
import { stat } from "node:fs/promises";
import { createServer } from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, "..");
const host = process.env.HOST || "0.0.0.0";
const port = Number(process.env.PORT || 8765);

const types = new Map([
  [".css", "text/css; charset=utf-8"],
  [".html", "text/html; charset=utf-8"],
  [".js", "text/javascript; charset=utf-8"],
  [".json", "application/json; charset=utf-8"],
  [".md", "text/markdown; charset=utf-8"],
  [".pdf", "application/pdf"],
  [".png", "image/png"],
  [".jpg", "image/jpeg"],
  [".jpeg", "image/jpeg"],
  [".svg", "image/svg+xml; charset=utf-8"],
  [".txt", "text/plain; charset=utf-8"],
]);

function send(res, status, body, contentType = "text/plain; charset=utf-8") {
  res.writeHead(status, { "content-type": contentType });
  res.end(body);
}

async function resolveFile(urlPath) {
  const decoded = decodeURIComponent(urlPath.split("?")[0]);
  const clean = decoded.replace(/^\/+/, "");
  let candidate = path.resolve(root, clean || "index.html");

  if (!candidate.startsWith(root + path.sep) && candidate !== root) {
    return null;
  }

  let info;
  try {
    info = await stat(candidate);
  } catch {
    return null;
  }

  if (info.isDirectory()) {
    candidate = path.join(candidate, "index.html");
    try {
      info = await stat(candidate);
    } catch {
      return null;
    }
  }

  return info.isFile() ? candidate : null;
}

const server = createServer(async (req, res) => {
  if (!req.url || !["GET", "HEAD"].includes(req.method || "")) {
    send(res, 405, "Method not allowed");
    return;
  }

  const file = await resolveFile(req.url);
  if (!file) {
    send(res, 404, "Not found");
    return;
  }

  const contentType = types.get(path.extname(file).toLowerCase()) || "application/octet-stream";
  res.writeHead(200, { "content-type": contentType });

  if (req.method === "HEAD") {
    res.end();
    return;
  }

  createReadStream(file).pipe(res);
});

server.listen(port, host, () => {
  const visibleHost = host === "0.0.0.0" ? "127.0.0.1" : host;
  console.log(`Serving ${root} at http://${visibleHost}:${port}/`);
});
