import { mkdir, readdir, readFile, rm, writeFile } from "node:fs/promises";
import { spawn } from "node:child_process";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, "..");
const contentDir = path.join(root, "content", "blogs");
const outputDir = path.join(root, "blogs");
const templatePath = path.join(root, "templates", "blog-post.html");

function run(command, args) {
  return new Promise(function(resolve, reject) {
    const child = spawn(command, args, { cwd: root, stdio: "inherit" });
    child.on("error", reject);
    child.on("exit", function(code) {
      if (code === 0) resolve();
      else reject(new Error(command + " exited with code " + code));
    });
  });
}

function parseFrontMatter(source) {
  if (!source.startsWith("---\n")) return { metadata: {}, body: source };
  const end = source.indexOf("\n---", 4);
  if (end === -1) return { metadata: {}, body: source };

  const raw = source.slice(4, end).trim();
  const body = source.slice(end + 5).replace(/^\n/, "");
  const metadata = {};

  for (const line of raw.split(/\r?\n/)) {
    const match = line.match(/^([A-Za-z0-9_-]+):\s*(.*)$/);
    if (!match) continue;
    const key = match[1].trim();
    let value = match[2].trim();
    value = value.replace(/^['"]|['"]$/g, "");
    metadata[key] = value;
  }

  return { metadata, body };
}

function firstHeading(body) {
  const match = body.match(/^#\s+(.+)$/m);
  return match ? match[1].trim() : "";
}

function firstParagraph(body) {
  const cleaned = body
    .replace(/^#\s+.+$/gm, "")
    .split(/\n{2,}/)
    .map(function(part) { return part.replace(/\s+/g, " ").trim(); })
    .find(function(part) { return part && !part.startsWith("$$") && !part.startsWith("-") && !part.startsWith("!"); });

  return cleaned || "";
}

function stripFirstHeading(body) {
  return body.replace(/^#\s+.+\n?/, "").replace(/^\n+/, "");
}

function slugify(value) {
  const slug = value
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");

  return slug || "post";
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function yamlValue(value) {
  return JSON.stringify(String(value));
}

async function markdownFiles() {
  await mkdir(contentDir, { recursive: true });
  const entries = await readdir(contentDir, { withFileTypes: true });
  return entries
    .filter(function(entry) { return entry.isFile() && entry.name.endsWith(".md"); })
    .map(function(entry) { return path.join(contentDir, entry.name); })
    .sort(function(a, b) { return a.localeCompare(b); });
}

function makeIndex(posts) {
  let items;
  if (posts.length) {
    items = posts.map(function(post) {
      const meta = post.date ? "              <p class=\"meta\">" + escapeHtml(post.date) + "</p>" : "";
      return [
        "            <li>",
        "              <h3><a href=\"" + escapeHtml(post.slug) + "/\">" + escapeHtml(post.title) + "</a></h3>",
        meta,
        "            </li>"
      ].filter(Boolean).join("\n");
    }).join("\n");
  } else {
    items = [
      "            <li>",
      "              <h3>No posts yet</h3>",
      "              <p>Markdown posts added to <code>content/blogs/</code> will appear here after running <code>npm run build</code>.</p>",
      "            </li>"
    ].join("\n");
  }

  return [
    "<!doctype html>",
    "<html lang=\"en\">",
    "  <head>",
    "    <meta charset=\"utf-8\">",
    "    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">",
    "    <meta name=\"author\" content=\"Artem Frenk\">",
    "    <meta name=\"description\" content=\"Blog posts and notes by Artem Frenk.\">",
    "    <title>Blogs | Artem Frenk</title>",
    "    <link rel=\"stylesheet\" href=\"../assets/site.css\">",
    "  </head>",
    "  <body>",
    "    <div class=\"site-shell\">",
    "      <header class=\"masthead\">",
    "        <nav class=\"site-nav\" aria-label=\"Primary navigation\">",
    "          <a href=\"/\">Home</a>",
    "          <a href=\"/papers/\">Papers</a>",
    "          <a href=\"/blogs/\" aria-current=\"page\">Blogs</a>",
    "        </nav>",
    "      </header>",
    "",
    "      <main id=\"content\" class=\"page-grid page-grid-single\">",
    "        <article>",
    "          <p class=\"kicker\">Blogs</p>",
    "          <h1 class=\"page-title\">Blogs</h1>",
    "",
    "          <ul class=\"item-list\">",
    items,
    "          </ul>",
    "        </article>",
    "      </main>",
    "    </div>",
    "  </body>",
    "</html>",
    ""
  ].join("\n");
}

const files = await markdownFiles();
await mkdir(outputDir, { recursive: true });

const posts = [];
const seenSlugs = new Map();

for (const file of files) {
  const source = await readFile(file, "utf8");
  const parsed = parseFrontMatter(source);
  if (!parsed.body.trim()) continue;
  const basename = path.basename(file, ".md");
  const inferredTitle = firstHeading(parsed.body) || basename;
  const title = parsed.metadata.title || inferredTitle;
  const baseSlug = slugify(parsed.metadata.slug || title || basename);
  const count = seenSlugs.get(baseSlug) || 0;
  seenSlugs.set(baseSlug, count + 1);
  const slug = count === 0 ? baseSlug : baseSlug + "-" + (count + 1);
  const description = parsed.metadata.description || firstParagraph(parsed.body).slice(0, 180);
  const date = parsed.metadata.date || "";
  const body = parsed.metadata.title ? parsed.body : stripFirstHeading(parsed.body);
  const tempFile = path.join(os.tmpdir(), "personal-site-blog-" + process.pid + "-" + slug + ".md");
  const normalized = "---\n" +
    "title: " + yamlValue(title) + "\n" +
    (date ? "date: " + yamlValue(date) + "\n" : "") +
    (description ? "description: " + yamlValue(description) + "\n" : "") +
    "---\n\n" + body;

  await writeFile(tempFile, normalized, "utf8");

  const postDir = path.join(outputDir, slug);
  await rm(postDir, { recursive: true, force: true });
  await mkdir(postDir, { recursive: true });

  await run("pandoc", [
    tempFile,
    "--from", "markdown+yaml_metadata_block+tex_math_dollars+tex_math_single_backslash+raw_tex",
    "--to", "html5",
    "--standalone",
    "--mathjax=https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js",
    "--template", templatePath,
    "--output", path.join(postDir, "index.html")
  ]);

  await rm(tempFile, { force: true });
  posts.push({ title, date, description, slug });
}

posts.sort(function(a, b) {
  if (a.date && b.date && a.date !== b.date) return b.date.localeCompare(a.date);
  if (a.date && !b.date) return -1;
  if (!a.date && b.date) return 1;
  return a.title.localeCompare(b.title);
});

const activeSlugs = new Set(posts.map(function(post) { return post.slug; }));
for (const entry of await readdir(outputDir, { withFileTypes: true })) {
  if (entry.isDirectory() && !activeSlugs.has(entry.name)) {
    await rm(path.join(outputDir, entry.name), { recursive: true, force: true });
  }
}

await writeFile(path.join(outputDir, "index.html"), makeIndex(posts), "utf8");
console.log("Built " + posts.length + " blog post" + (posts.length === 1 ? "" : "s") + ".");
