# Personal-Site-Real

Static personal site with generated Markdown/TeX blog posts.

## Build blogs

```bash
npm run build
```

This converts `content/blogs/*.md` into routed static pages under `blogs/` using Pandoc and MathJax.

## Local preview

```bash
npm run dev
```

By default this serves the repo at `http://127.0.0.1:8765/`. Override with `PORT=3000 npm run dev`.

## Production

Run `npm run build`, then serve the repository root as static files. No Python runtime is required. If your host expects a Node start command, use `npm start` and set `PORT` if required.
