#!/usr/bin/env bash
set -euo pipefail

case "${1:-pages}" in
  pages|blogs|reading-list|release)
    if ! command -v pandoc >/dev/null 2>&1; then
      if ! command -v uv >/dev/null 2>&1; then
        printf 'error: pandoc is required for %s. Add it to PATH or install uv.\n' "${1:-pages}" >&2
        exit 127
      fi
      if ! pandoc_path=$(uv run --locked python -c 'import pypandoc; print(pypandoc.get_pandoc_path())'); then
        printf 'error: uv could not provide pandoc for %s. Check the uv error above.\n' "${1:-pages}" >&2
        exit 127
      fi
      if [[ ! -f "$pandoc_path" || ! -x "$pandoc_path" ]]; then
        printf 'error: uv returned a non-executable pandoc path: %s\n' "$pandoc_path" >&2
        exit 127
      fi
      export PATH="$(dirname "$pandoc_path"):$PATH"
    fi
    ;;
esac

case "${1:-pages}" in
  pages) exec zig build pages ;;
  blogs) exec zig build blogs ;;
  reading-list) exec zig build reading-list ;;
  test) exec zig build test ;;
  serve) shift; exec zig build run -- "$@" ;;
  release) zig build pages && exec zig build -Doptimize=ReleaseSafe ;;
  *) printf 'usage: %s {pages|blogs|reading-list|test|serve|release}\n' "$0" >&2; exit 2 ;;
esac
