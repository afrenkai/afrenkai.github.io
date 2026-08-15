#!/usr/bin/env bash
set -euo pipefail

case "${1:-blogs}" in
  blogs) exec zig build blogs ;;
  test) exec zig build test ;;
  serve) shift; exec zig build run -- "$@" ;;
  release) zig build blogs && exec zig build -Doptimize=ReleaseSafe ;;
  *) printf 'usage: %s {blogs|test|serve|release}\n' "$0" >&2; exit 2 ;;
esac
