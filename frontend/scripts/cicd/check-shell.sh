#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(git rev-parse --show-toplevel)"
cd "$ROOT_DIR"

if ! command -v shellcheck >/dev/null 2>&1; then
  echo "Error: shellcheck is required but not installed" >&2
  exit 127
fi

mapfile -t shell_files < <(git ls-files -- '*.sh')

if [[ ${#shell_files[@]} -eq 0 ]]; then
  echo 'No shell scripts found.'
  exit 0
fi

shellcheck --format=gcc "${shell_files[@]}"
