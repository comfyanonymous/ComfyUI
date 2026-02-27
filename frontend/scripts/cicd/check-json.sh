#!/bin/bash
set -euo pipefail

usage() {
  echo "Usage: $0 [--debug]" >&2
}

debug=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --debug)
      debug=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
  shift
done

# Validate JSON syntax in tracked files using jq
if ! command -v jq >/dev/null 2>&1; then
  echo "Error: jq is required but not installed" >&2
  exit 127
fi

EXCLUDE_PATTERNS=(
  '**/tsconfig*.json'
)

if [ -n "${JSON_LINT_EXCLUDES:-}" ]; then
  # shellcheck disable=SC2206
  EXCLUDE_PATTERNS+=( ${JSON_LINT_EXCLUDES} )
fi

pathspecs=(-- '*.json')
for pattern in "${EXCLUDE_PATTERNS[@]}"; do
  if [[ ${pattern:0:1} == ':' ]]; then
    pathspecs+=("$pattern")
  else
    pathspecs+=(":(glob,exclude)${pattern}")
  fi
done

mapfile -t json_files < <(git ls-files "${pathspecs[@]}")

if [ "${#json_files[@]}" -eq 0 ]; then
  echo 'No JSON files found.'
  exit 0
fi

if [ "$debug" -eq 1 ]; then
  echo 'JSON files to validate:'
  printf '  %s\n' "${json_files[@]}"
fi

failed=0
for file in "${json_files[@]}"; do
  if ! jq -e . "$file" >/dev/null; then
    echo "Invalid JSON syntax: $file" >&2
    failed=1
  fi
done

if [ "$failed" -ne 0 ]; then
  echo 'JSON validation failed.' >&2
  exit 1
fi

echo 'All JSON files are valid.'
