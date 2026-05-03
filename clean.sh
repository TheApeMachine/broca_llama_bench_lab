#!/usr/bin/env zsh

set -euo pipefail

# Change this to your root directory (or pass as argument)
ROOT_DIR="${1:-.}"

# Find all regular files recursively (skip Python virtualenv trees)
find "$ROOT_DIR" -type f -not -path '*/.venv/*' | while read -r file; do
  # Process file with awk
  awk '
    {
      lines[NR] = $0
    }
    END {
      start = 1
      end = NR

      # Remove first line if blank
      if (NR > 0 && lines[1] ~ /^[[:space:]]*$/) {
        start = 2
      }

      # Remove up to last 3 lines if they are blank
      removed = 0
      while (end >= start && removed < 3 && lines[end] ~ /^[[:space:]]*$/) {
        end--
        removed++
      }

      # Print remaining lines
      for (i = start; i <= end; i++) {
        print lines[i]
      }
    }
  ' "$file" > "$file.tmp" && mv "$file.tmp" "$file"
done