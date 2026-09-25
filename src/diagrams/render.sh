#!/usr/bin/env bash
# Rebuild the README diagram; TeX intermediates remain temporary.
set -euo pipefail
source_dir="$(cd "$(dirname "$0")" && pwd)"
repo_dir="$(cd "$source_dir/../.." && pwd)"
output_dir="$repo_dir/datasets/figures/architecture"
build_dir="$(mktemp -d)"
trap 'rm -rf "$build_dir"' EXIT
cp "$source_dir/modpinn.tex" "$build_dir/modpinn.tex"
mkdir -p "$output_dir"
for theme in light dark; do
  if [[ "$theme" == dark ]]; then
    printf '\\def\\darkmode{1}\n' > "$build_dir/$theme.tex"
  else
    : > "$build_dir/$theme.tex"
  fi
  printf '\\input{modpinn.tex}\n' >> "$build_dir/$theme.tex"
  tectonic --keep-logs --outdir "$build_dir" "$build_dir/$theme.tex"
  pdftocairo -svg "$build_dir/$theme.pdf" "$output_dir/modpinn-$theme.svg"
done
