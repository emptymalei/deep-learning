#!/usr/bin/env bash
# Export every marimo notebook in playgrounds/ to a static WebAssembly site
# under dl/assets/playgrounds/<name>/, which the site build copies as-is.
#
# The output is generated (and gitignored); scripts/build_site.sh runs this
# before building the site. To export the playgrounds alone:
#
#     scripts/build_playgrounds.sh
#
# Requires uv (https://docs.astral.sh/uv/). Set MARIMO_VERSION to pin marimo,
# e.g. MARIMO_VERSION=0.25.2 scripts/build_playgrounds.sh
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
src_dir="${repo_root}/playgrounds"
out_root="${repo_root}/dl/assets/playgrounds"
marimo="marimo${MARIMO_VERSION:+==${MARIMO_VERSION}}"

shopt -s nullglob
notebooks=("${src_dir}"/*.py)
if [ ${#notebooks[@]} -eq 0 ]; then
    echo "No notebooks found in ${src_dir}" >&2
    exit 1
fi

for nb in "${notebooks[@]}"; do
    name="$(basename "${nb}" .py)"
    out="${out_root}/${name}"
    echo "==> ${nb#"${repo_root}"/} -> ${out#"${repo_root}"/}"
    rm -rf "${out}"
    # --execute runs the notebook once at build time and embeds its outputs,
    # so readers see the plots while Pyodide is still loading.
    uvx --from "${marimo}" marimo export html-wasm "${nb}" \
        -o "${out}" --mode run --no-show-code --execute
    # The site build would turn .md/.py/.ipynb files into pages.
    # The export only has marimo's AI-assistant prompt (CLAUDE.md), which the
    # page does not load.
    find "${out}" -type f \( -name '*.md' -o -name '*.py' -o -name '*.ipynb' \) \
        -delete
done

du -sh "${out_root}"/*
