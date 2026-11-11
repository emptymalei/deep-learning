#!/usr/bin/env bash
# Build dl.leima.is with Zensical into build/site/. The deploy workflow runs
# exactly this, so a local run reproduces the published site:
#
#     scripts/build_site.sh
#     uvx zensical==0.1.1 serve -f build/mkdocs.yml   # optional: live preview
#
# Steps:
#   1. scripts/build_playgrounds.sh: marimo playgrounds -> dl/assets/playgrounds/
#   2. scripts/stage_site.py: dl/ -> build/docs/ + build/mkdocs.yml, doing the
#      work of the MkDocs plugins Zensical cannot run (notebooks, citations)
#   3. zensical build -> build/site/
#
# Requires uv (https://docs.astral.sh/uv/). Tool versions are pinned below and
# can be overridden through the environment.
set -euo pipefail

: "${ZENSICAL_VERSION:=0.1.1}"
: "${MARIMO_VERSION:=0.25.2}"
export MARIMO_VERSION

cd "$(dirname "${BASH_SOURCE[0]}")/.."

scripts/build_playgrounds.sh
uv run --python 3.12 scripts/stage_site.py
uvx --from "zensical==${ZENSICAL_VERSION}" zensical build -f build/mkdocs.yml
