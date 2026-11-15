# Agent Guide

How to work in this repository, for AI coding agents and people alike.

## What This Is

The source of the book [Time Series with Deep Learning Quick Bite](https://dl.leima.is), built with [Zensical](https://zensical.org) and published to GitHub Pages from `main`.

- `dl/`: the book. Markdown pages, with the navigation in `mkdocs.yml` (`nav`).
- `dl/notebooks/*.py`: notebooks as [jupytext](https://jupytext.readthedocs.io) Python files. The paired `.ipynb` files are git-ignored.
- `notebook-outputs/*.ipynb.gz`: outputs saved from local notebook runs (`scripts/stage_site.py --save-outputs`), shown on the notebook pages. The build never executes notebooks.
- `dl/notebooks/ts_dl_utils/`: small helpers the notebooks import (data modules, evaluator, naive forecasters).
- `dl/assets/references/*.bib`: the bibliography for `[@key]` citations.
- `dl/assets/widgets/`: interactive widgets (plain JavaScript, no build plugin).
- `playgrounds/*.py`: marimo notebooks, exported to WebAssembly into `dl/assets/playgrounds/` at build time.
- `scripts/`: the site build and its helpers.

## Build and Check

The site build only needs [uv](https://docs.astral.sh/uv/):

```bash
scripts/build_site.sh                           # build/site/
uvx zensical==0.1.1 serve -f build/mkdocs.yml   # live preview of build/docs/
```

`scripts/stage_site.py` stages `dl/` into `build/docs/` first: it converts the notebooks to Markdown, turns citations into footnotes and writes the generated `build/mkdocs.yml`. Never edit anything under `build/`.

Checks to run before a pull request, all lightweight:

```bash
pre-commit run --all-files
scripts/build_site.sh            # must report "No issues found"
python3 scripts/fix_links.py     # every Markdown link resolves from its source file
```

Do not execute the notebooks to verify a change, locally or in CI. They train models; some run for hours (`diffusion_model`, `time_vae`, `time_vae_poison`), download large datasets (`hierarchical_forecasting_mint`, `transformer-ts-nixtla-testing-m5`) or need more than 5 GB of memory. To check a dependency change, import the libraries, `py_compile` the notebooks and probe the changed APIs on toy data.

The notebook environment is managed with Poetry (`poetry install`); the Python version is in `.python-version`.

## Writing Conventions

- Material-flavoured Markdown (admonitions, tabs, `pymdownx` extensions) that builds with Zensical. Interactive pieces must not depend on MkDocs plugins.
- Write links relative to the source file, e.g. `../concepts/kl-divergence.md`, never relative to the rendered URL. Link to notebooks by their `.py` source; staging points those links at the rendered page.
- Cite with `[@key]` (several: `[@a; @b]`), where the key is in `dl/assets/references/*.bib`.
- New pages go into `nav` in `mkdocs.yml`; pages outside the nav still build but are hard to find.
- Widgets are `<div class="dl-widget" data-widget="name" markdown>…</div>` blocks; the content inside is the fallback for readers without JavaScript. See the README for how to add one.
- No tracking: no analytics, no cookie banners, no third-party scripts that track readers.

## Front Matter

Pages may start with YAML front matter. The fields:

| Field | Type | Meaning |
|---|---|---|
| `tags` | list of strings | Tags for the tags index (`dl/tags.md`). |
| `summary` | string | One or two sentences on what the page covers. Becomes the page's description in `llms.txt`. |
| `prerequisites` | list of paths | Pages to read first, relative to `dl/` (e.g. `time-series/timeseries-basics.ar.md`). |
| `related` | list of paths | Pages on neighbouring topics, relative to `dl/`. |
| `notebook` | path | The notebook behind the page, relative to `dl/` (e.g. `notebooks/rnn_timeseries.py`). |

`scripts/stage_site.py` renders them: `summary` becomes an "In one minute" box under the title, `prerequisites` become "Read first" links in that box, and `related` become "Related" links at the end of the page. `summary` is also the page's description in `llms.txt`.

### Page Shape

New and reworked pages follow one shape: the "In one minute" box (from front matter), intuition with a figure or diagram, the math (long derivations folded with `??? note "Show the math"`), code, a "Try it" widget where one helps, then pitfalls. See the Evaluation and Metrics chapter for the reference.

### Glossary

`includes/glossary.md` holds abbreviation definitions (`*[MAE]: Mean absolute error…`). It is appended to every page, so each term shows a tooltip wherever it appears as a word. Add a term there once instead of defining it on every page.

A tap or click on a term opens its definition in a panel that stays until it is closed (`dl/assets/js/glossary-panel.js`): a sheet at the bottom of the screen on phones, a card on the right on wider screens. Keep definitions to a sentence or two of plain text; they are read from the `title` attribute.

The right sidebar of every page lists, under the table of contents, the glossary terms that occur on the page and the page's links to and from other pages of the book. Both are generated from the built pages by `scripts/stage_site.py --post-build` (`add_sidebar_sections`); there is nothing to maintain by hand.

## For AI Assistants Reading the Book

- `https://dl.leima.is/llms.txt` lists every page, by nav section, with a link to its Markdown version.
- `https://dl.leima.is/llms-full.txt` is the whole book in one Markdown file.
- Every page has a Markdown copy at `<page URL>index.md`, e.g. `https://dl.leima.is/time-series/timeseries-basics.ar/index.md`.

These are generated by the `llmstxt` plugin settings in `mkdocs.yml`; `scripts/stage_site.py` derives the sections from the nav.

## Changes

Work on a branch and open a pull request against `main`; merging to `main` deploys the site. Keep pull requests focused and say in the description how the change was checked.

Every pull request from a branch of this repository gets a site preview at `https://dl.leima.is/pr-preview/pr-<number>/`, linked in a comment on the pull request and removed when it closes. See "Pull Request Previews" in the README.
