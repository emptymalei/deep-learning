# Deep Learning

When I switched to data science, I built [my digital garden, datumorphism](https://datumorphism.leima.is/). I deliberately designed this digital garden as my second brain. As a result, most of the articles are fragments of knowledge and require context to understand them.

Making bricks is easy but assembling them into a house is not. So I have decided to use this repository to practice my house-building techniques.

I do not have a finished blueprint yet. But I have a framework in my mind: I want to consolidate some of my thoughts and learnings in a good way. However, I do not want to compile a reference book, as [datumorphism](https://datumorphism.leima.is/) already serves this purpose. I should create stories.


## How to Contribute

This repository contains mostly markdown files. To make sure we have the same conventions, we have added markdownlint tools to pre-commit. So please install [pre-commit](https://pre-commit.com/) then run the following command the first time you cloned the repository.

```bash
pre-commit install
```

### Build and Preview the Site

dl.leima.is is built with [Zensical](https://zensical.org), the successor to Material for MkDocs, by `scripts/build_site.sh`. The deploy workflow runs the same script. It only needs [uv](https://docs.astral.sh/uv/):

```bash
scripts/build_site.sh                           # builds build/site/
uvx zensical==0.1.1 serve -f build/mkdocs.yml   # live preview
```

Zensical reads `mkdocs.yml` but cannot run third-party MkDocs plugins yet. So the script first exports the marimo playgrounds, then runs `scripts/stage_site.py`, which converts notebooks and citations into plain Markdown under `build/` (git-ignored). `zensical serve` watches `build/docs/`, so after editing a page under `dl/`, rerun `uv run --python 3.12 scripts/stage_site.py`.

Notebook outputs are taken from your local, git-ignored `.ipynb` files when they exist. CI has none, so the published notebook pages show code only. Run `python3 scripts/fix_links.py` to check that every Markdown link resolves relative to its source file.

### Python Environment

The project uses Python 3.12 (see `.python-version`) and Poetry 2 for the notebooks. Install the requirements using

```
poetry env use 3.12
poetry install
```

The site build does not need this environment; it only uses uv.

### Developing Notebooks

We use jupytext to sync the `.py` files to `.ipynb` files. `.ipynb` files are ignore in git.
Please pair the `.py` file with the `.ipynb` using jupytext in jupyterlab first.

### Interactive Widgets

Interactive explainers run entirely in the reader's browser and need no build plugin. Put a `dl-widget` block in a page; whatever is inside it is the fallback for readers without JavaScript, for print, and for text exports.

```html
<div class="dl-widget" data-widget="backtest-windows" markdown>

![Sliding Blocks](../assets/timeseries-evaluation.forecasting/block-slide.png)

</div>
```

Available widgets live in `dl/assets/widgets/`:

| Widget | What it shows | Source of truth |
|---|---|---|
| `backtest-windows` | Sliding vs expanding backtest folds, with a gap | `backtest-windows.js` |
| `vega` | A Vega-Lite chart exported from Python with Altair (`data-spec="path/from/docs/root.vl.json"`) | `scripts/build_charts.py` |
| `embed` | A click-to-load iframe, e.g. a marimo WebAssembly playground (`data-src`, `data-height`) | `playgrounds/*.py` |

To add a widget:

1. Create `dl/assets/widgets/<name>.js` that calls `dlWidgets.register("<name>", (el, api) => { ... })`. The `api` gives theme colours (`api.colors()`, `api.onThemeChange(cb)`), `api.reducedMotion()`, `api.resolve(path)` for files relative to the docs root, and `api.loadScript(url)` for pinned CDN libraries.
2. Add the file to `extra_javascript` in `mkdocs.yml` after `assets/widgets/loader.js`.
3. Respect `prefers-reduced-motion` (no autoplay) and check the widget in light mode, dark mode and at phone width.

To regenerate Altair charts, run `uv run --python 3.12 scripts/build_charts.py`.

marimo playgrounds (`playgrounds/*.py`) are exported to WebAssembly by `scripts/build_playgrounds.sh` into the git-ignored `dl/assets/playgrounds/`. `scripts/build_site.sh` runs it, with marimo pinned there, so every build and deploy includes them. When the export is missing, the `embed` widget shows its Markdown fallback.
