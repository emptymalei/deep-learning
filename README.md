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

Zensical reads `mkdocs.yml` but cannot run third-party MkDocs plugins yet. So the script first exports the marimo playgrounds, then runs `scripts/stage_site.py`, which converts notebooks and citations into plain Markdown under `build/` (git-ignored). `zensical serve` watches `build/docs/`, so after editing a page under `dl/`, rerun `uv run --python 3.14 scripts/stage_site.py`.

Notebook outputs come from your local, git-ignored `.ipynb` files when they exist, otherwise from the outputs saved in `notebook-outputs/` (see below). Neither the build nor CI executes notebooks. Run `python3 scripts/fix_links.py` to check that every Markdown link resolves relative to its source file.

### Python Environment

The project uses Python 3.14 (see `.python-version`) and Poetry 2 for the notebooks. If you have no Python 3.14 yet, `uv python install 3.14` or `pyenv install 3.14` provides one. Install the requirements using

```
poetry env use 3.14
poetry install
```

The site build does not need this environment; it only uses uv.

### Developing Notebooks

We use jupytext to sync the `.py` files to `.ipynb` files. `.ipynb` files are ignore in git.
Please pair the `.py` file with the `.ipynb` using jupytext in jupyterlab first.

To publish a notebook's outputs, run it locally, then save the outputs of your local runs and commit `notebook-outputs/`:

```bash
uv run --python 3.14 scripts/stage_site.py --save-outputs
```

This writes a cleaned, gzipped copy of each run notebook to `notebook-outputs/<name>.ipynb.gz`: progress-bar widgets and error tracebacks are dropped, long logs are trimmed and images wider than 1600 px are scaled down. The site build matches outputs to cells by their code, so a cell you changed after the run shows no output until you run and save again.

### Pull Request Previews

Every pull request from a branch of this repository is built and published as a preview:

```text
https://dl.leima.is/pr-preview/pr-<number>/
```

The "Publish Docs Test" workflow comments on the pull request with the link and updates the preview on every push. Each page of a preview carries a `noindex` tag and a badge in the bottom left corner with the pull request and the commit it was built from. After a push it can take a minute or two before the page updates.

How it works:

- Previews live in `pr-preview/pr-<number>/` on the `gh-pages` branch, next to the published site (`scripts/pr_preview.sh`). The site uses relative links, so it runs from a subfolder as built.
- The deploy from `main` replaces the branch but carries the previews of open pull requests over (`scripts/deploy_site.sh`).
- The "PR Preview Cleanup" workflow removes a preview when its pull request is merged or closed.
- Pull requests from forks are built but not previewed, because their token cannot write to the repository.

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
| `metrics-playground` | Drag forecast points and watch MAE, MSE, RMSE, MAPE and sMAPE react | `metrics-playground.js` |
| `crps-explorer` | A normal forecast's CDF against the observed value, with the exact CRPS | `crps-explorer.js` |
| `window-explorer` | Time delay embedding: slide input and target windows along a series and see the rows it produces. `data-gap` adds a gap slider and `data-labels="dataset"` shows the same thing as `DataFrameDataset` data points | `window-explorer.js` |
| `positional-encoding` | The transformer's sinusoidal positional encodings as a heatmap, with the similarity between positions | `positional-encoding.js` |

To add a widget:

1. Create `dl/assets/widgets/<name>.js` that calls `dlWidgets.register("<name>", (el, api) => { ... })`. The `api` gives theme colours (`api.colors()`, `api.onThemeChange(cb)`), `api.reducedMotion()`, `api.resolve(path)` for files relative to the docs root, and `api.loadScript(url)` for pinned CDN libraries.
2. Add the file to `extra_javascript` in `mkdocs.yml` after `assets/widgets/loader.js`.
3. Respect `prefers-reduced-motion` (no autoplay) and check the widget in light mode, dark mode and at phone width.

To regenerate Altair charts, run `uv run --python 3.14 scripts/build_charts.py`.

marimo playgrounds (`playgrounds/*.py`) are exported to WebAssembly by `scripts/build_playgrounds.sh` into the git-ignored `dl/assets/playgrounds/`. `scripts/build_site.sh` runs it, with marimo pinned there, so every build and deploy includes them. When the export is missing, the `embed` widget shows its Markdown fallback.

### Reading with AI Assistants

The build publishes [`llms.txt`](https://dl.leima.is/llms.txt), [`llms-full.txt`](https://dl.leima.is/llms-full.txt) and a Markdown copy of every page (`<page URL>index.md`), configured by the `llmstxt` plugin in `mkdocs.yml`. Each page also has a copy-as-Markdown icon (Zensical's `content.action.copy`) and an icon with a small menu to ask Claude, ChatGPT or Gemini about the page (`dl/assets/js/page-actions.js`); nothing is sent until the reader picks one.

Conventions for AI coding agents (and people) are in [AGENTS.md](AGENTS.md), including the front matter fields. A design for an MCP server is in [design/mcp-server.md](design/mcp-server.md).
