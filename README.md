# Deep Learning

When I switched to data science, I built [my digital garden, datumorphism](https://datumorphism.leima.is/). I deliberately designed this digital garden as my second brain. As a result, most of the articles are fragments of knowledge and require context to understand them.

Making bricks is easy but assembling them into a house is not. So I have decided to use this repository to practice my house-building techniques.

I do not have a finished blueprint yet. But I have a framework in my mind: I want to consolidate some of my thoughts and learnings in a good way. However, I do not want to compile a reference book, as [datumorphism](https://datumorphism.leima.is/) already serves this purpose. I should create stories.


## How to Contribute

This repository contains mostly markdown files. To make sure we have the same conventions, we have added markdownlint tools to pre-commit. So please install [pre-commit](https://pre-commit.com/) then run the following command the first time you cloned the repository.

```bash
pre-commit install
```

### Preview Requires Python

The project uses Python 3.12 (see `.python-version`) and Poetry 2. Install the requirements using

```
poetry env use 3.12
poetry install
```

`poetry install` also installs the notebook dependencies (PyTorch and friends). To only build the docs, `poetry install --only main --no-root` is enough, which is what CI does.

Preview the docs:

```python
poetry run mkdocs serve -s
```

### Preview with Zensical (trial)

[Zensical](https://zensical.org) is the successor to Material for MkDocs. It reads `mkdocs.yml` but cannot run third-party MkDocs plugins yet, so `scripts/stage_site.py` first converts notebooks and citations into plain Markdown under `build/` (git-ignored):

```bash
scripts/build_playgrounds.sh                 # optional: marimo playgrounds
uv run --python 3.12 scripts/stage_site.py   # build/docs + build/mkdocs.yml
uvx zensical serve -f build/mkdocs.yml       # or: uvx zensical build -f build/mkdocs.yml
```

Notebook outputs are taken from your local, git-ignored `.ipynb` files when they exist. Run `python3 scripts/fix_links.py` to check that every Markdown link resolves relative to its source file, which both MkDocs and Zensical need.

### Developing Notebooks

We use jupytext to sync the `.py` files to `.ipynb` files. `.ipynb` files are ignore in git.
Please pair the `.py` file with the `.ipynb` using jupytext in jupyterlab first.

### Interactive Widgets

Interactive explainers run entirely in the reader's browser and need no MkDocs plugin, so they also work with Zensical. Put a `dl-widget` block in a page; whatever is inside it is the fallback for readers without JavaScript, for print, and for text exports.

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

### Optional Requirements

> The pdf generation is done by the [mkdocs-with-pdf](https://github.com/orzih/mkdocs-with-pdf) plugin.

To generate PDF locally, please install [cairo, Pango and GDK-PixBuf ](https://doc.courtbouillon.org/weasyprint/latest/first_steps.html#macos).


#### Install pango on Mac

When installing pango on Mac using homebrew, the path for `DYLD_LIBRARY_PATH` are not automatically updated. So we need to add the correct path for pango, harfbuzz, and fontconfig. For example,

```
export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:/Users/itsme/homebrew/Cellar/pango/1.48.8/lib:/Users/itsme/homebrew/Cellar/harfbuzz/2.8.2/lib:/Users/itsme/homebrew/Cellar/fontconfig/2.13.1/lib
```
