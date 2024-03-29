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

Install the requirements using

```
poetry install
```

Preview the docs:

```python
poetry run mkdocs serve -s
```

### Developing Notebooks

We use jupytext to sync the `.py` files to `.ipynb` files. `.ipynb` files are ignore in git.
Please pair the `.py` file with the `.ipynb` using jupytext in jupyterlab first.

### Optional Requirements

> The pdf generation is done by the [mkdocs-with-pdf](https://github.com/orzih/mkdocs-with-pdf) plugin.

To generate PDF locally, please install [cairo, Pango and GDK-PixBuf ](https://doc.courtbouillon.org/weasyprint/latest/first_steps.html#macos).


#### Install pango on Mac

When installing pango on Mac using homebrew, the path for `DYLD_LIBRARY_PATH` are not automatically updated. So we need to add the correct path for pango, harfbuzz, and fontconfig. For example,

```
export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:/Users/itsme/homebrew/Cellar/pango/1.48.8/lib:/Users/itsme/homebrew/Cellar/harfbuzz/2.8.2/lib:/Users/itsme/homebrew/Cellar/fontconfig/2.13.1/lib
```
