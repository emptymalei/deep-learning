# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "jupytext>=1.16",
#     "nbformat>=5.10",
#     "pybtex>=0.24",
#     "pyyaml>=6.0",
# ]
# ///
"""Stage the book for Zensical.

Zensical reads mkdocs.yml but cannot run third-party MkDocs plugins yet. This
script does their work ahead of the build, so the site no longer depends on
them:

* mkdocs-jupyter: jupytext ``.py`` notebooks become Markdown pages. When a
  paired ``.ipynb`` with outputs exists locally (it is git-ignored), its
  outputs are merged in and images are written next to the page.
* mkdocs-bibtex: ``[@key]`` / ``[@a; @b]`` citations become footnotes whose
  text comes from ``dl/assets/references/*.bib``.
* git-authors and print-site are dropped.

Usage, from the repository root:

    uv run --python 3.12 scripts/stage_site.py          # stage into build/
    uvx zensical build -f build/mkdocs.yml              # build into build/site
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import shutil
import subprocess
from pathlib import Path

import jupytext
import nbformat
import yaml
from pybtex.database import parse_file

ROOT = Path(__file__).resolve().parents[1]
SRC_CONFIG = ROOT / "mkdocs.yml"

# Plugins Zensical cannot run (handled here or dropped) and plugins that are
# built in (kept as configured).
DROP_PLUGINS = {
    "mkdocs-jupyter",
    "bibtex",
    "git-authors",
    "print-site",
    "exclude-search",
    "autorefs",
}

# Git-ignored folders that are generated on purpose and must be published.
GENERATED = ("assets/playgrounds",)

CITATION = re.compile(r"\[(@[A-Za-z0-9_:.\-]+(?:\s*;\s*@[A-Za-z0-9_:.\-]+)*)\]")


# --------------------------------------------------------------------------
# YAML that round-trips MkDocs' !!python/name tags.
# --------------------------------------------------------------------------
class Tagged(str):
    tag = ""


class Loader(yaml.SafeLoader):
    pass


class Dumper(yaml.SafeDumper):
    pass


def _tagged(loader, suffix, node):
    t = Tagged(loader.construct_scalar(node))
    t.tag = "tag:yaml.org,2002:python/name:" + suffix
    return t


Loader.add_multi_constructor("tag:yaml.org,2002:python/name:", _tagged)
Dumper.add_representer(Tagged, lambda d, v: d.represent_scalar(v.tag, str(v)))


# --------------------------------------------------------------------------
# Citations
# --------------------------------------------------------------------------
def load_bibliography(bib_dir: Path) -> dict:
    entries = {}
    for bib in sorted(bib_dir.glob("*.bib")):
        try:
            entries.update(parse_file(str(bib), bib_format="bibtex").entries)
        except Exception as err:  # keep going; report the broken file
            print(f"  ! could not parse {bib.name}: {err}")
    return entries


def _clean(text: str) -> str:
    return re.sub(r"[{}]", "", text).replace("\\&", "&").replace("--", "–").strip()


def format_entry(entry) -> str:
    people = entry.persons.get("author") or entry.persons.get("editor") or []
    names = []
    for p in people[:3]:
        last = _clean(" ".join(p.last_names))
        initials = "".join(n[0] + "." for n in p.first_names + p.middle_names if n)
        names.append(f"{last} {initials}".strip())
    authors = ", ".join(names) + (" et al." if len(people) > 3 else "")
    f = entry.fields
    title = _clean(f.get("title", ""))
    venue = _clean(
        f.get("journal")
        or f.get("booktitle")
        or f.get("publisher")
        or f.get("howpublished", "")
    )
    year = f.get("year", "")
    link = f.get("url") or (f"https://doi.org/{f['doi']}" if f.get("doi") else "")
    if (
        not link
        and f.get("eprint")
        and "arxiv" in f.get("archiveprefix", f.get("eprinttype", "")).lower()
    ):
        link = f"https://arxiv.org/abs/{f['eprint']}"
    parts = [p for p in (authors, title, f"*{venue}*" if venue else "", year) if p]
    text = ". ".join(parts) + "."
    return f"{text} [{link}]({link})" if link else text


def resolve_citations(
    markdown: str, bibliography: dict, page: str, missing: set
) -> str:
    order: list[str] = []

    def replace(m: re.Match) -> str:
        refs = []
        for key in (k.strip().lstrip("@") for k in m.group(1).split(";")):
            if key not in bibliography:
                missing.add((page, key))
                refs.append(f"[{key}?]")
                continue
            if key not in order:
                order.append(key)
            refs.append(f"[^bib-{key}]")
        return "".join(refs)

    # Leave fenced code blocks alone.
    chunks = re.split(r"(^```.*?^```)", markdown, flags=re.S | re.M)
    for i in range(0, len(chunks), 2):
        chunks[i] = CITATION.sub(replace, chunks[i])
    out = "".join(chunks)
    if order:
        out = (
            out.rstrip()
            + "\n\n"
            + "\n".join(f"[^bib-{k}]: {format_entry(bibliography[k])}" for k in order)
            + "\n"
        )
    return out


# --------------------------------------------------------------------------
# Notebooks
# --------------------------------------------------------------------------
def notebook_to_markdown(py_path: Path, out_md: Path) -> int:
    nb = jupytext.read(py_path)
    ipynb = py_path.with_suffix(".ipynb")
    if ipynb.exists():
        try:
            from jupytext.combine import combine_inputs_with_outputs

            nb = combine_inputs_with_outputs(nb, nbformat.read(ipynb, as_version=4))
        except Exception as err:
            print(f"  ! could not merge outputs from {ipynb.name}: {err}")

    asset_dir = out_md.parent / "assets" / out_md.stem
    lines: list[str] = []
    n_out = 0
    for idx, cell in enumerate(nb.cells):
        if cell.cell_type == "markdown":
            lines.append(cell.source.strip() + "\n")
        elif cell.cell_type == "code" and cell.source.strip():
            lines.append("```python\n" + cell.source.rstrip() + "\n```\n")
            outputs = cell.get("outputs", [])
            n_out += len(outputs)
            # Merge all stream text of a cell (training logs, progress bars) into one block.
            stream = "".join(
                "".join(o.get("text", ""))
                for o in outputs
                if o.get("output_type") == "stream"
            )
            if stream.strip():
                lines.append(render_stream(stream))
            for j, output in enumerate(outputs):
                if output.get("output_type") != "stream":
                    lines.append(
                        render_output(output, asset_dir, f"cell{idx}-{j}", out_md.stem)
                    )
    out_md.write_text("\n".join(l for l in lines if l))
    return n_out


def render_stream(text: str, max_lines: int = 12, keep_lines: int = 60) -> str:
    # Progress bars rewrite the same line with \r; keep only the final state.
    rows = [r.split("\r")[-1] for r in text.rstrip().split("\n")]
    rows = [r for r in rows if r.strip()]
    if len(rows) > keep_lines:
        rows = (
            rows[:5]
            + [f"... ({len(rows) - keep_lines + 5} lines omitted) ..."]
            + rows[-(keep_lines - 5) :]
        )
    block = "```text\n" + "\n".join(rows) + "\n```"
    if len(rows) <= max_lines:
        return block + "\n"
    indented = "\n".join("    " + r if r else r for r in block.split("\n"))
    return f'??? note "Output ({len(rows)} lines)"\n\n{indented}\n'


def render_output(output, asset_dir: Path, name: str, stem: str) -> str:
    otype = output.get("output_type")
    if otype in ("execute_result", "display_data"):
        data = output.get("data", {})
        if "application/vnd.jupyter.widget-view+json" in data:
            return (
                ""  # live ipywidgets (e.g. tqdm progress bars) have no static meaning
            )
        if "application/vnd.plotly.v1+json" in data:
            asset_dir.mkdir(parents=True, exist_ok=True)
            path = asset_dir / f"{name}.plotly.json"
            path.write_text(json.dumps(data["application/vnd.plotly.v1+json"]))
            spec = f"notebooks/assets/{stem}/{path.name}"
            return (
                f'<div class="dl-widget" data-widget="plotly" data-spec="{spec}" markdown>\n\n'
                "*Interactive Plotly figure: enable JavaScript to view it.*\n\n</div>\n"
            )
        for mime, ext in (
            ("image/png", "png"),
            ("image/jpeg", "jpg"),
            ("image/svg+xml", "svg"),
        ):
            if mime in data:
                asset_dir.mkdir(parents=True, exist_ok=True)
                payload = data[mime]
                path = asset_dir / f"{name}.{ext}"
                if ext == "svg":
                    path.write_text("".join(payload))
                else:
                    path.write_bytes(base64.b64decode("".join(payload)))
                return f"![output](assets/{stem}/{path.name})\n"
        if "text/markdown" in data:
            return "".join(data["text/markdown"]) + "\n"
        if "text/html" in data and "<table" in "".join(data["text/html"]):
            return (
                '<div class="nb-output" markdown="0">\n'
                + "".join(data["text/html"])
                + "\n</div>\n"
            )
        if "text/plain" in data:
            return "```text\n" + "".join(data["text/plain"]).rstrip() + "\n```\n"
    if otype == "error":
        return (
            "```text\n"
            + output.get("ename", "Error")
            + ": "
            + output.get("evalue", "")
            + "\n```\n"
        )
    return ""


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
def rewrite_nav(node):
    if isinstance(node, str):
        return (
            node[:-3] + ".md"
            if node.startswith("notebooks/") and node.endswith(".py")
            else node
        )
    if isinstance(node, list):
        return [rewrite_nav(n) for n in node]
    if isinstance(node, dict):
        return {k: rewrite_nav(v) for k, v in node.items()}
    return node


def stage_config(cfg: dict, out_dir: Path) -> dict:
    cfg = dict(cfg)
    cfg["docs_dir"] = "docs"
    cfg["site_dir"] = "site"
    cfg["site_url"] = "https://dl.leima.is/"
    cfg["repo_name"] = "emptymalei/deep-learning"
    cfg["edit_uri"] = "edit/main/dl/"
    theme = dict(cfg["theme"])
    if "custom_dir" in theme:
        theme["custom_dir"] = str(
            (ROOT / theme["custom_dir"]).relative_to(out_dir, walk_up=True)
        )
    for key in ("include_search_page", "search_index_only"):
        theme.pop(key, None)
    cfg["theme"] = theme
    plugins = []
    for p in cfg.get("plugins", []):
        name = p if isinstance(p, str) else next(iter(p))
        if name not in DROP_PLUGINS:
            plugins.append(p)
    cfg["plugins"] = plugins
    cfg["nav"] = rewrite_nav(cfg["nav"])
    return cfg


def copy_docs(src_docs: Path, docs: Path) -> int:
    """Copy what git would publish: tracked files and new, non-ignored files.

    Local, git-ignored data (datasets, checkpoints, paired .ipynb files) stays
    out, so a local build matches CI. Folders in GENERATED are added on top.
    """
    listed = subprocess.run(
        [
            "git",
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
            "--",
            str(src_docs),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split("\0")
    files = [ROOT / f for f in listed if f]
    for rel in GENERATED:
        gen = src_docs / rel
        if gen.exists():
            files += [p for p in gen.rglob("*") if p.is_file()]
    n = 0
    for f in files:
        if not f.is_file():  # deleted in the working tree
            continue
        dest = docs / f.relative_to(src_docs)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dest)
        n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="build", help="output folder (default: build)")
    args = ap.parse_args()

    out = (ROOT / args.out).resolve()
    cfg = yaml.load(SRC_CONFIG.read_text(), Loader=Loader)
    src_docs = ROOT / cfg.get("docs_dir", "docs")
    docs = out / "docs"

    if docs.exists():
        shutil.rmtree(docs)
    copied = copy_docs(src_docs, docs)
    print(
        f"copied {copied} files from {src_docs.relative_to(ROOT)}/ -> {docs.relative_to(ROOT)}/"
    )

    # Notebooks -> Markdown
    notebooks = sorted((src_docs / "notebooks").glob("*.py"))
    total_outputs = 0
    for py in notebooks:
        md = docs / "notebooks" / (py.stem + ".md")
        total_outputs += notebook_to_markdown(py, md)
        (docs / "notebooks" / py.name).unlink(missing_ok=True)
    print(
        f"notebooks: {len(notebooks)} converted, {total_outputs} outputs merged from local .ipynb files"
    )

    # Tags index: the old Material marker -> the listing marker Zensical understands.
    for md in docs.rglob("*.md"):
        text = md.read_text()
        if "[TAGS]" in text:
            md.write_text(text.replace("[TAGS]", "<!-- material/tags -->"))

    # Citations -> footnotes
    bib_dir = ROOT / "dl" / "assets" / "references"
    bibliography = load_bibliography(bib_dir)
    missing: set = set()
    cited_pages = 0
    for md in docs.rglob("*.md"):
        text = md.read_text()
        if "[@" not in text:
            continue
        new = resolve_citations(text, bibliography, str(md.relative_to(docs)), missing)
        if new != text:
            cited_pages += 1
            md.write_text(new)
    print(
        f"citations: {len(bibliography)} bib entries, resolved on {cited_pages} pages"
    )
    for page, key in sorted(missing):
        print(f"  ! missing bib key {key!r} on {page}")

    staged = stage_config(cfg, out)
    dumped = yaml.dump(staged, Dumper=Dumper, sort_keys=False, allow_unicode=True)
    # Write `!!python/name:x` the way MkDocs configs do, without a trailing ''.
    dumped = re.sub(r"(!!python/name:\S+) ''", r"\1", dumped)
    (out / "mkdocs.yml").write_text(
        "# Generated by scripts/stage_site.py from ../mkdocs.yml. Do not edit.\n"
        + dumped
    )
    print(f"config: {(out / 'mkdocs.yml').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
