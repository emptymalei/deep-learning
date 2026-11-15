# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "jupytext>=1.16",
#     "nbformat>=5.10",
#     "pillow>=10",
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

    uv run --python 3.14 scripts/stage_site.py          # stage into build/
    uvx zensical build -f build/mkdocs.yml              # build into build/site
    uv run --python 3.14 scripts/stage_site.py --post-build   # fix notebook edit links
"""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import io
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

# Outputs saved from local notebook runs (`--save-outputs`), used when there is
# no local .ipynb, as in CI. Notebooks are never executed by the build.
OUTPUTS_DIR = ROOT / "notebook-outputs"

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
def load_outputs(py_path: Path):
    """The notebook with outputs for a .py: the local .ipynb, else the saved one."""
    local = py_path.with_suffix(".ipynb")
    if local.exists():
        return nbformat.read(local, as_version=4)
    saved = OUTPUTS_DIR / f"{py_path.stem}.ipynb.gz"
    if saved.exists():
        return nbformat.reads(
            gzip.decompress(saved.read_bytes()).decode(), as_version=4
        )
    return None


DROP_MIMES = (
    "application/vnd.jupyter.widget-view+json",  # live widgets, e.g. tqdm bars
    "application/vnd.jupyter.widget-state+json",
)
STRIP_MIMES = ("application/vnd.microsoft.datawrangler.viewer.v0+json",)
MAX_IMAGE_WIDTH = 1600
MAX_STREAM_LINES = 120


def _downscale(b64: str, fmt: str) -> str:
    from PIL import Image

    img = Image.open(io.BytesIO(base64.b64decode(b64)))
    if img.width <= MAX_IMAGE_WIDTH:
        return b64
    height = round(img.height * MAX_IMAGE_WIDTH / img.width)
    img = img.resize((MAX_IMAGE_WIDTH, height), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format=fmt, optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def clean_outputs(nb) -> int:
    """Keep what a reader of the page needs; return the outputs kept."""
    nb.metadata.pop("widgets", None)
    kept = 0
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        cell.metadata = {}
        new, stream = [], []
        for out in cell.get("outputs", []):
            kind = out.get("output_type")
            if kind == "error":
                continue
            if kind == "stream":
                stream.extend("".join(out.get("text", "")).splitlines(keepends=True))
                continue
            data = out.get("data", {})
            if any(m in data for m in DROP_MIMES):
                continue
            for m in STRIP_MIMES:
                data.pop(m, None)
            for mime, fmt in (("image/png", "PNG"), ("image/jpeg", "JPEG")):
                if mime in data:
                    data[mime] = _downscale("".join(data[mime]), fmt)
            new.append(out)
        if stream:
            text = stream[-MAX_STREAM_LINES:]
            if len(stream) > MAX_STREAM_LINES:
                text.insert(
                    0, f"... ({len(stream) - MAX_STREAM_LINES} earlier lines)\n"
                )
            new.insert(
                0, nbformat.v4.new_output("stream", name="stdout", text="".join(text))
            )
        cell.outputs = new
        kept += len(new)
    return kept


def save_outputs(src_docs: Path) -> None:
    """Save the outputs of local notebook runs into notebook-outputs/."""
    OUTPUTS_DIR.mkdir(exist_ok=True)
    for py in sorted((src_docs / "notebooks").glob("*.py")):
        local = py.with_suffix(".ipynb")
        if not local.exists():
            continue
        nb = nbformat.read(local, as_version=4)
        if not any(c.get("outputs") for c in nb.cells if c.cell_type == "code"):
            continue
        kept = clean_outputs(nb)
        if not kept:
            continue
        # nbformat 4.5: every cell has an id (some editors add ids to older
        # notebooks without bumping the minor version).
        nb.nbformat_minor = max(nb.get("nbformat_minor", 0), 5)
        for i, cell in enumerate(nb.cells):
            if not cell.get("id"):
                cell["id"] = hashlib.sha1(f"{i}{cell.source}".encode()).hexdigest()[:8]
        nbformat.validate(nb)
        target = OUTPUTS_DIR / f"{py.stem}.ipynb.gz"
        target.write_bytes(gzip.compress(nbformat.writes(nb).encode(), mtime=0))
        print(
            f"saved {kept:4d} outputs  {target.relative_to(ROOT)} ({target.stat().st_size / 1e6:.2f} MB)"
        )


def notebook_to_markdown(py_path: Path, out_md: Path) -> int:
    nb = jupytext.read(py_path)
    outputs_nb = load_outputs(py_path)
    if outputs_nb is not None:
        try:
            from jupytext.combine import combine_inputs_with_outputs

            # Cells are matched by their code: a cell that changed since the
            # run gets no (stale) output.
            nb = combine_inputs_with_outputs(nb, outputs_nb)
        except Exception as err:
            print(f"  ! could not merge outputs into {py_path.name}: {err}")

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


def _nav_pages(item):
    if isinstance(item, str):
        yield item
    elif isinstance(item, dict):
        for value in item.values():
            yield from _nav_pages(value)
    elif isinstance(item, list):
        for value in item:
            yield from _nav_pages(value)


def _summary(md: Path) -> str | None:
    """The `summary` front matter field of a staged page, if any."""
    text = md.read_text()
    end = text.find("\n---", 4) if text.startswith("---\n") else -1
    if end < 0:
        return None
    meta = yaml.safe_load(text[4:end]) or {}
    summary = meta.get("summary") if isinstance(meta, dict) else None
    return " ".join(str(summary).split()) if summary else None


def llms_sections(nav: list, docs: Path) -> dict:
    """llms.txt sections: one per top-level nav entry, in nav order, then the
    pages that are not in the nav. A page's `summary` becomes its description."""

    def entry(path: str):
        summary = _summary(docs / path)
        return {path: summary} if summary else path

    sections: dict = {}
    seen: set = set()
    for item in nav:
        title, value = (
            next(iter(item.items())) if isinstance(item, dict) else ("Pages", item)
        )
        files = [
            p
            for p in _nav_pages(value)
            if p.endswith(".md")
            and p != "tags.md"
            and p not in seen
            and (docs / p).exists()
        ]
        seen.update(files)
        if files:
            sections.setdefault(title, []).extend(entry(p) for p in files)
    rest = sorted(
        str(p.relative_to(docs))
        for p in docs.rglob("*.md")
        if not str(p.relative_to(docs)).startswith("assets/")
        and p.name != "tags.md"
        and str(p.relative_to(docs)) not in seen
    )
    if rest:
        sections["Other pages"] = [entry(p) for p in rest]
    return sections


def stage_config(cfg: dict, out_dir: Path) -> dict:
    cfg = dict(cfg)
    cfg["docs_dir"] = "docs"
    cfg["site_dir"] = "site"
    theme = dict(cfg["theme"])
    if "custom_dir" in theme:
        theme["custom_dir"] = str(
            (ROOT / theme["custom_dir"]).relative_to(out_dir, walk_up=True)
        )
    cfg["theme"] = theme
    plugins = []
    for p in cfg.get("plugins", []):
        name = p if isinstance(p, str) else next(iter(p))
        if name not in DROP_PLUGINS:
            plugins.append(p)
    cfg["plugins"] = plugins
    cfg["nav"] = rewrite_nav(cfg["nav"])
    for i, p in enumerate(cfg["plugins"]):
        if isinstance(p, dict) and "llmstxt" in p:
            conf = dict(p["llmstxt"] or {})
            conf.setdefault("sections", llms_sections(cfg["nav"], out_dir / "docs"))
            cfg["plugins"][i] = {"llmstxt": conf}
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


def link_notebook_sources(site: Path, src_docs: Path) -> int:
    """Point the edit/view buttons of converted notebooks at their .py source.

    Zensical derives those links from the page's source file, which for a
    notebook is the generated .md that does not exist in the repository.
    """
    changed = 0
    for py in sorted((src_docs / "notebooks").glob("*.py")):
        page = site / "notebooks" / py.stem / "index.html"
        if not page.exists():
            continue
        html = page.read_text()
        new = html.replace(f'/dl/notebooks/{py.stem}.md"', f'/dl/notebooks/{py.name}"')
        if new != html:
            page.write_text(new)
            changed += 1
    return changed


def fingerprint_assets(site: Path, cfg: dict) -> int:
    """Append a content hash to the site's own CSS and JS links.

    The files keep their names (`assets/widgets/widgets.css`), and the host
    lets browsers cache them for hours, so without this a reader could get a
    new page with an old stylesheet or script after a deploy.
    """
    assets = {}
    for path in cfg.get("extra_css", []) + cfg.get("extra_javascript", []):
        path = path if isinstance(path, str) else path.get("path", "")
        if "://" in path or not (site / path).is_file():
            continue
        digest = hashlib.sha256((site / path).read_bytes()).hexdigest()[:8]
        assets[f'{path}"'] = f'{path}?v={digest}"'
    pages = 0
    for html in site.rglob("*.html"):
        if "assets/playgrounds" in str(html):
            continue
        text = html.read_text()
        new = text
        for old, versioned in assets.items():
            new = new.replace(old, versioned)
        if new != text:
            html.write_text(new)
            pages += 1
    return pages


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="build", help="output folder (default: build)")
    ap.add_argument(
        "--save-outputs",
        action="store_true",
        help="save the outputs of your local notebook runs into notebook-outputs/",
    )
    ap.add_argument(
        "--post-build",
        action="store_true",
        help="after `zensical build`: fix the edit/view links of notebook pages",
    )
    args = ap.parse_args()

    out = (ROOT / args.out).resolve()
    cfg = yaml.load(SRC_CONFIG.read_text(), Loader=Loader)
    src_docs = ROOT / cfg.get("docs_dir", "docs")
    docs = out / "docs"

    if args.save_outputs:
        save_outputs(src_docs)
        return

    if args.post_build:
        n = link_notebook_sources(out / "site", src_docs)
        print(
            f"post-build: edit/view links of {n} notebook pages point at the .py source"
        )
        n = fingerprint_assets(out / "site", cfg)
        print(f"post-build: versioned CSS/JS links on {n} pages")
        return

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
        f"notebooks: {len(notebooks)} converted, {total_outputs} outputs merged from local or saved runs"
    )

    # Links to notebook sources now point at the converted pages, and the old
    # Material tags marker becomes the listing marker Zensical understands.
    nb_link = re.compile(r"(\]\((?:\.\./)*(?:notebooks/)?[\w.-]+)\.py(\)|#)")
    for md in docs.rglob("*.md"):
        text = md.read_text()
        new = text.replace("[TAGS]", "<!-- material/tags -->")
        new = nb_link.sub(
            lambda m: (
                m.group(1) + ".md" + m.group(2)
                if (md.parent / (m.group(1).split("](", 1)[1] + ".md"))
                .resolve()
                .exists()
                else m.group(0)
            ),
            new,
        )
        if new != text:
            md.write_text(new)

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
