# /// script
# requires-python = ">=3.10"
# ///
"""Rewrite Markdown links so they resolve relative to the source file.

Many pages were written with links relative to the *rendered* URL: with
directory URLs, ``dl/time-series/foo.md`` is served at ``/time-series/foo/``,
so ``../assets/x.png`` happened to work in MkDocs. Such links break on
GitHub, in editors, in llms.txt / Markdown exports and in Zensical, which
resolves links relative to the source file (as MkDocs does for valid ones).

    python3 scripts/fix_links.py            # report only
    python3 scripts/fix_links.py --write    # rewrite in place
    python3 scripts/fix_links.py --write dl/time-series/foo.md   # some files

Exit code 1 if any link is broken either way, so it can run in CI.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "dl"

INLINE = re.compile(r"(!?\[[^\]\n]*\]\()([^)\s]+)((?:\s+\"[^\"]*\")?\))")
# Link reference definitions, but not footnote definitions ([^note]: text).
REFDEF = re.compile(r"^(\s{0,3}\[(?!\^)[^\]]+\]:\s*)(\S+)", re.M)
FENCE = re.compile(r"(^\s*```.*?^\s*```)", re.S | re.M)
EXTERNAL = re.compile(r"^(?:[a-z][a-z0-9+.-]*:|//|#)", re.I)


def candidates(path: Path):
    yield path
    yield path.with_name(path.name + ".md")
    yield path.with_name(path.name + ".py")
    yield path / "index.md"


def resolve(base: Path, target: str) -> Path | None:
    p = (base / target).resolve()
    for c in candidates(p):
        if c.is_file() and DOCS in c.parents:
            return c
    return None


def fix_target(md: Path, target: str, problems: list) -> str:
    if EXTERNAL.match(target):
        return target
    path, sep, anchor = target.partition("#")
    if not path:
        return target
    if resolve(md.parent, path):
        return target  # already source-relative
    url_base = md.parent if md.name == "index.md" else md.parent / md.stem
    found = resolve(url_base, path)
    if not found:
        problems.append(target)
        return target
    rel = os.path.relpath(found, md.parent).replace(os.sep, "/")
    return rel + (sep + anchor if sep else "")


def process(md: Path, write: bool) -> tuple[int, list]:
    text = md.read_text()
    problems: list = []
    changed = 0

    def sub_inline(m):
        nonlocal changed
        new = fix_target(md, m.group(2), problems)
        changed += new != m.group(2)
        return m.group(1) + new + m.group(3)

    def sub_ref(m):
        nonlocal changed
        new = fix_target(md, m.group(2), problems)
        changed += new != m.group(2)
        return m.group(1) + new

    parts = FENCE.split(text)
    for i in range(0, len(parts), 2):  # skip fenced code blocks
        parts[i] = REFDEF.sub(sub_ref, INLINE.sub(sub_inline, parts[i]))
    new_text = "".join(parts)
    if write and new_text != text:
        md.write_text(new_text)
    return changed, problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true", help="rewrite files in place")
    ap.add_argument(
        "files", nargs="*", type=Path, help="Markdown files (default: all under dl/)"
    )
    args = ap.parse_args()

    if args.files:
        files = [f.resolve() for f in args.files]
    else:
        listed = subprocess.run(
            [
                "git",
                "ls-files",
                "--cached",
                "--others",
                "--exclude-standard",
                "dl/*.md",
                "dl/**/*.md",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()
        files = [ROOT / f for f in listed]

    total, broken = 0, 0
    for md in sorted(files):
        changed, problems = process(md, args.write)
        total += changed
        rel = md.relative_to(ROOT)
        if changed:
            print(f"{'fixed' if args.write else 'would fix'} {changed:3d}  {rel}")
        for p in problems:
            broken += 1
            print(f"broken       {rel}: {p}")
    print(
        f"\n{total} links {'rewritten' if args.write else 'to rewrite'}, {broken} broken"
    )
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())
