"""Mark a built site as the preview of a pull request.

    python3 scripts/mark_preview.py build/site <pr-number> <commit-sha>

Every HTML file gets a `noindex` robots tag, so search engines keep previews
out of their results. Book pages also get a small badge with the pull request
and the commit, which makes it easy to tell that the page is a preview and
whether it is the latest one.
"""

import html
import os
import sys
from pathlib import Path

NOINDEX = '<meta name="robots" content="noindex, nofollow">'
BADGE = """<a class="dl-preview-badge" href="{url}" style="position:fixed;left:.6rem;bottom:.6rem;z-index:10;\
padding:.25rem .6rem;border-radius:1rem;font:600 12px/1.4 system-ui,sans-serif;text-decoration:none;\
background:#ff6e42;color:#fff;box-shadow:0 1px 4px rgba(0,0,0,.3)">Preview: PR #{pr} at {sha}</a>\
<style>@media print{{.dl-preview-badge{{display:none}}}}</style>"""


def main(site: Path, pr: str, sha: str) -> None:
    if not pr.isdigit():
        raise SystemExit(f"the pull request number must be digits: {pr}")
    server = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
    repo = os.environ.get("GITHUB_REPOSITORY", "emptymalei/deep-learning")
    badge = BADGE.format(
        url=html.escape(f"{server}/{repo}/pull/{pr}"), pr=pr, sha=html.escape(sha[:7])
    )
    marked = badged = 0
    for path in site.rglob("*.html"):
        text = path.read_text(encoding="utf-8")
        if NOINDEX in text or "<head>" not in text:
            continue
        text = text.replace("<head>", "<head>" + NOINDEX, 1)
        marked += 1
        # Only pages of the book: not the playground exports shown in iframes.
        if 'class="md-header' in text and "</body>" in text:
            head, _, tail = text.rpartition("</body>")
            text = head + badge + "</body>" + tail
            badged += 1
        path.write_text(text, encoding="utf-8")
    print(f"preview of PR #{pr}: noindex on {marked} pages, badge on {badged}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    main(Path(sys.argv[1]), sys.argv[2], sys.argv[3])
