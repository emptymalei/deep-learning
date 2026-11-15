# Design: An MCP Server for the Book

Status: proposal, not built. Source: DTS-13 (Deep Learning TS).

## Goal

Let AI assistants that speak the [Model Context Protocol](https://modelcontextprotocol.io) (Claude, ChatGPT connectors, IDE agents) search the book, read pages and notebooks, and cite its references, without scraping HTML. It must cost nothing to run and must not track readers.

`llms.txt` and the per-page Markdown already cover assistants that only fetch URLs. The MCP server adds search and structured answers on top of the same content.

## Tools

All tools are read-only.

| Tool | Input | Output |
|---|---|---|
| `search` | `query` (string), `limit` (int, default 5, max 20) | Ranked pages: `title`, `url`, `markdown_url`, `section`, `snippet`, `score`. |
| `get_page` | `page`: a path relative to `dl/` (`time-series/timeseries-basics.ar.md`) or a page URL | `title`, `url`, `section`, `summary`, `markdown`, `prerequisites`, `related`, `notebook`. |
| `get_notebook` | `name`: notebook stem (`rnn_timeseries`) or path | `path`, `page_url`, `github_url`, `source` (the jupytext `.py`), `libraries` (top-level imports). |
| `cite` | `key` (bib key) or `page` | The formatted reference(s), with DOI or URL, in the book's citation style. For a page: every reference it cites. |

`prerequisites`, `related`, `notebook` and `summary` come from the page front matter described in `AGENTS.md`; empty when a page does not set them.

Optionally, each page can also be exposed as an MCP resource (`dl://page/<path>`), so clients that browse resources see the table of contents.

## Data

The server reads data the site build already produces, plus one small index:

- Page Markdown: `<page URL>index.md` from the `llmstxt` plugin.
- Page list and sections: `llms.txt`.
- New: `site/mcp/index.json`, written by `scripts/stage_site.py --post-build`, with one record per page (path, URL, title, section, summary, front matter fields, cited keys), one per notebook (stem, path, page, imports) and the bibliography rendered with the same formatter the footnotes use (about 100 entries).
- Search: a [MiniSearch](https://github.com/lucaong/minisearch) index serialised into `site/mcp/search.json` at build time, over titles, headings and body text. The Zensical search index (`search.json`) is an alternative, but its format is internal to the theme.

Everything is published with the site on GitHub Pages, so a content change needs no redeploy of the server: the Worker fetches the JSON over HTTPS and keeps it in the Cloudflare cache (for example `max-age=600`, matching GitHub Pages).

## Hosting

A Cloudflare Worker on the free plan (100,000 requests a day), stateless:

- The Streamable HTTP transport at `/mcp`, with the [`@modelcontextprotocol/sdk`](https://github.com/modelcontextprotocol/typescript-sdk) server. No Durable Objects, no sessions, no storage, so it stays on the free plan.
- Route: `mcp.dl.leima.is` (a Worker custom domain on the existing Cloudflare zone).
- No authentication: the content is public. Cloudflare's rate limiting rule protects the free quota.
- No logging beyond Cloudflare's defaults; no analytics. Queries are not stored.
- Code in `mcp/` in this repository (`wrangler.toml`, `src/index.ts`, tests), deployed with `wrangler deploy` from a GitHub Action on pushes to `main` that touch `mcp/`. That Action needs a `CLOUDFLARE_API_TOKEN` secret with Workers edit rights, which the owner has to create.

## Alternatives Considered

- **Static only (no server)**: `llms.txt` and per-page Markdown already work for fetch-based assistants; no search, no structured citations.
- **Python server on a VM**: runs the same code as the build, but costs money and needs maintenance.
- **Bundling the data into the Worker**: faster cold reads, but every content change would need a Worker redeploy.

## Plan

1. Build step: write `site/mcp/index.json` and `site/mcp/search.json` in `scripts/stage_site.py --post-build`, with a unit test on a tiny docs tree.
2. Worker with `get_page`, `get_notebook` and `cite` (exact lookups), tested with the MCP Inspector against a local `wrangler dev`.
3. `search` over the MiniSearch index.
4. Deploy to `mcp.dl.leima.is`; document the connector URL in the README and `llms.txt`.

## Open Questions

- Is a subdomain (`mcp.dl.leima.is`) acceptable, or should it live under `dl.leima.is/mcp` via a Worker route?
- Should `get_notebook` also return rendered outputs, once notebook outputs are published (DTS-7)?
