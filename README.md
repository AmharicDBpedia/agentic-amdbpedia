# agentic-amdbpedia

Map Amharic Wikipedia infobox fields to the English DBpedia ontology. Hybrid
retrieval over the real ontology, a local LLM (Gemma 2 9B via Ollama) with
Groq fallback, MCP tools, and a SvelteKit chat in Amharic. The chat proposes
mappings, lets you edit each row, and downloads the finished
`{{TemplateMapping}}` wikitext for `mappings.dbpedia.org`.

## Architecture

![Architecture: SvelteKit frontend → HTTP API; Groq reads the message and writes the reply; the LangGraph pipeline runs extract → predict → format → save; retrieval merges dense + BM25 with learned aliases, then the LLM reranks or proposes; results land in the mapping card.](docs/architecture.svg)

A message takes five steps: Groq reads it (`/v1/assist/understand`), a link
is fetched if needed, the pipeline extracts fields → predicts properties →
formats the wikitext → saves a run record (`/v1/preview`, streamed), Groq
writes the Amharic reply (`/v1/assist/summarize`), and the mapping card
appears. Download writes the current rows as `{{TemplateMapping}}`
wikitext; publishing to the live wiki (`/v1/publish`) is implemented but
disabled in the UI for now.

Retrieval is local (dense + BM25 with reciprocal rank fusion over ~2,948
properties). On a confident hit the LLM reranks; otherwise it proposes a
property checked against the real ontology, with "none" as an option. Picks
are labelled **AI-proposed** in the UI. Downloaded (or published) rows
become learned aliases on the next request.

## Run a published release

```bash
curl -LO https://github.com/AmharicDBpedia/agentic-amdbpedia/releases/latest/download/docker-compose.release.yml
cp .env.example .env   # add GROQ_API_KEY and optionally bot credentials
TAG=v0.1.0 docker compose -f docker-compose.release.yml up -d
```

Frontend at `http://localhost:3000`, API at `http://localhost:8001`. Pin a
`vX.Y.Z` for anything you'll come back to; `latest` drifts. Images live at
`ghcr.io/amharicdbpedia/agentic-amdbpedia-{backend,frontend}` and ship for
`linux/amd64` and `linux/arm64`.

## Run from source

Prerequisites: Python 3.11, [uv](https://docs.astral.sh/uv/), Node.js +
pnpm, Docker (for PostgreSQL and the published Ollama image), a Groq API
key. Optional: a local Ollama install with `gemma2:9b`.

```bash
cp .env.example .env         # set GROQ_API_KEY; keep DATABASE_URL for Postgres
docker compose up -d postgres
just run-http                # API on :8001
# in another terminal
cd frontend && pnpm install && pnpm run dev --open   # UI on :5173
```

First startup downloads the embedding models and builds the ontology index
in `data/.cache/`. On a machine with less than ~7 GB of free RAM, skip
Ollama and run Groq-only:

```bash
env -u GROQ_API_KEY PREDICT_MODEL=groq/qwen/qwen3.8-27b PREDICT_FALLBACK_MODEL=none just run-http
```

Local Gemma, opt-in:

```bash
docker compose --profile local-llm up -d ollama
docker compose exec ollama ollama pull gemma2:9b   # ~5.4 GB, once
```

Set `PREDICT_MODEL=gemma2` once Ollama is up. An exported `GROQ_API_KEY` in
`~/.zshrc` or `~/.bashrc` takes precedence over `.env` and will fail with
`expired_api_key` if stale, hence the `env -u` dropper above.

## Using the app

The UI is in Amharic.

- Paste **infobox wikitext**, an **am.wikipedia.org link**, a **template
  link** (e.g. `https://am.wikipedia.org/wiki/መለጠፊያ:የምግብ_መረጃ`), or a
  single **field name**.
- Each row has a **pencil** (edit) and a **trash** (drop) button. Edit
  opens an inline input + ontology search; confirming marks the row
  **ተረጋግጧል**. Edited rows get an amber bar, added rows green, removed
  rows stay struck through until you restore them.
- The footer shows the change count and **ሁሉንም መልስ** (undo all).
  **Download** saves the current rows as `Mapping_am-<template>.wiki`.
- The chat is per browser tab (sessionStorage). **ውይይቱን አጽዳ** (Clear
  chat) warns about edits that have not been downloaded yet.

> Publishing to `mappings.dbpedia.org` from the UI currently fails with
> `writeapidenied`: that wiki denies the `writeapi` right to anonymous
> users, so every API login is refused. Download → paste by hand until a
> wiki admin restores the MediaWiki default
> `$wgGroupPermissions['*']['writeapi'] = true;`.

## Cut a release

From a clean `main`:

```bash
just release 0.2.0
git push origin main v0.2.0
```

That bumps `pyproject.toml` + `frontend/package.json`, prepends a
`## [v0.2.0]` section to `CHANGELOG.md`, commits, and tags. The release
workflow (`.github/workflows/release.yml`) builds and pushes multi-arch
images, then opens a GitHub Release with that CHANGELOG section as the
notes and `docker-compose.release.yml` attached.

## MCP server

```bash
just run-server                                         # stdio MCP server
uv run python scripts/print_desktop_config.py           # Claude Desktop block
```

Tools: `find_semantic_match`, `generate_mapping_syntax`,
`resources://benchmarks/latest`. Credentials stay in `.env`, not in the
desktop config.

## Common commands

```bash
just test              # fast unit tests
just lint              # ruff + strict mypy
just test-integration  # real-model integration tests
just test-e2e          # full pipeline tests
just test-perf         # latency checks
just validate-corpus   # ontology data sanity
```

Integration and end-to-end runs download real embedding models on first
run, so they need network access.

## Documentation

- [`implementation.md`](implementation.md) — roadmap and implementation detail
- [`examples/demo.md`](examples/demo.md) — complete demo transcripts
- [`frontend/README.md`](frontend/README.md) — routes and API contracts
- [`frontend/static/how-to-use.html`](frontend/static/how-to-use.html) — in-app Amharic usage guide
- [`frontend/static/documentation.html`](frontend/static/documentation.html) — in-app English concept guide
- [`corpus-refresh.md`](corpus-refresh.md) — refreshing ontology and mapping data
- [`evaluation/results.md`](evaluation/results.md) — retrieval evaluation results
- [`CHANGELOG.md`](CHANGELOG.md) — notable changes

Traceability lives in [`docs/traceability.md`](docs/traceability.md); the
automated suites link the main system boundaries to executable evidence.
