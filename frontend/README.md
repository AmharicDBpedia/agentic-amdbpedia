# agentic-amdbpedia frontend

The web UI for the Amharic infobox → DBpedia mapping workflow: SvelteKit 5,
TypeScript, Tailwind v4, scaffolded with `sv create` and installed with pnpm.

## What's real vs. planned

The UI, routing, and API client are real and functional, and every endpoint
this frontend calls lives on `agentic-amdbpedia`'s own backend
(`mcp_server/http_app.py`, run with `uvicorn mcp_server.http_app:create_app
--factory` alongside the MCP stdio server) — this frontend has no remaining
dependency on `agentic-dbpedia` at all. `getCoverageStats` used to call
`agentic-dbpedia`'s `/api/statistics/summary`, an endpoint that (found live)
never actually existed there; it's since moved to `GET
{CROSS_LINGUAL_URL}/v1/coverage`, computed from this repo's own review
queue (`db/session.py::coverage_stats`) rather than depending on a
DEF-extraction-output crawl. A screen whose endpoint isn't reachable fails
closed into a visible "not reachable yet" message instead of crashing or
showing fake data — that's intentional, not a bug to fix later.

| Screen                 | Calls                                                     | Status                                                                 |
| ---------------------- | --------------------------------------------------------- | ---------------------------------------------------------------------- |
| Chat (`/`)             | `POST /v1/assist/understand`, `POST /v1/assist/summarize` | **Exists**: Groq reads each message and writes the final reply         |
| Chat (`/`)             | `POST /v1/preview` (SSE), `POST /v1/find-semantic-match`  | **Exists**: `mcp_server/http_app.py`                                   |
| Chat (`/`)             | `POST /v1/publish`                                        | **Exists**: writes `Mapping am:<template>` on mappings.dbpedia.org     |
| Coverage (`/coverage`) | `GET /v1/coverage`                                        | **Exists**: `mcp_server/http_app.py` + `db/session.py::coverage_stats` |

## Language

Every user-facing string is in Amharic (labels, placeholders, buttons,
dialogs, tooltips, `aria-label`s and errors), and Groq's replies are
requested in Amharic. DBpedia property and class names, `Mapping am:` page
titles and the «Publish» button stay as they are, since the Amharic replies
refer to «Publish» by name. The sidebar links to the English usage guide,
`static/documentation.html`, as a full page load (`data-sveltekit-reload`) so
the router doesn't treat it as an app route.

## Chat (`/`)

One conversation and one message box. There are no user accounts, so there
is no chat history: the conversation is kept in the browser tab
(`sessionStorage`, `src/lib/chat.svelte.ts`) so a reload keeps unpublished
edits, and it is gone when the tab closes or the user presses **Clear chat**
in the sidebar (which asks first if some edits are unpublished).

The message can be an infobox, an am.wikipedia.org link, or a question about
a field. The optional DBpedia class is the **Class: auto** chip next to the
send button. An infobox or link runs the extract → predict → format pipeline
over SSE and ends in a mapping card; a field question runs a grounded
lookup.

## Editing and publishing

Each infobox result is one mapping card
(`src/lib/components/MappingCard.svelte`), and every infobox field is a row
of the same table:

- The DBpedia property of every row is an editable box, with an ontology
  search beside it (the same `find-semantic-match` endpoint), so a fix is
  picked from real candidates instead of typed from memory.
- Changes are marked against the model's prediction
  (`src/lib/mapping-diff.ts`): edited rows have an amber bar and "edited
  from X · undo", added rows a green bar, removed rows stay in the table
  struck through with a restore button. AI-proposed rows are flagged
  "check this". The status chip and the footer count the changes, and
  **Undo all** puts the prediction back.
- Fields the pipeline couldn't map are empty rows with the closest
  suggestions; picking or typing a property adds them.
- **Publish** opens a confirmation that lists exactly what will be written,
  with each change shown before → after. A failed publish shows the real
  server reason (`PublishFailedError`, `src/lib/api.ts`).

## Auth

Deliberately skipped for this iteration — no auth headers, no login flow.
This is an internal tool for a small set of maintainers; add a shared token
or real auth once that becomes a problem, not before.

## Component library

**shadcn-svelte** primitives (`src/lib/components/ui/*` — button, input,
textarea, label, card, badge, table, checkbox, separator, tooltip,
alert-dialog, skeleton, sonner) built on **Bits UI**, styled with Tailwind
v4's CSS-variable theme (`src/routes/layout.css`, "new-york" style, a
warm neutral palette with its own accent color, tuned for both light and
dark) plus **`mode-watcher`** for a real light/dark toggle and
**Font Awesome** (`svelte-fa` + `@fortawesome/free-solid-svg-icons`) for
icons — kept to the small set that actually clarifies a control's purpose
(search, remove, restore, add, spinner, status, nav), not decorative ones.
The shadcn-svelte CLI's newer `init` flow requires an interactive TTY to
pick a design-system preset (confirmed: piped stdin is ignored outright,
so it can't be scripted) — these files were hand-authored to match its
actual generated output instead of fought into working non-interactively;
`pnpm dlx shadcn-svelte@latest add <name>` should still work against
`components.json` for adding more.

App-specific composites live one level up in `src/lib/components/`
(`AppSidebar`, `ModeToggle`, `ConfidencePill`, `StepTracker`,
`MappingCard`) rather than copy-pasted per page.

## Running it

```bash
pnpm install
cp .env.example .env   # point at your local backend, or leave the default
pnpm run dev --open
```

`pnpm run check` runs `svelte-check`; `pnpm run lint` / `pnpm run format`
wrap prettier + eslint, both already configured by `sv create`. `pnpm run
test` runs Vitest (`@testing-library/svelte` + jsdom). The tests mount the
real components: `chat-turn-mutation.test.ts` drives the chat from message
to publish (editing, Undo all, remove and restore, unmapped fields, the
class chip), and `chat-store.test.ts` covers the tab-only chat and the
sidebar's Clear chat.
