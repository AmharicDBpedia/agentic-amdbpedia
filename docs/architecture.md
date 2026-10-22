# Architecture

## Domain

This project supports **cross-lingual semantic web engineering** between
**Amharic Wikipedia** and the **English DBpedia ontology**. It helps editors
find schema-specific properties such as `length`, `openingDate`, and IATA or
ICAO identifiers without relying on literal translation alone.

**Retrieval-augmented generation is required** because ontology properties
are specialized and must be grounded in the real DBpedia schema.
**MCP tools are required** to expose that retrieval safely and to generate
deterministic mapping syntax instead of letting a language model invent XML.

## Architecture

![Architecture: SvelteKit frontend → HTTP API; Groq reads the message and writes the reply; the LangGraph pipeline runs extract → predict → format → save; retrieval merges dense + BM25 with learned aliases, then the LLM reranks or proposes; results land in the mapping card.](./architecture.svg)

A message takes five steps:

1. **Groq reads it** (`/v1/assist/understand`): an infobox, a Wikipedia link,
   or a field inside a question, and which DBpedia class it belongs to.
2. **A link is fetched** from `am.wikipedia.org`. For a template page, the
   template's declared parameters are mapped directly.
3. **The pipeline runs** (`/v1/preview`, streamed over SSE): extract the
   fields, predict a property for each, format the mapping syntax, and save
   a run record.
4. **Groq writes the reply** in Amharic (`/v1/assist/summarize`), and the
   result appears as an editable mapping card.
5. **Download** writes the current rows as `{{TemplateMapping}}` wikitext.
   The server-side `/v1/publish` endpoint for writing to
   `Mapping am:<template>` on `mappings.dbpedia.org` is implemented but
   disabled in the UI; it still produces **deterministic** wikitext from the
   final mapping set rather than letting the LLM invent XML.

### Retrieval

Retrieval runs locally and merges dense (sentence-transformer embeddings)
and sparse (BM25) rankings using reciprocal rank fusion over roughly 2,948
DBpedia properties. When it finds a confident match, the LLM reranks the
candidates with the target class in view. When it doesn't — the usual case
for Amharic vocabulary not yet in the corpus — the LLM first proposes a
property that is then validated against the real ontology. If the proposal
is rejected, the LLM picks among the dense model's unverified top 10, with
"none" as an explicit option.

Picks the LLM made (vs. retrieval) are labelled **AI-proposed** in the UI.
Fields that stay unmapped show their closest suggestions for a reviewer to
pick from. Downloaded (or published) rows are saved per class in
`data/learned_aliases.json` and consulted before retrieval on the next
request.

### Groq: understanding the message and writing the reply

Groq does the two language jobs around the mapping, never the mapping itself:

- **Before:** it reads what you typed and decides what to run. That's an
  infobox, a Wikipedia link, or a single field pulled out of a question like
  "የጋዜጣ ዋና_አዘጋጅ በDBpedia ምን ይባላል?" (field `ዋና_አዘጋጅ`, class
  `Newspaper`). It also infers the DBpedia class from an infobox's template
  name. A class is only used if it really exists in the ontology, and a field
  only if you actually typed it. Prompt-injection text never reaches Groq.
- **After:** it turns the result into a short reply in Amharic: what mapped,
  what the AI model suggested and should be checked, and what is still
  unmapped with the closest suggestions. A reply that names a property not
  in the result is discarded and replaced by a plain summary.

If Groq is unavailable, simple local rules take over for both steps.

### Language model

Every LLM call tries the models in order:

1. `PREDICT_MODEL`, by default the local **Gemma 2 9B via Ollama**.
2. `PREDICT_FALLBACK_MODEL`, by default `groq/<GROQ_MODEL_FAST>` using your
   `GROQ_API_KEY`. It's used whenever Ollama isn't running or times out. Set
   it to `none` to run local-only.

Without a GPU, Gemma 2 9B needs several seconds per answer, so
`PREDICT_LLM_TIMEOUT_SECONDS` defaults to 60. It also needs about 6.3 GB of
free RAM. On a machine with less, make Groq the mapping model instead (see
the README's "Run from source" section).

## Requirements Traceability

See [`traceability.md`](traceability.md) for the matrix linking system
boundaries to executable test evidence.

## Future Work

- Expand the Amharic alias corpus and evaluation set.
- Add authentication and authorization before exposing the publish endpoint
  beyond localhost.
- Automate post-publication DBpedia extraction and verification.
- Fine-tune the reranker from published and corrected mappings.
- Package the services for production deployment.
