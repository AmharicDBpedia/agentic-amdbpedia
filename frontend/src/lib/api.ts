// Typed client for the cross-lingual HTTP API -- the only backend this
// frontend calls. Every endpoint here is documented with its contract; each
// function is labeled PLANNED or EXISTING in its own doc comment. Each
// function fails closed with BackendUnavailableError rather than throwing an
// opaque network error, so callers can render an honest "not connected yet"
// state instead of a crash.
import { CROSS_LINGUAL_URL } from './config';
import type {
	AgentStep,
	CoverageStats,
	MappingCandidate,
	MappingSyntax,
	PredictedMapping,
	UnmappedField
} from './types';

export class BackendUnavailableError extends Error {}

/**
 * The backend understood the publish request but the write itself failed
 * (missing MediaWiki bot credentials, a rejected edit, ...):
 * `POST /v1/publish` answers 502 with `{error_type, message}`. Distinct from
 * `BackendUnavailableError` so the chat can show the real reason.
 */
export class PublishFailedError extends Error {
	errorType: string;

	constructor(message: string, errorType: string) {
		super(message);
		this.errorType = errorType;
	}
}

async function safeFetch(input: string, init?: RequestInit): Promise<Response> {
	try {
		return await fetch(input, init);
	} catch (cause) {
		throw new BackendUnavailableError(`Could not reach ${input}`, { cause });
	}
}

type PreviewEvent =
	| AgentStep
	| ({
			node: 'result';
			mappings: PredictedMapping[];
			// Run record id -- sent back on publish so coverage stays right.
			reviewItemId: string | null;
			// The page a publish writes (`Mapping am:<templateName>`) and the
			// class it maps to (may have been derived from the template).
			templateName?: string | null;
			domainClass?: string;
			// Pipeline notes: fields skipped as already-published, fields
			// with no confident match, an auto-derived domain class. Absent
			// on older backends -- treat undefined as [].
			warnings?: string[];
			// Fields neither retrieval nor the LLM fallback could map, for
			// the chat's in-place "assign this one yourself" step. Absent on
			// older backends -- treat undefined as [].
			unmappedFields?: UnmappedField[];
	  } & MappingSyntax);

/**
 * True for a real `wikipedia.org` article link (any language edition) --
 * shared with `frontend/src/routes/+page.svelte` so "does this input go
 * through the URL-fetch path" is decided in exactly one place, matching
 * `mcp_server/wiki_fetch.py::parse_article_url`'s own host check.
 */
export function looksLikeWikipediaUrl(text: string): boolean {
	try {
		const url = new URL(text.trim());
		return /^([a-z0-9-]+\.)?wikipedia\.org$/i.test(url.hostname);
	} catch {
		return false;
	}
}

/**
 * EXISTING: POST {CROSS_LINGUAL_URL}/v1/preview — text/event-stream
 * (mcp_server/http_app.py, refs implementation.md 16.3). Streams the
 * mapping pipeline's (16.2) progress for a pasted infobox **or** a real
 * Wikipedia article link (`mcp_server/wiki_fetch.py` fetches its wikitext
 * server-side first, reported as its own `fetch_source_article` step,
 * before the same extract → predict → format → persist sequence runs).
 * Each SSE `data:` line is one JSON-encoded step; the final event carries
 * `node: "result"` with the predicted mappings. `targetClass` defaults to
 * `"Thing"` server-side when omitted.
 *
 * Was originally pointed at agentic-dbpedia (`AGENTIC_DBPEDIA_URL`) — that
 * predates this session settling on cross-lingual owning the full
 * pipeline (agentic-dbpedia is DEF-extraction-only); repointed here for
 * the same reason every other endpoint here lives there.
 */
export async function* previewMapping(
	source: string,
	targetClass?: string
): AsyncGenerator<PreviewEvent> {
	const body = looksLikeWikipediaUrl(source)
		? { url: source.trim(), target_class: targetClass }
		: { infobox: source, target_class: targetClass };
	const res = await safeFetch(`${CROSS_LINGUAL_URL}/v1/preview`, {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify(body)
	});
	if (!res.ok || !res.body) {
		throw new BackendUnavailableError(`preview failed: ${res.status}`);
	}

	const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
	let buffer = '';
	while (true) {
		const { value, done } = await reader.read();
		if (done) break;
		buffer += value;
		const events = buffer.split('\n\n');
		buffer = events.pop() ?? '';
		for (const chunk of events) {
			const dataLine = chunk.split('\n').find((line) => line.startsWith('data:'));
			if (!dataLine) continue;
			yield JSON.parse(dataLine.slice('data:'.length).trim()) as PreviewEvent;
		}
	}
}

/**
 * EXISTING: POST {CROSS_LINGUAL_URL}/v1/find-semantic-match
 * (mcp_server/http_app.py, refs implementation.md 16.3) — the same
 * find_semantic_match tool already exposed over MCP, mirrored over HTTP so
 * this frontend (and agentic-dbpedia's pipeline) can call it directly.
 */
export async function findSemanticMatch(
	amharicProperty: string,
	targetClass?: string
): Promise<{
	status: 'ok' | 'no_match' | 'rejected';
	matches: MappingCandidate[];
	message?: string;
	/** Unverified similarity matches, sent with `no_match`. */
	suggestions?: MappingCandidate[];
	/** Set when the language model could not be reached to try. */
	llm_error?: string | null;
}> {
	const res = await safeFetch(`${CROSS_LINGUAL_URL}/v1/find-semantic-match`, {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify({ amharic_property: amharicProperty, target_class: targetClass })
	});
	if (!res.ok) throw new BackendUnavailableError(`find_semantic_match failed: ${res.status}`);
	return res.json();
}

/** What Groq made of the user's message (`mcp_server/assistant.py`). */
export interface InputPlan {
	intent: 'infobox' | 'url' | 'field' | 'unsupported' | 'rejected';
	/** What to run: the raw infobox, the URL, or the extracted field name. */
	text: string;
	target_class: string | null;
	language: 'am' | 'en';
	reply: string | null;
	source: 'groq' | 'rules';
}

/**
 * POST /v1/assist/understand -- Groq turns free text into a plan (intent,
 * field, DBpedia class). Falls back to local rules if the call itself
 * fails, so a Groq outage never blocks mapping.
 */
export async function understandInput(text: string, targetClass?: string): Promise<InputPlan> {
	const local = (): InputPlan => ({
		intent: looksLikeWikipediaUrl(text) ? 'url' : text.includes('{{') ? 'infobox' : 'field',
		text: text.trim(),
		target_class: targetClass ?? null,
		language: 'en',
		reply: null,
		source: 'rules'
	});
	try {
		const res = await fetch(`${CROSS_LINGUAL_URL}/v1/assist/understand`, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({ text, target_class: targetClass })
		});
		return res.ok ? await res.json() : local();
	} catch {
		return local();
	}
}

/**
 * POST /v1/assist/summarize -- Groq writes the final reply for a result.
 * The server discards any reply naming a property not in the result.
 * Resolves to `null` (show no prose, the table still stands) on failure.
 */
export async function summarizeResult(
	kind: 'infobox' | 'field',
	result: Record<string, unknown>,
	language: 'am' | 'en' = 'en'
): Promise<{ message: string; source: 'groq' | 'rules' } | null> {
	try {
		const res = await fetch(`${CROSS_LINGUAL_URL}/v1/assist/summarize`, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({ kind, result, language })
		});
		return res.ok ? await res.json() : null;
	} catch {
		return null;
	}
}

/**
 * POST /v1/publish -- write the (possibly edited) mapping to
 * `Mapping am:<templateName>` on the live wiki. `consent: true` is sent
 * only after the chat's confirmation dialog. A failed write rejects with
 * `PublishFailedError` carrying the server's reason.
 */
export async function publishMapping(request: {
	templateName: string;
	domainClass: string;
	mappings: PredictedMapping[];
	originalMappings?: PredictedMapping[];
	reviewItemId?: string | null;
}): Promise<{ status: 'published'; title: string; url: string }> {
	const res = await safeFetch(`${CROSS_LINGUAL_URL}/v1/publish`, {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify({
			template_name: request.templateName,
			domain_class: request.domainClass,
			mappings: request.mappings.map((m) => ({
				templateProperty: m.templateProperty,
				ontologyProperty: m.ontologyProperty
			})),
			original_mappings: request.originalMappings?.map((m) => ({
				templateProperty: m.templateProperty,
				ontologyProperty: m.ontologyProperty
			})),
			review_item_id: request.reviewItemId ?? undefined,
			consent: true
		})
	});
	if (!res.ok) {
		const body = await res.json().catch(() => null);
		if (body && typeof body.message === 'string') {
			throw new PublishFailedError(body.message, body.error_type ?? 'unknown');
		}
		throw new BackendUnavailableError(`publish failed: ${res.status}`);
	}
	return res.json();
}

/**
 * EXISTING: GET {CROSS_LINGUAL_URL}/v1/coverage (mcp_server/http_app.py).
 *
 * Was originally pointed at agentic-dbpedia's `/api/statistics/summary` --
 * that endpoint never actually existed there (its real routes are
 * `/api/statistics/latest`/`generate`/`runs`, and even those compute a
 * different thing: raw DEF extraction-output triple counts). Repointed at
 * cross-lingual's own review queue, for the same reason every other
 * endpoint here already lives there: agentic-dbpedia is DEF-extraction
 * -only. See `db/session.py::coverage_stats`'s docstring for exactly what
 * "coverage" means computed this way.
 */
export async function getCoverageStats(): Promise<CoverageStats> {
	const res = await safeFetch(`${CROSS_LINGUAL_URL}/v1/coverage`);
	if (!res.ok) throw new BackendUnavailableError(`statistics failed: ${res.status}`);
	return res.json();
}
