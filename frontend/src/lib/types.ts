export interface MappingCandidate {
	property: string;
	class: string;
	score: number;
	payload?: Record<string, unknown>;
}

export interface AgentStep {
	node: string;
	status: 'running' | 'done' | 'error';
	detail?: string;
	timestamp?: string;
}

export interface PredictedMapping {
	templateProperty: string;
	ontologyProperty: string;
	confidence: number;
	// How this row was arrived at: `retrieval` (the ontology search, the
	// default), `llm` (chosen by the language model because retrieval
	// found nothing confident — treat as a weak guess), `learned` (a
	// reviewer approved this exact field for this class before), or
	// `manual` (a reviewer assigned it by hand from the chat). Absent on
	// older backends — treat undefined as `retrieval`.
	source?: 'retrieval' | 'llm' | 'learned' | 'manual';
	// What the reviewer did with this prediction in the chat: `confirmed`
	// (clicked ✓), `rejected` (clicked ✗ and should fix it). Undefined
	// means "not yet acknowledged". Not sent on publish — the backend
	// doesn't care; this is only to drive the actions column.
	ack?: 'confirmed' | 'rejected';
}

/** A field neither retrieval nor the LLM fallback could map, carried on
 * the pipeline's final result event so the chat can offer a "search and
 * assign this one yourself" step. `value` is truncated server-side. */
export interface UnmappedField {
	name: string;
	value: string;
	// Ungated similarity matches -- unverified, for a human to pick from.
	// Absent on older backends.
	suggestions?: MappingCandidate[];
}

/**
 * The pipeline's `format_mapping_syntax` step always computes both of
 * these deterministically alongside `mappings` (`mcp_server/pipeline.py`)
 * — the same MediaWiki `{{TemplateMapping ...}}` wikitext a publish would
 * actually write, and its XML-syntax equivalent. Not persisted on
 * `ReviewItem`: cheaply regenerable from `mappings`/`domainClass` at any
 * time, so nothing worth duplicating in Postgres.
 */
export interface MappingSyntax {
	mappingWikitext: string;
	xmlRules: string;
}

export interface ChatMessage {
	role: 'user' | 'assistant';
	content: string;
}

export type ReviewStatus = 'pending_review' | 'approved' | 'rejected' | 'published';

export interface ReviewItem {
	id: string;
	templateName: string;
	domainClass: string;
	status: ReviewStatus;
	submittedAt: string;
	mappings: PredictedMapping[];
}

export interface CoverageStats {
	totalTemplates: number;
	mappedTemplates: number;
	coveragePercent: number;
	lastRunAt?: string;
}
