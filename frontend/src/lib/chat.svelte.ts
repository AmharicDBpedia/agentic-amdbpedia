// The one conversation on screen. There are no user accounts, so there is
// no chat history: the conversation lives in this browser tab only
// (sessionStorage), surviving a reload so unpublished edits aren't lost,
// and is gone when the tab closes or the user presses "Clear chat".
import { browser } from '$app/environment';
import { changesFor } from './mapping-diff';
import type { AgentStep, MappingCandidate, PredictedMapping, UnmappedField } from './types';

/** Groq's final reply for a turn (`POST /v1/assist/summarize`). */
export interface TurnFeedback {
	message: string;
	source: 'groq' | 'rules';
}

export interface PipelineTurn {
	id: string;
	kind: 'pipeline';
	input: string;
	targetClass?: string;
	steps: AgentStep[];
	// `null` is "the run hasn't finished yet"; `[]` is "finished, nothing
	// confident enough to map".
	mappings: PredictedMapping[] | null;
	mappingWikitext?: string;
	xmlRules?: string;
	// The run record the pipeline saved; sent back on publish.
	reviewItemId?: string | null;
	// What a publish writes: `Mapping am:<templateName>` mapped to `domainClass`.
	templateName?: string | null;
	domainClass?: string;
	// The editable copy shown in the mapping card (starts as `mappings`).
	// Publishing sends exactly these rows.
	edited?: PredictedMapping[];
	// Pipeline notes: fields skipped as already-published, fields with no
	// confident match, which model answered, an auto-derived class.
	warnings?: string[];
	// Fields the pipeline couldn't map. Never shrinks: a field counts as
	// still unmapped while no row in `edited` maps it, so "Undo all" brings
	// it back.
	unmappedFields?: UnmappedField[];
	feedback?: TurnFeedback | null;
	publishing?: boolean;
	publishError?: string;
	published?: { title: string; url: string };
	// Snapshot of `edited` the last time the user clicked Download, as JSON.
	// `hasUnpublishedEdits` compares it with the current `edited` to tell
	// "unsaved" from "already downloaded" without threading a flag through
	// every mutation path.
	downloadedSnapshot?: string;
	running: boolean;
	error?: string;
}

export interface AnswerTurn {
	id: string;
	kind: 'answer';
	input: string;
	targetClass?: string;
	// The field Groq extracted from a free-form question, when it differs
	// from the raw input.
	field?: string;
	matches: MappingCandidate[] | null;
	noMatch: boolean;
	suggestions?: MappingCandidate[];
	llmUnavailable?: boolean;
	// A reply with nothing to map (greeting, off-topic) or a rejection.
	reply?: string;
	feedback?: TurnFeedback | null;
	running: boolean;
	error?: string;
}

export type Turn = PipelineTurn | AnswerTurn;

const STORAGE_KEY = 'amdbpedia:chat:v2';
// The old multi-chat history (one entry per saved chat). Removed on load.
const LEGACY_HISTORY_KEY = 'amdbpedia:chat-sessions:v1';

// A turn still running when the tab reloaded will never finish: its
// request died with the old page.
export function settleInterrupted(turns: Turn[]): Turn[] {
	for (const turn of turns) {
		if (turn.kind === 'pipeline') turn.publishing = false;
		if (!turn.running) continue;
		turn.running = false;
		turn.error ??= 'ገጹ እንደገና ስለተጫነ ተቋርጧል። እንደገና ለማስኬድ መልእክቱን ደግመው ይላኩ።';
	}
	return turns;
}

export function loadTurns(): Turn[] {
	if (!browser) return [];
	try {
		localStorage.removeItem(LEGACY_HISTORY_KEY);
	} catch {
		// Disabled site data: nothing to clean up anyway.
	}
	try {
		const raw = sessionStorage.getItem(STORAGE_KEY);
		const parsed = raw ? JSON.parse(raw) : [];
		return Array.isArray(parsed) ? settleInterrupted(parsed) : [];
	} catch (err) {
		console.error('Failed to load the chat from sessionStorage:', err);
		return [];
	}
}

export const chat = $state<{ turns: Turn[] }>({ turns: loadTurns() });

export function saveChat(turns: Turn[] = $state.snapshot(chat.turns)) {
	if (!browser) return;
	try {
		sessionStorage.setItem(STORAGE_KEY, JSON.stringify(turns));
	} catch (err) {
		// Private browsing / storage quota -- the chat still works, it just
		// won't survive a reload.
		console.error('Failed to save the chat to sessionStorage:', err);
	}
}

export function clearChat() {
	chat.turns.length = 0;
	if (!browser) return;
	try {
		sessionStorage.removeItem(STORAGE_KEY);
	} catch {
		// Nothing stored, nothing to clear.
	}
}

/** True when some mapping card has changes that have not been downloaded
 * (or published) yet. A card the user already Downloaded does not count,
 * even if `edited` still differs from `mappings`. */
export function hasUnpublishedEdits(turns: Turn[] = chat.turns): boolean {
	return turns.some((turn) => {
		if (turn.kind !== 'pipeline' || turn.published) return false;
		if (turn.mappings === null || turn.edited === undefined) return false;
		if (changesFor(turn.mappings, turn.edited).total === 0) return false;
		return JSON.stringify(turn.edited) !== turn.downloadedSnapshot;
	});
}
