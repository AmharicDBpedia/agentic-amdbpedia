// Chat history, persisted client-side. There's no backend "conversations"
// table to back this with, so this is deliberately
// localStorage-only: per-browser, not shared across devices or reviewers.
// Good enough for "the thread I was just looking at is still there after I
// navigate away and back" without inventing backend storage this app
// doesn't otherwise need.
import { browser } from '$app/environment';
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
	// The editable copy shown in the chat table (starts as `mappings`).
	// Publishing sends exactly these rows.
	edited?: PredictedMapping[];
	// Pipeline notes: fields skipped as already-published, fields with no
	// confident match, which model answered, an auto-derived class.
	warnings?: string[];
	// Fields the pipeline couldn't map; each can be assigned from the chat.
	unmappedFields?: UnmappedField[];
	// Older saved chats kept hand-assigned rows here; folded into `edited`.
	manualMappings?: PredictedMapping[];
	feedback?: TurnFeedback | null;
	publishing?: boolean;
	publishError?: string;
	published?: { title: string; url: string };
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

export interface ChatSession {
	id: string;
	title: string;
	turns: Turn[];
	updatedAt: number;
}

const STORAGE_KEY = 'amdbpedia:chat-sessions:v1';
const MAX_SESSIONS = 50;

// Chats saved before in-chat editing have no `edited` rows. Filling them in
// here, not while rendering, because Svelte forbids writing state from a
// template expression (state_unsafe_mutation).
export function withEditedRows(loaded: ChatSession[]): ChatSession[] {
	for (const session of loaded) {
		for (const turn of session.turns ?? []) {
			if (turn.kind !== 'pipeline' || turn.edited || !turn.mappings) continue;
			turn.edited = [...turn.mappings, ...(turn.manualMappings ?? [])].map((row) => ({ ...row }));
		}
	}
	return loaded;
}

function loadSessions(): ChatSession[] {
	if (!browser) return [];
	try {
		const raw = localStorage.getItem(STORAGE_KEY);
		if (!raw) return [];
		const parsed = JSON.parse(raw);
		return Array.isArray(parsed) ? withEditedRows(parsed) : [];
	} catch (err) {
		console.error('Failed to load chat history from localStorage:', err);
		return [];
	}
}

export const sessions = $state<ChatSession[]>(loadSessions());
export const active = $state<{ id: string | null }>({ id: sessions[0]?.id ?? null });

function persist() {
	if (!browser) return;
	try {
		localStorage.setItem(STORAGE_KEY, JSON.stringify($state.snapshot(sessions)));
	} catch (err) {
		// Private browsing / storage quota / disabled site data -- a chat
		// still works for the session, it just won't survive a reload.
		console.error('Failed to persist chat history to localStorage:', err);
	}
}

export function deriveTitle(text: string): string {
	const oneLine = text.replace(/\s+/g, ' ').trim();
	if (!oneLine) return 'New chat';
	return oneLine.length > 48 ? oneLine.slice(0, 48) + '…' : oneLine;
}

export function newSession(): string {
	const id = crypto.randomUUID();
	sessions.unshift({ id, title: 'New chat', turns: [], updatedAt: Date.now() });
	if (sessions.length > MAX_SESSIONS) sessions.length = MAX_SESSIONS;
	active.id = id;
	persist();
	return id;
}

export function ensureActiveSession(): ChatSession {
	const existing = sessions.find((s) => s.id === active.id);
	if (existing) return existing;
	newSession();
	return sessions[0];
}

export function currentSession(): ChatSession | undefined {
	return sessions.find((s) => s.id === active.id);
}

export function selectSession(id: string) {
	active.id = id;
}

/** Used by the sidebar's "New chat" button: only actually creates a new
 * (empty) session if the current one already has content, so repeated
 * clicks don't pile up empty "New chat" placeholders in the history. */
export function startNewChat(): string {
	const existing = currentSession();
	if (!existing || existing.turns.length > 0) return newSession();
	return existing.id;
}

export function touchSession(id: string) {
	const index = sessions.findIndex((s) => s.id === id);
	if (index === -1) return;
	const session = sessions[index];
	session.updatedAt = Date.now();
	if (session.title === 'New chat' && session.turns.length > 0) {
		session.title = deriveTitle(session.turns[0].input);
	}
	// Most-recently-touched floats to the top, matching every chat product's
	// history ordering.
	if (index > 0) {
		sessions.splice(index, 1);
		sessions.unshift(session);
	}
	persist();
}

export function deleteSession(id: string) {
	const index = sessions.findIndex((s) => s.id === id);
	if (index === -1) return;
	sessions.splice(index, 1);
	if (active.id === id) active.id = sessions[0]?.id ?? null;
	persist();
}
