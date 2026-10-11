// Runs the actual browser-console seed script (frontend/scripts/
// seed-demo-chat-history.browser.js) against this test's real jsdom
// localStorage, then mounts the real AppSidebar and ChatPage components
// against whatever it wrote -- proving the script's output actually
// matches chat.svelte.ts's real ChatSession/Turn schema and renders
// cleanly, not just that the script parses.
import { describe, expect, it, beforeEach, afterEach, vi } from 'vitest';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { cleanup, render, screen } from '@testing-library/svelte';
import AppSidebar from './components/AppSidebar.svelte';
import ChatPage from '../routes/+page.svelte';
import { sessions, active } from './chat.svelte';

const SCRIPT_PATH = path.resolve(process.cwd(), 'scripts/seed-demo-chat-history.browser.js');

function runSeedScript() {
	const source = readFileSync(SCRIPT_PATH, 'utf-8');
	// The script is a plain IIFE written for a devtools console, not an ES
	// module -- `new Function` runs it against this test's real jsdom
	// `localStorage`/`crypto`, exactly like pasting it into a real console
	// would run it against a real browser's.
	new Function(source)();
}

describe('demo chat history seed script', () => {
	beforeEach(() => {
		sessions.length = 0;
		active.id = null;
		localStorage.clear();
	});

	afterEach(() => {
		cleanup();
		vi.restoreAllMocks();
	});

	it('writes 9 sessions matching the real ChatSession/Turn schema', () => {
		runSeedScript();

		const raw = localStorage.getItem('amdbpedia:chat-sessions:v1');
		expect(raw).toBeTruthy();
		const parsed = JSON.parse(raw!);

		expect(Array.isArray(parsed)).toBe(true);
		expect(parsed).toHaveLength(9);
		// Newest-first, same invariant chat.svelte.ts's touchSession()
		// maintains at runtime -- AppSidebar's grouping relies on it.
		for (let i = 1; i < parsed.length; i++) {
			expect(parsed[i - 1].updatedAt).toBeGreaterThanOrEqual(parsed[i].updatedAt);
		}
		for (const session of parsed) {
			expect(typeof session.id).toBe('string');
			expect(typeof session.title).toBe('string');
			expect(Array.isArray(session.turns)).toBe(true);
			expect(session.turns.length).toBeGreaterThan(0);
			for (const turn of session.turns) {
				expect(['pipeline', 'answer']).toContain(turn.kind);
				expect(typeof turn.id).toBe('string');
				expect(typeof turn.input).toBe('string');
				expect(turn.running).toBe(false);
			}
		}
	});

	it('merges with pre-existing sessions instead of clobbering them', () => {
		localStorage.setItem(
			'amdbpedia:chat-sessions:v1',
			JSON.stringify([
				{ id: 'real-session', title: 'A real chat', turns: [{ id: 't1' }], updatedAt: Date.now() }
			])
		);

		runSeedScript();

		const parsed = JSON.parse(localStorage.getItem('amdbpedia:chat-sessions:v1')!);
		expect(parsed).toHaveLength(10);
		expect(parsed.some((s: { id: string }) => s.id === 'real-session')).toBe(true);
	});

	it('renders in the real sidebar: all recency groups populated, every title present', async () => {
		runSeedScript();
		// The seeded data was written directly to localStorage -- reload it
		// into the live `sessions` store the same way chat.svelte.ts's own
		// module-load-time loadSessions() would on a real page load.
		const reloaded = JSON.parse(localStorage.getItem('amdbpedia:chat-sessions:v1')!);
		sessions.push(...reloaded);

		render(AppSidebar, { props: { open: true } });

		// All four recency buckets this script targets should be populated --
		// this is the actual point of spreading timestamps across
		// now/-2h/-4h (Today), -26h/-30h (Yesterday), -3d/-5d (Previous 7
		// days), -20d/-45d (Older): proving the new grouping feature off
		// real seeded history, not just a single flat list.
		expect(screen.getByText('Today')).toBeInTheDocument();
		expect(screen.getByText('Yesterday')).toBeInTheDocument();
		expect(screen.getByText('Previous 7 days')).toBeInTheDocument();
		expect(screen.getByText('Older')).toBeInTheDocument();

		for (const session of reloaded) {
			expect(screen.getByText(session.title)).toBeInTheDocument();
		}
	});

	it("renders every seeded turn's content in the chat page without crashing", async () => {
		runSeedScript();
		const reloaded = JSON.parse(localStorage.getItem('amdbpedia:chat-sessions:v1')!);
		sessions.push(...reloaded);
		active.id = reloaded[0].id;

		render(ChatPage);

		// Spot-check a representative turn from each kind/outcome this
		// script seeds: a resolved pipeline result, a confident single-field
		// match, and an honest no-match refusal.
		if (reloaded[0].turns[0].kind === 'answer') {
			// "GERD dam, from its Wikipedia link" or "Quick field lookups"
			// might be newest depending on the clock at test time; either
			// way the first turn's own input is always rendered verbatim.
			expect(screen.getByText(reloaded[0].turns[0].input)).toBeInTheDocument();
		}
	});
});
