import { describe, expect, it, beforeEach, afterEach, vi } from 'vitest';
import { tick } from 'svelte';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/svelte';
import AppSidebar from './components/AppSidebar.svelte';
import {
	chat,
	clearChat,
	hasUnpublishedEdits,
	loadTurns,
	saveChat,
	type PipelineTurn
} from './chat.svelte';

// jsdom has no SvelteKit router; Clear chat only needs to ask for '/'.
vi.mock('$app/navigation', () => ({ goto: vi.fn() }));

function bridgeTurn(overrides: Partial<PipelineTurn> = {}): PipelineTurn {
	const mappings = [{ templateProperty: 'ርዝመት', ontologyProperty: 'length', confidence: 0.75 }];
	return {
		id: crypto.randomUUID(),
		kind: 'pipeline',
		input: '{{Infobox bridge}}',
		steps: [],
		mappings,
		edited: mappings.map((r) => ({ ...r })),
		running: false,
		...overrides
	};
}

describe('the single chat', () => {
	beforeEach(() => {
		chat.turns.length = 0;
		sessionStorage.clear();
		localStorage.clear();
	});

	afterEach(() => cleanup());

	it('survives a reload in the same tab and settles a run the reload cut off', () => {
		saveChat([bridgeTurn({ running: true, mappings: null, edited: undefined })]);
		const [turn] = loadTurns();
		expect(turn.running).toBe(false);
		expect(turn.error).toMatch(/ገጹ እንደገና ስለተጫነ ተቋርጧል/);
	});

	it('drops the old multi-chat history', () => {
		localStorage.setItem('amdbpedia:chat-sessions:v1', '[]');
		loadTurns();
		expect(localStorage.getItem('amdbpedia:chat-sessions:v1')).toBeNull();
	});

	it('knows when a card has edits nobody published', () => {
		const turn = bridgeTurn();
		expect(hasUnpublishedEdits([turn])).toBe(false);
		turn.edited![0].ontologyProperty = 'totalLength';
		expect(hasUnpublishedEdits([turn])).toBe(true);
		turn.published = { title: 'Mapping am:Infobox bridge', url: 'https://example.org' };
		expect(hasUnpublishedEdits([turn])).toBe(false);
	});

	it('does not flag a card whose current edits match the last download', () => {
		const turn = bridgeTurn();
		turn.edited![0].ontologyProperty = 'totalLength';
		expect(hasUnpublishedEdits([turn])).toBe(true);
		// User clicks Download → MappingCard records the snapshot.
		turn.downloadedSnapshot = JSON.stringify(turn.edited);
		expect(hasUnpublishedEdits([turn])).toBe(false);
		// A new edit after download re-flags the card.
		turn.edited![0].ontologyProperty = 'span';
		expect(hasUnpublishedEdits([turn])).toBe(true);
	});

	it('clearChat empties the chat and its saved copy', () => {
		chat.turns.push(bridgeTurn());
		saveChat();
		clearChat();
		expect(chat.turns).toHaveLength(0);
		expect(loadTurns()).toEqual([]);
	});
});

describe('sidebar', () => {
	beforeEach(() => {
		chat.turns.length = 0;
		localStorage.clear();
	});

	afterEach(() => cleanup());

	it('has no chat history, only Clear chat, Chat, Coverage and the usage guide', () => {
		render(AppSidebar, { props: { open: true } });
		expect(screen.getByRole('button', { name: /ውይይቱን አጽዳ/ })).toBeDisabled();
		expect(screen.getByTitle('ውይይት')).toBeInTheDocument();
		expect(screen.getByTitle('ሽፋን')).toBeInTheDocument();
		const guide = screen.getByTitle('የአጠቃቀም መመሪያ');
		expect(guide).toHaveAttribute('href', '/how-to-use.html');
		expect(screen.getByTitle('Documentation')).toHaveAttribute('href', '/documentation.html');
		// A static page: the client router must not handle it.
		expect(guide).toHaveAttribute('data-sveltekit-reload');
		expect(screen.queryByPlaceholderText(/search chats/i)).toBeNull();
		expect(screen.queryByText('New chat')).toBeNull();
	});

	it('clears straight away when nothing is unpublished', async () => {
		chat.turns.push(bridgeTurn());
		render(AppSidebar, { props: { open: true } });
		await tick();
		screen.getByRole('button', { name: /ውይይቱን አጽዳ/ }).click();
		await waitFor(() => expect(chat.turns).toHaveLength(0));
	});

	it('asks before throwing away unpublished edits', async () => {
		const turn = bridgeTurn();
		turn.edited![0].ontologyProperty = 'totalLength';
		chat.turns.push(turn);
		render(AppSidebar, { props: { open: true } });
		await tick();

		screen.getByRole('button', { name: /ውይይቱን አጽዳ/ }).click();
		expect(await screen.findByText('ውይይቱ ይጽዳ?')).toBeInTheDocument();
		expect(chat.turns).toHaveLength(1);

		await fireEvent.click(screen.getByRole('button', { name: 'ማረሙን ቀጥል' }));
		await waitFor(() => expect(screen.queryByText('ውይይቱ ይጽዳ?')).toBeNull());
		expect(chat.turns).toHaveLength(1);
	});

	it('collapsing hides the labels but keeps every button reachable', async () => {
		render(AppSidebar, { props: { open: true } });
		expect(screen.getByText('ሽፋን')).toBeInTheDocument();

		screen.getByRole('button', { name: 'የጎን ማውጫውን ሰብስብ' }).click();
		await tick();
		expect(screen.queryByText('ሽፋን')).toBeNull();
		expect(screen.getByTitle('ሽፋን')).toBeInTheDocument();
		expect(screen.getByTitle('ውይይቱን አጽዳ')).toBeInTheDocument();

		screen.getByRole('button', { name: 'የጎን ማውጫውን ዘርጋ' }).click();
		await tick();
		expect(screen.getByText('ሽፋን')).toBeInTheDocument();
	});
});
