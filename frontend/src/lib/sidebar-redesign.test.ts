import { describe, expect, it, beforeEach, afterEach, vi } from 'vitest';
import { tick } from 'svelte';
import { cleanup, render, screen, waitFor } from '@testing-library/svelte';
import AppSidebar from './components/AppSidebar.svelte';
import {
	sessions,
	active,
	ensureActiveSession,
	startNewChat,
	touchSession,
	type PipelineTurn
} from './chat.svelte';

function makeTurn(input: string): PipelineTurn {
	return {
		id: crypto.randomUUID(),
		kind: 'pipeline',
		input,
		steps: [],
		mappings: null,
		running: false
	};
}

describe('sidebar redesign', () => {
	beforeEach(() => {
		sessions.length = 0;
		active.id = null;
		localStorage.clear();
	});

	afterEach(() => {
		cleanup();
		vi.restoreAllMocks();
	});

	it('search filters the chat list by title, live', async () => {
		render(AppSidebar, { props: { open: true } });

		const first = ensureActiveSession();
		first.turns.push(makeTurn('{{Infobox bridge | ርዝመት = 1,700 ሜትር}}'));
		touchSession(first.id);
		await tick();

		// startNewChat() is what the sidebar's own "New chat" button calls --
		// ensureActiveSession() alone would just keep returning the same
		// (already-populated) session.
		startNewChat();
		const second = ensureActiveSession();
		second.turns.push(makeTurn('አይካኦ_ኮድ'));
		touchSession(second.id);
		await tick();

		expect(screen.getByText(/Infobox bridge/)).toBeInTheDocument();
		expect(screen.getByText('አይካኦ_ኮድ')).toBeInTheDocument();

		const searchBox = screen.getByPlaceholderText('Search chats…') as HTMLInputElement;
		searchBox.value = 'አይካኦ';
		searchBox.dispatchEvent(new Event('input', { bubbles: true }));
		await tick();

		expect(screen.getByText('አይካኦ_ኮድ')).toBeInTheDocument();
		expect(screen.queryByText(/Infobox bridge/)).toBeNull();
	});

	it('deleting a chat requires confirmation -- the session survives an unconfirmed click', async () => {
		render(AppSidebar, { props: { open: true } });

		const session = ensureActiveSession();
		session.turns.push(makeTurn('{{Infobox bridge | ርዝመት = 1,700 ሜትር}}'));
		touchSession(session.id);
		await tick();

		expect(sessions).toHaveLength(1);

		const deleteButton = screen.getByRole('button', { name: 'Delete chat' });
		deleteButton.click();
		await tick();

		// A click alone must not have deleted anything yet -- only the
		// confirmation dialog's own Delete action does.
		expect(sessions).toHaveLength(1);
		expect(await screen.findByText('Delete this chat?')).toBeInTheDocument();

		const confirmButton = screen.getByRole('button', { name: 'Delete' });
		confirmButton.click();

		await waitFor(() => {
			expect(sessions).toHaveLength(0);
		});
	});

	it('collapsing the sidebar hides text labels but keeps the icons reachable', async () => {
		render(AppSidebar, { props: { open: true } });

		expect(screen.getByText('New chat')).toBeInTheDocument();
		expect(screen.getByText('Coverage')).toBeInTheDocument();

		const collapseButton = screen.getByRole('button', { name: 'Collapse sidebar' });
		collapseButton.click();
		await tick();

		expect(screen.queryByText('New chat')).toBeNull();
		expect(screen.queryByText('Coverage')).toBeNull();
		// The link itself -- and its accessible name via `title` -- must
		// still be there; collapsing removes the label text, not the nav.
		expect(screen.getByTitle('Coverage')).toBeInTheDocument();

		screen.getByRole('button', { name: 'Expand sidebar' }).click();
		await tick();

		expect(screen.getByText('New chat')).toBeInTheDocument();
	});
});
