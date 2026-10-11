<script lang="ts">
	import { page } from '$app/state';
	import { resolve } from '$app/paths';
	import { browser } from '$app/environment';
	import { cn } from '$lib/utils.js';
	import { sessions, active, startNewChat, selectSession, deleteSession } from '$lib/chat.svelte';
	import { Button } from '$lib/components/ui/button/index.js';
	import { Input } from '$lib/components/ui/input/index.js';
	import { Separator } from '$lib/components/ui/separator/index.js';
	import * as AlertDialog from '$lib/components/ui/alert-dialog/index.js';
	import logo from '$lib/assets/dbpedia-am-logo.png';
	import Fa from 'svelte-fa';
	import {
		faPlus,
		faChartSimple,
		faTrash,
		faMagnifyingGlass,
		faChevronLeft,
		faChevronRight
	} from '@fortawesome/free-solid-svg-icons';

	let { open = $bindable(true) }: { open?: boolean } = $props();

	// Icon-rail collapse (desktop only -- on mobile the sidebar is a
	// full-width overlay, collapsing it to icons there would just hide the
	// content behind a dim strip with no way back). Persisted the same
	// browser-guarded, fail-quiet way chat.svelte.ts already persists chat
	// history: a per-browser convenience, not data that needs to survive a
	// cleared profile.
	const COLLAPSE_KEY = 'amdbpedia:sidebar-collapsed:v1';

	function loadCollapsed(): boolean {
		if (!browser) return false;
		try {
			return localStorage.getItem(COLLAPSE_KEY) === '1';
		} catch {
			return false;
		}
	}

	let collapsed = $state(loadCollapsed());

	$effect(() => {
		const value = collapsed;
		if (!browser) return;
		try {
			localStorage.setItem(COLLAPSE_KEY, value ? '1' : '0');
		} catch {
			// Private browsing / disabled site data -- collapse state just
			// won't survive a reload.
		}
	});

	const links = [{ href: resolve('/coverage'), label: 'Coverage', icon: faChartSimple }];

	const isChatRoute = $derived(page.url.pathname === resolve('/'));

	function relativeTime(ms: number): string {
		const diff = Date.now() - ms;
		const minute = 60_000;
		const hour = 60 * minute;
		const day = 24 * hour;
		if (diff < minute) return 'just now';
		if (diff < hour) return `${Math.floor(diff / minute)}m ago`;
		if (diff < day) return `${Math.floor(diff / hour)}h ago`;
		if (diff < 7 * day) return `${Math.floor(diff / day)}d ago`;
		return new Date(ms).toLocaleDateString();
	}

	let query = $state('');
	const filteredSessions = $derived.by(() => {
		const needle = query.trim().toLowerCase();
		return needle ? sessions.filter((s) => s.title.toLowerCase().includes(needle)) : sessions;
	});

	// Every mainstream chat product (ChatGPT, Claude, Gemini) groups
	// history the same way -- a flat list stops being scannable once there
	// are more than a handful of sessions. `sessions` is already
	// newest-first (chat.svelte.ts's touchSession floats the most recently
	// touched one to the top), so a single pass preserves that order
	// within each bucket.
	const groupedSessions = $derived.by(() => {
		const nowDate = new Date();
		// A fresh Date built from y/m/d, not `.setHours()` mutating an
		// existing instance -- eslint's svelte/prefer-svelte-reactivity
		// flags the latter as a stray mutable Date a component shouldn't
		// hold onto; this is a one-shot read, never retained.
		const startOfToday = new Date(
			nowDate.getFullYear(),
			nowDate.getMonth(),
			nowDate.getDate()
		).getTime();
		const startOfYesterday = startOfToday - 86_400_000;
		const sevenDaysAgo = startOfToday - 7 * 86_400_000;
		const buckets: { label: string; sessions: typeof sessions }[] = [
			{ label: 'Today', sessions: [] },
			{ label: 'Yesterday', sessions: [] },
			{ label: 'Previous 7 days', sessions: [] },
			{ label: 'Older', sessions: [] }
		];
		for (const session of filteredSessions) {
			if (session.updatedAt >= startOfToday) buckets[0].sessions.push(session);
			else if (session.updatedAt >= startOfYesterday) buckets[1].sessions.push(session);
			else if (session.updatedAt >= sevenDaysAgo) buckets[2].sessions.push(session);
			else buckets[3].sessions.push(session);
		}
		return buckets.filter((bucket) => bucket.sessions.length > 0);
	});

	let confirmDeleteId = $state<string | null>(null);
	const confirmDeleteSession = $derived(sessions.find((s) => s.id === confirmDeleteId));

	function requestDelete(event: MouseEvent, id: string) {
		event.preventDefault();
		event.stopPropagation();
		confirmDeleteId = id;
	}

	function confirmDelete(event: Event) {
		event.preventDefault();
		if (confirmDeleteId) deleteSession(confirmDeleteId);
		confirmDeleteId = null;
	}
</script>

{#if open}
	<button
		type="button"
		class="fixed inset-0 z-30 bg-black/50 md:hidden"
		onclick={() => (open = false)}
		aria-label="Close sidebar"
	></button>
{/if}

<aside
	class={cn(
		'fixed inset-y-0 left-0 z-40 flex w-72 shrink-0 flex-col border-r bg-sidebar text-sidebar-foreground transition-[transform,width] md:relative md:translate-x-0',
		open ? 'translate-x-0' : '-translate-x-full',
		collapsed ? 'md:w-16' : 'md:w-72'
	)}
>
	<div class={cn('flex p-3.5', collapsed ? 'flex-col items-center gap-2' : 'items-center gap-2.5')}>
		<img src={logo} alt="Amharic DBpedia" class="size-8 shrink-0 object-contain" />
		{#if !collapsed}
			<div class="min-w-0 flex-1">
				<p class="truncate text-sm font-semibold tracking-tight">agentic-amdbpedia</p>
				<p class="truncate text-xs text-sidebar-foreground/55">Amharic → DBpedia mapping</p>
			</div>
		{/if}
		<button
			type="button"
			class="hidden shrink-0 rounded-md p-1.5 text-sidebar-foreground/50 hover:bg-sidebar-accent/60 hover:text-sidebar-foreground md:block"
			onclick={() => (collapsed = !collapsed)}
			title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
			aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
		>
			<Fa icon={collapsed ? faChevronRight : faChevronLeft} class="size-3.5" />
		</button>
	</div>

	<div class="px-2.5">
		<Button
			variant="outline"
			class={cn('w-full gap-2 bg-transparent', collapsed ? 'justify-center px-0' : 'justify-start')}
			href={resolve('/')}
			title="New chat"
			onclick={() => {
				startNewChat();
				open = false;
			}}
		>
			<Fa icon={faPlus} class="size-3.5 shrink-0" />
			{#if !collapsed}New chat{/if}
		</Button>
	</div>

	<Separator class="my-3" />

	<nav class="space-y-1 px-2.5">
		{#each links as link (link.href)}
			{@const isActive = page.url.pathname === link.href}
			<a
				href={link.href}
				onclick={() => (open = false)}
				title={link.label}
				class={cn(
					'relative flex items-center gap-2.5 rounded-md px-2.5 py-2 text-sm transition-colors',
					collapsed && 'justify-center',
					isActive
						? 'bg-sidebar-accent font-medium text-sidebar-accent-foreground'
						: 'text-sidebar-foreground/80 hover:bg-sidebar-accent/60 hover:text-sidebar-accent-foreground'
				)}
			>
				<Fa icon={link.icon} class="size-3.5 shrink-0" />
				{#if !collapsed}
					<span class="flex-1">{link.label}</span>
				{/if}
			</a>
		{/each}
	</nav>

	{#if !collapsed}
		{#if sessions.length > 0}
			<Separator class="my-3" />

			<div class="px-2.5 pb-2">
				<div class="relative">
					<Fa
						icon={faMagnifyingGlass}
						class="pointer-events-none absolute top-1/2 left-2.5 size-3 -translate-y-1/2 text-sidebar-foreground/40"
					/>
					<Input
						type="search"
						bind:value={query}
						placeholder="Search chats…"
						class="h-8 border-sidebar-border bg-transparent pl-7 text-sm shadow-none"
					/>
				</div>
			</div>

			<div class="min-h-0 flex-1 space-y-3 overflow-y-auto px-2.5 pb-3">
				{#each groupedSessions as group (group.label)}
					<div>
						<p
							class="px-2 pb-1 text-xs font-medium tracking-wide text-sidebar-foreground/50 uppercase"
						>
							{group.label}
						</p>
						<div class="space-y-0.5">
							{#each group.sessions as session (session.id)}
								{@const isActive = isChatRoute && active.id === session.id}
								<div
									class={cn(
										'group flex items-center rounded-md',
										isActive
											? 'bg-sidebar-accent text-sidebar-accent-foreground'
											: 'hover:bg-sidebar-accent/60'
									)}
								>
									<a
										href={resolve('/')}
										onclick={() => {
											selectSession(session.id);
											open = false;
										}}
										class="min-w-0 flex-1 px-2.5 py-2 text-sm"
									>
										<p class="truncate" title={session.title}>{session.title}</p>
										<p class="text-xs text-sidebar-foreground/45">
											{relativeTime(session.updatedAt)}
										</p>
									</a>
									<button
										type="button"
										class="mr-1.5 shrink-0 rounded p-1.5 text-sidebar-foreground/40 opacity-0 group-hover:opacity-100 hover:text-destructive"
										onclick={(event) => requestDelete(event, session.id)}
										aria-label="Delete chat"
										title="Delete chat"
									>
										<Fa icon={faTrash} class="size-3" />
									</button>
								</div>
							{/each}
						</div>
					</div>
				{/each}
				{#if groupedSessions.length === 0}
					<p class="px-2.5 py-4 text-center text-xs text-sidebar-foreground/45">
						No chats match "{query}"
					</p>
				{/if}
			</div>
		{:else}
			<div class="flex-1"></div>
		{/if}
	{:else}
		<div class="flex-1"></div>
	{/if}
</aside>

<AlertDialog.Root
	open={confirmDeleteId !== null}
	onOpenChange={(isOpen) => {
		if (!isOpen) confirmDeleteId = null;
	}}
>
	<AlertDialog.Content>
		<AlertDialog.Header>
			<AlertDialog.Title>Delete this chat?</AlertDialog.Title>
			<AlertDialog.Description>
				{#if confirmDeleteSession}
					<span class="font-medium text-foreground">{confirmDeleteSession.title}</span> and its {confirmDeleteSession
						.turns.length}
					message{confirmDeleteSession.turns.length === 1 ? '' : 's'} will be removed from this browser.
					This can't be undone. Anything already published is unaffected — deleting the chat doesn't delete
					what it submitted.
				{/if}
			</AlertDialog.Description>
		</AlertDialog.Header>
		<AlertDialog.Footer>
			<AlertDialog.Cancel>Cancel</AlertDialog.Cancel>
			<AlertDialog.Action
				class="bg-destructive text-white hover:bg-destructive/90"
				onclick={confirmDelete}
			>
				Delete
			</AlertDialog.Action>
		</AlertDialog.Footer>
	</AlertDialog.Content>
</AlertDialog.Root>
