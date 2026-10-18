<script lang="ts">
	import { page } from '$app/state';
	import { asset, resolve } from '$app/paths';
	import { goto } from '$app/navigation';
	import { browser } from '$app/environment';
	import { cn } from '$lib/utils.js';
	import { chat, clearChat, hasUnpublishedEdits } from '$lib/chat.svelte';
	import { Button } from '$lib/components/ui/button/index.js';
	import { Separator } from '$lib/components/ui/separator/index.js';
	import * as AlertDialog from '$lib/components/ui/alert-dialog/index.js';
	import logo from '$lib/assets/dbpedia-am-logo.png';
	import Fa from 'svelte-fa';
	import {
		faBook,
		faBroom,
		faChartSimple,
		faChevronLeft,
		faChevronRight,
		faComments
	} from '@fortawesome/free-solid-svg-icons';

	let { open = $bindable(true) }: { open?: boolean } = $props();

	// Icon-rail collapse (desktop only -- on mobile the sidebar is a
	// full-width overlay, collapsing it to icons there would just hide the
	// content behind a dim strip with no way back). A per-browser
	// convenience, so it fails quietly when site data is unavailable.
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

	const links = [
		{ href: resolve('/'), label: 'ውይይት', icon: faComments, page: false },
		{ href: resolve('/coverage'), label: 'ሽፋን', icon: faChartSimple, page: false },
		// The static usage guide (static/documentation.html), not an app
		// route: it needs a full page load, or the router shows the 404 page.
		{ href: asset('/documentation.html'), label: 'የአጠቃቀም መመሪያ', icon: faBook, page: true }
	];

	// Clearing drops edits nobody published, so that case asks first.
	let confirmClear = $state(false);

	function requestClear() {
		if (hasUnpublishedEdits()) {
			confirmClear = true;
			return;
		}
		clear();
	}

	function clear() {
		clearChat();
		confirmClear = false;
		open = false;
		if (page.url.pathname !== resolve('/')) goto(resolve('/'));
	}
</script>

{#if open}
	<button
		type="button"
		class="fixed inset-0 z-30 bg-black/50 md:hidden"
		onclick={() => (open = false)}
		aria-label="የጎን ማውጫውን ዝጋ"
	></button>
{/if}

<aside
	class={cn(
		'fixed inset-y-0 left-0 z-40 flex w-64 shrink-0 flex-col border-r bg-sidebar text-sidebar-foreground transition-[transform,width] md:relative md:translate-x-0',
		open ? 'translate-x-0' : '-translate-x-full',
		collapsed ? 'md:w-16' : 'md:w-64'
	)}
>
	<div class={cn('flex p-3.5', collapsed ? 'flex-col items-center gap-2' : 'items-center gap-2.5')}>
		<img src={logo} alt="የአማርኛ DBpedia" class="size-8 shrink-0 object-contain" />
		{#if !collapsed}
			<div class="min-w-0 flex-1">
				<p class="truncate text-sm font-semibold tracking-tight">agentic-amdbpedia</p>
				<p class="truncate text-xs text-sidebar-foreground/55">የአማርኛ → DBpedia ማዛመጃ</p>
			</div>
		{/if}
		<button
			type="button"
			class="hidden shrink-0 rounded-md p-1.5 text-sidebar-foreground/50 hover:bg-sidebar-accent/60 hover:text-sidebar-foreground md:block"
			onclick={() => (collapsed = !collapsed)}
			title={collapsed ? 'የጎን ማውጫውን ዘርጋ' : 'የጎን ማውጫውን ሰብስብ'}
			aria-label={collapsed ? 'የጎን ማውጫውን ዘርጋ' : 'የጎን ማውጫውን ሰብስብ'}
		>
			<Fa icon={collapsed ? faChevronRight : faChevronLeft} class="size-3.5" />
		</button>
	</div>

	<div class="px-2.5">
		<Button
			variant="outline"
			class={cn('w-full gap-2 bg-transparent', collapsed ? 'justify-center px-0' : 'justify-start')}
			title="ውይይቱን አጽዳ"
			disabled={chat.turns.length === 0}
			onclick={requestClear}
		>
			<Fa icon={faBroom} class="size-3.5 shrink-0" />
			{#if !collapsed}ውይይቱን አጽዳ{/if}
		</Button>
	</div>

	<Separator class="my-3" />

	<nav class="space-y-1 px-2.5">
		{#each links as link (link.href)}
			{@const isActive = page.url.pathname === link.href}
			<!-- eslint-disable svelte/no-navigation-without-resolve -- hrefs come from resolve() or asset() above -->
			<a
				href={link.href}
				data-sveltekit-reload={link.page || undefined}
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
			<!-- eslint-enable svelte/no-navigation-without-resolve -->
		{/each}
	</nav>

	<div class="flex-1"></div>
	{#if !collapsed}
		<p class="px-4 pb-4 text-xs leading-relaxed text-sidebar-foreground/45">
			ውይይቱ በዚህ ትር ውስጥ ብቻ ይቀመጣል። ትሩን ሲዘጉ ወይም ሲያጸዱት ይጠፋል፤ የታተመው ግን ዊኪው ላይ ይቆያል።
		</p>
	{/if}
</aside>

<AlertDialog.Root bind:open={confirmClear}>
	<AlertDialog.Content>
		<AlertDialog.Header>
			<AlertDialog.Title>ውይይቱ ይጽዳ?</AlertDialog.Title>
			<AlertDialog.Description>
				አንዳንድ ዝምድናዎች ያልታተሙ እርማቶች አሏቸው። ውይይቱን ማጽዳት እነዚያን እርማቶች ያጠፋል። ቀድሞ የታተመው ዊኪው ላይ ይቆያል።
			</AlertDialog.Description>
		</AlertDialog.Header>
		<AlertDialog.Footer>
			<AlertDialog.Cancel>ማረሙን ቀጥል</AlertDialog.Cancel>
			<AlertDialog.Action class="bg-destructive text-white hover:bg-destructive/90" onclick={clear}>
				ውይይቱን አጽዳ
			</AlertDialog.Action>
		</AlertDialog.Footer>
	</AlertDialog.Content>
</AlertDialog.Root>
