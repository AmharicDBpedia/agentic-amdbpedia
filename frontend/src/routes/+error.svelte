<script lang="ts">
	// Root error page. It renders inside +layout.svelte, so the sidebar
	// (Clear chat, Chat, Coverage) stays available on a 404.
	import { page } from '$app/state';
	import { resolve } from '$app/paths';
	import { Button } from '$lib/components/ui/button/index.js';
	import Fa from 'svelte-fa';
	import { faArrowLeft, faComments, faChartSimple } from '@fortawesome/free-solid-svg-icons';

	const notFound = $derived(page.status === 404);
	// /review was removed when mappings moved to "edit and publish in the
	// chat"; old bookmarks land here.
	const wasReviewQueue = $derived(/\/review(\/|$)/.test(page.url.pathname));
</script>

<svelte:head>
	<title>{notFound ? 'ገጹ አልተገኘም' : 'ስህተት'} · agentic-amdbpedia</title>
</svelte:head>

<div class="flex h-full items-center justify-center overflow-y-auto px-4 py-10">
	<div class="w-full max-w-md text-center">
		<p class="text-6xl font-semibold tracking-tight text-muted-foreground/40 tabular-nums">
			{page.status}
		</p>
		<h1 class="mt-3 text-2xl font-semibold tracking-tight">
			{notFound ? 'ገጹ አልተገኘም' : 'የሆነ ችግር ተፈጥሯል'}
		</h1>
		<p class="mt-2 text-sm text-muted-foreground">
			{#if wasReviewQueue}
				የግምገማ ወረፋው ተወግዷል። ዝምድናዎች አሁን በውይይቱ ውስጥ በቀጥታ ይታረማሉ፣ ይታተማሉም።
			{:else if notFound}
				<span class="font-mono break-all">{page.url.pathname}</span> የሚባል ገጽ የለም።
			{:else}
				{page.error?.message ?? 'ያልተጠበቀ ስህተት ተፈጥሯል።'}
			{/if}
		</p>

		<div class="mt-6 flex flex-wrap justify-center gap-2">
			<Button href={resolve('/')} class="gap-2">
				<Fa icon={faComments} class="size-3.5" />
				ወደ ውይይቱ
			</Button>
			<Button href={resolve('/coverage')} variant="outline" class="gap-2">
				<Fa icon={faChartSimple} class="size-3.5" />
				ሽፋን
			</Button>
			<Button variant="ghost" class="gap-2" onclick={() => history.back()}>
				<Fa icon={faArrowLeft} class="size-3.5" />
				ተመለስ
			</Button>
		</div>
	</div>
</div>
