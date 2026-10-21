<script lang="ts">
	import { onMount } from 'svelte';
	import { BackendUnavailableError, getCoverageStats } from '$lib/api';
	import type { CoverageStats } from '$lib/types';
	import { Card, CardDescription, CardHeader, CardTitle } from '$lib/components/ui/card/index.js';
	import { Skeleton } from '$lib/components/ui/skeleton/index.js';
	import Fa from 'svelte-fa';
	import { faTriangleExclamation } from '@fortawesome/free-solid-svg-icons';

	let stats = $state<CoverageStats | null>(null);
	let error = $state<string | null>(null);

	async function load() {
		try {
			stats = await getCoverageStats();
		} catch (err) {
			console.error('Failed to load coverage stats:', err);
			error =
				err instanceof BackendUnavailableError
					? 'cross-lingual አይገኝም — የHTTP አገልጋዩ (just run-http) መሮጡን ያረጋግጡ።'
					: `የሽፋን መረጃውን ሲጫን ያልተጠበቀ ስህተት፦ ${err instanceof Error ? err.message : String(err)}`;
		}
	}

	onMount(load);
</script>

<div class="mx-auto h-full max-w-5xl overflow-y-auto px-4 py-6 md:px-8 md:py-10">
	<div class="mb-8">
		<h1 class="text-2xl font-semibold tracking-tight">Extraction coverage</h1>
		<p class="mt-1 text-sm text-muted-foreground">
			ይህ ሂደት ካስተናገዳቸው infobox templates ውስጥ ስንቶቹ mapping በቀጥታ ዊኪው ላይ እንደታተመ ያሳያል።
		</p>
	</div>

	{#if error}
		<div
			class="flex items-center gap-2 rounded-lg border border-warning/40 bg-warning/10 px-4 py-3 text-sm text-warning-foreground"
		>
			<Fa icon={faTriangleExclamation} class="size-4 shrink-0" />
			{error}
		</div>
	{:else if !stats}
		<div class="grid grid-cols-3 gap-4">
			<Skeleton class="h-24 w-full" />
			<Skeleton class="h-24 w-full" />
			<Skeleton class="h-24 w-full" />
		</div>
	{:else}
		<div class="grid grid-cols-3 gap-4">
			<Card>
				<CardHeader>
					<CardDescription>Templates</CardDescription>
					<CardTitle class="text-2xl tabular-nums">{stats.totalTemplates}</CardTitle>
				</CardHeader>
			</Card>
			<Card>
				<CardHeader>
					<CardDescription>የተዛመዱ</CardDescription>
					<CardTitle class="text-2xl tabular-nums">{stats.mappedTemplates}</CardTitle>
				</CardHeader>
			</Card>
			<Card>
				<CardHeader>
					<CardDescription>ሽፋን</CardDescription>
					<CardTitle class="text-2xl tabular-nums">{stats.coveragePercent.toFixed(1)}%</CardTitle>
				</CardHeader>
			</Card>
		</div>
	{/if}
</div>
