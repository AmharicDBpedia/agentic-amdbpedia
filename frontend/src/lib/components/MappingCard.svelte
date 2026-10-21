<script lang="ts">
	// One infobox result: every field in a single table (mapped, unmapped
	// and removed), with each change marked and a footer to publish.
	import { BackendUnavailableError, findSemanticMatch } from '$lib/api';
	import type { PipelineTurn } from '$lib/chat.svelte';
	import { changesFor, originalFor, removedRows, rowChange } from '$lib/mapping-diff';
	import { buildTemplateMapping } from '$lib/mapping-wikitext';
	import type { MappingCandidate, PredictedMapping, UnmappedField } from '$lib/types';
	import { cn } from '$lib/utils.js';
	import { Button } from '$lib/components/ui/button/index.js';
	import { Input } from '$lib/components/ui/input/index.js';
	import ConfidencePill from './ConfidencePill.svelte';
	import Fa from 'svelte-fa';
	import {
		faCheck,
		faCircleCheck,
		faDownload,
		faMagnifyingGlass,
		faPen,
		faPlus,
		faRotateLeft,
		faSpinner,
		faTrash
	} from '@fortawesome/free-solid-svg-icons';

	let { turn }: { turn: PipelineTurn } = $props();

	const original = $derived(turn.mappings ?? []);
	const rows = $derived(turn.edited ?? []);
	const readonly = $derived(Boolean(turn.published));
	const changes = $derived(changesFor(original, rows));
	const removed = $derived(removedRows(original, rows));
	const unmapped = $derived(
		(turn.unmappedFields ?? []).filter(
			(field) => !rows.some((row) => row.templateProperty === field.name)
		)
	);
	const domainClass = $derived(turn.domainClass ?? turn.targetClass ?? 'Thing');
	const canDownload = $derived(rows.length > 0 && !readonly);
	const downloadName = $derived(
		`Mapping_am-${(turn.templateName ?? 'template').replace(/\s+/g, '_')}.wiki`
	);

	const changeSummary = $derived(
		[
			changes.edited && `${changes.edited} ታርሟል`,
			changes.added && `${changes.added} ተጨምሯል`,
			changes.removed && `${changes.removed} ተወግዷል`
		]
			.filter(Boolean)
			.join(' · ')
	);

	// A property the user chose counts as confirmed (100%); typing the
	// model's own answer back in is not a change and keeps its confidence.
	function setProperty(row: PredictedMapping, value: string) {
		const source = originalFor(original, row.templateProperty);
		row.ontologyProperty = value;
		row.confidence = source?.ontologyProperty === value ? source.confidence : 1;
	}

	function undoRow(row: PredictedMapping) {
		const source = originalFor(original, row.templateProperty);
		if (!source) return;
		row.ontologyProperty = source.ontologyProperty;
		row.confidence = source.confidence;
		if (source.source) row.source = source.source;
		else delete row.source;
	}

	function removeRow(row: PredictedMapping) {
		turn.edited = rows.filter((r) => r !== row);
	}

	function confirmRow(row: PredictedMapping) {
		row.ack = 'confirmed';
	}

	// Rows are read-only until the pencil opens edit mode for one row at a
	// time; the tick closes it (and confirms the pick). The search panel is
	// only reachable while a row is being edited.
	let editing = $state<string | null>(null);

	function startEdit(row: PredictedMapping) {
		editing = row.templateProperty;
	}

	function stopEdit(row: PredictedMapping) {
		if (editing === row.templateProperty) editing = null;
		openSearch = null;
		confirmRow(row);
	}

	function restoreRow(row: PredictedMapping) {
		turn.edited = [...rows, { ...row }];
	}

	function undoAll() {
		turn.edited = original.map((row) => ({ ...row }));
		openSearch = null;
		editing = null;
	}

	function assign(field: string, property: string, confidence: number) {
		const name = property.trim();
		if (!name) return;
		turn.edited = [
			...rows,
			{ templateProperty: field, ontologyProperty: name, confidence, source: 'manual' }
		];
		delete drafts[field];
		openSearch = null;
	}

	// What the user typed into an unmapped field's property box.
	let drafts = $state<Record<string, string>>({});

	// One ontology search panel open at a time, keyed by field name. Picks
	// come from real ontology candidates instead of a freehand guess.
	let openSearch = $state<string | null>(null);
	let searchQuery = $state('');
	let searchResults = $state<MappingCandidate[] | null>(null);
	let searching = $state(false);
	let searchError = $state<string | null>(null);

	function toggleSearch(field: string) {
		if (openSearch === field) {
			openSearch = null;
			return;
		}
		openSearch = field;
		searchQuery = field;
		searchResults = null;
		searchError = null;
		runSearch();
	}

	async function runSearch() {
		const query = searchQuery.trim();
		if (!query) return;
		searching = true;
		searchError = null;
		try {
			const result = await findSemanticMatch(query, domainClass);
			searchResults = result.status === 'no_match' ? [] : result.matches;
		} catch (err) {
			searchError =
				err instanceof BackendUnavailableError
					? 'አገልጋዩ አሁን አይገኝም።'
					: `ፍለጋው አልተሳካም፦ ${err instanceof Error ? err.message : String(err)}`;
		} finally {
			searching = false;
		}
	}

	function pick(field: string, candidate: MappingCandidate) {
		const row = rows.find((r) => r.templateProperty === field);
		if (row) {
			setProperty(row, candidate.property);
			openSearch = null;
		} else {
			assign(field, candidate.property, candidate.score);
		}
	}

	let newField = $state('');
	let newProperty = $state('');

	function addRow() {
		const field = newField.trim();
		if (!field || !newProperty.trim() || rows.some((r) => r.templateProperty === field)) return;
		assign(field, newProperty, 1);
		newField = '';
		newProperty = '';
	}

	// Live publishing is wired in the backend (POST /v1/publish) but gated
	// off from the UI for now -- the reviewer downloads the generated
	// {{TemplateMapping ...}} wikitext and pastes it into the wiki page by
	// hand. The publish button will come back later.
	function download() {
		if (!canDownload) return;
		const wikitext = buildTemplateMapping(domainClass, rows);
		const blob = new Blob([wikitext], { type: 'text/plain;charset=utf-8' });
		const url = URL.createObjectURL(blob);
		const link = document.createElement('a');
		link.href = url;
		link.download = downloadName;
		document.body.appendChild(link);
		link.click();
		link.remove();
		URL.revokeObjectURL(url);
	}
</script>

{#snippet searchPanel(field: string)}
	<div class="mt-2 rounded-lg border bg-background p-2">
		<form
			class="flex gap-1.5"
			onsubmit={(e) => {
				e.preventDefault();
				runSearch();
			}}
		>
			<Input bind:value={searchQuery} placeholder="የአማርኛ መስክ ወይም የእንግሊዝኛ ፍንጭ" class="h-7 text-xs" />
			<Button type="submit" size="sm" class="h-7" disabled={searching}>
				{#if searching}
					<Fa icon={faSpinner} class="size-3.5 animate-spin" />
				{:else}
					ፈልግ
				{/if}
			</Button>
		</form>
		{#if searchError}
			<p class="mt-1.5 text-xs text-destructive">{searchError}</p>
		{:else if searchResults && searchResults.length === 0}
			<p class="mt-1.5 text-xs text-muted-foreground">እርግጠኛ የኦንቶሎጂ ዝምድና የለም።</p>
		{:else if searchResults}
			<ul class="mt-1.5 flex flex-wrap gap-1.5">
				{#each searchResults as candidate (candidate.property)}
					<li>
						<button
							type="button"
							class="flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs hover:border-primary/50 hover:bg-accent"
							onclick={() => pick(field, candidate)}
						>
							<span class="font-mono">{candidate.property}</span>
							<ConfidencePill confidence={candidate.score} />
						</button>
					</li>
				{/each}
			</ul>
		{/if}
	</div>
{/snippet}

{#snippet searchButton(field: string)}
	<Button
		variant="outline"
		size="icon"
		class="size-8 shrink-0"
		onclick={() => toggleSearch(field)}
		title="ኦንቶሎጂውን ፈልግ"
		aria-label="ለ{field} ኦንቶሎጂውን ፈልግ"
	>
		<Fa icon={faMagnifyingGlass} class="size-3.5" />
	</Button>
{/snippet}

{#snippet fieldCell(name: string, value?: string)}
	<p class="font-mono text-sm break-all">{name}</p>
	{#if value}
		<p class="mt-0.5 truncate text-xs text-muted-foreground" title={value}>= {value}</p>
	{/if}
{/snippet}

<section
	class={cn(
		'overflow-hidden rounded-xl border bg-card shadow-sm',
		changes.total > 0 && !readonly && 'border-warning/60',
		readonly && 'border-success/50'
	)}
>
	<header
		class="flex flex-col items-start gap-x-3 gap-y-2 border-b bg-muted/40 px-3 py-3 sm:flex-row sm:items-center sm:px-4"
	>
		<div class="w-full min-w-0 sm:w-auto sm:flex-1">
			<p class="text-sm font-semibold break-words">
				{#if turn.templateName}
					<span class="font-mono">Mapping am:{turn.templateName}</span>
				{:else}
					ዝምድና
				{/if}
				<span class="font-normal text-muted-foreground">→ {domainClass}</span>
			</p>
			{#if !readonly}
				<p class="mt-0.5 flex items-center gap-1.5 text-xs text-muted-foreground">
					<Fa icon={faPen} class="size-2.5" />
					ከታች ያሉት ባህርያት በሙሉ ሊታረሙ ይችላሉ። ለውጦች እስኪታተሙ ድረስ ምልክት ይደረግባቸዋል።
				</p>
			{/if}
		</div>
		{#if readonly}
			<span
				class="flex items-center gap-1.5 rounded-full bg-success/15 px-2.5 py-1 text-xs font-medium text-success"
			>
				<Fa icon={faCircleCheck} class="size-3" />
				ታትሟል
			</span>
		{:else}
			<span
				class={cn(
					'rounded-full px-2.5 py-1 text-xs font-medium',
					changes.total > 0
						? 'bg-warning/15 text-warning-foreground dark:text-warning'
						: 'bg-muted text-muted-foreground'
				)}
				data-testid="card-status"
			>
				ረቂቅ{changeSummary ? ` · ${changeSummary}` : ''}
			</span>
		{/if}
	</header>

	<div class="overflow-x-auto">
		<table class="w-full text-sm">
			<thead>
				<tr class="border-b text-left text-xs text-muted-foreground">
					<th class="w-[30%] px-3 py-2 font-medium sm:w-[38%] sm:px-4">የመረጃ ሳጥን መስክ</th>
					<th class="px-3 py-2 font-medium sm:px-4">DBpedia ባህርይ</th>
					{#if !readonly}<th class="w-10 px-2 py-2"><span class="sr-only">ተግባራት</span></th>{/if}
				</tr>
			</thead>
			<tbody>
				{#each rows as row, i (i)}
					{@const change = rowChange(original, row)}
					{@const aiPick = row.source === 'llm' && change === null}
					{@const isEditing = editing === row.templateProperty && !readonly}
					<tr
						class={cn(
							'border-b border-l-4 align-top last:border-b-0',
							change === 'edited' && 'border-l-warning bg-warning/5',
							change === 'added' && 'border-l-success bg-success/5',
							change === null && 'border-l-transparent'
						)}
						data-change={change ?? 'none'}
					>
						<td class="px-3 py-2.5 sm:px-4">{@render fieldCell(row.templateProperty)}</td>
						<td class="px-3 py-2.5 sm:px-4">
							{#if !isEditing}
								<span class="font-mono">{row.ontologyProperty}</span>
							{:else}
								<div class="flex items-center gap-1.5">
									<div class="relative min-w-0 flex-1">
										<Input
											value={row.ontologyProperty}
											oninput={(e) => setProperty(row, e.currentTarget.value)}
											aria-label="DBpedia ባህርይ ለ{row.templateProperty}"
											class={cn(
												'h-8 pr-7 font-mono',
												change === 'edited' && 'border-warning focus-visible:ring-warning/40',
												aiPick && row.ack !== 'confirmed' && 'border-dashed border-warning'
											)}
										/>
										<Fa
											icon={faPen}
											class="pointer-events-none absolute top-1/2 right-2.5 size-2.5 -translate-y-1/2 text-muted-foreground/60"
										/>
									</div>
									{@render searchButton(row.templateProperty)}
								</div>
							{/if}
							<div class="mt-1 flex flex-wrap items-center gap-x-2 gap-y-1 text-xs">
								{#if change === 'edited'}
									<span class="text-warning-foreground dark:text-warning">
										የቀድሞው፦
										<span class="font-mono line-through"
											>{originalFor(original, row.templateProperty)?.ontologyProperty}</span
										>
									</span>
									{#if !readonly}
										<button
											type="button"
											class="font-medium underline underline-offset-2 hover:no-underline"
											onclick={() => undoRow(row)}>መልስ</button
										>
									{/if}
								{:else if change === 'added'}
									<span class="text-success">በእርስዎ የተጨመረ</span>
								{:else}
									{#if row.ack === 'confirmed'}
										<span class="flex items-center gap-1 text-success">
											<Fa icon={faCheck} class="size-3" />
											ተረጋግጧል
										</span>
									{:else if aiPick}
										<span
											class="rounded bg-warning/15 px-1 py-0.5 font-medium text-warning-foreground dark:text-warning"
											title="ፍለጋው እርግጠኛ ውጤት ስላላገኘ ተመረጠ።">recommended</span
										>
									{:else if row.source === 'learned'}
										<span
											class="rounded bg-accent px-1 py-0.5 font-medium text-accent-foreground"
											title="ለዚሁ መስክና ክፍል ቀደም ሲል ታትሟል።">የተማረ</span
										>
									{/if}
									<ConfidencePill confidence={row.confidence} />
								{/if}
							</div>
							{#if openSearch === row.templateProperty && isEditing}
								{@render searchPanel(row.templateProperty)}
							{/if}
						</td>
						{#if !readonly}
							<td class="px-2 py-2.5">
								<div class="flex items-center gap-1">
									{#if isEditing}
										<Button
											variant="ghost"
											size="icon"
											class="size-8 text-muted-foreground hover:text-success"
											onclick={() => stopEdit(row)}
											title="አርትዖትን ጨርስ"
											aria-label="{row.templateProperty} አርትዖትን ጨርስ"
										>
											<Fa icon={faCheck} class="size-3.5" />
										</Button>
									{:else}
										<Button
											variant="ghost"
											size="icon"
											class="size-8 text-muted-foreground hover:text-foreground"
											onclick={() => startEdit(row)}
											title="አርትዕ"
											aria-label="{row.templateProperty} አርትዕ"
										>
											<Fa icon={faPen} class="size-3.5" />
										</Button>
									{/if}
									<Button
										variant="ghost"
										size="icon"
										class="size-8 text-muted-foreground hover:text-destructive"
										onclick={() => removeRow(row)}
										title="ይህን መስክ አስወግድ"
										aria-label="{row.templateProperty} አስወግድ"
									>
										<Fa icon={faTrash} class="size-3.5" />
									</Button>
								</div>
							</td>
						{/if}
					</tr>
				{/each}

				{#if !readonly}
					{#each unmapped as field, i (i)}
						{@render unmappedRow(field)}
					{/each}

					{#each removed as row, i (i)}
						<tr class="border-b border-l-4 border-l-destructive/50 bg-muted/30 align-top">
							<td class="px-3 py-2.5 text-muted-foreground line-through sm:px-4">
								{@render fieldCell(row.templateProperty)}
							</td>
							<td class="px-3 py-2.5 sm:px-4">
								<span class="font-mono text-muted-foreground line-through"
									>{row.ontologyProperty}</span
								>
								<p class="mt-1 text-xs text-muted-foreground">ተወግዷል፣ አይታተምም</p>
							</td>
							<td class="px-2 py-2.5">
								<Button
									variant="ghost"
									size="icon"
									class="size-8 text-muted-foreground hover:text-foreground"
									onclick={() => restoreRow(row)}
									title="ወደነበረበት መልስ"
									aria-label="{row.templateProperty} ወደነበረበት መልስ"
								>
									<Fa icon={faRotateLeft} class="size-3.5" />
								</Button>
							</td>
						</tr>
					{/each}

					<tr class="border-t border-dashed">
						<td class="px-3 py-2.5 sm:px-4">
							<Input
								bind:value={newField}
								placeholder="+ ሌላ የመረጃ ሳጥን መስክ"
								aria-label="አዲስ የመረጃ ሳጥን መስክ"
								class="h-8 border-dashed font-mono text-xs placeholder:font-sans"
							/>
						</td>
						<td class="px-3 py-2.5 sm:px-4">
							<Input
								bind:value={newProperty}
								placeholder="DBpedia ባህርይ"
								aria-label="አዲስ DBpedia ባህርይ"
								class="h-8 border-dashed font-mono text-xs placeholder:font-sans"
								onkeydown={(e) => {
									if (e.key === 'Enter') addRow();
								}}
							/>
						</td>
						<td class="px-2 py-2.5">
							<Button
								variant="outline"
								size="icon"
								class="size-8"
								disabled={!newField.trim() || !newProperty.trim()}
								onclick={addRow}
								title="መስክ ጨምር"
								aria-label="መስክ ጨምር"
							>
								<Fa icon={faPlus} class="size-3.5" />
							</Button>
						</td>
					</tr>
				{/if}
			</tbody>
		</table>
	</div>

	<footer
		class="flex flex-wrap items-center gap-x-3 gap-y-2 border-t bg-muted/40 px-3 py-3 sm:px-4"
	>
		{#if turn.published}
			<p class="flex flex-wrap items-center gap-1.5 text-sm text-success">
				<Fa icon={faCircleCheck} class="size-3.5" />
				ታትሟል፦
				<!-- eslint-disable svelte/no-navigation-without-resolve -- external wiki page, not an app route -->
				<a
					href={turn.published.url}
					target="_blank"
					rel="noreferrer"
					class="font-mono underline underline-offset-4">{turn.published.title}</a
				>
				<!-- eslint-enable svelte/no-navigation-without-resolve -->
			</p>
		{:else}
			<div class="min-w-0 flex-1 text-xs text-muted-foreground">
				{#if changes.total > 0}
					<span class="font-medium text-foreground">{changes.total} ለውጦች</span>
					·
					<button
						type="button"
						class="underline underline-offset-2 hover:text-foreground"
						onclick={undoAll}>ሁሉንም መልስ</button
					>
				{/if}
			</div>
			<div class="flex flex-col items-end gap-0.5">
				<Button size="sm" disabled={!canDownload} onclick={download}>
					<Fa icon={faDownload} class="size-3.5" />
					Download · {rows.length} ዝምድናዎች
				</Button>
				<span class="text-[11px] text-muted-foreground">{downloadName}</span>
			</div>
		{/if}
	</footer>
</section>

{#snippet unmappedRow(field: UnmappedField)}
	<tr
		class="border-b border-l-4 border-dashed border-l-muted-foreground/30 align-top"
		data-change="unmapped"
	>
		<td class="px-3 py-2.5 sm:px-4">{@render fieldCell(field.name, field.value)}</td>
		<td class="px-3 py-2.5 sm:px-4">
			<div class="flex items-center gap-1.5">
				<Input
					value={drafts[field.name] ?? ''}
					oninput={(e) => (drafts[field.name] = e.currentTarget.value)}
					onkeydown={(e) => {
						if (e.key === 'Enter') assign(field.name, drafts[field.name] ?? '', 1);
					}}
					placeholder="አልተዛመደም፤ ባህርይ ይጻፉ ወይም ይፈልጉ"
					aria-label="DBpedia ባህርይ ለ{field.name}"
					class="h-8 border-dashed font-mono text-xs placeholder:font-sans"
				/>
				{@render searchButton(field.name)}
			</div>
			{#if field.suggestions?.length}
				<div class="mt-1.5 flex flex-wrap items-center gap-1.5">
					<span class="text-[11px] text-muted-foreground">ቅርብ (ያልተረጋገጡ)፦</span>
					{#each field.suggestions as suggestion (suggestion.property)}
						<button
							type="button"
							class="rounded-full border border-dashed px-2.5 py-0.5 font-mono text-xs hover:border-primary/50 hover:bg-accent"
							title="ይህን ባህርይ ተጠቀም"
							onclick={() => assign(field.name, suggestion.property, suggestion.score)}
						>
							{suggestion.property}
						</button>
					{/each}
				</div>
			{/if}
			{#if openSearch === field.name}
				{@render searchPanel(field.name)}
			{/if}
		</td>
		<td class="px-2 py-2.5">
			{#if drafts[field.name]?.trim()}
				<Button
					variant="outline"
					size="icon"
					class="size-8"
					onclick={() => assign(field.name, drafts[field.name] ?? '', 1)}
					title="ይህን ባህርይ ተጠቀም"
					aria-label="{field.name} አዛምድ"
				>
					<Fa icon={faPlus} class="size-3.5" />
				</Button>
			{/if}
		</td>
	</tr>
{/snippet}
