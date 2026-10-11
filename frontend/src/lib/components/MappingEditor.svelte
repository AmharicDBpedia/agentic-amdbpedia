<script lang="ts">
	import { BackendUnavailableError, findSemanticMatch } from '$lib/api';
	import type { MappingCandidate, PredictedMapping } from '$lib/types';
	import { cn } from '$lib/utils.js';
	import { Button } from '$lib/components/ui/button/index.js';
	import { Input } from '$lib/components/ui/input/index.js';
	import {
		Table,
		TableBody,
		TableCell,
		TableHead,
		TableHeader,
		TableRow
	} from '$lib/components/ui/table/index.js';
	import ConfidencePill from './ConfidencePill.svelte';
	import Fa from 'svelte-fa';
	import {
		faMagnifyingGlass,
		faTrash,
		faRotateLeft,
		faPlus,
		faSpinner
	} from '@fortawesome/free-solid-svg-icons';

	let {
		mappings = $bindable(),
		original,
		domainClass,
		readonly = false
	}: {
		mappings: PredictedMapping[];
		original: PredictedMapping[];
		domainClass: string;
		readonly?: boolean;
	} = $props();

	function originalFor(templateProperty: string) {
		return original.find((row) => row.templateProperty === templateProperty);
	}

	const removedOriginals = $derived(
		original.filter((row) => !mappings.some((m) => m.templateProperty === row.templateProperty))
	);

	function isEdited(row: PredictedMapping) {
		const source = originalFor(row.templateProperty);
		return source !== undefined && source.ontologyProperty !== row.ontologyProperty;
	}

	function isNew(row: PredictedMapping) {
		return originalFor(row.templateProperty) === undefined;
	}

	function resetRow(index: number) {
		const source = originalFor(mappings[index].templateProperty);
		if (!source) return;
		mappings[index] = { ...source };
	}

	function removeRow(index: number) {
		mappings.splice(index, 1);
	}

	function restoreRow(row: PredictedMapping) {
		mappings.push({ ...row });
	}

	function applyOntologyEdit(index: number, value: string) {
		mappings[index].ontologyProperty = value;
		mappings[index].confidence = 1;
	}

	// Per-row "find a match" panel -- reuses the same retrieval endpoint the
	// Mapping Assistant chat panel uses, so a correction is picked from real
	// grounded candidates instead of freehand-typed and possibly misspelled.
	let suggestOpenIndex = $state<number | null>(null);
	let suggestQuery = $state('');
	let suggestResults = $state<MappingCandidate[] | null>(null);
	let suggestLoading = $state(false);
	let suggestError = $state<string | null>(null);

	function openSuggest(index: number) {
		suggestOpenIndex = suggestOpenIndex === index ? null : index;
		suggestQuery = mappings[index].templateProperty;
		suggestResults = null;
		suggestError = null;
		if (suggestOpenIndex !== null) runSuggest();
	}

	async function runSuggest() {
		if (!suggestQuery.trim()) return;
		suggestLoading = true;
		suggestError = null;
		try {
			const result = await findSemanticMatch(suggestQuery, domainClass || undefined);
			suggestResults = result.status === 'no_match' ? [] : result.matches;
		} catch (err) {
			suggestError =
				err instanceof BackendUnavailableError
					? 'Retrieval backend not reachable right now.'
					: 'Search failed unexpectedly.';
		} finally {
			suggestLoading = false;
		}
	}

	function applySuggestion(index: number, candidate: MappingCandidate) {
		applyOntologyEdit(index, candidate.property);
		suggestOpenIndex = null;
	}

	let newTemplateProperty = $state('');
	let newOntologyProperty = $state('');

	function addRow() {
		if (!newTemplateProperty.trim() || !newOntologyProperty.trim()) return;
		mappings.push({
			templateProperty: newTemplateProperty.trim(),
			ontologyProperty: newOntologyProperty.trim(),
			confidence: 1
		});
		newTemplateProperty = '';
		newOntologyProperty = '';
	}
</script>

{#snippet sourceBadge(row: PredictedMapping)}
	{#if row.source === 'llm'}
		<span
			class="mt-1 inline-block rounded bg-warning/15 px-1 py-0.5 text-[10px] font-medium text-warning-foreground"
			title="Chosen by the language model because retrieval found nothing confident -- check it before publishing."
			>AI-proposed</span
		>
	{:else if row.source === 'learned'}
		<span
			class="mt-1 inline-block rounded bg-accent px-1 py-0.5 text-[10px] font-medium text-accent-foreground"
			title="Published for this field and class before.">learned</span
		>
	{:else if row.source === 'manual'}
		<span
			class="mt-1 inline-block rounded bg-accent px-1 py-0.5 text-[10px] font-medium text-accent-foreground"
			>added by you</span
		>
	{/if}
{/snippet}

<div class="overflow-hidden rounded-lg border">
	<Table>
		<TableHeader>
			<TableRow class="hover:bg-transparent">
				<TableHead>Template property</TableHead>
				<TableHead>Ontology property</TableHead>
				<TableHead>Confidence</TableHead>
				{#if !readonly}
					<TableHead class="w-10"><span class="sr-only">Actions</span></TableHead>
				{/if}
			</TableRow>
		</TableHeader>
		<TableBody>
			{#each mappings as row, index (index)}
				<TableRow class="align-top">
					<TableCell class="font-mono">
						{#if readonly}
							{row.templateProperty}
						{:else}
							<Input bind:value={row.templateProperty} class="h-8 font-mono" />
						{/if}
					</TableCell>
					<TableCell class="whitespace-normal">
						{#if readonly}
							<span class="font-mono">{row.ontologyProperty}</span>
							{@render sourceBadge(row)}
						{:else}
							<div class="flex items-center gap-1.5">
								<Input
									value={row.ontologyProperty}
									oninput={(e) => applyOntologyEdit(index, e.currentTarget.value)}
									class={cn(
										'h-8 font-mono',
										isEdited(row) && 'border-warning focus-visible:ring-warning/40'
									)}
								/>
								<Button
									variant="outline"
									size="icon"
									class="size-8 shrink-0"
									onclick={() => openSuggest(index)}
									title="Search real ontology candidates"
								>
									<Fa icon={faMagnifyingGlass} class="size-3.5" />
								</Button>
							</div>
							{#if !isEdited(row)}{@render sourceBadge(row)}{/if}
							{#if isEdited(row)}
								<p class="mt-1 flex items-center gap-1 text-xs text-warning">
									edited from
									<span class="font-mono"
										>{originalFor(row.templateProperty)?.ontologyProperty}</span
									>
									<button
										type="button"
										class="underline underline-offset-2 hover:no-underline"
										onclick={() => resetRow(index)}
									>
										reset
									</button>
								</p>
							{:else if isNew(row)}
								<p class="mt-1 text-xs text-primary">added by reviewer</p>
							{/if}

							{#if suggestOpenIndex === index}
								<div class="mt-2 rounded-lg border bg-muted/40 p-2">
									<form
										class="flex gap-1.5"
										onsubmit={(e) => {
											e.preventDefault();
											runSuggest();
										}}
									>
										<Input
											bind:value={suggestQuery}
											placeholder="Amharic field or English hint"
											class="h-7 text-xs"
										/>
										<Button type="submit" size="sm" class="h-7" disabled={suggestLoading}>
											{#if suggestLoading}
												<Fa icon={faSpinner} class="size-3.5 animate-spin" />
											{:else}
												Search
											{/if}
										</Button>
									</form>
									{#if suggestError}
										<p class="mt-1.5 text-xs text-warning">{suggestError}</p>
									{:else if suggestResults && suggestResults.length === 0}
										<p class="mt-1.5 text-xs text-muted-foreground">No confident match found.</p>
									{:else if suggestResults}
										<ul class="mt-1.5 flex flex-wrap gap-1.5">
											{#each suggestResults as candidate (candidate.property)}
												<li>
													<button
														type="button"
														class="rounded-full border px-2 py-1 text-xs hover:border-primary hover:text-primary"
														onclick={() => applySuggestion(index, candidate)}
													>
														<span class="font-mono">{candidate.property}</span>
														<span class="text-muted-foreground">
															{Math.round(candidate.score * 100)}%
														</span>
													</button>
												</li>
											{/each}
										</ul>
									{/if}
								</div>
							{/if}
						{/if}
					</TableCell>
					<TableCell>
						<ConfidencePill confidence={row.confidence} />
					</TableCell>
					{#if !readonly}
						<TableCell>
							<Button
								variant="ghost"
								size="icon"
								class="size-8 text-muted-foreground hover:text-destructive"
								onclick={() => removeRow(index)}
								title="Remove this mapping"
							>
								<Fa icon={faTrash} class="size-3.5" />
							</Button>
						</TableCell>
					{/if}
				</TableRow>
			{/each}

			{#if !readonly}
				<TableRow class="bg-muted/30 hover:bg-muted/30">
					<TableCell>
						<Input
							bind:value={newTemplateProperty}
							placeholder="new template property"
							class="h-8 border-dashed font-mono text-xs"
						/>
					</TableCell>
					<TableCell>
						<Input
							bind:value={newOntologyProperty}
							placeholder="new ontology property"
							class="h-8 border-dashed font-mono text-xs"
						/>
					</TableCell>
					<TableCell class="text-xs text-muted-foreground">—</TableCell>
					<TableCell>
						<Button
							variant="outline"
							size="sm"
							class="h-8"
							disabled={!newTemplateProperty.trim() || !newOntologyProperty.trim()}
							onclick={addRow}
						>
							<Fa icon={faPlus} class="size-3.5" />
							Add
						</Button>
					</TableCell>
				</TableRow>
			{/if}
		</TableBody>
	</Table>
</div>

{#if !readonly && removedOriginals.length > 0}
	<div class="mt-2 flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
		<span>Removed from the model's prediction:</span>
		{#each removedOriginals as row (row.templateProperty)}
			<button
				type="button"
				class="flex items-center gap-1 rounded-full border px-2 py-1 font-mono hover:border-foreground/40 hover:text-foreground"
				onclick={() => restoreRow(row)}
			>
				<Fa icon={faRotateLeft} class="size-3" />
				{row.templateProperty} to {row.ontologyProperty}
			</button>
		{/each}
	</div>
{/if}
