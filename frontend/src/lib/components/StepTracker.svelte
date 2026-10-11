<script lang="ts">
	import type { AgentStep } from '$lib/types';
	import Fa from 'svelte-fa';
	import { faCircleCheck, faCircleXmark, faSpinner } from '@fortawesome/free-solid-svg-icons';

	let { steps }: { steps: AgentStep[] } = $props();

	const stepLabels: Record<string, string> = {
		fetch_source_article: 'Fetch Wikipedia article',
		extract_infobox_fields: 'Extract infobox fields',
		predict_properties: 'Predict ontology properties',
		format_mapping_syntax: 'Generate mapping XML',
		persist_review_item: 'Save run record'
	};
</script>

{#if steps.length > 0}
	<ol class="space-y-2.5">
		{#each steps as step, i (i)}
			<li class="flex items-center gap-2.5 text-sm">
				{#if step.status === 'error'}
					<Fa icon={faCircleXmark} class="size-3.5 shrink-0 text-destructive" />
				{:else if step.status === 'done'}
					<Fa icon={faCircleCheck} class="size-3.5 shrink-0 text-success" />
				{:else}
					<Fa icon={faSpinner} class="size-3.5 shrink-0 animate-spin text-primary" />
				{/if}
				<span class={step.status === 'done' ? 'text-muted-foreground' : 'text-foreground'}>
					{stepLabels[step.node] ?? step.node}
				</span>
				{#if step.detail && step.status !== 'done'}
					<span class="text-xs text-muted-foreground">— {step.detail}</span>
				{/if}
			</li>
		{/each}
	</ol>
{/if}
