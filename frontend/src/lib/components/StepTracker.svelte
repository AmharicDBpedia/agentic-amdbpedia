<script lang="ts">
	import type { AgentStep } from '$lib/types';
	import Fa from 'svelte-fa';
	import { faCircleCheck, faCircleXmark, faSpinner } from '@fortawesome/free-solid-svg-icons';

	// Once the run is over the steps fold into one line, unless one failed.
	let { steps, finished = false }: { steps: AgentStep[]; finished?: boolean } = $props();

	const stepLabels: Record<string, string> = {
		fetch_source_article: 'Wikipedia source ማምጣት',
		extract_infobox_fields: 'infobox fields ማውጣት',
		predict_properties: 'ontology properties መተንበይ',
		format_mapping_syntax: 'mapping XML ማመንጨት',
		persist_review_item: 'run record ማስቀመጥ'
	};

	const failed = $derived(steps.some((step) => step.status === 'error'));
</script>

{#snippet list()}
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
{/snippet}

{#if steps.length > 0}
	{#if finished && !failed}
		<details class="group text-sm">
			<summary
				class="flex cursor-pointer items-center gap-2 text-xs text-muted-foreground hover:text-foreground"
			>
				<Fa icon={faCircleCheck} class="size-3 text-success" />
				{steps.length} steps ተጠናቋል
				<span class="group-open:hidden">· አሳይ</span>
			</summary>
			<div class="mt-2.5 pl-1">{@render list()}</div>
		</details>
	{:else}
		{@render list()}
	{/if}
{/if}
