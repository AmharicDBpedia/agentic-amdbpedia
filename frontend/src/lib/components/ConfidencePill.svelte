<script lang="ts">
	import { Badge, type BadgeVariant } from '$lib/components/ui/badge/index.js';

	let { confidence }: { confidence: number } = $props();

	const variant: BadgeVariant = $derived(
		confidence >= 0.75 ? 'success' : confidence >= 0.4 ? 'warning' : 'destructive'
	);
</script>

<Badge {variant} class="tabular-nums" title="የፍለጋ እርግጠኝነት">
	{Math.round(confidence * 100)}%
</Badge>
