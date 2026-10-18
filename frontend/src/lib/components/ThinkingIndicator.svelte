<script lang="ts">
	let { label = 'በማሰብ ላይ' }: { label?: string } = $props();
</script>

<div class="flex items-center gap-2 text-sm text-muted-foreground">
	<span class="flex items-center gap-1">
		<span class="size-1.5 animate-bounce rounded-full bg-primary/60 [animation-delay:-0.3s]"></span>
		<span class="size-1.5 animate-bounce rounded-full bg-primary/60 [animation-delay:-0.15s]"
		></span>
		<span class="size-1.5 animate-bounce rounded-full bg-primary/60"></span>
	</span>
	{label}…
</div>
