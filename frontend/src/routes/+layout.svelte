<script lang="ts">
	import './layout.css';
	import logo from '$lib/assets/dbpedia-am-logo.png';
	import { ModeWatcher } from 'mode-watcher';
	import { Toaster } from '$lib/components/ui/sonner/index.js';
	import { Button } from '$lib/components/ui/button/index.js';
	import AppSidebar from '$lib/components/AppSidebar.svelte';
	import ModeToggle from '$lib/components/ModeToggle.svelte';
	import Fa from 'svelte-fa';
	import { faBars } from '@fortawesome/free-solid-svg-icons';

	let { children } = $props();

	let sidebarOpen = $state(false);
</script>

<svelte:head><title>agentic-amdbpedia</title><link rel="icon" href={logo} /></svelte:head>

<ModeWatcher />
<Toaster richColors closeButton expand position="top-center" />

<div class="flex h-full overflow-hidden">
	<AppSidebar bind:open={sidebarOpen} />

	<div class="flex min-h-0 min-w-0 flex-1 flex-col">
		<header
			class="flex shrink-0 items-center gap-2 border-b px-3 py-2 md:justify-end md:border-0 md:px-4 md:py-3"
		>
			<Button
				variant="ghost"
				size="icon"
				class="size-8 md:hidden"
				onclick={() => (sidebarOpen = !sidebarOpen)}
				aria-label="የጎን ማውጫውን ክፈት ወይም ዝጋ"
			>
				<Fa icon={faBars} class="size-4" />
			</Button>
			<img src={logo} alt="የአማርኛ DBpedia" class="size-6 object-contain md:hidden" />
			<span class="text-sm font-semibold md:hidden">agentic-amdbpedia</span>
			<div class="flex-1 md:hidden"></div>
			<ModeToggle />
		</header>
		<main class="min-h-0 flex-1 overflow-hidden">
			{@render children()}
		</main>
	</div>
</div>
