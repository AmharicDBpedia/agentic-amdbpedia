<script lang="ts">
	import { mode, toggleMode } from 'mode-watcher';
	import { Button } from '$lib/components/ui/button/index.js';
	import Fa from 'svelte-fa';
	import { faSun, faMoon } from '@fortawesome/free-solid-svg-icons';
</script>

<Button onclick={toggleMode} variant="ghost" size="icon" class="size-8" aria-label="ገጽታ ቀይር">
	<Fa icon={mode.current === 'dark' ? faSun : faMoon} class="size-4" />
</Button>
