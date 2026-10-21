<script lang="ts">
	import {
		BackendUnavailableError,
		findSemanticMatch,
		looksLikeWikipediaUrl,
		previewMapping,
		summarizeResult,
		understandInput
	} from '$lib/api';
	import { type PipelineTurn, type AnswerTurn, chat, saveChat } from '$lib/chat.svelte';
	import { Button } from '$lib/components/ui/button/index.js';
	import { Textarea } from '$lib/components/ui/textarea/index.js';
	import ConfidencePill from '$lib/components/ConfidencePill.svelte';
	import MappingCard from '$lib/components/MappingCard.svelte';
	import StepTracker from '$lib/components/StepTracker.svelte';
	import ThinkingIndicator from '$lib/components/ThinkingIndicator.svelte';
	import logo from '$lib/assets/dbpedia-am-logo.png';
	import Fa from 'svelte-fa';
	import { faArrowUp } from '@fortawesome/free-solid-svg-icons';

	let input = $state('');
	let busy = $state(false);

	// Pasted Wikipedia URLs arrive percent-encoded (am.wikipedia.org wraps
	// every Amharic character as %XX). Showing them raw in the message
	// bubble looks like noise, so decode for display -- the string the
	// backend fetched is still the one in `turn.input`.
	// `decodeURI` is deliberately avoided: it refuses to decode percent-
	// encoded reserved chars (`%3A`, `%2F`, `%3F`, ...), so a URL with an
	// encoded colon (common when copied from logs or tooling) would still
	// show as `መለጠፊያ%3Aየሶፍትዌር…`. `decodeURIComponent` decodes everything.
	function displayInput(text: string): string {
		if (!looksLikeWikipediaUrl(text)) return text;
		try {
			return decodeURIComponent(text);
		} catch {
			return text;
		}
	}
	// The message Groq is reading right now (before we know its turn kind).
	let understanding = $state<string | null>(null);
	let scrollAnchor: HTMLDivElement | undefined = $state();

	const turns = $derived(chat.turns);

	// Every change (streamed steps, edits, publishes) is kept for this tab.
	$effect(() => {
		saveChat($state.snapshot(chat.turns));
	});

	// Splits Groq's reply on `backticks` so property names render as code
	// without ever injecting HTML.
	const feedbackParts = (message: string) =>
		message.split('`').map((text, index) => ({ text, code: index % 2 === 1 }));

	function scrollToBottom() {
		requestAnimationFrame(() => scrollAnchor?.scrollIntoView({ behavior: 'smooth', block: 'end' }));
	}

	const unreachable = 'cross-lingual አይገኝም — የHTTP አገልጋዩ (just run-http) መሮጡን ያረጋግጡ።';

	async function submit(text?: string) {
		const value = (text ?? input).trim();
		if (!value || busy) return;
		input = '';
		busy = true;

		const id = crypto.randomUUID();

		// 1. Groq reads the message: infobox, link, or a field (possibly
		//    inside a question), and which DBpedia class it belongs to.
		understanding = value;
		scrollToBottom();
		const plan = await understandInput(value);
		understanding = null;
		const cls = plan.target_class ?? undefined;

		if (plan.intent === 'url' || plan.intent === 'infobox') {
			chat.turns.push({
				id,
				kind: 'pipeline',
				input: value,
				targetClass: cls,
				steps: [],
				mappings: null,
				running: true
			});
			// Re-read the pushed element: Svelte 5 proxies objects inserted
			// into $state, and only the proxy's mutations reach the DOM.
			const turn = chat.turns[chat.turns.length - 1] as PipelineTurn;
			scrollToBottom();
			try {
				// 2. The mapping itself: retrieval + Gemma (Ollama).
				for await (const event of previewMapping(plan.text, cls)) {
					if ('mappings' in event) {
						turn.mappings = event.mappings;
						turn.edited = event.mappings.map((row) => ({ ...row }));
						turn.mappingWikitext = event.mappingWikitext;
						turn.xmlRules = event.xmlRules;
						turn.reviewItemId = event.reviewItemId;
						turn.templateName = event.templateName ?? null;
						turn.domainClass = event.domainClass ?? cls;
						turn.warnings = event.warnings ?? [];
						turn.unmappedFields = event.unmappedFields ?? [];
					} else {
						turn.steps = [...turn.steps, event];
					}
					scrollToBottom();
				}
				// A Groq summary of the mapping would just repeat what the card
				// already shows (edits, AI-proposed, unmapped), so skip it.
			} catch (err) {
				console.error('Mapping pipeline failed:', err);
				turn.error =
					err instanceof BackendUnavailableError
						? unreachable
						: `ያልተጠበቀ ስህተት፦ ${err instanceof Error ? err.message : String(err)}`;
			} finally {
				turn.running = false;
			}
		} else {
			chat.turns.push({
				id,
				kind: 'answer',
				input: value,
				targetClass: cls,
				field: plan.intent === 'field' && plan.text !== value ? plan.text : undefined,
				matches: null,
				noMatch: false,
				running: plan.intent === 'field'
			});
			const turn = chat.turns[chat.turns.length - 1] as AnswerTurn;
			scrollToBottom();
			if (plan.intent === 'rejected') {
				turn.error = plan.reply ?? 'ውድቅ ተደርጓል፦ የprompt-injection ሙከራ ተገኝቷል።';
			} else if (plan.intent === 'unsupported') {
				turn.reply = plan.reply ?? 'የመረጃ ሳጥን፣ የዊኪፔዲያ አድራሻ ወይም የመስክ ስም ያስገቡ።';
			} else {
				try {
					const result = await findSemanticMatch(plan.text, cls);
					if (result.status === 'rejected') {
						turn.error = result.message ?? 'ውድቅ ተደርጓል፦ የprompt-injection ሙከራ ተገኝቷል።';
					} else {
						if (result.status === 'no_match') {
							turn.noMatch = true;
							turn.suggestions = result.suggestions ?? [];
							turn.llmUnavailable = Boolean(result.llm_error);
						} else {
							turn.matches = result.matches;
						}
						turn.feedback = await summarizeResult('field', result, 'am');
					}
				} catch (err) {
					console.error('Assistant lookup failed:', err);
					turn.error =
						err instanceof BackendUnavailableError
							? unreachable
							: `ያልተጠበቀ ስህተት፦ ${err instanceof Error ? err.message : String(err)}`;
				} finally {
					turn.running = false;
				}
			}
		}

		busy = false;
		scrollToBottom();
	}

	function onKeydown(event: KeyboardEvent) {
		if (event.key === 'Enter' && !event.shiftKey) {
			event.preventDefault();
			submit();
		}
	}

	const examples = [
		{
			label: 'የመረጃ ሳጥን ለማዛመድ ይለጥፉ',
			value: `{{Infobox bridge\n| ስም = ደደሳ ድልድይ\n| ርዝመት = 1,700 ሜትር\n}}`
		},
		{ label: 'አንድ መስክ ምን እንደሚዛመድ ይጠይቁ', value: 'አይካኦ_ኮድ' },
		{
			label: 'የመለጠፊያ አድራሻ',
			value: 'https://am.wikipedia.org/wiki/መለጠፊያ:የምግብ_መረጃ'
		},
		{
			label: 'ወይም የዊኪፔዲያ ገጽ አድራሻ',
			value: 'https://am.wikipedia.org/wiki/ታላቁ_የኢትዮጵያ_ሕዳሴ_ግድብ'
		}
	];
</script>

{#snippet feedbackBubble(message: string)}
	<div class="rounded-2xl rounded-tl-sm bg-muted/60 px-4 py-2.5 text-sm leading-relaxed">
		{#each feedbackParts(message) as part, i (i)}
			{#if part.code}<code class="font-mono text-[0.85em]">{part.text}</code>{:else}{part.text}{/if}
		{/each}
	</div>
{/snippet}

<div class="mx-auto flex h-full max-w-4xl flex-col">
	<div class="flex-1 overflow-y-auto px-4 py-6 md:px-8">
		{#if turns.length === 0}
			<div class="flex h-full flex-col items-center justify-center gap-6 text-center">
				<img src={logo} alt="የአማርኛ DBpedia" class="size-16 object-contain" />
				<div>
					<h1 class="text-2xl font-semibold tracking-tight">የአማርኛ → DBpedia ማዛመጃ</h1>
					<p class="mt-2 text-sm text-muted-foreground">
						የመረጃ ሳጥን፣ የዊኪፔዲያ የገጽ ወይም የመለጠፊያ አድራሻ ይለጥፉ፣ ወይም ስለ መስክ ይጠይቁ። ውጤቱን እዚሁ ያርሙ፤ «Publish»
						እስከሚጫኑ ድረስ ምንም አይታተምም።
					</p>
				</div>
				<div class="flex flex-wrap justify-center gap-2">
					{#each examples as example (example.label)}
						<button
							type="button"
							class="rounded-full border px-3.5 py-1.5 text-sm hover:border-primary/50 hover:bg-accent"
							onclick={() => submit(example.value)}
						>
							{example.label}
						</button>
					{/each}
				</div>
			</div>
		{:else}
			<div class="flex flex-col gap-6">
				{#each turns as turn (turn.id)}
					<div class="flex justify-end">
						<div class="max-w-[85%]">
							<pre
								class="max-h-60 overflow-y-auto rounded-2xl rounded-br-sm bg-primary px-4 py-2.5 text-sm whitespace-pre-wrap text-primary-foreground">{displayInput(
									turn.input
								)}</pre>
							{#if turn.targetClass}
								<p class="mt-1 pr-1 text-right text-xs text-muted-foreground">
									ክፍል፦ {turn.targetClass}
								</p>
							{/if}
						</div>
					</div>

					<div class="flex flex-col gap-3">
						{#if turn.kind === 'pipeline'}
							{#if turn.running && turn.steps.length === 0}
								<ThinkingIndicator
									label={looksLikeWikipediaUrl(turn.input)
										? 'ገጹን በማምጣት ላይ'
										: 'መስኮችን በማውጣትና በመተንበይ ላይ'}
								/>
							{/if}
							<StepTracker steps={turn.steps} finished={!turn.running} />
							{#if turn.mappings && turn.edited && (turn.mappings.length > 0 || turn.edited.length > 0 || (turn.unmappedFields?.length ?? 0) > 0)}
								<MappingCard {turn} />
							{:else if turn.mappings}
								<p class="text-sm text-muted-foreground">ከዚህ የመረጃ ሳጥን በእርግጠኝነት የተዛመደ ባህርይ የለም።</p>
							{/if}
						{:else if turn.kind === 'answer'}
							{#if turn.field}
								<p class="text-xs text-muted-foreground">
									መስኩን በመፈለግ ላይ፦ <span class="font-mono">{turn.field}</span>
								</p>
							{/if}
							{#if turn.reply}
								<div class="rounded-2xl rounded-tl-sm bg-muted/60 px-4 py-2.5 text-sm">
									{turn.reply}
								</div>
							{/if}
							{#if turn.feedback}
								{@render feedbackBubble(turn.feedback.message)}
							{/if}
							{#if turn.running}
								<ThinkingIndicator label="ኦንቶሎጂውን በመፈለግ ላይ" />
							{:else if turn.noMatch}
								<p class="text-sm">ለዚህ ቃል በኦንቶሎጂው ውስጥ እርግጠኛ ዝምድና አልተገኘም።</p>
								{#if turn.llmUnavailable}
									<p class="mt-1 text-xs text-muted-foreground">
										የቋንቋ ሞዴሉ ሊገኝ ስላልቻለ አልተጠየቀም። Ollama ን ያስጀምሩ ወይም GROQ_API_KEY ን ያረጋግጡ።
									</p>
								{/if}
								{#if turn.suggestions?.length}
									<p class="mt-2 text-xs text-muted-foreground">
										በተመሳሳይነት ቅርብ የሆኑት (ያልተረጋገጡ — ከመጠቀምዎ በፊት ያረጋግጡ)፦
									</p>
									<ul class="mt-1.5 flex flex-wrap gap-1.5">
										{#each turn.suggestions as match (match.property)}
											<li
												class="flex items-center gap-1.5 rounded-full border border-dashed px-2.5 py-1 text-sm text-muted-foreground"
											>
												<span class="font-mono">{match.property}</span>
											</li>
										{/each}
									</ul>
								{/if}
							{:else if turn.matches}
								<ul class="flex flex-wrap gap-1.5">
									{#each turn.matches as match (match.property)}
										<li class="flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-sm">
											<span class="font-mono">{match.property}</span>
											{#if match.payload?.source === 'llm'}
												<span
													class="rounded bg-warning/15 px-1 py-0.5 text-[10px] font-medium text-warning-foreground"
													title="ፍለጋው እርግጠኛ ውጤት ስላላገኘ በቋንቋ ሞዴሉ የተመረጠ — ከመጠቀምዎ በፊት ያረጋግጡ።"
												>
													በAI የተጠቆመ
												</span>
											{:else if match.payload?.source === 'learned'}
												<span
													class="rounded bg-accent px-1 py-0.5 text-[10px] font-medium text-accent-foreground"
													title="ለዚሁ መስክና ክፍል ቀደም ሲል ታትሟል።"
												>
													የተማረ
												</span>
											{:else}
												<ConfidencePill confidence={match.score} />
											{/if}
										</li>
									{/each}
								</ul>
							{/if}
						{/if}

						{#if turn.error}
							<p
								class="rounded-lg border border-warning/40 bg-warning/10 px-3 py-2 text-sm text-warning-foreground"
							>
								{turn.error}
							</p>
						{/if}
					</div>
				{/each}
			</div>
		{/if}
		{#if understanding}
			<div class="mt-6 flex flex-col gap-3">
				<div class="flex justify-end">
					<pre
						class="max-h-60 max-w-[85%] overflow-y-auto rounded-2xl rounded-br-sm bg-primary px-4 py-2.5 text-sm whitespace-pre-wrap text-primary-foreground">{displayInput(
							understanding
						)}</pre>
				</div>
				<ThinkingIndicator label="መልእክትዎን በማንበብ ላይ" />
			</div>
		{/if}
		<div bind:this={scrollAnchor}></div>
	</div>

	<div class="shrink-0 px-4 pb-4 md:px-8">
		<div class="flex items-end gap-2 rounded-2xl border bg-card p-2 shadow-sm">
			<Textarea
				bind:value={input}
				onkeydown={onKeydown}
				placeholder="የመረጃ ሳጥን፣ የዊኪፔዲያ አድራሻ ወይም ስለ መስክ ይጠይቁ…"
				rows={1}
				class="max-h-48 min-h-9 flex-1 resize-none border-none bg-transparent px-2 py-1.5 shadow-none focus-visible:ring-0"
			/>
			<Button
				size="icon"
				class="mb-0.5 size-8 shrink-0 rounded-full"
				disabled={busy || !input.trim()}
				onclick={() => submit()}
				aria-label="ላክ"
			>
				<Fa icon={faArrowUp} class="size-3.5" />
			</Button>
		</div>
	</div>
</div>
