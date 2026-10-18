<script lang="ts">
	import {
		BackendUnavailableError,
		PublishFailedError,
		findSemanticMatch,
		looksLikeWikipediaUrl,
		previewMapping,
		publishMapping,
		summarizeResult,
		understandInput
	} from '$lib/api';
	import {
		type PipelineTurn,
		type AnswerTurn,
		currentSession,
		ensureActiveSession,
		touchSession
	} from '$lib/chat.svelte';
	import type { MappingCandidate, PredictedMapping } from '$lib/types';
	import { Button } from '$lib/components/ui/button/index.js';
	import { Input } from '$lib/components/ui/input/index.js';
	import { Textarea } from '$lib/components/ui/textarea/index.js';
	import * as AlertDialog from '$lib/components/ui/alert-dialog/index.js';
	import ConfidencePill from '$lib/components/ConfidencePill.svelte';
	import MappingEditor from '$lib/components/MappingEditor.svelte';
	import StepTracker from '$lib/components/StepTracker.svelte';
	import ThinkingIndicator from '$lib/components/ThinkingIndicator.svelte';
	import logo from '$lib/assets/dbpedia-am-logo.png';
	import Fa from 'svelte-fa';
	import { toast } from 'svelte-sonner';
	import {
		faArrowUp,
		faCircleCheck,
		faCloudUploadAlt,
		faSpinner
	} from '@fortawesome/free-solid-svg-icons';

	let input = $state('');
	let targetClass = $state('');
	let busy = $state(false);
	// The message Groq is reading right now (before we know its turn kind).
	let understanding = $state<string | null>(null);
	let scrollAnchor: HTMLDivElement | undefined = $state();

	const turns = $derived(currentSession()?.turns ?? []);

	// In-chat "search the ontology and assign this unmapped field yourself"
	// state, keyed `${turn.id}::${fieldName}`. Deliberately not part of the
	// persisted chat session -- it's transient per-visit scratch state, so
	// the effect below rehydrates an empty entry for every unmapped field
	// after a reload.
	interface AssignState {
		query: string;
		running: boolean;
		matches: MappingCandidate[] | null;
		error?: string;
	}
	let assignByField = $state<Record<string, AssignState>>({});

	const assignKey = (turnId: string, field: string) => `${turnId}::${field}`;

	$effect(() => {
		for (const turn of turns) {
			if (turn.kind !== 'pipeline' || !turn.unmappedFields) continue;
			for (const field of turn.unmappedFields) {
				const key = assignKey(turn.id, field.name);
				if (!assignByField[key]) {
					assignByField[key] = { query: field.name, running: false, matches: null };
				}
			}
		}
	});

	async function searchForField(turn: PipelineTurn, fieldName: string) {
		const entry = assignByField[assignKey(turn.id, fieldName)];
		if (!entry || entry.running) return;
		const query = entry.query.trim();
		if (!query) return;
		entry.running = true;
		entry.error = undefined;
		entry.matches = null;
		try {
			const result = await findSemanticMatch(query, turn.targetClass);
			entry.matches = result.matches;
		} catch (err) {
			entry.error =
				err instanceof BackendUnavailableError
					? 'cross-lingual is not reachable.'
					: `Lookup failed: ${err instanceof Error ? err.message : String(err)}`;
		} finally {
			entry.running = false;
		}
	}

	// The rows shown in the editable table and sent on publish. Chats saved
	// before in-chat editing kept hand-assigned rows in `manualMappings`.
	const rowsFor = (turn: PipelineTurn): PredictedMapping[] =>
		turn.edited ?? [...(turn.mappings ?? []), ...(turn.manualMappings ?? [])];

	function ensureEdited(turn: PipelineTurn): PredictedMapping[] {
		if (!turn.edited) turn.edited = rowsFor(turn).map((row) => ({ ...row }));
		return turn.edited;
	}

	// Splits Groq's reply on `backticks` so property names render as code
	// without ever injecting HTML.
	const feedbackParts = (message: string) =>
		message.split('`').map((text, index) => ({ text, code: index % 2 === 1 }));

	function attachManualMapping(
		turn: PipelineTurn,
		fieldName: string,
		property: string,
		score: number
	) {
		ensureEdited(turn).push({
			templateProperty: fieldName,
			ontologyProperty: property,
			confidence: score,
			source: 'manual'
		});
		turn.unmappedFields = (turn.unmappedFields ?? []).filter((f) => f.name !== fieldName);
		delete assignByField[assignKey(turn.id, fieldName)];
		const session = currentSession();
		if (session) touchSession(session.id);
	}

	function scrollToBottom() {
		requestAnimationFrame(() => scrollAnchor?.scrollIntoView({ behavior: 'smooth', block: 'end' }));
	}

	const unreachable =
		'cross-lingual is not reachable — check that the HTTP server (just run-http) is running.';

	async function submit(text?: string) {
		const value = (text ?? input).trim();
		if (!value || busy) return;
		input = '';
		busy = true;

		const session = ensureActiveSession();
		const id = crypto.randomUUID();
		const wantsClass = targetClass.trim() || undefined;

		// 1. Groq reads the message: infobox, link, or a field (possibly
		//    inside a question), and which DBpedia class it belongs to.
		understanding = value;
		scrollToBottom();
		const plan = await understandInput(value, wantsClass);
		understanding = null;
		const cls = plan.target_class ?? wantsClass;

		if (plan.intent === 'url' || plan.intent === 'infobox') {
			session.turns.push({
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
			const turn = session.turns[session.turns.length - 1] as PipelineTurn;
			touchSession(session.id);
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
				// 3. Groq turns the result into the final reply.
				if (turn.mappings) {
					turn.feedback = await summarizeResult(
						'infobox',
						{
							domainClass: turn.domainClass,
							mappings: turn.mappings,
							unmappedFields: turn.unmappedFields
						},
						'am'
					);
				}
			} catch (err) {
				console.error('Mapping pipeline failed:', err);
				turn.error =
					err instanceof BackendUnavailableError
						? unreachable
						: `Unexpected error: ${err instanceof Error ? err.message : String(err)}`;
			} finally {
				turn.running = false;
				touchSession(session.id);
			}
		} else {
			session.turns.push({
				id,
				kind: 'answer',
				input: value,
				targetClass: cls,
				field: plan.intent === 'field' && plan.text !== value ? plan.text : undefined,
				matches: null,
				noMatch: false,
				running: plan.intent === 'field'
			});
			const turn = session.turns[session.turns.length - 1] as AnswerTurn;
			touchSession(session.id);
			scrollToBottom();
			if (plan.intent === 'rejected') {
				turn.error = plan.reply ?? 'Rejected: prompt-injection attempt detected.';
			} else if (plan.intent === 'unsupported') {
				turn.reply = plan.reply ?? 'የመረጃ ሳጥን፣ የዊኪፔዲያ አድራሻ ወይም የመስክ ስም ያስገቡ።';
			} else {
				try {
					const result = await findSemanticMatch(plan.text, cls);
					if (result.status === 'rejected') {
						turn.error = result.message ?? 'Rejected: prompt-injection attempt detected.';
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
							: `Unexpected error: ${err instanceof Error ? err.message : String(err)}`;
				} finally {
					turn.running = false;
					touchSession(session.id);
				}
			}
			touchSession(session.id);
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

	// Publishing writes to the live mappings wiki -- a real, hard-to-reverse
	// edit -- so it always goes through this confirmation dialog first.
	let confirmPublishTurnId = $state<string | null>(null);

	const canPublish = (turn: PipelineTurn) =>
		Boolean(turn.templateName) && rowsFor(turn).length > 0 && !turn.published;

	function requestPublish(turn: PipelineTurn) {
		if (canPublish(turn)) confirmPublishTurnId = turn.id;
	}

	async function publish(turn: PipelineTurn) {
		if (!turn.templateName || turn.publishing) return;
		turn.publishing = true;
		turn.publishError = undefined;
		const request = publishMapping({
			templateName: turn.templateName,
			domainClass: turn.domainClass ?? turn.targetClass ?? 'Thing',
			mappings: rowsFor(turn),
			originalMappings: turn.mappings ?? [],
			reviewItemId: turn.reviewItemId
		});
		toast.promise(request, {
			loading: 'Publishing to mappings.dbpedia.org…',
			success: (result) => {
				turn.published = { title: result.title, url: result.url };
				return `Published ${result.title}.`;
			},
			error: (err) => {
				turn.publishError =
					err instanceof PublishFailedError
						? err.message
						: 'Could not reach the backend to publish.';
				return `Publish failed: ${turn.publishError}`;
			}
		});
		try {
			await request;
		} catch {
			// already surfaced via the toast.promise error callback above
		} finally {
			turn.publishing = false;
			confirmPublishTurnId = null;
			const session = currentSession();
			if (session) touchSession(session.id);
		}
	}

	const examples = [
		{
			label: 'የመረጃ ሳጥን ለማዛመድ ይለጥፉ',
			value: `{{Infobox bridge\n| ስም = ደደሳ ድልድይ\n| ርዝመት = 1,700 ሜትር\n}}`
		},
		{ label: 'አንድ መስክ ምን እንደሚዛመድ ይጠይቁ', value: 'አይካኦ_ኮድ' },
		{
			label: 'ወይም የዊኪፔዲያ አድራሻ',
			value: 'https://am.wikipedia.org/wiki/ታላቁ_የኢትዮጵያ_ሕዳሴ_ግድብ'
		}
	];
</script>

<div class="mx-auto flex h-full max-w-3xl flex-col">
	<div class="flex-1 overflow-y-auto px-4 py-6 md:px-8">
		{#if turns.length === 0}
			<div class="flex h-full flex-col items-center justify-center gap-6 text-center">
				<img src={logo} alt="Amharic DBpedia" class="size-16 object-contain" />
				<div>
					<h1 class="text-2xl font-semibold tracking-tight">የአማርኛ → DBpedia ማዛመጃ</h1>
					<p class="mt-2 text-sm text-muted-foreground">
						የመረጃ ሳጥን፣ የዊኪፔዲያ የገጽ አድራሻ ይለጥፉ፣ ወይም ስለ መስክ ይጠይቁ። ውጤቱን እዚህ ያርሙ፤
						«Publish» እስከሚጫኑ ድረስ ምንም አይታተምም።
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
								class="rounded-2xl rounded-br-sm bg-primary px-4 py-2.5 text-sm whitespace-pre-wrap text-primary-foreground">{turn.input}</pre>
							{#if turn.targetClass}
								<p class="mt-1 pr-1 text-right text-xs text-muted-foreground">
									class: {turn.targetClass}
								</p>
							{/if}
						</div>
					</div>

					<div class="flex flex-col gap-3">
						{#if turn.kind === 'pipeline'}
							{#if turn.running && turn.steps.length === 0}
								<ThinkingIndicator
									label={looksLikeWikipediaUrl(turn.input)
										? 'Fetching the article'
										: 'Extracting and predicting'}
								/>
							{/if}
							<StepTracker steps={turn.steps} />
							{#if turn.feedback}
								<div
									class="rounded-2xl rounded-tl-sm bg-muted/60 px-4 py-2.5 text-sm leading-relaxed"
								>
									{#each feedbackParts(turn.feedback.message) as part, i (i)}
										{#if part.code}<code class="font-mono text-[0.85em]">{part.text}</code
											>{:else}{part.text}{/if}
									{/each}
								</div>
							{/if}
							{#if turn.mappings && turn.edited && (turn.mappings.length > 0 || turn.edited.length > 0)}
								{@const rows = turn.edited}
								<MappingEditor
									bind:mappings={turn.edited}
									original={turn.mappings}
									domainClass={turn.domainClass ?? turn.targetClass ?? 'Thing'}
									readonly={Boolean(turn.published)}
								/>
								{#if turn.mappingWikitext}
									<details class="rounded-lg border">
										<summary
											class="cursor-pointer px-3 py-2 text-sm font-medium text-muted-foreground hover:text-foreground"
										>
											View mapping wikitext
										</summary>
										<pre
											class="overflow-x-auto border-t bg-muted/40 px-3 py-2.5 font-mono text-xs whitespace-pre-wrap">{turn.mappingWikitext}</pre>
									</details>
								{/if}
								{#if turn.xmlRules}
									<details class="rounded-lg border">
										<summary
											class="cursor-pointer px-3 py-2 text-sm font-medium text-muted-foreground hover:text-foreground"
										>
											View mapping XML
										</summary>
										<pre
											class="overflow-x-auto border-t bg-muted/40 px-3 py-2.5 font-mono text-xs whitespace-pre-wrap">{turn.xmlRules}</pre>
									</details>
								{/if}
								{#if turn.published}
									<p class="flex flex-wrap items-center gap-1.5 text-sm text-success">
										<Fa icon={faCircleCheck} class="size-3.5" />
										Published
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
									<div class="flex flex-wrap items-center gap-2">
										<Button
											size="sm"
											disabled={turn.publishing || !canPublish(turn)}
											onclick={() => requestPublish(turn)}
										>
											<Fa
												icon={turn.publishing ? faSpinner : faCloudUploadAlt}
												class="size-3.5 {turn.publishing ? 'animate-spin' : ''}"
											/>
											Publish {rows.length} mapping{rows.length === 1 ? '' : 's'}
										</Button>
										{#if !turn.templateName}
											<span class="text-xs text-muted-foreground"
												>No template name found, so there is no page to publish to.</span
											>
										{:else}
											<span class="text-xs text-muted-foreground"
												>to <span class="font-mono">Mapping am:{turn.templateName}</span></span
											>
										{/if}
									</div>
									{#if turn.publishError}
										<p class="text-sm text-destructive">{turn.publishError}</p>
									{/if}
								{/if}
							{:else if turn.mappings}
								<p class="text-sm text-muted-foreground">
									No properties were confidently mapped from that infobox.
								</p>
							{/if}
							{#if turn.mappings && turn.warnings && turn.warnings.length > 0}
								<ul class="space-y-1 text-xs text-muted-foreground">
									{#each turn.warnings as note (note)}
										<li class="flex gap-1.5">
											<span aria-hidden="true">·</span><span>{note}</span>
										</li>
									{/each}
								</ul>
							{/if}
							{#if turn.mappings && turn.unmappedFields && turn.unmappedFields.length > 0}
								<div class="rounded-lg border border-dashed p-3">
									<p class="mb-2.5 text-xs text-muted-foreground">
										{turn.unmappedFields.length} field{turn.unmappedFields.length === 1 ? '' : 's'} couldn't
										be mapped automatically. Pick a suggestion or search the ontology to add
										{turn.unmappedFields.length === 1 ? 'it' : 'them'} to the table, or leave
										{turn.unmappedFields.length === 1 ? 'it' : 'them'} out of the publish.
									</p>
									<div class="flex flex-col gap-2.5">
										{#each turn.unmappedFields as field (field.name)}
											{@const entry = assignByField[assignKey(turn.id, field.name)]}
											<div class="rounded-md bg-muted/40 p-2.5">
												<div class="flex flex-wrap items-baseline gap-x-2">
													<span class="font-mono text-xs">{field.name}</span>
													{#if field.value}
														<span class="truncate text-xs text-muted-foreground"
															>= {field.value}</span
														>
													{/if}
												</div>
												{#if field.suggestions?.length}
													<div class="mt-2 flex flex-wrap items-center gap-1.5">
														<span class="text-[11px] text-muted-foreground"
															>Closest (unverified):</span
														>
														{#each field.suggestions as suggestion (suggestion.property)}
															<button
																type="button"
																class="rounded-full border border-dashed px-2.5 py-0.5 font-mono text-xs hover:border-primary/50 hover:bg-accent"
																title="Assign this property to the field"
																onclick={() =>
																	attachManualMapping(
																		turn,
																		field.name,
																		suggestion.property,
																		suggestion.score
																	)}
															>
																{suggestion.property}
															</button>
														{/each}
													</div>
												{/if}
												{#if entry}
													<div class="mt-2 flex gap-2">
														<Input
															class="h-8 text-xs"
															placeholder="ኦንቶሎጂውን ይፈልጉ…"
															bind:value={entry.query}
															onkeydown={(e) => {
																if (e.key === 'Enter') {
																	e.preventDefault();
																	searchForField(turn, field.name);
																}
															}}
														/>
														<Button
															size="sm"
															variant="outline"
															disabled={entry.running}
															onclick={() => searchForField(turn, field.name)}
														>
															{entry.running ? 'Searching…' : 'Search'}
														</Button>
													</div>
													{#if entry.error}
														<p class="mt-1.5 text-xs text-destructive">{entry.error}</p>
													{:else if entry.matches && entry.matches.length === 0}
														<p class="mt-1.5 text-xs text-muted-foreground">
															No confident ontology match for that term.
														</p>
													{:else if entry.matches}
														<ul class="mt-2 flex flex-wrap gap-1.5">
															{#each entry.matches as match (match.property)}
																<li>
																	<button
																		type="button"
																		class="flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs hover:border-primary/50 hover:bg-accent"
																		onclick={() =>
																			attachManualMapping(
																				turn,
																				field.name,
																				match.property,
																				match.score
																			)}
																	>
																		<span class="font-mono">{match.property}</span>
																		<ConfidencePill confidence={match.score} />
																	</button>
																</li>
															{/each}
														</ul>
													{/if}
												{/if}
											</div>
										{/each}
									</div>
								</div>
							{/if}
						{:else if turn.kind === 'answer'}
							{#if turn.field}
								<p class="text-xs text-muted-foreground">
									Looking up the field <span class="font-mono">{turn.field}</span>
								</p>
							{/if}
							{#if turn.reply}
								<div class="rounded-2xl rounded-tl-sm bg-muted/60 px-4 py-2.5 text-sm">
									{turn.reply}
								</div>
							{/if}
							{#if turn.feedback}
								<div
									class="rounded-2xl rounded-tl-sm bg-muted/60 px-4 py-2.5 text-sm leading-relaxed"
								>
									{#each feedbackParts(turn.feedback.message) as part, i (i)}
										{#if part.code}<code class="font-mono text-[0.85em]">{part.text}</code
											>{:else}{part.text}{/if}
									{/each}
								</div>
							{/if}
							{#if turn.running}
								<ThinkingIndicator label="Searching the ontology" />
							{:else if turn.noMatch}
								<p class="text-sm">No confident match found in the ontology for that term.</p>
								{#if turn.llmUnavailable}
									<p class="mt-1 text-xs text-muted-foreground">
										The language model could not be reached, so it wasn't asked. Start Ollama or
										check GROQ_API_KEY.
									</p>
								{/if}
								{#if turn.suggestions?.length}
									<p class="mt-2 text-xs text-muted-foreground">
										Closest by similarity (unverified — check before using):
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
													title="Chosen by the language model because retrieval found nothing confident -- verify before using."
												>
													AI-proposed
												</span>
											{:else if match.payload?.source === 'learned'}
												<span
													class="rounded bg-accent px-1 py-0.5 text-[10px] font-medium text-accent-foreground"
													title="A reviewer approved this field for this class before."
												>
													learned
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
						class="max-w-[85%] rounded-2xl rounded-br-sm bg-primary px-4 py-2.5 text-sm whitespace-pre-wrap text-primary-foreground">{understanding}</pre>
				</div>
				<ThinkingIndicator label="Reading your message" />
			</div>
		{/if}
		<div bind:this={scrollAnchor}></div>
	</div>

	<div class="shrink-0 px-4 pb-4 md:px-8">
		<div class="rounded-2xl border bg-card shadow-sm">
			<div class="px-3 pt-2.5">
				<Input
					bind:value={targetClass}
					placeholder="ዒላማ ክፍል (አማራጭ፣ ለምሳሌ Bridge)"
					class="h-6 border-none bg-transparent px-0 text-xs shadow-none focus-visible:ring-0"
				/>
			</div>
			<div class="flex items-end gap-2 p-2">
				<Textarea
					bind:value={input}
					onkeydown={onKeydown}
					placeholder="የመረጃ ሳጥን፣ የዊኪፔዲያ አድራሻ ወይም ስለ መስክ ይጠይቁ…"
					rows={1}
					class="max-h-48 min-h-9 resize-none border-none bg-transparent px-2 py-1.5 shadow-none focus-visible:ring-0"
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
</div>

{#each turns as turn (turn.id)}
	{#if turn.kind === 'pipeline'}
		<AlertDialog.Root
			open={confirmPublishTurnId === turn.id}
			onOpenChange={(open) => {
				if (!open) confirmPublishTurnId = null;
			}}
		>
			<AlertDialog.Content>
				<AlertDialog.Header>
					<AlertDialog.Title>Publish to mappings.dbpedia.org?</AlertDialog.Title>
					<AlertDialog.Description>
						This writes
						<span class="font-mono text-foreground">{rowsFor(turn).length}</span>
						mapping{rowsFor(turn).length === 1 ? '' : 's'} to
						<span class="font-mono text-foreground">Mapping am:{turn.templateName}</span>
						on the live wiki immediately, using the configured bot account. Mappings already on that page
						are kept; a field already mapped there is updated. It is a real edit, not a preview.
					</AlertDialog.Description>
				</AlertDialog.Header>
				{#if rowsFor(turn).length > 0}
					<ul class="space-y-1 rounded-lg border bg-muted/40 p-2">
						{#each rowsFor(turn) as mapping (mapping.templateProperty)}
							<li class="font-mono text-xs">
								{mapping.templateProperty} to
								<span class="text-foreground">{mapping.ontologyProperty}</span>
							</li>
						{/each}
					</ul>
				{/if}
				<AlertDialog.Footer>
					<AlertDialog.Cancel>Cancel</AlertDialog.Cancel>
					<AlertDialog.Action
						disabled={Boolean(turn.publishing)}
						onclick={(e) => {
							e.preventDefault();
							publish(turn);
						}}
					>
						{#if turn.publishing}
							<Fa icon={faSpinner} class="size-3.5 animate-spin" />
						{/if}
						Publish
					</AlertDialog.Action>
				</AlertDialog.Footer>
			</AlertDialog.Content>
		</AlertDialog.Root>
	{/if}
{/each}
