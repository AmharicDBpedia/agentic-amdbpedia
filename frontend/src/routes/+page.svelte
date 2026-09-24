<script lang="ts">
	import { resolve } from '$app/paths';
	import {
		BackendUnavailableError,
		DecisionFailedError,
		decideReview,
		findSemanticMatch,
		looksLikeWikipediaUrl,
		previewMapping
	} from '$lib/api';
	import {
		type PipelineTurn,
		type AnswerTurn,
		currentSession,
		ensureActiveSession,
		touchSession
	} from '$lib/chat.svelte';
	import type { MappingCandidate } from '$lib/types';
	import { Button } from '$lib/components/ui/button/index.js';
	import { Input } from '$lib/components/ui/input/index.js';
	import { Textarea } from '$lib/components/ui/textarea/index.js';
	import {
		Table,
		TableBody,
		TableCell,
		TableHead,
		TableHeader,
		TableRow
	} from '$lib/components/ui/table/index.js';
	import * as AlertDialog from '$lib/components/ui/alert-dialog/index.js';
	import ConfidencePill from '$lib/components/ConfidencePill.svelte';
	import StatusBadge from '$lib/components/StatusBadge.svelte';
	import StepTracker from '$lib/components/StepTracker.svelte';
	import ThinkingIndicator from '$lib/components/ThinkingIndicator.svelte';
	import logo from '$lib/assets/dbpedia-am-logo.png';
	import Fa from 'svelte-fa';
	import { toast } from 'svelte-sonner';
	import {
		faArrowUp,
		faArrowRight,
		faCircleCheck,
		faCircleXmark,
		faCloudUploadAlt,
		faSpinner
	} from '@fortawesome/free-solid-svg-icons';

	let input = $state('');
	let targetClass = $state('');
	let busy = $state(false);
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

	// Auto-predicted rows plus any the reviewer assigned by hand -- the set
	// that gets approved and, later, published.
	const rowsFor = (turn: PipelineTurn) => [
		...(turn.mappings ?? []),
		...(turn.manualMappings ?? [])
	];

	function attachManualMapping(
		turn: PipelineTurn,
		fieldName: string,
		property: string,
		score: number
	) {
		turn.manualMappings = [
			...(turn.manualMappings ?? []),
			{
				templateProperty: fieldName,
				ontologyProperty: property,
				confidence: score,
				source: 'manual'
			}
		];
		turn.unmappedFields = (turn.unmappedFields ?? []).filter((f) => f.name !== fieldName);
		delete assignByField[assignKey(turn.id, fieldName)];
		const session = currentSession();
		if (session) touchSession(session.id);
	}

	function looksLikeInfobox(text: string): boolean {
		return /\{\{\s*infobox/i.test(text) || text.trim().startsWith('{{');
	}

	// A Wikipedia link runs the exact same extract → predict → format →
	// persist pipeline as a pasted infobox -- the backend just fetches the
	// article's wikitext first (mcp_server/wiki_fetch.py) and hands it to
	// the same extract_first_infobox() call, which already finds "the
	// first infobox-like template" in arbitrarily larger wikitext. So this
	// only needs to decide routing, not build a third turn kind.
	function runsThePipeline(text: string): boolean {
		return looksLikeInfobox(text) || looksLikeWikipediaUrl(text);
	}

	function scrollToBottom() {
		requestAnimationFrame(() => scrollAnchor?.scrollIntoView({ behavior: 'smooth', block: 'end' }));
	}

	async function submit(text?: string) {
		const value = (text ?? input).trim();
		if (!value || busy) return;
		input = '';
		busy = true;

		const session = ensureActiveSession();
		const id = crypto.randomUUID();
		const wantsClass = targetClass.trim() || undefined;

		if (runsThePipeline(value)) {
			session.turns.push({
				id,
				kind: 'pipeline',
				input: value,
				targetClass: wantsClass,
				steps: [],
				mappings: null,
				running: true
			});
			// `session` is an element of the `sessions` $state array, and
			// Svelte 5 wraps a plain object in its own reactive proxy the
			// moment it's inserted into a $state container -- that proxy,
			// not the object literal above, is what the template actually
			// reads. Re-reading the just-pushed element by reference here
			// (rather than continuing to mutate the object literal) is what
			// makes every mutation below actually reach the DOM; mutating
			// the pre-push literal directly is silently invisible to Svelte
			// (confirmed live: the "Thinking…" indicator never clears,
			// steps/mappings never appear, because turn.steps/turn.mappings
			// assignments were landing on a detached object nothing was
			// still watching).
			const turn = session.turns[session.turns.length - 1] as PipelineTurn;
			touchSession(session.id);
			scrollToBottom();
			try {
				for await (const event of previewMapping(value, wantsClass)) {
					if ('mappings' in event) {
						turn.mappings = event.mappings;
						turn.mappingWikitext = event.mappingWikitext;
						turn.xmlRules = event.xmlRules;
						turn.reviewItemId = event.reviewItemId;
						turn.warnings = event.warnings ?? [];
						turn.unmappedFields = event.unmappedFields ?? [];
					} else {
						turn.steps = [...turn.steps, event];
					}
					scrollToBottom();
				}
			} catch (err) {
				console.error('Mapping pipeline failed:', err);
				turn.error =
					err instanceof BackendUnavailableError
						? 'cross-lingual is not reachable — check that the HTTP server (just run-http) is running.'
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
				targetClass: wantsClass,
				matches: null,
				noMatch: false,
				running: true
			});
			// See the comment above the pipeline branch's push -- same fix,
			// same reason.
			const turn = session.turns[session.turns.length - 1] as AnswerTurn;
			touchSession(session.id);
			scrollToBottom();
			try {
				const result = await findSemanticMatch(value, wantsClass);
				if (result.status === 'rejected') {
					turn.error = result.message ?? 'Rejected: prompt-injection attempt detected.';
				} else if (result.status === 'no_match') {
					turn.noMatch = true;
				} else {
					turn.matches = result.matches;
				}
			} catch (err) {
				console.error('Assistant lookup failed:', err);
				turn.error =
					err instanceof BackendUnavailableError
						? 'cross-lingual is not reachable — check that the HTTP server (just run-http) is running.'
						: `Unexpected error: ${err instanceof Error ? err.message : String(err)}`;
			} finally {
				turn.running = false;
				touchSession(session.id);
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

	// Approve/reject right here, on the review-queue row this exact chat
	// turn just created -- previously the only way to act on it was to
	// leave for /review, find the same row again, and decide there.
	// Deliberately approve/reject only, no correction editing here (still a
	// /review-only action) -- but publishing IS available from here too,
	// see `publish()` below, so the whole human-in-the-loop (approve or
	// reject, then optionally publish live) can finish without ever
	// leaving the chat.
	async function decide(turn: PipelineTurn, decision: 'approved' | 'rejected') {
		if (!turn.reviewItemId || turn.deciding) return;
		const id = turn.reviewItemId;
		turn.deciding = true;
		turn.decisionError = undefined;

		// Fields the reviewer assigned by hand in the chat ride along on the
		// approve as `corrected_mappings` -- the decision endpoint already
		// replaces the item's mappings with these before logging the
		// decision as training data, so no separate write is needed.
		const manual = turn.manualMappings ?? [];
		const request =
			decision === 'approved' && manual.length > 0
				? decideReview(id, decision, {
						correctedMappings: [...(turn.mappings ?? []), ...manual]
					})
				: decideReview(id, decision);
		toast.promise(request, {
			loading: decision === 'approved' ? 'Approving…' : 'Rejecting…',
			success: (updated) => {
				turn.reviewStatus = updated.status;
				return `Marked as ${updated.status.replace('_', ' ')}.`;
			},
			error: (err) => {
				turn.decisionError =
					err instanceof DecisionFailedError
						? err.message
						: 'Could not record that decision — backend not reachable.';
				return turn.decisionError;
			}
		});

		try {
			await request;
		} catch {
			// already surfaced via the toast.promise error callback above
		} finally {
			turn.deciding = false;
		}
	}

	// Which turn's publish-confirmation dialog is open, if any -- at most
	// one at a time, matching /review's own `confirmPublishFor` pattern.
	// Publishing is a real, outward-facing, hard-to-reverse write to the
	// live MediaWiki (mcp_server.publish.publish_mapping), so it always
	// goes through this confirmation step, in chat or on /review alike --
	// never a bare one-click action.
	let confirmPublishTurnId = $state<string | null>(null);

	function requestPublish(turn: PipelineTurn) {
		if (!turn.reviewItemId || turn.reviewStatus !== 'approved') return;
		confirmPublishTurnId = turn.id;
	}

	async function publish(turn: PipelineTurn) {
		if (!turn.reviewItemId || turn.publishing) return;
		const id = turn.reviewItemId;
		turn.publishing = true;
		turn.publishError = undefined;

		// decision: 'approved' again is deliberate, not a re-approval --
		// mcp_server.http_app.decide_review always applies the decision
		// (idempotent when already approved) and only then, given
		// `publish: true`, attempts the live write. A failed publish
		// leaves the review at "approved" (never a silent partial state),
		// so the button just becomes retryable.
		const request = decideReview(id, 'approved', { publish: true });
		toast.promise(request, {
			loading: `Publishing ${turn.mappingWikitext ? 'mapping' : 'to the live wiki'}…`,
			success: (updated) => {
				turn.reviewStatus = updated.status;
				return updated.status === 'published'
					? 'Published to the live wiki.'
					: `Marked as ${updated.status.replace('_', ' ')}.`;
			},
			error: (err) => {
				turn.publishError =
					err instanceof DecisionFailedError
						? err.message
						: 'Could not reach the backend to publish.';
				if (err instanceof DecisionFailedError && err.review) {
					turn.reviewStatus = err.review.status;
				}
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
		}
	}

	const examples = [
		{
			label: 'Paste an infobox to map',
			value: `{{Infobox bridge\n| ስም = ደደሳ ድልድይ\n| ርዝመት = 1,700 ሜትር\n}}`
		},
		{ label: 'Ask what a field maps to', value: 'አይካኦ_ኮድ' },
		{
			label: 'Or a Wikipedia link',
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
					<h1 class="text-2xl font-semibold tracking-tight">Amharic → DBpedia Mapping Assistant</h1>
					<p class="mt-2 text-sm text-muted-foreground">
						Paste an infobox, a Wikipedia article link, or ask about a single field. Nothing is
						published until a reviewer approves it on the
						<a href={resolve('/review')} class="text-primary underline underline-offset-4"
							>Review Queue</a
						>.
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
							{#if turn.mappings && turn.mappings.length > 0}
								{@const rows = rowsFor(turn)}
								<div class="overflow-hidden rounded-lg border">
									<Table>
										<TableHeader>
											<TableRow class="hover:bg-transparent">
												<TableHead>Template property</TableHead>
												<TableHead>Ontology property</TableHead>
												<TableHead>Confidence</TableHead>
											</TableRow>
										</TableHeader>
										<TableBody>
											{#each rows as mapping (mapping.templateProperty)}
												<TableRow>
													<TableCell class="font-mono">{mapping.templateProperty}</TableCell>
													<TableCell class="font-mono">
														{mapping.ontologyProperty}
														{#if mapping.source === 'llm'}
															<span
																class="ml-1.5 rounded bg-warning/15 px-1 py-0.5 text-[10px] font-medium text-warning-foreground"
																title="Proposed by the language model because retrieval found nothing -- verify before approving."
															>
																AI-proposed
															</span>
														{:else if mapping.source === 'manual'}
															<span
																class="ml-1.5 rounded bg-accent px-1 py-0.5 text-[10px] font-medium text-accent-foreground"
															>
																added by you
															</span>
														{/if}
													</TableCell>
													<TableCell><ConfidencePill confidence={mapping.confidence} /></TableCell>
												</TableRow>
											{/each}
										</TableBody>
									</Table>
								</div>
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
								{#if turn.reviewStatus === 'approved'}
									<div class="flex flex-wrap items-center gap-2">
										<StatusBadge status={turn.reviewStatus} />
										<Button
											size="sm"
											disabled={turn.publishing}
											onclick={() => requestPublish(turn)}
										>
											<Fa
												icon={turn.publishing ? faSpinner : faCloudUploadAlt}
												class="size-3.5 {turn.publishing ? 'animate-spin' : ''}"
											/>
											Publish to live wiki
										</Button>
										<a
											href={resolve('/review')}
											class="text-xs text-muted-foreground underline underline-offset-4 hover:text-foreground"
										>
											View in Review Queue
										</a>
									</div>
									{#if turn.publishError}
										<p class="text-sm text-destructive">{turn.publishError}</p>
									{/if}
								{:else if turn.reviewStatus}
									<div class="flex w-fit items-center gap-2">
										<StatusBadge status={turn.reviewStatus} />
										<a
											href={resolve('/review')}
											class="text-xs text-muted-foreground underline underline-offset-4 hover:text-foreground"
										>
											View in Review Queue
										</a>
									</div>
								{:else if turn.reviewItemId}
									<div class="flex flex-wrap items-center gap-2">
										<Button
											size="sm"
											variant="outline"
											class="border-success/40 text-success hover:bg-success/10 hover:text-success"
											disabled={turn.deciding}
											onclick={() => decide(turn, 'approved')}
										>
											<Fa
												icon={turn.deciding ? faSpinner : faCircleCheck}
												class="size-3.5 {turn.deciding ? 'animate-spin' : ''}"
											/>
											Approve
										</Button>
										<Button
											size="sm"
											variant="outline"
											class="border-destructive/40 text-destructive hover:bg-destructive/10 hover:text-destructive"
											disabled={turn.deciding}
											onclick={() => decide(turn, 'rejected')}
										>
											<Fa icon={faCircleXmark} class="size-3.5" />
											Reject
										</Button>
										<a
											href={resolve('/review')}
											class="flex items-center gap-1.5 text-xs text-muted-foreground underline underline-offset-4 hover:text-foreground"
										>
											or correct it on the Review Queue
											<Fa icon={faArrowRight} class="size-2.5 shrink-0" />
										</a>
									</div>
									{#if turn.decisionError}
										<p class="text-sm text-destructive">{turn.decisionError}</p>
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
										be mapped automatically. Search the ontology and assign
										{turn.unmappedFields.length === 1 ? 'it' : 'them'} here, or leave
										{turn.unmappedFields.length === 1 ? 'it' : 'them'} for the Review Queue.
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
												{#if entry}
													<div class="mt-2 flex gap-2">
														<Input
															class="h-8 text-xs"
															placeholder="Search the ontology…"
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
							{#if turn.running}
								<ThinkingIndicator label="Searching the ontology" />
							{:else if turn.noMatch}
								<p class="text-sm">No confident match found in the ontology for that term.</p>
							{:else if turn.matches}
								<ul class="flex flex-wrap gap-1.5">
									{#each turn.matches as match (match.property)}
										<li class="flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-sm">
											<span class="font-mono">{match.property}</span>
											<ConfidencePill confidence={match.score} />
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
		<div bind:this={scrollAnchor}></div>
	</div>

	<div class="shrink-0 px-4 pb-4 md:px-8">
		<div class="rounded-2xl border bg-card shadow-sm">
			<div class="px-3 pt-2.5">
				<Input
					bind:value={targetClass}
					placeholder="Target class (optional, e.g. Bridge)"
					class="h-6 border-none bg-transparent px-0 text-xs shadow-none focus-visible:ring-0"
				/>
			</div>
			<div class="flex items-end gap-2 p-2">
				<Textarea
					bind:value={input}
					onkeydown={onKeydown}
					placeholder="Paste an infobox, a Wikipedia link, or ask about a field…"
					rows={1}
					class="max-h-48 min-h-9 resize-none border-none bg-transparent px-2 py-1.5 shadow-none focus-visible:ring-0"
				/>
				<Button
					size="icon"
					class="mb-0.5 size-8 shrink-0 rounded-full"
					disabled={busy || !input.trim()}
					onclick={() => submit()}
					aria-label="Send"
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
					<AlertDialog.Title>Publish this mapping live?</AlertDialog.Title>
					<AlertDialog.Description>
						This writes
						<span class="font-mono text-foreground">{rowsFor(turn).length}</span>
						mapping{rowsFor(turn).length === 1 ? '' : 's'} to
						<span class="font-mono text-foreground">mappings.dbpedia.org</span> immediately, using a MediaWiki
						Bot Password — a real edit, not a preview.
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
