import { describe, expect, it, beforeEach, afterEach, vi } from 'vitest';
import { cleanup, render, screen, waitFor } from '@testing-library/svelte';
import ChatPage from '../routes/+page.svelte';
import { chat } from './chat.svelte';
import * as api from './api';
import type { AgentStep, MappingSyntax, PredictedMapping } from './types';

const GERD_URL = 'https://am.wikipedia.org/wiki/ታላቁ_የኢትዮጵያ_ሕዳሴ_ግድብ';

describe('extracting an infobox from a Wikipedia link, in the chat', () => {
	beforeEach(() => {
		chat.turns.length = 0;
		// Groq routing/feedback are server calls; keep them local here.
		vi.spyOn(api, 'understandInput').mockImplementation(async (text, cls) => ({
			intent: api.looksLikeWikipediaUrl(text) ? 'url' : 'field',
			text,
			target_class: cls ?? null,
			language: 'en',
			reply: null,
			source: 'rules'
		}));
		vi.spyOn(api, 'summarizeResult').mockResolvedValue(null);
	});

	afterEach(() => {
		cleanup();
		vi.restoreAllMocks();
	});

	it('routes a pasted Wikipedia link through the pipeline (not the plain-question lookup)', async () => {
		async function* fakeStream(): AsyncGenerator<
			| AgentStep
			| ({
					node: 'result';
					mappings: PredictedMapping[];
					reviewItemId: string | null;
			  } & MappingSyntax)
		> {
			yield {
				node: 'fetch_source_article',
				status: 'done',
				detail: 'Fetched "ታላቁ_የኢትዮጵያ_ሕዳሴ_ግድብ" from Wikipedia'
			};
			yield { node: 'extract_infobox_fields', status: 'done', detail: 'Extracting' };
			yield { node: 'predict_properties', status: 'done', detail: 'Predicting' };
			yield { node: 'format_mapping_syntax', status: 'done', detail: 'Formatting' };
			yield { node: 'persist_review_item', status: 'done', detail: 'Saving' };
			yield {
				node: 'result',
				mappings: [{ templateProperty: 'ወንዝ', ontologyProperty: 'river', confidence: 0.6 }],
				mappingWikitext: '',
				xmlRules: '',
				reviewItemId: 'review-gerd'
			};
		}
		const previewSpy = vi.spyOn(api, 'previewMapping').mockReturnValue(fakeStream());
		const findSemanticMatchSpy = vi.spyOn(api, 'findSemanticMatch');

		render(ChatPage);

		const textarea = screen.getByPlaceholderText(/የመረጃ ሳጥን/) as HTMLTextAreaElement;
		textarea.value = GERD_URL;
		textarea.dispatchEvent(new Event('input', { bubbles: true }));

		await waitFor(() => {
			expect(screen.getByRole('button', { name: 'ላክ' })).not.toBeDisabled();
		});
		screen.getByRole('button', { name: 'ላክ' }).click();

		// The actual thing under test: a pasted link must run the pipeline
		// (previewMapping), never the plain-question lookup
		// (findSemanticMatch) -- confirmed by which real api.ts function
		// actually got called, not just what rendered.
		await waitFor(() => {
			expect(previewSpy).toHaveBeenCalledWith(GERD_URL, undefined);
		});
		expect(findSemanticMatchSpy).not.toHaveBeenCalled();

		// The fetch step is real UI, not just a silently-consumed event --
		// StepTracker's label map (frontend/src/lib/components/StepTracker.svelte)
		// has to know this node name or it would fall back to the raw
		// "fetch_source_article" node string instead.
		await waitFor(() => {
			expect(screen.getByText('የዊኪፔዲያ መለጠፊያን ማምጣት')).toBeInTheDocument();
			expect(screen.getByDisplayValue('river')).toBeInTheDocument();
		});
	});

	it('surfaces a failed fetch as a visible error step, not a silent no-op', async () => {
		async function* failingStream(): AsyncGenerator<
			| AgentStep
			| ({
					node: 'result';
					mappings: PredictedMapping[];
					reviewItemId: string | null;
			  } & MappingSyntax)
		> {
			yield {
				node: 'fetch_source_article',
				status: 'error',
				detail: 'Could not reach am.wikipedia.org: HTTP 404: Not Found'
			};
			yield {
				node: 'result',
				mappings: [],
				mappingWikitext: '',
				xmlRules: '',
				reviewItemId: null
			};
		}
		vi.spyOn(api, 'previewMapping').mockReturnValue(failingStream());

		render(ChatPage);

		const textarea = screen.getByPlaceholderText(/የመረጃ ሳጥን/) as HTMLTextAreaElement;
		textarea.value = 'https://am.wikipedia.org/wiki/ThisPageDoesNotExist';
		textarea.dispatchEvent(new Event('input', { bubbles: true }));

		await waitFor(() => {
			expect(screen.getByRole('button', { name: 'ላክ' })).not.toBeDisabled();
		});
		screen.getByRole('button', { name: 'ላክ' }).click();

		await waitFor(() => {
			expect(screen.getByText('የዊኪፔዲያ መለጠፊያን ማምጣት')).toBeInTheDocument();
			expect(screen.getByText(/404: Not Found/)).toBeInTheDocument();
		});
	});
});
