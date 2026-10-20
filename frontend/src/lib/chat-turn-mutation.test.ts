import { describe, expect, it, beforeEach, afterEach, vi } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/svelte';
import ChatPage from '../routes/+page.svelte';
import { chat } from './chat.svelte';
import * as api from './api';
import type { AgentStep, MappingSyntax, PredictedMapping } from './types';

const RUN_ID = 'run-123';

// Exactly what previewMapping() yields: step events, then the result event.
async function* fakeStream(): AsyncGenerator<
	| AgentStep
	| ({
			node: 'result';
			mappings: PredictedMapping[];
			reviewItemId: string | null;
			templateName?: string | null;
			domainClass?: string;
			warnings?: string[];
			unmappedFields?: { name: string; value: string }[];
	  } & MappingSyntax)
> {
	yield { node: 'extract_infobox_fields', status: 'done', detail: 'Extracting' };
	yield { node: 'predict_properties', status: 'done', detail: 'Predicting' };
	yield {
		node: 'result',
		templateName: 'Infobox bridge',
		domainClass: 'Bridge',
		mappings: [{ templateProperty: 'ርዝመት', ontologyProperty: 'length', confidence: 0.75 }],
		mappingWikitext:
			'{{TemplateMapping\n | mapToClass = Bridge\n | mappings =\n  {{PropertyMapping | templateProperty = ርዝመት | ontologyProperty = length }}\n}}',
		xmlRules: '<TemplateMapping mapToClass="dbo:Bridge">...</TemplateMapping>',
		reviewItemId: RUN_ID,
		warnings: ['Skipped 3 field(s) already present in the published Amharic mappings.'],
		unmappedFields: [{ name: 'ካርታ_መግለጫ', value: 'a caption' }]
	};
}

function plan(overrides: Partial<api.InputPlan> = {}): api.InputPlan {
	return {
		intent: 'infobox',
		text: '{{Infobox bridge\n| ርዝመት = 1,700 ሜትር\n}}',
		target_class: 'Bridge',
		language: 'en',
		reply: null,
		source: 'groq',
		...overrides
	};
}

async function send(text: string) {
	render(ChatPage);
	const textarea = screen.getByPlaceholderText(/የመረጃ ሳጥን/) as HTMLTextAreaElement;
	textarea.value = text;
	textarea.dispatchEvent(new Event('input', { bubbles: true }));
	await waitFor(() => {
		expect(screen.getByRole('button', { name: 'ላክ' })).not.toBeDisabled();
	});
	screen.getByRole('button', { name: 'ላክ' }).click();
}

const submitBridgeInfobox = () => send('{{Infobox bridge\n| ርዝመት = 1,700 ሜትር\n}}');

describe('chat: Groq in, pipeline, Groq out, edit and download in place', () => {
	beforeEach(() => {
		chat.turns.length = 0;
		vi.spyOn(api, 'understandInput').mockResolvedValue(plan());
		vi.spyOn(api, 'summarizeResult').mockResolvedValue({
			message: 'I mapped 1 field: `length`. Edit the table, then press Publish.',
			source: 'groq'
		});
	});

	afterEach(() => {
		cleanup();
		vi.restoreAllMocks();
	});

	it('streams the run into an editable table with the Groq reply and notes', async () => {
		const previewSpy = vi.spyOn(api, 'previewMapping').mockReturnValue(fakeStream());

		await submitBridgeInfobox();

		await waitFor(() => expect(screen.getByText('የመለጠፊያ መስኮችን ማውጣት')).toBeInTheDocument());
		// Groq's class is passed on to the pipeline.
		expect(previewSpy).toHaveBeenCalledWith(plan().text, 'Bridge');
		await waitFor(() => expect(screen.getByDisplayValue('length')).toBeInTheDocument());
		expect(screen.getByText('የማዛመጃ wikitext')).toBeInTheDocument();
		expect(screen.getByText('የማዛመጃ XML')).toBeInTheDocument();
		// The pipeline's response is only the mapping card: no Groq summary
		// bubble, no warnings list.
		expect(screen.queryByText(/I mapped 1 field/)).not.toBeInTheDocument();
		expect(
			screen.queryByText(/Skipped 3 field\(s\) already present in the published Amharic mappings/)
		).not.toBeInTheDocument();
		expect(api.summarizeResult).not.toHaveBeenCalled();
	});

	it('has no approve, reject or review-queue step', async () => {
		vi.spyOn(api, 'previewMapping').mockReturnValue(fakeStream());

		await submitBridgeInfobox();
		await screen.findByDisplayValue('length');

		expect(screen.queryByRole('button', { name: /approve/i })).not.toBeInTheDocument();
		expect(screen.queryByRole('button', { name: /reject/i })).not.toBeInTheDocument();
		expect(screen.queryByText(/review queue/i)).not.toBeInTheDocument();
		// Publish is gated off in the UI for now; the primary action is Download.
		expect(screen.queryByRole('button', { name: /publish/i })).not.toBeInTheDocument();
		expect(screen.getByRole('button', { name: /download · 1/i })).toBeInTheDocument();
	});

	it('downloads the current (edited) mapping as {{TemplateMapping}} wikitext', async () => {
		vi.spyOn(api, 'previewMapping').mockReturnValue(fakeStream());
		// publishMapping must not be called from the UI any more.
		const publishSpy = vi.spyOn(api, 'publishMapping');

		await submitBridgeInfobox();
		const cell = (await screen.findByDisplayValue('length')) as HTMLInputElement;
		await fireEvent.input(cell, { target: { value: 'totalLength' } });

		const captured: { name: string | null; text: string | null } = { name: null, text: null };
		const origBlob = globalThis.Blob;
		const origCreate = URL.createObjectURL;
		const origRevoke = URL.revokeObjectURL;
		const origClick = HTMLAnchorElement.prototype.click;
		// jsdom's Blob has no .text(); capture the string synchronously here.
		globalThis.Blob = class extends origBlob {
			constructor(parts: BlobPart[] = [], opts?: BlobPropertyBag) {
				super(parts, opts);
				captured.text = parts.map(String).join('');
			}
		} as typeof Blob;
		URL.createObjectURL = () => 'blob:test';
		URL.revokeObjectURL = () => {};
		HTMLAnchorElement.prototype.click = function () {
			if (this.download) captured.name = this.download;
			origClick.call(this);
		};

		try {
			screen.getByRole('button', { name: /download · 1/i }).click();
		} finally {
			globalThis.Blob = origBlob;
			URL.createObjectURL = origCreate;
			URL.revokeObjectURL = origRevoke;
			HTMLAnchorElement.prototype.click = origClick;
		}

		expect(captured.name).toBe('Mapping_am-Infobox_bridge.wiki');
		expect(captured.text).toBe(
			[
				'{{TemplateMapping',
				' | mapToClass = Bridge',
				' | mappings =',
				'  {{PropertyMapping | templateProperty = ርዝመት | ontologyProperty = totalLength }}',
				'}}'
			].join('\n')
		);
		expect(publishSpy).not.toHaveBeenCalled();
	});

	it('adds an unmapped field from the chat and includes it in the download', async () => {
		vi.spyOn(api, 'previewMapping').mockReturnValue(fakeStream());
		const findSpy = vi.spyOn(api, 'findSemanticMatch').mockResolvedValue({
			status: 'ok',
			matches: [{ property: 'depiction', class: 'Thing', score: 0.9 }]
		});

		await submitBridgeInfobox();
		// The unmapped field is a row of the same table, not a separate box.
		const empty = await screen.findByLabelText('DBpedia ባህርይ ለካርታ_መግለጫ');
		expect(empty).toHaveValue('');
		screen.getByRole('button', { name: 'ለካርታ_መግለጫ ኦንቶሎጂውን ፈልግ' }).click();
		await waitFor(() => expect(findSpy).toHaveBeenCalledWith('ካርታ_መግለጫ', 'Bridge'));

		(await screen.findByRole('button', { name: /depiction/i })).click();
		await waitFor(() => {
			expect(screen.getByLabelText('DBpedia ባህርይ ለካርታ_መግለጫ')).toHaveValue('depiction');
			expect(screen.getByText('በእርስዎ የተጨመረ')).toBeInTheDocument();
			expect(screen.getByTestId('card-status')).toHaveTextContent('ረቂቅ · 1 ተጨምሯል');
		});

		expect(screen.getByRole('button', { name: /download · 2/i })).toBeEnabled();
	});

	it('looks up the field Groq pulled out of a free-form question', async () => {
		vi.spyOn(api, 'understandInput').mockResolvedValue(
			plan({ intent: 'field', text: 'ዋና_አዘጋጅ', target_class: 'Newspaper', language: 'am' })
		);
		const findSpy = vi.spyOn(api, 'findSemanticMatch').mockResolvedValue({
			status: 'ok',
			matches: [{ property: 'editor', class: 'Newspaper', score: 0.3, payload: { source: 'llm' } }]
		});
		const previewSpy = vi.spyOn(api, 'previewMapping');

		await send('የጋዜጣ ዋና_አዘጋጅ በDBpedia ምን ይባላል?');

		await waitFor(() => expect(findSpy).toHaveBeenCalledWith('ዋና_አዘጋጅ', 'Newspaper'));
		expect(previewSpy).not.toHaveBeenCalled();
		await waitFor(() => {
			expect(screen.getByText('ዋና_አዘጋጅ', { selector: 'span' })).toBeInTheDocument();
			expect(screen.getByText('በAI የተጠቆመ')).toBeInTheDocument();
		});
		expect(api.summarizeResult).toHaveBeenCalledWith(
			'field',
			expect.objectContaining({ status: 'ok' }),
			'am'
		);
	});

	it('answers an off-topic message without running anything', async () => {
		vi.spyOn(api, 'understandInput').mockResolvedValue(
			plan({ intent: 'unsupported', text: 'hello', reply: 'I map Amharic infobox fields.' })
		);
		const findSpy = vi.spyOn(api, 'findSemanticMatch');
		const previewSpy = vi.spyOn(api, 'previewMapping');

		await send('hello');

		await waitFor(() =>
			expect(screen.getByText('I map Amharic infobox fields.')).toBeInTheDocument()
		);
		expect(findSpy).not.toHaveBeenCalled();
		expect(previewSpy).not.toHaveBeenCalled();
	});

	it('marks an edited property and Undo all puts the prediction back', async () => {
		vi.spyOn(api, 'previewMapping').mockReturnValue(fakeStream());

		await submitBridgeInfobox();
		const cell = await screen.findByLabelText('DBpedia ባህርይ ለርዝመት');
		expect(screen.getByTestId('card-status')).toHaveTextContent(/^ረቂቅ$/);
		expect(screen.getByText('እስካሁን ምንም ለውጥ የለም')).toBeInTheDocument();

		await fireEvent.input(cell, { target: { value: 'totalLength' } });

		await waitFor(() => {
			expect(screen.getByText(/የቀድሞው፦/)).toBeInTheDocument();
			expect(cell.closest('tr')).toHaveAttribute('data-change', 'edited');
			expect(screen.getByTestId('card-status')).toHaveTextContent('ረቂቅ · 1 ታርሟል');
			expect(screen.getByText('1 ለውጦች')).toBeInTheDocument();
		});

		screen.getByRole('button', { name: 'ሁሉንም መልስ' }).click();
		await waitFor(() => {
			expect(screen.getByLabelText('DBpedia ባህርይ ለርዝመት')).toHaveValue('length');
			expect(screen.queryByText(/የቀድሞው፦/)).not.toBeInTheDocument();
			expect(screen.getByText('እስካሁን ምንም ለውጥ የለም')).toBeInTheDocument();
		});
	});

	it('keeps a removed row in the table, struck through, until it is restored', async () => {
		vi.spyOn(api, 'previewMapping').mockReturnValue(fakeStream());

		await submitBridgeInfobox();
		await screen.findByLabelText('DBpedia ባህርይ ለርዝመት');

		// A predicted row starts with ✓/✗; the trash can appears only after
		// the reviewer acknowledges the prediction.
		expect(screen.queryByRole('button', { name: 'ርዝመት አስወግድ' })).not.toBeInTheDocument();
		screen.getByRole('button', { name: 'ርዝመት ትክክል አይደለም' }).click();
		(await screen.findByRole('button', { name: 'ርዝመት አስወግድ' })).click();
		await waitFor(() => {
			expect(screen.getByText('ተወግዷል፣ አይታተምም')).toBeInTheDocument();
			expect(screen.getByTestId('card-status')).toHaveTextContent('ረቂቅ · 1 ተወግዷል');
			expect(screen.getByRole('button', { name: /download · 0/i })).toBeDisabled();
		});

		screen.getByRole('button', { name: 'ርዝመት ወደነበረበት መልስ' }).click();
		await waitFor(() => {
			expect(screen.getByLabelText('DBpedia ባህርይ ለርዝመት')).toHaveValue('length');
			expect(screen.getByRole('button', { name: /download · 1/i })).toBeEnabled();
		});
	});

	it('leaves class inference to Groq -- the composer has no class picker', async () => {
		const understandSpy = vi.spyOn(api, 'understandInput');
		vi.spyOn(api, 'previewMapping').mockReturnValue(fakeStream());
		render(ChatPage);

		expect(screen.queryByLabelText('ዒላማ ክፍል')).not.toBeInTheDocument();
		expect(screen.queryByText(/ክፍል፦/)).not.toBeInTheDocument();

		const textarea = screen.getByPlaceholderText(/የመረጃ ሳጥን/);
		await fireEvent.input(textarea, { target: { value: '{{Infobox bridge}}' } });
		screen.getByRole('button', { name: 'ላክ' }).click();
		await waitFor(() => expect(understandSpy).toHaveBeenCalledWith('{{Infobox bridge}}'));
	});

	it('decodes percent-encoded Amharic Wikipedia URLs when echoing them back', async () => {
		vi.spyOn(api, 'previewMapping').mockReturnValue(fakeStream());
		vi.spyOn(api, 'understandInput').mockResolvedValue(
			plan({ intent: 'url', text: 'https://am.wikipedia.org/wiki/x', target_class: null })
		);

		await send(
			'https://am.wikipedia.org/wiki/%E1%88%98%E1%88%88%E1%8C%A0%E1%8D%8A%E1%8B%AB:%E1%8B%A8%E1%88%B6%E1%8D%8D%E1%89%B5%E1%8B%8C%E1%88%AD_%E1%88%98%E1%88%A8%E1%8C%83_%E1%88%B3%E1%8C%A5%E1%8A%95'
		);
		await screen.findByText('https://am.wikipedia.org/wiki/መለጠፊያ:የሶፍትዌር_መረጃ_ሳጥን');
	});

	it('a predicted row shows ✓/✗ until the reviewer acts, then the fix tools', async () => {
		vi.spyOn(api, 'previewMapping').mockReturnValue(fakeStream());

		await submitBridgeInfobox();
		await screen.findByLabelText('DBpedia ባህርይ ለርዝመት');

		// Starts in "pending": ✓ and ✗ in the actions column, no search or trash.
		const confirm = screen.getByRole('button', { name: 'ርዝመት ተረጋግጧል' });
		expect(screen.getByRole('button', { name: 'ርዝመት ትክክል አይደለም' })).toBeInTheDocument();
		expect(screen.queryByRole('button', { name: 'ለርዝመት ኦንቶሎጂውን ፈልግ' })).not.toBeInTheDocument();
		expect(screen.queryByRole('button', { name: 'ርዝመት አስወግድ' })).not.toBeInTheDocument();

		// ✓ leaves a confirmed badge and gives the row a delete button (so a
		// confirmed row can still be removed if the reviewer changes their mind).
		confirm.click();
		await waitFor(() => {
			expect(screen.getByText('ተረጋግጧል')).toBeInTheDocument();
			expect(screen.queryByRole('button', { name: 'ርዝመት ተረጋግጧል' })).not.toBeInTheDocument();
			expect(screen.getByRole('button', { name: 'ርዝመት አስወግድ' })).toBeInTheDocument();
		});
	});

	it('✗ on a prediction opens the ontology search for a fix', async () => {
		vi.spyOn(api, 'previewMapping').mockReturnValue(fakeStream());
		const findSpy = vi
			.spyOn(api, 'findSemanticMatch')
			.mockResolvedValue({ status: 'ok', matches: [] });

		await submitBridgeInfobox();
		await screen.findByLabelText('DBpedia ባህርይ ለርዝመት');

		screen.getByRole('button', { name: 'ርዝመት ትክክል አይደለም' }).click();
		await waitFor(() => {
			expect(findSpy).toHaveBeenCalledWith('ርዝመት', 'Bridge');
			expect(screen.getByRole('button', { name: 'ለርዝመት ኦንቶሎጂውን ፈልግ' })).toBeInTheDocument();
			expect(screen.getByRole('button', { name: 'ርዝመት አስወግድ' })).toBeInTheDocument();
		});
	});

	it('typing the predicted property back in is not an edit', async () => {
		vi.spyOn(api, 'previewMapping').mockReturnValue(fakeStream());

		await submitBridgeInfobox();
		const cell = await screen.findByLabelText('DBpedia ባህርይ ለርዝመት');
		await fireEvent.input(cell, { target: { value: 'len' } });
		await fireEvent.input(cell, { target: { value: 'length' } });

		await waitFor(() => {
			expect(cell.closest('tr')).toHaveAttribute('data-change', 'none');
			expect(screen.getByText('75%')).toBeInTheDocument();
			expect(screen.getByText('እስካሁን ምንም ለውጥ የለም')).toBeInTheDocument();
		});
	});
});
