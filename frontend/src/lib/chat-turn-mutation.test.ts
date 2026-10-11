import { describe, expect, it, beforeEach, afterEach, vi } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/svelte';
import ChatPage from '../routes/+page.svelte';
import { sessions, active } from './chat.svelte';
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
	const textarea = screen.getByPlaceholderText(/Paste an infobox/i) as HTMLTextAreaElement;
	textarea.value = text;
	textarea.dispatchEvent(new Event('input', { bubbles: true }));
	await waitFor(() => {
		expect(screen.getByRole('button', { name: /send/i })).not.toBeDisabled();
	});
	screen.getByRole('button', { name: /send/i }).click();
}

const submitBridgeInfobox = () => send('{{Infobox bridge\n| ርዝመት = 1,700 ሜትር\n}}');

describe('chat: Groq in, pipeline, Groq out, edit and publish in place', () => {
	beforeEach(() => {
		sessions.length = 0;
		active.id = null;
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

		await waitFor(() => expect(screen.getByText('Extract infobox fields')).toBeInTheDocument());
		// Groq's class is passed on to the pipeline.
		expect(previewSpy).toHaveBeenCalledWith(plan().text, 'Bridge');
		await waitFor(() => expect(screen.getByDisplayValue('length')).toBeInTheDocument());
		await waitFor(() => {
			expect(screen.getByText(/I mapped 1 field/)).toBeInTheDocument();
			expect(screen.getByText('View mapping wikitext')).toBeInTheDocument();
			expect(screen.getByText('View mapping XML')).toBeInTheDocument();
			expect(
				screen.getByText(/Skipped 3 field\(s\) already present in the published Amharic mappings/)
			).toBeInTheDocument();
		});
		expect(api.summarizeResult).toHaveBeenCalledWith(
			'infobox',
			expect.objectContaining({ domainClass: 'Bridge' }),
			'en'
		);
	});

	it('has no approve, reject or review-queue step', async () => {
		vi.spyOn(api, 'previewMapping').mockReturnValue(fakeStream());

		await submitBridgeInfobox();
		await screen.findByDisplayValue('length');

		expect(screen.queryByRole('button', { name: /approve/i })).not.toBeInTheDocument();
		expect(screen.queryByRole('button', { name: /reject/i })).not.toBeInTheDocument();
		expect(screen.queryByText(/review queue/i)).not.toBeInTheDocument();
		expect(screen.getByRole('button', { name: /publish 1 mapping/i })).toBeInTheDocument();
	});

	it('publishes the edited table only after the confirmation dialog', async () => {
		vi.spyOn(api, 'previewMapping').mockReturnValue(fakeStream());
		const publishSpy = vi.spyOn(api, 'publishMapping').mockResolvedValue({
			status: 'published',
			title: 'Mapping am:Infobox bridge',
			url: 'https://mappings.dbpedia.org/index.php/Mapping_am:Infobox_bridge'
		});

		await submitBridgeInfobox();
		const cell = (await screen.findByDisplayValue('length')) as HTMLInputElement;

		// Edit right in the chat.
		await fireEvent.input(cell, { target: { value: 'totalLength' } });

		screen.getByRole('button', { name: /publish 1 mapping/i }).click();
		expect(publishSpy).not.toHaveBeenCalled();

		(await screen.findByRole('button', { name: /^publish$/i })).click();
		await waitFor(() => {
			expect(publishSpy).toHaveBeenCalledWith({
				templateName: 'Infobox bridge',
				domainClass: 'Bridge',
				mappings: [{ templateProperty: 'ርዝመት', ontologyProperty: 'totalLength', confidence: 1 }],
				originalMappings: [
					{ templateProperty: 'ርዝመት', ontologyProperty: 'length', confidence: 0.75 }
				],
				reviewItemId: RUN_ID
			});
		});
		await waitFor(() => {
			expect(screen.getByText('Mapping am:Infobox bridge')).toBeInTheDocument();
			expect(screen.queryByRole('button', { name: /publish 1 mapping/i })).not.toBeInTheDocument();
		});
	});

	it('shows the server reason when a publish fails', async () => {
		vi.spyOn(api, 'previewMapping').mockReturnValue(fakeStream());
		vi.spyOn(api, 'publishMapping').mockRejectedValue(
			new api.PublishFailedError(
				'MediaWiki bot credentials are not configured',
				'mediawiki_credentials'
			)
		);

		await submitBridgeInfobox();
		await screen.findByDisplayValue('length');
		screen.getByRole('button', { name: /publish 1 mapping/i }).click();
		(await screen.findByRole('button', { name: /^publish$/i })).click();

		await waitFor(() => {
			expect(screen.getByText('MediaWiki bot credentials are not configured')).toBeInTheDocument();
		});
	});

	it('adds an unmapped field from the chat and publishes it with the rest', async () => {
		vi.spyOn(api, 'previewMapping').mockReturnValue(fakeStream());
		const findSpy = vi.spyOn(api, 'findSemanticMatch').mockResolvedValue({
			status: 'ok',
			matches: [{ property: 'depiction', class: 'Thing', score: 0.9 }]
		});
		const publishSpy = vi.spyOn(api, 'publishMapping').mockResolvedValue({
			status: 'published',
			title: 'Mapping am:Infobox bridge',
			url: 'https://mappings.dbpedia.org/index.php/Mapping_am:Infobox_bridge'
		});

		await submitBridgeInfobox();
		await screen.findByText(/couldn't be mapped automatically/i);
		screen.getByRole('button', { name: /^search$/i }).click();
		await waitFor(() => expect(findSpy).toHaveBeenCalled());

		(await screen.findByRole('button', { name: /depiction/i })).click();
		await waitFor(() => {
			expect(screen.getByDisplayValue('depiction')).toBeInTheDocument();
			expect(screen.queryByText(/couldn't be mapped automatically/i)).not.toBeInTheDocument();
		});

		screen.getByRole('button', { name: /publish 2 mappings/i }).click();
		(await screen.findByRole('button', { name: /^publish$/i })).click();
		await waitFor(() => {
			expect(publishSpy).toHaveBeenCalledWith(
				expect.objectContaining({
					mappings: [
						{ templateProperty: 'ርዝመት', ontologyProperty: 'length', confidence: 0.75 },
						{
							templateProperty: 'ካርታ_መግለጫ',
							ontologyProperty: 'depiction',
							confidence: 0.9,
							source: 'manual'
						}
					]
				})
			);
		});
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
			expect(screen.getByText('AI-proposed')).toBeInTheDocument();
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
});
