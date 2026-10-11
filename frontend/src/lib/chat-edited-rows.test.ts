import { describe, expect, it } from 'vitest';
import { withEditedRows, type ChatSession, type PipelineTurn } from './chat.svelte';
import type { PredictedMapping } from './types';

const row = (templateProperty: string, ontologyProperty: string) =>
	({ templateProperty, ontologyProperty, confidence: 1 }) as PredictedMapping;

describe('withEditedRows (saved chats from before in-chat editing)', () => {
	it('fills edited rows at load time so the template never writes state', () => {
		const turn = {
			kind: 'pipeline',
			mappings: [row('ስም', 'name')],
			manualMappings: [row('ከፍታ', 'elevation')]
		} as unknown as PipelineTurn;
		const [session] = withEditedRows([{ id: 's', turns: [turn] } as unknown as ChatSession]);
		const edited = (session.turns[0] as PipelineTurn).edited!;
		expect(edited.map((r) => r.ontologyProperty)).toEqual(['name', 'elevation']);
		// A copy, so editing the table never changes the original prediction.
		expect(edited[0]).not.toBe(turn.mappings![0]);
	});

	it('leaves existing edits and turns without mappings alone', () => {
		const edited = [row('ስም', 'foaf:name')];
		const kept = {
			kind: 'pipeline',
			mappings: [row('ስም', 'name')],
			edited
		} as unknown as PipelineTurn;
		const pending = { kind: 'pipeline', mappings: null } as unknown as PipelineTurn;
		withEditedRows([{ id: 's', turns: [kept, pending] } as unknown as ChatSession]);
		expect(kept.edited).toBe(edited);
		expect(pending.edited).toBeUndefined();
	});
});
