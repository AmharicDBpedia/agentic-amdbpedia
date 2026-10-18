import { describe, expect, it } from 'vitest';
import { changesFor, removedRows, rowChange } from './mapping-diff';
import type { PredictedMapping } from './types';

const row = (templateProperty: string, ontologyProperty: string): PredictedMapping => ({
	templateProperty,
	ontologyProperty,
	confidence: 0.8
});

const predicted = [row('ስም', 'name'), row('ርዝመት', 'length'), row('ወንዝ', 'river')];

describe('mapping-diff', () => {
	it('counts nothing for an untouched prediction', () => {
		expect(
			changesFor(
				predicted,
				predicted.map((r) => ({ ...r }))
			)
		).toEqual({
			edited: 0,
			added: 0,
			removed: 0,
			total: 0
		});
	});

	it('tells edited, added and removed rows apart by infobox field', () => {
		const rows = [row('ስም', 'name'), row('ርዝመት', 'totalLength'), row('ካርታ', 'depiction')];
		expect(rowChange(predicted, rows[0])).toBeNull();
		expect(rowChange(predicted, rows[1])).toBe('edited');
		expect(rowChange(predicted, rows[2])).toBe('added');
		expect(removedRows(predicted, rows).map((r) => r.templateProperty)).toEqual(['ወንዝ']);
		expect(changesFor(predicted, rows)).toEqual({ edited: 1, added: 1, removed: 1, total: 3 });
	});
});
