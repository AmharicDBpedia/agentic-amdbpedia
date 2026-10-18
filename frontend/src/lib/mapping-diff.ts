// What the user changed in a mapping card, compared with the model's
// prediction. Rows are matched by template property (the infobox field).
import type { PredictedMapping } from './types';

export type RowChange = 'edited' | 'added' | null;

export interface MappingChanges {
	edited: number;
	added: number;
	removed: number;
	total: number;
}

export function originalFor(
	original: PredictedMapping[],
	templateProperty: string
): PredictedMapping | undefined {
	return original.find((row) => row.templateProperty === templateProperty);
}

export function rowChange(original: PredictedMapping[], row: PredictedMapping): RowChange {
	const source = originalFor(original, row.templateProperty);
	if (!source) return 'added';
	return source.ontologyProperty !== row.ontologyProperty ? 'edited' : null;
}

export function removedRows(
	original: PredictedMapping[],
	rows: PredictedMapping[]
): PredictedMapping[] {
	return original.filter((o) => !rows.some((row) => row.templateProperty === o.templateProperty));
}

export function changesFor(original: PredictedMapping[], rows: PredictedMapping[]): MappingChanges {
	let edited = 0;
	let added = 0;
	for (const row of rows) {
		const change = rowChange(original, row);
		if (change === 'edited') edited++;
		else if (change === 'added') added++;
	}
	const removed = removedRows(original, rows).length;
	return { edited, added, removed, total: edited + added + removed };
}
