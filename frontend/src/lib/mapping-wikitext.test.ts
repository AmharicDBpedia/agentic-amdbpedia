import { describe, expect, it } from 'vitest';
import { buildTemplateMapping } from './mapping-wikitext';
import type { PredictedMapping } from './types';

const row = (templateProperty: string, ontologyProperty: string): PredictedMapping => ({
	templateProperty,
	ontologyProperty,
	confidence: 1
});

describe('buildTemplateMapping', () => {
	// The exact bytes the backend writes (mcp_server/publish.py:: _render_template_mapping).
	// Keeping these in sync is deliberate: a page a reviewer downloads and
	// pastes into the wiki parses back cleanly into the project's corpus.
	it('matches the backend format byte for byte', () => {
		expect(buildTemplateMapping('Bridge', [row('ርዝመት', 'length'), row('ስፋት', 'width')])).toBe(
			[
				'{{TemplateMapping',
				' | mapToClass = Bridge',
				' | mappings =',
				'  {{PropertyMapping | templateProperty = ርዝመት | ontologyProperty = length }}',
				'  {{PropertyMapping | templateProperty = ስፋት | ontologyProperty = width }}',
				'}}'
			].join('\n')
		);
	});

	it('still emits the mapToClass line when there are no rows', () => {
		expect(buildTemplateMapping('Zoo', [])).toBe(
			'{{TemplateMapping\n | mapToClass = Zoo\n | mappings =\n}}'
		);
	});
});
