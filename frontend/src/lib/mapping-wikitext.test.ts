import { describe, expect, it } from 'vitest';
import { UnsafeMappingValueError, buildTemplateMapping } from './mapping-wikitext';
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

	// Guard: a value containing MediaWiki parameter separators (|, =, {{,
	// }}, newline) would silently truncate the template on paste. Throw so
	// the caller refuses the download instead of producing broken wikitext.
	it.each([
		['templateProperty', row('width | height', 'length'), 'templateProperty'],
		['templateProperty with =', row('a = b', 'length'), 'templateProperty'],
		['templateProperty with }}', row('foo}}bar', 'length'), 'templateProperty'],
		['templateProperty with newline', row('foo\nbar', 'length'), 'templateProperty'],
		['ontologyProperty with |', row('ok', 'length|width'), 'ontologyProperty'],
		['ontologyProperty with {{', row('ok', '{{sneaky}}'), 'ontologyProperty']
	])('rejects %s', (_label, badRow, field) => {
		expect(() => buildTemplateMapping('Bridge', [badRow])).toThrow(UnsafeMappingValueError);
		try {
			buildTemplateMapping('Bridge', [badRow]);
		} catch (err) {
			expect(err).toBeInstanceOf(UnsafeMappingValueError);
			expect((err as UnsafeMappingValueError).field).toBe(field);
		}
	});

	it('rejects an unsafe mapToClass too', () => {
		expect(() => buildTemplateMapping('Bridge|evil', [])).toThrow(UnsafeMappingValueError);
	});
});
