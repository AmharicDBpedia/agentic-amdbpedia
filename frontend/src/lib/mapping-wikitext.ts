// Builds the {{TemplateMapping | mapToClass = ... | mappings = ...}}
// wikitext a Mapping am:<template> page expects. The format matches the
// backend's `mcp_server/publish.py::_render_template_mapping` byte-for-byte
// so a page assembled from this download reads back cleanly into the
// repo's own corpus the next time it refreshes.
import type { PredictedMapping } from './types';

export class UnsafeMappingValueError extends Error {
	constructor(
		public readonly field: 'templateProperty' | 'ontologyProperty' | 'mapToClass',
		public readonly value: string
	) {
		super(
			`${field}=${JSON.stringify(value)} contains a character MediaWiki template syntax can't carry`
		);
	}
}

// MediaWiki's {{Template|k=v|...}} syntax uses these as parameter separators
// -- a value containing any of them would terminate the parameter early and
// quietly corrupt the mapping on paste. Newline is forbidden for the same
// reason: the pipeline parser (rag/ontology.py's AmharicMappingIndex) reads
// one PropertyMapping per line.
const UNSAFE = /[|=\n\r]|\{\{|\}\}/;

function assertSafe(
	field: 'templateProperty' | 'ontologyProperty' | 'mapToClass',
	value: string
): void {
	if (UNSAFE.test(value)) throw new UnsafeMappingValueError(field, value);
}

export function buildTemplateMapping(
	domainClass: string,
	mappings: readonly PredictedMapping[]
): string {
	assertSafe('mapToClass', domainClass);
	const lines = ['{{TemplateMapping', ` | mapToClass = ${domainClass}`, ' | mappings ='];
	for (const row of mappings) {
		assertSafe('templateProperty', row.templateProperty);
		assertSafe('ontologyProperty', row.ontologyProperty);
		lines.push(
			`  {{PropertyMapping | templateProperty = ${row.templateProperty}` +
				` | ontologyProperty = ${row.ontologyProperty} }}`
		);
	}
	lines.push('}}');
	return lines.join('\n');
}
