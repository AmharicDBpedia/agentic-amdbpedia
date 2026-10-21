// Builds the {{TemplateMapping | mapToClass = ... | mappings = ...}}
// wikitext a Mapping am:<template> page expects. The format matches the
// backend's `mcp_server/publish.py::_render_template_mapping` byte-for-byte
// so a page assembled from this download reads back cleanly into the
// repo's own corpus the next time it refreshes.
import type { PredictedMapping } from './types';

export function buildTemplateMapping(
	domainClass: string,
	mappings: readonly PredictedMapping[]
): string {
	const lines = ['{{TemplateMapping', ` | mapToClass = ${domainClass}`, ' | mappings ='];
	for (const row of mappings) {
		lines.push(
			`  {{PropertyMapping | templateProperty = ${row.templateProperty}` +
				` | ontologyProperty = ${row.ontologyProperty} }}`
		);
	}
	lines.push('}}');
	return lines.join('\n');
}
