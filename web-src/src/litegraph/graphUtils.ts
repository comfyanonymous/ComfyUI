// These are utils that operate on an `app instance`

import { ComfyGraph } from './comfyGraph.js';
import { formatDate } from '../scripts/utils.js';

export function applyTextReplacements(graph: ComfyGraph, value: string) {
    return value.replace(/%([^%]+)%/g, function (match, text) {
        const split = text.split('.');
        if (split.length !== 2) {
            // Special handling for dates
            if (split[0].startsWith('date:')) {
                return formatDate(split[0].substring(5), new Date());
            }

            if (text !== 'width' && text !== 'height') {
                // Dont warn on standard replacements
                console.warn('Invalid replacement pattern', text);
            }
            return match;
        }

        // Find node with matching S&R property name
        let nodes = graph.nodes.filter(n => n.properties?.['Node name for S&R'] === split[0]);
        // If we cant, see if there is a node with that title
        if (!nodes?.length) {
            nodes = graph?.nodes.filter(n => n.title === split[0]);
        }
        if (!nodes?.length) {
            console.warn('Unable to find node', split[0]);
            return match;
        }

        if (nodes?.length > 1) {
            console.warn('Multiple nodes matched', split[0], 'using first match');
        }

        const node = nodes[0];

        const widget = node.widgets?.find(w => w.name === split[1]);
        if (!widget) {
            console.warn('Unable to find widget', split[1], 'on node', split[0], node);
            return match;
        }

        return ((widget.value ?? '') + '').replaceAll(/\/|\\/g, '_');
    });
}

export const loadWorkflow = async (): Promise<boolean> => {
    try {
        const json = localStorage.getItem('workflow');
        if (json) {
            const workflow = JSON.parse(json);
            await this.loadGraphData(workflow);
            return true;
        }
    } catch (err) {
        console.error('Error loading previous workflow', err);
    }

    return false;
};
