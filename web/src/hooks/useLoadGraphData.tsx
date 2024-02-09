import React, { ReactNode, useState } from 'react';
import { defaultGraph } from '../scripts/defaultGraph.ts';
import { sanitizeNodeName } from '../scripts/utils.ts';
import { ComfyError } from '../types/many.ts';
import { ErrorHint } from '../components/errors/ErrorHint.tsx';
import { WorkflowLoadError } from '../components/errors/WorkflowLoadError.tsx';
import { useComfyDialog } from '../context/comfyDialogContext.tsx';
import { useComfyApp } from '../context/appContext.tsx';
import { useGraph } from '../context/graphContext.tsx';
import { MissingNodesError } from '../components/errors/MissingNodesError.tsx';
import { LiteGraph } from 'litegraph.js';

// import { logging } from '../scripts/logging.ts';

export function useLoadGraphData() {
    const { clean: cleanApp } = useComfyApp();
    const { showDialog } = useComfyDialog();
    const { graph } = useGraph();
    const [errorHint, setErrorHint] = useState<ReactNode[]>([]);

    const loadGraphData = (graphData?: any, clean: boolean = true) => {
        if (clean) {
            cleanApp();
        }

        let reset_invalid_values = false;
        if (!graphData) {
            graphData = defaultGraph;
            reset_invalid_values = true;
        }

        if (typeof structuredClone === 'undefined') {
            graphData = JSON.parse(JSON.stringify(graphData));
        } else {
            graphData = structuredClone(graphData);
        }

        const missingNodeTypes: string[] = [];
        // await extensionManager.invokeExtensionsAsync('beforeConfigureGraph', graphData, missingNodeTypes);
        for (const k in graphData.nodes) {
            const n = graphData.nodes[k];

            // Patch T2IAdapterLoader to ControlNetLoader since they are the same node now
            if (n.type == 'T2IAdapterLoader') n.type = 'ControlNetLoader';
            if (n.type == 'ConditioningAverage ') n.type = 'ConditioningAverage'; //typo fix
            if (n.type == 'SDV_img2vid_Conditioning') n.type = 'SVD_img2vid_Conditioning'; //typo fix

            // Find missing node types
            if (!(n.type in LiteGraph.registered_node_types)) {
                missingNodeTypes.push(n.type);
                n.type = sanitizeNodeName(n.type);
            }
        }

        try {
            graph.configure(graphData);
        } catch (error) {
            const err = error as ComfyError;
            // Try extracting filename to see if it was caused by an extension script
            const filename = err.fileName || (err.stack || '').match(/(\/extensions\/.*\.js)/)?.[1];
            const pos = (filename || '').indexOf('/extensions/');
            if (pos > -1) {
                setErrorHint(prevHints => [
                    ...prevHints,
                    <ErrorHint key={filename} script={filename?.substring(pos) ?? ''} />,
                ]);
            }

            // Show dialog to let the user know something went wrong loading the data
            showDialog(<WorkflowLoadError err={err} errorHint={errorHint} />);
            return;
        }

        for (const node of graph?.nodes || []) {
            const size = node.computeSize();
            size[0] = Math.max(node.size[0], size[0]);
            size[1] = Math.max(node.size[1], size[1]);
            node.size = size;

            if (node.widgets) {
                // If you break something in the backend and want to patch workflows in the frontend
                // This is the place to do this
                for (const widget of node.widgets) {
                    if (node.type == 'KSampler' || node.type == 'KSamplerAdvanced') {
                        if (widget.name == 'sampler_name') {
                            if (widget.value.startsWith('sample_')) {
                                widget.value = widget.value.slice(7);
                            }
                        }
                    }
                    if (node.type == 'KSampler' || node.type == 'KSamplerAdvanced' || node.type == 'PrimitiveNode') {
                        if (widget.name == 'control_after_generate') {
                            if (widget.value === true) {
                                widget.value = 'randomize';
                            } else if (widget.value === false) {
                                widget.value = 'fixed';
                            }
                        }
                    }
                    if (reset_invalid_values) {
                        if (widget.type == 'combo') {
                            if (!widget.options.values.includes(widget.value) && widget.options.values.length > 0) {
                                widget.value = widget.options.values[0];
                            }
                        }
                    }
                }
            }

            // TO DO: check if this behavior changed at all; we went from
            // invokeExtensions to invokeExtensionsAsync here
            // extensionManager.invokeExtensionsAsync('loadedGraphNode', node);
        }

        if (missingNodeTypes.length) {
            showDialog(<MissingNodesError nodeTypes={missingNodeTypes} />);
            // TODO: Add logging
            // logging.addEntry('Comfy.App', 'warn', {
            //     MissingNodes: missingNodeTypes,
            // });
        }

        // await extensionManager.invokeExtensionsAsync('afterConfigureGraph', missingNodeTypes);
    };

    const loadWorkflow = (): boolean => {
        try {
            const json = localStorage.getItem('workflow');
            const hasWorkflow = json !== '{}' && !!json;

            if (hasWorkflow) {
                const workflow = JSON.parse(json);
                loadGraphData(workflow);
                return true;
            }
        } catch (err) {
            console.error('Error loading previous workflow', err);
        }

        return false;
    };

    return {
        loadGraphData,
        loadWorkflow,
    };
}
