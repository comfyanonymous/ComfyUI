import React from 'react';
import { createContext, useState, useEffect, ReactNode } from 'react';
import { createUseContextHook } from './hookCreator';
import { LastNodeErrors, QueueItem } from '../types/many';
import { ApiEventEmitter } from './apiContext';
import { ComfyMessage, ComfyMessage_QueueStatus, WorkflowStep } from '../../autogen_web_ts/comfy_request.v1';
import { useApiContext } from './apiContext';
import { useGraph } from './graphContext';

export interface IJobQueueContext {
    queue: QueueItem[];
    lastNodeErrors: LastNodeErrors;
}

export const JobQueueContextProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
    const [queue, setQueue] = useState<QueueItem[]>([]);
    const [lastNodeErrors, setLastNodeErrors] = useState<LastNodeErrors>({});
    const { graph, canvas } = useGraph();
    const { runWorkflow } = useApiContext();

    // Update the queue-state as messages come in from the API
    useEffect(() => {
        const update = ({ detail }: { detail: ComfyMessage_QueueStatus }) => {
            // TO DO: update queue and last node errors based on this message
            // setQueue(message);
            // setLastNodeErrors(message);
            // this.lastNodeErrors = res.node_errors;
            //         if (this.lastNodeErrors) {
            //             let errors = Array.isArray(this.lastNodeErrors)
            //                 ? this.lastNodeErrors
            //                 : Object.keys(this.lastNodeErrors);
            //             if (errors.length > 0) {
            //                 this.canvas?.draw(true, true);
            //             }
            //         }
            //     } catch (error: unknown) {
            //         const err = error as ComfyPromptError;
            //         const formattedError = this.#formatPromptError(err);
            //         this.ui.dialog.show(formattedError);
            //         if (err.response) {
            //             this.lastNodeErrors = err.response.node_errors;
            //             this.canvas?.draw(true, true);
            //         }
            //         break;
            //     }
        };

        const abortController = new AbortController();

        ApiEventEmitter.addEventListener('queue_status', update, { signal: abortController.signal });

        return () => {
            abortController.abort();
        };
    }, []);

    // When a job completes, allow widgets to run callbacks
    // e.g. random seed after every gen
    function runNodeCallbacks(workflow: Record<string, WorkflowStep>) {
        for (const nodeId of Object.keys(workflow)) {
            // TO DO: this assumes that nodeIds are always stringified numbers, which is the default for LiteGraph.js
            const node = graph.getNodeById(Number(nodeId));
            if (node?.widgets) {
                for (const widget of node.widgets) {
                    if (widget.afterQueued) {
                        widget.afterQueued();
                    }
                }
            }
        }

        canvas.draw(true, true);
    }

    // This is the oriignal queue-prompt functionality
    // Idk if it's useful for anything anymore?
    async function queuePrompt(batchCount: number = 1) {
        try {
            for (let i = 0; i < batchCount; i++) {
                const { serializedGraph, apiWorkflow } = graph.serializeGraph();
                try {
                    const res = await runWorkflow(apiWorkflow, serializedGraph);

                    runNodeCallbacks(apiWorkflow);
                    // ui.queue.update();
                } catch (error: unknown) {
                    console.error(error);
                }
            }
        } catch (error: unknown) {
            console.error(error);
        }
    }

    return <JobQueueContext.Provider value={{ queue, lastNodeErrors }}>{children}</JobQueueContext.Provider>;
};

const JobQueueContext = createContext<IJobQueueContext | undefined>(undefined);
export const useJobQueue = createUseContextHook(
    JobQueueContext,
    'useJobQueueContext must be used within a JobQueueContextProvider'
);
