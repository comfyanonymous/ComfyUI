// This class tracks state of the job-queue, and provides methods to add or remove jobs from it

import { api } from './api';
import { ComfyCanvas } from './comfyCanvas';

export type QueueItem = {
    number: number;
    batchCount: number;
};

export class JobQueue {
    /** List of entries to queue */
    #queueItems: QueueItem[] = [];
    get queueItems(): QueueItem[] {
        return this.#queueItems;
    }

    /** If the queue is currently being processed */
    #processingQueue: boolean = false;
    get processingQueue(): boolean {
        return this.#processingQueue;
    }

    /** The last node errors from the API */
    lastNodeErrors = null;

    /** Reference to a ComfyCanvas that can be drawn to */
    canvas: ComfyCanvas;

    constructor(canvas: ComfyCanvas) {
        this.canvas = canvas;
    }

    async queuePrompt(number: number, batchCount = 1) {
        this.#queueItems.push({ number, batchCount });

        // Only have one action process the items so each one gets a unique seed correctly
        if (this.#processingQueue) {
            return;
        }

        this.#processingQueue = true;
        this.lastNodeErrors = null;

        try {
            while (this.#queueItems.length > 0) {
                const queueItem = this.#queueItems.pop();
                if (queueItem) {
                    ({ number, batchCount } = queueItem);

                    for (let i = 0; i < batchCount; i++) {
                        const p = await this.canvas.graph.graphToWorkflow();

                        try {
                            const res = await this.api.queuePrompt(number, p);
                            this.lastNodeErrors = res.node_errors;

                            if (this.lastNodeErrors) {
                                let errors = Array.isArray(this.lastNodeErrors)
                                    ? this.lastNodeErrors
                                    : Object.keys(this.lastNodeErrors);
                                if (errors.length > 0) {
                                    this.canvas?.draw(true, true);
                                }
                            }
                        } catch (error: unknown) {
                            const err = error as ComfyPromptError;

                            const formattedError = this.#formatPromptError(err);
                            this.ui.dialog.show(formattedError);
                            if (err.response) {
                                this.lastNodeErrors = err.response.node_errors;
                                this.canvas?.draw(true, true);
                            }
                            break;
                        }

                        if (p.workflow) {
                            for (const n of p.workflow.nodes) {
                                const node = this.graph?.getNodeById(n.id);
                                if (node?.widgets) {
                                    for (const widget of node.widgets) {
                                        // Allow widgets to run callbacks after a prompt has been queued
                                        // e.g. random seed after every gen
                                        if (widget.afterQueued) {
                                            widget.afterQueued();
                                        }
                                    }
                                }
                            }
                        }

                        this.canvas?.draw(true, true);
                        await this.ui.queue.update();
                    }
                }
            }
        } finally {
            this.#processingQueue = false;
        }
        this.api.dispatchEvent(new CustomEvent('promptQueued', { detail: { number, batchCount } }));
    }
}
