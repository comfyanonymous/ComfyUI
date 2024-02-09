import { ComfyObjectInfo } from '../types/comfy';
import {
    EmbeddingsResponse,
    HistoryResponse,
    IComfyApi,
    QueueResponse,
    SettingsResponse,
    SystemStatsResponse,
    UserConfigResponse,
} from '../types/api';
import { WorkflowStep } from '../types/many';
import { SerializedGraph } from '../types/litegraph';
import { JobCreated } from '../../autogen_web_ts/comfy_request.v1';

type storeUserDataOptions = RequestInit & { stringify?: boolean; throwOnError?: boolean };

export class ComfyApi extends EventTarget implements IComfyApi {
    socket: WebSocket | null = null;
    api_host: string;
    api_base: string;
    user: string | undefined;
    clientId: string | undefined;

    /** Set of custom message types */
    #registered: Set<string>;

    constructor(host?: string) {
        super();
        this.#registered = new Set();
        this.api_host = host ?? location.host;
        this.api_base = location.pathname.split('/').slice(0, -1).join('/');
        this.clientId = sessionStorage.getItem('clientId')!;
    }

    /** Initialises sockets for realtime updates */
    init() {
        this.#connectToServer();
    }

    apiURL(route: string) {
        return this.api_host + route;
    }

    fetchApi(route: string, options?: RequestInit) {
        if (!options) {
            options = {};
        }
        if (!options.headers) {
            options.headers = {};
        }
        // Assuming `this.user` is of type `string | undefined`
        // (options.headers as Record<string, string>)['Comfy-User'] = this.user || '';

        // return fetch(this.apiURL(route), options) as Promise<T>;
        return fetch(this.apiURL(route), options);
    }

    addEventListener(
        type: string,
        callback: EventListenerOrEventListenerObject,
        options?: boolean | AddEventListenerOptions
    ): void {
        super.addEventListener(type, callback, options);
        this.#registered.add(type);
    }

    /**
     * Poll status  for colab and other things that don't support websockets.
     */
    #pollQueue() {
        setInterval(async () => {
            try {
                const resp = await this.fetchApi('/prompt');
                const status = await resp.json();
                this.dispatchEvent(new CustomEvent('status', { detail: status }));
            } catch (error) {
                this.dispatchEvent(new CustomEvent('status', { detail: null }));
            }
        }, 1000);
    }

    /**
     * ComfyUI name: `#createSocket`
     * Connects to the server for realtime updates
     * @param {boolean} isReconnect If the socket is connection is a reconnect attempt
     */
    #connectToServer(isReconnect = false) {
        if (this.socket) {
            return;
        }

        let opened = false;
        let existingSession = window.name;
        if (existingSession) {
            existingSession = '?clientId=' + existingSession;
        }
        this.socket = new WebSocket(
            `ws${window.location.protocol === 'https:' ? 's' : ''}://${this.api_host}${
                this.api_base
            }/ws${existingSession}`
        );
        this.socket.binaryType = 'arraybuffer';

        this.socket.addEventListener('open', () => {
            opened = true;
            if (isReconnect) {
                this.dispatchEvent(new CustomEvent('reconnected'));
            }
        });

        this.socket.addEventListener('error', () => {
            if (this.socket) this.socket.close();
            if (!isReconnect && !opened) {
                this.#pollQueue();
            }
        });

        this.socket.addEventListener('close', () => {
            setTimeout(() => {
                this.socket = null;
                this.#connectToServer(true);
            }, 300);
            if (opened) {
                this.dispatchEvent(new CustomEvent('status', { detail: null }));
                this.dispatchEvent(new CustomEvent('reconnecting'));
            }
        });

        this.socket.addEventListener('message', event => {
            try {
                if (event.data instanceof ArrayBuffer) {
                    const view = new DataView(event.data);
                    const eventType = view.getUint32(0);
                    const buffer = event.data.slice(4);
                    switch (eventType) {
                        case 1:
                            const view2 = new DataView(event.data);
                            const imageType = view2.getUint32(0);
                            let imageMime;
                            switch (imageType) {
                                case 1:
                                default:
                                    imageMime = 'image/jpeg';
                                    break;
                                case 2:
                                    imageMime = 'image/png';
                            }
                            const imageBlob = new Blob([buffer.slice(4)], { type: imageMime });
                            this.dispatchEvent(new CustomEvent('b_preview', { detail: imageBlob }));
                            break;
                        default:
                            throw new Error(`Unknown binary websocket message of type ${eventType}`);
                    }
                } else {
                    const msg = JSON.parse(event.data);
                    switch (msg.type) {
                        case 'status':
                            if (msg.data.sid) {
                                this.clientId = msg.data.sid;
                                if (this.clientId) window.name = this.clientId;
                            }
                            this.dispatchEvent(new CustomEvent('status', { detail: msg.data.status }));
                            break;
                        case 'progress':
                            this.dispatchEvent(new CustomEvent('progress', { detail: msg.data }));
                            break;
                        case 'executing':
                            this.dispatchEvent(new CustomEvent('executing', { detail: msg.data.node }));
                            break;
                        case 'executed':
                            this.dispatchEvent(new CustomEvent('executed', { detail: msg.data }));
                            break;
                        case 'execution_start':
                            this.dispatchEvent(new CustomEvent('execution_start', { detail: msg.data }));
                            break;
                        case 'execution_error':
                            this.dispatchEvent(new CustomEvent('execution_error', { detail: msg.data }));
                            break;
                        case 'execution_cached':
                            this.dispatchEvent(new CustomEvent('execution_cached', { detail: msg.data }));
                            break;
                        default:
                            if (this.#registered.has(msg.type)) {
                                this.dispatchEvent(new CustomEvent(msg.type, { detail: msg.data }));
                            } else {
                                throw new Error(`Unknown message type ${msg.type}`);
                            }
                    }
                }
            } catch (error) {
                console.warn('Unhandled message:', event.data, error);
            }
        });
    }

    /**
     * Gets a list of extension urls
     * @returns An array of script urls to import
     */
    async getExtensions(): Promise<string[]> {
        const resp = await this.fetchApi('/extensions', { cache: 'no-store' });
        return (await resp.json()).map((route: string) => this.apiURL(route));
    }

    /**
     * Gets a list of embedding names
     * @returns An array of script urls to import
     */
    async getEmbeddings(): Promise<EmbeddingsResponse> {
        const resp = await this.fetchApi('/embeddings', { cache: 'no-store' });
        return await resp.json();
    }

    /**
     * Loads node object definitions for the graph
     * @returns The node definitions
     */
    async getNodeDefs(): Promise<Record<string, ComfyObjectInfo>> {
        const resp = await this.fetchApi('/object_info', { cache: 'no-store' });
        return await resp.json();
    }

    async queuePrompt(
        apiWorkflow: Record<string, WorkflowStep>,
        serializedGraph?: SerializedGraph
    ): Promise<JobCreated> {
        const body = {
            client_id: this.clientId,
            workflow: apiWorkflow,
            extra_data: { extra_pnginfo: { serializedGraph } },
        };

        const res = await this.fetchApi('/prompt', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(body),
        });

        if (res.status !== 200) {
            throw {
                response: await res.json(),
            };
        }

        return await res.json();
    }

    /**
     * Loads a list of items (queue or history)
     * @param {string} type The type of items to load, queue or history
     * @returns The items of the specified type grouped by their status
     */
    async getItems(type: string) {
        if (type === 'queue') {
            return this.getQueue();
        }
        return this.getHistory();
    }

    /**
     * Gets the current state of the queue
     * @returns The currently running and queued items
     */
    async getQueue() {
        try {
            const res = await this.fetchApi('/queue');
            const data = (await res.json()) as QueueResponse;
            return {
                // Running action uses a different endpoint for cancelling
                Running: data.queue_running.map(prompt => ({
                    prompt,
                    remove: { name: 'Cancel', cb: () => this.interrupt() },
                })),
                Pending: data.queue_pending.map(prompt => ({ prompt })),
            };
        } catch (error) {
            console.error(error);
            return { Running: [], Pending: [] };
        }
    }

    /**
     * Gets the prompt execution history
     * @returns Prompt history including node outputs
     */
    async getHistory(max_items = 200) {
        try {
            const res = await this.fetchApi(`/history?max_items=${max_items}`);
            if (!res.ok) {
                throw new Error(`Error fetching history: ${res.status} ${res.statusText}`);
            }

            const history = (await res.json()) as HistoryResponse;
            return { History: Object.values(history) };
        } catch (error) {
            console.error(error);
            return { History: [] };
        }
    }

    /**
     * Gets system & device stats
     * @returns System stats such as python version, OS, per device info
     */
    async getSystemStats(): Promise<SystemStatsResponse> {
        const res = await this.fetchApi('/system_stats');
        if (!res.ok) {
            throw new Error(`Error fetching system stats: ${res.status} ${res.statusText}`);
        }

        return await res.json();
    }

    /**
     * Sends a POST request to the API
     * @param {*} type The endpoint to post to
     * @param {*} body Optional POST data
     */
    async #postItem(type: string, body?: object) {
        try {
            await this.fetchApi('/' + type, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: body ? JSON.stringify(body) : undefined,
            });
        } catch (error) {
            console.error(error);
        }
    }

    /**
     * Deletes an item from the specified list
     * @param {string} type The type of item to delete, queue or history
     * @param {number} id The id of the item to delete
     */
    async deleteItem(type: string, id: number) {
        await this.#postItem(type, { delete: [id] });
    }

    /**
     * Clears the specified list
     * @param {string} type The type of list to clear, queue or history
     */
    async clearItems(type: string) {
        await this.#postItem(type, { clear: true });
    }

    /**
     * Interrupts the execution of the running prompt
     */
    async interrupt() {
        await this.#postItem('interrupt');
    }

    /**
     * Gets user configuration data and where data should be stored
     * @returns { Promise<{ storage: "server" | "browser", users?: Promise<string, unknown>, migrated?: boolean }> }
     */
    async getUserConfig(): Promise<UserConfigResponse> {
        const response = await this.fetchApi('/users');
        return await response.json();
    }

    /**
     * Creates a new user
     * @param { string } username
     * @returns The fetch response
     */
    createUser(username: string) {
        return this.fetchApi('/users', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ username }),
        });
    }

    /**
     * Gets all setting values for the current user
     * @returns { Promise<string, unknown> } A dictionary of id -> value
     */
    async getSettings(): Promise<SettingsResponse> {
        const response = await this.fetchApi('/settings');
        return await response.json();
    }

    /**
     * Gets a setting for the current user
     * @param { string } id The id of the setting to fetch
     * @returns { Promise<unknown> } The setting value
     */
    async getSetting(id: string) {
        const response = await this.fetchApi(`/settings/${encodeURIComponent(id)}`);
        return await response.json();
    }

    /**
     * Stores a dictionary of settings for the current user
     * @param { Record<string, unknown> } settings Dictionary of setting id -> value to save
     * @returns { Promise<void> }
     */
    async storeSettings(settings: Record<string, unknown>) {
        return this.fetchApi(`/settings`, {
            method: 'POST',
            body: JSON.stringify(settings),
        });
    }

    /**
     * Stores a setting for the current user
     * @param { string } id The id of the setting to update
     * @param { unknown } value The value of the setting
     * @returns { Promise<void> }
     */
    async storeSetting(id: string, value: Record<string, any>) {
        return this.fetchApi(`/settings/${encodeURIComponent(id)}`, {
            method: 'POST',
            body: JSON.stringify(value),
        });
    }

    /**
     * Gets a user data file for the current user
     * @param { string } file The name of the userdata file to load
     * @param { RequestInit } [options]
     * @returns { Promise<unknown> } The fetch response object
     */
    async getUserData(file: string, options: RequestInit) {
        return this.fetchApi(`/userdata/${encodeURIComponent(file)}`, options);
    }

    /**
     * Stores a user data file for the current user
     * @param { string } file The name of the userdata file to save
     * @param { unknown } data The data to save to the file
     * @param { RequestInit & { stringify?: boolean, throwOnError?: boolean } } [options]
     * @returns { Promise<void> }
     */
    async storeUserData(
        file: string,
        data: BodyInit,
        options: storeUserDataOptions = { stringify: true, throwOnError: true }
    ) {
        const resp = await this.fetchApi(`/userdata/${encodeURIComponent(file)}`, {
            method: 'POST',
            body: options?.stringify ? JSON.stringify(data) : data,
            ...options,
        });
        if (resp.status !== 200) {
            throw new Error(`Error storing user data file '${file}': ${resp.status} ${resp.statusText}`);
        }
    }
}

//
// /** This will create and submit a workflow. ComfyUI Terminology: `queuePrompt` */
// export function submitCurrentWorkflow() {
//
//     try {
//         while (this.#queueItems.length > 0) {
//             const queueItem = this.#queueItems.pop();
//             if (queueItem) {
//                 ({ number, batchCount } = queueItem);
//
//                 for (let i = 0; i < batchCount; i++) {
//                     const p = await this.canvas.graph.serializeGraph();
//
//                     try {
//                         const res = await this.api.queuePrompt(number, p);
//                         this.lastNodeErrors = res.node_errors;
//
//                         if (this.lastNodeErrors) {
//                             let errors = Array.isArray(this.lastNodeErrors)
//                                 ? this.lastNodeErrors
//                                 : Object.keys(this.lastNodeErrors);
//                             if (errors.length > 0) {
//                                 this.canvas?.draw(true, true);
//                             }
//                         }
//                     } catch (error: unknown) {
//                         const err = error as ComfyPromptError;
//
//                         const formattedError = this.#formatPromptError(err);
//                         this.ui.dialog.show(formattedError);
//                         if (err.response) {
//                             this.lastNodeErrors = err.response.node_errors;
//                             this.canvas?.draw(true, true);
//                         }
//                         break;
//                     }
//
//                     if (p.workflow) {
//                         for (const n of p.workflow.nodes) {
//                             const node = this.graph?.getNodeById(n.id);
//                             if (node?.widgets) {
//                                 for (const widget of node.widgets) {
//                                     // Allow widgets to run callbacks after a prompt has been queued
//                                     // e.g. random seed after every gen
//                                     if (widget.afterQueued) {
//                                         widget.afterQueued();
//                                     }
//                                 }
//                             }
//                         }
//                     }
//
//                     this.canvas?.draw(true, true);
//                     await this.ui.queue.update();
//                 }
//             }
//         }
//     } finally {
//         this.#processingQueue = false;
//     }
//     this.api.dispatchEvent(new CustomEvent('promptQueued', { detail: { number, batchCount } }));
// }
//
// // Again, all custom-nodes are written with the assumption that `api` is a singleton
// // object already instantiated.
// // export const api = app.api;

export const api = new ComfyApi('http://127.0.0.1:8188');
