import React from 'react';
import { createContext, useState, useEffect, ReactNode, useCallback } from 'react';
import ReconnectingWebSocket from 'reconnecting-websocket';
import { api } from '../scripts/api';
import { createUseContextHook } from './hookCreator';
import { createChannel, createClient, Metadata } from 'nice-grpc-web';
import { ComfyClient, ComfyDefinition, ComfyMessage, JobCreated } from '../../autogen_web_ts/comfy_request.v1.ts';
import { WorkflowStep } from '../../autogen_web_ts/comfy_request.v1.ts';
import { SerializedGraph } from '../types/litegraph';
import {
    IComfyApi,
    ComfyItemURLType,
    EmbeddingsResponse,
    HistoryResponse,
    UserConfigResponse,
    SystemStatsResponse,
    SettingsResponse,
    storeUserDataOptions,
    ComfyHistoryItems,
} from '../types/api.ts';
import API_URL from './apiUrl';
import { ComfyObjectInfo } from '../types/comfy';

// This is injected into index.html by `start.py`
declare global {
    interface Window {
        API_KEY?: string;
        SERVER_URL: string;
        SERVER_PROTOCOL: ProtocolType;
    }
}

type ProtocolType = 'grpc' | 'ws';

interface IApiContext extends Partial<IComfyApi> {
    sessionId?: string;
    connectionStatus: string;
    comfyClient: ComfyClient | null;
    requestMetadata?: Metadata;
    runWorkflow: (workflow: Record<string, WorkflowStep>, serializedGraph?: SerializedGraph) => Promise<JobCreated>;
}

enum ApiStatus {
    CONNECTING = 'connecting',
    OPEN = 'open',
    CLOSING = 'closing',
    CLOSED = 'closed',
}

// Non-react component
export const ApiEventEmitter = new EventTarget();

// TO DO: implement this
const handleComfyMessage = (message: ComfyMessage) => {
    ApiEventEmitter.dispatchEvent(new CustomEvent('room', { detail: message }));
};

// Use polling as a backup strategy incase the websocket fails to connect
const pollingFallback = () => {
    const intervalId = setInterval(async () => {
        try {
            const resp = await api.fetchApi('/prompt');
            const status = await resp.json();
            ApiEventEmitter.dispatchEvent(new CustomEvent('status', { detail: status }));
        } catch (error) {
            ApiEventEmitter.dispatchEvent(new CustomEvent('status', { detail: null }));
        }
    }, 1000);

    // Cleanup function
    return () => clearInterval(intervalId);
};

export const ApiContextProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
    // TO DO: add possible auth in here as well?
    const [sessionId, setSessionId] = useState<string | undefined>(undefined);
    // const [socket, setSocket] = useState<ReconnectingWebSocket | null>(null);
    const [serverUrl, setServerUrl] = useState<string>(window.SERVER_URL);
    const [connectionStatus, setConnectionStatus] = useState<string>(ApiStatus.CLOSED);
    const [serverProtocol, setServerProtocol] = useState<ProtocolType>(window.SERVER_PROTOCOL);
    const [requestMetadata, setRequestMetadata] = useState<Metadata | undefined>(undefined);

    // Only used for when serverProtocol is grpc. Used to both send messages and stream results
    const [comfyClient, setComfyClient] = useState<ComfyClient>(
        createClient(ComfyDefinition, createChannel(serverUrl))
    );

    // Recreate ComfyClient as needed
    useEffect(() => {
        if (serverProtocol === 'grpc') {
            const channel = createChannel(serverUrl);
            const newComfyClient = createClient(ComfyDefinition, channel);
            setComfyClient(newComfyClient);
            // No cleanup is explicitly required for gRPC client
        }
    }, [serverUrl, serverProtocol]);

    // Establish a connection to local-server if we're using websockets
    useEffect(() => {
        if (serverProtocol === 'ws') {
            const socket = new ReconnectingWebSocket(serverUrl, undefined, { maxReconnectionDelay: 300 });
            socket.binaryType = 'arraybuffer';
            let cleanupPolling = () => {};

            socket.addEventListener('open', () => {
                setConnectionStatus(ApiStatus.OPEN);
            });

            socket.addEventListener('error', () => {
                if (!(socket.readyState === socket.OPEN)) {
                    // The websocket failed to open; use a fallback instead
                    socket.close();
                    cleanupPolling = pollingFallback();
                }
                setConnectionStatus(ApiStatus.CLOSED);
            });

            socket.addEventListener('close', () => {
                setConnectionStatus(ApiStatus.CONNECTING);
            });

            setConnectionStatus(ApiStatus.CONNECTING);

            return () => {
                socket.close();
                cleanupPolling();
            };
        }
    }, [serverUrl, serverProtocol]);

    // Once we have a session-id, subscribe to the stream of results using grpc
    useEffect(() => {
        if (sessionId === undefined) return;
        if (serverProtocol !== 'grpc') return;

        const abortController = new AbortController();

        const stream = comfyClient.streamRoom({ session_id: sessionId }, { signal: abortController.signal });

        setConnectionStatus(ApiStatus.CONNECTING);

        (async () => {
            let first = true;
            for await (const message of stream) {
                if (first) {
                    first = false;
                    setConnectionStatus(ApiStatus.OPEN);
                }
                handleComfyMessage(message);
            }
        })()
            .then(() => {
                setConnectionStatus(ApiStatus.CLOSED);
            })
            .catch(error => {
                setConnectionStatus(ApiStatus.CLOSED);
                console.error(error);
            });

        // Cleanup stream
        return () => {
            abortController.abort();
        };
    }, [comfyClient, sessionId, serverProtocol]);

    // Update metadata based on api-key / login status
    useEffect(() => {
        const metadata = new Metadata();
        if (window.API_KEY) metadata.set('api-key', window.API_KEY);
        setRequestMetadata(metadata);
    }, []);

    // This is the function used to submit jobs to the server
    // ComfyUI terminology: 'queuePrompt'
    const runWorkflow = useCallback(
        async (workflow: Record<string, WorkflowStep>, serializedGraph?: SerializedGraph): Promise<JobCreated> => {
            if (serverProtocol === 'grpc' && comfyClient) {
                // Use gRPC server
                const request = {
                    workflow,
                    serializedGraph,
                    inputFiles: [],
                    output_config: undefined,
                    worker_wait_duration: undefined,
                    session_id: sessionId,
                };

                const res = await comfyClient.runWorkflow(request, { metadata: requestMetadata });

                // Update the assigned sessionId
                if (res.session_id !== sessionId) {
                    setSessionId(res.session_id);
                }

                return res;
            } else {
                // Use REST server
                const headers: Record<string, string> = {
                    'Content-Type': 'application/json',
                };

                // Convert Metadata to headers
                if (requestMetadata) {
                    for (const [key, values] of requestMetadata) {
                        // Since values is an array, join them with a comma.
                        headers[key] = values.join(', ');
                    }
                }

                const res = await fetch(`${serverUrl}/prompt`, {
                    method: 'POST',
                    headers,
                    body: JSON.stringify({
                        workflow,
                        serializedGraph,
                    }),
                });

                if (res.status !== 200) {
                    throw {
                        response: await res.json(),
                    };
                }

                return await res.json();
            }
        },
        [requestMetadata, serverUrl, serverProtocol, comfyClient, sessionId]
    );

    // putting these functions here for now cos we might want to find a way to go with the gRPC and the fetch requests
    const generateURL = (route: string): string => {
        return serverUrl + route;
    };

    const fetchApi = (route: string, options?: RequestInit) => {
        if (!options) {
            options = {};
        }
        if (!options.headers) {
            options.headers = {};
        }

        return fetch(generateURL(route), options);
    };

    /**
     * Gets a list of embedding names
     * @returns An array of script urls to import
     */
    const getEmbeddings = async (): Promise<EmbeddingsResponse> => {
        const resp = await fetchApi(API_URL.GET_EMBEDDINGS, { cache: 'no-store' });
        return await resp.json();
    };

    /**
     * Loads node object definitions for the graph
     * @returns The node definitions
     */
    const getNodeDefs = async (): Promise<Record<string, ComfyObjectInfo>> => {
        const resp = await fetchApi(API_URL.GET_NODE_DEFS, { cache: 'no-store' });
        return await resp.json();
    };

    /**
     * Gets the prompt execution history
     * @returns Prompt history including node outputs
     */
    const getHistory = async (max_items: number = 200): Promise<ComfyHistoryItems> => {
        try {
            const res = await fetchApi(API_URL.GET_HISTORY(max_items));
            if (!res.ok) {
                throw new Error(`Error fetching history: ${res.status} ${res.statusText}`);
            }

            const history = (await res.json()) as HistoryResponse;
            return { History: Object.values(history) };
        } catch (error) {
            console.error(error);
            return { History: [] };
        }
    };

    /**
     * Gets system & device stats
     * @returns System stats such as python version, OS, per device info
     */
    const getSystemStats = async (): Promise<SystemStatsResponse> => {
        const res = await fetchApi(API_URL.GET_SYSTEM_STATS);
        if (!res.ok) {
            throw new Error(`Error fetching system stats: ${res.status} ${res.statusText}`);
        }

        return await res.json();
    };

    /**
     * Sends a POST request to the API
     * @param {*} type The endpoint to post to: queue or history
     * @param {*} body Optional POST data
     */
    const postItem = async (type: ComfyItemURLType, body?: object): Promise<void> => {
        try {
            await fetchApi('/' + type, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: body ? JSON.stringify(body) : undefined,
            });
        } catch (error) {
            console.error(error);
        }
    };

    /**
     * Deletes an item from the specified list
     * @param {string} type The type of item to delete, queue or history
     * @param {number} id The id of the item to delete
     */
    const deleteItem = async (type: ComfyItemURLType, id: number) => {
        await postItem(type, { delete: [id] });
    };

    /**
     * Clears the specified list
     * @param {string} type The type of list to clear, queue or history
     */
    const clearItems = async (type: ComfyItemURLType) => {
        await postItem(type, { clear: true });
    };

    /**
     * Interrupts the execution of the running prompt
     */
    const interrupt = async (): Promise<void> => {
        await postItem('interrupt');
    };

    /**
     * Gets user configuration data and where data should be stored
     * @returns { Promise<{ storage: "server" | "browser", users?: Promise<string, unknown>, migrated?: boolean }> }
     */
    const getUserConfig = async (): Promise<UserConfigResponse> => {
        const response = await fetchApi(API_URL.GET_USER_CONFIG);
        return await response.json();
    };

    /**
     * Creates a new user
     * @param { string } username
     * @returns The fetch response
     */
    const createUser = (username: string) => {
        return fetchApi(API_URL.CREATE_USER, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ username }),
        });
    };

    /**
     * Gets all setting values for the current user
     * @returns { Promise<string, unknown> } A dictionary of id -> value
     */
    const getSettings = async (): Promise<SettingsResponse> => {
        const response = await fetchApi(API_URL.GET_SETTINGS);
        return await response.json();
    };

    /**
     * Gets a setting for the current user
     * @param { string } id The id of the setting to fetch
     * @returns { Promise<unknown> } The setting value
     */
    const getSetting = async (id: string) => {
        const response = await fetchApi(API_URL.GET_SETTING_BY_ID(id));
        return await response.json();
    };

    /**
     * Stores a dictionary of settings for the current user
     * @param { Record<string, unknown> } settings Dictionary of setting id -> value to save
     * @returns { Promise<void> }
     */
    const storeSettings = (settings: Record<string, unknown>) => {
        return fetchApi(API_URL.STORE_SETTINGS, {
            method: 'POST',
            body: JSON.stringify(settings),
        });
    };

    /**
     * Stores a setting for the current user
     * @param { string } id The id of the setting to update
     * @param { unknown } value The value of the setting
     * @returns { Promise<void> }
     */
    const storeSetting = (id: string, value: Record<string, any>) => {
        return fetchApi(`/settings/${encodeURIComponent(id)}`, {
            method: 'POST',
            body: JSON.stringify(value),
        });
    };

    /**
     * Gets a user data file for the current user
     * @param { string } file The name of the userData file to load
     * @param { RequestInit } [options]
     * @returns { Promise<unknown> } The fetch response object
     */
    const getUserData = async (file: string, options: RequestInit) => {
        return await fetchApi(API_URL.GET_USER_DATA_FILE(file), options);
    };

    /**
     * Stores a user data file for the current user
     * @param { string } file The name of the userData file to save
     * @param { unknown } data The data to save to the file
     * @param { RequestInit & { stringify?: boolean, throwOnError?: boolean } } [options]
     * @returns { Promise<void> }
     */
    const storeUserData = async (
        file: string,
        data: BodyInit,
        options: storeUserDataOptions = { stringify: true, throwOnError: true }
    ) => {
        const resp = await fetchApi(API_URL.STORE_USER_DATA_FILE(file), {
            method: 'POST',
            body: options?.stringify ? JSON.stringify(data) : data,
            ...options,
        });
        if (resp.status !== 200) {
            throw new Error(`Error storing user data file '${file}': ${resp.status} ${resp.statusText}`);
        }
    };

    return (
        <ApiContext.Provider
            value={{
                sessionId,
                connectionStatus,
                comfyClient,
                requestMetadata,
                runWorkflow,

                // API functions
                getEmbeddings,
                getNodeDefs,
                getHistory,
                getSystemStats,
                deleteItem,
                clearItems,
                interrupt,
                getUserConfig,
                createUser,
                getSettings,
                storeSettings,
                storeSetting,
                getUserData,
                storeUserData,
            }}
        >
            {children}
        </ApiContext.Provider>
    );
};

const ApiContext = createContext<IApiContext | undefined>(undefined);

export const useApiContext = createUseContextHook(ApiContext, 'useApiContext must be used within a ApiContextProvider');
