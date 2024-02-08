import React from 'react';
import { createContext, useContext, useState, useEffect, ReactNode, useCallback } from 'react';
import ReconnectingWebSocket from 'reconnecting-websocket';
import { api } from '../scripts/api';
import { createUseContextHook } from './hookCreator';
import { createChannel, createClient, Metadata } from 'nice-grpc-web';
import { ComfyClient, ComfyDefinition, ComfyMessage, JobCreated } from '../../autogen_web_ts/comfy_request.v1.ts';
import { WorkflowStep } from '../../autogen_web_ts/comfy_request.v1.ts';
import { SerializedGraph } from '../types/litegraph';

// This is injected into index.html by `start.py`
declare global {
    interface Window {
        API_KEY?: string;
        SERVER_URL: string;
        SERVER_PROTOCOL: ProtocolType;
    }
}

type ProtocolType = 'grpc' | 'ws';

interface IApiContext {
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

    return (
        <ApiContext.Provider
            value={{
                sessionId,
                connectionStatus,
                comfyClient,
                requestMetadata,
                runWorkflow,
            }}
        >
            {children}
        </ApiContext.Provider>
    );
};

const ApiContext = createContext<IApiContext | undefined>(undefined);

export const useApiContext = createUseContextHook(ApiContext, 'useApiContext must be used within a ApiContextProvider');
