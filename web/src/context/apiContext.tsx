import React from 'react';
import { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import ReconnectingWebSocket from 'reconnecting-websocket';
import { api, ComfyApi } from '../scripts/api';
import { createUseContextHook } from './hookCreator';
import { IComfyApi } from '../types/api.ts';
import { createChannel, createClient, Metadata } from 'nice-grpc';
import { ComfyClient, ComfyDefinition, ComfyMessage } from '../../autogen_web_ts/comfy_request.v1.ts';

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
}

enum ApiStatus {
    CONNECTING = 'connecting',
    OPEN = 'open',
    CLOSING = 'closing',
    CLOSED = 'closed',
}

// Non-react component
const ApiEventEmitter = new EventTarget();

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
    const [comfyClient, setComfyClient] = useState<ComfyClient | null>(null);

    // establish a connection to the server
    useEffect(() => {
        if (serverProtocol === 'grpc') {
            const channel = createChannel(serverUrl);
            const comfyClient: ComfyClient = createClient(ComfyDefinition, channel);

            setComfyClient(comfyClient);

            // Cleanup connection
            return () => channel.close();
        } else {
            // Assumed to be websocket protocol
            const socket = new ReconnectingWebSocket(serverUrl, undefined, { maxReconnectionDelay: 300 });
            socket.binaryType = 'arraybuffer';

            socket.addEventListener('open', () => {
                setConnectionStatus(ApiStatus.OPEN);
            });

            let cleanupPolling = () => {};
            socket.addEventListener('error', () => {
                if (!(socket.readyState == socket.OPEN)) {
                    // The websocket failed to open; use a fallback instead
                    socket.close();
                    cleanupPolling = pollingFallback();
                }
                setConnectionStatus(ApiStatus.CLOSED);
            });

            socket.addEventListener('close', () => {
                // Will automatically try to reconnect
                setConnectionStatus(ApiStatus.CONNECTING);
            });

            setConnectionStatus(ApiStatus.CONNECTING);

            return () => {
                socket.close();
                cleanupPolling();
            };
        }
    }, [serverUrl, serverProtocol]);

    // If we are in a session-id, subscribe to the stream of results
    useEffect(() => {
        if (comfyClient === null) return;

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
    }, [comfyClient, sessionId]);

    useEffect(() => {
        const metadata = new Metadata();
        if (window.API_KEY) metadata.set('api-key', window.API_KEY);
        setRequestMetadata(metadata);
    }, []);

    return (
        <ApiContext.Provider
            value={{
                sessionId,
                connectionStatus,
                comfyClient,
                requestMetadata,
            }}
        >
            {children}
        </ApiContext.Provider>
    );
};

const ApiContext = createContext<IApiContext | undefined>(undefined);
export const useApiContext = createUseContextHook(ApiContext, 'useApiContext must be used within a ApiContextProvider');
