import React from 'react';
import { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import ReconnectingWebSocket from 'reconnecting-websocket';
import { api, ComfyApi } from '../scripts/api';
import { createUseContextHook } from './hookCreator';
import { IComfyApi } from '../types/api.ts';
import { createChannel, createClient } from 'nice-grpc';
import { ComfyClient, ComfyDefinition } from '../../autogen_web_ts/comfy_request.v1.ts';

// This is injected into index.html by `start.py`
declare global {
    interface Window {
        SERVER_URL: string;
    }
}

interface ApiContextType {
    api: IComfyApi;
    ApiEventEmitter: EventTarget;
    connectionStatus: string;
    sessionId: string | null;
}

enum ApiStatus {
    CONNECTING = 'connecting',
    OPEN = 'open',
    CLOSING = 'closing',
    CLOSED = 'closed',
}

// Non-react component
const ApiEventEmitter = new EventTarget();

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
    // Return cleanup function
    return () => clearInterval(intervalId);
};

export const ApiContextProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
    // TO DO: add possible auth in here as well?
    const [sessionId, setSessionId] = useState<string | null>(null);
    const [socket, setSocket] = useState<ReconnectingWebSocket | null>(null);
    const [serverUrl, setServerUrl] = useState<string>(window.SERVER_URL);
    const [connectionStatus, setApiStatus] = useState<string>(ApiStatus.CLOSED);

    // websocket client
    useEffect(() => {
        let suffix = '';
        if (sessionId) {
            suffix = '?clientId=' + sessionId;
        }
        const socket = new ReconnectingWebSocket(serverUrl, undefined, { maxReconnectionDelay: 300 });
        socket.binaryType = 'arraybuffer';

        socket.addEventListener('open', () => {
            setApiStatus(ApiStatus.OPEN);
        });

        let cleanupPolling = () => {};
        socket.addEventListener('error', () => {
            if (!(socket.readyState == socket.OPEN)) {
                // The websocket failed to open; use a fallback insetad
                socket.close();
                cleanupPolling = pollingFallback();
            }
            setApiStatus(ApiStatus.CLOSED);
        });

        socket.addEventListener('close', () => {
            // Will automatically try to reconnect
            setApiStatus(ApiStatus.CONNECTING);
        });

        setSocket(socket);
        setApiStatus(ApiStatus.CONNECTING);

        return () => {
            socket.close();
            cleanupPolling();
        };
    }, [serverUrl, sessionId]);

    // gRPC client
    useEffect(() => {
        const channel = createChannel('localhost:8080');
        const client: ComfyClient = createClient(ComfyDefinition, channel);

        // Cleanup connection
        return () => {
            channel.close();
        };
    }, [socket]);

    return (
        <ApiContext.Provider
            value={{
                // TODO: we shouldn't hardcode this
                api: new ComfyApi('http://127.0.0.1:8188'),
                ApiEventEmitter,
                connectionStatus,
                sessionId,
            }}
        >
            {children}
        </ApiContext.Provider>
    );
};

const ApiContext = createContext<ApiContextType | undefined>(undefined);
export const useApiContext = createUseContextHook(ApiContext, 'useApiContext must be used within a ApiContextProvider');
