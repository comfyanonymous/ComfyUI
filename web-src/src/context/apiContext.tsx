import React from 'react';
import { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import ReconnectingWebSocket from 'reconnecting-websocket';
import { api, ComfyApi } from '../scripts/api';
import { createUseContextHook } from './hookCreator';
import { IComfyApi } from '../types/api.ts';

interface ApiContextType {
    api: IComfyApi;
    ApiEventEmitter: EventTarget;
    apiStatus: string;
    sessionId: string | null;
}

enum ApiStatus {
    CONNECTING = 'connecting',
    OPEN = 'open',
    CLOSING = 'closing',
    CLOSED = 'closed',
}

const ApiContext = createContext<ApiContextType | undefined>(undefined);
export const useApiContext = createUseContextHook(ApiContext, 'useApiContext must be used within a ApiContextProvider');

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
    const [api_host, setApiHost] = useState<string>(location.host);
    const [api_base, setApiBase] = useState<string>(location.pathname.split('/').slice(0, -1).join('/'));
    // const [apiEventEmitter, _] = useState<EventTarget>(new EventTarget());
    const [apiStatus, setApiStatus] = useState<string>(ApiStatus.CLOSED);

    useEffect(() => {
        let suffix = '';
        if (sessionId) {
            suffix = '?clientId=' + sessionId;
        }
        const socket = new ReconnectingWebSocket(
            `ws${window.location.protocol === 'https:' ? 's' : ''}://${api_host}${api_base}/ws${suffix}`,
            undefined,
            { maxReconnectionDelay: 300 }
        );
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
    }, [api_host, api_base, sessionId]);

    return (
        <ApiContext.Provider
            value={{
                api: new ComfyApi('http://127.0.0.1:8188'),
                ApiEventEmitter,
                apiStatus,
                sessionId,
            }}
        >
            {children}
        </ApiContext.Provider>
    );
};
