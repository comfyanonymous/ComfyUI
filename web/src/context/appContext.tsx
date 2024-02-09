import React, { ReactNode, useState, useEffect } from 'react';
import { createUseContextHook } from './hookCreator';
import { loadExtensions } from '../pluginSystem/loadExtensions.ts';
import { usePlugin } from '../pluginSystem/pluginContext.tsx';
import { ComfyGraph } from '../litegraph/comfyGraph.ts';
import { LastExecutionError } from '../types/interfaces.ts';
import { api } from '../scripts/api.tsx';

interface ComfyAppContextType {
    enableWorkflowAutoSave: (graph: ComfyGraph) => void;
    disableWorkflowAutoSave: () => void;
    clean: () => void;
    storageLocation: string | null;
    isNewUserSession: boolean;
    lastExecutionError: LastExecutionError | null;
}

const ComfyAppContext = React.createContext<ComfyAppContextType | null>(null);
export const useComfyApp = createUseContextHook(
    ComfyAppContext,
    'useComfyApp must be used within a ComfyAppContextProvider'
);

// Though the comfier app is a singleton,
// I think it makes sense to have it in the context for usage in the web app
export const ComfyAppContextProvider = ({ children }: { children: ReactNode }) => {
    const { install } = usePlugin();
    const [saveInterval, setSaveInterval] = useState<NodeJS.Timeout | null>(null);
    const [storageLocation, setStorageLocation] = useState<string | null>(null);
    const [isNewUserSession, setIsNewUserSession] = useState<boolean>(false);
    const [lastExecutionError, setLastExecutionError] = useState<LastExecutionError | null>(null);

    useEffect(() => {
        const loadPlugins = async () => {
            const webModuleUrls = await api.getExtensions();
            // const webModuleUrls = await getLocalExtensions();
            const comfyPlugins = await loadExtensions(webModuleUrls);
            install(comfyPlugins);
        };
        loadPlugins().catch((err: unknown) => console.error(err));
    }, [install, api]);

    const enableWorkflowAutoSave = (graph: ComfyGraph) => {
        const interval = setInterval(
            () => localStorage.setItem('workflow', JSON.stringify(graph.serializeGraph())),
            1000
        );
    };

    const disableWorkflowAutoSave = () => {
        if (!saveInterval) return;

        clearInterval(saveInterval);
        setSaveInterval(null);
    };

    const clean = () => {
        disableWorkflowAutoSave();
    };

    return (
        <ComfyAppContext.Provider
            value={{
                clean,
                storageLocation,
                isNewUserSession,
                lastExecutionError,
                enableWorkflowAutoSave,
                disableWorkflowAutoSave,
            }}
        >
            {children}
        </ComfyAppContext.Provider>
    );
};
