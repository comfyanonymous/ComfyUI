import React, { ReactNode, useState, useEffect } from 'react';
import { createUseContextHook } from './hookCreator';
import { IComfyApp } from '../types/interfaces.ts';
import { ComfyApp } from '../scripts/app2.ts';
import { getLocalExtensions } from '../pluginStore/utils/findLocalExtensions';
import { loadExtensions } from '../pluginStore/utils/loadExtensions';
import { usePlugin } from './pluginContext';
import { useApiContext } from './apiContext';

interface ComfyAppContextType {
    app: IComfyApp;
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
    const [app, setApp] = useState<IComfyApp>(ComfyApp.getInstance());

    useEffect(() => {
        const loadPlugins = async () => {
            const webModuleUrls = await api.getExtensions();
            // const webModuleUrls = await getLocalExtensions();
            const comfyPlugins = await loadExtensions(webModuleUrls);
            install(comfyPlugins);
        };
        loadPlugins().catch((err: unknown) => console.error(err));
    }, [install, api]);

    return <ComfyAppContext.Provider value={{ app }}>{children}</ComfyAppContext.Provider>;
};
