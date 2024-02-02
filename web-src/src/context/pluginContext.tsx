// The container is used to provider dependency resolution for plugins

import React, { useState, useEffect } from 'react';
import { createUseContextHook } from './hookCreator';
import { getLocalExtensions } from '../pluginStore/utils/findLocalExtensions';
import { loadExtensions } from '../pluginStore/utils/loadExtensions';
import { defaultSerializeGraph } from '../litegraph/comfyGraph';
import { usePluginStore } from '../pluginStore';

interface IExtensionContext {}

const ExtensionContext = React.createContext<IExtensionContext | null>(null);

export const ExtensionContextProvider: React.FC = ({ children }) => {
    const pluginStore = usePluginStore();
    const [serializeGraph, setSerializeGraph] = useState(() => defaultSerializeGraph);

    useEffect(() => {
        const loadPlugins = async () => {
            // const webModuleUrls = await api.getExtensions();
            const webModuleUrls = await getLocalExtensions();
            const comfyPlugins = await loadExtensions(webModuleUrls);
            pluginStore.install(comfyPlugins);
        };
        loadPlugins().catch((err: unknown) => console.error(err));
    }, [pluginStore]);

    return <ExtensionContext.Provider value={{}}>{children}</ExtensionContext.Provider>;
};

export const usePlugin = createUseContextHook(ExtensionContext, 'Plugin Context not found');
