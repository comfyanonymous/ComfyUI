// The container is used to provider dependency resolution for plugins

import React, { useState, useEffect } from 'react';
import { Container } from 'inversify';
import { createUseContextHook } from './hookCreator';
import { getLocalExtensions } from '../extension_manager/findLocalExtensions';
import { loadExtensions } from '../extension_manager/loadExtensions';
import { ExtensionManager } from '../extension_manager/extensionManager';

interface IPluginContext {
    container: Container;
}

const PluginContext = React.createContext<IPluginContext | null>(null);

export const PluginContextProvider: React.FC = ({ children }) => {
    const [container, setContainer] = useState(new Container());
    const [extensionManager, setExtensionManager] = useState(new ExtensionManager());

    useEffect(() => {
        const loadPlugins = async () => {
            // const webModuleUrls = await api.getExtensions();
            const webModuleUrls = await getLocalExtensions();
            const comfyPlugins = await loadExtensions(webModuleUrls);
            extensionManager.registerPlugins(comfyPlugins);
        };
        loadPlugins().catch((err: unknown) => console.error(err));
    }, [container, extensionManager]);

    // First, load dependencies
    // then add them to the container

    return <PluginContext.Provider value={{ container }}>{children}</PluginContext.Provider>;
};

export const usePlugin = createUseContextHook(PluginContext, 'Plugin Context not found');
