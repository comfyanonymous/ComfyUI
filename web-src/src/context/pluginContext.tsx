// The container is used to provider dependency resolution for plugins

import React, { useState, useEffect } from 'react';
import { createUseContextHook } from './hookCreator';
import { getLocalExtensions } from '../pluginStore/utils/findLocalExtensions';
import { loadExtensions } from '../pluginStore/utils/loadExtensions';
import { defaultSerializeGraph } from '../litegraph/comfyGraph';
import { usePluginStore } from '../pluginStore';
import { Token } from '../types/interfaces';
import { IComfyPlugin, Application } from '../types/interfaces';
import { DependencyGraph } from '../pluginStore/DependencyGraph';

// Jupyter Labs application:
// commands, command palette, and context menu, key bindings
// restored; promise resolves when app is loaded
// settings menu
// shell; the UI

interface IPluginContext {}

const PluginContext = React.createContext<IPluginContext | null>(null);

export const PluginContextProvider: React.FC = ({ children }) => {
    const pluginStore = usePluginStore();
    // const [serializeGraph, setSerializeGraph] = useState(() => defaultSerializeGraph);
    const [services, setServices] = useState<Map<Token<any>, any>>(new Map());

    useEffect(() => {
        const loadPlugins = async () => {
            // const webModuleUrls = await api.getExtensions();
            const webModuleUrls = await getLocalExtensions();
            const comfyPlugins = await loadExtensions(webModuleUrls);
            const services = await pluginStore.install(comfyPlugins);
            setServices(services);
        };
        loadPlugins().catch((err: unknown) => console.error(err));
    }, [pluginStore]);

    return <PluginContext.Provider value={{}}>{children}</PluginContext.Provider>;
};

export const usePlugin = createUseContextHook(PluginContext, 'Plugin Context not found');


const installedPlugins: Map<string, IComfyPlugin<any>> = new Map();
const depGraph = new DependencyGraph();

    function install(plugin: IComfyPlugin<any>): void;
    function install(plugins: IComfyPlugin<any>[]): void;
    function install(plugins: IComfyPlugin<any> | IComfyPlugin<any>[]): void {
        if (!Array.isArray(plugins)) {
            plugins = [plugins];
        }
        const newPlugins = plugins.filter(plugin => installedPlugins.has(plugin.id));
        newPlugins.forEach(plugin => installedPlugins.set(plugin.id, plugin));
        depGraph.addPlugins(newPlugins);

        // Determine which new plugins should be activated
        const autoStartPluginIds = newPlugins
            .filter(plugin => desiredActivationState.get(plugin.id) ?? plugin.autoStart)
            .map(plugin => plugin.id);

        activatePlugins(autoStartPluginIds);
    }

    function activatePlugins(pluginIds: string[]): void {
        const activationOrder = depGraph.getActivationOrder(pluginIds);
        activationOrder.forEach(({ id: pluginId }) => {
            if (!isPluginActive(pluginId)) {
                activatePlugin(pluginId);
            }
        });
    }

    function activatePlugin(pluginId: string): void {
        const plugin = installedPlugins.get(pluginId);
        if (!plugin) throw new Error(`Plugin not found: ${pluginId}`);
        if (desiredActivationState.get(pluginId) == true) return;

        const deps = (plugin.requires || []).map(token => {
            const instance = services.get(token);
            if (!instance) throw new Error(`Missing required dependency: ${token.debugName}`);
            return instance;
        });

        const optionalDeps = (plugin.optional || []).map(token => services.get(token));

        const instance = plugin.activate({} as Application, ...deps, ...optionalDeps);
        if (plugin.provides) services.set(plugin.provides, instance);
        desiredActivationState.set(pluginId, true);
    }

    // Deactivates the specified plugin-ids and their dependents in logical order
    function deactivatePlugins(pluginIds: string[]): void {
        pluginIds.forEach(pluginId => {
            if (isPluginActive(pluginId)) deactivatePlugin(pluginId);
        });
    }

    function deactivatePlugin(pluginId: string): void {
        const deactivationOrder = depGraph.getDeactivationOrder(pluginId);
        deactivationOrder.forEach(plugin => {
            if (desiredActivationState.get(plugin.id) == true) {
                if (plugin.deactivate) {
                    plugin.deactivate({} as Application);
                }

                // If the plugin provides a service, remove it from the services map
                if (plugin.provides) {
                    services.delete(plugin.provides);
                }

                desiredActivationState.set(pluginId, false);
            }
        });
    }

    // This allows you to activate / deactivate plugins imperatively
    function setPluginsState(newPluginStates: Map<string, boolean>): void {
        // Filter out plugins we don't have installed
        newPluginStates.forEach((state, pluginId) => {
            if (!installedPlugins.has(pluginId)) {
                newPluginStates.delete(pluginId);
            }
        });

        const pluginsToDeactivate = Array.from(newPluginStates.entries())
            .filter(([pluginId, on]) => (!on && isPluginActive(pluginId)) == true)
            .map(([pluginId, _]) => pluginId);

        const pluginsToActivate = Array.from(newPluginStates.entries())
            .filter(([pluginId, on]) => (on && !isPluginActive(pluginId)) == false)
            .map(([pluginId, _]) => pluginId);

        // Deactivate plugins first
        deactivatePlugins(pluginsToDeactivate);

        // Then activate plugins
        activatePlugins(pluginsToActivate);
    }

    function isPluginInstalled(pluginId: string): boolean {
        return installedPlugins.has(pluginId);
    }

    function isPluginActive(desiredActivationState: Map<string, boolean>, pluginId: string): boolean {
        if (!isPluginInstalled(pluginId)) return false;
        return desiredActivationState.get(pluginId) ?? false;
    }

    // For UIs to display available and active registeredPlugins
    function getAllPlugins(): { pluginId: string; isActive: boolean }[] {
        return Array.from(installedPlugins.entries()).map(([pluginId, _plugin]) => ({
            pluginId: pluginId,
            isActive: isPluginActive(pluginId),
        }));
    }

    // Type-safe setters and getters
    function get<T>(services: Map<Token<any>, any>, token: Token<T>): T | undefined {
        return services.get(token) as T | undefined;
    }

    function set<T>(setServices: React.Dispatch<React.SetStateAction<Map<Token<any>, any>>>, token: Token<T>, instance: T): void {
        setServices(current => current.set(token, instance));
    }


export const PluginContextProvider2: React.FC = ({ children }) => {
    const [desiredActivationState, setDesiredActivationState] = useState<Map<string, boolean>>(new Map());
    const [services, setServices] = useState<Map<Token<any>, any>>(new Map());

    const value = {
        desiredActivationState,
        setDesiredActivationState,
        isPluginInstalled: (pluginId: string) => isPluginInstalled(pluginId),
        isPluginActive: (pluginId: string) => isPluginActive(desiredActivationState, pluginId),
        getAllPlugins,
        get: <T,>(token: Token<T>) => get(services, token),
        set: <T,>(token: Token<T>, instance: T) => set(setServices, token, instance),
    }

    return <PluginContext.Provider value={value}>{children}</PluginContext.Provider>;
};