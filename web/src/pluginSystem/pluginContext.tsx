// TO DO: this context needs a lot of work to verify it's correct

import React, { useState, useEffect } from 'react';
import { createUseContextHook } from '../context/hookCreator';
import { IComfyPlugin, Application } from '../types/interfaces';
import { DependencyGraph } from './DependencyGraph';

// Jupyter Labs application:
// commands, command palette, and context menu, key bindings
// restored; promise resolves when app is loaded
// settings menum
// shell; the UI

// Not kept in React state; this is derived from the installedPlugins
const depGraph = new DependencyGraph();

export const PluginProvider: React.FC = ({ children }) => {
    const [services, setServices] = useState<Map<string, any>>(new Map());
    const [desiredActivationState, setDesiredActivationState] = useState<Map<string, boolean>>(new Map());
    const [installedPlugins, setInstalledPlugins] = useState<Map<string, IComfyPlugin<any>>>(new Map());

    // TO DO: move these functions OUTSIDE of React scope, or at least useCallback them

    // function install(plugin: IComfyPlugin<any>): void;
    // function install(plugins: IComfyPlugin<any>[]): void;
    function install(plugins: IComfyPlugin<any> | IComfyPlugin<any>[]): void {
        if (!Array.isArray(plugins)) {
            plugins = [plugins];
        }
        const newPlugins = plugins.filter(plugin => installedPlugins.has(plugin.id));

        depGraph.addPlugins(newPlugins);

        // Determine which new plugins should be activated
        const autoStartPluginIds = newPlugins
            .filter(plugin => desiredActivationState.get(plugin.id) ?? plugin.autoStart)
            .map(plugin => plugin.id);

        // Update react state
        setInstalledPlugins(current => {
            const updatedPlugins = new Map(current);
            newPlugins.forEach(plugin => updatedPlugins.set(plugin.id, plugin));
            return updatedPlugins;
        });

        activatePlugins(autoStartPluginIds);
    }

    function activatePlugins(pluginIds: string[]): void {
        const activationOrder = depGraph.getActivationOrder(pluginIds);
        activationOrder.forEach(({ id: pluginId }) => {
            if (!isPluginActive(pluginId)) {
                const plugin = installedPlugins.get(pluginId);
                if (!plugin) throw new Error(`Plugin not found: ${pluginId}`);
                if (desiredActivationState.get(pluginId) == true) return;

                const deps = (plugin.requires || []).map(id => {
                    const instance = services.get(id);
                    if (!instance) throw new Error(`Missing required dependency: ${id}`);
                    return instance;
                });

                const optionalDeps = (plugin.optional || []).map(token => services.get(token));

                const instance = plugin.activate({} as Application, ...deps, ...optionalDeps);
                if (plugin.provides) services.set(plugin.provides, instance);
                desiredActivationState.set(pluginId, true);
            }
        });

        // Update react state
        setServices(services);
        setDesiredActivationState(desiredActivationState);
    }

    // Deactivates the specified plugin-ids and their dependents in logical order
    function deactivatePlugins(pluginIds: string[]): void {
        pluginIds.forEach(pluginId => {
            if (!isPluginActive(pluginId)) return;
            const deactivationOrder = depGraph.getDeactivationOrder(pluginId);
            deactivationOrder.forEach(plugin => {
                if (desiredActivationState.get(plugin.id) == true) {
                    if (plugin.deactivate) {
                        // May or may not be asynchronous
                        Promise.resolve(plugin.deactivate({} as Application)).catch((err: unknown) =>
                            console.error(err)
                        );
                    }

                    // If the plugin provides a service, remove it from the services map
                    if (plugin.provides) {
                        services.delete(plugin.provides);
                    }

                    desiredActivationState.set(pluginId, false);
                }
            });
        });

        // Update react state
        setDesiredActivationState(desiredActivationState);
        setServices(services);
    }

    function isPluginInstalled(pluginId: string): boolean {
        return installedPlugins.has(pluginId);
    }

    function isPluginActive(pluginId: string): boolean {
        if (!isPluginInstalled(pluginId)) return false;
        return desiredActivationState.get(pluginId) ?? false;
    }

    // For UIs to display available and active registeredPlugins
    function getAllPlugins() {
        return Array.from(installedPlugins.entries()).map(([pluginId, _plugin]) => ({
            pluginId: pluginId,
            isActive: isPluginActive(pluginId),
        }));
    }

    // Type-safe setters and getters
    function getService<T>(id: string): T | undefined {
        return services.get(id) as T | undefined;
    }

    // function setService<T>(id: string, instance: T): void {
    //     setServices(current => current.set(id, instance));
    // }

    // This allows you to activate / deactivate plugins imperatively
    function setActivationState(newPluginStates: Map<string, boolean>): void {
        // Filter out plugins we don't have installed
        // newPluginStates.forEach((state, pluginId) => {
        //     if (!installedPlugins.has(pluginId)) {
        //         newPluginStates.delete(pluginId);
        //     }
        // });

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

    const value = {
        services,
        install,
        activatePlugins,
        deactivatePlugins,
        isPluginInstalled,
        isPluginActive,
        getAllPlugins,
        getService,
        setActivationState,
    };

    return <PluginContext.Provider value={value}>{children}</PluginContext.Provider>;
};

interface IPluginContext {
    services: Map<string, any>;
    install: (plugins: IComfyPlugin<any> | IComfyPlugin<any>[]) => void;
    activatePlugins: (pluginIds: string[]) => void;
    deactivatePlugins: (pluginIds: string[]) => void;
    isPluginInstalled: (pluginId: string) => boolean;
    isPluginActive: (pluginId: string) => boolean;
    getAllPlugins: () => { pluginId: string; isActive: boolean }[];
    getService: <T>(id: string) => T | undefined;
    setActivationState: (newPluginStates: Map<string, boolean>) => void;
}

const PluginContext = React.createContext<IPluginContext | null>(null);
export const usePlugin = createUseContextHook(PluginContext, 'Plugin Context not found');
