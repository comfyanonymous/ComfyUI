import { Event } from './Event';
import { EventCallableRegsitry } from './EventCallableRegsitry';
import type { IComfyPlugin, Token } from '../types/interfaces';
import { DependencyGraph } from './DependencyGraph';

export class PluginStore {
    private functionArray: Map<string, any> = new Map<string, (...args: any[]) => any>();
    private _eventCallableRegistry: EventCallableRegsitry = new EventCallableRegsitry();

    // Holds the IComfyPlugin objects
    private installedPlugins: Map<string, IComfyPlugin<any>> = new Map();

    // Tracks the desired activation state of each plugin (active == true). Key is plugin-id
    private desiredActivationState: Map<string, boolean> = new Map();

    // Holds services provided by each plugin, to be used as dependencies
    private services: Map<Token<any>, any> = new Map();

    // Used so plugins can be de/activated in the order required by their dependencies
    private depGraph: DependencyGraph = new DependencyGraph();

    install(plugin: IComfyPlugin<any>): void;
    install(plugins: IComfyPlugin<any>[]): void;
    install(plugins: IComfyPlugin<any> | IComfyPlugin<any>[]): void {
        if (!Array.isArray(plugins)) {
            plugins = [plugins];
        }
        const newPlugins = plugins.filter(plugin => !this.installedPlugins.has(plugin.id));
        newPlugins.forEach(plugin => this.installedPlugins.set(plugin.id, plugin));
        this.depGraph.addPlugins(newPlugins);

        // Determine which new plugins should be activated
        const autoStartPluginIds = newPlugins
            .filter(plugin => this.desiredActivationState.get(plugin.id) ?? plugin.autoStart)
            .map(plugin => plugin.id);

        this.activatePlugins(autoStartPluginIds);
    }

    activatePlugins(pluginIds: string[]): void {
        const activationOrder = this.depGraph.getActivationOrder(pluginIds);
        activationOrder.forEach(({ id: pluginId }) => {
            if (!this.isPluginActive(pluginId)) {
                this.activatePlugin(pluginId);
            }
        });
    }

    private activatePlugin(pluginId: string): void {
        const plugin = this.installedPlugins.get(pluginId);
        if (!plugin) throw new Error(`Plugin not found: ${pluginId}`);
        if (this.desiredActivationState.get(pluginId) == true) return;

        const deps = (plugin.requires || []).map(token => {
            const instance = this.services.get(token);
            if (!instance) throw new Error(`Missing required dependency: ${token.debugName}`);
            return instance;
        });

        const optionalDeps = (plugin.optional || []).map(token => this.services.get(token));

        const instance = plugin.activate(this, ...deps, ...optionalDeps);
        if (plugin.provides) this.services.set(plugin.provides, instance);
        this.desiredActivationState.set(pluginId, true);
    }

    // Deactivates the specified plugin-ids and their dependents in logical order
    deactivatePlugins(pluginIds: string[]): void {
        pluginIds.forEach(pluginId => {
            if (this.isPluginActive(pluginId)) this.deactivatePlugin(pluginId);
        });
    }

    private deactivatePlugin(pluginId: string): void {
        const deactivationOrder = this.depGraph.getDeactivationOrder(pluginId);
        deactivationOrder.forEach(plugin => {
            if (this.desiredActivationState.get(plugin.id) == true) {
                if (plugin.deactivate) {
                    plugin.deactivate();
                }

                // If the plugin provides a service, remove it from the services map
                if (plugin.provides) {
                    this.services.delete(plugin.provides);
                }

                this.desiredActivationState.set(pluginId, false);
            }
        });
    }

    // This allows you to activate / deactivate plugins imperatively
    setPluginsState(newPluginStates: Map<string, boolean>): void {
        // Filter out plugins we don't have installed
        newPluginStates.forEach((state, pluginId) => {
            if (!this.installedPlugins.has(pluginId)) {
                newPluginStates.delete(pluginId);
            }
        });

        const pluginsToDeactivate = Array.from(newPluginStates.entries())
            .filter(([pluginId, on]) => (!on && this.isPluginActive(pluginId)) == true)
            .map(([pluginId, _]) => pluginId);

        const pluginsToActivate = Array.from(newPluginStates.entries())
            .filter(([pluginId, on]) => (on && !this.isPluginActive(pluginId)) == false)
            .map(([pluginId, _]) => pluginId);

        // Deactivate plugins first
        this.deactivatePlugins(pluginsToDeactivate);

        // Then activate plugins
        this.activatePlugins(pluginsToActivate);
    }

    isPluginInstalled(pluginId: string): boolean {
        return this.installedPlugins.has(pluginId);
    }

    isPluginActive(pluginId: string): boolean {
        if (!this.isPluginInstalled(pluginId)) return false;
        return this.desiredActivationState.get(pluginId) ?? false;
    }

    // For UIs to display available and active registeredPlugins
    getAllPlugins(): { pluginId: string; isActive: boolean }[] {
        return Array.from(this.installedPlugins.entries()).map(([pluginId, _plugin]) => ({
            pluginId: pluginId,
            isActive: this.isPluginActive(pluginId),
        }));
    }

    addFunction(key: string, fn: (...args: any[]) => any) {
        this.functionArray.set(key, fn);
    }

    executeFunction(key: string, ...args: any[]): any {
        const fn = this.functionArray.get(key);
        if (fn) {
            return fn(...args);
        }
        console.error('No function added for the key ' + key + '.');
    }

    removeFunction(key: string): void {
        this.functionArray.delete(key);
    }

    addEventListener<EventType = Event>(name: string, callback: (event: EventType) => void) {
        this._eventCallableRegistry.addEventListener(name, callback);
    }
    removeEventListener<EventType = Event>(name: string, callback: (event: EventType) => void) {
        this._eventCallableRegistry.removeEventListener(name, callback);
    }
    dispatchEvent<EventType extends Event = Event>(event: EventType) {
        this._eventCallableRegistry.dispatchEvent(event);
    }

    // Type-safe setters and getters
    set<T>(token: Token<T>, instance: T): void {
        this.services.set(token, instance);
    }
    get<T>(token: Token<T>): T | undefined {
        return this.services.get(token) as T | undefined;
    }
}

// Meant to be used as a singleton
export const pluginStore = new PluginStore();
