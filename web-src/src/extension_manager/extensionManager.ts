import { IComfyPlugin, Application, Token } from '../types/interfaces';
import { DependencyGraph } from './dependencyGraph';

// TO DO: fill this in later with an actual object
const app = {} as Application;

// Singleton class
export class ExtensionManager {
    private static instance: ExtensionManager;

    // Stores all registered plugins
    private registeredPlugins: Map<string, IComfyPlugin<any>> = new Map();

    // Set of all currently active Plugins, using plugin-id
    private activePlugins: Set<string> = new Set();

    // Stores instances of all services provided by activated plugins
    private instances: Map<Token<any>, any> = new Map();

    // Stores the dependency graph between plugin-services
    private depGraph: DependencyGraph = new DependencyGraph();

    // Gets or creates the singleton instance of `ExtensionManager`
    public static getInstance(): ExtensionManager {
        if (!ExtensionManager.instance) {
            ExtensionManager.instance = new ExtensionManager();
        }
        return ExtensionManager.instance;
    }

    // This is idempotent; a plugin can only be registered once. Plugin ids should be globally unique
    // All plugins with "autoStart" set to true will automatically be activated
    registerPlugins(plugins: IComfyPlugin<any>[]): void {
        const newPlugins = plugins.filter(plugin => !this.registeredPlugins.has(plugin.id));
        newPlugins.forEach(plugin => this.registeredPlugins.set(plugin.id, plugin));
        this.depGraph.addPlugins(newPlugins);

        // Activate all plugins with `autoStart` set to true
        const autoStartPluginIds = newPlugins.filter(plugin => plugin.autoStart).map(plugin => plugin.id);
        this.activatePlugins(autoStartPluginIds);
    }

    // Activates the specified plugin-ids and their dependencies in logical order
    activatePlugins(pluginIds: string[]): void {
        const activationOrder = this.depGraph.getActivationOrder(pluginIds);
        activationOrder.forEach(plugin => {
            if (this.registeredPlugins.has(plugin.id) && !this.isPluginActive(plugin.id)) {
                this.activatePlugin(plugin.id);
            }
        });
    }

    private activatePlugin(pluginId: string): void {
        const plugin = this.registeredPlugins.get(pluginId);
        if (!plugin) throw new Error(`Plugin not found: ${pluginId}`);
        if (this.activePlugins.has(pluginId)) return;

        // Find all required dependencies
        const deps = (plugin.requires || []).map(token => {
            const instance = this.instances.get(token);
            if (!instance) throw new Error(`Missing required dependency: ${token.name}`);
            return instance;
        });

        // Include the optional dependencies, if they exist
        const optionalDeps = (plugin.optional || []).map(token => this.instances.get(token));

        // Activate plugin, and if it provides a service, save it
        const instance = plugin.activate(app, ...deps, ...optionalDeps);
        if (plugin.provides) this.instances.set(plugin.provides, instance);
        this.activePlugins.add(pluginId);
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
            if (this.activePlugins.has(plugin.id)) {
                if (plugin.deactivate) {
                    plugin.deactivate(app);
                }

                // If the plugin provides a service, remove it from the instances map
                if (plugin.provides) {
                    this.instances.delete(plugin.provides);
                }

                // Remove the plugin from the list of active plugins
                this.activePlugins.delete(plugin.id);
            }
        });
    }

    isPluginActive(pluginId: string): boolean {
        return this.activePlugins.has(pluginId);
    }

    // For UIs to display available and active registeredPlugins
    getAllPlugins(): { pluginId: string; isActive: boolean }[] {
        return Array.from(this.registeredPlugins.entries()).map(([pluginId, _plugin]) => ({
            pluginId: pluginId,
            isActive: this.isPluginActive(pluginId),
        }));
    }
}
