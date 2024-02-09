import type { IComfyPlugin } from '../types/interfaces';

// Used to instantiate registeredPlugins in the correct order
export class DependencyGraph {
    private nodes: Map<string, DependencyNode> = new Map();

    addPlugins(plugins: IComfyPlugin<any>[]): void {
        // First pass: Add all nodes to the graph without dependencies
        plugins.forEach(plugin => {
            if (!this.nodes.has(plugin.id)) {
                const node = new DependencyNode(plugin);
                this.nodes.set(plugin.id, node);
            }
        });

        // Second pass: Now that all nodes are added, establish dependencies
        plugins.forEach(plugin => {
            const node = this.nodes.get(plugin.id)!;
            (plugin.requires || []).forEach(depId => {
                const depNode = this.findNodeById(depId);
                if (depNode) {
                    node.addDependency(depNode);
                } else {
                    console.error(`Plugin dependency not found: ${depId}`);
                    return;
                }
            });
        });
    }

    getActivationOrder(startingNodes: string[]): IComfyPlugin<any>[] {
        const order: IComfyPlugin<any>[] = [];
        const visited = new Set<string>();
        const tempMark = new Set<string>();
        const startingNodesSet = new Set(startingNodes);

        const visit = (node: DependencyNode) => {
            if (tempMark.has(node.plugin.id)) {
                throw new Error(`Circular dependency detected: ${node.plugin.id}`);
            }
            if (!visited.has(node.plugin.id)) {
                tempMark.add(node.plugin.id);
                node.dependencies.forEach(dep => visit(dep));
                tempMark.delete(node.plugin.id);
                visited.add(node.plugin.id);
                order.push(node.plugin);
            }
        };

        // Filter the nodes to only include those in the startingNodes list
        Array.from(this.nodes.values())
            .filter(node => startingNodesSet.has(node.plugin.id))
            .forEach(node => visit(node));

        return order;
    }

    // Returns a list of plugins to deactivate if the specified plugin is deactivated
    getDeactivationOrder(pluginId: string): IComfyPlugin<any>[] {
        const toDeactivate: IComfyPlugin<any>[] = [];
        const stack = [pluginId];
        const visited = new Set<string>(); // Added to track visited nodes

        while (stack.length > 0) {
            const currentId = stack.pop()!;
            // Check if the currentId has already been visited
            if (visited.has(currentId)) {
                continue; // Skip the iteration to prevent infinite loop
            }
            visited.add(currentId); // Mark the currentId as visited

            const currentNode = this.nodes.get(currentId);
            if (currentNode) {
                if (!toDeactivate.find(plugin => plugin.id === currentNode.plugin.id)) {
                    toDeactivate.push(currentNode.plugin);
                    currentNode.dependents.forEach(dependent => {
                        if (!stack.includes(dependent.plugin.id) && !visited.has(dependent.plugin.id)) {
                            // Check if not visited
                            stack.push(dependent.plugin.id);
                        }
                    });
                }
            }
        }

        return toDeactivate.reverse();
    }

    private findNodeById(id: string): DependencyNode | undefined {
        return Array.from(this.nodes.values()).find(node => node.plugin.provides === id);
    }
}

// Used to create a dependency tree
class DependencyNode {
    plugin: IComfyPlugin<any>;
    dependencies: DependencyNode[] = [];
    dependents: DependencyNode[] = [];

    constructor(plugin: IComfyPlugin<any>) {
        this.plugin = plugin;
    }

    addDependency(node: DependencyNode): void {
        this.dependencies.push(node);
        node.dependents.push(this);
    }
}
