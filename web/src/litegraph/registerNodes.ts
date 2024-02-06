// In order to instantiate a node in LiteGraph, it must be defined (registered) with
// LiteGraph first.

import { LiteGraph } from 'litegraph.js';
import { ComfyNode } from './comfyNode';
import { widgetState } from './widgetFactory';
import { ComfyObjectInfo } from '../types/comfy';
import { api } from '../scripts/api';

/** Load node-definitions from the server and register them */
export async function registerNodes() {
    const defs = await api.getNodeDefs();
    await registerNodesFromDefs(defs);
    // await extensionManager.invokeExtensionsAsync('registerCustomNodes');
}

export async function registerNodesFromDefs(defs: Record<string, ComfyObjectInfo>) {
    // await extensionManager.invokeExtensionsAsync('addCustomNodeDefs', defs);

    // Refresh list of known widgets
    await widgetState.refresh();

    // Register a node for each definition
    for (const nodeId in defs) {
        registerNodeDef(nodeId, defs[nodeId]);
    }
}

// Register a node class so it can be listed when we want to create a new one
export async function registerNodeDef(nodeId: string, nodeData: any) {
    // Capture nodeData and app in a closure and return a new constructor function
    const comfyNodeConstructor = class extends ComfyNode {
        static title: string;
        static comfyClass: string;
        static nodeData: any;

        constructor() {
            super(nodeData);
        }
    };

    // Add these properties in as well to maintain consistency for the extension-message broadcast
    // we're about to do. Idk if any of the extensions actually use this info though.
    comfyNodeConstructor.title = nodeData.display_name || nodeData.name;
    comfyNodeConstructor.comfyClass = nodeData.name;
    comfyNodeConstructor.nodeData = nodeData;

    // await extensionManager.invokeExtensionsAsync('beforeRegisterNodeDef', comfyNodeConstructor, nodeData);

    LiteGraph.registerNodeType(nodeId, comfyNodeConstructor);
}
