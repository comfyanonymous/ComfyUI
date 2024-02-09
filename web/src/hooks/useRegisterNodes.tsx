import { api } from '../scripts/api.tsx';
import { ComfyNode } from '../litegraph/comfyNode.ts';
import { ComfyObjectInfo } from '../types/comfy.ts';
import { widgetState } from '../litegraph/widgetFactory.ts';
import { LiteGraph } from 'litegraph.js';

// TO DO: we need an alternative to the invokeExtensionsAsync callbacks
// The purpose of this was so that the extensions could manually over-ride the IComfyNode constructor
// We need some sort of impoted custom nodes as well

export function useRegisterNodes() {
    // const { invokeExtensionsAsync } = useExtensionManager();

    const registerNodes = async () => {
        // Load node definitions from the backend
        const defs = await api.getNodeDefs();
        await registerNodesFromDefs(defs);
        // await invokeExtensionsAsync('registerCustomNodes');
    };

    // Register a node class so it can be listed when we want to create a new one
    const registerNodeDef = (nodeId: string, nodeData: any) => {
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
    };

    const registerNodesFromDefs = async (defs: Record<string, ComfyObjectInfo>) => {
        // await extensionManager.invokeExtensionsAsync('addCustomNodeDefs', defs);

        // Refresh list of known widgets
        await widgetState.refresh();

        // Register a node for each definition
        for (const nodeId in defs) {
            registerNodeDef(nodeId, defs[nodeId]);
        }
    };

    return {
        registerNodes,
        registerNodeDef,
        registerNodesFromDefs,
    };
}
