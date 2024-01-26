import { LGraph } from 'litegraph.js';
import type {ComfyApp} from './app';
import type {ComfyNode} from './comfyNode';

export class ComfyGraph extends LGraph {
    app: ComfyApp;

    // Overwrite _nodes as ComfyNode[] rather than LGraphNode[]
    // @ts-expect-error
    get nodes(): ComfyNode[] {
        // @ts-expect-error
        return super.nodes as ComfyNode[];
    }

    constructor(app: ComfyApp, ...args: ConstructorParameters<typeof LGraph>) {
        super(...args);
        this.app = app;
    }

    // I don't like this `configure` or `onConfigure` pattern; seeems really newbish
    // But I'm keeping it for now, because that's how ComfyUI did it
    configure(data: object, keep_old?: boolean): boolean | undefined {
        // Flag that the graph is configuring to prevent nodes from running checks while its still loading
        this.app.configuringGraph = true;
        try {
            return super.configure(data, keep_old);
        } finally {
            this.app.configuringGraph = false;
        }
    }

    onConfigure(data: object): void {
        // Fire callbacks before the onConfigure, this is used by widget inputs to setup the config
        for (const node of this.nodes) {
            node.onGraphConfigured?.();
        }

        const r = super.onConfigure ? super.onConfigure(data) : undefined;

        // Fire after onConfigure, used by primitves to generate widget using input nodes config
        for (const node of this.nodes) {
            node.onAfterGraphConfigured?.();
        }

        // Why does ComfyUI return anything here? This is always void foom what I can tell
        return r;
    }

    // ===== Override LGraph methods, repalcing LGraphNode with ComfyNode =====

    add(node: ComfyNode, skip_compute_order?: boolean): void {
        super.add(node, skip_compute_order);
    }

    onNodeAdded(node: ComfyNode): void {
        super.onNodeAdded(node);
    }

    remove(node: ComfyNode): void {
        super.remove(node);
    }

    getNodeById(id: number): ComfyNode | undefined {
        return super.getNodeById(id) as ComfyNode; // Cast the result to ComfyNode
    }

    getAncestors(node: ComfyNode): ComfyNode[] {
        return super.getAncestors(node) as ComfyNode[]; // Cast the result to ComfyNode[]
    }

    // findNodesByClass<T extends LGraphNode>(
    //   classObject: LGraphNodeConstructor<T>
    // ): T[] {
    //   return super.findNodesByClass(classObject) as T[]; // Cast the result to an array of ComfyNode
    // }

    // findNodesByType<T extends ComfyNode = ComfyNode>(type: string): T[] {
    //   return super.findNodesByType(type) as T[]; // Cast the result to an array of ComfyNode
    // }

    // findNodeByTitle<T extends ComfyNode = ComfyNode>(title: string): T | null {
    //   return super.findNodeByTitle(title) as T; // Cast the result to ComfyNode
    // }

    // findNodesByTitle<T extends ComfyNode = ComfyNode>(title: string): T[] {
    //   return super.findNodesByTitle(title) as T[]; // Cast the result to an array of ComfyNode
    // }

    // getNodeOnPos<T extends ComfyNode = ComfyNode>(
    //   x: number,
    //   y: number,
    //   node_list?: ComfyNode[],
    //   margin?: number
    // ): T | null {
    //   return super.getNodeOnPos(x, y, node_list, margin) as T; // Cast the result to ComfyNode
    // }

    beforeChange(info?: ComfyNode): void {
        super.beforeChange(info);
    }

    afterChange(info?: ComfyNode): void {
        super.afterChange(info);
    }

    connectionChange(node: ComfyNode): void {
        super.connectionChange(node);
    }
}
