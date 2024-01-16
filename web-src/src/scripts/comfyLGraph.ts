import {LGraph, LGraphNode} from "litegraph.js";
import {ComfyApp} from "./app";
import {ComfyNode} from "./comfyNode";

export class ComfyLGraph extends LGraph {
    app: ComfyApp
    nodes: ComfyNode[]

    onConfigure?: () => void

    constructor(app: ComfyApp) {
        super();

        this.app = app
        this.nodes = []
    }

    getNodeById(id: number) {
        return super.getNodeById(id) as ComfyNode;
    }
}