// This class acts as a gateway for plugins

import {IComfyApp} from '../types/interfaces';
import {ComfyGraph} from "../litegraph/comfyGraph.ts";

// Single class
export class ComfyApp implements IComfyApp {
    private static instance: IComfyApp;

    private saveInterval: NodeJS.Timeout | null = null;

    public static getInstance() {
        if (!ComfyApp.instance) {
            ComfyApp.instance = new ComfyApp();
        }
        return ComfyApp.instance;
    }

    public enableWorkflowAutoSave(graph: ComfyGraph) {
        this.saveInterval = setInterval(
            () => localStorage.setItem('workflow', JSON.stringify(graph.serializeGraph())),
            1000
        );
    }

    public disableWorkflowAutoSave() {
        if (!this.saveInterval) return;

        clearInterval(this.saveInterval);
        this.saveInterval = null;
    }

    // TODO: implement
    clean() {}

    // LiteGraph
    // the api
    // ui components?
}
