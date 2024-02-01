// This class acts as a gateway for plugins

import { IComfyApp } from '../types/interfaces';

// Single class
export class ComfyApp implements IComfyApp {
    private static instance: IComfyApp;

    getInstance() {
        if (!ComfyApp.instance) {
            ComfyApp.instance = new ComfyApp();
        }
        return ComfyApp.instance;
    }

    // LiteGraph
    // the api
    // ui components?
}
