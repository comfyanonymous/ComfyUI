// This is the main entry point of the app. Main lifecycle:
// 1. external script loads module
// 2. external script calls api(api-options, load-external = true), to configure the api
//      1a. we query the api for extensions to load
//      1b. we dynamically import the extensions
//      1c. the extensions register themselvs with the extensionManager
// 2. external script calls app.setup(canvasElement), where the app will be mounted
//      2a. app creates a new ComfyGraph and ComfyCanvas
//

import { LiteGraph } from 'litegraph.js';
import { api } from './api';
import { loadWebExtensions } from '../extension_manager/loadWebExtensions';
import { ExtensionManager } from '../extension_manager/extensionManager';

// Ask the api-server what front-end extensions to load, if any, and then load them
const webModuleUrls = await api.getExtensions();
const comfyPlugins = await loadWebExtensions(webModuleUrls);
const extManager = ExtensionManager.getInstance();
extManager.registerPlugins(comfyPlugins);

export async function mountLiteGraph(mainCanvas: HTMLCanvasElement) {
    await userSettings.setUser();
    // await this.ui.settings.load();

    // Mount the LiteGraph in the DOM
    mainCanvas.style.touchAction = 'none';
    const canvasEl = (this.canvasEl = Object.assign(mainCanvas, {
        id: 'graph-canvas',
    }));
    canvasEl.tabIndex = 1;
    document.body.prepend(canvasEl);

    addDomClippingSetting();

    this.graph = new ComfyGraph();
    this.canvas = new ComfyCanvas(canvasEl, this.graph);
    this.ctx = canvasEl.getContext('2d');

    LiteGraph.release_link_on_empty_shows_menu = true;
    LiteGraph.alt_drag_do_clone_nodes = true;

    this.graph.start();

    // Load previous workflow
    const restored = await loadWorkflow();

    // We failed to restore a workflow so load the default
    if (!restored) {
        await this.loadGraphData();
    }

    // Save current workflow automatically
    this.saveInterval = setInterval(
        () => localStorage.setItem('workflow', JSON.stringify(this.graph?.serialize())),
        1000
    );

    this.#addDropHandler();
    this.#addCopyHandler();
    this.#addPasteHandler();
    this.#addKeyboardHandler();
    this.#addApiUpdateHandlers(api);

    await extensionManager.invokeExtensionsAsync('setup');
}
