// This is the main entry point of the app. Main lifecycle:
// 1. external script loads module
// 2. external script calls api(api-options, load-external = true), to configure the api
//      1a. we query the api for extensions to load
//      1b. we dynamically import the extensions
//      1c. the extensions register themselvs with the extensionManager
// 2. external script calls app.setup(canvasElement), where the app will be mounted
//      2a. app creates a new ComfyGraph and ComfyCanvas
//

import { api } from './api';
import { loadWebExtensions } from '../pluginManager/loadExtensions';
import { ExtensionManager } from '../extension_manager/extensionManager';

// Ask the api-server what front-end extensions to load, if any, and then load them
const webModuleUrls = await api.getExtensions();
const comfyPlugins = await loadWebExtensions(webModuleUrls);

const extManager = ExtensionManager.getInstance();
extManager.registerPlugins(comfyPlugins);
