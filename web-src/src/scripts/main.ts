// This is the main entry point of the app. Main lifecycle:
// 1. external script loads module
// 2. external script calls api(api-options, load-external = true), to configure the api
//      1a. we query the api for extensions to load
//      1b. we dynamically import the extensions
//      1c. the extensions register themselvs with the extensionManager
// 2. external script calls app.setup(canvasElement), where the app will be mounted
//      2a. app creates a new ComfyGraph and ComfyCanvas
//

import { app } from './app';
import { api } from './api';
import { loadCustomNodes } from './loadCustomNodes';

// Ask the api-server what custom-nodes to load, if any, and then load them
const jsModuleUrls = await api.getExtensions();
await loadCustomNodes(jsModuleUrls);

// Every custom-node is built with the assumption that ComfyApp is a singleton
// class that is already instantiated and can be imported here.
export { app };
