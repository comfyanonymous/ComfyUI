// This is the main entry point of the app

import { app } from './app';
import { api } from './api';
import { loadCustomNodes } from './loadCustomNodes';

// Ask the api-server what custom-nodes to load, if any, and then load them
const jsModuleUrls = await api.getExtensions();
await loadCustomNodes(jsModuleUrls);

// Every custom-node is built with the assumption that ComfyApp is a singleton
// class that is already instantiated and can be imported here.
export { app };
