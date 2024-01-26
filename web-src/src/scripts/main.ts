import { ComfyApp } from './app';
import { api } from './api';

// Ask the api-server what extensions to load, if any, and then load them
const extensionUrls = await api.getExtensions();
await ComfyApp.loadExtensions(extensionUrls);

// Every custom-node is built with the assumption that ComfyApp is a singleton
// class that is already instantiated and can be imported here.
export const app = new ComfyApp();
