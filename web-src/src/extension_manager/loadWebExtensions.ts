import { logging } from '../scripts/logging';
import { IComfyPlugin } from '../types/interfaces';

// Loads all specified .js-files into the window in parallel
export async function loadWebExtensions(webModuleUrls: string[]) {
    logging.addEntry('Comfy.App', 'debug', { Extensions: webModuleUrls });

    const extensionPromises = webModuleUrls.map(async ext => {
        try {
            const module = await import(ext);
            if (module.default && typeof module.default === 'object') {
                return module.default as IComfyPlugin<any>;
            } else {
                console.error('Module does not export a default object', ext);
                return null;
            }
        } catch (error) {
            console.error('Error loading extension', ext, error);
        }
    });

    const extensions = await Promise.all(extensionPromises);
    return extensions.filter((ext): ext is IComfyPlugin<any> => ext !== null);
}
