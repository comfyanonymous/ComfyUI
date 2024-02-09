import { logging } from '../scripts/logging';
import { IComfyPlugin, ModuleWithPlugins } from '../types/interfaces';

// Loads all specified .js-files into the window in parallel
// extensionPaths can be either a URL or a local-filesystem path
export async function loadExtensions(extensionPaths: string[]) {
    logging.addEntry('Comfy.App', 'debug', { Extensions: extensionPaths });

    const extensionPromises = extensionPaths.map(async ext => {
        try {
            const module = (await import(ext)) as ModuleWithPlugins<any>;
            if (module.default) {
                const defaults = Array.isArray(module.default) ? module.default : [module.default];
                // Filter out non-objects
                return defaults.filter(item => typeof item === 'object');
            } else {
                console.error('Module does not export a default', ext);
                return null;
            }
        } catch (error) {
            console.error('Error loading extension', ext, error);
        }
    });

    const extensions = await Promise.all(extensionPromises);
    return extensions.flat().filter((ext): ext is IComfyPlugin<any> => ext !== null);
}
