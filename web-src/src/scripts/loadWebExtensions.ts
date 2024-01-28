import { logging } from './logging';

/** Loads all specified .js-files into the window in parallel */
export async function loadWebExtensions(jsModuleUrls: string[]) {
    logging.addEntry('Comfy.App', 'debug', { Extensions: jsModuleUrls });

    const extensionPromises = jsModuleUrls.map(async ext => {
        try {
            await import(ext);
        } catch (error) {
            console.error('Error loading extension', ext, error);
        }
    });

    await Promise.all(extensionPromises);
}
