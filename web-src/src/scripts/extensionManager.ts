// Right now, ComfyApp uses a service-oriented paradigm for extensions, rather
// than an event-driven paradigm. This means ComfyApp records all of the
// extensions in a list, and invokes them in order. In the future, we may switch
// from service-oriented to event-driven, which would be simpler.

import { ComfyExtension } from '../types/interfaces';

class ExtensionManager {
    private static instance: ExtensionManager;
    private extensions: ComfyExtension[] = [];

    private constructor() {}

    static getInstance(): ExtensionManager {
        if (!this.instance) {
            this.instance = new ExtensionManager();
        }
        return this.instance;
    }

    registerExtension(extension: ComfyExtension): void {
        if (!extension.name) {
            throw new Error("Extensions must have a 'name' property.");
        }
        if (this.extensions.find(ext => ext.name === extension.name)) {
            throw new Error(`Extension named '${extension.name}' already registered.`);
        }
        this.extensions.push(extension);
    }

    /** Invoke an async extension callback
     * Each callback will be invoked concurrently */
    async invokeExtensionsAsync<K extends keyof ComfyExtension>(method: K, ...args: any[]): Promise<any[]> {
        return await Promise.all(
            this.extensions
                .map(async ext => {
                    const func = ext[method];
                    if (typeof func === 'function') {
                        try {
                            // @ts-ignore
                            return await func(...args);
                        } catch (error) {
                            console.error(
                                `Error calling extension '${ext.name}' method '${String(method)}'`,
                                { error },
                                { extension: ext },
                                { args }
                            );
                        }
                    }
                })
                .filter(promise => promise !== undefined) // Filter out undefined promises from non-existent methods
        );
    }
}

export const extensionManager = ExtensionManager.getInstance();
