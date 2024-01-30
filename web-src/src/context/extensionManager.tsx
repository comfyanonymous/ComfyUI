import {createContext, ReactNode, useContext, useState} from "react";
import {ComfyExtension} from "../types/interfaces.ts";

interface ExtensionManagerContextType {
    extensions: ComfyExtension[]
    registerExtension: (extension: ComfyExtension) => void
    invokeExtensionsAsync: <K extends keyof ComfyExtension>(method: K, ...args: any[]) => Promise<any[]>
}

const ExtensionManagerContext = createContext<ExtensionManagerContextType | null>(null)

export function useExtensionManager() {
    const context = useContext(ExtensionManagerContext);
    if (!context) {
        throw new Error("useExtensionManager must be used within a ExtensionManagerProvider");
    }

    return context
}

export function ExtensionManagerProvider({children}: { children: ReactNode }) {
    const [extensions, setExtensions] = useState<ComfyExtension[]>([])

    const registerExtension = (extension: ComfyExtension) => {
        if (!extension.name) {
            throw new Error("Extensions must have a 'name' property.");
        }

        if (extensions.find(ext => ext.name === extension.name)) {
            throw new Error(`Extension named '${extension.name}' already registered.`);
        }
        setExtensions((prev) => [...prev, extension])
    }

    /** Invoke an async extension callback
     * Each callback will be invoked concurrently */
    const invokeExtensionsAsync = async <K extends keyof ComfyExtension>(method: K, ...args: any[]): Promise<any[]> => {
        return await Promise.all(
            extensions
                .map(async ext => {
                    const func = ext[method];
                    if (typeof func === 'function') {
                        try {
                            // @ts-ignore
                            return await func(...args);
                        } catch (error) {
                            console.error(
                                `Error calling extension '${ext.name}' method '${String(method)}'`,
                                {error},
                                {extension: ext},
                                {args}
                            );
                        }
                    }
                })
                .filter(promise => promise !== undefined) // Filter out undefined promises from non-existent methods
        );
    }

    return (
        <ExtensionManagerContext.Provider
            value={{
                extensions,
                registerExtension,
                invokeExtensionsAsync
            }}
        >
            {children}
        </ExtensionManagerContext.Provider>
    )
}
