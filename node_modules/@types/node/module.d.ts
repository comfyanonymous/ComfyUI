/**
 * @since v0.3.7
 * @experimental
 */
declare module "module" {
    import { URL } from "node:url";
    import { MessagePort } from "node:worker_threads";
    namespace Module {
        /**
         * The `module.syncBuiltinESMExports()` method updates all the live bindings for
         * builtin `ES Modules` to match the properties of the `CommonJS` exports. It
         * does not add or remove exported names from the `ES Modules`.
         *
         * ```js
         * const fs = require('node:fs');
         * const assert = require('node:assert');
         * const { syncBuiltinESMExports } = require('node:module');
         *
         * fs.readFile = newAPI;
         *
         * delete fs.readFileSync;
         *
         * function newAPI() {
         *   // ...
         * }
         *
         * fs.newAPI = newAPI;
         *
         * syncBuiltinESMExports();
         *
         * import('node:fs').then((esmFS) => {
         *   // It syncs the existing readFile property with the new value
         *   assert.strictEqual(esmFS.readFile, newAPI);
         *   // readFileSync has been deleted from the required fs
         *   assert.strictEqual('readFileSync' in fs, false);
         *   // syncBuiltinESMExports() does not remove readFileSync from esmFS
         *   assert.strictEqual('readFileSync' in esmFS, true);
         *   // syncBuiltinESMExports() does not add names
         *   assert.strictEqual(esmFS.newAPI, undefined);
         * });
         * ```
         * @since v12.12.0
         */
        function syncBuiltinESMExports(): void;
        /**
         * `path` is the resolved path for the file for which a corresponding source map
         * should be fetched.
         * @since v13.7.0, v12.17.0
         * @return Returns `module.SourceMap` if a source map is found, `undefined` otherwise.
         */
        function findSourceMap(path: string, error?: Error): SourceMap;
        interface SourceMapPayload {
            file: string;
            version: number;
            sources: string[];
            sourcesContent: string[];
            names: string[];
            mappings: string;
            sourceRoot: string;
        }
        interface SourceMapping {
            generatedLine: number;
            generatedColumn: number;
            originalSource: string;
            originalLine: number;
            originalColumn: number;
        }
        interface SourceOrigin {
            /**
             * The name of the range in the source map, if one was provided
             */
            name?: string;
            /**
             * The file name of the original source, as reported in the SourceMap
             */
            fileName: string;
            /**
             * The 1-indexed lineNumber of the corresponding call site in the original source
             */
            lineNumber: number;
            /**
             * The 1-indexed columnNumber of the corresponding call site in the original source
             */
            columnNumber: number;
        }
        /**
         * @since v13.7.0, v12.17.0
         */
        class SourceMap {
            /**
             * Getter for the payload used to construct the `SourceMap` instance.
             */
            readonly payload: SourceMapPayload;
            constructor(payload: SourceMapPayload);
            /**
             * Given a line offset and column offset in the generated source
             * file, returns an object representing the SourceMap range in the
             * original file if found, or an empty object if not.
             *
             * The object returned contains the following keys:
             *
             * The returned value represents the raw range as it appears in the
             * SourceMap, based on zero-indexed offsets, _not_ 1-indexed line and
             * column numbers as they appear in Error messages and CallSite
             * objects.
             *
             * To get the corresponding 1-indexed line and column numbers from a
             * lineNumber and columnNumber as they are reported by Error stacks
             * and CallSite objects, use `sourceMap.findOrigin(lineNumber, columnNumber)`
             * @param lineOffset The zero-indexed line number offset in the generated source
             * @param columnOffset The zero-indexed column number offset in the generated source
             */
            findEntry(lineOffset: number, columnOffset: number): SourceMapping;
            /**
             * Given a 1-indexed `lineNumber` and `columnNumber` from a call site in the generated source,
             * find the corresponding call site location in the original source.
             *
             * If the `lineNumber` and `columnNumber` provided are not found in any source map,
             * then an empty object is returned.
             * @param lineNumber The 1-indexed line number of the call site in the generated source
             * @param columnNumber The 1-indexed column number of the call site in the generated source
             */
            findOrigin(lineNumber: number, columnNumber: number): SourceOrigin | {};
        }
        /** @deprecated Use `ImportAttributes` instead */
        interface ImportAssertions extends ImportAttributes {}
        interface ImportAttributes extends NodeJS.Dict<string> {
            type?: string | undefined;
        }
        type ModuleFormat = "builtin" | "commonjs" | "json" | "module" | "wasm";
        type ModuleSource = string | ArrayBuffer | NodeJS.TypedArray;
        interface GlobalPreloadContext {
            port: MessagePort;
        }
        /**
         * @deprecated This hook will be removed in a future version.
         * Use `initialize` instead. When a loader has an `initialize` export, `globalPreload` will be ignored.
         *
         * Sometimes it might be necessary to run some code inside of the same global scope that the application runs in.
         * This hook allows the return of a string that is run as a sloppy-mode script on startup.
         *
         * @param context Information to assist the preload code
         * @return Code to run before application startup
         */
        type GlobalPreloadHook = (context: GlobalPreloadContext) => string;
        /**
         * The `initialize` hook provides a way to define a custom function that runs in the hooks thread
         * when the hooks module is initialized. Initialization happens when the hooks module is registered via `register`.
         *
         * This hook can receive data from a `register` invocation, including ports and other transferrable objects.
         * The return value of `initialize` can be a `Promise`, in which case it will be awaited before the main application thread execution resumes.
         */
        type InitializeHook<Data = any> = (data: Data) => void | Promise<void>;
        interface ResolveHookContext {
            /**
             * Export conditions of the relevant `package.json`
             */
            conditions: string[];
            /**
             * @deprecated Use `importAttributes` instead
             */
            importAssertions: ImportAttributes;
            /**
             *  An object whose key-value pairs represent the assertions for the module to import
             */
            importAttributes: ImportAttributes;
            /**
             * The module importing this one, or undefined if this is the Node.js entry point
             */
            parentURL: string | undefined;
        }
        interface ResolveFnOutput {
            /**
             * A hint to the load hook (it might be ignored)
             */
            format?: ModuleFormat | null | undefined;
            /**
             * @deprecated Use `importAttributes` instead
             */
            importAssertions?: ImportAttributes | undefined;
            /**
             * The import attributes to use when caching the module (optional; if excluded the input will be used)
             */
            importAttributes?: ImportAttributes | undefined;
            /**
             * A signal that this hook intends to terminate the chain of `resolve` hooks.
             * @default false
             */
            shortCircuit?: boolean | undefined;
            /**
             * The absolute URL to which this input resolves
             */
            url: string;
        }
        /**
         * The `resolve` hook chain is responsible for resolving file URL for a given module specifier and parent URL, and optionally its format (such as `'module'`) as a hint to the `load` hook.
         * If a format is specified, the load hook is ultimately responsible for providing the final `format` value (and it is free to ignore the hint provided by `resolve`);
         * if `resolve` provides a format, a custom `load` hook is required even if only to pass the value to the Node.js default `load` hook.
         *
         * @param specifier The specified URL path of the module to be resolved
         * @param context
         * @param nextResolve The subsequent `resolve` hook in the chain, or the Node.js default `resolve` hook after the last user-supplied resolve hook
         */
        type ResolveHook = (
            specifier: string,
            context: ResolveHookContext,
            nextResolve: (
                specifier: string,
                context?: ResolveHookContext,
            ) => ResolveFnOutput | Promise<ResolveFnOutput>,
        ) => ResolveFnOutput | Promise<ResolveFnOutput>;
        interface LoadHookContext {
            /**
             * Export conditions of the relevant `package.json`
             */
            conditions: string[];
            /**
             * The format optionally supplied by the `resolve` hook chain
             */
            format: ModuleFormat;
            /**
             * @deprecated Use `importAttributes` instead
             */
            importAssertions: ImportAttributes;
            /**
             *  An object whose key-value pairs represent the assertions for the module to import
             */
            importAttributes: ImportAttributes;
        }
        interface LoadFnOutput {
            format: ModuleFormat;
            /**
             * A signal that this hook intends to terminate the chain of `resolve` hooks.
             * @default false
             */
            shortCircuit?: boolean | undefined;
            /**
             * The source for Node.js to evaluate
             */
            source?: ModuleSource;
        }
        /**
         * The `load` hook provides a way to define a custom method of determining how a URL should be interpreted, retrieved, and parsed.
         * It is also in charge of validating the import assertion.
         *
         * @param url The URL/path of the module to be loaded
         * @param context Metadata about the module
         * @param nextLoad The subsequent `load` hook in the chain, or the Node.js default `load` hook after the last user-supplied `load` hook
         */
        type LoadHook = (
            url: string,
            context: LoadHookContext,
            nextLoad: (url: string, context?: LoadHookContext) => LoadFnOutput | Promise<LoadFnOutput>,
        ) => LoadFnOutput | Promise<LoadFnOutput>;
    }
    interface RegisterOptions<Data> {
        parentURL: string | URL;
        data?: Data | undefined;
        transferList?: any[] | undefined;
    }
    interface Module extends NodeModule {}
    class Module {
        static runMain(): void;
        static wrap(code: string): string;
        static createRequire(path: string | URL): NodeRequire;
        static builtinModules: string[];
        static isBuiltin(moduleName: string): boolean;
        static Module: typeof Module;
        static register<Data = any>(
            specifier: string | URL,
            parentURL?: string | URL,
            options?: RegisterOptions<Data>,
        ): void;
        static register<Data = any>(specifier: string | URL, options?: RegisterOptions<Data>): void;
        constructor(id: string, parent?: Module);
    }
    global {
        interface ImportMeta {
            /**
             * The directory name of the current module. This is the same as the `path.dirname()` of the `import.meta.filename`.
             * **Caveat:** only present on `file:` modules.
             */
            dirname: string;
            /**
             * The full absolute path and filename of the current module, with symlinks resolved.
             * This is the same as the `url.fileURLToPath()` of the `import.meta.url`.
             * **Caveat:** only local modules support this property. Modules not using the `file:` protocol will not provide it.
             */
            filename: string;
            /**
             * The absolute `file:` URL of the module.
             */
            url: string;
            /**
             * Provides a module-relative resolution function scoped to each module, returning
             * the URL string.
             *
             * Second `parent` parameter is only used when the `--experimental-import-meta-resolve`
             * command flag enabled.
             *
             * @since v20.6.0
             *
             * @param specifier The module specifier to resolve relative to `parent`.
             * @param parent The absolute parent module URL to resolve from.
             * @returns The absolute (`file:`) URL string for the resolved module.
             */
            resolve(specifier: string, parent?: string | URL | undefined): string;
        }
    }
    export = Module;
}
declare module "node:module" {
    import module = require("module");
    export = module;
}
