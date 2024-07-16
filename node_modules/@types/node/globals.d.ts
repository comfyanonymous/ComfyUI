export {}; // Make this a module

// #region Fetch and friends
// Conditional type aliases, used at the end of this file.
// Will either be empty if lib-dom is included, or the undici version otherwise.
type _Request = typeof globalThis extends { onmessage: any } ? {} : import("undici-types").Request;
type _Response = typeof globalThis extends { onmessage: any } ? {} : import("undici-types").Response;
type _FormData = typeof globalThis extends { onmessage: any } ? {} : import("undici-types").FormData;
type _Headers = typeof globalThis extends { onmessage: any } ? {} : import("undici-types").Headers;
type _RequestInit = typeof globalThis extends { onmessage: any } ? {}
    : import("undici-types").RequestInit;
type _ResponseInit = typeof globalThis extends { onmessage: any } ? {}
    : import("undici-types").ResponseInit;
type _File = typeof globalThis extends { onmessage: any } ? {} : import("node:buffer").File;
// #endregion Fetch and friends

declare global {
    // Declare "static" methods in Error
    interface ErrorConstructor {
        /** Create .stack property on a target object */
        captureStackTrace(targetObject: object, constructorOpt?: Function): void;

        /**
         * Optional override for formatting stack traces
         *
         * @see https://v8.dev/docs/stack-trace-api#customizing-stack-traces
         */
        prepareStackTrace?: ((err: Error, stackTraces: NodeJS.CallSite[]) => any) | undefined;

        stackTraceLimit: number;
    }

    /*-----------------------------------------------*
    *                                               *
    *                   GLOBAL                      *
    *                                               *
    ------------------------------------------------*/

    // For backwards compability
    interface NodeRequire extends NodeJS.Require {}
    interface RequireResolve extends NodeJS.RequireResolve {}
    interface NodeModule extends NodeJS.Module {}

    var process: NodeJS.Process;
    var console: Console;

    var __filename: string;
    var __dirname: string;

    var require: NodeRequire;
    var module: NodeModule;

    // Same as module.exports
    var exports: any;

    /**
     * Only available if `--expose-gc` is passed to the process.
     */
    var gc: undefined | (() => void);

    // #region borrowed
    // from https://github.com/microsoft/TypeScript/blob/38da7c600c83e7b31193a62495239a0fe478cb67/lib/lib.webworker.d.ts#L633 until moved to separate lib
    /** A controller object that allows you to abort one or more DOM requests as and when desired. */
    interface AbortController {
        /**
         * Returns the AbortSignal object associated with this object.
         */

        readonly signal: AbortSignal;
        /**
         * Invoking this method will set this object's AbortSignal's aborted flag and signal to any observers that the associated activity is to be aborted.
         */
        abort(reason?: any): void;
    }

    /** A signal object that allows you to communicate with a DOM request (such as a Fetch) and abort it if required via an AbortController object. */
    interface AbortSignal extends EventTarget {
        /**
         * Returns true if this AbortSignal's AbortController has signaled to abort, and false otherwise.
         */
        readonly aborted: boolean;
        readonly reason: any;
        onabort: null | ((this: AbortSignal, event: Event) => any);
        throwIfAborted(): void;
    }

    var AbortController: typeof globalThis extends { onmessage: any; AbortController: infer T } ? T
        : {
            prototype: AbortController;
            new(): AbortController;
        };

    var AbortSignal: typeof globalThis extends { onmessage: any; AbortSignal: infer T } ? T
        : {
            prototype: AbortSignal;
            new(): AbortSignal;
            abort(reason?: any): AbortSignal;
            timeout(milliseconds: number): AbortSignal;
        };
    // #endregion borrowed

    // #region Disposable
    interface SymbolConstructor {
        /**
         * A method that is used to release resources held by an object. Called by the semantics of the `using` statement.
         */
        readonly dispose: unique symbol;

        /**
         * A method that is used to asynchronously release resources held by an object. Called by the semantics of the `await using` statement.
         */
        readonly asyncDispose: unique symbol;
    }

    interface Disposable {
        [Symbol.dispose](): void;
    }

    interface AsyncDisposable {
        [Symbol.asyncDispose](): PromiseLike<void>;
    }
    // #endregion Disposable

    // #region ArrayLike.at()
    interface RelativeIndexable<T> {
        /**
         * Takes an integer value and returns the item at that index,
         * allowing for positive and negative integers.
         * Negative integers count back from the last item in the array.
         */
        at(index: number): T | undefined;
    }
    interface String extends RelativeIndexable<string> {}
    interface Array<T> extends RelativeIndexable<T> {}
    interface ReadonlyArray<T> extends RelativeIndexable<T> {}
    interface Int8Array extends RelativeIndexable<number> {}
    interface Uint8Array extends RelativeIndexable<number> {}
    interface Uint8ClampedArray extends RelativeIndexable<number> {}
    interface Int16Array extends RelativeIndexable<number> {}
    interface Uint16Array extends RelativeIndexable<number> {}
    interface Int32Array extends RelativeIndexable<number> {}
    interface Uint32Array extends RelativeIndexable<number> {}
    interface Float32Array extends RelativeIndexable<number> {}
    interface Float64Array extends RelativeIndexable<number> {}
    interface BigInt64Array extends RelativeIndexable<bigint> {}
    interface BigUint64Array extends RelativeIndexable<bigint> {}
    // #endregion ArrayLike.at() end

    /**
     * @since v17.0.0
     *
     * Creates a deep clone of an object.
     */
    function structuredClone<T>(
        value: T,
        transfer?: { transfer: ReadonlyArray<import("worker_threads").TransferListItem> },
    ): T;

    /*----------------------------------------------*
    *                                               *
    *               GLOBAL INTERFACES               *
    *                                               *
    *-----------------------------------------------*/
    namespace NodeJS {
        interface CallSite {
            /**
             * Value of "this"
             */
            getThis(): unknown;

            /**
             * Type of "this" as a string.
             * This is the name of the function stored in the constructor field of
             * "this", if available.  Otherwise the object's [[Class]] internal
             * property.
             */
            getTypeName(): string | null;

            /**
             * Current function
             */
            getFunction(): Function | undefined;

            /**
             * Name of the current function, typically its name property.
             * If a name property is not available an attempt will be made to try
             * to infer a name from the function's context.
             */
            getFunctionName(): string | null;

            /**
             * Name of the property [of "this" or one of its prototypes] that holds
             * the current function
             */
            getMethodName(): string | null;

            /**
             * Name of the script [if this function was defined in a script]
             */
            getFileName(): string | undefined;

            /**
             * Current line number [if this function was defined in a script]
             */
            getLineNumber(): number | null;

            /**
             * Current column number [if this function was defined in a script]
             */
            getColumnNumber(): number | null;

            /**
             * A call site object representing the location where eval was called
             * [if this function was created using a call to eval]
             */
            getEvalOrigin(): string | undefined;

            /**
             * Is this a toplevel invocation, that is, is "this" the global object?
             */
            isToplevel(): boolean;

            /**
             * Does this call take place in code defined by a call to eval?
             */
            isEval(): boolean;

            /**
             * Is this call in native V8 code?
             */
            isNative(): boolean;

            /**
             * Is this a constructor call?
             */
            isConstructor(): boolean;

            /**
             * is this an async call (i.e. await, Promise.all(), or Promise.any())?
             */
            isAsync(): boolean;

            /**
             * is this an async call to Promise.all()?
             */
            isPromiseAll(): boolean;

            /**
             * returns the index of the promise element that was followed in
             * Promise.all() or Promise.any() for async stack traces, or null
             * if the CallSite is not an async
             */
            getPromiseIndex(): number | null;

            getScriptNameOrSourceURL(): string;
            getScriptHash(): string;

            getEnclosingColumnNumber(): number;
            getEnclosingLineNumber(): number;
            getPosition(): number;

            toString(): string;
        }

        interface ErrnoException extends Error {
            errno?: number | undefined;
            code?: string | undefined;
            path?: string | undefined;
            syscall?: string | undefined;
        }

        interface ReadableStream extends EventEmitter {
            readable: boolean;
            read(size?: number): string | Buffer;
            setEncoding(encoding: BufferEncoding): this;
            pause(): this;
            resume(): this;
            isPaused(): boolean;
            pipe<T extends WritableStream>(destination: T, options?: { end?: boolean | undefined }): T;
            unpipe(destination?: WritableStream): this;
            unshift(chunk: string | Uint8Array, encoding?: BufferEncoding): void;
            wrap(oldStream: ReadableStream): this;
            [Symbol.asyncIterator](): AsyncIterableIterator<string | Buffer>;
        }

        interface WritableStream extends EventEmitter {
            writable: boolean;
            write(buffer: Uint8Array | string, cb?: (err?: Error | null) => void): boolean;
            write(str: string, encoding?: BufferEncoding, cb?: (err?: Error | null) => void): boolean;
            end(cb?: () => void): this;
            end(data: string | Uint8Array, cb?: () => void): this;
            end(str: string, encoding?: BufferEncoding, cb?: () => void): this;
        }

        interface ReadWriteStream extends ReadableStream, WritableStream {}

        interface RefCounted {
            ref(): this;
            unref(): this;
        }

        type TypedArray =
            | Uint8Array
            | Uint8ClampedArray
            | Uint16Array
            | Uint32Array
            | Int8Array
            | Int16Array
            | Int32Array
            | BigUint64Array
            | BigInt64Array
            | Float32Array
            | Float64Array;
        type ArrayBufferView = TypedArray | DataView;

        interface Require {
            (id: string): any;
            resolve: RequireResolve;
            cache: Dict<NodeModule>;
            /**
             * @deprecated
             */
            extensions: RequireExtensions;
            main: Module | undefined;
        }

        interface RequireResolve {
            (id: string, options?: { paths?: string[] | undefined }): string;
            paths(request: string): string[] | null;
        }

        interface RequireExtensions extends Dict<(m: Module, filename: string) => any> {
            ".js": (m: Module, filename: string) => any;
            ".json": (m: Module, filename: string) => any;
            ".node": (m: Module, filename: string) => any;
        }
        interface Module {
            /**
             * `true` if the module is running during the Node.js preload
             */
            isPreloading: boolean;
            exports: any;
            require: Require;
            id: string;
            filename: string;
            loaded: boolean;
            /** @deprecated since v14.6.0 Please use `require.main` and `module.children` instead. */
            parent: Module | null | undefined;
            children: Module[];
            /**
             * @since v11.14.0
             *
             * The directory name of the module. This is usually the same as the path.dirname() of the module.id.
             */
            path: string;
            paths: string[];
        }

        interface Dict<T> {
            [key: string]: T | undefined;
        }

        interface ReadOnlyDict<T> {
            readonly [key: string]: T | undefined;
        }
    }

    interface RequestInit extends _RequestInit {}

    function fetch(
        input: string | URL | globalThis.Request,
        init?: RequestInit,
    ): Promise<Response>;

    interface Request extends _Request {}
    var Request: typeof globalThis extends {
        onmessage: any;
        Request: infer T;
    } ? T
        : typeof import("undici-types").Request;

    interface ResponseInit extends _ResponseInit {}

    interface Response extends _Response {}
    var Response: typeof globalThis extends {
        onmessage: any;
        Response: infer T;
    } ? T
        : typeof import("undici-types").Response;

    interface FormData extends _FormData {}
    var FormData: typeof globalThis extends {
        onmessage: any;
        FormData: infer T;
    } ? T
        : typeof import("undici-types").FormData;

    interface Headers extends _Headers {}
    var Headers: typeof globalThis extends {
        onmessage: any;
        Headers: infer T;
    } ? T
        : typeof import("undici-types").Headers;

    interface File extends _File {}
    var File: typeof globalThis extends {
        onmessage: any;
        File: infer T;
    } ? T
        : typeof import("node:buffer").File;
}
