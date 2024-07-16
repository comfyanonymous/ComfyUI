/**
 * The `timer` module exposes a global API for scheduling functions to
 * be called at some future period of time. Because the timer functions are
 * globals, there is no need to call `require('node:timers')` to use the API.
 *
 * The timer functions within Node.js implement a similar API as the timers API
 * provided by Web Browsers but use a different internal implementation that is
 * built around the Node.js [Event Loop](https://nodejs.org/en/docs/guides/event-loop-timers-and-nexttick/#setimmediate-vs-settimeout).
 * @see [source](https://github.com/nodejs/node/blob/v20.12.2/lib/timers.js)
 */
declare module "timers" {
    import { Abortable } from "node:events";
    import {
        setImmediate as setImmediatePromise,
        setInterval as setIntervalPromise,
        setTimeout as setTimeoutPromise,
    } from "node:timers/promises";
    interface TimerOptions extends Abortable {
        /**
         * Set to `false` to indicate that the scheduled `Timeout`
         * should not require the Node.js event loop to remain active.
         * @default true
         */
        ref?: boolean | undefined;
    }
    let setTimeout: typeof global.setTimeout;
    let clearTimeout: typeof global.clearTimeout;
    let setInterval: typeof global.setInterval;
    let clearInterval: typeof global.clearInterval;
    let setImmediate: typeof global.setImmediate;
    let clearImmediate: typeof global.clearImmediate;
    global {
        namespace NodeJS {
            // compatibility with older typings
            interface Timer extends RefCounted {
                hasRef(): boolean;
                refresh(): this;
                [Symbol.toPrimitive](): number;
            }
            /**
             * This object is created internally and is returned from `setImmediate()`. It
             * can be passed to `clearImmediate()` in order to cancel the scheduled
             * actions.
             *
             * By default, when an immediate is scheduled, the Node.js event loop will continue
             * running as long as the immediate is active. The `Immediate` object returned by `setImmediate()` exports both `immediate.ref()` and `immediate.unref()` functions that can be used to
             * control this default behavior.
             */
            class Immediate implements RefCounted {
                /**
                 * When called, requests that the Node.js event loop _not_ exit so long as the `Immediate` is active. Calling `immediate.ref()` multiple times will have no
                 * effect.
                 *
                 * By default, all `Immediate` objects are "ref'ed", making it normally unnecessary
                 * to call `immediate.ref()` unless `immediate.unref()` had been called previously.
                 * @since v9.7.0
                 * @return a reference to `immediate`
                 */
                ref(): this;
                /**
                 * When called, the active `Immediate` object will not require the Node.js event
                 * loop to remain active. If there is no other activity keeping the event loop
                 * running, the process may exit before the `Immediate` object's callback is
                 * invoked. Calling `immediate.unref()` multiple times will have no effect.
                 * @since v9.7.0
                 * @return a reference to `immediate`
                 */
                unref(): this;
                /**
                 * If true, the `Immediate` object will keep the Node.js event loop active.
                 * @since v11.0.0
                 */
                hasRef(): boolean;
                _onImmediate: Function; // to distinguish it from the Timeout class
                /**
                 * Cancels the immediate. This is similar to calling `clearImmediate()`.
                 * @since v20.5.0
                 */
                [Symbol.dispose](): void;
            }
            /**
             * This object is created internally and is returned from `setTimeout()` and `setInterval()`. It can be passed to either `clearTimeout()` or `clearInterval()` in order to cancel the
             * scheduled actions.
             *
             * By default, when a timer is scheduled using either `setTimeout()` or `setInterval()`, the Node.js event loop will continue running as long as the
             * timer is active. Each of the `Timeout` objects returned by these functions
             * export both `timeout.ref()` and `timeout.unref()` functions that can be used to
             * control this default behavior.
             */
            class Timeout implements Timer {
                /**
                 * When called, requests that the Node.js event loop _not_ exit so long as the`Timeout` is active. Calling `timeout.ref()` multiple times will have no effect.
                 *
                 * By default, all `Timeout` objects are "ref'ed", making it normally unnecessary
                 * to call `timeout.ref()` unless `timeout.unref()` had been called previously.
                 * @since v0.9.1
                 * @return a reference to `timeout`
                 */
                ref(): this;
                /**
                 * When called, the active `Timeout` object will not require the Node.js event loop
                 * to remain active. If there is no other activity keeping the event loop running,
                 * the process may exit before the `Timeout` object's callback is invoked. Calling `timeout.unref()` multiple times will have no effect.
                 * @since v0.9.1
                 * @return a reference to `timeout`
                 */
                unref(): this;
                /**
                 * If true, the `Timeout` object will keep the Node.js event loop active.
                 * @since v11.0.0
                 */
                hasRef(): boolean;
                /**
                 * Sets the timer's start time to the current time, and reschedules the timer to
                 * call its callback at the previously specified duration adjusted to the current
                 * time. This is useful for refreshing a timer without allocating a new
                 * JavaScript object.
                 *
                 * Using this on a timer that has already called its callback will reactivate the
                 * timer.
                 * @since v10.2.0
                 * @return a reference to `timeout`
                 */
                refresh(): this;
                [Symbol.toPrimitive](): number;
                /**
                 * Cancels the timeout.
                 * @since v20.5.0
                 */
                [Symbol.dispose](): void;
            }
        }
        /**
         * Schedules execution of a one-time `callback` after `delay` milliseconds.
         *
         * The `callback` will likely not be invoked in precisely `delay` milliseconds.
         * Node.js makes no guarantees about the exact timing of when callbacks will fire,
         * nor of their ordering. The callback will be called as close as possible to the
         * time specified.
         *
         * When `delay` is larger than `2147483647` or less than `1`, the `delay` will be set to `1`. Non-integer delays are truncated to an integer.
         *
         * If `callback` is not a function, a `TypeError` will be thrown.
         *
         * This method has a custom variant for promises that is available using `timersPromises.setTimeout()`.
         * @since v0.0.1
         * @param callback The function to call when the timer elapses.
         * @param [delay=1] The number of milliseconds to wait before calling the `callback`.
         * @param args Optional arguments to pass when the `callback` is called.
         * @return for use with {@link clearTimeout}
         */
        function setTimeout<TArgs extends any[]>(
            callback: (...args: TArgs) => void,
            ms?: number,
            ...args: TArgs
        ): NodeJS.Timeout;
        // util.promisify no rest args compability
        // eslint-disable-next-line @typescript-eslint/no-invalid-void-type
        function setTimeout(callback: (args: void) => void, ms?: number): NodeJS.Timeout;
        namespace setTimeout {
            const __promisify__: typeof setTimeoutPromise;
        }
        /**
         * Cancels a `Timeout` object created by `setTimeout()`.
         * @since v0.0.1
         * @param timeout A `Timeout` object as returned by {@link setTimeout} or the `primitive` of the `Timeout` object as a string or a number.
         */
        function clearTimeout(timeoutId: NodeJS.Timeout | string | number | undefined): void;
        /**
         * Schedules repeated execution of `callback` every `delay` milliseconds.
         *
         * When `delay` is larger than `2147483647` or less than `1`, the `delay` will be
         * set to `1`. Non-integer delays are truncated to an integer.
         *
         * If `callback` is not a function, a `TypeError` will be thrown.
         *
         * This method has a custom variant for promises that is available using `timersPromises.setInterval()`.
         * @since v0.0.1
         * @param callback The function to call when the timer elapses.
         * @param [delay=1] The number of milliseconds to wait before calling the `callback`.
         * @param args Optional arguments to pass when the `callback` is called.
         * @return for use with {@link clearInterval}
         */
        function setInterval<TArgs extends any[]>(
            callback: (...args: TArgs) => void,
            ms?: number,
            ...args: TArgs
        ): NodeJS.Timeout;
        // util.promisify no rest args compability
        // eslint-disable-next-line @typescript-eslint/no-invalid-void-type
        function setInterval(callback: (args: void) => void, ms?: number): NodeJS.Timeout;
        namespace setInterval {
            const __promisify__: typeof setIntervalPromise;
        }
        /**
         * Cancels a `Timeout` object created by `setInterval()`.
         * @since v0.0.1
         * @param timeout A `Timeout` object as returned by {@link setInterval} or the `primitive` of the `Timeout` object as a string or a number.
         */
        function clearInterval(intervalId: NodeJS.Timeout | string | number | undefined): void;
        /**
         * Schedules the "immediate" execution of the `callback` after I/O events'
         * callbacks.
         *
         * When multiple calls to `setImmediate()` are made, the `callback` functions are
         * queued for execution in the order in which they are created. The entire callback
         * queue is processed every event loop iteration. If an immediate timer is queued
         * from inside an executing callback, that timer will not be triggered until the
         * next event loop iteration.
         *
         * If `callback` is not a function, a `TypeError` will be thrown.
         *
         * This method has a custom variant for promises that is available using `timersPromises.setImmediate()`.
         * @since v0.9.1
         * @param callback The function to call at the end of this turn of the Node.js `Event Loop`
         * @param args Optional arguments to pass when the `callback` is called.
         * @return for use with {@link clearImmediate}
         */
        function setImmediate<TArgs extends any[]>(
            callback: (...args: TArgs) => void,
            ...args: TArgs
        ): NodeJS.Immediate;
        // util.promisify no rest args compability
        // eslint-disable-next-line @typescript-eslint/no-invalid-void-type
        function setImmediate(callback: (args: void) => void): NodeJS.Immediate;
        namespace setImmediate {
            const __promisify__: typeof setImmediatePromise;
        }
        /**
         * Cancels an `Immediate` object created by `setImmediate()`.
         * @since v0.9.1
         * @param immediate An `Immediate` object as returned by {@link setImmediate}.
         */
        function clearImmediate(immediateId: NodeJS.Immediate | undefined): void;
        function queueMicrotask(callback: () => void): void;
    }
}
declare module "node:timers" {
    export * from "timers";
}
