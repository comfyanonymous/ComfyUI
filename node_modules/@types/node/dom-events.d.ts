export {}; // Don't export anything!

//// DOM-like Events
// NB: The Event / EventTarget / EventListener implementations below were copied
// from lib.dom.d.ts, then edited to reflect Node's documentation at
// https://nodejs.org/api/events.html#class-eventtarget.
// Please read that link to understand important implementation differences.

// This conditional type will be the existing global Event in a browser, or
// the copy below in a Node environment.
type __Event = typeof globalThis extends { onmessage: any; Event: any } ? {}
    : {
        /** This is not used in Node.js and is provided purely for completeness. */
        readonly bubbles: boolean;
        /** Alias for event.stopPropagation(). This is not used in Node.js and is provided purely for completeness. */
        cancelBubble: () => void;
        /** True if the event was created with the cancelable option */
        readonly cancelable: boolean;
        /** This is not used in Node.js and is provided purely for completeness. */
        readonly composed: boolean;
        /** Returns an array containing the current EventTarget as the only entry or empty if the event is not being dispatched. This is not used in Node.js and is provided purely for completeness. */
        composedPath(): [EventTarget?];
        /** Alias for event.target. */
        readonly currentTarget: EventTarget | null;
        /** Is true if cancelable is true and event.preventDefault() has been called. */
        readonly defaultPrevented: boolean;
        /** This is not used in Node.js and is provided purely for completeness. */
        readonly eventPhase: 0 | 2;
        /** The `AbortSignal` "abort" event is emitted with `isTrusted` set to `true`. The value is `false` in all other cases. */
        readonly isTrusted: boolean;
        /** Sets the `defaultPrevented` property to `true` if `cancelable` is `true`. */
        preventDefault(): void;
        /** This is not used in Node.js and is provided purely for completeness. */
        returnValue: boolean;
        /** Alias for event.target. */
        readonly srcElement: EventTarget | null;
        /** Stops the invocation of event listeners after the current one completes. */
        stopImmediatePropagation(): void;
        /** This is not used in Node.js and is provided purely for completeness. */
        stopPropagation(): void;
        /** The `EventTarget` dispatching the event */
        readonly target: EventTarget | null;
        /** The millisecond timestamp when the Event object was created. */
        readonly timeStamp: number;
        /** Returns the type of event, e.g. "click", "hashchange", or "submit". */
        readonly type: string;
    };

// See comment above explaining conditional type
type __EventTarget = typeof globalThis extends { onmessage: any; EventTarget: any } ? {}
    : {
        /**
         * Adds a new handler for the `type` event. Any given `listener` is added only once per `type` and per `capture` option value.
         *
         * If the `once` option is true, the `listener` is removed after the next time a `type` event is dispatched.
         *
         * The `capture` option is not used by Node.js in any functional way other than tracking registered event listeners per the `EventTarget` specification.
         * Specifically, the `capture` option is used as part of the key when registering a `listener`.
         * Any individual `listener` may be added once with `capture = false`, and once with `capture = true`.
         */
        addEventListener(
            type: string,
            listener: EventListener | EventListenerObject,
            options?: AddEventListenerOptions | boolean,
        ): void;
        /** Dispatches a synthetic event event to target and returns true if either event's cancelable attribute value is false or its preventDefault() method was not invoked, and false otherwise. */
        dispatchEvent(event: Event): boolean;
        /** Removes the event listener in target's event listener list with the same type, callback, and options. */
        removeEventListener(
            type: string,
            listener: EventListener | EventListenerObject,
            options?: EventListenerOptions | boolean,
        ): void;
    };

interface EventInit {
    bubbles?: boolean;
    cancelable?: boolean;
    composed?: boolean;
}

interface EventListenerOptions {
    /** Not directly used by Node.js. Added for API completeness. Default: `false`. */
    capture?: boolean;
}

interface AddEventListenerOptions extends EventListenerOptions {
    /** When `true`, the listener is automatically removed when it is first invoked. Default: `false`. */
    once?: boolean;
    /** When `true`, serves as a hint that the listener will not call the `Event` object's `preventDefault()` method. Default: false. */
    passive?: boolean;
    /** The listener will be removed when the given AbortSignal object's `abort()` method is called. */
    signal?: AbortSignal;
}

interface EventListener {
    (evt: Event): void;
}

interface EventListenerObject {
    handleEvent(object: Event): void;
}

import {} from "events"; // Make this an ambient declaration
declare global {
    /** An event which takes place in the DOM. */
    interface Event extends __Event {}
    var Event: typeof globalThis extends { onmessage: any; Event: infer T } ? T
        : {
            prototype: __Event;
            new(type: string, eventInitDict?: EventInit): __Event;
        };

    /**
     * EventTarget is a DOM interface implemented by objects that can
     * receive events and may have listeners for them.
     */
    interface EventTarget extends __EventTarget {}
    var EventTarget: typeof globalThis extends { onmessage: any; EventTarget: infer T } ? T
        : {
            prototype: __EventTarget;
            new(): __EventTarget;
        };
}
