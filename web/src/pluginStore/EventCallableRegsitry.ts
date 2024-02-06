import { Event } from './Event';

declare global {
    interface Window {
        debug?: boolean;
    }
}

export class EventCallableRegsitry {
    registry: Map<string, Array<(event: Event) => object>>;

    constructor() {
        this.registry = new Map();
    }

    addEventListener(name: string, callback: any) {
        const callbacks = this.registry.get(name);
        if (callbacks) {
            callbacks.push(callback);
        } else {
            this.registry.set(name, [callback]);
        }
    }

    removeEventListener(name: string, callback: any) {
        const callbacks = this.registry.get(name);
        if (callbacks) {
            const indexOf = callbacks.indexOf(callback);
            if (indexOf > -1) {
                callbacks.splice(indexOf, 1);
            }
        }
    }

    dispatchEvent(event: Event) {
        if (window['debug']) {
            console.log('DEBUG::', 'Event:', event.name);
        }
        const callbacks = this.registry.get(event.name);
        if (callbacks) {
            for (const callback of callbacks) {
                callback(event);
            }
        }
    }
}
