export function getResolver(timeout = 5000) {
    const resolver = {};
    resolver.id = generateId(8);
    resolver.completed = false;
    resolver.resolved = false;
    resolver.rejected = false;
    resolver.promise = new Promise((resolve, reject) => {
        resolver.reject = (e) => {
            resolver.completed = true;
            resolver.rejected = true;
            reject(e);
        };
        resolver.resolve = (data) => {
            resolver.completed = true;
            resolver.resolved = true;
            resolve(data);
        };
    });
    resolver.timeout = setTimeout(() => {
        if (!resolver.completed) {
            resolver.reject();
        }
    }, timeout);
    return resolver;
}
const DEBOUNCE_FN_TO_PROMISE = new WeakMap();
export function debounce(fn, ms = 64) {
    if (!DEBOUNCE_FN_TO_PROMISE.get(fn)) {
        DEBOUNCE_FN_TO_PROMISE.set(fn, wait(ms).then(() => {
            DEBOUNCE_FN_TO_PROMISE.delete(fn);
            fn();
        }));
    }
    return DEBOUNCE_FN_TO_PROMISE.get(fn);
}
export function check(value, msg = "", ...args) {
    if (!value) {
        console.error(msg, ...(args || []));
        throw new Error(msg || "Error");
    }
}
export function wait(ms = 16) {
    if (ms === 16) {
        return new Promise((resolve) => {
            requestAnimationFrame(() => {
                resolve();
            });
        });
    }
    return new Promise((resolve) => {
        setTimeout(() => {
            resolve();
        }, ms);
    });
}
export function deepFreeze(obj) {
    const propNames = Reflect.ownKeys(obj);
    for (const name of propNames) {
        const value = obj[name];
        if ((value && typeof value === "object") || typeof value === "function") {
            deepFreeze(value);
        }
    }
    return Object.freeze(obj);
}
function dec2hex(dec) {
    return dec.toString(16).padStart(2, "0");
}
export function generateId(length) {
    const arr = new Uint8Array(length / 2);
    crypto.getRandomValues(arr);
    return Array.from(arr, dec2hex).join("");
}
export function getObjectValue(obj, objKey, def) {
    if (!obj || !objKey)
        return def;
    const keys = objKey.split(".");
    const key = keys.shift();
    const found = obj[key];
    if (keys.length) {
        return getObjectValue(found, keys.join("."), def);
    }
    return found;
}
export function setObjectValue(obj, objKey, value, createMissingObjects = true) {
    if (!obj || !objKey)
        return obj;
    const keys = objKey.split(".");
    const key = keys.shift();
    if (obj[key] === undefined) {
        if (!createMissingObjects) {
            return;
        }
        obj[key] = {};
    }
    if (!keys.length) {
        obj[key] = value;
    }
    else {
        if (typeof obj[key] != "object") {
            obj[key] = {};
        }
        setObjectValue(obj[key], keys.join("."), value, createMissingObjects);
    }
    return obj;
}
export function moveArrayItem(arr, itemOrFrom, to) {
    const from = typeof itemOrFrom === "number" ? itemOrFrom : arr.indexOf(itemOrFrom);
    arr.splice(to, 0, arr.splice(from, 1)[0]);
}
export function removeArrayItem(arr, itemOrIndex) {
    const index = typeof itemOrIndex === "number" ? itemOrIndex : arr.indexOf(itemOrIndex);
    arr.splice(index, 1);
}
export function injectCss(href) {
    if (document.querySelector(`link[href^="${href}"]`)) {
        return Promise.resolve();
    }
    return new Promise((resolve) => {
        const link = document.createElement("link");
        link.setAttribute("rel", "stylesheet");
        link.setAttribute("type", "text/css");
        const timeout = setTimeout(resolve, 1000);
        link.addEventListener("load", (e) => {
            clearInterval(timeout);
            resolve();
        });
        link.href = href;
        document.head.appendChild(link);
    });
}
export function defineProperty(instance, property, desc) {
    var _a, _b, _c, _d, _e, _f;
    const existingDesc = Object.getOwnPropertyDescriptor(instance, property);
    if ((existingDesc === null || existingDesc === void 0 ? void 0 : existingDesc.configurable) === false) {
        throw new Error(`Error: rgthree-comfy cannot define un-configurable property "${property}"`);
    }
    if ((existingDesc === null || existingDesc === void 0 ? void 0 : existingDesc.get) && desc.get) {
        const descGet = desc.get;
        desc.get = () => {
            existingDesc.get.apply(instance, []);
            return descGet.apply(instance, []);
        };
    }
    if ((existingDesc === null || existingDesc === void 0 ? void 0 : existingDesc.set) && desc.set) {
        const descSet = desc.set;
        desc.set = (v) => {
            existingDesc.set.apply(instance, [v]);
            return descSet.apply(instance, [v]);
        };
    }
    desc.enumerable = (_b = (_a = desc.enumerable) !== null && _a !== void 0 ? _a : existingDesc === null || existingDesc === void 0 ? void 0 : existingDesc.enumerable) !== null && _b !== void 0 ? _b : true;
    desc.configurable = (_d = (_c = desc.configurable) !== null && _c !== void 0 ? _c : existingDesc === null || existingDesc === void 0 ? void 0 : existingDesc.configurable) !== null && _d !== void 0 ? _d : true;
    if (!desc.get && !desc.set) {
        desc.writable = (_f = (_e = desc.writable) !== null && _e !== void 0 ? _e : existingDesc === null || existingDesc === void 0 ? void 0 : existingDesc.writable) !== null && _f !== void 0 ? _f : true;
    }
    return Object.defineProperty(instance, property, desc);
}
export function areDataViewsEqual(a, b) {
    if (a.byteLength !== b.byteLength) {
        return false;
    }
    for (let i = 0; i < a.byteLength; i++) {
        if (a.getUint8(i) !== b.getUint8(i)) {
            return false;
        }
    }
    return true;
}
function looksLikeBase64(source) {
    return source.length > 500 || source.startsWith("data:");
}
export function areArrayBuffersEqual(a, b) {
    if (a == b || !a || !b) {
        return a == b;
    }
    return areDataViewsEqual(new DataView(a), new DataView(b));
}
export function getCanvasImageData(image) {
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");
    canvas.width = image.width;
    canvas.height = image.height;
    ctx.drawImage(image, 0, 0);
    const imageData = ctx.getImageData(0, 0, image.width, image.height);
    return [canvas, ctx, imageData];
}
export async function convertToBase64(source) {
    if (source instanceof Promise) {
        source = await source;
    }
    if (typeof source === "string" && looksLikeBase64(source)) {
        return source;
    }
    if (typeof source === "string" || source instanceof Blob || source instanceof ArrayBuffer) {
        return convertToBase64(await loadImage(source));
    }
    if (source instanceof HTMLImageElement) {
        const [canvas, ctx, imageData] = getCanvasImageData(source);
        return convertToBase64(canvas);
    }
    if (source instanceof HTMLCanvasElement) {
        return source.toDataURL("image/png");
    }
    throw Error("Unknown source to convert to base64.");
}
export async function convertToArrayBuffer(source) {
    if (source instanceof Promise) {
        source = await source;
    }
    if (source instanceof ArrayBuffer) {
        return source;
    }
    if (typeof source === "string") {
        if (looksLikeBase64(source)) {
            var binaryString = atob(source.replace(/^.*?;base64,/, ""));
            var bytes = new Uint8Array(binaryString.length);
            for (var i = 0; i < binaryString.length; i++) {
                bytes[i] = binaryString.charCodeAt(i);
            }
            return bytes.buffer;
        }
        return convertToArrayBuffer(await loadImage(source));
    }
    if (source instanceof HTMLImageElement) {
        const [canvas, ctx, imageData] = getCanvasImageData(source);
        return convertToArrayBuffer(canvas);
    }
    if (source instanceof HTMLCanvasElement) {
        return convertToArrayBuffer(source.toDataURL());
    }
    if (source instanceof Blob) {
        return source.arrayBuffer();
    }
    throw Error("Unknown source to convert to arraybuffer.");
}
export async function loadImage(source) {
    if (source instanceof Promise) {
        source = await source;
    }
    if (source instanceof HTMLImageElement) {
        return loadImage(source.src);
    }
    if (source instanceof Blob) {
        return loadImage(source.arrayBuffer());
    }
    if (source instanceof HTMLCanvasElement) {
        return loadImage(source.toDataURL());
    }
    if (source instanceof ArrayBuffer) {
        var binary = "";
        var bytes = new Uint8Array(source);
        var len = bytes.byteLength;
        for (var i = 0; i < len; i++) {
            binary += String.fromCharCode(bytes[i]);
        }
        return loadImage(`data:${getMimeTypeFromArrayBuffer(bytes)};base64,${btoa(binary)}`);
    }
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.addEventListener("load", () => {
            resolve(img);
        });
        img.addEventListener("error", () => {
            reject(img);
        });
        img.src = source;
    });
}
function getMimeTypeFromArrayBuffer(buffer) {
    const len = 4;
    if (buffer.length >= len) {
        let signatureArr = new Array(len);
        for (let i = 0; i < len; i++)
            signatureArr[i] = buffer[i].toString(16);
        const signature = signatureArr.join("").toUpperCase();
        switch (signature) {
            case "89504E47":
                return "image/png";
            case "47494638":
                return "image/gif";
            case "25504446":
                return "application/pdf";
            case "FFD8FFDB":
            case "FFD8FFE0":
                return "image/jpeg";
            case "504B0304":
                return "application/zip";
            default:
                return null;
        }
    }
    return null;
}
export class Broadcaster extends EventTarget {
    constructor(channelName) {
        super();
        this.queue = {};
        this.queue = {};
        this.channel = new BroadcastChannel(channelName);
        this.channel.addEventListener("message", (e) => {
            this.onMessage(e);
        });
    }
    getId() {
        let id;
        do {
            id = generateId(6);
        } while (this.queue[id]);
        return id;
    }
    async broadcastAndWait(action, payload, options) {
        const id = this.getId();
        this.queue[id] = getResolver(options === null || options === void 0 ? void 0 : options.timeout);
        this.channel.postMessage({
            id,
            action,
            payload,
        });
        let response;
        try {
            response = await this.queue[id].promise;
        }
        catch (e) {
            console.log("CAUGHT", e);
            response = [];
        }
        return response;
    }
    broadcast(action, payload) {
        this.channel.postMessage({
            id: this.getId(),
            action,
            payload,
        });
    }
    reply(replyId, action, payload) {
        this.channel.postMessage({
            id: this.getId(),
            replyId,
            action,
            payload,
        });
    }
    openWindowAndWaitForMessage(rgthreePath, windowName) {
        const id = this.getId();
        this.queue[id] = getResolver();
        const win = window.open(`/rgthree/${rgthreePath}#broadcastLoadMsgId=${id}`, windowName);
        return { window: win, promise: this.queue[id].promise };
    }
    onMessage(e) {
        var _a, _b;
        const msgId = ((_a = e.data) === null || _a === void 0 ? void 0 : _a.replyId) || "";
        const queueItem = this.queue[msgId];
        if (queueItem) {
            if (queueItem.completed) {
                console.error(`${msgId} already completed..`);
            }
            queueItem.deferment = queueItem.deferment || { data: [] };
            queueItem.deferment.data.push(e.data.payload);
            queueItem.deferment.timeout && clearTimeout(queueItem.deferment.timeout);
            queueItem.deferment.timeout = setTimeout(() => {
                queueItem.resolve(queueItem.deferment.data);
            }, 250);
        }
        else {
            this.dispatchEvent(new CustomEvent("rgthree-broadcast-message", {
                detail: Object.assign({ replyTo: (_b = e.data) === null || _b === void 0 ? void 0 : _b.id }, e.data),
            }));
        }
    }
    addMessageListener(callback, options) {
        return super.addEventListener("rgthree-broadcast-message", callback, options);
    }
}
const broadcastChannelMap = new Map();
export function broadcastOnChannel(channel, action, payload) {
    let queue = broadcastChannelMap.get(channel);
    if (!queue) {
        broadcastChannelMap.set(channel, {});
        queue = broadcastChannelMap.get(channel);
    }
    let id;
    do {
        id = generateId(6);
    } while (queue[id]);
    queue[id] = getResolver();
    channel.postMessage({
        id,
        action,
        payload,
    });
    return queue[id].promise;
}
