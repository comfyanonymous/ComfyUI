/**
 * @fileoverview
 * A bunch of shared utils that can be used in ComfyUI, as well as in any single-HTML pages.
 */

export type Resolver<T> = {
  id: string;
  completed: boolean;
  resolved: boolean;
  rejected: boolean;
  promise: Promise<T>;
  resolve: (data: T) => void;
  reject: (e?: Error) => void;
  timeout: number | null;
  deferment?: {data?: any; timeout?: number | null; signal?: string};
};

/**
 * Returns a new `Resolver` type that allows creating a "disconnected" `Promise` that can be
 * returned and resolved separately.
 */
export function getResolver<T>(timeout: number = 5000): Resolver<T> {
  const resolver: Partial<Resolver<T>> = {};
  resolver.id = generateId(8);
  resolver.completed = false;
  resolver.resolved = false;
  resolver.rejected = false;
  resolver.promise = new Promise((resolve, reject) => {
    resolver.reject = (e?: Error) => {
      resolver.completed = true;
      resolver.rejected = true;
      reject(e);
    };
    resolver.resolve = (data: T) => {
      resolver.completed = true;
      resolver.resolved = true;
      resolve(data);
    };
  });
  resolver.timeout = setTimeout(() => {
    if (!resolver.completed) {
      resolver.reject!();
    }
  }, timeout);
  return resolver as Resolver<T>;
}

/** The WeakMap for debounced functions. */
const DEBOUNCE_FN_TO_PROMISE: WeakMap<Function, Promise<void>> = new WeakMap();

/**
 * Debounces a function call so it is only called once in the initially provided ms even if asked
 * to be called multiple times within that period.
 */
export function debounce(fn: Function, ms = 64) {
  if (!DEBOUNCE_FN_TO_PROMISE.get(fn)) {
    DEBOUNCE_FN_TO_PROMISE.set(
      fn,
      wait(ms).then(() => {
        DEBOUNCE_FN_TO_PROMISE.delete(fn);
        fn();
      }),
    );
  }
  return DEBOUNCE_FN_TO_PROMISE.get(fn);
}

/** Checks that a value is not falsy. */
export function check(value: any, msg = "", ...args: any[]): asserts value {
  if (!value) {
    console.error(msg, ...(args || []));
    throw new Error(msg || "Error");
  }
}

/** Waits a certain number of ms, as a `Promise.` */
export function wait(ms = 16): Promise<void> {
  // Special logic, if we're waiting 16ms, then trigger on next frame.
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

/** Deeply freezes the passed in object. */
export function deepFreeze<T extends Object>(obj: T): T {
  // Retrieve the property names defined on object
  const propNames = Reflect.ownKeys(obj);

  // Freeze properties before freezing self
  for (const name of propNames) {
    const value = (obj as any)[name];
    if ((value && typeof value === "object") || typeof value === "function") {
      deepFreeze(value);
    }
  }
  return Object.freeze(obj);
}

function dec2hex(dec: number) {
  return dec.toString(16).padStart(2, "0");
}

/** Generates an unique id of a specific length. */
export function generateId(length: number) {
  const arr = new Uint8Array(length / 2);
  crypto.getRandomValues(arr);
  return Array.from(arr, dec2hex).join("");
}

/**
 * Returns the deep value of an object given a dot-delimited key.
 */
export function getObjectValue(obj: {[key: string]: any}, objKey: string, def?: any) {
  if (!obj || !objKey) return def;

  const keys = objKey.split(".");
  const key = keys.shift()!;
  const found = obj[key];
  if (keys.length) {
    return getObjectValue(found, keys.join("."), def);
  }
  return found;
}

/**
 * Sets the deep value of an object given a dot-delimited key.
 *
 * By default, missing objects will be created while settng the path.  If `createMissingObjects` is
 * set to false, then the setting will be abandoned if the key path is missing an intermediate
 * value. For example:
 *
 *   setObjectValue({a: {z: false}}, 'a.b.c', true); // {a: {z: false, b: {c: true } } }
 *   setObjectValue({a: {z: false}}, 'a.b.c', true, false); // {a: {z: false}}
 *
 */
export function setObjectValue(obj: any, objKey: string, value: any, createMissingObjects = true) {
  if (!obj || !objKey) return obj;

  const keys = objKey.split(".");
  const key = keys.shift()!;
  if (obj[key] === undefined) {
    if (!createMissingObjects) {
      return;
    }
    obj[key] = {};
  }
  if (!keys.length) {
    obj[key] = value;
  } else {
    if (typeof obj[key] != "object") {
      obj[key] = {};
    }
    setObjectValue(obj[key], keys.join("."), value, createMissingObjects);
  }
  return obj;
}

/**
 * Moves an item in an array (by item or its index) to another index.
 */
export function moveArrayItem(arr: any[], itemOrFrom: any, to: number) {
  const from = typeof itemOrFrom === "number" ? itemOrFrom : arr.indexOf(itemOrFrom);
  arr.splice(to, 0, arr.splice(from, 1)[0]!);
}

/**
 * Moves an item in an array (by item or its index) to another index.
 */
export function removeArrayItem<T>(arr: T[], itemOrIndex: T | number) {
  const index = typeof itemOrIndex === "number" ? itemOrIndex : arr.indexOf(itemOrIndex);
  arr.splice(index, 1);
}

/**
 * Injects CSS into the page with a promise when complete.
 */
export function injectCss(href: string): Promise<void> {
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

/**
 * Calls `Object.defineProperty` with special care around getters and setters to call out to a
 * parent getter or setter (like a super.set call)   to ensure any side effects up the chain
 * are still invoked.
 */
export function defineProperty(instance: any, property: string, desc: PropertyDescriptor) {
  const existingDesc = Object.getOwnPropertyDescriptor(instance, property);
  if (existingDesc?.configurable === false) {
    throw new Error(`Error: rgthree-comfy cannot define un-configurable property "${property}"`);
  }

  if (existingDesc?.get && desc.get) {
    const descGet = desc.get;
    desc.get = () => {
      existingDesc.get!.apply(instance, []);
      return descGet!.apply(instance, []);
    };
  }
  if (existingDesc?.set && desc.set) {
    const descSet = desc.set;
    desc.set = (v: any) => {
      existingDesc.set!.apply(instance, [v]);
      return descSet!.apply(instance, [v]);
    };
  }

  desc.enumerable = desc.enumerable ?? existingDesc?.enumerable ?? true;
  desc.configurable = desc.configurable ?? existingDesc?.configurable ?? true;
  if (!desc.get && !desc.set) {
    desc.writable = desc.writable ?? existingDesc?.writable ?? true;
  }
  return Object.defineProperty(instance, property, desc);
}

/**
 * Determines if two DataViews are equal.
 */
export function areDataViewsEqual(a: DataView, b: DataView) {
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

/**
 * A cheap check if the source looks like base64.
 */
function looksLikeBase64(source: string) {
  return source.length > 500 || source.startsWith("data:");
}

/**
 * Determines if two ArrayBuffers are equal.
 */
export function areArrayBuffersEqual(a?: ArrayBuffer | null, b?: ArrayBuffer | null) {
  if (a == b || !a || !b) {
    return a == b;
  }
  return areDataViewsEqual(new DataView(a), new DataView(b));
}

/**
 * Returns canvas image data for an HTML Image.
 */
export function getCanvasImageData(
  image: HTMLImageElement,
): [HTMLCanvasElement, CanvasRenderingContext2D, ImageData] {
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d")!;
  canvas.width = image.width;
  canvas.height = image.height;
  ctx.drawImage(image, 0, 0);
  const imageData = ctx.getImageData(0, 0, image.width, image.height);
  return [canvas, ctx, imageData];
}

/** Union of types for image conversion. */
type ImageConverstionTypes = string | Blob | ArrayBuffer | HTMLImageElement | HTMLCanvasElement;

/**
 * Converts an ImageConverstionTypes to a base64 string.
 */
export async function convertToBase64(
  source: ImageConverstionTypes | Promise<ImageConverstionTypes>,
): Promise<string> {
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

/**
 * Converts an ImageConverstionTypes to an image array buffer.
 */
export async function convertToArrayBuffer(
  source: ImageConverstionTypes | Promise<ImageConverstionTypes>,
): Promise<ArrayBuffer> {
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

/**
 * Loads an image into an HTMLImageElement.
 */
export async function loadImage(
  source: ImageConverstionTypes | Promise<ImageConverstionTypes>,
): Promise<HTMLImageElement> {
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
      binary += String.fromCharCode(bytes[i]!);
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

/**
 * Determines the mime type from an array buffer.
 */
function getMimeTypeFromArrayBuffer(buffer: Uint8Array) {
  const len = 4;
  if (buffer.length >= len) {
    let signatureArr = new Array(len);
    for (let i = 0; i < len; i++) signatureArr[i] = buffer[i]!.toString(16);
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

type BroadcasterMessage<T extends {}> = {
  id: string;
  replyId?: string;
  action: string;
  window: Window;
  port: MessagePort;
  payload?: T;
};

type BroadcasterMessageOptions = {
  timeout?: number;
  listenForReply?: boolean;
};

/**
 * A Broadcaster is a wrapper around a BroadcastChannel for communication with other windows.
 */
export class Broadcaster<OutPayload extends {}, InPayload extends {}> extends EventTarget {
  private channel: BroadcastChannel;
  private queue: {[key: string]: Resolver<InPayload[]>} = {};

  constructor(channelName: string) {
    super();
    this.queue = {};
    this.channel = new BroadcastChannel(channelName);
    this.channel.addEventListener("message", (e) => {
      this.onMessage(e);
    });
  }

  /**
   * Returns a unique id within the queue.
   */
  private getId() {
    let id: string;
    do {
      id = generateId(6);
    } while (this.queue[id]);
    return id;
  }

  /**
   * Broadcasts an action, and waits for a response, with a timeout before cancelling.
   */
  async broadcastAndWait(
    action: string,
    payload?: OutPayload,
    options?: BroadcasterMessageOptions,
  ): Promise<InPayload[]> {
    const id = this.getId();
    this.queue[id] = getResolver<InPayload[]>(options?.timeout);
    this.channel.postMessage({
      id,
      action,
      payload,
    });
    let response: InPayload[];
    try {
      response = await this.queue[id]!.promise;
    } catch (e) {
      console.log("CAUGHT", e);
      response = [];
    }
    return response;
  }

  broadcast(action: string, payload?: OutPayload) {
    this.channel.postMessage({
      id: this.getId(),
      action,
      payload,
    });
  }

  reply(replyId: string, action: string, payload?: OutPayload) {
    this.channel.postMessage({
      id: this.getId(),
      replyId,
      action,
      payload,
    });
  }

  openWindowAndWaitForMessage(rgthreePath: string, windowName?: string) {
    const id = this.getId();
    this.queue[id] = getResolver();
    const win = window.open(`/rgthree/${rgthreePath}#broadcastLoadMsgId=${id}`, windowName);
    return {window: win, promise: this.queue[id]!.promise};
  }

  onMessage(e: MessageEvent<BroadcasterMessage<InPayload>>) {
    const msgId = e.data?.replyId || "";
    const queueItem = this.queue[msgId];
    if (queueItem) {
      if (queueItem.completed) {
        console.error(`${msgId} already completed..`);
      }
      queueItem.deferment = queueItem.deferment || {data: []};
      queueItem.deferment.data.push(e.data.payload);
      queueItem.deferment.timeout && clearTimeout(queueItem.deferment.timeout);
      queueItem.deferment.timeout = setTimeout(() => {
        queueItem.resolve(queueItem.deferment!.data);
      }, 250);
    } else {
      this.dispatchEvent(
        new CustomEvent("rgthree-broadcast-message", {
          detail: Object.assign({replyTo: e.data?.id}, e.data),
        }),
      );
    }
  }

  addMessageListener(callback: EventListener, options?: any) {
    return super.addEventListener("rgthree-broadcast-message", callback, options);
  }
}

const broadcastChannelMap: Map<BroadcastChannel, {[key: string]: Resolver<any>}> = new Map();

export function broadcastOnChannel<T extends {}>(
  channel: BroadcastChannel,
  action: string,
  payload?: T,
) {
  let queue = broadcastChannelMap.get(channel);
  if (!queue) {
    broadcastChannelMap.set(channel, {});
    queue = broadcastChannelMap.get(channel)!;
  }
  let id: string;
  do {
    id = generateId(6);
  } while (queue[id]);
  queue[id] = getResolver();
  channel.postMessage({
    id,
    action,
    payload,
  });
  return queue[id]!.promise;
}
