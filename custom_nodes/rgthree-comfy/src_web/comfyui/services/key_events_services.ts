/**
 * A service responsible for capturing keys within LiteGraph's canvas, and outside of it, allowing
 * nodes and other services to confidently determine what's going on.
 */
class KeyEventService extends EventTarget {
  readonly downKeys: { [key: string]: boolean } = {};
  readonly shiftDownKeys: { [key: string]: boolean } = {};

  ctrlKey = false;
  altKey = false;
  metaKey = false;
  shiftKey = false;

  private readonly isMac: boolean = !!(
    navigator.platform?.toLocaleUpperCase().startsWith("MAC") ||
    (navigator as any).userAgentData?.platform?.toLocaleUpperCase().startsWith("MAC")
  );

  constructor() {
    super();
    this.initialize();
  }

  initialize() {
    const that = this;
    // [🤮] Sometimes ComfyUI and/or LiteGraph stop propagation of key events which makes it hard
    // to determine if keys are currently pressed. To attempt to get around this, we'll hijack
    // LiteGraph's processKey to try to get better consistency.
    const processKey = LGraphCanvas.prototype.processKey;
    LGraphCanvas.prototype.processKey = function (e: KeyboardEvent) {
      if (e.type === "keydown" || e.type === "keyup") {
        that.handleKeyDownOrUp(e);
      }
      return processKey.apply(this, [...arguments] as any) as any;
    };

    // Now that ComfyUI has more non-canvas UI (like the top bar), we listen on window as well, and
    // de-dupe when we get multiple events from both window and/or LiteGraph.
    window.addEventListener("keydown", (e) => {
      that.handleKeyDownOrUp(e);
    });
    window.addEventListener("keyup", (e) => {
      that.handleKeyDownOrUp(e);
    });

    // If we get a visibilitychange, then clear the keys since we can't listen for keys up/down when
    // not visible.
    document.addEventListener("visibilitychange", (e) => {
      this.clearKeydowns();
    });

    // If we get a blur, then also clear the keys since we can't listen for keys up/down when
    // blurred. This can happen w/o a visibilitychange, like a browser alert.
    window.addEventListener("blur", (e) => {
      this.clearKeydowns();
    });
  }

  /**
   * Adds a new queue item, unless the last is the same.
   */
  handleKeyDownOrUp(e: KeyboardEvent) {
    const key = e.key.toLocaleUpperCase();
    // If we're already down, or already up, then ignore and don't fire.
    if ((e.type === 'keydown' && this.downKeys[key] === true)
        || (e.type === 'keyup' && this.downKeys[key] === undefined)) {
      return;
    }

    this.ctrlKey = !!e.ctrlKey;
    this.altKey = !!e.altKey;
    this.metaKey = !!e.metaKey;
    this.shiftKey = !!e.shiftKey;
    if (e.type === "keydown") {
      this.downKeys[key] = true;
      this.dispatchCustomEvent("keydown", { originalEvent: e });

      // If SHIFT is pressed down as well, then we need to keep track of this separetly to "release"
      // it once SHIFT is also released.
      if (this.shiftKey && key !== 'SHIFT') {
        this.shiftDownKeys[key] = true;
      }
    } else if (e.type === "keyup") {
      // See https://github.com/rgthree/rgthree-comfy/issues/238
      // A little bit of a hack, but Mac reportedly does something odd with copy/paste. ComfyUI
      // gobbles the copy event propagation, but it happens for paste too and reportedly 'Enter' which
      // I can't find a reason for in LiteGraph/comfy. So, for Mac only, whenever we lift a Command
      // (META) key, we'll also clear any other keys.
      if (key === "META" && this.isMac) {
        this.clearKeydowns();
      } else {
        delete this.downKeys[key];
      }

      // If we're releasing the SHIFT key, then we may also be releasing all other keys we pressed
      // during the SHIFT key as well. We should get an additional keydown for them after.
      if (key === 'SHIFT') {
        for (const key in this.shiftDownKeys) {
          delete this.downKeys[key];
          delete this.shiftDownKeys[key];
        }
      }
      this.dispatchCustomEvent("keyup", { originalEvent: e });
    }

  }

  private clearKeydowns() {
    this.ctrlKey = false;
    this.altKey = false;
    this.metaKey = false;
    this.shiftKey = false;
    for (const key in this.downKeys) delete this.downKeys[key];
  }

  /**
   * Wraps `dispatchEvent` for easier CustomEvent dispatching.
   */
  private dispatchCustomEvent(event: string, detail?: any) {
    if (detail != null) {
      return this.dispatchEvent(new CustomEvent(event, { detail }));
    }
    return this.dispatchEvent(new CustomEvent(event));
  }

  /**
   * Parses a shortcut string.
   *
   *   - 's' => ['S']
   *   - 'shift + c' => ['SHIFT', 'C']
   *   - 'shift + meta + @' => ['SHIFT', 'META', '@']
   *   - 'shift + + + @' => ['SHIFT', '__PLUS__', '=']
   *   - '+ + p' => ['__PLUS__', 'P']
   */
  private getKeysFromShortcut(shortcut: string | string[]) {
    let keys;
    if (typeof shortcut === "string") {
      // Rip all spaces out. Note, Comfy swallows space, so we don't have to handle it. Otherwise,
      // we would require space to be fed as "Space" or "Spacebar" instead of " ".
      shortcut = shortcut.replace(/\s/g, "");
      // Change a real "+" to something we can encode.
      shortcut = shortcut.replace(/^\+/, "__PLUS__").replace(/\+\+/, "+__PLUS__");
      keys = shortcut.split("+").map((i) => i.replace("__PLUS__", "+"));
    } else {
      keys = [...shortcut];
    }
    return keys.map((k) => k.toLocaleUpperCase());
  }

  /**
   * Checks if all keys passed in are down.
   */
  areAllKeysDown(keys: string | string[]) {
    keys = this.getKeysFromShortcut(keys);
    return keys.every((k) => {
      return this.downKeys[k];
    });
  }

  /**
   * Checks if only the keys passed in are down; optionally and additionally allowing "shift" key.
   */
  areOnlyKeysDown(keys: string | string[], alsoAllowShift = false) {
    keys = this.getKeysFromShortcut(keys);
    const allKeysDown = this.areAllKeysDown(keys);
    const downKeysLength = Object.values(this.downKeys).length;
    // All keys are down and they're the only ones.
    if (allKeysDown && keys.length === downKeysLength) {
      return true;
    }
    // Special case allowing the shift key in addition to the shortcut keys. This helps when a user
    // may had originally defined "$" as a shortcut, but needs to press "shift + $" since it's an
    // upper key character, etc.
    if (alsoAllowShift && !keys.includes("SHIFT") && keys.length === downKeysLength - 1) {
      // If we're holding down shift, have one extra key held down, and the original keys don't
      // include shift, then we're good to go.
      return allKeysDown && this.areAllKeysDown(["SHIFT"]);
    }
    return false;
  }
}

/** The KeyEventService singleton. */
export const SERVICE = new KeyEventService();
