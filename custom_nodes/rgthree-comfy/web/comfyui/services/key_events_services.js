class KeyEventService extends EventTarget {
    constructor() {
        var _a, _b, _c;
        super();
        this.downKeys = {};
        this.shiftDownKeys = {};
        this.ctrlKey = false;
        this.altKey = false;
        this.metaKey = false;
        this.shiftKey = false;
        this.isMac = !!(((_a = navigator.platform) === null || _a === void 0 ? void 0 : _a.toLocaleUpperCase().startsWith("MAC")) ||
            ((_c = (_b = navigator.userAgentData) === null || _b === void 0 ? void 0 : _b.platform) === null || _c === void 0 ? void 0 : _c.toLocaleUpperCase().startsWith("MAC")));
        this.initialize();
    }
    initialize() {
        const that = this;
        const processKey = LGraphCanvas.prototype.processKey;
        LGraphCanvas.prototype.processKey = function (e) {
            if (e.type === "keydown" || e.type === "keyup") {
                that.handleKeyDownOrUp(e);
            }
            return processKey.apply(this, [...arguments]);
        };
        window.addEventListener("keydown", (e) => {
            that.handleKeyDownOrUp(e);
        });
        window.addEventListener("keyup", (e) => {
            that.handleKeyDownOrUp(e);
        });
        document.addEventListener("visibilitychange", (e) => {
            this.clearKeydowns();
        });
        window.addEventListener("blur", (e) => {
            this.clearKeydowns();
        });
    }
    handleKeyDownOrUp(e) {
        const key = e.key.toLocaleUpperCase();
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
            if (this.shiftKey && key !== 'SHIFT') {
                this.shiftDownKeys[key] = true;
            }
        }
        else if (e.type === "keyup") {
            if (key === "META" && this.isMac) {
                this.clearKeydowns();
            }
            else {
                delete this.downKeys[key];
            }
            if (key === 'SHIFT') {
                for (const key in this.shiftDownKeys) {
                    delete this.downKeys[key];
                    delete this.shiftDownKeys[key];
                }
            }
            this.dispatchCustomEvent("keyup", { originalEvent: e });
        }
    }
    clearKeydowns() {
        this.ctrlKey = false;
        this.altKey = false;
        this.metaKey = false;
        this.shiftKey = false;
        for (const key in this.downKeys)
            delete this.downKeys[key];
    }
    dispatchCustomEvent(event, detail) {
        if (detail != null) {
            return this.dispatchEvent(new CustomEvent(event, { detail }));
        }
        return this.dispatchEvent(new CustomEvent(event));
    }
    getKeysFromShortcut(shortcut) {
        let keys;
        if (typeof shortcut === "string") {
            shortcut = shortcut.replace(/\s/g, "");
            shortcut = shortcut.replace(/^\+/, "__PLUS__").replace(/\+\+/, "+__PLUS__");
            keys = shortcut.split("+").map((i) => i.replace("__PLUS__", "+"));
        }
        else {
            keys = [...shortcut];
        }
        return keys.map((k) => k.toLocaleUpperCase());
    }
    areAllKeysDown(keys) {
        keys = this.getKeysFromShortcut(keys);
        return keys.every((k) => {
            return this.downKeys[k];
        });
    }
    areOnlyKeysDown(keys, alsoAllowShift = false) {
        keys = this.getKeysFromShortcut(keys);
        const allKeysDown = this.areAllKeysDown(keys);
        const downKeysLength = Object.values(this.downKeys).length;
        if (allKeysDown && keys.length === downKeysLength) {
            return true;
        }
        if (alsoAllowShift && !keys.includes("SHIFT") && keys.length === downKeysLength - 1) {
            return allKeysDown && this.areAllKeysDown(["SHIFT"]);
        }
        return false;
    }
}
export const SERVICE = new KeyEventService();
