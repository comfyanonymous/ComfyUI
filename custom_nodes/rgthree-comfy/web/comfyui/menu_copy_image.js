import { app } from "../../scripts/app.js";
const clipboardSupportedPromise = new Promise(async (resolve) => {
    try {
        const result = await navigator.permissions.query({ name: "clipboard-write" });
        resolve(result.state === "granted");
        return;
    }
    catch (e) {
        try {
            if (!navigator.clipboard.write) {
                throw new Error();
            }
            new ClipboardItem({ "image/png": new Blob([], { type: "image/png" }) });
            resolve(true);
            return;
        }
        catch (e) {
            resolve(false);
        }
    }
});
app.registerExtension({
    name: "rgthree.CopyImageToClipboard",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name.toLowerCase().includes("image")) {
            if (await clipboardSupportedPromise) {
                const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
                nodeType.prototype.getExtraMenuOptions = function (canvas, options) {
                    var _a, _b;
                    options = (_a = getExtraMenuOptions === null || getExtraMenuOptions === void 0 ? void 0 : getExtraMenuOptions.call(this, canvas, options)) !== null && _a !== void 0 ? _a : options;
                    if ((_b = this.imgs) === null || _b === void 0 ? void 0 : _b.length) {
                        let img = this.imgs[this.imageIndex || 0] || this.imgs[this.overIndex || 0] || this.imgs[0];
                        const foundIdx = options.findIndex((option) => { var _a; return (_a = option === null || option === void 0 ? void 0 : option.content) === null || _a === void 0 ? void 0 : _a.includes("Copy Image"); });
                        if (img && foundIdx === -1) {
                            const menuItem = {
                                content: "Copy Image (rgthree)",
                                callback: () => {
                                    const canvas = document.createElement("canvas");
                                    const ctx = canvas.getContext("2d");
                                    canvas.width = img.naturalWidth;
                                    canvas.height = img.naturalHeight;
                                    ctx.drawImage(img, 0, 0, img.naturalWidth, img.naturalHeight);
                                    canvas.toBlob((blob) => {
                                        navigator.clipboard.write([new ClipboardItem({ "image/png": blob })]);
                                    });
                                },
                            };
                            let idx = options.findIndex((option) => { var _a; return (_a = option === null || option === void 0 ? void 0 : option.content) === null || _a === void 0 ? void 0 : _a.includes("Open Image"); }) + 1;
                            if (idx != null) {
                                options.splice(idx, 0, menuItem);
                            }
                            else {
                                options.unshift(menuItem);
                            }
                        }
                    }
                    return [];
                };
            }
        }
    },
});
