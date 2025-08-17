const shimCache = new Map();
async function shimComfyUiModule(moduleName, prop) {
    var _a, _b;
    let module = shimCache.get(moduleName);
    if (!module) {
        if ((_a = window.comfyAPI) === null || _a === void 0 ? void 0 : _a[moduleName]) {
            module = (_b = window.comfyAPI) === null || _b === void 0 ? void 0 : _b[moduleName];
        }
        else {
            module = await import(`./comfyui_shim_${moduleName}.js`);
        }
        if (!module) {
            throw new Error(`Module ${moduleName} could not be loaded.`);
        }
        shimCache.set(moduleName, module);
    }
    if (prop) {
        if (!module[prop]) {
            throw new Error(`Property ${prop} on module ${moduleName} could not be loaded.`);
        }
        return module[prop];
    }
    return module;
}
export async function getPngMetadata(file) {
    const fn = (await shimComfyUiModule("pnginfo", "getPngMetadata"));
    return fn(file);
}
export async function getWebpMetadata(file) {
    const fn = (await shimComfyUiModule("pnginfo", "getWebpMetadata"));
    return fn(file);
}
