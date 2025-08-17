/**
 * [🤮] At some point the new ComfyUI frontend stopped loading it's source as modules and started
 * bundling them together. This removed the ability for to import the individual modules (like the
 * api.js or pnginfo.js) in stand alone pages, like link_fixer.
 *
 * So, what do we have to do? Well, we have to fork, hardcode, or port what we would want to load
 * from ComfyUI as our own, independant files again, which is unforunate for several reasons;
 * duplicate code, risk of falling behind, etc...
 *
 * Anyway, this file is a shim that will either detect we're in the ComfyUI app and pass through the
 * bundled module from the ComfyUI global or load from our own code when that's not available
 * (because we're not in the actual ComfyUI UI).
 */

import type {getPngMetadata, getWebpMetadata} from "typings/comfy.js";

const shimCache = new Map<string, any>();

async function shimComfyUiModule(moduleName: string, prop?: string) {
  let module = shimCache.get(moduleName);
  if (!module) {
    if (window.comfyAPI?.[moduleName]) {
      module = window.comfyAPI?.[moduleName];
    } else {
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

export async function getPngMetadata(file: File | Blob) {
  const fn = (await shimComfyUiModule("pnginfo", "getPngMetadata")) as getPngMetadata;
  return fn(file);
}

export async function getWebpMetadata(file: File | Blob) {
  const fn = (await shimComfyUiModule("pnginfo", "getWebpMetadata")) as getWebpMetadata;
  return fn(file);
}
