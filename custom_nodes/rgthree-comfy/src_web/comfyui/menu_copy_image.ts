import type {IContextMenuValue, LGraphCanvas} from "@comfyorg/frontend";
import type {ComfyNodeDef} from "typings/comfy.js";

import {app} from "scripts/app.js";

const clipboardSupportedPromise = new Promise<boolean>(async (resolve) => {
  try {
    // MDN says to check this, but it doesn't work in Mozilla... however, in secure contexts
    // (localhost included), it's given by default if the user has it flagged.. so we should be
    // able to check in the latter ClipboardItem too.
    const result = await navigator.permissions.query({name: "clipboard-write"} as any);
    resolve(result.state === "granted");
    return;
  } catch (e) {
    try {
      if (!navigator.clipboard.write) {
        throw new Error();
      }
      new ClipboardItem({"image/png": new Blob([], {type: "image/png"})});
      resolve(true);
      return;
    } catch (e) {
      resolve(false);
    }
  }
});

/**
 * Adds a "Copy Image" to images in similar fashion to the "native" Open Image and Save Image
 * options.
 */
app.registerExtension({
  name: "rgthree.CopyImageToClipboard",
  async beforeRegisterNodeDef(nodeType: typeof LGraphNode, nodeData: ComfyNodeDef) {
    if (nodeData.name.toLowerCase().includes("image")) {
      if (await clipboardSupportedPromise) {
        const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function (
          canvas: LGraphCanvas,
          options: (IContextMenuValue<unknown> | null)[],
        ): (IContextMenuValue<unknown> | null)[] {
          options = getExtraMenuOptions?.call(this, canvas, options) ?? options;
          // If we already have a copy image somehow, then let's skip ours.
          if (this.imgs?.length) {
            let img =
              this.imgs[this.imageIndex || 0] || this.imgs[this.overIndex || 0] || this.imgs[0];
            const foundIdx = options.findIndex((option) => option?.content?.includes("Copy Image"));
            if (img && foundIdx === -1) {
              const menuItem: IContextMenuValue = {
                content: "Copy Image (rgthree)",
                callback: () => {
                  const canvas = document.createElement("canvas");
                  const ctx = canvas.getContext("2d")!;
                  canvas.width = img.naturalWidth;
                  canvas.height = img.naturalHeight;
                  ctx.drawImage(img, 0, 0, img.naturalWidth, img.naturalHeight);
                  canvas.toBlob((blob) => {
                    navigator.clipboard.write([new ClipboardItem({"image/png": blob!})]);
                  });
                },
              };
              let idx = options.findIndex((option) => option?.content?.includes("Open Image")) + 1;
              if (idx != null) {
                options.splice(idx, 0, menuItem);
              } else {
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
