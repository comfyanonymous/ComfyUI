import { IComfyClipspace, SerializedNodeObject } from '../types/interfaces.ts';
import type { ComfyWidget } from '../types/comfyWidget.ts';
import type { ComfyGraph } from '../litegraph/comfyGraph.ts';
import type { ComfyNode } from '../litegraph/comfyNode.ts';
import { ComfyFile } from '../types/many.ts';

export class ComfyClipspace implements IComfyClipspace {
    private static _instance: ComfyClipspace;

    graph: ComfyGraph | null;

    clipspace: SerializedNodeObject | null;
    clipspace_return_node: ComfyNode | null;

    clipspace_invalidate_handler: (() => void) | null;
    openClipspace?: () => void;

    private constructor() {
        this.graph = null;
        this.clipspace = null;
        this.clipspace_invalidate_handler = null;
        this.clipspace_return_node = null;
    }

    static getInstance() {
        if (!ComfyClipspace._instance) {
            ComfyClipspace._instance = new ComfyClipspace();
        }

        return ComfyClipspace._instance;
    }

    onClipspaceEditorSave() {
        if (this.clipspace_return_node) {
            this.pasteFromClipspace(this.clipspace_return_node);
        }
    }

    onClipspaceEditorClosed() {
        this.clipspace_return_node = null;
    }

    copyToClipspace(node: ComfyNode) {
        let widgets = null;
        if (node.widgets) {
            widgets = node.widgets.map(({ type, name, value }) => ({
                type,
                name,
                value,
            })) as ComfyWidget[];
        }

        let imgs = undefined;
        let orig_imgs = undefined;
        if (node.imgs != undefined) {
            imgs = [];
            orig_imgs = [];

            for (let i = 0; i < node.imgs.length; i++) {
                imgs[i] = new Image();
                imgs[i].src = node.imgs[i].src;
                orig_imgs[i] = imgs[i];
            }
        }

        let selectedIndex = 0;
        if (node.imageIndex) {
            selectedIndex = node.imageIndex;
        }

        this.clipspace = {
            widgets: widgets,
            imgs: imgs,
            original_imgs: orig_imgs,
            images: node.images,
            selectedIndex: selectedIndex,
            img_paste_mode: 'selected', // reset to default imf_paste_mode state on copy action
        };

        this.clipspace_return_node = null;

        if (this.clipspace_invalidate_handler) {
            this.clipspace_invalidate_handler();
        }
    }

    pasteFromClipspace(node: ComfyNode, outputs: Record<string, any> = {}) {
        if (this.clipspace) {
            // image paste
            if (this.clipspace.imgs && node.imgs) {
                if (node.images && this.clipspace.images) {
                    if (this.clipspace['img_paste_mode'] == 'selected') {
                        node.images = [this.clipspace.images[this.clipspace['selectedIndex']] as HTMLImageElement];
                    } else {
                        node.images = this.clipspace.images;
                    }

                    if (outputs[node.id + '']) outputs[node.id + ''].images = node.images;
                }

                if (this.clipspace.imgs) {
                    // deep-copy to cut link with clipspace
                    if (this.clipspace['img_paste_mode'] == 'selected') {
                        const img = new Image();
                        img.src = (this.clipspace.imgs[this.clipspace['selectedIndex']] as HTMLImageElement).src;
                        node.imgs = [img];
                        node.imageIndex = 0;
                    } else {
                        const imgs = [];
                        for (let i = 0; i < this.clipspace.imgs.length; i++) {
                            imgs[i] = new Image();
                            imgs[i].src = (this.clipspace.imgs[i] as HTMLImageElement).src;
                            node.imgs = imgs;
                        }
                    }
                }
            }

            if (node.widgets) {
                if (this.clipspace.images) {
                    const clip_image = this.clipspace.images[this.clipspace['selectedIndex']] as ComfyFile;
                    const index = node.widgets.findIndex(obj => obj.name === 'image');
                    if (index >= 0) {
                        if (
                            node.widgets[index].type != 'image' &&
                            typeof node.widgets[index].value == 'string' &&
                            clip_image.filename
                        ) {
                            node.widgets[index].value =
                                (clip_image.subfolder ? clip_image.subfolder + '/' : '') +
                                clip_image.filename +
                                (clip_image.type ? ` [${clip_image.type}]` : '');
                        } else {
                            node.widgets[index].value = clip_image;
                        }
                    }
                }
                if (this.clipspace.widgets) {
                    this.clipspace.widgets.forEach(({ type, name, value }) => {
                        const prop = Object.values(node.widgets).find(obj => obj.type === type && obj.name === name);
                        if (prop && prop.type != 'button') {
                            value = value as ComfyFile;
                            if (prop.type != 'image' && typeof prop.value == 'string' && value.filename) {
                                prop.value =
                                    (value.subfolder ? value.subfolder + '/' : '') +
                                    value.filename +
                                    (value.type ? ` [${value.type}]` : '');
                            } else {
                                prop.value = value;
                                prop.callback?.(value);
                            }
                        }
                    });
                }
            }

            this.graph?.setDirtyCanvas(true, true);
        }
    }
}

export const clipspace = ComfyClipspace.getInstance();
