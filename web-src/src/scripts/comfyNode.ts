import { LiteGraph, LGraphNode, Vector2 } from 'litegraph.js';
import { ComfyObjectInfo } from '../types/comfy';
import { api } from './api';
import { $el } from './utils2';
import { calculateGrid, getImageTop, is_all_same_aspect_ratio } from './helpers';
import { calculateImageGrid, createImageHost } from './ui/imagePreview';
import { ComfyNodeConfig } from '../types/comfy';
import { ComfyWidget, comfyWidgetTypes } from './comfyWidget';

interface Point {
    x: number;
    y: number;
    width: number;
    height: number;
}

function intersect(a: Point, b: Point) {
    const x = Math.max(a.x, b.x);
    const num1 = Math.min(a.x + a.width, b.x + b.width);
    const y = Math.max(a.y, b.y);
    const num2 = Math.min(a.y + a.height, b.y + b.height);
    if (num1 >= x && num2 >= y) return [x, y, num1 - x, num2 - y];
    else return null;
}

export function getClipPath(node: ComfyNode, element: Element, elRect: Point) {
    if (app.canvas) {
        const selectedNode = Object.values(app.canvas.selected_nodes)[0];
        if (selectedNode && selectedNode !== node) {
            const MARGIN = 7;
            const scale = app.canvas?.ds.scale;

            const bounding = selectedNode.getBounding();
            const intersection = intersect(
                {
                    x: elRect.x / scale,
                    y: elRect.y / scale,
                    width: elRect.width / scale,
                    height: elRect.height / scale,
                },
                {
                    x: selectedNode.pos[0] + app.canvas.ds.offset[0] - MARGIN,
                    y: selectedNode.pos[1] + app.canvas.ds.offset[1] - LiteGraph.NODE_TITLE_HEIGHT - MARGIN,
                    width: bounding[2] + MARGIN + MARGIN,
                    height: bounding[3] + MARGIN + MARGIN,
                }
            );

            if (!intersection) {
                return '';
            }

            const widgetRect = element.getBoundingClientRect();
            const clipX = intersection[0] - widgetRect.x / scale + 'px';
            const clipY = intersection[1] - widgetRect.y / scale + 'px';
            const clipWidth = intersection[2] + 'px';
            const clipHeight = intersection[3] + 'px';
            const path = `polygon(0% 0%, 0% 100%, ${clipX} 100%, ${clipX} ${clipY}, calc(${clipX} + ${clipWidth}) ${clipY}, calc(${clipX} + ${clipWidth}) calc(${clipY} + ${clipHeight}), ${clipX} calc(${clipY} + ${clipHeight}), ${clipX} 100%, 100% 100%, 100% 0%)`;
            return path;
        }
    }
    return '';
}

export function addDomClippingSetting(app: ComfyApp) {
    app.ui.settings.addSetting({
        id: 'Comfy.DOMClippingEnabled',
        name: 'Enable DOM element clipping (enabling may reduce performance)',
        type: 'boolean',
        defaultValue: true,
        onChange(value: boolean) {},
    });
}

export const ANIM_PREVIEW_WIDGET = '$$comfy_animation_preview';

const SIZE = Symbol();

interface ComfyOptionsHost {
    el: Element;
    updateImages: (imgs: (HTMLImageElement | string)[]) => void;
    getHeight: () => void;
    onDraw: () => void;
}

interface AddDOMWidgetOptions {
    host?: ComfyOptionsHost;
    getHeight?: ComfyOptionsHost['getHeight'];
    onDraw?: ComfyOptionsHost['onDraw'];
    hideOnZoom?: boolean;
    selectOn?: string[];
    getValue?: () => string | undefined;
    setValue?: (value: any) => string | undefined | void;
    beforeResize?: (node: ComfyNode) => void;
    afterResize?: (node: ComfyNode) => void;
}

// TO DO: replace 'any' types with actually useful types
export class ComfyNode extends LGraphNode {
    app: ComfyApp; // reference to the app this node is attached to
    // title: string;
    category: any;
    // comfyClass: string;
    imageIndex?: number | null;
    imageOffset: number | undefined;
    animatedImages: any | undefined;
    imgs?: HTMLImageElement[] | null;
    images: any[] | undefined;
    // nodeData: any;
    serialize_widgets: boolean;
    widgets: ComfyWidget[]; // idk how to type widgets yet
    resetExecution: boolean;
    pointerWasDown: boolean | null;

    /** Widgets can add event-handlers to nodes. These are some of them. */
    onGraphConfigured?: () => void;
    onAfterGraphConfigured?: () => void;
    onExecutionStart?: (...args: any[]) => void;
    onDragDrop?(e: DragEvent): boolean;
    onDragOver?(e: DragEvent): boolean;
    pasteFile?(file: File): boolean | object;

    // not sure what type the `output` param is yet
    onExecuted?: (output: any) => void;

    // not sure what type the `defs` param is yet
    refreshComboInNode?: (defs: Record<string, ComfyObjectInfo>) => void;

    onResize?: (size: number[]) => void;

    callback?: (args: any) => void;

    inputHeight: number | null;
    freeWidgetSpace: number | null;
    imageRects: [number, number, number, number][] | null;
    overIndex: number | null;
    pointerDown: { pos: Vector2; index: number | null } | null;
    preview: HTMLImageElement | string | string[] | null;

    [SIZE]: boolean | null;

    constructor(nodeData: any, app: ComfyApp) {
        super();
        this.app = app;
        // this.title = nodeData.display_name || nodeData.name;
        this.category = nodeData.category;
        // this.comfyClass = nodeData.name;
        // this.nodeData = nodeData;
        this.widgets = [];
        this.resetExecution = false;
        this.pointerDown = null;
        this.overIndex = null;
        this.pointerWasDown = null;
        this.inputHeight = null;
        this.freeWidgetSpace = null;
        this.imageRects = null;
        this.preview = null;
        this[SIZE] = null;

        let inputs = nodeData['input']['required'];
        if (nodeData['input']['optional'] != undefined) {
            inputs = { ...nodeData['input']['required'], ...nodeData['input']['optional'] };
        }
        const config: ComfyNodeConfig = {
            minWidth: 1,
            minHeight: 1,
            widget: {
                options: {
                    forceInput: false,
                    defaultInput: '',
                },
            },
        };

        for (const inputName in inputs) {
            const inputData = inputs[inputName];
            const type = inputData[0];

            let widgetCreated = true;
            const widgetType = this.getWidgetType(inputData, inputName, app);
            if (widgetType) {
                if (widgetType === 'COMBO') {
                    Object.assign(config, this.app.widgets?.COMBO(this, inputName, inputData, app, '') || {});
                } else {
                    Object.assign(config, this.app.widgets?.[widgetType](this, inputName, inputData, app, '') || {});
                }
            } else {
                // Node connection inputs
                this.addInput(inputName, type);
                widgetCreated = false;
            }

            if (widgetCreated && inputData[1]?.forceInput && config?.widget) {
                if (!config.widget.options) config.widget.options = {};
                config.widget.options.forceInput = inputData[1].forceInput;
            }
            if (widgetCreated && inputData[1]?.defaultInput && config?.widget) {
                if (!config.widget.options) config.widget.options = {};
                config.widget.options.defaultInput = inputData[1].defaultInput;
            }
        }

        for (const o in nodeData['output']) {
            let output = nodeData['output'][o];
            if (output instanceof Array) output = 'COMBO';
            const outputName = nodeData['output_name'][o] || output;
            const outputShape = nodeData['output_is_list'][o] ? LiteGraph.GRID_SHAPE : LiteGraph.CIRCLE_SHAPE;
            this.addOutput(outputName, output, { shape: outputShape });
        }

        const s = this.computeSize();
        this.size = [Math.max(config.minWidth, s[0] * 1.5), Math.max(config.minHeight, s[1])];
        this.serialize_widgets = true;

        app.invokeExtensionsAsync('nodeCreated', this);
    }

    getWidgetType(inputData: any, inputName: string, app: ComfyApp): string | null {
        return app.getWidgetType(inputData, inputName);
    }

    // TO DO: this only makes sense if this is a node with imageIndex defined
    onKeyDown(e: KeyboardEvent) {
        // TO DO: there was something about 'originalKeyDown.apply' here, but I don't know what it was for

        if (this.flags.collapsed || !this.imgs || this.imageIndex === null) {
            return;
        }

        let handled = false;

        // Handle left and right arrow keys
        if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
            if (this.imageIndex) {
                if (e.key === 'ArrowLeft') {
                    this.imageIndex -= 1;
                } else if (e.key === 'ArrowRight') {
                    this.imageIndex += 1;
                }

                // Wrap around the image index
                this.imageIndex %= this.imgs.length;
                if (this.imageIndex < 0) {
                    this.imageIndex = this.imgs.length + this.imageIndex;
                }
            }
            handled = true;
        } else if (e.key === 'Escape') {
            // Handle escape key
            this.imageIndex = null;
            handled = true;
        }

        // If an action was handled, prevent the default action and stop propagation
        if (handled) {
            e.preventDefault();
            e.stopPropagation();
            return false;
        }
    }

    setSizeForImage(force: boolean = false) {
        if (!force && this.animatedImages) return;

        if (this.inputHeight || (this.freeWidgetSpace && this.freeWidgetSpace > 210)) {
            this.setSize(this.size);
            return;
        }
        const minHeight = getImageTop(this) + 220;
        if (this.size[1] < minHeight) {
            this.setSize([this.size[0], minHeight]);
        }
    }

    /**
     * Adds Custom drawing logic for nodes
     * e.g. Draws images and handles thumbnail navigation on nodes that output images
     */
    onDrawBackground(ctx: any) {
        const app = this.app;
        if (!this.flags.collapsed) {
            let imgURLs: (HTMLImageElement | string | string[])[] = [];
            let imagesChanged = false;

            const output = app.nodeOutputs[this.id + ''];
            if (output?.images) {
                this.animatedImages = output?.animated?.find(Boolean);
                if (this.images !== output.images) {
                    this.images = output.images;
                    imagesChanged = true;
                    imgURLs = imgURLs.concat(
                        output.images.map((params: string) => {
                            return api.apiURL(
                                '/view?' +
                                    new URLSearchParams(params).toString() +
                                    (this.animatedImages ? '' : app.getPreviewFormatParam()) +
                                    app.getRandParam()
                            );
                        })
                    );
                }
            }

            const preview = app.nodePreviewImages?.[this.id + ''];
            if (preview && this.preview !== preview) {
                this.preview = preview;
                imagesChanged = true;
                if (preview != null) {
                    imgURLs.push(preview);
                }
            }

            if (imagesChanged) {
                this.imageIndex = null;
                if (imgURLs.length > 0) {
                    Promise.all(
                        imgURLs.map(src => {
                            return new Promise<HTMLImageElement | null>(r => {
                                const img = new Image();
                                img.onload = () => r(img);
                                img.onerror = () => r(null);
                                if (typeof src === 'string') {
                                    img.src = src;
                                }
                            });
                        })
                    ).then(imgs => {
                        if ((!output || this.images === output.images) && (!preview || this.preview === preview)) {
                            this.imgs = imgs.filter(Boolean) as HTMLImageElement[];
                            this.setSizeForImage?.();
                            // app.graph.setDirtyCanvas(true);
                            app.graph?.setDirtyCanvas(true, false);
                        }
                    });
                } else {
                    this.imgs = null;
                }
            }

            if (this.imgs?.length) {
                const widgetIdx = this.widgets?.findIndex(w => w.name === ANIM_PREVIEW_WIDGET);

                if (this.animatedImages) {
                    // Instead of using the canvas we'll use a IMG
                    if (widgetIdx > -1) {
                        // Replace content
                        const widget = this.widgets[widgetIdx];
                        widget.options.host.updateImages(this.imgs);
                    } else {
                        const host = createImageHost(this);
                        this.setSizeForImage(true);
                        const widget = this.addDOMWidget(ANIM_PREVIEW_WIDGET, 'img', host.el as HTMLElement, {
                            host,
                            getHeight: host.getHeight,
                            onDraw: host.onDraw,
                            hideOnZoom: false,
                        });
                        if (widget) {
                            widget.serializeValue = () => undefined;
                            widget.options.host.updateImages(this.imgs);
                        }
                    }
                    return;
                }

                if (widgetIdx > -1) {
                    this.widgets[widgetIdx].onRemove?.();
                    this.widgets.splice(widgetIdx, 1);
                }

                const canvas = app.graph?.list_of_graphcanvas[0];
                const mouse = canvas?.graph_mouse;
                if (mouse && !canvas.pointer_is_down && this.pointerDown) {
                    if (mouse[0] === this.pointerDown.pos[0] && mouse[1] === this.pointerDown.pos[1]) {
                        this.imageIndex = this.pointerDown.index;
                    }
                    this.pointerDown = null;
                }

                let imageIndex = this.imageIndex;
                const numImages = this.imgs.length;
                if (numImages === 1 && !imageIndex) {
                    this.imageIndex = imageIndex = 0;
                }

                const top = getImageTop(this);
                var shiftY = top;

                let dw = this.size[0];
                let dh = this.size[1];
                dh -= shiftY;

                if (imageIndex == null) {
                    let cellWidth: number, cellHeight: number, shiftX: number, cell_padding: number, cols: number;

                    const compact_mode = is_all_same_aspect_ratio(this.imgs);
                    if (!compact_mode) {
                        // use rectangle cell style and border line
                        cell_padding = 2;
                        const { cell_size, columns, rows } = calculateGrid(dw, dh, numImages);
                        cols = columns;

                        cellWidth = cell_size;
                        cellHeight = cell_size;
                        shiftX = (dw - cell_size * cols) / 2;
                        shiftY = (dh - cell_size * rows) / 2 + top;
                    } else {
                        cell_padding = 0;
                        ({ cellWidth, cellHeight, cols, shiftX } = calculateImageGrid(this.imgs, dw, dh) as {
                            cellWidth: number;
                            cellHeight: number;
                            cols: number;
                            shiftX: number;
                        });
                    }

                    let anyHovered = false;
                    this.imageRects = [];
                    for (let i = 0; i < numImages; i++) {
                        const img = this.imgs[i];
                        const row = Math.floor(i / cols);
                        const col = i % cols;
                        const x = col * cellWidth + shiftX;
                        const y = row * cellHeight + shiftY;
                        if (mouse && !anyHovered) {
                            anyHovered = LiteGraph.isInsideRectangle(
                                mouse[0],
                                mouse[1],
                                x + this.pos[0],
                                y + this.pos[1],
                                cellWidth,
                                cellHeight
                            );
                            if (anyHovered) {
                                if (canvas) {
                                    this.overIndex = i;
                                    let value = 110;
                                    if (canvas.pointer_is_down) {
                                        if (!this.pointerDown || this.pointerDown.index !== i) {
                                            this.pointerDown = { index: i, pos: [...mouse] };
                                        }
                                        value = 125;
                                    }
                                    ctx.filter = `contrast(${value}%) brightness(${value}%)`;
                                    canvas.canvas.style.cursor = 'pointer';
                                }
                            }
                        }
                        this.imageRects.push([x, y, cellWidth, cellHeight]);

                        let wratio = cellWidth / img.width;
                        let hratio = cellHeight / img.height;
                        var ratio = Math.min(wratio, hratio);

                        let imgHeight = ratio * img.height;
                        let imgY = row * cellHeight + shiftY + (cellHeight - imgHeight) / 2;
                        let imgWidth = ratio * img.width;
                        let imgX = col * cellWidth + shiftX + (cellWidth - imgWidth) / 2;

                        ctx.drawImage(
                            img,
                            imgX + cell_padding,
                            imgY + cell_padding,
                            imgWidth - cell_padding * 2,
                            imgHeight - cell_padding * 2
                        );
                        if (!compact_mode) {
                            // rectangle cell and border line style
                            ctx.strokeStyle = '#8F8F8F';
                            ctx.lineWidth = 1;
                            ctx.strokeRect(
                                x + cell_padding,
                                y + cell_padding,
                                cellWidth - cell_padding * 2,
                                cellHeight - cell_padding * 2
                            );
                        }

                        ctx.filter = 'none';
                    }

                    if (!anyHovered) {
                        this.pointerDown = null;
                        this.overIndex = null;
                    }
                } else {
                    // Draw individual
                    let w = this.imgs[imageIndex].naturalWidth;
                    let h = this.imgs[imageIndex].naturalHeight;

                    const scaleX = dw / w;
                    const scaleY = dh / h;
                    const scale = Math.min(scaleX, scaleY, 1);

                    w *= scale;
                    h *= scale;

                    let x = (dw - w) / 2;
                    let y = (dh - h) / 2 + shiftY;
                    ctx.drawImage(this.imgs[imageIndex], x, y, w, h);

                    const drawButton = (x: number, y: number, sz: number, text: string) => {
                        if (!mouse) return false;
                        const hovered = LiteGraph.isInsideRectangle(
                            mouse[0],
                            mouse[1],
                            x + this.pos[0],
                            y + this.pos[1],
                            sz,
                            sz
                        );
                        let fill = '#333';
                        let textFill = '#fff';
                        let isClicking = false;
                        if (hovered) {
                            if (canvas) {
                                canvas.canvas.style.cursor = 'pointer';
                                if (canvas.pointer_is_down) {
                                    fill = '#1e90ff';
                                    isClicking = true;
                                } else {
                                    fill = '#eee';
                                    textFill = '#000';
                                }
                            }
                        } else {
                            this.pointerWasDown = null;
                        }

                        ctx.fillStyle = fill;
                        ctx.beginPath();
                        ctx.roundRect(x, y, sz, sz, [4]);
                        ctx.fill();
                        ctx.fillStyle = textFill;
                        ctx.font = '12px Arial';
                        ctx.textAlign = 'center';
                        ctx.fillText(text, x + 15, y + 20);

                        return isClicking;
                    };

                    if (numImages > 1) {
                        if (this.imageIndex && this.pointerDown) {
                            if (drawButton(dw - 40, dh + top - 40, 30, `${this.imageIndex + 1}/${numImages}`)) {
                                let i = this.imageIndex + 1 >= numImages ? 0 : this.imageIndex + 1;
                                // if (!this.pointerDown || !this.pointerDown.index === i) {
                                if (!this.pointerDown || !(this.pointerDown.index === i)) {
                                    if (mouse) {
                                        this.pointerDown = { index: i, pos: [...mouse] };
                                    }
                                }
                            }
                        }

                        if (drawButton(dw - 40, top + 10, 30, `x`)) {
                            // if (!this.pointerDown || !this.pointerDown.index === null)) {
                            if (!this.pointerDown || !(this.pointerDown.index === null)) {
                                if (mouse) {
                                    this.pointerDown = { index: null, pos: [...mouse] };
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /**
     * Adds special context menu handling for nodes
     * e.g. this adds Open Image functionality for nodes that show images
     */
    getExtraMenuOptions(_: any, options: { content: string; callback: () => void }[]) {
        if (this.imgs) {
            // If this node has images then we add an open in new tab item
            let img: HTMLImageElement | undefined;
            if (this.imageIndex != null) {
                // An image is selected so select that
                img = this.imgs[this.imageIndex];
            } else if (this.overIndex != null) {
                // No image is selected but one is hovered
                img = this.imgs[this.overIndex];
            }
            if (img) {
                options.unshift(
                    {
                        content: 'Open Image',
                        callback: () => {
                            if (img) {
                                let url = new URL(img.src);
                                url.searchParams.delete('preview');
                                window.open(url, '_blank');
                            }
                        },
                    },
                    ...getCopyImageOption(img),
                    {
                        content: 'Save Image',
                        callback: () => {
                            if (img) {
                                const a = document.createElement('a');
                                let url = new URL(img.src);
                                url.searchParams.delete('preview');
                                a.href = url.toString();
                                a.setAttribute('download', new URLSearchParams(url.search).get('filename') || '');
                                document.body.append(a);
                                a.click();
                                requestAnimationFrame(() => a.remove());
                            }
                        },
                    }
                );
            }
        }

        options.push({
            content: 'Bypass',
            callback: () => {
                // if (this.mode === 4) this.mode = 0;
                // else this.mode = 4;
                if (this.mode === LiteGraph.NEVER) this.mode = LiteGraph.ALWAYS;
                else this.mode = LiteGraph.NEVER;
                this.graph?.change();
            },
        });

        // prevent conflict of clipspace content
        if (!this.app.clipspace_return_node) {
            options.push({
                content: 'Copy (Clipspace)',
                callback: () => {
                    this.app.copyToClipspace(this);
                },
            });

            if (this.app.clipspace != null) {
                options.push({
                    content: 'Paste (Clipspace)',
                    callback: () => {
                        this.app.pasteFromClipspace(this);
                    },
                });
            }

            if (ComfyApp.isImageNode(this)) {
                options.push({
                    content: 'Open in MaskEditor',
                    callback: () => {
                        this.app.copyToClipspace(this);
                        this.app.clipspace_return_node = this;
                        if (this.app.open_maskeditor) {
                            this.app.open_maskeditor();
                        }
                    },
                });
            }
        }
    }

    addDOMWidget(
        name: string,
        type: string,
        element: HTMLElement,
        options: AddDOMWidgetOptions
    ): ComfyWidget | undefined {
        options = { hideOnZoom: true, selectOn: ['focus', 'click'], ...options };

        if (!element.parentElement) {
            document.body.append(element);
        }

        let mouseDownHandler: (event: MouseEvent) => void;
        if (element.blur) {
            mouseDownHandler = event => {
                if (!element.contains(event.target as Node)) {
                    element.blur();
                }
            };
            document.addEventListener('mousedown', mouseDownHandler);
        }

        const { app } = this;

        const widget: ComfyWidget = {
            name,
            type: type as comfyWidgetTypes,
            get value() {
                return options.getValue?.() ?? undefined;
            },
            set value(v) {
                options.setValue?.(v);
                widget.callback?.(widget.value);
            },
            draw: function (ctx, node, widgetWidth, y, widgetHeight) {
                if (widget.computedHeight == null) {
                    adjustWidgetSizesAndPositions(node, node.size);
                }

                const hidden =
                    node.flags?.collapsed ||
                    (!!options.hideOnZoom && app.canvas && app.canvas.ds.scale < 0.5) ||
                    (widget.computedHeight && widget.computedHeight <= 0) ||
                    widget.type === 'converted-widget' ||
                    widget.type === 'hidden';
                element.hidden = hidden;
                element.style.display = hidden ? 'none' : '';
                if (hidden) {
                    widget.options.onHide?.(widget);
                    return;
                }

                const margin = 10;
                const elRect = ctx.canvas.getBoundingClientRect();
                const transform = new DOMMatrix()
                    .scaleSelf(elRect.width / ctx.canvas.width, elRect.height / ctx.canvas.height)
                    .multiplySelf(ctx.getTransform())
                    .translateSelf(margin, margin + y);

                const scale = new DOMMatrix().scaleSelf(transform.a, transform.d);

                Object.assign(element.style, {
                    transformOrigin: '0 0',
                    transform: scale,
                    left: `${transform.a + transform.e}px`,
                    top: `${transform.d + transform.f}px`,
                    width: `${widgetWidth - margin * 2}px`,
                    height: `${(widget.computedHeight ?? 50) - margin * 2}px`,
                    position: 'absolute',
                    zIndex: app.graph?.nodes.indexOf(node as ComfyNode),
                });

                if (app.ui.settings.getSettingValue('Comfy.DOMClippingEnabled', true)) {
                    element.style.clipPath = getClipPath(node as ComfyNode, element, elRect);
                    element.style.willChange = 'clip-path';
                }

                this.options.onDraw?.(widget);
            },
            element,
            options,
            onRemove() {
                if (mouseDownHandler) {
                    document.removeEventListener('mousedown', mouseDownHandler);
                }
                element.remove();
            },
        };

        if (options.selectOn) {
            for (const evt of options.selectOn) {
                element.addEventListener(evt, () => {
                    this.app.canvas?.selectNode(this);
                    this.app.canvas?.bringToFront(this);
                });
            }
        }

        this.addCustomWidget<ComfyWidget>(widget);
        app.elementWidgets.add(this);

        const collapse = this.collapse;
        this.collapse = function (...args: [force: boolean]) {
            collapse.apply(this, args);
            if (this.flags?.collapsed) {
                element.hidden = true;
                element.style.display = 'none';
            }
        };

        const onRemoved = this.onRemoved;
        this.onRemoved = function (...args: []) {
            element.remove();
            app.elementWidgets.delete(this);
            onRemoved?.apply(this, args);
        };

        if (!this[SIZE]) {
            this[SIZE] = true;
            const onResize = this.onResize;
            this.onResize = function (...args: [size: number[]]) {
                options.beforeResize?.call(widget, this);
                adjustWidgetSizesAndPositions(this, args[0]);
                onResize?.apply(this, args);
                options.afterResize?.call(widget, this);
            };
        }

        return widget;
    }
}

function getCopyImageOption(img: HTMLImageElement) {
    if (typeof window.ClipboardItem === 'undefined') return [];
    return [
        {
            content: 'Copy Image',
            callback: async () => {
                const url = new URL(img.src);
                url.searchParams.delete('preview');

                const writeImage = async (blob: Blob | null) => {
                    if (blob) {
                        await navigator.clipboard.write([
                            new ClipboardItem({
                                [blob.type]: blob,
                            }),
                        ]);
                    }
                };

                try {
                    const data = await fetch(url);
                    const blob = await data.blob();
                    try {
                        await writeImage(blob);
                    } catch (error) {
                        // Chrome seems to only support PNG on write, convert and try again
                        if (blob.type !== 'image/png') {
                            const canvas = $el('canvas', {
                                width: img.naturalWidth,
                                height: img.naturalHeight,
                            }) as HTMLCanvasElement;
                            const ctx = canvas.getContext('2d');
                            let image: HTMLImageElement | ImageBitmap;

                            if (typeof window.createImageBitmap === 'undefined') {
                                image = new Image();
                                const p = new Promise((resolve, reject) => {
                                    if (image instanceof HTMLImageElement) {
                                        image.onload = resolve;
                                        image.onerror = reject;
                                    }
                                }).finally(() => {
                                    URL.revokeObjectURL((image as HTMLImageElement).src);
                                });
                                image.src = URL.createObjectURL(blob);
                                await p;
                            } else {
                                image = await createImageBitmap(blob);
                            }
                            try {
                                ctx?.drawImage(image, 0, 0);
                                canvas.toBlob(writeImage, 'image/png');
                            } finally {
                                if ('close' in image && typeof image.close === 'function') {
                                    image.close();
                                }
                            }

                            return;
                        }
                        throw error;
                    }
                } catch (error: unknown) {
                    alert('Error copying image: ' + (error as Error).message);
                }
            },
        },
    ];
}

type DOMWidgetInfo = {
    minHeight: number;
    prefHeight: number;
    w: ComfyWidget;
    diff?: number;
};

export function adjustWidgetSizesAndPositions(node: ComfyNode, size: number[]) {
    if (node.widgets?.[0]?.last_y == null) return;

    let y = node.widgets[0].last_y;
    let freeSpace = size[1] - y;

    let widgetHeight = 0;
    let dom: DOMWidgetInfo[] = [];
    for (const w of node.widgets) {
        if (w.type === 'converted-widget') {
            // Ignore
            delete w.computedHeight;
        } else if (w.computeSize) {
            widgetHeight += w.computeSize(node.size[0])[1] + 4;
        } else if (w.element) {
            // Extract DOM widget size info
            const styles = getComputedStyle(w.element);
            let minHeight =
                w.options.getMinHeight?.() ?? parseInt(styles.getPropertyValue('--comfy-widget-min-height'));
            let maxHeight =
                w.options.getMaxHeight?.() ?? parseInt(styles.getPropertyValue('--comfy-widget-max-height'));

            let prefHeight = w.options.getHeight?.() ?? styles.getPropertyValue('--comfy-widget-height');
            if (prefHeight.endsWith?.('%')) {
                prefHeight = size[1] * (parseFloat(prefHeight.substring(0, prefHeight.length - 1)) / 100);
            } else {
                prefHeight = parseInt(prefHeight);
                if (isNaN(minHeight)) {
                    minHeight = prefHeight;
                }
            }
            if (isNaN(minHeight)) {
                minHeight = 50;
            }
            if (!isNaN(maxHeight)) {
                if (!isNaN(prefHeight)) {
                    prefHeight = Math.min(prefHeight, maxHeight);
                } else {
                    prefHeight = maxHeight;
                }
            }
            dom.push({
                minHeight,
                prefHeight,
                w,
            });
        } else {
            widgetHeight += LiteGraph.NODE_WIDGET_HEIGHT + 4;
        }
    }

    freeSpace -= widgetHeight;

    // Calculate sizes with all widgets at their min height
    const prefGrow = []; // Nodes that want to grow to their prefd size
    const canGrow = []; // Nodes that can grow to auto size
    let growBy = 0;
    for (const d of dom) {
        freeSpace -= d.minHeight;
        if (isNaN(d.prefHeight)) {
            canGrow.push(d);
            d.w.computedHeight = d.minHeight;
        } else {
            const diff = d.prefHeight - d.minHeight;
            if (diff > 0) {
                prefGrow.push(d);
                growBy += diff;
                d.diff = diff;
            } else {
                d.w.computedHeight = d.minHeight;
            }
        }
    }

    if (node.imgs && !node.widgets.find(w => w.name === ANIM_PREVIEW_WIDGET)) {
        // Allocate space for image
        freeSpace -= 220;
    }

    node.freeWidgetSpace = freeSpace;

    if (freeSpace < 0) {
        // Not enough space for all widgets so we need to grow
        size[1] -= freeSpace;
        node?.graph?.setDirtyCanvas(true, true);
    } else {
        // Share the space between each
        const growDiff = freeSpace - growBy;
        if (growDiff > 0) {
            // All pref sizes can be fulfilled
            freeSpace = growDiff;
            for (const d of prefGrow) {
                d.w.computedHeight = d.prefHeight;
            }
        } else {
            // We need to grow evenly
            const shared = -growDiff / prefGrow.length;
            for (const d of prefGrow) {
                d.w.computedHeight = d.prefHeight - shared;
            }
            freeSpace = 0;
        }

        if (freeSpace > 0 && canGrow.length) {
            // Grow any that are auto height
            const shared = freeSpace / canGrow.length;
            for (const d of canGrow) {
                d.w.computedHeight = shared + (d.w.computedHeight || 0);
            }
        }
    }

    // Position each of the widgets
    for (const w of node.widgets) {
        w.y = y;
        if (w.computedHeight) {
            y += w.computedHeight;
        } else if (w.computeSize) {
            y += w.computeSize(node.size[0])[1] + 4;
        } else {
            y += LiteGraph.NODE_WIDGET_HEIGHT + 4;
        }
    }
}
