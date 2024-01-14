import { ComfyApp } from './app';
import { LGraphNode } from 'litegraph.js';

// TO DO: replace 'any' types with actually useful types
export class ComfyNode extends LGraphNode {
    app: ComfyApp; // reference to the app this node is attached to
    title: string;
    category: any;
    comfyClass: string;
    imageIndex: number | undefined;
    animatedImages: any | undefined;
    imgs: any | undefined;
    images: any[] | undefined;
    nodeData: any;
    serialize_widgets: boolean;

    constructor(nodeData: any, app: ComfyApp) {
        super();
        this.app = app;
        this.title = nodeData.display_name || nodeData.name;
        this.category = nodeData.category;
        this.comfyClass = nodeData.name;
        this.nodeData = nodeData;

        let inputs = nodeData['input']['required'];
        if (nodeData['input']['optional'] != undefined) {
            inputs = { ...nodeData['input']['required'], ...nodeData['input']['optional'] };
        }
        const config = { minWidth: 1, minHeight: 1 };
        for (const inputName in inputs) {
            const inputData = inputs[inputName];
            const type = inputData[0];

            let widgetCreated = true;
            const widgetType = this.getWidgetType(inputData, inputName, app);
            if (widgetType) {
                if (widgetType === 'COMBO') {
                    Object.assign(config, app.widgets.COMBO(this, inputName, inputData, app) || {});
                } else {
                    Object.assign(config, app.widgets[widgetType](this, inputName, inputData, app) || {});
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

    // getWidgetType(inputData: any, inputName: string, app: any): string | null {
    //     // Implementation needed
    //     return null; // Replace with actual widget type computation
    // }

    // TO DO: this only makes sense if this is a node with imageIndex defined
    onKeyDown(e: KeyboardEvent) {
        // TO DO: there was something about 'originalKeyDown.apply' here, but I don't know what it was for

        if (this.flags.collapsed || !this.imgs || this.imageIndex === null) {
            return;
        }

        let handled = false;

        // Handle left and right arrow keys
        if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
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

        if (this.inputHeight || this.freeWidgetSpace > 210) {
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
            let imgURLs: string[] = [];
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

            const preview = app.nodePreviewImages[this.id + ''];
            if (this.preview !== preview) {
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
                            return new Promise(r => {
                                const img = new Image();
                                img.onload = () => r(img);
                                img.onerror = () => r(null);
                                img.src = src;
                            });
                        })
                    ).then(imgs => {
                        if ((!output || this.images === output.images) && (!preview || this.preview === preview)) {
                            this.imgs = imgs.filter(Boolean);
                            this.setSizeForImage?.();
                            app.graph.setDirtyCanvas(true);
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
                        const widget = this.addDOMWidget(ANIM_PREVIEW_WIDGET, 'img', host.el, {
                            host,
                            getHeight: host.getHeight,
                            onDraw: host.onDraw,
                            hideOnZoom: false,
                        });
                        widget.serializeValue = () => undefined;
                        widget.options.host.updateImages(this.imgs);
                    }
                    return;
                }

                if (widgetIdx > -1) {
                    this.widgets[widgetIdx].onRemove?.();
                    this.widgets.splice(widgetIdx, 1);
                }

                const canvas = app.graph.list_of_graphcanvas[0];
                const mouse = canvas.graph_mouse;
                if (!canvas.pointer_is_down && this.pointerDown) {
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
                    var cellWidth, cellHeight, shiftX, cell_padding, cols;

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
                        ({ cellWidth, cellHeight, cols, shiftX } = calculateImageGrid(this.imgs, dw, dh));
                    }

                    let anyHovered = false;
                    this.imageRects = [];
                    for (let i = 0; i < numImages; i++) {
                        const img = this.imgs[i];
                        const row = Math.floor(i / cols);
                        const col = i % cols;
                        const x = col * cellWidth + shiftX;
                        const y = row * cellHeight + shiftY;
                        if (!anyHovered) {
                            anyHovered = LiteGraph.isInsideRectangle(
                                mouse[0],
                                mouse[1],
                                x + this.pos[0],
                                y + this.pos[1],
                                cellWidth,
                                cellHeight
                            );
                            if (anyHovered) {
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

                    const drawButton = (x, y, sz, text) => {
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
                            canvas.canvas.style.cursor = 'pointer';
                            if (canvas.pointer_is_down) {
                                fill = '#1e90ff';
                                isClicking = true;
                            } else {
                                fill = '#eee';
                                textFill = '#000';
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
                        if (drawButton(dw - 40, dh + top - 40, 30, `${this.imageIndex + 1}/${numImages}`)) {
                            let i = this.imageIndex + 1 >= numImages ? 0 : this.imageIndex + 1;
                            if (!this.pointerDown || !this.pointerDown.index === i) {
                                this.pointerDown = { index: i, pos: [...mouse] };
                            }
                        }

                        if (drawButton(dw - 40, top + 10, 30, `x`)) {
                            if (!this.pointerDown || !this.pointerDown.index === null) {
                                this.pointerDown = { index: null, pos: [...mouse] };
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
                if (this.mode === 4) this.mode = 0;
                else this.mode = 4;
                this.graph?.change();
            },
        });

        // prevent conflict of clipspace content
        if (!ComfyApp.clipspace_return_node) {
            options.push({
                content: 'Copy (Clipspace)',
                callback: () => {
                    ComfyApp.copyToClipspace(this);
                },
            });

            if (ComfyApp.clipspace != null) {
                options.push({
                    content: 'Paste (Clipspace)',
                    callback: () => {
                        ComfyApp.pasteFromClipspace(this);
                    },
                });
            }

            if (ComfyApp.isImageNode(this)) {
                options.push({
                    content: 'Open in MaskEditor',
                    callback: () => {
                        ComfyApp.copyToClipspace(this);
                        ComfyApp.clipspace_return_node = this;
                        if (ComfyApp.open_maskeditor) {
                            ComfyApp.open_maskeditor();
                        }
                    },
                });
            }
        }
    }
}
