import { LiteGraph, LGraphNode, LGraphCanvas, BuiltInSlotType, BuiltInSlotShape } from "../lib/litegraph.core.js"
import { ComfyWidgets } from "./widgets.js";
import { iterateNodeDefOutputs, iterateNodeDefInputs } from "./nodeDef.js";
import { api } from "./api.js";
import { ComfyApp } from "./app.js"

export class ComfyGraphNode extends LGraphNode {
	constructor(title) {
		super(title)
		this.serialize_widgets = true;
	}

	onKeyDown(e) {
		if (super.onKeyDown && super.onKeyDown.apply(this, e) === false) {
			return false;
		}

		if (this.flags.collapsed || !this.imgs || this.imageIndex === null) {
			return;
		}

		let handled = false;

		if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
			if (e.key === "ArrowLeft") {
				this.imageIndex -= 1;
			} else if (e.key === "ArrowRight") {
				this.imageIndex += 1;
			}
			this.imageIndex %= this.imgs.length;

			if (this.imageIndex < 0) {
				this.imageIndex = this.imgs.length + this.imageIndex;
			}
			handled = true;
		} else if (e.key === "Escape") {
			this.imageIndex = null;
			handled = true;
		}

		if (handled === true) {
			e.preventDefault();
			e.stopImmediatePropagation();
			return false;
		}
	}

	/*
	 * SnapToGrid functionality
	 */
	onResize() {
		if (app.shiftDown) {
			const w = LiteGraph.CANVAS_GRID_SIZE * Math.round(node.size[0] / LiteGraph.CANVAS_GRID_SIZE);
			const h = LiteGraph.CANVAS_GRID_SIZE * Math.round(node.size[1] / LiteGraph.CANVAS_GRID_SIZE);
			node.size[0] = w;
			node.size[1] = h;
		}
		return super.onResize?.();
	}

	/**
	 * Adds special context menu handling for nodes
	 * e.g. this adds Open Image functionality for nodes that show images
	 * @param {*} node The node to add the menu handler
	 */
	getExtraMenuOptions(_, options) {
		if (super.getExtraMenuOptions)
			super.getExtraMenuOptions(_, options);

		if (this.imgs) {
			// If this node has images then we add an open in new tab item
			let img;
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
						content: "Open Image",
						callback: () => {
							let url = new URL(img.src);
							url.searchParams.delete('preview');
							window.open(url, "_blank")
						},
					},
					{
						content: "Save Image",
						callback: () => {
							const a = document.createElement("a");
							let url = new URL(img.src);
							url.searchParams.delete('preview');
							a.href = url;
							a.setAttribute("download", new URLSearchParams(url.search).get("filename"));
							document.body.append(a);
							a.click();
							requestAnimationFrame(() => a.remove());
						},
					}
				);
			}
		}

		options.push({
			content: "Bypass",
			callback: (obj) => { if (this.mode === 4) this.mode = 0; else this.mode = 4; this.graph.change(); }
		});

		// prevent conflict of clipspace content
		if(!ComfyApp.clipspace_return_node) {
			options.push({
				content: "Copy (Clipspace)",
				callback: (obj) => { ComfyApp.copyToClipspace(this); }
			});

			if(ComfyApp.clipspace != null) {
				options.push({
					content: "Paste (Clipspace)",
					callback: () => { ComfyApp.pasteFromClipspace(this); }
				});
			}

			if(ComfyApp.isImageNode(this)) {
				options.push({
					content: "Open in MaskEditor",
					callback: (obj) => {
						ComfyApp.copyToClipspace(this);
						ComfyApp.clipspace_return_node = this;
						ComfyApp.open_maskeditor();
					}
				});
			}
		}
	};

	getImageTop() {
		let shiftY;
		if (this.imageOffset != null) {
			shiftY = this.imageOffset;
		} else {
			if (this.widgets?.length) {
				const w = this.widgets[this.widgets.length - 1];
				shiftY = w.last_y;
				if (w.computeSize) {
					shiftY += w.computeSize()[1] + 4;
				}
				else if(w.computedHeight) {
					shiftY += w.computedHeight;
				}
				else {
					shiftY += LiteGraph.NODE_WIDGET_HEIGHT + 4;
				}
			} else {
				shiftY = this.computeSize()[1];
			}
		}
		return shiftY;
	}

	setSizeForImage() {
		if (this.inputHeight) {
			this.setSize(this.size);
			return;
		}
		const minHeight = this.getImageTop() + 220;
		if (this.size[1] < minHeight) {
			this.setSize([this.size[0], minHeight]);
		}
	};

	onDrawBackground(ctx) {
		if (!this.flags.collapsed) {
			let imgURLs = []
			let imagesChanged = false

			const output = app.nodeOutputs[this.id + ""];
			if (output && output.images) {
				if (this.images !== output.images) {
					this.images = output.images;
					imagesChanged = true;
					imgURLs = imgURLs.concat(output.images.map(params => {
						return api.apiURL("/view?" + new URLSearchParams(params).toString() + app.getPreviewFormatParam());
					}))
				}
			}

			const preview = app.nodePreviewImages[this.id + ""]
			if (this.preview !== preview) {
				this.preview = preview
				imagesChanged = true;
				if (preview != null) {
					imgURLs.push(preview);
				}
			}

			if (imagesChanged) {
				this.imageIndex = null;
				if (imgURLs.length > 0) {
					Promise.all(
						imgURLs.map((src) => {
							return new Promise((r) => {
								const img = new Image();
								img.onload = () => r(img);
								img.onerror = () => r(null);
								img.src = src
							});
						})
					).then((imgs) => {
						if ((!output || this.images === output.images) && (!preview || this.preview === preview)) {
							this.imgs = imgs.filter(Boolean);
							this.setSizeForImage?.();
							app.graph.setDirtyCanvas(true);
						}
					});
				}
				else {
					this.imgs = null;
				}
			}

			if (this.imgs && this.imgs.length) {
				const canvas = graph.list_of_graphcanvas[0];
				const mouse = canvas.graph_mouse;
				if (!canvas.pointer_is_down && this.pointerDown) {
					if (mouse[0] === this.pointerDown.pos[0] && mouse[1] === this.pointerDown.pos[1]) {
						this.imageIndex = this.pointerDown.index;
					}
					this.pointerDown = null;
				}

				let w = this.imgs[0].naturalWidth;
				let h = this.imgs[0].naturalHeight;
				let imageIndex = this.imageIndex;
				const numImages = this.imgs.length;
				if (numImages === 1 && !imageIndex) {
					this.imageIndex = imageIndex = 0;
				}

				const shiftY = this.getImageTop();

				let dw = this.size[0];
				let dh = this.size[1];
				dh -= shiftY;

				if (imageIndex == null) {
					let best = 0;
					let cellWidth;
					let cellHeight;
					let cols = 0;
					let shiftX = 0;
					for (let c = 1; c <= numImages; c++) {
						const rows = Math.ceil(numImages / c);
						const cW = dw / c;
						const cH = dh / rows;
						const scaleX = cW / w;
						const scaleY = cH / h;

						const scale = Math.min(scaleX, scaleY, 1);
						const imageW = w * scale;
						const imageH = h * scale;
						const area = imageW * imageH * numImages;

						if (area > best) {
							best = area;
							cellWidth = imageW;
							cellHeight = imageH;
							cols = c;
							shiftX = c * ((cW - imageW) / 2);
						}
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
								canvas.canvas.style.cursor = "pointer";
							}
						}
						this.imageRects.push([x, y, cellWidth, cellHeight]);
						ctx.drawImage(img, x, y, cellWidth, cellHeight);
						ctx.filter = "none";
					}

					if (!anyHovered) {
						this.pointerDown = null;
						this.overIndex = null;
					}
				} else {
					// Draw individual
					const scaleX = dw / w;
					const scaleY = dh / h;
					const scale = Math.min(scaleX, scaleY, 1);

					w *= scale;
					h *= scale;

					let x = (dw - w) / 2;
					let y = (dh - h) / 2 + shiftY;
					ctx.drawImage(this.imgs[imageIndex], x, y, w, h);

					const drawButton = (x, y, sz, text) => {
						const hovered = LiteGraph.isInsideRectangle(mouse[0], mouse[1], x + this.pos[0], y + this.pos[1], sz, sz);
						let fill = "#333";
						let textFill = "#fff";
						let isClicking = false;
						if (hovered) {
							canvas.canvas.style.cursor = "pointer";
							if (canvas.pointer_is_down) {
								fill = "#1e90ff";
								isClicking = true;
							} else {
								fill = "#eee";
								textFill = "#000";
							}
						} else {
							this.pointerWasDown = null;
						}

						ctx.fillStyle = fill;
						ctx.beginPath();
						ctx.roundRect(x, y, sz, sz, [4]);
						ctx.fill();
						ctx.fillStyle = textFill;
						ctx.font = "12px Arial";
						ctx.textAlign = "center";
						ctx.fillText(text, x + 15, y + 20);

						return isClicking;
					};

					if (numImages > 1) {
						if (drawButton(x + w - 35, y + h - 35, 30, `${this.imageIndex + 1}/${numImages}`)) {
							let i = this.imageIndex + 1 >= numImages ? 0 : this.imageIndex + 1;
							if (!this.pointerDown || !this.pointerDown.index === i) {
								this.pointerDown = { index: i, pos: [...mouse] };
							}
						}

						if (drawButton(x + w - 35, y + 5, 30, `x`)) {
							if (!this.pointerDown || !this.pointerDown.index === null) {
								this.pointerDown = { index: null, pos: [...mouse] };
							}
						}
					}
				}
			}
		}
	};
}

// comfy class -> input name -> input config
const defaultInputConfigs = {}


export class ComfyBackendNode extends ComfyGraphNode {
    constructor(title, comfyClass, nodeDef) {
        super(title)
        this.type = comfyClass; // XXX: workaround dependency in LGraphNode.addInput()
        this.displayName = nodeDef.display_name;
        this.comfyNodeDef = nodeDef;
        this.comfyClass = comfyClass;
        this.isBackendNode = true;

        this.#setup(nodeDef)

        // if (nodeDef.output_node) {
        //     this.addOutput("OUTPUT", BuiltInSlotType.EVENT, { color_off: "rebeccapurple", color_on: "rebeccapurple" });
        // }
    }

    get isOutputNode() {
        return this.comfyNodeDef.output_node;
    }

    #setup(nodeDef) {
        defaultInputConfigs[this.type] = {}

		const widgets = Object.assign({}, ComfyWidgets, ComfyWidgets.customWidgets);

        for (const [inputName, inputData] of iterateNodeDefInputs(nodeDef)) {
            const config = {};

            const [type, opts] = inputData;

            if (opts?.forceInput) {
                if (Array.isArray(type)) {
                    throw new Error(`Can't have forceInput set to true for an enum type! ${type}`)
                }
                this.addInput(inputName, type);
            } else {
                if (Array.isArray(type)) {
                    // Enums
                    Object.assign(config, widgets.COMBO(this, inputName, inputData) || {});
                } else if (`${type}:${inputName}` in widgets) {
                    // Support custom ComfyWidgets by Type:Name
                    Object.assign(config, widgets[`${type}:${inputName}`](this, inputName, inputData) || {});
                } else if (type in widgets) {
                    // Standard type ComfyWidgets
                    Object.assign(config, widgets[type](this, inputName, inputData) || {});
                } else {
                    // Node connection inputs (backend)
                    this.addInput(inputName, type);
                }
            }

            if ("widgetNodeType" in config)
                ComfyBackendNode.defaultInputConfigs[this.type][inputName] = config.config
        }

        for (const output of iterateNodeDefOutputs(nodeDef)) {
            const outputShape = output.is_list ? BuiltInSlotShape.GRID_SHAPE : BuiltInSlotShape.CIRCLE_SHAPE;
            this.addOutput(output.name, output.type, { shape: outputShape });
        }

        // app.#invokeExtensionsAsync("nodeCreated", this);
    }

    // onExecuted(outputData) {
    //     console.warn("onExecuted outputs", outputData)
    //     this.triggerSlot(0, outputData)
    // }
}
