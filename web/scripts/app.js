import { ComfyLogging } from "./logging.js";
import { ComfyWidgets, initWidgets } from "./widgets.js";
import { ComfyUI, $el } from "./ui.js";
import { api } from "./api.js";
import { defaultGraph } from "./defaultGraph.js";
import { getPngMetadata, getWebpMetadata, importA1111, getLatentMetadata } from "./pnginfo.js";
import { addDomClippingSetting } from "./domWidget.js";
import { createImageHost, calculateImageGrid } from "./ui/imagePreview.js"

export const ANIM_PREVIEW_WIDGET = "$$comfy_animation_preview"

function sanitizeNodeName(string) {
	let entityMap = {
	'&': '',
	'<': '',
	'>': '',
	'"': '',
	"'": '',
	'`': '',
	'=': ''
	};
	return String(string).replace(/[&<>"'`=]/g, function fromEntityMap (s) {
		return entityMap[s];
	});
}

/**
 * @typedef {import("types/comfy").ComfyExtension} ComfyExtension
 */

export class ComfyApp {
	/**
	 * List of entries to queue
	 * @type {{number: number, batchCount: number}[]}
	 */
	#queueItems = [];
	/**
	 * If the queue is currently being processed
	 * @type {boolean}
	 */
	#processingQueue = false;

	/**
	 * Content Clipboard
	 * @type {serialized node object}
	 */
	static clipspace = null;
	static clipspace_invalidate_handler = null;
	static open_maskeditor = null;
	static clipspace_return_node = null;

	constructor() {
		this.ui = new ComfyUI(this);
		this.logging = new ComfyLogging(this);

		/**
		 * List of extensions that are registered with the app
		 * @type {ComfyExtension[]}
		 */
		this.extensions = [];

		/**
		 * Stores the execution output data for each node
		 * @type {Record<string, any>}
		 */
		this.nodeOutputs = {};

		/**
		 * Stores the preview image data for each node
		 * @type {Record<string, Image>}
		 */
		this.nodePreviewImages = {};

		/**
		 * If the shift key on the keyboard is pressed
		 * @type {boolean}
		 */
		this.shiftDown = false;
	}

	getPreviewFormatParam() {
		let preview_format = this.ui.settings.getSettingValue("Comfy.PreviewFormat");
		if(preview_format)
			return `&preview=${preview_format}`;
		else
			return "";
	}

	getRandParam() {
		return "&rand=" + Math.random();
	}

	static isImageNode(node) {
		return node.imgs || (node && node.widgets && node.widgets.findIndex(obj => obj.name === 'image') >= 0);
	}

	static onClipspaceEditorSave() {
		if(ComfyApp.clipspace_return_node) {
			ComfyApp.pasteFromClipspace(ComfyApp.clipspace_return_node);
		}
	}

	static onClipspaceEditorClosed() {
		ComfyApp.clipspace_return_node = null;
	}

	static copyToClipspace(node) {
		var widgets = null;
		if(node.widgets) {
			widgets = node.widgets.map(({ type, name, value }) => ({ type, name, value }));
		}

		var imgs = undefined;
		var orig_imgs = undefined;
		if(node.imgs != undefined) {
			imgs = [];
			orig_imgs = [];

			for (let i = 0; i < node.imgs.length; i++) {
				imgs[i] = new Image();
				imgs[i].src = node.imgs[i].src;
				orig_imgs[i] = imgs[i];
			}
		}

		var selectedIndex = 0;
		if(node.imageIndex) {
			selectedIndex = node.imageIndex;
		}

		ComfyApp.clipspace = {
			'widgets': widgets,
			'imgs': imgs,
			'original_imgs': orig_imgs,
			'images': node.images,
			'selectedIndex': selectedIndex,
			'img_paste_mode': 'selected' // reset to default im_paste_mode state on copy action
		};

		ComfyApp.clipspace_return_node = null;

		if(ComfyApp.clipspace_invalidate_handler) {
			ComfyApp.clipspace_invalidate_handler();
		}
	}

	static pasteFromClipspace(node) {
		if(ComfyApp.clipspace) {
			// image paste
			if(ComfyApp.clipspace.imgs && node.imgs) {
				if(node.images && ComfyApp.clipspace.images) {
					if(ComfyApp.clipspace['img_paste_mode'] == 'selected') {
						node.images = [ComfyApp.clipspace.images[ComfyApp.clipspace['selectedIndex']]];
					}
					else {
						node.images = ComfyApp.clipspace.images;
					}

					if(app.nodeOutputs[node.id + ""])
						app.nodeOutputs[node.id + ""].images = node.images;
				}

				if(ComfyApp.clipspace.imgs) {
					// deep-copy to cut link with clipspace
					if(ComfyApp.clipspace['img_paste_mode'] == 'selected') {
						const img = new Image();
						img.src = ComfyApp.clipspace.imgs[ComfyApp.clipspace['selectedIndex']].src;
						node.imgs = [img];
						node.imageIndex = 0;
					}
					else {
						const imgs = [];
						for(let i=0; i<ComfyApp.clipspace.imgs.length; i++) {
							imgs[i] = new Image();
							imgs[i].src = ComfyApp.clipspace.imgs[i].src;
							node.imgs = imgs;
						}
					}
				}
			}

			if(node.widgets) {
				if(ComfyApp.clipspace.images) {
					const clip_image = ComfyApp.clipspace.images[ComfyApp.clipspace['selectedIndex']];
					const index = node.widgets.findIndex(obj => obj.name === 'image');
					if(index >= 0) {
						if(node.widgets[index].type != 'image' && typeof node.widgets[index].value == "string" && clip_image.filename) {
							node.widgets[index].value = (clip_image.subfolder?clip_image.subfolder+'/':'') + clip_image.filename + (clip_image.type?` [${clip_image.type}]`:'');
						}
						else {
							node.widgets[index].value = clip_image;
						}
					}
				}
				if(ComfyApp.clipspace.widgets) {
					ComfyApp.clipspace.widgets.forEach(({ type, name, value }) => {
						const prop = Object.values(node.widgets).find(obj => obj.type === type && obj.name === name);
						if (prop && prop.type != 'button') {
							if(prop.type != 'image' && typeof prop.value == "string" && value.filename) {
								prop.value = (value.subfolder?value.subfolder+'/':'') + value.filename + (value.type?` [${value.type}]`:'');
							}
							else {
								prop.value = value;
								prop.callback(value);
							}
						}
					});
				}
			}

			app.graph.setDirtyCanvas(true);
		}
	}

	/**
	 * Invoke an extension callback
	 * @param {keyof ComfyExtension} method The extension callback to execute
	 * @param  {any[]} args Any arguments to pass to the callback
	 * @returns
	 */
	#invokeExtensions(method, ...args) {
		let results = [];
		for (const ext of this.extensions) {
			if (method in ext) {
				try {
					results.push(ext[method](...args, this));
				} catch (error) {
					console.error(
						`Error calling extension '${ext.name}' method '${method}'`,
						{ error },
						{ extension: ext },
						{ args }
					);
				}
			}
		}
		return results;
	}

	/**
	 * Invoke an async extension callback
	 * Each callback will be invoked concurrently
	 * @param {string} method The extension callback to execute
	 * @param  {...any} args Any arguments to pass to the callback
	 * @returns
	 */
	async #invokeExtensionsAsync(method, ...args) {
		return await Promise.all(
			this.extensions.map(async (ext) => {
				if (method in ext) {
					try {
						return await ext[method](...args, this);
					} catch (error) {
						console.error(
							`Error calling extension '${ext.name}' method '${method}'`,
							{ error },
							{ extension: ext },
							{ args }
						);
					}
				}
			})
		);
	}
	
	#addRestoreWorkflowView() {
		const serialize = LGraph.prototype.serialize;
		const self = this;
		LGraph.prototype.serialize = function() {
			const workflow = serialize.apply(this, arguments);

			// Store the drag & scale info in the serialized workflow if the setting is enabled
			if (self.enableWorkflowViewRestore.value) {
				if (!workflow.extra) {
					workflow.extra = {};
				}
				workflow.extra.ds = {
					scale: self.canvas.ds.scale,
					offset: self.canvas.ds.offset,
				};
			} else if (workflow.extra?.ds) {
				// Clear any old view data
				delete workflow.extra.ds;
			}

			return workflow;
		}
		this.enableWorkflowViewRestore = this.ui.settings.addSetting({
			id: "Comfy.EnableWorkflowViewRestore",
			name: "Save and restore canvas position and zoom level in workflows",
			type: "boolean",
			defaultValue: true
		});
	}

	/**
	 * Adds special context menu handling for nodes
	 * e.g. this adds Open Image functionality for nodes that show images
	 * @param {*} node The node to add the menu handler
	 */
	#addNodeContextMenuHandler(node) {
		function getCopyImageOption(img) {
			if (typeof window.ClipboardItem === "undefined") return [];
			return [
				{
					content: "Copy Image",
					callback: async () => {
						const url = new URL(img.src);
						url.searchParams.delete("preview");

						const writeImage = async (blob) => {
							await navigator.clipboard.write([
								new ClipboardItem({
									[blob.type]: blob,
								}),
							]);
						};

						try {
							const data = await fetch(url);
							const blob = await data.blob();
							try {
								await writeImage(blob);
							} catch (error) {
								// Chrome seems to only support PNG on write, convert and try again
								if (blob.type !== "image/png") {
									const canvas = $el("canvas", {
										width: img.naturalWidth,
										height: img.naturalHeight,
									});
									const ctx = canvas.getContext("2d");
									let image;
									if (typeof window.createImageBitmap === "undefined") {
										image = new Image();
										const p = new Promise((resolve, reject) => {
											image.onload = resolve;
											image.onerror = reject;
										}).finally(() => {
											URL.revokeObjectURL(image.src);
										});
										image.src = URL.createObjectURL(blob);
										await p;
									} else {
										image = await createImageBitmap(blob);
									}
									try {
										ctx.drawImage(image, 0, 0);
										canvas.toBlob(writeImage, "image/png");
									} finally {
										if (typeof image.close === "function") {
											image.close();
										}
									}

									return;
								}
								throw error;
							}
						} catch (error) {
							alert("Error copying image: " + (error.message ?? error));
						}
					},
				},
			];
		}

		node.prototype.getExtraMenuOptions = function (_, options) {
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
								url.searchParams.delete("preview");
								window.open(url, "_blank");
							},
						},
						...getCopyImageOption(img), 
						{
							content: "Save Image",
							callback: () => {
								const a = document.createElement("a");
								let url = new URL(img.src);
								url.searchParams.delete("preview");
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
				callback: (obj) => {
					if (this.mode === 4) this.mode = 0;
					else this.mode = 4;
					this.graph.change();
				},
			});

			// prevent conflict of clipspace content
			if (!ComfyApp.clipspace_return_node) {
				options.push({
					content: "Copy (Clipspace)",
					callback: (obj) => {
						ComfyApp.copyToClipspace(this);
					},
				});

				if (ComfyApp.clipspace != null) {
					options.push({
						content: "Paste (Clipspace)",
						callback: () => {
							ComfyApp.pasteFromClipspace(this);
						},
					});
				}

				if (ComfyApp.isImageNode(this)) {
					options.push({
						content: "Open in MaskEditor",
						callback: (obj) => {
							ComfyApp.copyToClipspace(this);
							ComfyApp.clipspace_return_node = this;
							ComfyApp.open_maskeditor();
						},
					});
				}
			}
		};
	}

	#addNodeKeyHandler(node) {
		const app = this;
		const origNodeOnKeyDown = node.prototype.onKeyDown;

		node.prototype.onKeyDown = function(e) {
			if (origNodeOnKeyDown && origNodeOnKeyDown.apply(this, e) === false) {
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
	}

	/**
	 * Adds Custom drawing logic for nodes
	 * e.g. Draws images and handles thumbnail navigation on nodes that output images
	 * @param {*} node The node to add the draw handler
	 */
	#addDrawBackgroundHandler(node) {
		const app = this;

		function getImageTop(node) {
			let shiftY;
			if (node.imageOffset != null) {
				shiftY = node.imageOffset;
			} else {
				if (node.widgets?.length) {
					const w = node.widgets[node.widgets.length - 1];
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
					shiftY = node.computeSize()[1];
				}
			}
			return shiftY;
		}

		node.prototype.setSizeForImage = function (force) {
			if(!force && this.animatedImages) return;

			if (this.inputHeight || this.freeWidgetSpace > 210) {
				this.setSize(this.size);
				return;
			}
			const minHeight = getImageTop(this) + 220;
			if (this.size[1] < minHeight) {
				this.setSize([this.size[0], minHeight]);
			}
		};

		node.prototype.onDrawBackground = function (ctx) {
			if (!this.flags.collapsed) {
				let imgURLs = []
				let imagesChanged = false

				const output = app.nodeOutputs[this.id + ""];
				if (output?.images) {
					this.animatedImages = output?.animated?.find(Boolean);
					if (this.images !== output.images) {
						this.images = output.images;
						imagesChanged = true;
						imgURLs = imgURLs.concat(
							output.images.map((params) => {
								return api.apiURL(
									"/view?" +
										new URLSearchParams(params).toString() +
										(this.animatedImages ? "" : app.getPreviewFormatParam()) + app.getRandParam()
								);
							})
						);
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

				function calculateGrid(w, h, n) {
					let columns, rows, cellsize;

					if (w > h) {
						cellsize = h;
						columns = Math.ceil(w / cellsize);
						rows = Math.ceil(n / columns);
					} else {
						cellsize = w;
						rows = Math.ceil(h / cellsize);
						columns = Math.ceil(n / rows);
					}

					while (columns * rows < n) {
						cellsize++;
						if (w >= h) {
							columns = Math.ceil(w / cellsize);
							rows = Math.ceil(n / columns);
						} else {
							rows = Math.ceil(h / cellsize);
							columns = Math.ceil(n / rows);
						}
					}

					const cell_size = Math.min(w/columns, h/rows);
					return {cell_size, columns, rows};
				}

				function is_all_same_aspect_ratio(imgs) {
					// assume: imgs.length >= 2
					let ratio = imgs[0].naturalWidth/imgs[0].naturalHeight;

					for(let i=1; i<imgs.length; i++) {
						let this_ratio = imgs[i].naturalWidth/imgs[i].naturalHeight;
						if(ratio != this_ratio)
							return false;
					}

					return true;
				}

				if (this.imgs?.length) {
					const widgetIdx = this.widgets?.findIndex((w) => w.name === ANIM_PREVIEW_WIDGET);
				
					if(this.animatedImages) {
						// Instead of using the canvas we'll use a IMG
						if(widgetIdx > -1) {
							// Replace content
							const widget = this.widgets[widgetIdx];
							widget.options.host.updateImages(this.imgs);
						} else {
							const host = createImageHost(this);
							this.setSizeForImage(true);
							const widget = this.addDOMWidget(ANIM_PREVIEW_WIDGET, "img", host.el, {
								host,
								getHeight: host.getHeight,
								onDraw: host.onDraw,
								hideOnZoom: false
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
						if(!compact_mode) {
							// use rectangle cell style and border line
							cell_padding = 2;
							const { cell_size, columns, rows } = calculateGrid(dw, dh, numImages);
							cols = columns;

							cellWidth = cell_size;
							cellHeight = cell_size;
							shiftX = (dw-cell_size*cols)/2;
							shiftY = (dh-cell_size*rows)/2 + top;
						}
						else {
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
									canvas.canvas.style.cursor = "pointer";
								}
							}
							this.imageRects.push([x, y, cellWidth, cellHeight]);

							let wratio = cellWidth/img.width;
							let hratio = cellHeight/img.height;
							var ratio = Math.min(wratio, hratio);

							let imgHeight = ratio * img.height;
							let imgY = row * cellHeight + shiftY + (cellHeight - imgHeight)/2;
							let imgWidth = ratio * img.width;
							let imgX = col * cellWidth + shiftX + (cellWidth - imgWidth)/2;

							ctx.drawImage(img, imgX+cell_padding, imgY+cell_padding, imgWidth-cell_padding*2, imgHeight-cell_padding*2);
							if(!compact_mode) {
								// rectangle cell and border line style
								ctx.strokeStyle = "#8F8F8F";
								ctx.lineWidth = 1;
								ctx.strokeRect(x+cell_padding, y+cell_padding, cellWidth-cell_padding*2, cellHeight-cell_padding*2);
							}

							ctx.filter = "none";
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
		};
	}

	/**
	 * Adds a handler allowing drag+drop of files onto the window to load workflows
	 */
	#addDropHandler() {
		// Get prompt from dropped PNG or json
		document.addEventListener("drop", async (event) => {
			event.preventDefault();
			event.stopPropagation();

			const n = this.dragOverNode;
			this.dragOverNode = null;
			// Node handles file drop, we dont use the built in onDropFile handler as its buggy
			// If you drag multiple files it will call it multiple times with the same file
			if (n && n.onDragDrop && (await n.onDragDrop(event))) {
				return;
			}
			// Dragging from Chrome->Firefox there is a file but its a bmp, so ignore that
			if (event.dataTransfer.files.length && event.dataTransfer.files[0].type !== "image/bmp") {
			await this.handleFile(event.dataTransfer.files[0]);
			} else {
				// Try loading the first URI in the transfer list
				const validTypes = ["text/uri-list", "text/x-moz-url"];
				const match = [...event.dataTransfer.types].find((t) => validTypes.find(v => t === v));
				if (match) {
					const uri = event.dataTransfer.getData(match)?.split("\n")?.[0];
					if (uri) {
						await this.handleFile(await (await fetch(uri)).blob());
					}
				}
			}
		});

		// Always clear over node on drag leave
		this.canvasEl.addEventListener("dragleave", async () => {
			if (this.dragOverNode) {
				this.dragOverNode = null;
				this.graph.setDirtyCanvas(false, true);
			}
		});

		// Add handler for dropping onto a specific node
		this.canvasEl.addEventListener(
			"dragover",
			(e) => {
				this.canvas.adjustMouseEvent(e);
				const node = this.graph.getNodeOnPos(e.canvasX, e.canvasY);
				if (node) {
					if (node.onDragOver && node.onDragOver(e)) {
						this.dragOverNode = node;

						// dragover event is fired very frequently, run this on an animation frame
						requestAnimationFrame(() => {
							this.graph.setDirtyCanvas(false, true);
						});
						return;
					}
				}
				this.dragOverNode = null;
			},
			false
		);
	}

	/**
	 * Adds a handler on paste that extracts and loads images or workflows from pasted JSON data
	 */
	#addPasteHandler() {
		document.addEventListener("paste", async (e) => {
			// ctrl+shift+v is used to paste nodes with connections
			// this is handled by litegraph
			if(this.shiftDown) return;

			let data = (e.clipboardData || window.clipboardData);
			const items = data.items;

			// Look for image paste data
			for (const item of items) {
				if (item.type.startsWith('image/')) {
					var imageNode = null;

					// If an image node is selected, paste into it
					if (this.canvas.current_node &&
						this.canvas.current_node.is_selected &&
						ComfyApp.isImageNode(this.canvas.current_node)) {
						imageNode = this.canvas.current_node;
					}

					// No image node selected: add a new one
					if (!imageNode) {
						const newNode = LiteGraph.createNode("LoadImage");
						newNode.pos = [...this.canvas.graph_mouse];
						imageNode = this.graph.add(newNode);
						this.graph.change();
					}
					const blob = item.getAsFile();
					imageNode.pasteFile(blob);
					return;
				}
			}

			// No image found. Look for node data
			data = data.getData("text/plain");
			let workflow;
			try {
				data = data.slice(data.indexOf("{"));
				workflow = JSON.parse(data);
			} catch (err) {
				try {
					data = data.slice(data.indexOf("workflow\n"));
					data = data.slice(data.indexOf("{"));
					workflow = JSON.parse(data);
				} catch (error) {}
			}

			if (workflow && workflow.version && workflow.nodes && workflow.extra) {
				await this.loadGraphData(workflow);
			}
			else {
				if (e.target.type === "text" || e.target.type === "textarea") {
					return;
				}

				// Litegraph default paste
				this.canvas.pasteFromClipboard();
			}


		});
	}


	/**
	 * Adds a handler on copy that serializes selected nodes to JSON
	 */
	#addCopyHandler() {
		document.addEventListener("copy", (e) => {
			if (e.target.type === "text" || e.target.type === "textarea") {
				// Default system copy
				return;
			}

			// copy nodes and clear clipboard
			if (e.target.className === "litegraph" && this.canvas.selected_nodes) {
				this.canvas.copyToClipboard();
				e.clipboardData.setData('text', ' '); //clearData doesn't remove images from clipboard
				e.preventDefault();
				e.stopImmediatePropagation();
				return false;
			}
		});
	}


	/**
	 * Handle mouse
	 *
	 * Move group by header
	 */
	#addProcessMouseHandler() {
		const self = this;

		const origProcessMouseDown = LGraphCanvas.prototype.processMouseDown;
		LGraphCanvas.prototype.processMouseDown = function(e) {
			// prepare for ctrl+shift drag: zoom start
			if(e.ctrlKey && e.shiftKey && e.buttons) {
				self.zoom_drag_start = [e.x, e.y, this.ds.scale];
				return;
			}

			const res = origProcessMouseDown.apply(this, arguments);

			this.selected_group_moving = false;

			if (this.selected_group && !this.selected_group_resizing) {
				var font_size =
					this.selected_group.font_size || LiteGraph.DEFAULT_GROUP_FONT_SIZE;
				var height = font_size * 1.4;

				// Move group by header
				if (LiteGraph.isInsideRectangle(e.canvasX, e.canvasY, this.selected_group.pos[0], this.selected_group.pos[1], this.selected_group.size[0], height)) {
					this.selected_group_moving = true;
				}
			}

			return res;
		}

		const origProcessMouseMove = LGraphCanvas.prototype.processMouseMove;
		LGraphCanvas.prototype.processMouseMove = function(e) {
			// handle ctrl+shift drag
			if(e.ctrlKey && e.shiftKey && self.zoom_drag_start) {
				// stop canvas zoom action
				if(!e.buttons) {
					self.zoom_drag_start = null;
					return;
				}

				// calculate delta
				let deltaY = e.y - self.zoom_drag_start[1];
				let startScale = self.zoom_drag_start[2];

				let scale = startScale - deltaY/100;

				this.ds.changeScale(scale, [this.ds.element.width/2, this.ds.element.height/2]);
				this.graph.change();

				return;
			}

			const orig_selected_group = this.selected_group;

			if (this.selected_group && !this.selected_group_resizing && !this.selected_group_moving) {
				this.selected_group = null;
			}

			const res = origProcessMouseMove.apply(this, arguments);

			if (orig_selected_group && !this.selected_group_resizing && !this.selected_group_moving) {
				this.selected_group = orig_selected_group;
			}

			return res;
		};
	}

	/**
	 * Handle keypress
	 *
	 * Ctrl + M mute/unmute selected nodes
	 */
	#addProcessKeyHandler() {
		const self = this;
		const origProcessKey = LGraphCanvas.prototype.processKey;
		LGraphCanvas.prototype.processKey = function(e) {
			if (!this.graph) {
				return;
			}

			var block_default = false;

			if (e.target.localName == "input") {
				return;
			}

			if (e.type == "keydown" && !e.repeat) {

				// Ctrl + M mute/unmute
				if (e.key === 'm' && e.ctrlKey) {
					if (this.selected_nodes) {
						for (var i in this.selected_nodes) {
							if (this.selected_nodes[i].mode === 2) { // never
								this.selected_nodes[i].mode = 0; // always
							} else {
								this.selected_nodes[i].mode = 2; // never
							}
						}
					}
					block_default = true;
				}

				// Ctrl + B bypass
				if (e.key === 'b' && e.ctrlKey) {
					if (this.selected_nodes) {
						for (var i in this.selected_nodes) {
							if (this.selected_nodes[i].mode === 4) { // never
								this.selected_nodes[i].mode = 0; // always
							} else {
								this.selected_nodes[i].mode = 4; // never
							}
						}
					}
					block_default = true;
				}

				// Alt + C collapse/uncollapse
				if (e.key === 'c' && e.altKey) {
					if (this.selected_nodes) {
						for (var i in this.selected_nodes) {
							this.selected_nodes[i].collapse()
						}
					}
					block_default = true;
				}

				// Ctrl+C Copy
				if ((e.key === 'c') && (e.metaKey || e.ctrlKey)) {
					// Trigger onCopy
					return true;
				}

				// Ctrl+V Paste
				if ((e.key === 'v' || e.key == 'V') && (e.metaKey || e.ctrlKey) && !e.shiftKey) {
					// Trigger onPaste
					return true;
				}

				if((e.key === '+') && e.altKey) {
					block_default = true;
					let scale = this.ds.scale * 1.1;
					this.ds.changeScale(scale, [this.ds.element.width/2, this.ds.element.height/2]);
					this.graph.change();
				}

				if((e.key === '-') && e.altKey) {
					block_default = true;
					let scale = this.ds.scale * 1 / 1.1;
					this.ds.changeScale(scale, [this.ds.element.width/2, this.ds.element.height/2]);
					this.graph.change();
				}
			}

			this.graph.change();

			if (block_default) {
				e.preventDefault();
				e.stopImmediatePropagation();
				return false;
			}

			// Fall through to Litegraph defaults
			return origProcessKey.apply(this, arguments);
		};
	}

	/**
	 * Draws group header bar
	 */
	#addDrawGroupsHandler() {
		const self = this;

		const origDrawGroups = LGraphCanvas.prototype.drawGroups;
		LGraphCanvas.prototype.drawGroups = function(canvas, ctx) {
			if (!this.graph) {
				return;
			}

			var groups = this.graph._groups;

			ctx.save();
			ctx.globalAlpha = 0.7 * this.editor_alpha;

			for (var i = 0; i < groups.length; ++i) {
				var group = groups[i];

				if (!LiteGraph.overlapBounding(this.visible_area, group._bounding)) {
					continue;
				} //out of the visible area

				ctx.fillStyle = group.color || "#335";
				ctx.strokeStyle = group.color || "#335";
				var pos = group._pos;
				var size = group._size;
				ctx.globalAlpha = 0.25 * this.editor_alpha;
				ctx.beginPath();
				var font_size =
					group.font_size || LiteGraph.DEFAULT_GROUP_FONT_SIZE;
				ctx.rect(pos[0] + 0.5, pos[1] + 0.5, size[0], font_size * 1.4);
				ctx.fill();
				ctx.globalAlpha = this.editor_alpha;
			}

			ctx.restore();

			const res = origDrawGroups.apply(this, arguments);
			return res;
		}
	}

	/**
	 * Draws node highlights (executing, drag drop) and progress bar
	 */
	#addDrawNodeHandler() {
		const origDrawNodeShape = LGraphCanvas.prototype.drawNodeShape;
		const self = this;

		LGraphCanvas.prototype.drawNodeShape = function (node, ctx, size, fgcolor, bgcolor, selected, mouse_over) {
			const res = origDrawNodeShape.apply(this, arguments);

			const nodeErrors = self.lastNodeErrors?.[node.id];

			let color = null;
			let lineWidth = 1;
			if (node.id === +self.runningNodeId) {
				color = "#0f0";
			} else if (self.dragOverNode && node.id === self.dragOverNode.id) {
				color = "dodgerblue";
			}
			else if (nodeErrors?.errors) {
				color = "red";
				lineWidth = 2;
			}
			else if (self.lastExecutionError && +self.lastExecutionError.node_id === node.id) {
				color = "#f0f";
				lineWidth = 2;
			}

			if (color) {
				const shape = node._shape || node.constructor.shape || LiteGraph.ROUND_SHAPE;
				ctx.lineWidth = lineWidth;
				ctx.globalAlpha = 0.8;
				ctx.beginPath();
				if (shape == LiteGraph.BOX_SHAPE)
					ctx.rect(-6, -6 - LiteGraph.NODE_TITLE_HEIGHT, 12 + size[0] + 1, 12 + size[1] + LiteGraph.NODE_TITLE_HEIGHT);
				else if (shape == LiteGraph.ROUND_SHAPE || (shape == LiteGraph.CARD_SHAPE && node.flags.collapsed))
					ctx.roundRect(
						-6,
						-6 - LiteGraph.NODE_TITLE_HEIGHT,
						12 + size[0] + 1,
						12 + size[1] + LiteGraph.NODE_TITLE_HEIGHT,
						this.round_radius * 2
					);
				else if (shape == LiteGraph.CARD_SHAPE)
					ctx.roundRect(
						-6,
						-6 - LiteGraph.NODE_TITLE_HEIGHT,
						12 + size[0] + 1,
						12 + size[1] + LiteGraph.NODE_TITLE_HEIGHT,
						[this.round_radius * 2, this.round_radius * 2, 2, 2]
				);
				else if (shape == LiteGraph.CIRCLE_SHAPE)
					ctx.arc(size[0] * 0.5, size[1] * 0.5, size[0] * 0.5 + 6, 0, Math.PI * 2);
				ctx.strokeStyle = color;
				ctx.stroke();
				ctx.strokeStyle = fgcolor;
				ctx.globalAlpha = 1;
			}

			if (self.progress && node.id === +self.runningNodeId) {
				ctx.fillStyle = "green";
				ctx.fillRect(0, 0, size[0] * (self.progress.value / self.progress.max), 6);
				ctx.fillStyle = bgcolor;
			}

			// Highlight inputs that failed validation
			if (nodeErrors) {
				ctx.lineWidth = 2;
				ctx.strokeStyle = "red";
				for (const error of nodeErrors.errors) {
					if (error.extra_info && error.extra_info.input_name) {
						const inputIndex = node.findInputSlot(error.extra_info.input_name)
						if (inputIndex !== -1) {
							let pos = node.getConnectionPos(true, inputIndex);
							ctx.beginPath();
							ctx.arc(pos[0] - node.pos[0], pos[1] - node.pos[1], 12, 0, 2 * Math.PI, false)
							ctx.stroke();
						}
					}
				}
			}

			return res;
		};

		const origDrawNode = LGraphCanvas.prototype.drawNode;
		LGraphCanvas.prototype.drawNode = function (node, ctx) {
			var editor_alpha = this.editor_alpha;
			var old_color = node.bgcolor;

			if (node.mode === 2) { // never
				this.editor_alpha = 0.4;
			}

			if (node.mode === 4) { // never
				node.bgcolor = "#FF00FF";
				this.editor_alpha = 0.2;
			}

			const res = origDrawNode.apply(this, arguments);

			this.editor_alpha = editor_alpha;
			node.bgcolor = old_color;

			return res;
		};
	}

	/**
	 * Handles updates from the API socket
	 */
	#addApiUpdateHandlers() {
		api.addEventListener("status", ({ detail }) => {
			this.ui.setStatus(detail);
		});

		api.addEventListener("reconnecting", () => {
			this.ui.dialog.show("Reconnecting...");
		});

		api.addEventListener("reconnected", () => {
			this.ui.dialog.close();
		});

		api.addEventListener("progress", ({ detail }) => {
			this.progress = detail;
			this.graph.setDirtyCanvas(true, false);
		});

		api.addEventListener("executing", ({ detail }) => {
			this.progress = null;
			this.runningNodeId = detail;
			this.graph.setDirtyCanvas(true, false);
			delete this.nodePreviewImages[this.runningNodeId]
		});

		api.addEventListener("executed", ({ detail }) => {
			const output = this.nodeOutputs[detail.node];
			if (detail.merge && output) {
				for (const k in detail.output ?? {}) {
					const v = output[k];
					if (v instanceof Array) {
						output[k] = v.concat(detail.output[k]);
					} else {
						output[k] = detail.output[k];
					}
				}
			} else {
				this.nodeOutputs[detail.node] = detail.output;
			}
			const node = this.graph.getNodeById(detail.node);
			if (node) {
				if (node.onExecuted)
					node.onExecuted(detail.output);
			}
		});

		api.addEventListener("execution_start", ({ detail }) => {
			this.runningNodeId = null;
			this.lastExecutionError = null
			this.graph._nodes.forEach((node) => {
				if (node.onExecutionStart)
					node.onExecutionStart()
			})
		});

		api.addEventListener("execution_error", ({ detail }) => {
			this.lastExecutionError = detail;
			const formattedError = this.#formatExecutionError(detail);
			this.ui.dialog.show(formattedError);
			this.canvas.draw(true, true);
		});

		api.addEventListener("b_preview", ({ detail }) => {
			const id = this.runningNodeId
			if (id == null)
				return;

			const blob = detail
			const blobUrl = URL.createObjectURL(blob)
			this.nodePreviewImages[id] = [blobUrl]
		});

		api.init();
	}

	#addKeyboardHandler() {
		window.addEventListener("keydown", (e) => {
			this.shiftDown = e.shiftKey;
		});
		window.addEventListener("keyup", (e) => {
			this.shiftDown = e.shiftKey;
		});
	}

	#addConfigureHandler() {
		const app = this;
		const configure = LGraph.prototype.configure;
		// Flag that the graph is configuring to prevent nodes from running checks while its still loading
		LGraph.prototype.configure = function () {
			app.configuringGraph = true;
			try {
				return configure.apply(this, arguments);
			} finally {
				app.configuringGraph = false;
			}
		};
	}

	#addAfterConfigureHandler() {
		const app = this;
		const onConfigure = app.graph.onConfigure;
		app.graph.onConfigure = function () {
			// Fire callbacks before the onConfigure, this is used by widget inputs to setup the config
			for (const node of app.graph._nodes) {
				node.onGraphConfigured?.();
			}
			
			const r = onConfigure?.apply(this, arguments);
			
			// Fire after onConfigure, used by primitves to generate widget using input nodes config
			for (const node of app.graph._nodes) {
				node.onAfterGraphConfigured?.();
			}

			return r;
		};
	}

	/**
	 * Loads all extensions from the API into the window in parallel
	 */
	async #loadExtensions() {
	    const extensions = await api.getExtensions();
	    this.logging.addEntry("Comfy.App", "debug", { Extensions: extensions });
	
	    const extensionPromises = extensions.map(async ext => {
	        try {
	            await import(api.apiURL(ext));
	        } catch (error) {
	            console.error("Error loading extension", ext, error);
	        }
	    });
	
	    await Promise.all(extensionPromises);
	}

	async #migrateSettings() {
		this.isNewUserSession = true;
		// Store all current settings
		const settings = Object.keys(this.ui.settings).reduce((p, n) => {
			const v = localStorage[`Comfy.Settings.${n}`];
			if (v) {
				try {
					p[n] = JSON.parse(v);
				} catch (error) {}
			}
			return p;
		}, {});

		await api.storeSettings(settings);
	}

	async #setUser() {
		const userConfig = await api.getUserConfig();
		this.storageLocation = userConfig.storage;
		if (typeof userConfig.migrated == "boolean") {
			// Single user mode migrated true/false for if the default user is created
			if (!userConfig.migrated && this.storageLocation === "server") {
				// Default user not created yet
				await this.#migrateSettings();
			}
			return;
		}

		this.multiUserServer = true;
		let user = localStorage["Comfy.userId"];
		const users = userConfig.users ?? {};
		if (!user || !users[user]) {
			// This will rarely be hit so move the loading to on demand
			const { UserSelectionScreen } = await import("./ui/userSelection.js");
		
			this.ui.menuContainer.style.display = "none";
			const { userId, username, created } = await new UserSelectionScreen().show(users, user);
			this.ui.menuContainer.style.display = "";

			user = userId;
			localStorage["Comfy.userName"] = username;
			localStorage["Comfy.userId"] = user;

			if (created) {
				api.user = user;
				await this.#migrateSettings();
			}
		}

		api.user = user;

		this.ui.settings.addSetting({
			id: "Comfy.SwitchUser",
			name: "Switch User",
			type: (name) => {
				let currentUser = localStorage["Comfy.userName"];
				if (currentUser) {
					currentUser = ` (${currentUser})`;
				}
				return $el("tr", [
					$el("td", [
						$el("label", {
							textContent: name,
						}),
					]),
					$el("td", [
						$el("button", {
							textContent: name + (currentUser ?? ""),
							onclick: () => {
								delete localStorage["Comfy.userId"];
								delete localStorage["Comfy.userName"];
								window.location.reload();
							},
						}),
					]),
				]);
			},
		});
	}

	/**
	 * Set up the app on the page
	 */
	async setup() {
		await this.#setUser();
		await this.ui.settings.load();
		await this.#loadExtensions();

		// Create and mount the LiteGraph in the DOM
		const mainCanvas = document.createElement("canvas")
		mainCanvas.style.touchAction = "none"
		const canvasEl = (this.canvasEl = Object.assign(mainCanvas, { id: "graph-canvas" }));
		canvasEl.tabIndex = "1";
		document.body.prepend(canvasEl);

		addDomClippingSetting();
		this.#addProcessMouseHandler();
		this.#addProcessKeyHandler();
		this.#addConfigureHandler();
		this.#addApiUpdateHandlers();
		this.#addRestoreWorkflowView();

		this.graph = new LGraph();

		this.#addAfterConfigureHandler();

		const canvas = (this.canvas = new LGraphCanvas(canvasEl, this.graph));
		this.ctx = canvasEl.getContext("2d");

		LiteGraph.release_link_on_empty_shows_menu = true;
		LiteGraph.alt_drag_do_clone_nodes = true;

		this.graph.start();

		function resizeCanvas() {
			// Limit minimal scale to 1, see https://github.com/comfyanonymous/ComfyUI/pull/845
			const scale = Math.max(window.devicePixelRatio, 1);
			const { width, height } = canvasEl.getBoundingClientRect();
			canvasEl.width = Math.round(width * scale);
			canvasEl.height = Math.round(height * scale);
			canvasEl.getContext("2d").scale(scale, scale);
			canvas.draw(true, true);
		}

		// Ensure the canvas fills the window
		resizeCanvas();
		window.addEventListener("resize", resizeCanvas);

		await this.#invokeExtensionsAsync("init");
		await this.registerNodes();
		initWidgets(this);

		// Load previous workflow
		let restored = false;
		try {
			const loadWorkflow = async (json) => {
				if (json) {
					const workflow = JSON.parse(json);
					await this.loadGraphData(workflow);
					return true;
				}
			};
			const clientId = api.initialClientId ?? api.clientId;
			restored =
				(clientId && (await loadWorkflow(sessionStorage.getItem(`workflow:${clientId}`)))) ||
				(await loadWorkflow(localStorage.getItem("workflow")));
		} catch (err) {
			console.error("Error loading previous workflow", err);
		}

		// We failed to restore a workflow so load the default
		if (!restored) {
			await this.loadGraphData();
		}

		// Save current workflow automatically
		setInterval(() => {
			const workflow = JSON.stringify(this.graph.serialize());
			localStorage.setItem("workflow", workflow);
			if (api.clientId) {
				sessionStorage.setItem(`workflow:${api.clientId}`, workflow);
			}
		}, 1000);

		this.#addDrawNodeHandler();
		this.#addDrawGroupsHandler();
		this.#addDropHandler();
		this.#addCopyHandler();
		this.#addPasteHandler();
		this.#addKeyboardHandler();

		await this.#invokeExtensionsAsync("setup");
	}

	/**
	 * Registers nodes with the graph
	 */
	async registerNodes() {
		const app = this;
		// Load node definitions from the backend
		const defs = await api.getNodeDefs();
		await this.registerNodesFromDefs(defs);
		await this.#invokeExtensionsAsync("registerCustomNodes");
	}

	getWidgetType(inputData, inputName) {
		const type = inputData[0];

		if (Array.isArray(type)) {
			return "COMBO";
		} else if (`${type}:${inputName}` in this.widgets) {
			return `${type}:${inputName}`;
		} else if (type in this.widgets) {
			return type;
		} else {
			return null;
		}
	}

	async registerNodeDef(nodeId, nodeData) {
		const self = this;
		const node = Object.assign(
			function ComfyNode() {
				var inputs = nodeData["input"]["required"];
				if (nodeData["input"]["optional"] != undefined) {
					inputs = Object.assign({}, nodeData["input"]["required"], nodeData["input"]["optional"]);
				}
				const config = { minWidth: 1, minHeight: 1 };
				for (const inputName in inputs) {
					const inputData = inputs[inputName];
					const type = inputData[0];

					let widgetCreated = true;
					const widgetType = self.getWidgetType(inputData, inputName);
					if(widgetType) {
						if(widgetType === "COMBO") {
							Object.assign(config, self.widgets.COMBO(this, inputName, inputData, app) || {});
						} else {
							Object.assign(config, self.widgets[widgetType](this, inputName, inputData, app) || {});
						}
					} else {
						// Node connection inputs
						this.addInput(inputName, type);
						widgetCreated = false;
					}

					if(widgetCreated && inputData[1]?.forceInput && config?.widget) {
						if (!config.widget.options) config.widget.options = {};
						config.widget.options.forceInput = inputData[1].forceInput;
					}
					if(widgetCreated && inputData[1]?.defaultInput && config?.widget) {
						if (!config.widget.options) config.widget.options = {};
						config.widget.options.defaultInput = inputData[1].defaultInput;
					}
				}

				for (const o in nodeData["output"]) {
					let output = nodeData["output"][o];
					if(output instanceof Array) output = "COMBO";
					const outputName = nodeData["output_name"][o] || output;
					const outputShape = nodeData["output_is_list"][o] ? LiteGraph.GRID_SHAPE : LiteGraph.CIRCLE_SHAPE ;
					this.addOutput(outputName, output, { shape: outputShape });
				}

				const s = this.computeSize();
				s[0] = Math.max(config.minWidth, s[0] * 1.5);
				s[1] = Math.max(config.minHeight, s[1]);
				this.size = s;
				this.serialize_widgets = true;

				app.#invokeExtensionsAsync("nodeCreated", this);
			},
			{
				title: nodeData.display_name || nodeData.name,
				comfyClass: nodeData.name,
				nodeData
			}
		);
		node.prototype.comfyClass = nodeData.name;

		this.#addNodeContextMenuHandler(node);
		this.#addDrawBackgroundHandler(node, app);
		this.#addNodeKeyHandler(node);

		await this.#invokeExtensionsAsync("beforeRegisterNodeDef", node, nodeData);
		LiteGraph.registerNodeType(nodeId, node);
		node.category = nodeData.category;
	}

    async registerNodesFromDefs(defs) {
		await this.#invokeExtensionsAsync("addCustomNodeDefs", defs);

		// Generate list of known widgets
		this.widgets = Object.assign(
			{},
			ComfyWidgets,
			...(await this.#invokeExtensionsAsync("getCustomWidgets")).filter(Boolean)
		);

		// Register a node for each definition
		for (const nodeId in defs) {
			this.registerNodeDef(nodeId, defs[nodeId]);
		}
	}

	loadTemplateData(templateData) {
		if (!templateData?.templates) {
			return;
		}

		const old = localStorage.getItem("litegrapheditor_clipboard");

		var maxY, nodeBottom, node;

		for (const template of templateData.templates) {
			if (!template?.data) {
				continue;
			}

			localStorage.setItem("litegrapheditor_clipboard", template.data);
			app.canvas.pasteFromClipboard();

			// Move mouse position down to paste the next template below

			maxY = false;

			for (const i in app.canvas.selected_nodes) {
				node = app.canvas.selected_nodes[i];

				nodeBottom = node.pos[1] + node.size[1];

				if (maxY === false || nodeBottom > maxY) {
					maxY = nodeBottom;
				}
			}

			app.canvas.graph_mouse[1] = maxY + 50;
		}

		localStorage.setItem("litegrapheditor_clipboard", old);
	}

	showMissingNodesError(missingNodeTypes, hasAddedNodes = true) {
		let seenTypes = new Set();

		this.ui.dialog.show(
			$el("div.comfy-missing-nodes", [
				$el("span", { textContent: "When loading the graph, the following node types were not found: " }),
				$el(
					"ul",
					Array.from(new Set(missingNodeTypes)).map((t) => {
						let children = [];
						if (typeof t === "object") {
							if(seenTypes.has(t.type)) return null;
							seenTypes.add(t.type);
							children.push($el("span", { textContent: t.type }));
							if (t.hint) {
								children.push($el("span", { textContent: t.hint }));
							}
							if (t.action) {
								children.push($el("button", { onclick: t.action.callback, textContent: t.action.text }));
							}
						} else {
							if(seenTypes.has(t)) return null;
							seenTypes.add(t);
							children.push($el("span", { textContent: t }));
						}
						return $el("li", children);
					}).filter(Boolean)
				),
				...(hasAddedNodes
					? [$el("span", { textContent: "Nodes that have failed to load will show as red on the graph." })]
					: []),
			])
		);
		this.logging.addEntry("Comfy.App", "warn", {
			MissingNodes: missingNodeTypes,
		});
	}

	/**
	 * Populates the graph with the specified workflow data
	 * @param {*} graphData A serialized graph object
	 * @param { boolean } clean If the graph state, e.g. images, should be cleared
	 */
	async loadGraphData(graphData, clean = true, restore_view = true) {
		if (clean !== false) {
			this.clean();
		}

		let reset_invalid_values = false;
		if (!graphData) {
			graphData = defaultGraph;
			reset_invalid_values = true;
		}

		if (typeof structuredClone === "undefined")
		{
			graphData = JSON.parse(JSON.stringify(graphData));
		}else
		{
			graphData = structuredClone(graphData);
		}

		const missingNodeTypes = [];
		await this.#invokeExtensionsAsync("beforeConfigureGraph", graphData, missingNodeTypes);
		for (let n of graphData.nodes) {
			// Patch T2IAdapterLoader to ControlNetLoader since they are the same node now
			if (n.type == "T2IAdapterLoader") n.type = "ControlNetLoader";
			if (n.type == "ConditioningAverage ") n.type = "ConditioningAverage"; //typo fix
			if (n.type == "SDV_img2vid_Conditioning") n.type = "SVD_img2vid_Conditioning"; //typo fix

			// Find missing node types
			if (!(n.type in LiteGraph.registered_node_types)) {
				missingNodeTypes.push(n.type);
				n.type = sanitizeNodeName(n.type);
			}
		}

		try {
			this.graph.configure(graphData);
			if (restore_view && this.enableWorkflowViewRestore.value && graphData.extra?.ds) {
				this.canvas.ds.offset = graphData.extra.ds.offset;
				this.canvas.ds.scale = graphData.extra.ds.scale;
			}
		} catch (error) {
			let errorHint = [];
			// Try extracting filename to see if it was caused by an extension script
			const filename = error.fileName || (error.stack || "").match(/(\/extensions\/.*\.js)/)?.[1];
			const pos = (filename || "").indexOf("/extensions/");
			if (pos > -1) {
				errorHint.push(
					$el("span", { textContent: "This may be due to the following script:" }),
					$el("br"),
					$el("span", {
						style: {
							fontWeight: "bold",
						},
						textContent: filename.substring(pos),
					})
				);
			}

			// Show dialog to let the user know something went wrong loading the data
			this.ui.dialog.show(
				$el("div", [
					$el("p", { textContent: "Loading aborted due to error reloading workflow data" }),
					$el("pre", {
						style: { padding: "5px", backgroundColor: "rgba(255,0,0,0.2)" },
						textContent: error.toString(),
					}),
					$el("pre", {
						style: {
							padding: "5px",
							color: "#ccc",
							fontSize: "10px",
							maxHeight: "50vh",
							overflow: "auto",
							backgroundColor: "rgba(0,0,0,0.2)",
						},
						textContent: error.stack || "No stacktrace available",
					}),
					...errorHint,
				]).outerHTML
			);

			return;
		}

		for (const node of this.graph._nodes) {
			const size = node.computeSize();
			size[0] = Math.max(node.size[0], size[0]);
			size[1] = Math.max(node.size[1], size[1]);
			node.size = size;

			if (node.widgets) {
				// If you break something in the backend and want to patch workflows in the frontend
				// This is the place to do this
				for (let widget of node.widgets) {
					if (node.type == "KSampler" || node.type == "KSamplerAdvanced") {
						if (widget.name == "sampler_name") {
							if (widget.value.startsWith("sample_")) {
								widget.value = widget.value.slice(7);
							}
						}
					}
					if (node.type == "KSampler" || node.type == "KSamplerAdvanced" || node.type == "PrimitiveNode") {
						if (widget.name == "control_after_generate") {
							if (widget.value === true) {
								widget.value = "randomize";
							} else if (widget.value === false) {
								widget.value = "fixed";
							}
						}
					}
					if (reset_invalid_values) {
						if (widget.type == "combo") {
							if (!widget.options.values.includes(widget.value) && widget.options.values.length > 0) {
								widget.value = widget.options.values[0];
							}
						}
					}
				}
			}

			this.#invokeExtensions("loadedGraphNode", node);
		}

		if (missingNodeTypes.length) {
			this.showMissingNodesError(missingNodeTypes);
		}
		await this.#invokeExtensionsAsync("afterConfigureGraph", missingNodeTypes);
	}

	/**
	 * Converts the current graph workflow for sending to the API
	 * @returns The workflow and node links
	 */
	async graphToPrompt() {
		for (const outerNode of this.graph.computeExecutionOrder(false)) {
			if (outerNode.widgets) {
				for (const widget of outerNode.widgets) {
					// Allow widgets to run callbacks before a prompt has been queued
					// e.g. random seed before every gen
					widget.beforeQueued?.();
				}
			}

			const innerNodes = outerNode.getInnerNodes ? outerNode.getInnerNodes() : [outerNode];
			for (const node of innerNodes) {
				if (node.isVirtualNode) {
					// Don't serialize frontend only nodes but let them make changes
					if (node.applyToGraph) {
						node.applyToGraph();
					}
				}
			}
		}

		const workflow = this.graph.serialize();
		const output = {};
		// Process nodes in order of execution
		for (const outerNode of this.graph.computeExecutionOrder(false)) {
			const skipNode = outerNode.mode === 2 || outerNode.mode === 4;
			const innerNodes = (!skipNode && outerNode.getInnerNodes) ? outerNode.getInnerNodes() : [outerNode];
			for (const node of innerNodes) {
				if (node.isVirtualNode) {
					continue;
				}

				if (node.mode === 2 || node.mode === 4) {
					// Don't serialize muted nodes
					continue;
				}

				const inputs = {};
				const widgets = node.widgets;

				// Store all widget values
				if (widgets) {
					for (const i in widgets) {
						const widget = widgets[i];
						if (!widget.options || widget.options.serialize !== false) {
							inputs[widget.name] = widget.serializeValue ? await widget.serializeValue(node, i) : widget.value;
						}
					}
				}

				// Store all node links
				for (let i in node.inputs) {
					let parent = node.getInputNode(i);
					if (parent) {
						let link = node.getInputLink(i);
						while (parent.mode === 4 || parent.isVirtualNode) {
							let found = false;
							if (parent.isVirtualNode) {
								link = parent.getInputLink(link.origin_slot);
								if (link) {
									parent = parent.getInputNode(link.target_slot);
									if (parent) {
										found = true;
									}
								}
							} else if (link && parent.mode === 4) {
								let all_inputs = [link.origin_slot];
								if (parent.inputs) {
									all_inputs = all_inputs.concat(Object.keys(parent.inputs))
									for (let parent_input in all_inputs) {
										parent_input = all_inputs[parent_input];
										if (parent.inputs[parent_input]?.type === node.inputs[i].type) {
											link = parent.getInputLink(parent_input);
											if (link) {
												parent = parent.getInputNode(parent_input);
											}
											found = true;
											break;
										}
									}
								}
							}

							if (!found) {
								break;
							}
						}

						if (link) {
							if (parent?.updateLink) {
								link = parent.updateLink(link);
							}
							if (link) {
								inputs[node.inputs[i].name] = [String(link.origin_id), parseInt(link.origin_slot)];
							}
						}
					}
				}

				let node_data = {
					inputs,
					class_type: node.comfyClass,
				};

				if (this.ui.settings.getSettingValue("Comfy.DevMode")) {
					// Ignored by the backend.
					node_data["_meta"] = {
						title: node.title,
					}
				}

				output[String(node.id)] = node_data;
			}
		}

		// Remove inputs connected to removed nodes

		for (const o in output) {
			for (const i in output[o].inputs) {
				if (Array.isArray(output[o].inputs[i])
					&& output[o].inputs[i].length === 2
					&& !output[output[o].inputs[i][0]]) {
					delete output[o].inputs[i];
				}
			}
		}

		return { workflow, output };
	}

	#formatPromptError(error) {
		if (error == null) {
			return "(unknown error)"
		}
		else if (typeof error === "string") {
			return error;
		}
		else if (error.stack && error.message) {
			return error.toString()
		}
		else if (error.response) {
			let message = error.response.error.message;
			if (error.response.error.details)
			message += ": " + error.response.error.details;
			for (const [nodeID, nodeError] of Object.entries(error.response.node_errors)) {
			message += "\n" + nodeError.class_type + ":"
				for (const errorReason of nodeError.errors) {
					message += "\n    - " + errorReason.message + ": " + errorReason.details
				}
			}
			return message
		}
		return "(unknown error)"
	}

	#formatExecutionError(error) {
		if (error == null) {
			return "(unknown error)"
		}

		const traceback = error.traceback.join("")
		const nodeId = error.node_id
		const nodeType = error.node_type

		return `Error occurred when executing ${nodeType}:\n\n${error.exception_message}\n\n${traceback}`
	}

	async queuePrompt(number, batchCount = 1) {
		this.#queueItems.push({ number, batchCount });

		// Only have one action process the items so each one gets a unique seed correctly
		if (this.#processingQueue) {
			return;
		}

		this.#processingQueue = true;
		this.lastNodeErrors = null;

		try {
			while (this.#queueItems.length) {
				({ number, batchCount } = this.#queueItems.pop());

				for (let i = 0; i < batchCount; i++) {
					const p = await this.graphToPrompt();

					try {
						const res = await api.queuePrompt(number, p);
						this.lastNodeErrors = res.node_errors;
						if (this.lastNodeErrors.length > 0) {
							this.canvas.draw(true, true);
						}
					} catch (error) {
						const formattedError = this.#formatPromptError(error)
						this.ui.dialog.show(formattedError);
						if (error.response) {
							this.lastNodeErrors = error.response.node_errors;
							this.canvas.draw(true, true);
						}
						break;
					}

					for (const n of p.workflow.nodes) {
						const node = graph.getNodeById(n.id);
						if (node.widgets) {
							for (const widget of node.widgets) {
								// Allow widgets to run callbacks after a prompt has been queued
								// e.g. random seed after every gen
								if (widget.afterQueued) {
									widget.afterQueued();
								}
							}
						}
					}

					this.canvas.draw(true, true);
					await this.ui.queue.update();
				}
			}
		} finally {
			this.#processingQueue = false;
		}
		api.dispatchEvent(new CustomEvent("promptQueued", { detail: { number, batchCount } }));
	}

	showErrorOnFileLoad(file) {
		this.ui.dialog.show(
			$el("div", [
				$el("p", {textContent: `Unable to find workflow in ${file.name}`})
			]).outerHTML
		);
	}

	/**
	 * Loads workflow data from the specified file
	 * @param {File} file
	 */
	async handleFile(file) {
		if (file.type === "image/png") {
			const pngInfo = await getPngMetadata(file);
			if (pngInfo?.workflow) {
				await this.loadGraphData(JSON.parse(pngInfo.workflow));
			} else if (pngInfo?.prompt) {
				this.loadApiJson(JSON.parse(pngInfo.prompt));
			} else if (pngInfo?.parameters) {
				importA1111(this.graph, pngInfo.parameters);
			} else {
				this.showErrorOnFileLoad(file);
			}
		} else if (file.type === "image/webp") {
			const pngInfo = await getWebpMetadata(file);
			// Support loading workflows from that webp custom node.
			const workflow = pngInfo?.workflow || pngInfo?.Workflow;
			const prompt = pngInfo?.prompt || pngInfo?.Prompt;

			if (workflow) {
				this.loadGraphData(JSON.parse(workflow));
			} else if (prompt) {
				this.loadApiJson(JSON.parse(prompt));
			} else {
				this.showErrorOnFileLoad(file);
			}
		} else if (file.type === "application/json" || file.name?.endsWith(".json")) {
			const reader = new FileReader();
			reader.onload = async () => {
				const jsonContent = JSON.parse(reader.result);
				if (jsonContent?.templates) {
					this.loadTemplateData(jsonContent);
				} else if(this.isApiJson(jsonContent)) {
					this.loadApiJson(jsonContent);
				} else {
					await this.loadGraphData(jsonContent);
				}
			};
			reader.readAsText(file);
		} else if (file.name?.endsWith(".latent") || file.name?.endsWith(".safetensors")) {
			const info = await getLatentMetadata(file);
			if (info.workflow) {
				await this.loadGraphData(JSON.parse(info.workflow));
			} else if (info.prompt) {
				this.loadApiJson(JSON.parse(info.prompt));
			} else {
				this.showErrorOnFileLoad(file);
			}
		} else {
			this.showErrorOnFileLoad(file);
		}
	}

	isApiJson(data) {
		return Object.values(data).every((v) => v.class_type);
	}

	loadApiJson(apiData) {
		const missingNodeTypes = Object.values(apiData).filter((n) => !LiteGraph.registered_node_types[n.class_type]);
		if (missingNodeTypes.length) {
			this.showMissingNodesError(missingNodeTypes.map(t => t.class_type), false);
			return;
		}

		const ids = Object.keys(apiData);
		app.graph.clear();
		for (const id of ids) {
			const data = apiData[id];
			const node = LiteGraph.createNode(data.class_type);
			node.id = isNaN(+id) ? id : +id;
			node.title = data._meta?.title ?? node.title
			graph.add(node);
		}

		for (const id of ids) {
			const data = apiData[id];
			const node = app.graph.getNodeById(id);
			for (const input in data.inputs ?? {}) {
				const value = data.inputs[input];
				if (value instanceof Array) {
					const [fromId, fromSlot] = value;
					const fromNode = app.graph.getNodeById(fromId);
					let toSlot = node.inputs?.findIndex((inp) => inp.name === input);
					if (toSlot == null || toSlot === -1) {
						try {
							// Target has no matching input, most likely a converted widget
							const widget = node.widgets?.find((w) => w.name === input);
							if (widget && node.convertWidgetToInput?.(widget)) {
								toSlot = node.inputs?.length - 1;
							}
						} catch (error) {}
					}
					if (toSlot != null || toSlot !== -1) {
						fromNode.connect(fromSlot, node, toSlot);
					}
				} else {
					const widget = node.widgets?.find((w) => w.name === input);
					if (widget) {
						widget.value = value;
						widget.callback?.(value);
					}
				}
			}
		}

		app.graph.arrange();
	}

	/**
	 * Registers a Comfy web extension with the app
	 * @param {ComfyExtension} extension
	 */
	registerExtension(extension) {
		if (!extension.name) {
			throw new Error("Extensions must have a 'name' property.");
		}
		if (this.extensions.find((ext) => ext.name === extension.name)) {
			throw new Error(`Extension named '${extension.name}' already registered.`);
		}
		this.extensions.push(extension);
	}

	/**
	 * Refresh combo list on whole nodes
	 */
	async refreshComboInNodes() {
		const defs = await api.getNodeDefs();

		for (const nodeId in defs) {
			this.registerNodeDef(nodeId, defs[nodeId]);
		}

		for(let nodeNum in this.graph._nodes) {
			const node = this.graph._nodes[nodeNum];
			const def = defs[node.type];

			// Allow primitive nodes to handle refresh
			node.refreshComboInNode?.(defs);

			if(!def)
				continue;

			for(const widgetNum in node.widgets) {
				const widget = node.widgets[widgetNum]
				if(widget.type == "combo" && def["input"]["required"][widget.name] !== undefined) {
					widget.options.values = def["input"]["required"][widget.name][0];

					if(widget.name != 'image' && !widget.options.values.includes(widget.value)) {
						widget.value = widget.options.values[0];
						widget.callback(widget.value);
					}
				}
			}
		}

		await this.#invokeExtensionsAsync("refreshComboInNodes", defs);
	}

	resetView() {
		app.canvas.ds.scale = 1;
		app.canvas.ds.offset = [0, 0]
		app.graph.setDirtyCanvas(true, true);
	}

	/**
	 * Clean current state
	 */
	clean() {
		this.nodeOutputs = {};
		this.nodePreviewImages = {}
		this.lastNodeErrors = null;
		this.lastExecutionError = null;
		this.runningNodeId = null;
	}
}

export const app = new ComfyApp();
