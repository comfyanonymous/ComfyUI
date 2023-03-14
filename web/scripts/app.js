import { ComfyWidgets } from "./widgets.js";
import { ComfyUI } from "./ui.js";
import { api } from "./api.js";
import { defaultGraph } from "./defaultGraph.js";
import { getPngMetadata } from "./pnginfo.js";

class ComfyApp {
	constructor() {
		this.ui = new ComfyUI(this);
		this.extensions = [];
		this.nodeOutputs = {};
	}

	/**
	 * Invoke an extension callback
	 * @param {string} method The extension callback to execute
	 * @param  {...any} args Any arguments to pass to the callback
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

	/**
	 * Adds special context menu handling for nodes
	 * e.g. this adds Open Image functionality for nodes that show images
	 * @param {*} node The node to add the menu handler
	 */
	#addNodeContextMenuHandler(node) {
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
					options.unshift({
						content: "Open Image",
						callback: () => window.open(img.src, "_blank"),
					});
				}
			}
		};
	}

	/**
	 * Adds Custom drawing logic for nodes
	 * e.g. Draws images and handles thumbnail navigation on nodes that output images
	 * @param {*} node The node to add the draw handler
	 */
	#addDrawBackgroundHandler(node) {
		const app = this;
		node.prototype.onDrawBackground = function (ctx) {
			if (!this.flags.collapsed) {
				const output = app.nodeOutputs[this.id + ""];
				if (output && output.images) {
					if (this.images !== output.images) {
						this.images = output.images;
						this.imgs = null;
						this.imageIndex = null;
						Promise.all(
							output.images.map((src) => {
								return new Promise((r) => {
									const img = new Image();
									img.onload = () => r(img);
									img.onerror = () => r(null);
									img.src = "/view/" + src;
								});
							})
						).then((imgs) => {
							if (this.images === output.images) {
								this.imgs = imgs.filter(Boolean);
								if (this.size[1] < 100) {
									this.size[1] = 250;
								}
								app.graph.setDirtyCanvas(true);
							}
						});
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
					let shiftY = this.type === "SaveImage" ? 55 : this.imageOffset || 0;
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
			if (n && n.onDragDrop && await n.onDragDrop(event)) {
				return;
			}

			await this.handleFile(event.dataTransfer.files[0]);
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
	 * Adds a handler on paste that extracts and loads workflows from pasted JSON data
	 */
	#addPasteHandler() {
		document.addEventListener("paste", (e) => {
			let data = (e.clipboardData || window.clipboardData).getData("text/plain");
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
				this.loadGraphData(workflow);
			}
		});
	}

	/**
	 * Draws currently node highlights and progress bar
	 */
	#addDrawNodeHandler() {
		const orig = LGraphCanvas.prototype.drawNodeShape;
		const self = this;
		LGraphCanvas.prototype.drawNodeShape = function (node, ctx, size, fgcolor, bgcolor, selected, mouse_over) {
			const res = orig.apply(this, arguments);

			let color = null;
			if (node.id === +self.runningNodeId) {
				color = "#0f0";
			} else if (self.dragOverNode && node.id === self.dragOverNode.id) {
				color = "dodgerblue";
			}

			if (color) {
				const shape = node._shape || node.constructor.shape || LiteGraph.ROUND_SHAPE;
				ctx.lineWidth = 1;
				ctx.globalAlpha = 0.8;
				ctx.beginPath();
				if (shape == LiteGraph.BOX_SHAPE)
					ctx.rect(-6, -6 + LiteGraph.NODE_TITLE_HEIGHT, 12 + size[0] + 1, 12 + size[1] + LiteGraph.NODE_TITLE_HEIGHT);
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
						-6 + LiteGraph.NODE_TITLE_HEIGHT,
						12 + size[0] + 1,
						12 + size[1] + LiteGraph.NODE_TITLE_HEIGHT,
						this.round_radius * 2,
						2
					);
				else if (shape == LiteGraph.CIRCLE_SHAPE)
					ctx.arc(size[0] * 0.5, size[1] * 0.5, size[0] * 0.5 + 6, 0, Math.PI * 2);
				ctx.strokeStyle = color;
				ctx.stroke();
				ctx.strokeStyle = fgcolor;
				ctx.globalAlpha = 1;

				if (self.progress) {
					ctx.fillStyle = "green";
					ctx.fillRect(0, 0, size[0] * (self.progress.value / self.progress.max), 6);
					ctx.fillStyle = bgcolor;
				}
			}

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
		});

		api.addEventListener("executed", ({ detail }) => {
			this.nodeOutputs[detail.node] = detail.output;
		});

		api.init();
	}

	/**
	 * Loads all extensions from the API into the window
	 */
	async #loadExtensions() {
		const extensions = await api.getExtensions();
		for (const ext of extensions) {
			try {
				await import(ext);
			} catch (error) {
				console.error("Error loading extension", ext, error);
			}
		}
	}

	/**
	 * Set up the app on the page
	 */
	async setup() {
		await this.#loadExtensions();

		// Create and mount the LiteGraph in the DOM
		const canvasEl = (this.canvasEl = Object.assign(document.createElement("canvas"), { id: "graph-canvas" }));
		document.body.prepend(canvasEl);

		this.graph = new LGraph();
		const canvas = (this.canvas = new LGraphCanvas(canvasEl, this.graph));
		this.ctx = canvasEl.getContext("2d");

		this.graph.start();

		function resizeCanvas() {
			canvasEl.width = canvasEl.offsetWidth;
			canvasEl.height = canvasEl.offsetHeight;
			canvas.draw(true, true);
		}

		// Ensure the canvas fills the window
		resizeCanvas();
		window.addEventListener("resize", resizeCanvas);

		await this.#invokeExtensionsAsync("init");
		await this.registerNodes();

		// Load previous workflow
		let restored = false;
		try {
			const json = localStorage.getItem("workflow");
			if (json) {
				const workflow = JSON.parse(json);
				this.loadGraphData(workflow);
				restored = true;
			}
		} catch (err) {}

		// We failed to restore a workflow so load the default
		if (!restored) {
			this.loadGraphData();
		}

		// Save current workflow automatically
		setInterval(() => localStorage.setItem("workflow", JSON.stringify(this.graph.serialize())), 1000);

		this.#addDrawNodeHandler();
		this.#addApiUpdateHandlers();
		this.#addDropHandler();
		this.#addPasteHandler();

		await this.#invokeExtensionsAsync("setup");
	}

	/**
	 * Registers nodes with the graph
	 */
	async registerNodes() {
		const app = this;
		// Load node definitions from the backend
		const defs = await api.getNodeDefs();
		await this.#invokeExtensionsAsync("addCustomNodeDefs", defs);

		// Generate list of known widgets
		const widgets = Object.assign(
			{},
			ComfyWidgets,
			...(await this.#invokeExtensionsAsync("getCustomWidgets")).filter(Boolean)
		);

		// Register a node for each definition
		for (const nodeId in defs) {
			const nodeData = defs[nodeId];
			const node = Object.assign(
				function ComfyNode() {
					const inputs = nodeData["input"]["required"];
					const config = { minWidth: 1, minHeight: 1 };
					for (const inputName in inputs) {
						const inputData = inputs[inputName];
						const type = inputData[0];

						if (Array.isArray(type)) {
							// Enums e.g. latent rotation
							this.addWidget("combo", inputName, type[0], () => {}, { values: type });
						} else if (`${type}:${inputName}` in widgets) {
							// Support custom widgets by Type:Name
							Object.assign(config, widgets[`${type}:${inputName}`](this, inputName, inputData, app) || {});
						} else if (type in widgets) {
							// Standard type widgets
							Object.assign(config, widgets[type](this, inputName, inputData, app) || {});
						} else {
							// Node connection inputs
							this.addInput(inputName, type);
						}
					}

					for (const output of nodeData["output"]) {
						this.addOutput(output, output);
					}

					const s = this.computeSize();
					s[0] = Math.max(config.minWidth, s[0] * 1.5);
					s[1] = Math.max(config.minHeight, s[1]);
					this.size = s;
					this.serialize_widgets = true;

					app.#invokeExtensionsAsync("nodeCreated", this);
				},
				{
					title: nodeData.name,
					comfyClass: nodeData.name,
				}
			);
			node.prototype.comfyClass = nodeData.name;

			this.#addNodeContextMenuHandler(node);
			this.#addDrawBackgroundHandler(node, app);

			await this.#invokeExtensionsAsync("beforeRegisterNodeDef", node, nodeData);
			LiteGraph.registerNodeType(nodeId, node);
			node.category = nodeData.category;
		}

		await this.#invokeExtensionsAsync("registerCustomNodes");
	}

	/**
	 * Populates the graph with the specified workflow data
	 * @param {*} graphData A serialized graph object
	 */
	loadGraphData(graphData) {
		if (!graphData) {
			graphData = defaultGraph;
		}
		this.graph.configure(graphData);

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
				}
			}

			this.#invokeExtensions("loadedGraphNode", node);
		}
	}

	/**
	 * Converts the current graph workflow for sending to the API
	 * @returns The workflow and node links
	 */
	async graphToPrompt() {
		const workflow = this.graph.serialize();
		const output = {};
		for (const n of workflow.nodes) {
			const node = this.graph.getNodeById(n.id);

			if (node.isVirtualNode) {
				// Don't serialize frontend only nodes
				continue;
			}

			const inputs = {};
			const widgets = node.widgets;

			// Store all widget values
			if (widgets) {
				for (const i in widgets) {
					const widget = widgets[i];
					if (!widget.options || widget.options.serialize !== false) {
						inputs[widget.name] = widget.serializeValue ? await widget.serializeValue(n, i) : widget.value;
					}
				}
			}

			// Store all node links
			for (let i in node.inputs) {
				let parent = node.getInputNode(i);
				if (parent) {
					let link;
					if (parent.isVirtualNode) {
						// Follow the path of virtual nodes until we reach the first real one
						while (parent != null) {
							link = parent.getInputLink(0);
							if (link) {
								const from = graph.getNodeById(link.origin_id);
								if (from.isVirtualNode) {
									parent = from;
								} else {
									parent = null;
								}
							} else {
								parent = null;
							}
						}
					} else {
						link = node.getInputLink(i);
					}

					if (link) {
						inputs[node.inputs[i].name] = [String(link.origin_id), parseInt(link.origin_slot)];
					}
				}
			}

			output[String(node.id)] = {
				inputs,
				class_type: node.comfyClass,
			};
		}

		return { workflow, output };
	}

	async queuePrompt(number) {
		const p = await this.graphToPrompt();

		try {
			await api.queuePrompt(number, p);
		} catch (error) {
			this.ui.dialog.show(error.response || error.toString());
			return;
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

	/**
	 * Loads workflow data from the specified file
	 * @param {File} file
	 */
	async handleFile(file) {
		if (file.type === "image/png") {
			const pngInfo = await getPngMetadata(file);
			if (pngInfo && pngInfo.workflow) {
				this.loadGraphData(JSON.parse(pngInfo.workflow));
			}
		} else if (file.type === "application/json" || file.name.endsWith(".json")) {
			const reader = new FileReader();
			reader.onload = () => {
				this.loadGraphData(JSON.parse(reader.result));
			};
			reader.readAsText(file);
		}
	}

	registerExtension(extension) {
		if (!extension.name) {
			throw new Error("Extensions must have a 'name' property.");
		}
		if (this.extensions.find((ext) => ext.name === extension.name)) {
			throw new Error(`Extension named '${extension.name}' already registered.`);
		}
		this.extensions.push(extension);
	}
}

export const app = new ComfyApp();
