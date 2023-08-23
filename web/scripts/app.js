import { ComfyLogging } from "./logging.js";
import { ComfyWidgets } from "./widgets.js";
import { ComfyUI, $el } from "./ui.js";
import { api } from "./api.js";
import { defaultGraph } from "./defaultGraph.js";
import { getPngMetadata, importA1111, getLatentMetadata } from "./pnginfo.js";
import { LiteGraph } from "../lib/litegraph.core.js"
import ComfyGraph from "./graph.js";
import ComfyGraphCanvas from "./graphCanvas.js";
import { ComfyBackendNode } from "./graphNode.js";

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
			this.nodeOutputs[detail.node] = detail.output;
			const node = this.graph.getNodeById(detail.node);
			if (node) {
				if (node.onExecuted)
					node.onExecuted(detail.output);
			}
		});

		api.addEventListener("execution_start", ({ detail }) => {
			this.runningNodeId = null;
			this.lastExecutionError = null
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

	/**
	 * Set up the app on the page
	 */
	async setup() {
		await this.#loadExtensions();

		// Create and mount the LiteGraph in the DOM
		const mainCanvas = document.createElement("canvas")
		mainCanvas.style.touchAction = "none"
		const canvasEl = (this.canvasEl = Object.assign(mainCanvas, { id: "graph-canvas" }));
		canvasEl.tabIndex = "1";
		document.body.prepend(canvasEl);

		this.graph = new ComfyGraph(this);
		const canvas = (this.canvas = new ComfyGraphCanvas(canvasEl, this.graph, this));
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

		// Load previous workflow
		let restored = false;
		try {
			const json = localStorage.getItem("workflow");
			if (json) {
				const workflow = JSON.parse(json);
				this.loadGraphData(workflow);
				restored = true;
			}
		} catch (err) {
			console.error("Error loading previous workflow", err);
		}

		// We failed to restore a workflow so load the default
		if (!restored) {
			this.loadGraphData();
		}

		// Save current workflow automatically
		setInterval(() => localStorage.setItem("workflow", JSON.stringify(this.graph.serialize())), 1000);

		this.#addApiUpdateHandlers();
		this.#addDropHandler();
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

		// Hide litegraph Subgraph from menu until it's implemented
		LiteGraph.registered_node_types["graph/subgraph"].hide_in_node_lists = true;
	}

    async registerNodesFromDefs(defs) {
		await this.#invokeExtensionsAsync("addCustomNodeDefs", defs);

		// // Generate list of known widgets
		// const widgets = Object.assign(
		// 	{},
		// 	ComfyWidgets,
		// 	...(await this.#invokeExtensionsAsync("getCustomWidgets")).filter(Boolean)
		// );

		// Generate list of known widgets
		const customWidgets = (await app.#invokeExtensionsAsync("getCustomWidgets")).filter(Boolean);
        ComfyWidgets.customWidgets = customWidgets;

		// Register a node for each definition
		for (const nodeId in defs) {
			const nodeData = defs[nodeId];

            const ctor = class extends ComfyBackendNode {
                constructor(title) {
                    super(title, nodeId, nodeData);
                }
            }

            const node = {
                class: ctor,
                title: nodeData.display_name || nodeData.name,
                desc: `ComfyNode: ${nodeId}`
            }

            node.type = nodeId;

			await this.#invokeExtensionsAsync("beforeRegisterNodeDef", node, nodeData);
			LiteGraph.registerNodeType(node);
			node.category = nodeData.category;
		}
	}

	/**
	 * Populates the graph with the specified workflow data
	 * @param {*} graphData A serialized graph object
	 */
	loadGraphData(graphData) {
		this.clean();

		let reset_invalid_values = false;
		if (!graphData) {
			graphData = structuredClone(defaultGraph);
			reset_invalid_values = true;
		}

		const missingNodeTypes = [];
		for (let n of graphData.nodes) {
			// Patch T2IAdapterLoader to ControlNetLoader since they are the same node now
			if (n.type == "T2IAdapterLoader") n.type = "ControlNetLoader";

			// Find missing node types
			if (!(n.type in LiteGraph.registered_node_types)) {
				missingNodeTypes.push(n.type);
			}
		}

		try {
			this.graph.configure(graphData);
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
			this.ui.dialog.show(
				`When loading the graph, the following node types were not found: <ul>${Array.from(new Set(missingNodeTypes)).map(
					(t) => `<li>${t}</li>`
				).join("")}</ul>Nodes that have failed to load will show as red on the graph.`
			);
			this.logging.addEntry("Comfy.App", "warn", {
				MissingNodes: missingNodeTypes,
			});
		}
	}

	/**
	 * Converts the current graph workflow for sending to the API
	 * @returns The workflow and node links
	 */
	async graphToPrompt() {
		const workflow = this.graph.serialize();
		const output = {};
		// Process nodes in order of execution
		for (const node of this.graph.computeExecutionOrder(false)) {
			const n = workflow.nodes.find((n) => n.id === node.id);

			if (node.isVirtualNode) {
				// Don't serialize frontend only nodes but let them make changes
				if (node.applyToGraph) {
					node.applyToGraph(workflow);
				}
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
						inputs[widget.name] = widget.serializeValue ? await widget.serializeValue(n, i) : widget.value;
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
									if (parent.inputs[parent_input].type === node.inputs[i].type) {
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
						inputs[node.inputs[i].name] = [String(link.origin_id), parseInt(link.origin_slot)];
					}
				}
			}

			output[String(node.id)] = {
				inputs,
				class_type: node.comfyClass,
			};
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
	}

	/**
	 * Loads workflow data from the specified file
	 * @param {File} file
	 */
	async handleFile(file) {
		if (file.type === "image/png") {
			const pngInfo = await getPngMetadata(file);
			if (pngInfo) {
				if (pngInfo.workflow) {
					this.loadGraphData(JSON.parse(pngInfo.workflow));
				} else if (pngInfo.parameters) {
					importA1111(this.graph, pngInfo.parameters);
				}
			}
		} else if (file.type === "application/json" || file.name?.endsWith(".json")) {
			const reader = new FileReader();
			reader.onload = () => {
				this.loadGraphData(JSON.parse(reader.result));
			};
			reader.readAsText(file);
		} else if (file.name?.endsWith(".latent") || file.name?.endsWith(".safetensors")) {
			const info = await getLatentMetadata(file);
			if (info.workflow) {
				this.loadGraphData(JSON.parse(info.workflow));
			}
		}
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

		for(let nodeNum in this.graph._nodes) {
			const node = this.graph._nodes[nodeNum];

			const def = defs[node.type];

			// HOTFIX: The current patch is designed to prevent the rest of the code from breaking due to primitive nodes,
			//         and additional work is needed to consider the primitive logic in the refresh logic.
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
