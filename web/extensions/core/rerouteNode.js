import { app } from "../../scripts/app.js";

// Node that allows you to redirect connections for cleaner graphs

// Context menu to change input/output orientation
function getOrientationMenu(value, options, e, menu, node) {
	const isInput = value.options.isInput
	const takenSlot = (isInput ? node.outputs[0].dir:node.inputs[0].dir) -1

	let availableDir = ["Up" ,"Down", "Left", "Right"]
	let availableValue = [LiteGraph.UP, LiteGraph.DOWN, LiteGraph.LEFT, LiteGraph.RIGHT]

	availableDir.splice(takenSlot, 1);
	availableValue.splice(takenSlot, 1);

	new LiteGraph.ContextMenu(
		availableDir,
		{
			event: e,
			parentMenu: menu,
			node: node,
			callback: (v, options, mouse_event, menu, node) => {
				if (!node) {
					return;
				}
		
				let dir = availableValue[Object.values(availableDir).indexOf(v)];
				
				if (isInput) {
					node.inputs[0].dir = dir;
				} else {
					node.outputs[0].dir = dir;
				}
		
				node.applyOrientation();
			}
		}
	);
}

app.registerExtension({
	name: "Comfy.RerouteNode",
	registerCustomNodes() {
		class RerouteNode {
			constructor() {
				if (!this.properties) {
					this.properties = {};
				}
				this.properties.showOutputText = RerouteNode.defaultVisibility;

				this.addInput("", "*", {nameLocked: true});
				this.addOutput(this.properties.showOutputText ? "*" : "", "*", {nameLocked: true});
				
				this.inputs[0].dir = LiteGraph.LEFT;
				this.outputs[0].dir = LiteGraph.RIGHT;

				this.onResize = function(_) {
					this.applyOrientation();
				}

				this.onDrawForeground = function(ctx, graphcanvas, canvas) {
					if (this.properties.showOutputText && graphcanvas.ds.scale > 0.5) {
						ctx.fillStyle = LiteGraph.NODE_TEXT_COLOR;
						ctx.font = graphcanvas.inner_text_font;
						ctx.textAlign = "center";

						ctx.fillText(this.getDisplayName(), this.size[0] / 2, this.size[1] / 2+5);
					}
				}

				this.onConfigure = function(data) {
					
					// update old reroute
					if (!this.inputs[0].dir) { this.inputs[0].dir = LiteGraph.LEFT; }
					if (!this.outputs[0].dir) { this.outputs[0].dir = LiteGraph.RIGHT; }

					if (this.inputs[0].label) { this.inputs[0].label = "" }
					if (this.outputs[0].label) { this.outputs[0].label = "" }
					
					if (!this.inputs[0].nameLocked) { this.inputs[0].nameLocked = true }
					if (!this.outputs[0].nameLocked) { this.outputs[0].nameLocked = true }

					// handle old horizontal property
					if (this.properties.horizontal) {
						this.inputs[0].dir = LiteGraph.UP;
						this.outputs[0].dir = LiteGraph.DOWN;
						delete this.properties.horizontal;
					}

					this.applyOrientation();
					app.graph.setDirtyCanvas(true, true);
				}

				this.onConnectionsChange = function (type, index, connected, link_info) {
					this.applyOrientation();

					// Prevent multiple connections to different types when we have no input
					if (connected && type === LiteGraph.OUTPUT) {
						// Ignore wildcard nodes as these will be updated to real types
						const types = new Set(this.outputs[0].links.map((l) => app.graph.links[l].type).filter((t) => t !== "*"));
						if (types.size > 1) {
							for (let i = 0; i < this.outputs[0].links.length - 1; i++) {
								const linkId = this.outputs[0].links[i];
								const link = app.graph.links[linkId];
								const node = app.graph.getNodeById(link.target_id);
								node.disconnectInput(link.target_slot);
							}
						}
					}

					// Find root input
					let currentNode = this;
					let updateNodes = [];
					let inputType = null;
					let inputNode = null;
					while (currentNode) {
						updateNodes.unshift(currentNode);
						const linkId = currentNode.inputs[0].link;
						if (linkId !== null) {
							const link = app.graph.links[linkId];
							const node = app.graph.getNodeById(link.origin_id);
							const type = node.constructor.type;
							if (type === "Reroute") {
								if (node === this) {
									// We've found a circle
									currentNode.disconnectInput(link.target_slot);
									currentNode = null;
								}
								else {
									// Move the previous node
									currentNode = node;
								}
							} else {
								// We've found the end
								inputNode = currentNode;
								inputType = node.outputs[link.origin_slot]?.type ?? null;
								break;
							}
						} else {
							// This path has no input node
							currentNode = null;
							break;
						}
					}

					// Find all outputs
					const nodes = [this];
					let outputType = null;
					while (nodes.length) {
						currentNode = nodes.pop();
						const outputs = (currentNode.outputs ? currentNode.outputs[0].links : []) || [];
						if (outputs.length) {
							for (const linkId of outputs) {
								const link = app.graph.links[linkId];

								// When disconnecting sometimes the link is still registered
								if (!link) continue;

								const node = app.graph.getNodeById(link.target_id);
								const type = node.constructor.type;

								if (type === "Reroute") {
									// Follow reroute nodes
									nodes.push(node);
									updateNodes.push(node);
								} else {
									// We've found an output
									const nodeOutType = node.inputs && node.inputs[link?.target_slot] && node.inputs[link.target_slot].type ? node.inputs[link.target_slot].type : null;
									if (inputType && nodeOutType !== inputType) {
										// The output doesnt match our input so disconnect it
										node.disconnectInput(link.target_slot);
									} else {
										outputType = nodeOutType;
									}
								}
							}
						} else {
							// No more outputs for this path
						}
					}

					const displayType = inputType || outputType || "*";
					const color = LGraphCanvas.link_type_colors[displayType];

					// Update the types of each node
					for (const node of updateNodes) {
						// If we dont have an input type we are always wildcard but we'll show the output type
						// This lets you change the output link to a different type and all nodes will update
						node.outputs[0].type = inputType || "*";
						node.__outputType = displayType;
						node.size = node.computeSize();
						node.applyOrientation();

						for (const l of node.outputs[0].links || []) {
							const link = app.graph.links[l];
							if (link) {
								link.color = color;
							}
						}
					}

					if (inputNode) {
						const link = app.graph.links[inputNode.inputs[0].link];
						if (link) {
							link.color = color;
						}
					}
				};

				this.clone = function () {
					const cloned = RerouteNode.prototype.clone.apply(this);
					let dir = this.outputs[0].dir
					cloned.removeOutput(0);
					cloned.addOutput(this.properties.showOutputText ? "*" : "", "*", {nameLocked: true});
					cloned.outputs[0].dir = dir
					cloned.size = cloned.computeSize();
					return cloned;
				};

				// This node is purely frontend and does not impact the resulting prompt so should not be serialized
				this.isVirtualNode = true;

				this.applyOrientation();
			}

			getExtraMenuOptions(_, options) {
				options.unshift(
					{
						content: (this.properties.showOutputText ? "Hide" : "Show") + " Type",
						callback: () => {
							this.properties.showOutputText = !this.properties.showOutputText;
							this.size = this.computeSize();
							this.applyOrientation();
							app.graph.setDirtyCanvas(true, true);
						},
					},
					{
						content: (RerouteNode.defaultVisibility ? "Hide" : "Show") + " Type By Default",
						callback: () => {
							RerouteNode.setDefaultTextVisibility(!RerouteNode.defaultVisibility);
						},
					},
					{
						content: "Input Orientation",
						has_submenu: true,
						options: {isInput: true},
						callback: getOrientationMenu
					},
					{
						content: "Output Orientation",
						has_submenu: true,
						options: {isInput: false},
						callback: getOrientationMenu
					},
				);
			}

			applyOrientation() {
				// Place inputs/outputs based on the direction
				function processInOut(node, slot) {
					if (!slot) { return; } // weird copy/paste fix

					const horizontal = ([LiteGraph.UP, LiteGraph.DOWN].indexOf(slot.dir) > -1);
					const reversed = ([LiteGraph.DOWN, LiteGraph.RIGHT].indexOf(slot.dir) > -1);

					if (horizontal) {
						slot.pos = [node.size[0] / 2, reversed ? node.size[1]:0];
					} else {
						slot.pos = [reversed ? node.size[0]:0, node.size[1] / 2];
					}
				}

				processInOut(this, this.inputs[0]);
				processInOut(this, this.outputs[0]);

				app.graph.setDirtyCanvas(true, true);
			}

			getDisplayName() {
				let displayName = this.__outputType;
				if (this.title !== "Reroute" && this.title !== "") {
					displayName = this.title;
				}
				return displayName;
			}

			computeSize() {
				return [
					this.properties.showOutputText && this.outputs
						? Math.max(75, LiteGraph.NODE_TEXT_SIZE * this.getDisplayName().length * 0.6)
						: 75,
					25,
				];
			}

			static setDefaultTextVisibility(visible) {
				RerouteNode.defaultVisibility = visible;
				if (visible) {
					localStorage["Comfy.RerouteNode.DefaultVisibility"] = "true";
				} else {
					delete localStorage["Comfy.RerouteNode.DefaultVisibility"];
				}
			}
		}

		// Load default visibility
		RerouteNode.setDefaultTextVisibility(!!localStorage["Comfy.RerouteNode.DefaultVisibility"]);

		LiteGraph.registerNodeType(
			"Reroute",
			Object.assign(RerouteNode, {
				title_mode: LiteGraph.NO_TITLE,
				title: "Reroute",
				collapsable: false,
			})
		);

		RerouteNode.category = "utils";
	},
	setup(app) {

		// adds "Add reroute" to right click canvas menu
		const orig = LGraphCanvas.prototype.getCanvasMenuOptions;
		LGraphCanvas.prototype.getCanvasMenuOptions = function () {
			const options = orig.apply(this, arguments);
			options.push(
				null,
				{ 
					content: "Add Reroute",
					callback: (value, options, mouse_event, menu, node) => {
						let newNode = LiteGraph.createNode("Reroute")

						newNode.pos = app.canvas.convertEventToCanvasOffset(mouse_event);
						newNode.pos[0] -= newNode.size[0]/2;
						newNode.pos[1] -= newNode.size[1]/2;

						app.graph.add(newNode);

					} 
				}
			);
			return options;
		};
	}
});
