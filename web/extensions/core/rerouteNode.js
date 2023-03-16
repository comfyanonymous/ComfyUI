import { app } from "../../scripts/app.js";

// Node that allows you to redirect connections for cleaner graphs

app.registerExtension({
	name: "Comfy.RerouteNode",
	registerCustomNodes() {
		class RerouteNode {
			constructor() {
				if (!this.properties) {
					this.properties = {};
				}
				this.properties.showOutputText = RerouteNode.defaultVisibility;

				this.addInput("", "*");
				this.addOutput(this.properties.showOutputText ? "*" : "", "*");

				this.onConnectionsChange = function (type, index, connected, link_info) {
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
								// Move the previous node
								currentNode = node;
							} else {
								// We've found the end
								inputNode = currentNode;
								inputType = node.outputs[link.origin_slot].type;
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
									const nodeOutType = node.inputs[link.target_slot].type;
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
						node.outputs[0].name = node.properties.showOutputText ? displayType : "";
						node.size = node.computeSize();

						for (const l of node.outputs[0].links || []) {
							app.graph.links[l].color = color;
						}
					}

					if (inputNode) {
						app.graph.links[inputNode.inputs[0].link].color = color;
					}
				};

				this.clone = function () {
					const cloned = RerouteNode.prototype.clone.apply(this);
					cloned.removeOutput(0);
					cloned.addOutput(this.properties.showOutputText ? "*" : "", "*");
					cloned.size = cloned.computeSize();
					return cloned;
				};

				// This node is purely frontend and does not impact the resulting prompt so should not be serialized
				this.isVirtualNode = true;
			}

			getExtraMenuOptions(_, options) {
				options.unshift(
					{
						content: (this.properties.showOutputText ? "Hide" : "Show") + " Type",
						callback: () => {
							this.properties.showOutputText = !this.properties.showOutputText;
							if (this.properties.showOutputText) {
								this.outputs[0].name = this.__outputType || this.outputs[0].type;
							} else {
								this.outputs[0].name = "";
							}
							this.size = this.computeSize();
							app.graph.setDirtyCanvas(true, true);
						},
					},
					{
						content: (RerouteNode.defaultVisibility ? "Hide" : "Show") + " Type By Default",
						callback: () => {
							RerouteNode.setDefaultTextVisibility(!RerouteNode.defaultVisibility);
						},
					}
				);
			}

			computeSize() {
				return [
					this.properties.showOutputText && this.outputs && this.outputs.length
						? Math.max(55, LiteGraph.NODE_TEXT_SIZE * this.outputs[0].name.length * 0.6 + 40)
						: 55,
					26,
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
});
