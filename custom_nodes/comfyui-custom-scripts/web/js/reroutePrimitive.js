import { app } from "../../../scripts/app.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";

const REROUTE_PRIMITIVE = "ReroutePrimitive|pysssss";
const MULTI_PRIMITIVE = "MultiPrimitive|pysssss";
const LAST_TYPE = Symbol("LastType");

app.registerExtension({
	name: "pysssss.ReroutePrimitive",
	init() {
		// On graph configure, fire onGraphConfigured to create widgets
		const graphConfigure = LGraph.prototype.configure;
		LGraph.prototype.configure = function () {
			const r = graphConfigure.apply(this, arguments);
			for (const n of app.graph._nodes) {
				if (n.type === REROUTE_PRIMITIVE) {
					n.onGraphConfigured();
				}
			}

			return r;
		};

		// Hide this node as it is no longer supported
		const getNodeTypesCategories = LiteGraph.getNodeTypesCategories;
		LiteGraph.getNodeTypesCategories = function() {
			return getNodeTypesCategories.apply(this, arguments).filter(c => !c.startsWith("__hidden__"));
		}

		const graphToPrompt = app.graphToPrompt;
		app.graphToPrompt = async function () {
			const res = await graphToPrompt.apply(this, arguments);

			const multiOutputs = [];
			for (const nodeId in res.output) {
				const output = res.output[nodeId];
				if (output.class_type === MULTI_PRIMITIVE) {
					multiOutputs.push({ id: nodeId, inputs: output.inputs });
				}
			}

			function permute(outputs) {
				function generatePermutations(inputs, currentIndex, currentPermutation, result) {
					if (currentIndex === inputs.length) {
						result.push({ ...currentPermutation });
						return;
					}

					const input = inputs[currentIndex];

					for (const k in input) {
						currentPermutation[currentIndex] = input[k];
						generatePermutations(inputs, currentIndex + 1, currentPermutation, result);
					}
				}

				const inputs = outputs.map((output) => output.inputs);
				const result = [];
				const current = new Array(inputs.length);

				generatePermutations(inputs, 0, current, result);

				return outputs.map((output, index) => ({
					...output,
					inputs: result.reduce((p, permutation) => {
						const count = Object.keys(p).length;
						p["value" + (count || "")] = permutation[index];
						return p;
					}, {}),
				}));
			}

			const permutations = permute(multiOutputs);
			for (let i = 0; i < permutations.length; i++) {
				res.output[multiOutputs[i].id].inputs = permutations[i].inputs;
			}

			return res;
		};
	},
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		function addOutputHandler() {
			// Finds the first non reroute output node down the chain
			nodeType.prototype.getFirstReroutedOutput = function (slot) {
				if (nodeData.name === MULTI_PRIMITIVE) {
					slot = 0;
				}
				const links = this.outputs[slot].links;
				if (!links) return null;

				const search = [];
				for (const l of links) {
					const link = app.graph.links[l];
					if (!link) continue;

					const node = app.graph.getNodeById(link.target_id);
					if (node.type !== REROUTE_PRIMITIVE && node.type !== MULTI_PRIMITIVE) {
						return { node, link };
					}
					search.push({ node, link });
				}

				for (const { link, node } of search) {
					const r = node.getFirstReroutedOutput(link.target_slot);
					if (r) {
						return r;
					}
				}
			};
		}

		if (nodeData.name === REROUTE_PRIMITIVE) {
			const configure = nodeType.prototype.configure || LGraphNode.prototype.configure;
			const onConnectionsChange = nodeType.prototype.onConnectionsChange;
			const onAdded = nodeType.prototype.onAdded;

			nodeType.title_mode = LiteGraph.NO_TITLE;

			function hasAnyInput(node) {
				for (const input of node.inputs) {
					if (input.link) {
						return true;
					}
				}
				return false;
			}

			// Remove input text
			nodeType.prototype.onAdded = function () {
				onAdded?.apply(this, arguments);
				this.inputs[0].label = "";
				this.outputs[0].label = "value";
				this.setSize(this.computeSize());
			};

			// Restore any widgets
			nodeType.prototype.onGraphConfigured = function () {
				if (hasAnyInput(this)) return;

				const outputNode = this.getFirstReroutedOutput(0);
				if (outputNode) {
					this.checkPrimitiveWidget(outputNode);
				}
			};

			// Check if we need to create (or remove) a widget on the node
			nodeType.prototype.checkPrimitiveWidget = function ({ node, link }) {
				let widgetType = link.type;
				let targetLabel = widgetType;
				const input = node.inputs[link.target_slot];
				if (input.widget?.config?.[0] instanceof Array) {
					targetLabel = input.widget.name;
					widgetType = "COMBO";
				}

				if (widgetType in ComfyWidgets) {
					if (!this.widgets?.length) {
						let v;
						if (this.widgets_values?.length) {
							v = this.widgets_values[0];
						}
						let config = [link.type, {}];
						if (input.widget?.config) {
							config = input.widget.config;
						}
						const { widget } = ComfyWidgets[widgetType](this, "value", config, app);
						if (v !== undefined && (!this[LAST_TYPE] || this[LAST_TYPE] === widgetType)) {
							widget.value = v;
						}
						this[LAST_TYPE] = widgetType;
					}
				} else if (this.widgets) {
					this.widgets.length = 0;
				}

				return targetLabel;
			};

			// Finds all input nodes from the current reroute
			nodeType.prototype.getReroutedInputs = function (slot) {
				let nodes = [{ node: this }];
				let node = this;
				while (node?.type === REROUTE_PRIMITIVE) {
					const input = node.inputs[slot];
					if (input.link) {
						const link = app.graph.links[input.link];
						node = app.graph.getNodeById(link.origin_id);
						slot = link.origin_slot;
						nodes.push({
							node,
							link,
						});
					} else {
						node = null;
					}
				}

				return nodes;
			};

			addOutputHandler();

			// Update the type of all reroutes in a chain
			nodeType.prototype.changeRerouteType = function (slot, type, label) {
				const color = LGraphCanvas.link_type_colors[type];
				const output = this.outputs[slot];
				this.inputs[slot].label = " ";
				output.label = label || (type === "*" ? "value" : type);
				output.type = type;

				// Process all linked outputs
				for (const linkId of output.links || []) {
					const link = app.graph.links[linkId];
					if (!link) continue;
					link.color = color;
					const node = app.graph.getNodeById(link.target_id);
					if (node.changeRerouteType) {
						// Recursively update reroutes
						node.changeRerouteType(link.target_slot, type, label);
					} else {
						// Validate links to 'real' nodes
						const theirType = node.inputs[link.target_slot].type;
						if (theirType !== type && theirType !== "*") {
							node.disconnectInput(link.target_slot);
						}
					}
				}

				if (this.inputs[slot].link) {
					const link = app.graph.links[this.inputs[slot].link];
					if (link) link.color = color;
				}
			};

			// Override configure so we can flag that we are configuring to avoid link validation breaking
			let configuring = false;
			nodeType.prototype.configure = function () {
				configuring = true;
				const r = configure?.apply(this, arguments);
				configuring = false;

				return r;
			};

			Object.defineProperty(nodeType, "title_mode", {
				get() {
					return app.canvas.current_node?.widgets?.length ? LiteGraph.NORMAL_TITLE : LiteGraph.NO_TITLE;
				},
			});

			nodeType.prototype.onConnectionsChange = function (type, _, connected, link_info) {
				// If configuring treat everything as OK as links may not be set by litegraph yet
				if (configuring) return;

				const isInput = type === LiteGraph.INPUT;
				const slot = isInput ? link_info.target_slot : link_info.origin_slot;

				let targetLabel = null;
				let targetNode = null;
				let targetType = "*";
				let targetSlot = slot;

				const inputPath = this.getReroutedInputs(slot);
				const rootInput = inputPath[inputPath.length - 1];
				const outputNode = this.getFirstReroutedOutput(slot);
				if (rootInput.node.type === REROUTE_PRIMITIVE) {
					// Our input node is a reroute, so see if we have an output
					if (outputNode) {
						targetType = outputNode.link.type;
					} else if (rootInput.node.widgets) {
						rootInput.node.widgets.length = 0;
					}
					targetNode = rootInput;
					targetSlot = rootInput.link?.target_slot ?? slot;
				} else {
					// We have a real input, so we want to use that type
					targetNode = inputPath[inputPath.length - 2];
					targetType = rootInput.node.outputs[rootInput.link.origin_slot].type;
					targetSlot = rootInput.link.target_slot;
				}

				if (this.widgets && inputPath.length > 1) {
					// We have an input node so remove our widget
					this.widgets.length = 0;
				}

				if (outputNode && rootInput.node.checkPrimitiveWidget) {
					// We have an output, check if we need to create a widget
					targetLabel = rootInput.node.checkPrimitiveWidget(outputNode);
				}

				// Trigger an update of the type to all child nodes
				targetNode.node.changeRerouteType(targetSlot, targetType, targetLabel);

				return onConnectionsChange?.apply(this, arguments);
			};

			// When collapsed fix the size to just the dot
			const computeSize = nodeType.prototype.computeSize || LGraphNode.prototype.computeSize;
			nodeType.prototype.computeSize = function () {
				const r = computeSize.apply(this, arguments);
				if (this.flags?.collapsed) {
					return [1, 25];
				} else if (this.widgets?.length) {
					return r;
				} else {
					let w = 75;
					if (this.outputs?.[0]?.label) {
						const t = LiteGraph.NODE_TEXT_SIZE * this.outputs[0].label.length * 0.6 + 30;
						if (t > w) {
							w = t;
						}
					}
					return [w, r[1]];
				}
			};

			// On collapse shrink the node to just a dot
			const collapse = nodeType.prototype.collapse || LGraphNode.prototype.collapse;
			nodeType.prototype.collapse = function () {
				collapse.apply(this, arguments);
				this.setSize(this.computeSize());
				requestAnimationFrame(() => {
					this.setDirtyCanvas(true, true);
				});
			};

			// Shift the bounding area up slightly as LiteGraph miscalculates it for collapsed nodes
			nodeType.prototype.onBounding = function (area) {
				if (this.flags?.collapsed) {
					area[1] -= 15;
				}
			};
		} else if (nodeData.name === MULTI_PRIMITIVE) {
			addOutputHandler();
			nodeType.prototype.onConnectionsChange = function (type, _, connected, link_info) {
				for (let i = 0; i < this.inputs.length - 1; i++) {
					if (!this.inputs[i].link) {
						this.removeInput(i--);
					}
				}
				if (this.inputs[this.inputs.length - 1].link) {
					this.addInput("v" + +new Date(), this.inputs[0].type).label = "value";
				}
			};
		}
	},
});
