import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { getWidgetType } from "../../scripts/widgets.js";

export const IS_GROUP_NODE = Symbol();
export const GROUP_DATA = Symbol();
const GROUP_SLOTS = Symbol();

export async function registerGroupNodes(groupNodes, source, prefix, missingNodeTypes) {
	if (!groupNodes) return;

	let extra = app.graph.extra;
	if (!extra) app.graph.extra = extra = {};
	let nodes = extra.groupNodes;
	if (!nodes) extra.groupNodes = nodes = {};

	for (const g in groupNodes) {
		const groupData = groupNodes[g];

		let hasMissing = false;
		for (const n of groupData.nodes) {
			// Find missing node types
			if (!(n.type in LiteGraph.registered_node_types)) {
				missingNodeTypes.push(n.type);
				hasMissing = true;
			}
		}

		if (hasMissing) continue;

		const def = buildNodeDef(groupData, g, globalDefs, source);
		if (prefix) {
			def.display_name = prefix + "/" + def.display_name;
		}
		await app.registerNodeDef(source + "/" + g, def);

		nodes[g] = groupNodes[g];
	}
}

function getLinks(config) {
	const linksFrom = {};
	const linksTo = {};

	// Extract links for easy lookup
	for (const l of config.links) {
		const [sourceNodeId, sourceNodeSlot, targetNodeId, targetNodeSlot] = l;

		// Skip links outside the copy config
		if (sourceNodeId == null) continue;

		if (!linksFrom[sourceNodeId]) {
			linksFrom[sourceNodeId] = {};
		}
		linksFrom[sourceNodeId][sourceNodeSlot] = l;

		if (!linksTo[targetNodeId]) {
			linksTo[targetNodeId] = {};
		}
		linksTo[targetNodeId][targetNodeSlot] = l;
	}
	return { linksTo, linksFrom };
}

function buildNodeDef(config, nodeName, defs, source = "workflow") {
	const slots = {
		inputs: {},
		widgets: {},
		outputs: {},
	};

	const newDef = {
		output: [],
		output_name: [],
		output_is_list: [],
		name: nodeName,
		display_name: nodeName,
		category: "group nodes" + ("/" + source),
		input: { required: {} },

		[IS_GROUP_NODE]: true,
		[GROUP_DATA]: config,
		[GROUP_SLOTS]: slots,
	};
	const links = getLinks(config);

	const seenInputs = {};
	const seenOutputs = {};

	let inputCount = 0;
	for (let nodeId = 0; nodeId < config.nodes.length; nodeId++) {
		const node = config.nodes[nodeId];
		let def = defs[node.type];

		const linksTo = links.linksTo[nodeId];
		const linksFrom = links.linksFrom[nodeId];

		if (!def) {
			// Special handling for reroutes to allow them to be used as inputs/outputs
			if (node.type === "Reroute") {
				if (linksTo && linksFrom) {
					// Being used internally
					continue;
				}

				let rerouteType = "*";
				if (linksFrom) {
					const [, , id, slot] = linksFrom["0"];
					rerouteType = config.nodes[id].inputs[slot].type;
				} else if (linksTo) {
					const [id, slot] = linksTo["0"];
					rerouteType = config.nodes[id].outputs[slot].type;
				} else {
					// Reroute used as a pipe
					for (const l of config.links) {
						if (l[2] === nodeId) {
							rerouteType = l[5];
							break;
						}
					}
				}

				def = {
					input: {
						required: {
							[rerouteType]: [rerouteType, {}],
						},
					},
					output: [rerouteType],
					output_name: [],
					output_is_list: [],
				};
			} else {
				// Front end only node
				console.warn("Skipping virtual node " + node.type + " when building group node " + nodeName);
				continue;
			}
		}

		const inputs = { ...def.input?.required, ...def.input?.optional };

		// Add inputs / widgets
		const inputNames = Object.keys(inputs);
		let linkInputId = 0;
		for (const inputName of inputNames) {
			const widgetType = getWidgetType(inputs[inputName], inputName);
			let prefix = node.title ?? node.type;
			let name = `${prefix} ${inputName}`;

			if (name in seenInputs) {
				prefix = `${node.title ?? node.type} ${++seenInputs[name]}`;
				name = `${prefix} ${inputName}`;
			} else {
				seenInputs[name] = 1;
			}

			if (widgetType) {
				// Store mapping to get a group widget name from an inner id + name
				if (!slots.widgets[nodeId]) slots.widgets[nodeId] = {};
				slots.widgets[nodeId][inputName] = name;
			} else {
				if (linksTo?.[linkInputId]) {
					linkInputId++;
					continue;
				}

				// Store a mapping to let us get the group node input for a specific slot on an inner node
				if (!slots.inputs[nodeId]) slots.inputs[nodeId] = {};
				slots.inputs[nodeId][linkInputId++] = inputCount++;
			}

			let inputDef = inputs[inputName];
			if (inputName === "seed" || inputName === "noise_seed") {
				inputDef = [...inputDef];
				inputDef[1] = { control_after_generate: `${prefix} control_after_generate`, ...inputDef[1] };
			}
			newDef.input.required[name] = inputDef;
		}

		// Add outputs
		for (let outputId = 0; outputId < def.output.length; outputId++) {
			if (linksFrom?.[outputId]) {
				continue;
			}

			slots.outputs[newDef.output.length] = {
				node: nodeId,
				slot: outputId,
			};

			newDef.output.push(def.output[outputId]);
			newDef.output_is_list.push(def.output_is_list[outputId]);

			const label = def.output_name?.[outputId] ?? def.output[outputId];
			let name = `${node.title ?? node.type} ${label}`;

			if (name in seenOutputs) {
				name = `${node.title ?? node.type} ${++seenOutputs[name]} ${label}`;
			} else {
				seenOutputs[name] = 1;
			}

			newDef.output_name.push(name);
		}
	}

	return newDef;
}

class ConvertToGroupAction {
	getName() {
		const name = prompt("Enter group name");
		if (!name) return;

		const nodeId = "workflow/" + name;

		if (app.graph.extra?.groupNodes?.[name]) {
			if (app.graph._nodes.find((n) => n.type === nodeId)) {
				alert(
					"An in use group node with this name already exists embedded in this workflow, please remove any instances or use a new name."
				);
				return;
			} else if (
				!confirm(
					"An group node with this name already exists embedded in this workflow, are you sure you want to overwrite it?"
				)
			) {
				return;
			}
		}

		return name;
	}

	async register(name) {
		// Use the built in copyToClipboard function to generate the node data we need
		const backup = localStorage.getItem("litegrapheditor_clipboard");
		app.canvas.copyToClipboard();
		const config = JSON.parse(localStorage.getItem("litegrapheditor_clipboard"));
		localStorage.setItem("litegrapheditor_clipboard", backup);

		// Store link types to allow reconstructing reroute types
		for (const link of config.links) {
			const origin = app.graph.getNodeById(link[4]);
			const type = origin.outputs[link[1]].type;
			link.push(type);
		}

		const def = buildNodeDef(config, name, globalDefs);
		await app.registerNodeDef("workflow/" + name, def);
		return { config, def };
	}

	findOutput(slots, link, index) {
		const outputMap = slots.outputs;
		for (const k in outputMap) {
			const o = outputMap[k];
			if (o.node === index && o.slot === link.origin_slot) {
				return +k;
			}
		}
	}

	linkOutputs(newNode, node, slots, index) {
		if (!node.outputs) return;
		for (const output of node.outputs) {
			if (!output.links) continue;
			// Clone the links as they'll be changed if we reconnect
			const links = [...output.links];
			for (const l of links) {
				const link = app.graph.links[l];
				if (!link) continue;

				const targetNode = app.graph.getNodeById(link.target_id);
				const slot = this.findOutput(slots, link, index);
				if (slot != null) {
					newNode.connect(slot, targetNode, link.target_slot);
				}
			}
		}
	}

	linkInputs(newNode, config, slots) {
		for (const link of config.links ?? []) {
			const [, originSlot, targetId, targetSlot, actualOriginId] = link;
			const originNode = app.graph.getNodeById(actualOriginId);
			if (!originNode) continue; // this node is in the group
			originNode.connect(originSlot, newNode.id, slots.inputs[targetId][targetSlot]);
		}
	}

	convert(name, config, def) {
		const newNode = LiteGraph.createNode("workflow/" + name);
		app.graph.add(newNode);

		let top;
		let left;
		let index = 0;
		const slots = def[GROUP_SLOTS];
		for (const id in app.canvas.selected_nodes) {
			const node = app.graph.getNodeById(id);
			if (left == null || node.pos[0] < left) {
				left = node.pos[0];
			}
			if (top == null || node.pos[1] < top) {
				top = node.pos[1];
			}

			this.linkOutputs(newNode, node, slots, index++);

			app.graph.remove(node);
		}

		this.linkInputs(newNode, config, slots);

		newNode.pos = [left, top];
		return newNode;
	}

	addOption(options, index) {
		options.splice(index + 1, null, {
			content: `Convert to Group Node`,
			disabled:
				Object.keys(app.canvas.selected_nodes || {}).length < 2 ||
				Object.values(app.canvas.selected_nodes).find((n) => n.constructor.nodeData?.[IS_GROUP_NODE]),
			callback: async () => {
				const name = this.getName();
				if (!name) return;

				let extra = app.graph.extra;
				if (!extra) app.graph.extra = extra = {};
				let groupNodes = extra.groupNodes;
				if (!groupNodes) extra.groupNodes = groupNodes = {};

				const { config, def } = await this.register(name);
				groupNodes[name] = config;

				return this.convert(name, config, def);
			},
		});
	}
}

const id = "Comfy.GroupNode";
let globalDefs;
const ext = {
	name: id,
	setup() {
		const getCanvasMenuOptions = LGraphCanvas.prototype.getCanvasMenuOptions;
		LGraphCanvas.prototype.getCanvasMenuOptions = function () {
			const options = getCanvasMenuOptions.apply(this, arguments);
			new ConvertToGroupAction().addOption(options, options.length);
			return options;
		};

		api.addEventListener("executing", ({ detail }) => {
			if (detail) {
				const node = app.graph.getNodeById(detail);
				if (!node) {
					const split = detail.split(":");
					if (split.length === 2) {
						const outerNode = app.graph.getNodeById(+split[0]);
						if (outerNode?.constructor.nodeData?.[IS_GROUP_NODE]) {
							outerNode.runningInternalNodeId = +split[1];
							api.dispatchEvent(new CustomEvent("executing", { detail: split[0] }));
						}
					}
				}
			}
		});

		api.addEventListener("executed", ({ detail }) => {
			const node = app.graph.getNodeById(detail.node);
			if (!node) {
				const split = detail.node.split(":");
				if (split.length === 2) {
					const outerNode = app.graph.getNodeById(+split[0]);
					if (outerNode?.constructor.nodeData?.[IS_GROUP_NODE]) {
						outerNode.runningInternalNodeId = null;
						api.dispatchEvent(
							new CustomEvent("executed", { detail: { ...detail, node: split[0], merge: !outerNode.resetExecution } })
						);
						outerNode.resetExecution = false;
					}
				}
			}
		});

		// Attach handlers after everything is registered to ensure all nodes are found
		for (const k in LiteGraph.registered_node_types) {
			const nodeType = LiteGraph.registered_node_types[k];

			if (nodeType.nodeData?.[IS_GROUP_NODE]) {
				continue;
			}

			const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
			nodeType.prototype.getExtraMenuOptions = function (_, options) {
				const r = getExtraMenuOptions?.apply?.(this, arguments);

				let i = options.findIndex((o) => o.content === "Outputs");
				if (i === -1) i = options.length;
				else i++;

				new ConvertToGroupAction().addOption(options, i);

				return r;
			};
		}
	},
	async beforeConfigureGraph(graphData, missingNodeTypes) {
		registerGroupNodes(graphData?.extra?.groupNodes, "workflow", undefined, missingNodeTypes);
	},
	addCustomNodeDefs(defs) {
		globalDefs = defs;
	},
	nodeCreated(node) {
		const def = node.constructor.nodeData;
		if (def?.[IS_GROUP_NODE]) {
			const config = def[GROUP_DATA];
			const slots = def[GROUP_SLOTS];

			const onExecutionStart = node.onExecutionStart;
			node.onExecutionStart = function () {
				node.resetExecution = true;
				return onExecutionStart?.apply(this, arguments);
			};

			// Draw custom collapse icon to identity this as a group
			const onDrawTitleBox = node.onDrawTitleBox;
			node.onDrawTitleBox = function (ctx, height, size, scale) {
				onDrawTitleBox?.apply(this, arguments);

				const fill = ctx.fillStyle;
				ctx.beginPath();
				ctx.rect(11, -height + 11, 2, 2);
				ctx.rect(14, -height + 11, 2, 2);
				ctx.rect(17, -height + 11, 2, 2);
				ctx.rect(11, -height + 14, 2, 2);
				ctx.rect(14, -height + 14, 2, 2);
				ctx.rect(17, -height + 14, 2, 2);
				ctx.rect(11, -height + 17, 2, 2);
				ctx.rect(14, -height + 17, 2, 2);
				ctx.rect(17, -height + 17, 2, 2);

				ctx.fillStyle = node.boxcolor || LiteGraph.NODE_DEFAULT_BOXCOLOR;
				ctx.fill();
				ctx.fillStyle = fill;
			};

			const onDrawForeground = node.onDrawForeground;
			node.onDrawForeground = function (ctx) {
				const r = onDrawForeground?.apply?.(this, arguments);
				if (+app.runningNodeId === this.id && this.runningInternalNodeId !== null) {
					const n = config.nodes[this.runningInternalNodeId];
					const message = `Running ${n.title || n.type}`;
					ctx.save();
					ctx.font = "12px sans-serif";
					const sz = ctx.measureText(message);
					ctx.fillStyle = node.boxcolor || LiteGraph.NODE_DEFAULT_BOXCOLOR;
					ctx.beginPath();
					ctx.roundRect(0, -LiteGraph.NODE_TITLE_HEIGHT - 20, sz.width + 12, 20, 5);
					ctx.fill();

					ctx.fillStyle = "#fff";
					ctx.fillText(message, 6, -LiteGraph.NODE_TITLE_HEIGHT - 6);
					ctx.restore();
				}
				return r;
			};

			const onNodeCreated = node.onNodeCreated;
			node.onNodeCreated = function () {
				for (let innerNodeId = 0; innerNodeId < config.nodes.length; innerNodeId++) {
					const values = config.nodes[innerNodeId].widgets_values;
					if (!values) continue;
					const widgets = slots.widgets?.[innerNodeId];
					if (!widgets) continue;

					const names = Object.values(widgets);
					let seedShift = 0;
					for (let i = 0; i < names.length; i++) {
						if (values[i + seedShift] == null) continue;
						const widgetIndex = this.widgets.findIndex((w) => w.name === names[i]);
						if (widgetIndex > -1) {
							this.widgets[widgetIndex].value = values[i + seedShift];
						}

						// We need to shift the value lookup for the widget values if its a seed
						if (
							names[i] === "seed" ||
							names[i] === "noise_seed" ||
							def.input.required[names[i]]?.[1]?.control_after_generate
						) {
							seedShift++;
							// As this is a seed we need to populate control_after_generate, which will be the next widget
							this.widgets[widgetIndex + 1].value = values[i + seedShift];
						}
					}
				}

				return onNodeCreated?.apply(this, arguments);
			};

			const getExtraMenuOptions = node.getExtraMenuOptions ?? node.prototype.getExtraMenuOptions;
			node.getExtraMenuOptions = function (_, options) {
				let i = options.findIndex((o) => o.content === "Outputs");
				if (i === -1) i = options.length;
				else i++;

				options.splice(i, 0, null, {
					content: "Convert to nodes",
					callback: () => {
						const backup = localStorage.getItem("litegrapheditor_clipboard");
						localStorage.setItem("litegrapheditor_clipboard", JSON.stringify(config));
						app.canvas.pasteFromClipboard();
						localStorage.setItem("litegrapheditor_clipboard", backup);

						// Calculate position shift
						const [x, y] = this.pos;
						let top;
						let left;
						const selectedIds = Object.keys(app.canvas.selected_nodes);
						const newNodes = [];
						for (const id of selectedIds) {
							const newNode = app.graph.getNodeById(id);
							newNodes.push(newNode);
							if (left == null || newNode.pos[0] < left) {
								left = newNode.pos[0];
							}
							if (top == null || newNode.pos[1] < top) {
								top = newNode.pos[1];
							}
						}

						// Shift each node
						for (const id in app.canvas.selected_nodes) {
							const newNode = app.graph.getNodeById(id);
							newNode.pos = [newNode.pos[0] - (left - x), newNode.pos[1] - (top - y)];
						}

						// Reconnect inputs
						const slots = def[GROUP_SLOTS];
						for (const nodeIndex in slots.inputs) {
							const id = selectedIds[nodeIndex];
							const newNode = app.graph.getNodeById(id);
							for (const inputId in slots.inputs[nodeIndex]) {
								const outerSlotId = slots.inputs[nodeIndex][inputId];
								if (outerSlotId == null) continue;
								const slot = node.inputs[outerSlotId];
								if (slot.link == null) continue;
								const link = app.graph.links[slot.link];
								//  connect this node output to the input of another node
								const originNode = app.graph.getNodeById(link.origin_id);
								originNode.connect(link.origin_slot, newNode, +inputId);
							}
						}

						// Reconnect outputs
						for (let outputId = 0; outputId < node.outputs?.length; outputId++) {
							const output = node.outputs[outputId];
							if (!output.links) continue;
							const links = [...output.links];
							for (const l of links) {
								const slot = slots.outputs[outputId];
								const link = app.graph.links[l];
								const targetNode = app.graph.getNodeById(link.target_id);
								const newNode = app.graph.getNodeById(selectedIds[slot.node]);
								newNode.connect(slot.slot, targetNode, link.target_slot);
							}
						}

						app.graph.remove(this);
						return newNodes;
					},
				});

				return getExtraMenuOptions?.apply(this, arguments);
			};

			node.updateLink = function (link) {
				// Replace the group node reference with the internal node
				link = { ...link };
				const output = slots.outputs[link.origin_slot];
				let innerNode = this.innerNodes[output.node];
				let l;
				while (innerNode.type === "Reroute") {
					l = innerNode.getInputLink(0);
					innerNode = innerNode.getInputNode(0);
				}

				link.origin_id = innerNode.id;
				link.origin_slot = l?.origin_slot ?? output.slot;
				return link;
			};

			node.getInnerNodes = function () {
				const links = getLinks(config);

				const innerNodes = config.nodes.map((n, i) => {
					const innerNode = LiteGraph.createNode(n.type);
					innerNode.configure(n);

					for (const innerWidget of innerNode.widgets ?? []) {
						const groupWidgetName = slots.widgets[i]?.[innerWidget.name];
						if (!groupWidgetName) continue;
						const groupWidget = node.widgets.find((w) => w.name === groupWidgetName);
						if (groupWidget) {
							innerWidget.value = groupWidget.value;
						}
					}

					innerNode.id = node.id + ":" + i;
					innerNode.getInputNode = function (slot) {
						if (!innerNode.comfyClass) slot = 0;
						const outerSlot = slots.inputs?.[i]?.[slot];
						if (outerSlot != null) {
							// Our inner node has a mapping to the group node inputs
							// return the input node from there
							const inputNode = node.getInputNode(outerSlot);
							return inputNode;
						}

						// Internal link
						const innerLink = links.linksTo[i]?.[slot];
						if (!innerLink) return null;

						const inputNode = innerNodes[innerLink[0]];
						return inputNode;
					};
					innerNode.getInputLink = function (slot) {
						const outerSlot = slots.inputs[i]?.[slot];
						if (outerSlot != null) {
							// The inner node is connected via the group node inputs
							const linkId = node.inputs[outerSlot].link;
							let link = app.graph.links[linkId];

							// Use the outer link, but update the target to the inner node
							link = {
								target_id: innerNode.id,
								target_slot: slot,
								...link,
							};
							return link;
						}

						let link = links.linksTo[i]?.[slot];
						if (!link) return null;
						// Use the inner link, but update the origin node to be inner node id
						link = {
							origin_id: node.id + ":" + link[0],
							origin_slot: link[1],
							target_id: node.id + ":" + i,
							target_slot: slot,
						};

						return link;
					};

					return innerNode;
				});

				this.innerNodes = innerNodes;

				return innerNodes;
			};
		}
	},
};

app.registerExtension(ext);
