// @ts-check
/// <reference path="../../web/types/litegraph.d.ts" />

const NODE = Symbol();

/**
 * @typedef { import("../../web/scripts/app")["app"] } app
 * @typedef { import("../../web/types/litegraph") } LG
 * @typedef { import("../../web/types/litegraph").IWidget } IWidget
 * @typedef { import("../../web/types/litegraph").ContextMenuItem } ContextMenuItem
 * @typedef { import("../../web/types/litegraph").INodeInputSlot } INodeInputSlot
 * @typedef { InstanceType<LG["LGraphNode"]> & { widgets?: Array<IWidget> } } LGNode
 * @typedef { { [k in keyof typeof Ez["util"]]: typeof Ez["util"][k] extends (app: any, ...rest: infer A) => infer R ? (...args: A) => R : never } } EzUtils
 * @typedef { (...args: EzOutput[] | [...EzOutput[], Record<string, unknown>]) => Array<EzOutput> & { $: EzNode, node: LG["LGraphNode"]} } EzNodeFactory
 * @typedef { ReturnType<EzNode["outputs"]>[0] } EzOutput
 */

class EzInput {
	/** @type { EzNode } */
	node;
	/** @type { INodeInputSlot } */
	input;
	/** @type { number } */
	index;

	/**
	 * @param { EzNode } node
	 * @param { INodeInputSlot } input
	 * @param { number } index
	 */
	constructor(node, input, index) {
		this.node = node;
		this.input = input;
		this.index = index;
	}
}

class EzNodeMenuItem {
	/** @type { EzNode } */
	node;
	/** @type { ContextMenuItem } */
	item;

	/**
	 * @param { EzNode } node
	 * @param { ContextMenuItem } item
	 */
	constructor(node, item) {
		this.node = node;
		this.item = item;
	}

	call(selectNode = true) {
		if (!this.item?.callback) throw new Error(`Menu Item ${this.item?.content ?? "[null]"} has no callback.`);
		if (selectNode) {
			this.node.select();
		}
		this.item.callback.call(this.node.node, undefined, undefined, undefined, undefined, this.node.node);
	}
}

class EzWidget {
	/** @type { EzNode } */
	node;
	/** @type { IWidget } */
	widget;

	/**
	 * @param { EzNode } node
	 * @param { IWidget } widget
	 */
	constructor(node, widget) {
		this.node = node;
		this.widget = widget;
	}

	get value() {
		return this.widget.value;
	}

	set value(v) {
		this.widget.value = v;
	}

	get isConvertedToInput() {
		// @ts-ignore : this type is valid for converted widgets
		return this.widget.type === "converted-widget";
	}

	convertToWidget() {
		if (!this.isConvertedToInput)
			throw new Error(`Widget ${this.widget.name} cannot be converted as it is already a widget.`);
		this.node.menu[`Convert ${this.widget.name} to widget`].call();
	}

	convertToInput() {
		if (this.isConvertedToInput)
			throw new Error(`Widget ${this.widget.name} cannot be converted as it is already an input.`);
		this.node.menu[`Convert ${this.widget.name} to input`].call();
	}
}

class EzNode {
	/** @type { app } */
	app;
	/** @type { LGNode } */
	node;
	/** @type { { length: number } & Record<string, EzInput> } */
	inputs;
	/** @type { Record<string, EzWidget> } */
	widgets;
	/** @type { Record<string, EzNodeMenuItem> } */
	menu;

	/**
	 * @param { app } app
	 * @param { LGNode } node
	 */
	constructor(app, node) {
		this.app = app;
		this.node = node;

		// @ts-ignore : this proxy returns the length
		this.inputs = new Proxy(
			{},
			{
				get: (_, p) => {
					if (typeof p !== "string") throw new Error(`Invalid widget name.`);
					if (p === "length") return this.node.inputs?.length ?? 0;
					const index = this.node.inputs.findIndex((i) => i.name === p);
					if (index === -1) throw new Error(`Unknown input "${p}" on node "${this.node.type}".`);
					return new EzInput(this, this.node.inputs[index], index);
				},
			}
		);

		this.widgets = new Proxy(
			{},
			{
				get: (_, p) => {
					if (typeof p !== "string") throw new Error(`Invalid widget name.`);
					const widget = this.node.widgets?.find((w) => w.name === p);
					if (!widget) throw new Error(`Unknown widget "${p}" on node "${this.node.type}".`);

					return new EzWidget(this, widget);
				},
			}
		);

		this.menu = new Proxy(
			{},
			{
				get: (_, p) => {
					if (typeof p !== "string") throw new Error(`Invalid menu item name.`);
					const options = this.menuItems();
					const option = options.find((o) => o?.content === p);
					if (!option) throw new Error(`Unknown menu item "${p}" on node "${this.node.type}".`);

					return new EzNodeMenuItem(this, option);
				},
			}
		);
	}

	get id() {
		return this.node.id;
	}

	menuItems() {
		return this.app.canvas.getNodeMenuOptions(this.node);
	}

	outputs() {
		return (
			this.node.outputs?.map((data, index) => {
				return {
					[NODE]: this.node,
					index,
					data,
				};
			}) ?? []
		);
	}

	select() {
		this.app.canvas.selectNode(this.node);
	}
}

class EzGraph {
	/** @type { app } */
	app;

	/**
	 * @param { app } app
	 */
	constructor(app) {
		this.app = app;
	}

	get nodes() {
		return this.app.graph._nodes.map((n) => new EzNode(this.app, n));
	}

	clear() {
		this.app.graph.clear();
	}

	arrange() {
		this.app.graph.arrange();
	}

	/**
	 * @param { number | LGNode | EzNode } obj
	 * @returns { EzNode }
	 */
	find(obj) {
		let match;
		let id;
		if (typeof obj === "number") {
			id = obj;
		} else {
			id = obj.id;
		}

		match = this.app.graph.getNodeById(id);

		if (!match) {
			throw new Error(`Unable to find node with ID ${id}.`);
		}

		return new EzNode(this.app, match);
	}

	/**
	 * @returns { Promise<void> }
	 */
	reload() {
		const graph = JSON.parse(JSON.stringify(this.app.graph.serialize()));
		return new Promise((r) => {
			this.app.graph.clear();
			setTimeout(() => {
				this.app.loadGraphData(graph);
				r();
			}, 10);
		});
	}
}

export const Ez = {
	/**
	 * Quickly build and interact with a ComfyUI graph
	 * @example
	 * const { ez, graph } = Ez.graph(app);
	 * graph.clear();
	 * const [model, clip, vae] = ez.CheckpointLoaderSimple();
	 * const [pos] = ez.CLIPTextEncode(clip, { text: "positive" });
	 * const [neg] = ez.CLIPTextEncode(clip, { text: "negative" });
	 * const [latent] = ez.KSampler(model, pos, neg, ...ez.EmptyLatentImage());
	 * const [image] = ez.VAEDecode(latent, vae);
	 * const saveNode = ez.SaveImage(image).node;
	 * console.log(saveNode);
	 * graph.arrange();
	 * @param { app } app
	 * @param { LG["LiteGraph"] } LiteGraph
	 * @param { LG["LGraphCanvas"] } LGraphCanvas
	 * @param { boolean } clearGraph
	 * @returns { { graph: EzGraph, ez: Record<string, EzNodeFactory> } }
	 */
	graph(app, LiteGraph, LGraphCanvas, clearGraph = true) {
		// Always set the active canvas so things work
		LGraphCanvas.active_canvas = app.canvas;

		if (clearGraph) {
			app.graph.clear();
		}

		// @ts-ignore : this proxy handles utility methods & node creation
		const factory = new Proxy(
			{},
			{
				get(_, p) {
					if (typeof p !== "string") throw new Error("Invalid node");
					const node = LiteGraph.createNode(p);
					if (!node) throw new Error(`Unknown node "${p}"`);
					app.graph.add(node);

					/**
					 * @param {Parameters<EzNodeFactory>} args
					 */
					return function (...args) {
						const ezNode = new EzNode(app, node);

						// console.log("Created " + node.type, "Populating:", args);
						let slot = 0;
						for (let i = 0; i < args.length; i++) {
							const arg = args[i];
							if (arg[NODE]) {
								arg[NODE].connect(arg.index, node, slot++);
							} else {
								for (const k in arg) {
									ezNode.widgets[k].value = arg[k];
								}
							}
						}

						const outputs = ezNode.outputs();
						outputs["$"] = ezNode;
						outputs["node"] = node;
						return outputs;
					};
				},
			}
		);

		return { graph: new EzGraph(app), ez: factory };
	},
};
