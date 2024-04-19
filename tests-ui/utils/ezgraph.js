// @ts-check
/// <reference path="../../web/types/litegraph.d.ts" />

/**
 * @typedef { import("../../web/scripts/app")["app"] } app
 * @typedef { import("../../web/types/litegraph") } LG
 * @typedef { import("../../web/types/litegraph").IWidget } IWidget
 * @typedef { import("../../web/types/litegraph").ContextMenuItem } ContextMenuItem
 * @typedef { import("../../web/types/litegraph").INodeInputSlot } INodeInputSlot
 * @typedef { import("../../web/types/litegraph").INodeOutputSlot } INodeOutputSlot
 * @typedef { InstanceType<LG["LGraphNode"]> & { widgets?: Array<IWidget> } } LGNode
 * @typedef { (...args: EzOutput[] | [...EzOutput[], Record<string, unknown>]) => EzNode } EzNodeFactory
 */

export class EzConnection {
	/** @type { app } */
	app;
	/** @type { InstanceType<LG["LLink"]> } */
	link;

	get originNode() {
		return new EzNode(this.app, this.app.graph.getNodeById(this.link.origin_id));
	}

	get originOutput() {
		return this.originNode.outputs[this.link.origin_slot];
	}

	get targetNode() {
		return new EzNode(this.app, this.app.graph.getNodeById(this.link.target_id));
	}

	get targetInput() {
		return this.targetNode.inputs[this.link.target_slot];
	}

	/**
	 * @param { app } app
	 * @param { InstanceType<LG["LLink"]> } link
	 */
	constructor(app, link) {
		this.app = app;
		this.link = link;
	}

	disconnect() {
		this.targetInput.disconnect();
	}
}

export class EzSlot {
	/** @type { EzNode } */
	node;
	/** @type { number } */
	index;

	/**
	 * @param { EzNode } node
	 * @param { number } index
	 */
	constructor(node, index) {
		this.node = node;
		this.index = index;
	}
}

export class EzInput extends EzSlot {
	/** @type { INodeInputSlot } */
	input;

	/**
	 * @param { EzNode } node
	 * @param { number } index
	 * @param { INodeInputSlot } input
	 */
	constructor(node, index, input) {
		super(node, index);
		this.input = input;
	}

	get connection() {
		const link = this.node.node.inputs?.[this.index]?.link;
		if (link == null) {
			return null;
		}
		return new EzConnection(this.node.app, this.node.app.graph.links[link]);
	}

	disconnect() {
		this.node.node.disconnectInput(this.index);
	}
}

export class EzOutput extends EzSlot {
	/** @type { INodeOutputSlot } */
	output;

	/**
	 * @param { EzNode } node
	 * @param { number } index
	 * @param { INodeOutputSlot } output
	 */
	constructor(node, index, output) {
		super(node, index);
		this.output = output;
	}

	get connections() {
		return (this.node.node.outputs?.[this.index]?.links ?? []).map(
			(l) => new EzConnection(this.node.app, this.node.app.graph.links[l])
		);
	}

	/**
	 * @param { EzInput } input
	 */
	connectTo(input) {
		if (!input) throw new Error("Invalid input");

		/**
		 * @type { LG["LLink"] | null }
		 */
		const link = this.node.node.connect(this.index, input.node.node, input.index);
		if (!link) {
			const inp = input.input;
			const inName = inp.name || inp.label || inp.type;
			throw new Error(
				`Connecting from ${input.node.node.type}#${input.node.id}[${inName}#${input.index}] -> ${this.node.node.type}#${this.node.id}[${
					this.output.name ?? this.output.type
				}#${this.index}] failed.`
			);
		}
		return link;
	}
}

export class EzNodeMenuItem {
	/** @type { EzNode } */
	node;
	/** @type { number } */
	index;
	/** @type { ContextMenuItem } */
	item;

	/**
	 * @param { EzNode } node
	 * @param { number } index
	 * @param { ContextMenuItem } item
	 */
	constructor(node, index, item) {
		this.node = node;
		this.index = index;
		this.item = item;
	}

	call(selectNode = true) {
		if (!this.item?.callback) throw new Error(`Menu Item ${this.item?.content ?? "[null]"} has no callback.`);
		if (selectNode) {
			this.node.select();
		}
		return this.item.callback.call(this.node.node, undefined, undefined, undefined, undefined, this.node.node);
	}
}

export class EzWidget {
	/** @type { EzNode } */
	node;
	/** @type { number } */
	index;
	/** @type { IWidget } */
	widget;

	/**
	 * @param { EzNode } node
	 * @param { number } index
	 * @param { IWidget } widget
	 */
	constructor(node, index, widget) {
		this.node = node;
		this.index = index;
		this.widget = widget;
	}

	get value() {
		return this.widget.value;
	}

	set value(v) {
		this.widget.value = v;
		this.widget.callback?.call?.(this.widget, v)
	}

	get isConvertedToInput() {
		// @ts-ignore : this type is valid for converted widgets
		return this.widget.type === "converted-widget";
	}

	getConvertedInput() {
		if (!this.isConvertedToInput) throw new Error(`Widget ${this.widget.name} is not converted to input.`);

		return this.node.inputs.find((inp) => inp.input["widget"]?.name === this.widget.name);
	}

	convertToWidget() {
		if (!this.isConvertedToInput)
			throw new Error(`Widget ${this.widget.name} cannot be converted as it is already a widget.`);
		var menu = this.node.menu["Convert Input to Widget"].item.submenu.options;
		var index = menu.findIndex(a => a.content == `Convert ${this.widget.name} to widget`);
		menu[index].callback.call();
	}

	convertToInput() {
		if (this.isConvertedToInput)
			throw new Error(`Widget ${this.widget.name} cannot be converted as it is already an input.`);
		var menu = this.node.menu["Convert Widget to Input"].item.submenu.options;
		var index = menu.findIndex(a => a.content == `Convert ${this.widget.name} to input`);
		menu[index].callback.call();
	}
}

export class EzNode {
	/** @type { app } */
	app;
	/** @type { LGNode } */
	node;

	/**
	 * @param { app } app
	 * @param { LGNode } node
	 */
	constructor(app, node) {
		this.app = app;
		this.node = node;
	}

	get id() {
		return this.node.id;
	}

	get inputs() {
		return this.#makeLookupArray("inputs", "name", EzInput);
	}

	get outputs() {
		return this.#makeLookupArray("outputs", "name", EzOutput);
	}

	get widgets() {
		return this.#makeLookupArray("widgets", "name", EzWidget);
	}

	get menu() {
		return this.#makeLookupArray(() => this.app.canvas.getNodeMenuOptions(this.node), "content", EzNodeMenuItem);
	}

	get isRemoved() {
		return !this.app.graph.getNodeById(this.id);
	}

	select(addToSelection = false) {
		this.app.canvas.selectNode(this.node, addToSelection);
	}

	// /**
	//  * @template { "inputs" | "outputs" } T
	//  * @param { T } type
	//  * @returns { Record<string, type extends "inputs" ? EzInput : EzOutput> & (type extends "inputs" ? EzInput [] : EzOutput[]) }
	//  */
	// #getSlotItems(type) {
	// 	// @ts-ignore : these items are correct
	// 	return (this.node[type] ?? []).reduce((p, s, i) => {
	// 		if (s.name in p) {
	// 			throw new Error(`Unable to store input ${s.name} on array as name conflicts.`);
	// 		}
	// 		// @ts-ignore
	// 		p.push((p[s.name] = new (type === "inputs" ? EzInput : EzOutput)(this, i, s)));
	// 		return p;
	// 	}, Object.assign([], { $: this }));
	// }

	/**
	 * @template { { new(node: EzNode, index: number, obj: any): any } } T
	 * @param { "inputs" | "outputs" | "widgets" | (() => Array<unknown>) } nodeProperty
	 * @param { string } nameProperty
	 * @param { T } ctor
	 * @returns { Record<string, InstanceType<T>> & Array<InstanceType<T>> }
	 */
	#makeLookupArray(nodeProperty, nameProperty, ctor) {
		const items = typeof nodeProperty === "function" ? nodeProperty() : this.node[nodeProperty];
		// @ts-ignore
		return (items ?? []).reduce((p, s, i) => {
			if (!s) return p;

			const name = s[nameProperty];
			const item = new ctor(this, i, s);
			// @ts-ignore
			p.push(item);
			if (name) {
				// @ts-ignore
				if (name in p) {
					throw new Error(`Unable to store ${nodeProperty} ${name} on array as name conflicts.`);
				}
			}
			// @ts-ignore
			p[name] = item;
			return p;
		}, Object.assign([], { $: this }));
	}
}

export class EzGraph {
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

	stringify() {
		return JSON.stringify(this.app.graph.serialize(), undefined);
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
			setTimeout(async () => {
				await this.app.loadGraphData(graph);
				r();
			}, 10);
		});
	}

	/**
	 * @returns { Promise<{
	 * 	workflow: {},
	 * 	output: Record<string, {
	 * 		class_name: string,
	 * 		inputs: Record<string, [string, number] | unknown>
	 * }>}> }
	 */
	toPrompt() {
		// @ts-ignore
		return this.app.graphToPrompt();
	}
}

export const Ez = {
	/**
	 * Quickly build and interact with a ComfyUI graph
	 * @example
	 * const { ez, graph } = Ez.graph(app);
	 * graph.clear();
	 * const [model, clip, vae] = ez.CheckpointLoaderSimple().outputs;
	 * const [pos] = ez.CLIPTextEncode(clip, { text: "positive" }).outputs;
	 * const [neg] = ez.CLIPTextEncode(clip, { text: "negative" }).outputs;
	 * const [latent] = ez.KSampler(model, pos, neg, ...ez.EmptyLatentImage().outputs).outputs;
	 * const [image] = ez.VAEDecode(latent, vae).outputs;
	 * const saveNode = ez.SaveImage(image);
	 * console.log(saveNode);
	 * graph.arrange();
	 * @param { app } app
	 * @param { LG["LiteGraph"] } LiteGraph
	 * @param { LG["LGraphCanvas"] } LGraphCanvas
	 * @param { boolean } clearGraph
	 * @returns { { graph: EzGraph, ez: Record<string, EzNodeFactory> } }
	 */
	graph(app, LiteGraph = window["LiteGraph"], LGraphCanvas = window["LGraphCanvas"], clearGraph = true) {
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
						const inputs = ezNode.inputs;

						let slot = 0;
						for (const arg of args) {
							if (arg instanceof EzOutput) {
								arg.connectTo(inputs[slot++]);
							} else {
								for (const k in arg) {
									ezNode.widgets[k].value = arg[k];
								}
							}
						}

						return ezNode;
					};
				},
			}
		);

		return { graph: new EzGraph(app), ez: factory };
	},
};
