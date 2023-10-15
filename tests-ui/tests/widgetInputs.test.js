/// <reference path="../node_modules/@types/jest/index.d.ts" />
// @ts-check

const { start } = require("../utils");
const lg = require("../utils/litegraph");


describe("widget inputs", () => {
	beforeEach(() => {
		lg.setup(global);
	});

	afterEach(() => {
		lg.teardown(global);
		jest.resetModules();
	});

	[
		{ name: "int", type: "INT", widget: "number", control: true },
		{ name: "float", type: "FLOAT", widget: "number", control: true },
		{ name: "text", type: "STRING" },
		{
			name: "customtext",
			type: "STRING",
			opt: { multiline: true },
		},
		{ name: "toggle", type: "BOOLEAN" },
		{ name: "combo", type: ["a", "b", "c"], control: true },
	].forEach((c) => {
		test(`widget conversion + primitive works on ${c.name}`, async () => {
			/**
			 * Test node with widgets of each type
			 * @type { import("../../web/types/comfy").ComfyObjectInfo } ComfyObjectInfo
			 */
			const WidgetTestNode = {
				category: "test",
				name: "WidgetTestNode",
				output_name: [],
				input: {
					required: {
						[c.name]: [c.type, c.opt ?? {}],
					},
				},
			};

			const { ez } = await start({
				mockNodeDefs: {
					WidgetTestNode,
				},
			});

			// Create test node and convert to input
			const n = ez.WidgetTestNode();
			const w = n.widgets[c.name];
			w.convertToInput();
			expect(w.isConvertedToInput).toBeTruthy();
			const input = w.getConvertedInput();
			expect(input).toBeTruthy();

			// Connect to primitive
			const p1 = ez.PrimitiveNode();
			// @ts-ignore : input is valid
			p1.outputs[0].connectTo(input);
			expect(p1.outputs[0].connectTo).toHaveLength(1);

			// Ensure widget is correct type
			const valueWidget = p1.widgets.value;
			expect(valueWidget.widget.type).toBe(c.widget ?? c.name);

			// Check if control_after_generate should be added
			if (c.control) {
				const controlWidget = p1.widgets.control_after_generate;
				expect(controlWidget.widget.type).toBe("combo");
			}

			// Ensure we dont have other widgets
			expect(p1.node.widgets).toHaveLength(1 + +!!c.control);
		});
	});

	test("converted widget works after reload", async () => {
		const { graph, ez } = await start();
		let n = ez.CheckpointLoaderSimple();

		const inputCount = n.inputs.length;

		// Convert ckpt name to an input
		n.widgets.ckpt_name.convertToInput();
		expect(n.widgets.ckpt_name.isConvertedToInput).toBeTruthy();
		expect(n.inputs.ckpt_name).toBeTruthy();
		expect(n.inputs.length).toEqual(inputCount + 1);

		// Convert back to widget and ensure input is removed
		n.widgets.ckpt_name.convertToWidget();
		expect(n.widgets.ckpt_name.isConvertedToInput).toBeFalsy();
		expect(n.inputs.ckpt_name).toBeFalsy();
		expect(n.inputs.length).toEqual(inputCount);

		// Convert again and reload the graph to ensure it maintains state
		n.widgets.ckpt_name.convertToInput();
		expect(n.inputs.length).toEqual(inputCount + 1);

		let primitive = ez.PrimitiveNode();
		primitive.outputs[0].connectTo(n.inputs.ckpt_name);

		await graph.reload();

		// Find the reloaded nodes in the graph
		n = graph.find(n);
		primitive = graph.find(primitive);

		// Ensure widget is converted
		expect(n.widgets.ckpt_name.isConvertedToInput).toBeTruthy();
		expect(n.inputs.ckpt_name).toBeTruthy();
		expect(n.inputs.length).toEqual(inputCount + 1);

		// Ensure primitive is connected
		let { connections } = primitive.outputs[0];
		expect(connections).toHaveLength(1);
		expect(connections[0].targetNode.id).toBe(n.node.id);

		// Disconnect & reconnect
		connections[0].disconnect();
		({ connections } = primitive.outputs[0]);
		expect(connections).toHaveLength(0);
		primitive.outputs[0].connectTo(n.inputs.ckpt_name);

		({ connections } = primitive.outputs[0]);
		expect(connections).toHaveLength(1);
		expect(connections[0].targetNode.id).toBe(n.node.id);

		// Convert back to widget and ensure input is removed
		n.widgets.ckpt_name.convertToWidget();
		expect(n.widgets.ckpt_name.isConvertedToInput).toBeFalsy();
		expect(n.inputs.ckpt_name).toBeFalsy();
		expect(n.inputs.length).toEqual(inputCount);
	});

	test("converted widget works on clone", async () => {
		const { graph, ez } = await start();
		let n = ez.CheckpointLoaderSimple();

		// Convert the widget to an input
		n.widgets.ckpt_name.convertToInput();
		expect(n.widgets.ckpt_name.isConvertedToInput).toBeTruthy();

		// Clone the node
		n.menu["Clone"].call();
		expect(graph.nodes).toHaveLength(2);
		const clone = graph.nodes[1];
		expect(clone.id).not.toEqual(n.id);

		// Ensure the clone has an input
		expect(clone.widgets.ckpt_name.isConvertedToInput).toBeTruthy();
		expect(clone.inputs.ckpt_name).toBeTruthy();

		// Ensure primitive connects to both nodes
		let primitive = ez.PrimitiveNode();
		primitive.outputs[0].connectTo(n.inputs.ckpt_name);
		primitive.outputs[0].connectTo(clone.inputs.ckpt_name);
		expect(primitive.outputs[0].connections).toHaveLength(2);

		// Convert back to widget and ensure input is removed
		clone.widgets.ckpt_name.convertToWidget();
		expect(clone.widgets.ckpt_name.isConvertedToInput).toBeFalsy();
		expect(clone.inputs.ckpt_name).toBeFalsy();
	});

	test("shows missing node error on custom node with converted input", async () => {
		const { graph } = await start();

		const dialogShow = jest.spyOn(graph.app.ui.dialog, "show")

		await graph.app.loadGraphData({
			last_node_id: 3,
			last_link_id: 4,
			nodes: [
				{
					id: 1,
					type: "TestNode",
					pos: [41.87329101561909, 389.7381480823742],
					size: { 0: 220, 1: 374 },
					flags: {},
					order: 1,
					mode: 0,
					inputs: [{ name: "test", type: "FLOAT", link: 4, widget: { name: "test" }, slot_index: 0 }],
					outputs: [],
					properties: { "Node name for S&R": "TestNode" },
					widgets_values: [1],
				},
				{
					id: 3,
					type: "PrimitiveNode",
					pos: [-312, 433],
					size: { 0: 210, 1: 82 },
					flags: {},
					order: 0,
					mode: 0,
					outputs: [{ links: [4], widget: { name: "test" } }],
					title: "test",
					properties: {},
				},
			],
			links: [[4, 3, 0, 1, 6, "FLOAT"]],
			groups: [],
			config: {},
			extra: {},
			version: 0.4,
		});

		expect(dialogShow).toBeCalledTimes(1);
		expect(dialogShow.mock.calls[0][0]).toContain("the following node types were not found");
		expect(dialogShow.mock.calls[0][0]).toContain("TestNode");
	});
});