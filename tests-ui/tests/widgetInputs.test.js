// @ts-check
/// <reference path="../node_modules/@types/jest/index.d.ts" />

const {
	start,
	makeNodeDef,
	checkBeforeAndAfterReload,
	assertNotNullOrUndefined,
	createDefaultWorkflow,
} = require("../utils");
const lg = require("../utils/litegraph");

/**
 * @typedef { import("../utils/ezgraph") } Ez
 * @typedef { ReturnType<Ez["Ez"]["graph"]>["ez"] } EzNodeFactory
 */

/**
 * @param { EzNodeFactory } ez
 * @param { InstanceType<Ez["EzGraph"]> } graph
 * @param { InstanceType<Ez["EzInput"]> } input
 * @param { string } widgetType
 * @param { number } controlWidgetCount
 * @returns
 */
async function connectPrimitiveAndReload(ez, graph, input, widgetType, controlWidgetCount = 0) {
	// Connect to primitive and ensure its still connected after
	let primitive = ez.PrimitiveNode();
	primitive.outputs[0].connectTo(input);

	await checkBeforeAndAfterReload(graph, async () => {
		primitive = graph.find(primitive);
		let { connections } = primitive.outputs[0];
		expect(connections).toHaveLength(1);
		expect(connections[0].targetNode.id).toBe(input.node.node.id);

		// Ensure widget is correct type
		const valueWidget = primitive.widgets.value;
		expect(valueWidget.widget.type).toBe(widgetType);

		// Check if control_after_generate should be added
		if (controlWidgetCount) {
			const controlWidget = primitive.widgets.control_after_generate;
			expect(controlWidget.widget.type).toBe("combo");
			if (widgetType === "combo") {
				const filterWidget = primitive.widgets.control_filter_list;
				expect(filterWidget.widget.type).toBe("string");
			}
		}

		// Ensure we dont have other widgets
		expect(primitive.node.widgets).toHaveLength(1 + controlWidgetCount);
	});

	return primitive;
}

describe("widget inputs", () => {
	beforeEach(() => {
		lg.setup(global);
	});

	afterEach(() => {
		lg.teardown(global);
	});

	[
		{ name: "int", type: "INT", widget: "number", control: 1 },
		{ name: "float", type: "FLOAT", widget: "number", control: 1 },
		{ name: "text", type: "STRING" },
		{
			name: "customtext",
			type: "STRING",
			opt: { multiline: true },
		},
		{ name: "toggle", type: "BOOLEAN" },
		{ name: "combo", type: ["a", "b", "c"], control: 2 },
	].forEach((c) => {
		test(`widget conversion + primitive works on ${c.name}`, async () => {
			const { ez, graph } = await start({
				mockNodeDefs: makeNodeDef("TestNode", { [c.name]: [c.type, c.opt ?? {}] }),
			});

			// Create test node and convert to input
			const n = ez.TestNode();
			const w = n.widgets[c.name];
			w.convertToInput();
			expect(w.isConvertedToInput).toBeTruthy();
			const input = w.getConvertedInput();
			expect(input).toBeTruthy();

			// @ts-ignore : input is valid here
			await connectPrimitiveAndReload(ez, graph, input, c.widget ?? c.name, c.control);
		});
	});

	test("converted widget works after reload", async () => {
		const { ez, graph } = await start();
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

		const primitive = await connectPrimitiveAndReload(ez, graph, n.inputs.ckpt_name, "combo", 2);

		// Disconnect & reconnect
		primitive.outputs[0].connections[0].disconnect();
		let { connections } = primitive.outputs[0];
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

		const dialogShow = jest.spyOn(graph.app.ui.dialog, "show");

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
		expect(dialogShow.mock.calls[0][0].innerHTML).toContain("the following node types were not found");
		expect(dialogShow.mock.calls[0][0].innerHTML).toContain("TestNode");
	});

	test("defaultInput widgets can be converted back to inputs", async () => {
		const { graph, ez } = await start({
			mockNodeDefs: makeNodeDef("TestNode", { example: ["INT", { defaultInput: true }] }),
		});

		// Create test node and ensure it starts as an input
		let n = ez.TestNode();
		let w = n.widgets.example;
		expect(w.isConvertedToInput).toBeTruthy();
		let input = w.getConvertedInput();
		expect(input).toBeTruthy();

		// Ensure it can be converted to
		w.convertToWidget();
		expect(w.isConvertedToInput).toBeFalsy();
		expect(n.inputs.length).toEqual(0);
		// and from
		w.convertToInput();
		expect(w.isConvertedToInput).toBeTruthy();
		input = w.getConvertedInput();

		// Reload and ensure it still only has 1 converted widget
		if (!assertNotNullOrUndefined(input)) return;

		await connectPrimitiveAndReload(ez, graph, input, "number", 1);
		n = graph.find(n);
		expect(n.widgets).toHaveLength(1);
		w = n.widgets.example;
		expect(w.isConvertedToInput).toBeTruthy();

		// Convert back to widget and ensure it is still a widget after reload
		w.convertToWidget();
		await graph.reload();
		n = graph.find(n);
		expect(n.widgets).toHaveLength(1);
		expect(n.widgets[0].isConvertedToInput).toBeFalsy();
		expect(n.inputs.length).toEqual(0);
	});

	test("forceInput widgets can not be converted back to inputs", async () => {
		const { graph, ez } = await start({
			mockNodeDefs: makeNodeDef("TestNode", { example: ["INT", { forceInput: true }] }),
		});

		// Create test node and ensure it starts as an input
		let n = ez.TestNode();
		let w = n.widgets.example;
		expect(w.isConvertedToInput).toBeTruthy();
		const input = w.getConvertedInput();
		expect(input).toBeTruthy();

		// Convert to widget should error
		expect(() => w.convertToWidget()).toThrow();

		// Reload and ensure it still only has 1 converted widget
		if (assertNotNullOrUndefined(input)) {
			await connectPrimitiveAndReload(ez, graph, input, "number", 1);
			n = graph.find(n);
			expect(n.widgets).toHaveLength(1);
			expect(n.widgets.example.isConvertedToInput).toBeTruthy();
		}
	});

	test("primitive can connect to matching combos on converted widgets", async () => {
		const { ez } = await start({
			mockNodeDefs: {
				...makeNodeDef("TestNode1", { example: [["A", "B", "C"], { forceInput: true }] }),
				...makeNodeDef("TestNode2", { example: [["A", "B", "C"], { forceInput: true }] }),
			},
		});

		const n1 = ez.TestNode1();
		const n2 = ez.TestNode2();
		const p = ez.PrimitiveNode();
		p.outputs[0].connectTo(n1.inputs[0]);
		p.outputs[0].connectTo(n2.inputs[0]);
		expect(p.outputs[0].connections).toHaveLength(2);
		const valueWidget = p.widgets.value;
		expect(valueWidget.widget.type).toBe("combo");
		expect(valueWidget.widget.options.values).toEqual(["A", "B", "C"]);
	});

	test("primitive can not connect to non matching combos on converted widgets", async () => {
		const { ez } = await start({
			mockNodeDefs: {
				...makeNodeDef("TestNode1", { example: [["A", "B", "C"], { forceInput: true }] }),
				...makeNodeDef("TestNode2", { example: [["A", "B"], { forceInput: true }] }),
			},
		});

		const n1 = ez.TestNode1();
		const n2 = ez.TestNode2();
		const p = ez.PrimitiveNode();
		p.outputs[0].connectTo(n1.inputs[0]);
		expect(() => p.outputs[0].connectTo(n2.inputs[0])).toThrow();
		expect(p.outputs[0].connections).toHaveLength(1);
	});

	test("combo output can not connect to non matching combos list input", async () => {
		const { ez } = await start({
			mockNodeDefs: {
				...makeNodeDef("TestNode1", {}, [["A", "B"]]),
				...makeNodeDef("TestNode2", { example: [["A", "B"], { forceInput: true }] }),
				...makeNodeDef("TestNode3", { example: [["A", "B", "C"], { forceInput: true }] }),
			},
		});

		const n1 = ez.TestNode1();
		const n2 = ez.TestNode2();
		const n3 = ez.TestNode3();

		n1.outputs[0].connectTo(n2.inputs[0]);
		expect(() => n1.outputs[0].connectTo(n3.inputs[0])).toThrow();
	});

	test("combo primitive can filter list when control_after_generate called", async () => {
		const { ez } = await start({
			mockNodeDefs: {
				...makeNodeDef("TestNode1", { example: [["A", "B", "C", "D", "AA", "BB", "CC", "DD", "AAA", "BBB"], {}] }),
			},
		});

		const n1 = ez.TestNode1();
		n1.widgets.example.convertToInput();
		const p = ez.PrimitiveNode();
		p.outputs[0].connectTo(n1.inputs[0]);

		const value = p.widgets.value;
		const control = p.widgets.control_after_generate.widget;
		const filter = p.widgets.control_filter_list;

		expect(p.widgets.length).toBe(3);
		control.value = "increment";
		expect(value.value).toBe("A");

		// Manually trigger after queue when set to increment
		control["afterQueued"]();
		expect(value.value).toBe("B");

		// Filter to items containing D
		filter.value = "D";
		control["afterQueued"]();
		expect(value.value).toBe("D");
		control["afterQueued"]();
		expect(value.value).toBe("DD");

		// Check decrement
		value.value = "BBB";
		control.value = "decrement";
		filter.value = "B";
		control["afterQueued"]();
		expect(value.value).toBe("BB");
		control["afterQueued"]();
		expect(value.value).toBe("B");

		// Check regex works
		value.value = "BBB";
		filter.value = "/[AB]|^C$/";
		control["afterQueued"]();
		expect(value.value).toBe("AAA");
		control["afterQueued"]();
		expect(value.value).toBe("BB");
		control["afterQueued"]();
		expect(value.value).toBe("AA");
		control["afterQueued"]();
		expect(value.value).toBe("C");
		control["afterQueued"]();
		expect(value.value).toBe("B");
		control["afterQueued"]();
		expect(value.value).toBe("A");

		// Check random
		control.value = "randomize";
		filter.value = "/D/";
		for (let i = 0; i < 100; i++) {
			control["afterQueued"]();
			expect(value.value === "D" || value.value === "DD").toBeTruthy();
		}

		// Ensure it doesnt apply when fixed
		control.value = "fixed";
		value.value = "B";
		filter.value = "C";
		control["afterQueued"]();
		expect(value.value).toBe("B");
	});

	describe("reroutes", () => {
		async function checkOutput(graph, values) {
			expect((await graph.toPrompt()).output).toStrictEqual({
				1: { inputs: { ckpt_name: "model1.safetensors" }, class_type: "CheckpointLoaderSimple" },
				2: { inputs: { text: "positive", clip: ["1", 1] }, class_type: "CLIPTextEncode" },
				3: { inputs: { text: "negative", clip: ["1", 1] }, class_type: "CLIPTextEncode" },
				4: {
					inputs: { width: values.width ?? 512, height: values.height ?? 512, batch_size: values?.batch_size ?? 1 },
					class_type: "EmptyLatentImage",
				},
				5: {
					inputs: {
						seed: 0,
						steps: 20,
						cfg: 8,
						sampler_name: "euler",
						scheduler: values?.scheduler ?? "normal",
						denoise: 1,
						model: ["1", 0],
						positive: ["2", 0],
						negative: ["3", 0],
						latent_image: ["4", 0],
					},
					class_type: "KSampler",
				},
				6: { inputs: { samples: ["5", 0], vae: ["1", 2] }, class_type: "VAEDecode" },
				7: {
					inputs: { filename_prefix: values.filename_prefix ?? "ComfyUI", images: ["6", 0] },
					class_type: "SaveImage",
				},
			});
		}

		async function waitForWidget(node) {
			// widgets are created slightly after the graph is ready
			// hard to find an exact hook to get these so just wait for them to be ready
			for (let i = 0; i < 10; i++) {
				await new Promise((r) => setTimeout(r, 10));
				if (node.widgets?.value) {
					return;
				}
			}
		}

		it("can connect primitive via a reroute path to a widget input", async () => {
			const { ez, graph } = await start();
			const nodes = createDefaultWorkflow(ez, graph);

			nodes.empty.widgets.width.convertToInput();
			nodes.sampler.widgets.scheduler.convertToInput();
			nodes.save.widgets.filename_prefix.convertToInput();

			let widthReroute = ez.Reroute();
			let schedulerReroute = ez.Reroute();
			let fileReroute = ez.Reroute();

			let widthNext = widthReroute;
			let schedulerNext = schedulerReroute;
			let fileNext = fileReroute;

			for (let i = 0; i < 5; i++) {
				let next = ez.Reroute();
				widthNext.outputs[0].connectTo(next.inputs[0]);
				widthNext = next;

				next = ez.Reroute();
				schedulerNext.outputs[0].connectTo(next.inputs[0]);
				schedulerNext = next;

				next = ez.Reroute();
				fileNext.outputs[0].connectTo(next.inputs[0]);
				fileNext = next;
			}

			widthNext.outputs[0].connectTo(nodes.empty.inputs.width);
			schedulerNext.outputs[0].connectTo(nodes.sampler.inputs.scheduler);
			fileNext.outputs[0].connectTo(nodes.save.inputs.filename_prefix);

			let widthPrimitive = ez.PrimitiveNode();
			let schedulerPrimitive = ez.PrimitiveNode();
			let filePrimitive = ez.PrimitiveNode();

			widthPrimitive.outputs[0].connectTo(widthReroute.inputs[0]);
			schedulerPrimitive.outputs[0].connectTo(schedulerReroute.inputs[0]);
			filePrimitive.outputs[0].connectTo(fileReroute.inputs[0]);
			expect(widthPrimitive.widgets.value.value).toBe(512);
			widthPrimitive.widgets.value.value = 1024;
			expect(schedulerPrimitive.widgets.value.value).toBe("normal");
			schedulerPrimitive.widgets.value.value = "simple";
			expect(filePrimitive.widgets.value.value).toBe("ComfyUI");
			filePrimitive.widgets.value.value = "ComfyTest";

			await checkBeforeAndAfterReload(graph, async () => {
				widthPrimitive = graph.find(widthPrimitive);
				schedulerPrimitive = graph.find(schedulerPrimitive);
				filePrimitive = graph.find(filePrimitive);
				await waitForWidget(filePrimitive);
				expect(widthPrimitive.widgets.length).toBe(2);
				expect(schedulerPrimitive.widgets.length).toBe(3);
				expect(filePrimitive.widgets.length).toBe(1);

				await checkOutput(graph, {
					width: 1024,
					scheduler: "simple",
					filename_prefix: "ComfyTest",
				});
			});
		});
		it("can connect primitive via a reroute path to multiple widget inputs", async () => {
			const { ez, graph } = await start();
			const nodes = createDefaultWorkflow(ez, graph);

			nodes.empty.widgets.width.convertToInput();
			nodes.empty.widgets.height.convertToInput();
			nodes.empty.widgets.batch_size.convertToInput();

			let reroute = ez.Reroute();
			let prevReroute = reroute;
			for (let i = 0; i < 5; i++) {
				const next = ez.Reroute();
				prevReroute.outputs[0].connectTo(next.inputs[0]);
				prevReroute = next;
			}

			const r1 = ez.Reroute(prevReroute.outputs[0]);
			const r2 = ez.Reroute(prevReroute.outputs[0]);
			const r3 = ez.Reroute(r2.outputs[0]);
			const r4 = ez.Reroute(r2.outputs[0]);

			r1.outputs[0].connectTo(nodes.empty.inputs.width);
			r3.outputs[0].connectTo(nodes.empty.inputs.height);
			r4.outputs[0].connectTo(nodes.empty.inputs.batch_size);

			let primitive = ez.PrimitiveNode();
			primitive.outputs[0].connectTo(reroute.inputs[0]);
			expect(primitive.widgets.value.value).toBe(1);
			primitive.widgets.value.value = 64;

			await checkBeforeAndAfterReload(graph, async (r) => {
				primitive = graph.find(primitive);
				await waitForWidget(primitive);

				// Ensure widget configs are merged
				expect(primitive.widgets.value.widget.options?.min).toBe(16); // width/height min
				expect(primitive.widgets.value.widget.options?.max).toBe(4096); // batch max
				expect(primitive.widgets.value.widget.options?.step).toBe(80); // width/height step * 10

				await checkOutput(graph, {
					width: 64,
					height: 64,
					batch_size: 64,
				});
			});
		});
	});
});
