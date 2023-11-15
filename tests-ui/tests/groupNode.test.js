// @ts-check
/// <reference path="../node_modules/@types/jest/index.d.ts" />

const { start, createDefaultWorkflow } = require("../utils");
const lg = require("../utils/litegraph");

describe("group node", () => {
	beforeEach(() => {
		lg.setup(global);
	});

	afterEach(() => {
		lg.teardown(global);
	});

	/**
	 *
	 * @param {*} app
	 * @param {*} graph
	 * @param {*} name
	 * @param {*} nodes
	 * @returns { Promise<InstanceType<import("../utils/ezgraph")["EzNode"]>> }
	 */
	async function convertToGroup(app, graph, name, nodes) {
		// Select the nodes we are converting
		for (const n of nodes) {
			n.select(true);
		}

		expect(Object.keys(app.canvas.selected_nodes).sort((a, b) => +a - +b)).toEqual(
			nodes.map((n) => n.id + "").sort((a, b) => +a - +b)
		);

		global.prompt = jest.fn().mockImplementation(() => name);
		const groupNode = await nodes[0].menu["Convert to Group Node"].call(false);

		// Check group name was requested
		expect(window.prompt).toHaveBeenCalled();

		// Ensure old nodes are removed
		for (const n of nodes) {
			expect(n.isRemoved).toBeTruthy();
		}

		expect(groupNode.type).toEqual("workflow/" + name);

		return graph.find(groupNode);
	}

	/**
	 * @param { Record<string, string> | number[] } idMap
	 * @param { Record<string, Record<string, unknown>> } valueMap
	 */
	function getOutput(idMap = {}, valueMap = {}) {
		if (idMap instanceof Array) {
			idMap = idMap.reduce((p, n) => {
				p[n] = n + "";
				return p;
			}, {});
		}
		const expected = {
			1: { inputs: { ckpt_name: "model1.safetensors", ...valueMap?.[1] }, class_type: "CheckpointLoaderSimple" },
			2: { inputs: { text: "positive", clip: ["1", 1], ...valueMap?.[2] }, class_type: "CLIPTextEncode" },
			3: { inputs: { text: "negative", clip: ["1", 1], ...valueMap?.[3] }, class_type: "CLIPTextEncode" },
			4: { inputs: { width: 512, height: 512, batch_size: 1, ...valueMap?.[4] }, class_type: "EmptyLatentImage" },
			5: {
				inputs: {
					seed: 0,
					steps: 20,
					cfg: 8,
					sampler_name: "euler",
					scheduler: "normal",
					denoise: 1,
					model: ["1", 0],
					positive: ["2", 0],
					negative: ["3", 0],
					latent_image: ["4", 0],
					...valueMap?.[5],
				},
				class_type: "KSampler",
			},
			6: { inputs: { samples: ["5", 0], vae: ["1", 2], ...valueMap?.[6] }, class_type: "VAEDecode" },
			7: { inputs: { filename_prefix: "ComfyUI", images: ["6", 0], ...valueMap?.[7] }, class_type: "SaveImage" },
		};

		for (const oldId in idMap) {
			const old = expected[oldId];
			delete expected[oldId];
			expected[idMap[oldId]] = old;

			for (const k in expected) {
				for (const input in expected[k].inputs) {
					const v = expected[k].inputs[input];
					if (v instanceof Array) {
						if (v[0] in idMap) {
							v[0] = idMap[v[0]];
						}
					}
				}
			}
		}

		return expected;
	}

	test("can be created from selected nodes", async () => {
		const { ez, graph, app } = await start();
		const nodes = createDefaultWorkflow(ez, graph);
		const group = await convertToGroup(app, graph, "test", [nodes.pos, nodes.neg, nodes.empty]);

		// Ensure links are now to the group node
		expect(group.inputs).toHaveLength(2);
		expect(group.outputs).toHaveLength(3);

		expect(group.inputs.map((i) => i.input.name)).toEqual(["CLIPTextEncode clip", "CLIPTextEncode 2 clip"]);
		expect(group.outputs.map((i) => i.output.name)).toEqual([
			"EmptyLatentImage LATENT",
			"CLIPTextEncode CONDITIONING",
			"CLIPTextEncode 2 CONDITIONING",
		]);

		// ckpt clip to both clip inputs on the group
		expect(nodes.ckpt.outputs.CLIP.connections.map((t) => [t.targetNode.id, t.targetInput.index])).toEqual([
			[group.id, 0],
			[group.id, 1],
		]);

		// group conditioning to sampler
		expect(
			group.outputs["CLIPTextEncode CONDITIONING"].connections.map((t) => [t.targetNode.id, t.targetInput.index])
		).toEqual([[nodes.sampler.id, 1]]);
		// group conditioning 2 to sampler
		expect(
			group.outputs["CLIPTextEncode 2 CONDITIONING"].connections.map((t) => [t.targetNode.id, t.targetInput.index])
		).toEqual([[nodes.sampler.id, 2]]);
		// group latent to sampler
		expect(
			group.outputs["EmptyLatentImage LATENT"].connections.map((t) => [t.targetNode.id, t.targetInput.index])
		).toEqual([[nodes.sampler.id, 3]]);
	});

	test("maintains all output links on conversion", async () => {
		const { ez, graph, app } = await start();
		const nodes = createDefaultWorkflow(ez, graph);
		const save2 = ez.SaveImage(...nodes.decode.outputs);
		const save3 = ez.SaveImage(...nodes.decode.outputs);
		// Ensure an output with multiple links maintains them on convert to group
		const group = await convertToGroup(app, graph, "test", [nodes.sampler, nodes.decode]);
		expect(group.outputs[0].connections.length).toBe(3);
		expect(group.outputs[0].connections[0].targetNode.id).toBe(nodes.save.id);
		expect(group.outputs[0].connections[1].targetNode.id).toBe(save2.id);
		expect(group.outputs[0].connections[2].targetNode.id).toBe(save3.id);

		// and they're still linked when converting back to nodes
		const newNodes = group.menu["Convert to nodes"].call();
		const decode = graph.find(newNodes.find((n) => n.type === "VAEDecode"));
		expect(decode.outputs[0].connections.length).toBe(3);
		expect(decode.outputs[0].connections[0].targetNode.id).toBe(nodes.save.id);
		expect(decode.outputs[0].connections[1].targetNode.id).toBe(save2.id);
		expect(decode.outputs[0].connections[2].targetNode.id).toBe(save3.id);
	});
	test("can be be converted back to nodes", async () => {
		const { ez, graph, app } = await start();
		const nodes = createDefaultWorkflow(ez, graph);
		const toConvert = [nodes.pos, nodes.neg, nodes.empty, nodes.sampler];
		const group = await convertToGroup(app, graph, "test", toConvert);

		// Edit some values to ensure they are set back onto the converted nodes
		expect(group.widgets["CLIPTextEncode text"].value).toBe("positive");
		group.widgets["CLIPTextEncode text"].value = "pos";
		expect(group.widgets["CLIPTextEncode 2 text"].value).toBe("negative");
		group.widgets["CLIPTextEncode 2 text"].value = "neg";
		expect(group.widgets["EmptyLatentImage width"].value).toBe(512);
		group.widgets["EmptyLatentImage width"].value = 1024;
		expect(group.widgets["KSampler sampler_name"].value).toBe("euler");
		group.widgets["KSampler sampler_name"].value = "ddim";
		expect(group.widgets["KSampler control_after_generate"].value).toBe("randomize");
		group.widgets["KSampler control_after_generate"].value = "fixed";

		/** @type { Array<any> } */
		group.menu["Convert to nodes"].call();

		// ensure widget values are set
		const pos = graph.find(nodes.pos.id);
		expect(pos.node.type).toBe("CLIPTextEncode");
		expect(pos.widgets["text"].value).toBe("pos");
		const neg = graph.find(nodes.neg.id);
		expect(neg.node.type).toBe("CLIPTextEncode");
		expect(neg.widgets["text"].value).toBe("neg");
		const empty = graph.find(nodes.empty.id);
		expect(empty.node.type).toBe("EmptyLatentImage");
		expect(empty.widgets["width"].value).toBe(1024);
		const sampler = graph.find(nodes.sampler.id);
		expect(sampler.node.type).toBe("KSampler");
		expect(sampler.widgets["sampler_name"].value).toBe("ddim");
		expect(sampler.widgets["control_after_generate"].value).toBe("fixed");

		// validate links
		expect(nodes.ckpt.outputs.CLIP.connections.map((t) => [t.targetNode.id, t.targetInput.index])).toEqual([
			[pos.id, 0],
			[neg.id, 0],
		]);

		expect(pos.outputs["CONDITIONING"].connections.map((t) => [t.targetNode.id, t.targetInput.index])).toEqual([
			[nodes.sampler.id, 1],
		]);

		expect(neg.outputs["CONDITIONING"].connections.map((t) => [t.targetNode.id, t.targetInput.index])).toEqual([
			[nodes.sampler.id, 2],
		]);

		expect(empty.outputs["LATENT"].connections.map((t) => [t.targetNode.id, t.targetInput.index])).toEqual([
			[nodes.sampler.id, 3],
		]);
	});
	test("it can embed reroutes as inputs", async () => {
		const { ez, graph, app } = await start();
		const nodes = createDefaultWorkflow(ez, graph);

		// Add and connect a reroute to the clip text encodes
		const reroute = ez.Reroute();
		nodes.ckpt.outputs.CLIP.connectTo(reroute.inputs[0]);
		reroute.outputs[0].connectTo(nodes.pos.inputs[0]);
		reroute.outputs[0].connectTo(nodes.neg.inputs[0]);

		// Convert to group and ensure we only have 1 input of the correct type
		const group = await convertToGroup(app, graph, "test", [nodes.pos, nodes.neg, nodes.empty, reroute]);
		expect(group.inputs).toHaveLength(1);
		expect(group.inputs[0].input.type).toEqual("CLIP");

		expect((await graph.toPrompt()).output).toEqual(getOutput());
	});
	test("it can embed reroutes as outputs", async () => {
		const { ez, graph, app } = await start();
		const nodes = createDefaultWorkflow(ez, graph);

		// Add a reroute with no output so we output IMAGE even though its used internally
		const reroute = ez.Reroute();
		nodes.decode.outputs.IMAGE.connectTo(reroute.inputs[0]);

		// Convert to group and ensure there is an IMAGE output
		const group = await convertToGroup(app, graph, "test", [nodes.decode, nodes.save, reroute]);
		expect(group.outputs).toHaveLength(1);
		expect(group.outputs[0].output.type).toEqual("IMAGE");
		expect((await graph.toPrompt()).output).toEqual(getOutput([nodes.decode.id, nodes.save.id]));
	});
	test("it can embed reroutes as pipes", async () => {
		const { ez, graph, app } = await start();
		const nodes = createDefaultWorkflow(ez, graph);

		// Use reroutes as a pipe
		const rerouteModel = ez.Reroute();
		const rerouteClip = ez.Reroute();
		const rerouteVae = ez.Reroute();
		nodes.ckpt.outputs.MODEL.connectTo(rerouteModel.inputs[0]);
		nodes.ckpt.outputs.CLIP.connectTo(rerouteClip.inputs[0]);
		nodes.ckpt.outputs.VAE.connectTo(rerouteVae.inputs[0]);

		const group = await convertToGroup(app, graph, "test", [rerouteModel, rerouteClip, rerouteVae]);

		expect(group.outputs).toHaveLength(3);
		expect(group.outputs.map((o) => o.output.type)).toEqual(["MODEL", "CLIP", "VAE"]);

		expect(group.outputs).toHaveLength(3);
		expect(group.outputs.map((o) => o.output.type)).toEqual(["MODEL", "CLIP", "VAE"]);

		group.outputs[0].connectTo(nodes.sampler.inputs.model);
		group.outputs[1].connectTo(nodes.pos.inputs.clip);
		group.outputs[1].connectTo(nodes.neg.inputs.clip);
	});
	test("creates with widget values from inner nodes", async () => {
		const { ez, graph, app } = await start();
		const nodes = createDefaultWorkflow(ez, graph);

		nodes.ckpt.widgets.ckpt_name.value = "model2.ckpt";
		nodes.pos.widgets.text.value = "hello";
		nodes.neg.widgets.text.value = "world";
		nodes.empty.widgets.width.value = 256;
		nodes.empty.widgets.height.value = 1024;
		nodes.sampler.widgets.seed.value = 1;
		nodes.sampler.widgets.control_after_generate.value = "increment";
		nodes.sampler.widgets.steps.value = 8;
		nodes.sampler.widgets.cfg.value = 4.5;
		nodes.sampler.widgets.sampler_name.value = "uni_pc";
		nodes.sampler.widgets.scheduler.value = "karras";
		nodes.sampler.widgets.denoise.value = 0.9;

		const group = await convertToGroup(app, graph, "test", [
			nodes.ckpt,
			nodes.pos,
			nodes.neg,
			nodes.empty,
			nodes.sampler,
		]);

		expect(group.widgets["CheckpointLoaderSimple ckpt_name"].value).toEqual("model2.ckpt");
		expect(group.widgets["CLIPTextEncode text"].value).toEqual("hello");
		expect(group.widgets["CLIPTextEncode 2 text"].value).toEqual("world");
		expect(group.widgets["EmptyLatentImage width"].value).toEqual(256);
		expect(group.widgets["EmptyLatentImage height"].value).toEqual(1024);
		expect(group.widgets["KSampler seed"].value).toEqual(1);
		expect(group.widgets["KSampler control_after_generate"].value).toEqual("increment");
		expect(group.widgets["KSampler steps"].value).toEqual(8);
		expect(group.widgets["KSampler cfg"].value).toEqual(4.5);
		expect(group.widgets["KSampler sampler_name"].value).toEqual("uni_pc");
		expect(group.widgets["KSampler scheduler"].value).toEqual("karras");
		expect(group.widgets["KSampler denoise"].value).toEqual(0.9);

		expect((await graph.toPrompt()).output).toEqual(
			getOutput([nodes.ckpt.id, nodes.pos.id, nodes.neg.id, nodes.empty.id, nodes.sampler.id], {
				[nodes.ckpt.id]: { ckpt_name: "model2.ckpt" },
				[nodes.pos.id]: { text: "hello" },
				[nodes.neg.id]: { text: "world" },
				[nodes.empty.id]: { width: 256, height: 1024 },
				[nodes.sampler.id]: {
					seed: 1,
					steps: 8,
					cfg: 4.5,
					sampler_name: "uni_pc",
					scheduler: "karras",
					denoise: 0.9,
				},
			})
		);
	});
	test("group inputs can be reroutes", async () => {
		const { ez, graph, app } = await start();
		const nodes = createDefaultWorkflow(ez, graph);
		const group = await convertToGroup(app, graph, "test", [nodes.pos, nodes.neg]);

		const reroute = ez.Reroute();
		nodes.ckpt.outputs.CLIP.connectTo(reroute.inputs[0]);

		reroute.outputs[0].connectTo(group.inputs[0]);
		reroute.outputs[0].connectTo(group.inputs[1]);

		expect((await graph.toPrompt()).output).toEqual(getOutput([nodes.pos.id, nodes.neg.id]));
	});
	test("group outputs can be reroutes", async () => {
		const { ez, graph, app } = await start();
		const nodes = createDefaultWorkflow(ez, graph);
		const group = await convertToGroup(app, graph, "test", [nodes.pos, nodes.neg]);

		const reroute1 = ez.Reroute();
		const reroute2 = ez.Reroute();
		group.outputs[0].connectTo(reroute1.inputs[0]);
		group.outputs[1].connectTo(reroute2.inputs[0]);

		reroute1.outputs[0].connectTo(nodes.sampler.inputs.positive);
		reroute2.outputs[0].connectTo(nodes.sampler.inputs.negative);

		expect((await graph.toPrompt()).output).toEqual(getOutput([nodes.pos.id, nodes.neg.id]));
	});
	test("groups can connect to each other", async () => {
		const { ez, graph, app } = await start();
		const nodes = createDefaultWorkflow(ez, graph);
		const group1 = await convertToGroup(app, graph, "test", [nodes.pos, nodes.neg]);
		const group2 = await convertToGroup(app, graph, "test2", [nodes.empty, nodes.sampler]);

		group1.outputs[0].connectTo(group2.inputs["KSampler positive"]);
		group1.outputs[1].connectTo(group2.inputs["KSampler negative"]);

		expect((await graph.toPrompt()).output).toEqual(
			getOutput([nodes.pos.id, nodes.neg.id, nodes.empty.id, nodes.sampler.id])
		);
	});
	test("displays generated image on group node", async () => {
		const { ez, graph, app } = await start();
		const nodes = createDefaultWorkflow(ez, graph);
		const group = await convertToGroup(app, graph, "test", [
			nodes.pos,
			nodes.neg,
			nodes.empty,
			nodes.sampler,
			nodes.decode,
			nodes.save,
		]);

		const { api } = require("../../web/scripts/api");
		api.dispatchEvent(new CustomEvent("execution_start", {}));
		api.dispatchEvent(new CustomEvent("executing", { detail: `${group.id}:3` }));
		// Event should be forwarded to group node id
		expect(+app.runningNodeId).toEqual(group.id);
		expect(group.node["imgs"]).toBeFalsy();
		api.dispatchEvent(
			new CustomEvent("executed", {
				detail: {
					node: `${group.id}:3`,
					output: {
						images: [
							{
								filename: "test.png",
								type: "output",
							},
						],
					},
				},
			})
		);

		// Trigger paint
		group.node.onDrawBackground?.(app.canvas.ctx, app.canvas.canvas);

		expect(group.node["images"]).toEqual([
			{
				filename: "test.png",
				type: "output",
			},
		]);
	});
	test("allows widgets to be converted to inputs", async () => {
		const { ez, graph, app } = await start();
		const nodes = createDefaultWorkflow(ez, graph);
		const group = await convertToGroup(app, graph, "test", [nodes.pos, nodes.neg]);
		group.widgets[0].convertToInput();

		const primitive = ez.PrimitiveNode();
		primitive.outputs[0].connectTo(group.inputs["CLIPTextEncode text"]);
		primitive.widgets[0].value = "hello";

		expect((await graph.toPrompt()).output).toEqual(
			getOutput([nodes.pos.id, nodes.neg.id], {
				[nodes.pos.id]: { text: "hello" },
			})
		);
	});
	test("can be copied", async () => {
		const { ez, graph, app } = await start();
		const nodes = createDefaultWorkflow(ez, graph);

		const group1 = await convertToGroup(app, graph, "test", [
			nodes.pos,
			nodes.neg,
			nodes.empty,
			nodes.sampler,
			nodes.decode,
			nodes.save,
		]);

		group1.widgets["CLIPTextEncode text"].value = "hello";
		group1.widgets["EmptyLatentImage width"].value = 256;
		group1.widgets["KSampler seed"].value = 1;

		// Clone the node
		group1.menu.Clone.call();
		expect(app.graph._nodes).toHaveLength(3);
		const group2 = graph.find(app.graph._nodes[2]);
		expect(group2.node.type).toEqual("workflow/test");
		expect(group2.id).not.toEqual(group1.id);

		// Reconnect ckpt
		nodes.ckpt.outputs.MODEL.connectTo(group2.inputs["KSampler model"]);
		nodes.ckpt.outputs.CLIP.connectTo(group2.inputs["CLIPTextEncode clip"]);
		nodes.ckpt.outputs.CLIP.connectTo(group2.inputs["CLIPTextEncode 2 clip"]);
		nodes.ckpt.outputs.VAE.connectTo(group2.inputs["VAEDecode vae"]);

		group2.widgets["CLIPTextEncode text"].value = "world";
		group2.widgets["EmptyLatentImage width"].value = 1024;
		group2.widgets["KSampler seed"].value = 100;

		let i = 0;
		expect((await graph.toPrompt()).output).toEqual({
			...getOutput([nodes.empty.id, nodes.pos.id, nodes.neg.id, nodes.sampler.id, nodes.decode.id, nodes.save.id], {
				[nodes.empty.id]: { width: 256 },
				[nodes.pos.id]: { text: "hello" },
				[nodes.sampler.id]: { seed: 1 },
			}),
			...getOutput(
				{
					[nodes.empty.id]: `${group2.id}:${i++}`,
					[nodes.pos.id]: `${group2.id}:${i++}`,
					[nodes.neg.id]: `${group2.id}:${i++}`,
					[nodes.sampler.id]: `${group2.id}:${i++}`,
					[nodes.decode.id]: `${group2.id}:${i++}`,
					[nodes.save.id]: `${group2.id}:${i++}`,
				},
				{
					[nodes.empty.id]: { width: 1024 },
					[nodes.pos.id]: { text: "world" },
					[nodes.sampler.id]: { seed: 100 },
				}
			),
		});

		graph.arrange();
	});
	test("is embedded in workflow", async () => {
		let { ez, graph, app } = await start();
		const nodes = createDefaultWorkflow(ez, graph);
		let group = await convertToGroup(app, graph, "test", [nodes.pos, nodes.neg]);
		const workflow = JSON.stringify((await graph.toPrompt()).workflow);

		// Clear the environment
		({ ez, graph, app } = await start({
			resetEnv: true,
		}));
		// Ensure the node isnt registered
		expect(() => ez["workflow/test"]).toThrow();

		// Reload the workflow
		await app.loadGraphData(JSON.parse(workflow));

		// Ensure the node is found
		group = graph.find(group);

		// Generate prompt and ensure it is as expected
		expect((await graph.toPrompt()).output).toEqual(
			getOutput({
				[nodes.pos.id]: `${group.id}:0`,
				[nodes.neg.id]: `${group.id}:1`,
			})
		);
	});
	test("shows missing node error on missing internal node when loading graph data", async () => {
		const { graph } = await start();

		const dialogShow = jest.spyOn(graph.app.ui.dialog, "show");
		await graph.app.loadGraphData({
			last_node_id: 3,
			last_link_id: 1,
			nodes: [
				{
					id: 3,
					type: "workflow/testerror",
				},
			],
			links: [],
			groups: [],
			config: {},
			extra: {
				groupNodes: {
					testerror: {
						nodes: [
							{
								type: "NotKSampler",
							},
							{
								type: "NotVAEDecode",
							},
						],
					},
				},
			},
		});

		expect(dialogShow).toBeCalledTimes(1);
		const call = dialogShow.mock.calls[0][0].innerHTML;
		expect(call).toContain("the following node types were not found");
		expect(call).toContain("NotKSampler");
		expect(call).toContain("NotVAEDecode");
		expect(call).toContain("workflow/testerror");
	});
	test("maintains widget inputs on conversion back to nodes", async () => {
		const { ez, graph, app } = await start();
		let pos = ez.CLIPTextEncode({ text: "positive" });
		pos.node.title = "Positive";
		let neg = ez.CLIPTextEncode({ text: "negative" });
		neg.node.title = "Negative";
		pos.widgets.text.convertToInput();
		neg.widgets.text.convertToInput();

		let primitive = ez.PrimitiveNode();
		primitive.outputs[0].connectTo(pos.inputs.text);
		primitive.outputs[0].connectTo(neg.inputs.text);

		const group = await convertToGroup(app, graph, "test", [pos, neg, primitive]);
		// These will both be the same due to the primitive
		expect(group.widgets["Positive text"].value).toBe("positive");
		expect(group.widgets["Negative text"].value).toBe("positive");

		const newNodes = group.menu["Convert to nodes"].call();
		pos = graph.find(newNodes.find((n) => n.title === "Positive"));
		neg = graph.find(newNodes.find((n) => n.title === "Negative"));
		primitive = graph.find(newNodes.find((n) => n.type === "PrimitiveNode"));

		expect(pos.inputs).toHaveLength(2);
		expect(neg.inputs).toHaveLength(2);
		expect(primitive.outputs[0].connections).toHaveLength(2);

		expect((await graph.toPrompt()).output).toEqual({
			1: { inputs: { text: "positive" }, class_type: "CLIPTextEncode" },
			2: { inputs: { text: "positive" }, class_type: "CLIPTextEncode" },
		});
	});
	test("adds widgets in node execution order", async () => {
		const { ez, graph, app } = await start();
		const scale = ez.LatentUpscale();
		const save = ez.SaveImage();
		const empty = ez.EmptyLatentImage();
		const decode = ez.VAEDecode();

		scale.outputs.LATENT.connectTo(decode.inputs.samples);
		decode.outputs.IMAGE.connectTo(save.inputs.images);
		empty.outputs.LATENT.connectTo(scale.inputs.samples);

		const group = await convertToGroup(app, graph, "test", [scale, save, empty, decode]);
		const widgets = group.widgets.map((w) => w.widget.name);
		expect(widgets).toStrictEqual([
			"EmptyLatentImage width",
			"EmptyLatentImage height",
			"EmptyLatentImage batch_size",
			"LatentUpscale upscale_method",
			"LatentUpscale width",
			"LatentUpscale height",
			"LatentUpscale crop",
			"SaveImage filename_prefix",
		]);
	});
	test("adds output for external links when converting to group", async () => {
		const { ez, graph, app } = await start();
		const img = ez.EmptyLatentImage();
		let decode = ez.VAEDecode(...img.outputs);
		const preview1 = ez.PreviewImage(...decode.outputs);
		const preview2 = ez.PreviewImage(...decode.outputs);

		const group = await convertToGroup(app, graph, "test", [img, decode, preview1]);

		// Ensure we have an output connected to the 2nd preview node
		expect(group.outputs.length).toBe(1);
		expect(group.outputs[0].connections.length).toBe(1);
		expect(group.outputs[0].connections[0].targetNode.id).toBe(preview2.id);

		// Convert back and ensure bothe previews are still connected
		group.menu["Convert to nodes"].call();
		decode = graph.find(decode);
		expect(decode.outputs[0].connections.length).toBe(2);
		expect(decode.outputs[0].connections[0].targetNode.id).toBe(preview1.id);
		expect(decode.outputs[0].connections[1].targetNode.id).toBe(preview2.id);
	});
	test("works with IMAGEUPLOAD widget", async () => {
		const { ez, graph, app } = await start();
		const img = ez.LoadImage();
		const preview1 = ez.PreviewImage(img.outputs[0]);

		const group = await convertToGroup(app, graph, "test", [img, preview1]);
		const widget = group.widgets["LoadImage upload"];
		expect(widget).toBeTruthy();
		expect(widget.widget.type).toBe("button");
	});
});
