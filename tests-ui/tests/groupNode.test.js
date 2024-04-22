// @ts-check
/// <reference path="../node_modules/@types/jest/index.d.ts" />

const { start, createDefaultWorkflow, getNodeDef, checkBeforeAndAfterReload } = require("../utils");
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
	 * @param { Record<string, string | number> | number[] } idMap
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

		// Map old IDs to new at the top level
		const mapped = {};
		for (const oldId in idMap) {
			mapped[idMap[oldId]] = expected[oldId];
			delete expected[oldId];
		}
		Object.assign(mapped, expected);

		// Map old IDs to new inside links
		for (const k in mapped) {
			for (const input in mapped[k].inputs) {
				const v = mapped[k].inputs[input];
				if (v instanceof Array) {
					if (v[0] in idMap) {
						v[0] = idMap[v[0]] + "";
					}
				}
			}
		}

		return mapped;
	}

	test("can be created from selected nodes", async () => {
		const { ez, graph, app } = await start();
		const nodes = createDefaultWorkflow(ez, graph);
		const group = await convertToGroup(app, graph, "test", [nodes.pos, nodes.neg, nodes.empty]);

		// Ensure links are now to the group node
		expect(group.inputs).toHaveLength(2);
		expect(group.outputs).toHaveLength(3);

		expect(group.inputs.map((i) => i.input.name)).toEqual(["clip", "CLIPTextEncode clip"]);
		expect(group.outputs.map((i) => i.output.name)).toEqual(["LATENT", "CONDITIONING", "CLIPTextEncode CONDITIONING"]);

		// ckpt clip to both clip inputs on the group
		expect(nodes.ckpt.outputs.CLIP.connections.map((t) => [t.targetNode.id, t.targetInput.index])).toEqual([
			[group.id, 0],
			[group.id, 1],
		]);

		// group conditioning to sampler
		expect(group.outputs["CONDITIONING"].connections.map((t) => [t.targetNode.id, t.targetInput.index])).toEqual([
			[nodes.sampler.id, 1],
		]);
		// group conditioning 2 to sampler
		expect(
			group.outputs["CLIPTextEncode CONDITIONING"].connections.map((t) => [t.targetNode.id, t.targetInput.index])
		).toEqual([[nodes.sampler.id, 2]]);
		// group latent to sampler
		expect(group.outputs["LATENT"].connections.map((t) => [t.targetNode.id, t.targetInput.index])).toEqual([
			[nodes.sampler.id, 3],
		]);
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
		expect(group.widgets["text"].value).toBe("positive");
		group.widgets["text"].value = "pos";
		expect(group.widgets["CLIPTextEncode text"].value).toBe("negative");
		group.widgets["CLIPTextEncode text"].value = "neg";
		expect(group.widgets["width"].value).toBe(512);
		group.widgets["width"].value = 1024;
		expect(group.widgets["sampler_name"].value).toBe("euler");
		group.widgets["sampler_name"].value = "ddim";
		expect(group.widgets["control_after_generate"].value).toBe("randomize");
		group.widgets["control_after_generate"].value = "fixed";

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
	test("can handle reroutes used internally", async () => {
		const { ez, graph, app } = await start();
		const nodes = createDefaultWorkflow(ez, graph);

		let reroutes = [];
		let prevNode = nodes.ckpt;
		for (let i = 0; i < 5; i++) {
			const reroute = ez.Reroute();
			prevNode.outputs[0].connectTo(reroute.inputs[0]);
			prevNode = reroute;
			reroutes.push(reroute);
		}
		prevNode.outputs[0].connectTo(nodes.sampler.inputs.model);

		const group = await convertToGroup(app, graph, "test", [...reroutes, ...Object.values(nodes)]);
		expect((await graph.toPrompt()).output).toEqual(getOutput());

		group.menu["Convert to nodes"].call();
		expect((await graph.toPrompt()).output).toEqual(getOutput());
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

		expect(group.widgets["ckpt_name"].value).toEqual("model2.ckpt");
		expect(group.widgets["text"].value).toEqual("hello");
		expect(group.widgets["CLIPTextEncode text"].value).toEqual("world");
		expect(group.widgets["width"].value).toEqual(256);
		expect(group.widgets["height"].value).toEqual(1024);
		expect(group.widgets["seed"].value).toEqual(1);
		expect(group.widgets["control_after_generate"].value).toEqual("increment");
		expect(group.widgets["steps"].value).toEqual(8);
		expect(group.widgets["cfg"].value).toEqual(4.5);
		expect(group.widgets["sampler_name"].value).toEqual("uni_pc");
		expect(group.widgets["scheduler"].value).toEqual("karras");
		expect(group.widgets["denoise"].value).toEqual(0.9);

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

		group1.outputs[0].connectTo(group2.inputs["positive"]);
		group1.outputs[1].connectTo(group2.inputs["negative"]);

		expect((await graph.toPrompt()).output).toEqual(
			getOutput([nodes.pos.id, nodes.neg.id, nodes.empty.id, nodes.sampler.id])
		);
	});
	test("groups can connect to each other via internal reroutes", async () => {
		const { ez, graph, app } = await start();

		const latent = ez.EmptyLatentImage();
		const vae = ez.VAELoader();
		const latentReroute = ez.Reroute();
		const vaeReroute = ez.Reroute();

		latent.outputs[0].connectTo(latentReroute.inputs[0]);
		vae.outputs[0].connectTo(vaeReroute.inputs[0]);

		const group1 = await convertToGroup(app, graph, "test", [latentReroute, vaeReroute]);
		group1.menu.Clone.call();
		expect(app.graph._nodes).toHaveLength(4);
		const group2 = graph.find(app.graph._nodes[3]);
		expect(group2.node.type).toEqual("workflow/test");
		expect(group2.id).not.toEqual(group1.id);

		group1.outputs.VAE.connectTo(group2.inputs.VAE);
		group1.outputs.LATENT.connectTo(group2.inputs.LATENT);

		const decode = ez.VAEDecode(group2.outputs.LATENT, group2.outputs.VAE);
		const preview = ez.PreviewImage(decode.outputs[0]);

		const output = {
			[latent.id]: { inputs: { width: 512, height: 512, batch_size: 1 }, class_type: "EmptyLatentImage" },
			[vae.id]: { inputs: { vae_name: "vae1.safetensors" }, class_type: "VAELoader" },
			[decode.id]: { inputs: { samples: [latent.id + "", 0], vae: [vae.id + "", 0] }, class_type: "VAEDecode" },
			[preview.id]: { inputs: { images: [decode.id + "", 0] }, class_type: "PreviewImage" },
		};
		expect((await graph.toPrompt()).output).toEqual(output);

		// Ensure missing connections dont cause errors
		group2.inputs.VAE.disconnect();
		delete output[decode.id].inputs.vae;
		expect((await graph.toPrompt()).output).toEqual(output);
	});
	test("displays generated image on group node", async () => {
		const { ez, graph, app } = await start();
		const nodes = createDefaultWorkflow(ez, graph);
		let group = await convertToGroup(app, graph, "test", [
			nodes.pos,
			nodes.neg,
			nodes.empty,
			nodes.sampler,
			nodes.decode,
			nodes.save,
		]);

		const { api } = require("../../web/scripts/api");

		api.dispatchEvent(new CustomEvent("execution_start", {}));
		api.dispatchEvent(new CustomEvent("executing", { detail: `${nodes.save.id}` }));
		// Event should be forwarded to group node id
		expect(+app.runningNodeId).toEqual(group.id);
		expect(group.node["imgs"]).toBeFalsy();
		api.dispatchEvent(
			new CustomEvent("executed", {
				detail: {
					node: `${nodes.save.id}`,
					display_node: `${nodes.save.id}`,
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

		// Reload
		const workflow = JSON.stringify((await graph.toPrompt()).workflow);
		await app.loadGraphData(JSON.parse(workflow));
		group = graph.find(group);

		// Trigger inner nodes to get created
		group.node["getInnerNodes"]();

		// Check it works for internal node ids
		api.dispatchEvent(new CustomEvent("execution_start", {}));
		api.dispatchEvent(new CustomEvent("executing", { detail: `${group.id}:5` }));
		// Event should be forwarded to group node id
		expect(+app.runningNodeId).toEqual(group.id);
		expect(group.node["imgs"]).toBeFalsy();
		api.dispatchEvent(
			new CustomEvent("executed", {
				detail: {
					node: `${group.id}:5`,
					display_node: `${group.id}:5`,
					output: {
						images: [
							{
								filename: "test2.png",
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
				filename: "test2.png",
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
		primitive.outputs[0].connectTo(group.inputs["text"]);
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

		group1.widgets["text"].value = "hello";
		group1.widgets["width"].value = 256;
		group1.widgets["seed"].value = 1;

		// Clone the node
		group1.menu.Clone.call();
		expect(app.graph._nodes).toHaveLength(3);
		const group2 = graph.find(app.graph._nodes[2]);
		expect(group2.node.type).toEqual("workflow/test");
		expect(group2.id).not.toEqual(group1.id);

		// Reconnect ckpt
		nodes.ckpt.outputs.MODEL.connectTo(group2.inputs["model"]);
		nodes.ckpt.outputs.CLIP.connectTo(group2.inputs["clip"]);
		nodes.ckpt.outputs.CLIP.connectTo(group2.inputs["CLIPTextEncode clip"]);
		nodes.ckpt.outputs.VAE.connectTo(group2.inputs["vae"]);

		group2.widgets["text"].value = "world";
		group2.widgets["width"].value = 1024;
		group2.widgets["seed"].value = 100;

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
		// This will use a primitive widget named 'value'
		expect(group.widgets.length).toBe(1);
		expect(group.widgets["value"].value).toBe("positive");

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
	test("correctly handles widget inputs", async () => {
		const { ez, graph, app } = await start();
		const upscaleMethods = (await getNodeDef("ImageScaleBy")).input.required["upscale_method"][0];

		const image = ez.LoadImage();
		const scale1 = ez.ImageScaleBy(image.outputs[0]);
		const scale2 = ez.ImageScaleBy(image.outputs[0]);
		const preview1 = ez.PreviewImage(scale1.outputs[0]);
		const preview2 = ez.PreviewImage(scale2.outputs[0]);
		scale1.widgets.upscale_method.value = upscaleMethods[1];
		scale1.widgets.upscale_method.convertToInput();

		const group = await convertToGroup(app, graph, "test", [scale1, scale2]);
		expect(group.inputs.length).toBe(3);
		expect(group.inputs[0].input.type).toBe("IMAGE");
		expect(group.inputs[1].input.type).toBe("IMAGE");
		expect(group.inputs[2].input.type).toBe("COMBO");

		// Ensure links are maintained
		expect(group.inputs[0].connection?.originNode?.id).toBe(image.id);
		expect(group.inputs[1].connection?.originNode?.id).toBe(image.id);
		expect(group.inputs[2].connection).toBeFalsy();

		// Ensure primitive gets correct type
		const primitive = ez.PrimitiveNode();
		primitive.outputs[0].connectTo(group.inputs[2]);
		expect(primitive.widgets.value.widget.options.values).toBe(upscaleMethods);
		expect(primitive.widgets.value.value).toBe(upscaleMethods[1]); // Ensure value is copied
		primitive.widgets.value.value = upscaleMethods[1];
		
		await checkBeforeAndAfterReload(graph, async (r) => {
			const scale1id = r ? `${group.id}:0` : scale1.id;
			const scale2id = r ? `${group.id}:1` : scale2.id;
			// Ensure widget value is applied to prompt
			expect((await graph.toPrompt()).output).toStrictEqual({
				[image.id]: { inputs: { image: "example.png", upload: "image" }, class_type: "LoadImage" },
				[scale1id]: {
					inputs: { upscale_method: upscaleMethods[1], scale_by: 1, image: [`${image.id}`, 0] },
					class_type: "ImageScaleBy",
				},
				[scale2id]: {
					inputs: { upscale_method: "nearest-exact", scale_by: 1, image: [`${image.id}`, 0] },
					class_type: "ImageScaleBy",
				},
				[preview1.id]: { inputs: { images: [`${scale1id}`, 0] }, class_type: "PreviewImage" },
				[preview2.id]: { inputs: { images: [`${scale2id}`, 0] }, class_type: "PreviewImage" },
			});
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
			"width",
			"height",
			"batch_size",
			"upscale_method",
			"LatentUpscale width",
			"LatentUpscale height",
			"crop",
			"filename_prefix",
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
	test("adds output for external links when converting to group when nodes are not in execution order", async () => {
		const { ez, graph, app } = await start();
		const sampler = ez.KSampler();
		const ckpt = ez.CheckpointLoaderSimple();
		const empty = ez.EmptyLatentImage();
		const pos = ez.CLIPTextEncode(ckpt.outputs.CLIP, { text: "positive" });
		const neg = ez.CLIPTextEncode(ckpt.outputs.CLIP, { text: "negative" });
		const decode1 = ez.VAEDecode(sampler.outputs.LATENT, ckpt.outputs.VAE);
		const save = ez.SaveImage(decode1.outputs.IMAGE);
		ckpt.outputs.MODEL.connectTo(sampler.inputs.model);
		pos.outputs.CONDITIONING.connectTo(sampler.inputs.positive);
		neg.outputs.CONDITIONING.connectTo(sampler.inputs.negative);
		empty.outputs.LATENT.connectTo(sampler.inputs.latent_image);

		const encode = ez.VAEEncode(decode1.outputs.IMAGE);
		const vae = ez.VAELoader();
		const decode2 = ez.VAEDecode(encode.outputs.LATENT, vae.outputs.VAE);
		const preview = ez.PreviewImage(decode2.outputs.IMAGE);
		vae.outputs.VAE.connectTo(encode.inputs.vae);

		const group = await convertToGroup(app, graph, "test", [vae, decode1, encode, sampler]);

		expect(group.outputs.length).toBe(3);
		expect(group.outputs[0].output.name).toBe("VAE");
		expect(group.outputs[0].output.type).toBe("VAE");
		expect(group.outputs[1].output.name).toBe("IMAGE");
		expect(group.outputs[1].output.type).toBe("IMAGE");
		expect(group.outputs[2].output.name).toBe("LATENT");
		expect(group.outputs[2].output.type).toBe("LATENT");

		expect(group.outputs[0].connections.length).toBe(1);
		expect(group.outputs[0].connections[0].targetNode.id).toBe(decode2.id);
		expect(group.outputs[0].connections[0].targetInput.index).toBe(1);

		expect(group.outputs[1].connections.length).toBe(1);
		expect(group.outputs[1].connections[0].targetNode.id).toBe(save.id);
		expect(group.outputs[1].connections[0].targetInput.index).toBe(0);

		expect(group.outputs[2].connections.length).toBe(1);
		expect(group.outputs[2].connections[0].targetNode.id).toBe(decode2.id);
		expect(group.outputs[2].connections[0].targetInput.index).toBe(0);

		expect((await graph.toPrompt()).output).toEqual({
			...getOutput({ 1: ckpt.id, 2: pos.id, 3: neg.id, 4: empty.id, 5: sampler.id, 6: decode1.id, 7: save.id }),
			[vae.id]: { inputs: { vae_name: "vae1.safetensors" }, class_type: vae.node.type },
			[encode.id]: { inputs: { pixels: ["6", 0], vae: [vae.id + "", 0] }, class_type: encode.node.type },
			[decode2.id]: { inputs: { samples: [encode.id + "", 0], vae: [vae.id + "", 0] }, class_type: decode2.node.type },
			[preview.id]: { inputs: { images: [decode2.id + "", 0] }, class_type: preview.node.type },
		});
	});
	test("works with IMAGEUPLOAD widget", async () => {
		const { ez, graph, app } = await start();
		const img = ez.LoadImage();
		const preview1 = ez.PreviewImage(img.outputs[0]);

		const group = await convertToGroup(app, graph, "test", [img, preview1]);
		const widget = group.widgets["upload"];
		expect(widget).toBeTruthy();
		expect(widget.widget.type).toBe("button");
	});
	test("internal primitive populates widgets for all linked inputs", async () => {
		const { ez, graph, app } = await start();
		const img = ez.LoadImage();
		const scale1 = ez.ImageScale(img.outputs[0]);
		const scale2 = ez.ImageScale(img.outputs[0]);
		ez.PreviewImage(scale1.outputs[0]);
		ez.PreviewImage(scale2.outputs[0]);

		scale1.widgets.width.convertToInput();
		scale2.widgets.height.convertToInput();

		const primitive = ez.PrimitiveNode();
		primitive.outputs[0].connectTo(scale1.inputs.width);
		primitive.outputs[0].connectTo(scale2.inputs.height);

		const group = await convertToGroup(app, graph, "test", [img, primitive, scale1, scale2]);
		group.widgets.value.value = 100;
		expect((await graph.toPrompt()).output).toEqual({
			1: {
				inputs: { image: img.widgets.image.value, upload: "image" },
				class_type: "LoadImage",
			},
			2: {
				inputs: { upscale_method: "nearest-exact", width: 100, height: 512, crop: "disabled", image: ["1", 0] },
				class_type: "ImageScale",
			},
			3: {
				inputs: { upscale_method: "nearest-exact", width: 512, height: 100, crop: "disabled", image: ["1", 0] },
				class_type: "ImageScale",
			},
			4: { inputs: { images: ["2", 0] }, class_type: "PreviewImage" },
			5: { inputs: { images: ["3", 0] }, class_type: "PreviewImage" },
		});
	});
	test("primitive control widgets values are copied on convert", async () => {
		const { ez, graph, app } = await start();
		const sampler = ez.KSampler();
		sampler.widgets.seed.convertToInput();
		sampler.widgets.sampler_name.convertToInput();

		let p1 = ez.PrimitiveNode();
		let p2 = ez.PrimitiveNode();
		p1.outputs[0].connectTo(sampler.inputs.seed);
		p2.outputs[0].connectTo(sampler.inputs.sampler_name);

		p1.widgets.control_after_generate.value = "increment";
		p2.widgets.control_after_generate.value = "decrement";
		p2.widgets.control_filter_list.value = "/.*/";

		p2.node.title = "p2";

		const group = await convertToGroup(app, graph, "test", [sampler, p1, p2]);
		expect(group.widgets.control_after_generate.value).toBe("increment");
		expect(group.widgets["p2 control_after_generate"].value).toBe("decrement");
		expect(group.widgets["p2 control_filter_list"].value).toBe("/.*/");

		group.widgets.control_after_generate.value = "fixed";
		group.widgets["p2 control_after_generate"].value = "randomize";
		group.widgets["p2 control_filter_list"].value = "/.+/";

		group.menu["Convert to nodes"].call();
		p1 = graph.find(p1);
		p2 = graph.find(p2);

		expect(p1.widgets.control_after_generate.value).toBe("fixed");
		expect(p2.widgets.control_after_generate.value).toBe("randomize");
		expect(p2.widgets.control_filter_list.value).toBe("/.+/");
	});
	test("internal reroutes work with converted inputs and merge options", async () => {
		const { ez, graph, app } = await start();
		const vae = ez.VAELoader();
		const latent = ez.EmptyLatentImage();
		const decode = ez.VAEDecode(latent.outputs.LATENT, vae.outputs.VAE);
		const scale = ez.ImageScale(decode.outputs.IMAGE);
		ez.PreviewImage(scale.outputs.IMAGE);

		const r1 = ez.Reroute();
		const r2 = ez.Reroute();

		latent.widgets.width.value = 64;
		latent.widgets.height.value = 128;

		latent.widgets.width.convertToInput();
		latent.widgets.height.convertToInput();
		latent.widgets.batch_size.convertToInput();

		scale.widgets.width.convertToInput();
		scale.widgets.height.convertToInput();

		r1.inputs[0].input.label = "hbw";
		r1.outputs[0].connectTo(latent.inputs.height);
		r1.outputs[0].connectTo(latent.inputs.batch_size);
		r1.outputs[0].connectTo(scale.inputs.width);

		r2.inputs[0].input.label = "wh";
		r2.outputs[0].connectTo(latent.inputs.width);
		r2.outputs[0].connectTo(scale.inputs.height);

		const group = await convertToGroup(app, graph, "test", [r1, r2, latent, decode, scale]);

		expect(group.inputs[0].input.type).toBe("VAE");
		expect(group.inputs[1].input.type).toBe("INT");
		expect(group.inputs[2].input.type).toBe("INT");

		const p1 = ez.PrimitiveNode();
		const p2 = ez.PrimitiveNode();
		p1.outputs[0].connectTo(group.inputs[1]);
		p2.outputs[0].connectTo(group.inputs[2]);

		expect(p1.widgets.value.widget.options?.min).toBe(16); // width/height min
		expect(p1.widgets.value.widget.options?.max).toBe(4096); // batch max
		expect(p1.widgets.value.widget.options?.step).toBe(80); // width/height step * 10

		expect(p2.widgets.value.widget.options?.min).toBe(16); // width/height min
		expect(p2.widgets.value.widget.options?.max).toBe(16384); // width/height max
		expect(p2.widgets.value.widget.options?.step).toBe(80); // width/height step * 10

		expect(p1.widgets.value.value).toBe(128);
		expect(p2.widgets.value.value).toBe(64);

		p1.widgets.value.value = 16;
		p2.widgets.value.value = 32;

		await checkBeforeAndAfterReload(graph, async (r) => {
			const id = (v) => (r ? `${group.id}:` : "") + v;
			expect((await graph.toPrompt()).output).toStrictEqual({
				1: { inputs: { vae_name: "vae1.safetensors" }, class_type: "VAELoader" },
				[id(2)]: { inputs: { width: 32, height: 16, batch_size: 16 }, class_type: "EmptyLatentImage" },
				[id(3)]: { inputs: { samples: [id(2), 0], vae: ["1", 0] }, class_type: "VAEDecode" },
				[id(4)]: {
					inputs: { upscale_method: "nearest-exact", width: 16, height: 32, crop: "disabled", image: [id(3), 0] },
					class_type: "ImageScale",
				},
				5: { inputs: { images: [id(4), 0] }, class_type: "PreviewImage" },
			});
		});
	});
	test("converted inputs with linked widgets map values correctly on creation", async () => {
		const { ez, graph, app } = await start();
		const k1 = ez.KSampler();
		const k2 = ez.KSampler();
		k1.widgets.seed.convertToInput();
		k2.widgets.seed.convertToInput();

		const rr = ez.Reroute();
		rr.outputs[0].connectTo(k1.inputs.seed);
		rr.outputs[0].connectTo(k2.inputs.seed);

		const group = await convertToGroup(app, graph, "test", [k1, k2, rr]);
		expect(group.widgets.steps.value).toBe(20);
		expect(group.widgets.cfg.value).toBe(8);
		expect(group.widgets.scheduler.value).toBe("normal");
		expect(group.widgets["KSampler steps"].value).toBe(20);
		expect(group.widgets["KSampler cfg"].value).toBe(8);
		expect(group.widgets["KSampler scheduler"].value).toBe("normal");
	});
	test("allow multiple of the same node type to be added", async () => {
		const { ez, graph, app } = await start();
		const nodes = [...Array(10)].map(() => ez.ImageScaleBy());
		const group = await convertToGroup(app, graph, "test", nodes);
		expect(group.inputs.length).toBe(10);
		expect(group.outputs.length).toBe(10);
		expect(group.widgets.length).toBe(20);
		expect(group.widgets.map((w) => w.widget.name)).toStrictEqual(
			[...Array(10)]
				.map((_, i) => `${i > 0 ? "ImageScaleBy " : ""}${i > 1 ? i + " " : ""}`)
				.flatMap((p) => [`${p}upscale_method`, `${p}scale_by`])
		);
	});
});
