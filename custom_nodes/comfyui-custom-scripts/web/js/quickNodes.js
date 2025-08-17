import { app } from "../../../scripts/app.js";

// Adds a bunch of context menu entries for quickly adding common steps

function addMenuHandler(nodeType, cb) {
	const getOpts = nodeType.prototype.getExtraMenuOptions;
	nodeType.prototype.getExtraMenuOptions = function () {
		const r = getOpts.apply(this, arguments);
		cb.apply(this, arguments);
		return r;
	};
}

function getOrAddVAELoader(node) {
	let vaeNode = app.graph._nodes.find((n) => n.type === "VAELoader");
	if (!vaeNode) {
		vaeNode = addNode("VAELoader", node);
	}
	return vaeNode;
}

function addNode(name, nextTo, options) {
	options = { select: true, shiftY: 0, before: false, ...(options || {}) };
	const node = LiteGraph.createNode(name);
	app.graph.add(node);
	node.pos = [
		options.before ? nextTo.pos[0] - node.size[0] - 30 : nextTo.pos[0] + nextTo.size[0] + 30,
		nextTo.pos[1] + options.shiftY,
	];
	if (options.select) {
		app.canvas.selectNode(node, false);
	}
	return node;
}

app.registerExtension({
	name: "pysssss.QuickNodes",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeData.input && nodeData.input.required) {
			const keys = Object.keys(nodeData.input.required);
			for (let i = 0; i < keys.length; i++) {
				if (nodeData.input.required[keys[i]][0] === "VAE") {
					addMenuHandler(nodeType, function (_, options) {
						options.unshift({
							content: "Use VAE",
							callback: () => {
								getOrAddVAELoader(this).connect(0, this, i);
							},
						});
					});
					break;
				}
			}
		}

		if (nodeData.name === "KSampler") {
			addMenuHandler(nodeType, function (_, options) {
				options.unshift(
					{
						content: "Add Blank Input",
						callback: () => {
							const imageNode = addNode("EmptyLatentImage", this, { before: true });
							imageNode.connect(0, this, 3);
						},
					},
					{
						content: "Add Hi-res Fix",
						callback: () => {
							const upscaleNode = addNode("LatentUpscale", this);
							this.connect(0, upscaleNode, 0);

							const sampleNode = addNode("KSampler", upscaleNode);

							for (let i = 0; i < 3; i++) {
								const l = this.getInputLink(i);
								if (l) {
									app.graph.getNodeById(l.origin_id).connect(l.origin_slot, sampleNode, i);
								}
							}

							upscaleNode.connect(0, sampleNode, 3);
						},
					},
					{
						content: "Add 2nd Pass",
						callback: () => {
							const upscaleNode = addNode("LatentUpscale", this);
							this.connect(0, upscaleNode, 0);

							const ckptNode = addNode("CheckpointLoaderSimple", this);
							const sampleNode = addNode("KSampler", ckptNode);

							const positiveLink = this.getInputLink(1);
							const negativeLink = this.getInputLink(2);
							const positiveNode = positiveLink
								? app.graph.add(app.graph.getNodeById(positiveLink.origin_id).clone())
								: addNode("CLIPTextEncode");
							const negativeNode = negativeLink
								? app.graph.add(app.graph.getNodeById(negativeLink.origin_id).clone())
								: addNode("CLIPTextEncode");

							ckptNode.connect(0, sampleNode, 0);
							ckptNode.connect(1, positiveNode, 0);
							ckptNode.connect(1, negativeNode, 0);
							positiveNode.connect(0, sampleNode, 1);
							negativeNode.connect(0, sampleNode, 2);
							upscaleNode.connect(0, sampleNode, 3);
						},
					},
					{
						content: "Add Save Image",
						callback: () => {
							const decodeNode = addNode("VAEDecode", this);
							this.connect(0, decodeNode, 0);

							getOrAddVAELoader(decodeNode).connect(0, decodeNode, 1);

							const saveNode = addNode("SaveImage", decodeNode);
							decodeNode.connect(0, saveNode, 0);
						},
					}
				);
			});
		}

		if (nodeData.name === "CheckpointLoaderSimple") {
			addMenuHandler(nodeType, function (_, options) {
				options.unshift({
					content: "Add Clip Skip",
					callback: () => {
						const clipSkipNode = addNode("CLIPSetLastLayer", this);
						const clipLinks = this.outputs[1].links ? this.outputs[1].links.map((l) => ({ ...graph.links[l] })) : [];

						this.disconnectOutput(1);
						this.connect(1, clipSkipNode, 0);

						for (const clipLink of clipLinks) {
							clipSkipNode.connect(0, clipLink.target_id, clipLink.target_slot);
						}
					},
				});
			});
		}

		if (
			nodeData.name === "CheckpointLoaderSimple" ||
			nodeData.name === "CheckpointLoader" ||
			nodeData.name === "CheckpointLoader|pysssss" ||
			nodeData.name === "LoraLoader" ||
			nodeData.name === "LoraLoader|pysssss"
		) {
			addMenuHandler(nodeType, function (_, options) {
				function addLora(type) {
					const loraNode = addNode(type, this);

					const modelLinks = this.outputs[0].links ? this.outputs[0].links.map((l) => ({ ...graph.links[l] })) : [];
					const clipLinks = this.outputs[1].links ? this.outputs[1].links.map((l) => ({ ...graph.links[l] })) : [];

					this.disconnectOutput(0);
					this.disconnectOutput(1);

					this.connect(0, loraNode, 0);
					this.connect(1, loraNode, 1);

					for (const modelLink of modelLinks) {
						loraNode.connect(0, modelLink.target_id, modelLink.target_slot);
					}

					for (const clipLink of clipLinks) {
						loraNode.connect(1, clipLink.target_id, clipLink.target_slot);
					}
				}
				options.unshift(
					{
						content: "Add LoRA",
						callback: () => addLora.call(this, "LoraLoader"),
					},
					{
						content: "Add 🐍 LoRA",
						callback: () => addLora.call(this, "LoraLoader|pysssss"),
					},
					{
						content: "Add Prompts",
						callback: () => {
							const positiveNode = addNode("CLIPTextEncode", this);
							const negativeNode = addNode("CLIPTextEncode", this, { shiftY: positiveNode.size[1] + 30 });

							this.connect(1, positiveNode, 0);
							this.connect(1, negativeNode, 0);
						},
					}
				);
			});
		}
	},
});
