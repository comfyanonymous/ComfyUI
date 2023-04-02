import { app } from "/scripts/app.js";

// Adds defaults for quickly adding nodes with middle click on the input/output

app.registerExtension({
	name: "Comfy.SlotDefaults",
	init() {
		LiteGraph.middle_click_slot_add_default_node = true;
		LiteGraph.slot_types_default_in = {
			MODEL: "CheckpointLoaderSimple",
			LATENT: "EmptyLatentImage",
			VAE: "VAELoader",
		};

		LiteGraph.slot_types_default_out = {
			LATENT: "VAEDecode",
			IMAGE: "SaveImage",
			CLIP: "CLIPTextEncode",
		};
	},
});
