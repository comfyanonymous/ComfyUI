import { app } from "/scripts/app.js";

// Adds an upload button to the nodes

app.registerExtension({
	name: "Comfy.UploadImage",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		switch (nodeData.name) {
		case "LoadImage":
		case "LoadImageMask":
			nodeData.input.required.upload = ["IMAGEUPLOAD"];
			break;
		}
	},
});
