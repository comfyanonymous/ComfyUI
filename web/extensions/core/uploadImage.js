import { app } from "/scripts/app.js";

app.registerExtension({
	name: "Comfy.UploadImage",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeData.name === "UploadImage" || nodeData.name === "UploadImageMask") {
			nodeData.input.required.upload = ["IMAGEUPLOAD"];
		}
	},
});
