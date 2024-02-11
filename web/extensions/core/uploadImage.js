import { app } from "../../scripts/app.js";

// Adds an upload button to the nodes

app.registerExtension({
	name: "ccniy.UploadImage",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeData?.input?.required?.image?.[1]?.image_upload === true) {
			nodeData.input.required.upload = ["IMAGEUPLOAD"];
		}
	},
});
