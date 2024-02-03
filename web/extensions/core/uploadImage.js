import { app } from "../../scripts/app.js";

// Adds an upload button to the nodes

app.registerExtension({
	name: "Comfy.UploadImage",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeData?.input?.required?.image?.[1]?.image_upload === true) {
			nodeData.input.required.upload = ["IMAGEUPLOAD"];
		}
		// 检查艺术家图像上传
		if (nodeData?.input?.required?.image?.[1]?.image_upload_artist === true) {
			nodeData.input.required.upload = ["ARTISTS_IMAGEUPLOAD"];
		}
		// 检查相机图像上传
		if (nodeData?.input?.required?.image?.[1]?.image_upload_camera === true) {
			nodeData.input.required.upload = ["CAMERAS_IMAGEUPLOAD"];
		}
		// 检查胶片图像上传
		if (nodeData?.input?.required?.image?.[1]?.image_upload_film === true) {
			nodeData.input.required.upload = ["FILMS_IMAGEUPLOAD"];
		}
		// 检查艺术运动图像上传
		if (nodeData?.input?.required?.image?.[1]?.image_upload_movement === true) {
			nodeData.input.required.upload = ["MOVEMENTS_IMAGEUPLOAD"];
		}
		// 检查风格图像上传
		if (nodeData?.input?.required?.image?.[1]?.image_upload_style === true) {
			nodeData.input.required.upload = ["STYLES_IMAGEUPLOAD"];


		}
	},
});
