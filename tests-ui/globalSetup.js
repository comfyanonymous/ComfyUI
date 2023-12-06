module.exports = async function () {
	global.ResizeObserver = class ResizeObserver {
		observe() {}
		unobserve() {}
		disconnect() {}
	};

	const { nop } = require("./utils/nopProxy");
	global.enableWebGLCanvas = nop;

	HTMLCanvasElement.prototype.getContext = nop;

	localStorage["Comfy.Settings.Comfy.Logging.Enabled"] = "false";
};
