const fs = require("fs");
const path = require("path");
const { nop } = require("../utils/nopProxy");

function forEachKey(cb) {
	for (const k of [
		"LiteGraph",
		"LGraph",
		"LLink",
		"LGraphNode",
		"LGraphGroup",
		"DragAndScale",
		"LGraphCanvas",
		"ContextMenu",
	]) {
		cb(k);
	}
}

export function setup(ctx) {
	const lg = fs.readFileSync(path.resolve("../web/lib/litegraph.core.js"), "utf-8");
	const globalTemp = {};
	(function (console) {
		eval(lg);
	}).call(globalTemp, nop);

	forEachKey((k) => (ctx[k] = globalTemp[k]));
	require(path.resolve("../web/lib/litegraph.extensions.js"));
}

export function teardown(ctx) {
	forEachKey((k) => delete ctx[k]);

	// Clear document after each run
	document.getElementsByTagName("html")[0].innerHTML = ""; 
}
