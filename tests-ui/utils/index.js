const { mockApi } = require("./setup");
const { Ez } = require("./ezgraph");

/**
 *
 * @param { Parameters<mockApi> } config
 * @returns
 */
export async function start(config = undefined) {
	mockApi(config);
	const { app } = require("../../web/scripts/app");
	await app.setup();
	return Ez.graph(app, global["LiteGraph"], global["LGraphCanvas"]);
}
