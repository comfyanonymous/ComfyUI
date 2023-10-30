const { mockApi } = require("./setup");
const { Ez } = require("./ezgraph");

/**
 *
 * @param { Parameters<mockApi>[0] } config
 * @returns
 */
export async function start(config = undefined) {
	mockApi(config);
	const { app } = require("../../web/scripts/app");
	await app.setup();
	return Ez.graph(app, global["LiteGraph"], global["LGraphCanvas"]);
}

/**
 * @param { ReturnType<Ez["graph"]>["graph"] } graph 
 * @param { (hasReloaded: boolean) => (Promise<void> | void) } cb 
 */
export async function checkBeforeAndAfterReload(graph, cb) {
	await cb(false);
	await graph.reload();
	await cb(true);
}

/**
 * @param { string } name 
 * @param { Record<string, string | [string | string[], any]> } input 
 * @param { (string | string[])[] | Record<string, string | string[]> } output
 * @returns { Record<string, import("../../web/types/comfy").ComfyObjectInfo> } 
 */
export function makeNodeDef(name, input, output = {}) {
	const nodeDef = {
		name,
		category: "test",
		output: [],
		output_name: [],
		output_is_list: [],
		input: {
			required: {}
		},
	};
	for(const k in input) {
		nodeDef.input.required[k] = typeof input[k] === "string" ? [input[k], {}] : [...input[k]];
	}
	if(output instanceof Array) {
		output = output.reduce((p, c) => {
			p[c] = c;
			return p;
		}, {})
	}
	for(const k in output) {
		nodeDef.output.push(output[k]);
		nodeDef.output_name.push(k);
		nodeDef.output_is_list.push(false);
	}

	return { [name]: nodeDef };
}

/**
/**
 * @template { any } T
 * @param { T } x
 * @returns { x is Exclude<T, null | undefined> }
 */
export function assertNotNullOrUndefined(x) {
	expect(x).not.toEqual(null);
	expect(x).not.toEqual(undefined);
	return true;
}