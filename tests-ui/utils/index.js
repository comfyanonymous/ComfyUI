const { mockApi } = require("./setup");
const { Ez } = require("./ezgraph");
const lg = require("./litegraph");
const fs = require("fs");
const path = require("path");

const html = fs.readFileSync(path.resolve(__dirname, "../../web/index.html"))

/**
 *
 * @param { Parameters<typeof mockApi>[0] & { 
 * 	resetEnv?: boolean, 
 * 	preSetup?(app): Promise<void>,
 *  localStorage?: Record<string, string> 
 * } } config
 * @returns
 */
export async function start(config = {}) {
	if(config.resetEnv) {
		jest.resetModules();
		jest.resetAllMocks();
        lg.setup(global);
		localStorage.clear();
		sessionStorage.clear();
	}

	Object.assign(localStorage, config.localStorage ?? {});
	document.body.innerHTML = html;

	mockApi(config);
	const { app } = require("../../web/scripts/app");
	config.preSetup?.(app);
	await app.setup();

	return { ...Ez.graph(app, global["LiteGraph"], global["LGraphCanvas"]), app };
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
			required: {},
		},
	};
	for (const k in input) {
		nodeDef.input.required[k] = typeof input[k] === "string" ? [input[k], {}] : [...input[k]];
	}
	if (output instanceof Array) {
		output = output.reduce((p, c) => {
			p[c] = c;
			return p;
		}, {});
	}
	for (const k in output) {
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

/**
 *
 * @param { ReturnType<Ez["graph"]>["ez"] } ez
 * @param { ReturnType<Ez["graph"]>["graph"] } graph
 */
export function createDefaultWorkflow(ez, graph) {
	graph.clear();
	const ckpt = ez.CheckpointLoaderSimple();

	const pos = ez.CLIPTextEncode(ckpt.outputs.CLIP, { text: "positive" });
	const neg = ez.CLIPTextEncode(ckpt.outputs.CLIP, { text: "negative" });

	const empty = ez.EmptyLatentImage();
	const sampler = ez.KSampler(
		ckpt.outputs.MODEL,
		pos.outputs.CONDITIONING,
		neg.outputs.CONDITIONING,
		empty.outputs.LATENT
	);

	const decode = ez.VAEDecode(sampler.outputs.LATENT, ckpt.outputs.VAE);
	const save = ez.SaveImage(decode.outputs.IMAGE);
	graph.arrange();

	return { ckpt, pos, neg, empty, sampler, decode, save };
}

export async function getNodeDefs() {
	const { api } = require("../../web/scripts/api");
	return api.getNodeDefs();
}

export async function getNodeDef(nodeId) {
	return (await getNodeDefs())[nodeId];
}