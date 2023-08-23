import { LiteGraph, LGraph } from "../lib/litegraph.core.js"

export default class ComfyGraph extends LGraph {
	constructor(app) {
		super();
		this.app = app;
	}
}
