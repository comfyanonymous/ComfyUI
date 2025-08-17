import { app } from "../../../scripts/app.js";

const REPEATER = "Repeater|pysssss";

app.registerExtension({
	name: "pysssss.Repeater",
	init() {
		const graphToPrompt = app.graphToPrompt;
		app.graphToPrompt = async function () {
			const res = await graphToPrompt.apply(this, arguments);

			const id = Date.now() + "_";
			let u = 0;

			let newNodes = {};
			const newRepeaters = {};
			for (const nodeId in res.output) {
				let output = res.output[nodeId];
				if (output.class_type === REPEATER) {
					const isMulti = output.inputs.output === "multi";
					if (output.inputs.node_mode === "create") {
						// We need to clone the input for every repeat
						const orig = res.output[output.inputs.source[0]];
						if (isMulti) {
							if (!newRepeaters[nodeId]) {
								newRepeaters[nodeId] = [];
								newRepeaters[nodeId][output.inputs.repeats - 1] = nodeId;
							}
						}
						for (let i = 0; i < output.inputs.repeats - 1; i++) {
							const clonedInputId = id + ++u;

							if (isMulti) {
								// If multi create we need to clone the repeater too
								newNodes[clonedInputId] = structuredClone(orig);

								output = structuredClone(output);

								const clonedRepeaterId = id + ++u;
								newNodes[clonedRepeaterId] = output;
								output.inputs["source"][0] = clonedInputId;

								newRepeaters[nodeId][i] = clonedRepeaterId;
							} else {
								newNodes[clonedInputId] = orig;
							}
							output.inputs[clonedInputId] = [clonedInputId, output.inputs.source[1]];
						}
					} else if (isMulti) {
						newRepeaters[nodeId] = Array(output.inputs.repeats).fill(nodeId);
					}
				}
			}

			Object.assign(res.output, newNodes);
			newNodes = {};

			for (const nodeId in res.output) {
				const output = res.output[nodeId];
				for (const k in output.inputs) {
					const v = output.inputs[k];
					if (v instanceof Array) {
						const repeaterId = v[0];
						const source = newRepeaters[repeaterId];
						if (source) {
							v[0] = source.pop();
							v[1] = 0;
						}
					}
				}
			}

			// Object.assign(res.output, newNodes);

			return res;
		};
	},
	beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeData.name === REPEATER) {
			const SETUP_OUTPUTS = Symbol();
			nodeType.prototype[SETUP_OUTPUTS] = function (repeats) {
				if (repeats == null) {
					repeats = this.widgets[0].value;
				}
				while (this.outputs.length > repeats) {
					this.removeOutput(repeats);
				}
				const id = Date.now() + "_";
				let u = 0;
				while (this.outputs.length < repeats) {
					this.addOutput(id + ++u, "*", { label: "*" });
				}
			};

			const onAdded = nodeType.prototype.onAdded;
			nodeType.prototype.onAdded = function () {
				const self = this;
				const repeatsCb = this.widgets[0].callback;
				this.widgets[0].callback = async function () {
					const v = (await repeatsCb?.apply(this, arguments)) ?? this.value;
					if (self.widgets[1].value === "multi") {
						self[SETUP_OUTPUTS](v);
					}
					return v;
				};

				const outputCb = this.widgets[1].callback;
				this.widgets[1].callback = async function () {
					const v = (await outputCb?.apply(this, arguments)) ?? this.value;
					if (v === "single") {
						self.outputs[0].shape = 6;
						self[SETUP_OUTPUTS](1);
					} else {
						delete self.outputs[0].shape;
						self[SETUP_OUTPUTS]();
					}
					return v;
				};
				return onAdded?.apply(this, arguments);
			};
		}
	},
});
