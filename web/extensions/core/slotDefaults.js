import { app } from "/scripts/app.js";
import { ComfyWidgets } from "/scripts/widgets.js";
// Adds defaults for quickly adding nodes with middle click on the input/output

app.registerExtension({
	name: "Comfy.SlotDefaults",
	suggestionsNumber: null,
	init() {
		LiteGraph.middle_click_slot_add_default_node = true;
		this.suggestionsNumber = app.ui.settings.addSetting({
			id: "Comfy.NodeSuggestions.number",
			name: "number of nodes suggestions",
			type: "slider",
			attrs: {
				min: 1,
				max: 100,
				step: 1,
			},
			defaultValue: 5,
			onChange: (newVal, oldVal) => {
				this.setDefaults(newVal);
			}
		});
	},
	slot_types_default_out: {},
	slot_types_default_in: {},
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
				var nodeId = nodeData.name;
		var inputs = [];
		inputs = nodeData["input"]["required"]; //only show required inputs to reduce the mess also not logical to create node with optional inputs
		for (const inputKey in inputs) {
			var input = (inputs[inputKey]);
			if (typeof input[0] !== "string") continue;

			var type = input[0]
			if (type in ComfyWidgets) {
				var customProperties = input[1]
				if (!(customProperties?.forceInput)) continue; //ignore widgets that don't force input
			}

			if (!(type in this.slot_types_default_out)) {
				this.slot_types_default_out[type] = ["Reroute"];
			}
			if (this.slot_types_default_out[type].includes(nodeId)) continue;
			this.slot_types_default_out[type].push(nodeId);
		} 

		var outputs = nodeData["output"];
		for (const key in outputs) {
			var type = outputs[key];
			if (!(type in this.slot_types_default_in)) {
				this.slot_types_default_in[type] = ["Reroute"];// ["Reroute", "Primitive"];  primitive doesn't always work :'()
			}

			this.slot_types_default_in[type].push(nodeId);
		}
		var maxNum = this.suggestionsNumber ? this.suggestionsNumber.value : 5;
		this.setDefaults(maxNum);
	},
	setDefaults(maxNum) {

		LiteGraph.slot_types_default_out = {};
		LiteGraph.slot_types_default_in = {};

		for (const type in this.slot_types_default_out) {
			LiteGraph.slot_types_default_out[type] = this.slot_types_default_out[type].slice(0, maxNum);
		}
		for (const type in this.slot_types_default_in) {
			LiteGraph.slot_types_default_in[type] = this.slot_types_default_in[type].slice(0, maxNum);
		}
	}
});
