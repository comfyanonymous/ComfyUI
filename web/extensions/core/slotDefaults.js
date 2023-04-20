import { app } from "/scripts/app.js";
import { ComfyWidgets } from "/scripts/widgets.js";
// Adds defaults for quickly adding nodes with middle click on the input/output

app.registerExtension({
	name: "Comfy.SlotDefaults",
	init() {
		LiteGraph.middle_click_slot_add_default_node = true;
	},
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		var nodeId = nodeData.name;
		var inputs = [];
		//if (nodeData["input"]["optional"] != undefined) {
		//	inputs = Object.assign({}, nodeData["input"]["required"], nodeData["input"]["optional"]);
		//} else {
		inputs = nodeData["input"]["required"]; //only show required inputs to reduce the mess also not logica to create node with optional inputs
		//}
		for (const inputKey in inputs) {
			var input = (inputs[inputKey]);
			//make sure input[0] is a string
			if (typeof input[0] !== "string") continue;

			//	for (const slotKey in inputs[inputKey]) {
			var type = input[0]
			if (type in ComfyWidgets) {
				var customProperties = input[1]
				//console.log(customProperties)
				if (!(customProperties?.forceInput)) continue; //ignore widgets that don't force input
			}

			if (!(type in LiteGraph.slot_types_default_out)) {
				LiteGraph.slot_types_default_out[type] = ["Reroute"];
			}
			if (LiteGraph.slot_types_default_out[type].includes(nodeId)) continue;
			LiteGraph.slot_types_default_out[type].push(nodeId);
			//	}
		}

		var outputs = nodeData["output"];
		for (const key in outputs) {
			var type = outputs[key];
			if (!(type in LiteGraph.slot_types_default_in)) {
				LiteGraph.slot_types_default_in[type] = ["Reroute"];// ["Reroute", "Primitive"];  primitive doesn't always work :'()
			}
			LiteGraph.slot_types_default_in[type].push(nodeId);
		}

	},
});
