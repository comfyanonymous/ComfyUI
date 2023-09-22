import { app } from "../../scripts/app.js";

const REGEX_MULTIPLE_INPUT = /(.+)\[(\d*)\]$/

/**
* move an existing input slot to another slot position
* @method moveInput
* @param {number} slot
* @param {number} to_slot
*/
function moveInput(node, slot, to_slot) {
	if (slot == to_slot) {
		return
	}

	var slot_info = node.inputs.splice(slot, 1); // remove
	node.inputs.splice(to_slot, 0, slot_info[0]) // add

	for (var i = Math.min(slot, to_slot); i <= Math.max(slot, to_slot); i++) {
		if (!node.inputs[i]) {
			continue;
		}
		var link = node.graph.links[node.inputs[i].link];
		if (!link) {
			continue;
		}
		link.target_slot = i
	}

	if (node.onInputMoved) {
		node.onInputMoved(slot, to_slot, slot_info[0]);
	}
	node.setDirtyCanvas(true, true);
}

app.registerExtension({
	name: "Comfy.MultipleInputs",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		const origOnInputAdded = nodeType.prototype.onInputAdded
		nodeType.prototype.onInputAdded = function (input) {
			const r = origOnInputAdded ? origOnInputAdded.apply(this, arguments) : undefined;
			if (input?.multiple) {
				var m = input.name.match(REGEX_MULTIPLE_INPUT)
				if (m) {
					if (input?.label == null) {
						input.label = m[1]
					}
				} else {
					if (input?.label == null) {
						input.label = input.name
					}
					input.name = input.name + "[0]"
				}
			}
			return r
		}

		const origOnConnectionsChange = nodeType.prototype.onConnectionsChange
		nodeType.prototype.onConnectionsChange = function (connectionKind, slot, connected, linkInfo, inputInfo) {
			const r = origOnConnectionsChange ? origOnConnectionsChange.apply(this, arguments) : undefined;
			if (connectionKind == LiteGraph.INPUT) {
				var m = inputInfo.name.match(REGEX_MULTIPLE_INPUT)
				if (m && inputInfo?.multiple) {
					var count = parseInt(m[2] || 0);
					var inputCountNotConnected = 0;
					var lastInputSlot = null

					for (var i = 0; i < this.inputs.length; i++) {
						var mi = this.inputs[i].name.match(REGEX_MULTIPLE_INPUT)
						if (mi && mi[1] === m[1]) {
							lastInputSlot = i
							if (this.inputs[i].link) {
								count = Math.max(count, parseInt(mi[2] || 0));
								inputCountNotConnected = 0
							} else {
								inputCountNotConnected++
							}
						}
					}

					count++

					if (inputCountNotConnected == 0 && connected && inputInfo.link !== null) {
						var inputName = m[1] + "[" + count + "]";
						var inputType = inputInfo.type;

						var extraInfo = {}
						for (var i in inputInfo) {
							if (i !== "name" && i !== "type" && i !== "link") {
								extraInfo[i] = inputInfo[i]
							}
						}

						this.addInput(inputName, inputType, extraInfo)

						moveInput(this, this.inputs.length - 1, lastInputSlot + 1)
					}

					inputCountNotConnected = 0;
					var removeInputs = []

					for (var i = 0; i < this.inputs.length; i++) {
						var mi = this.inputs[i].name.match(REGEX_MULTIPLE_INPUT)
						if (mi && mi[1] === m[1]) {
							if (this.inputs[i].link) {
								removeInputs = []
								inputCountNotConnected = 0
							} else {
								inputCountNotConnected++
								if (inputCountNotConnected > 1) {
									removeInputs.unshift(this.inputs[i].name)
								}
							}
						}
					}

					for (var i = this.inputs.length - 1; i >= 0; i--) {
						if (removeInputs.includes(this.inputs[i].name)) {
							this.removeInput(i)
						}
					}
				}
			}
			return r;
		}
	},
});
