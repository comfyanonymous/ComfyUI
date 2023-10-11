import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

class InMemorySubflow {
  constructor() {}

  onConfigure() {
    console.log(this);
  }

  getExportedOutput(slot) {
    return this.subflow.extras.outputSlots[slot];
  };

  getExportedInput(slot) {
    return this.subflow.extras.inputSlots[slot];
  };

  async refreshNode(subflow) {
    if (!subflow) return;
    this.updateSubflowPrompt(subflow);
    this.refreshPins(subflow);
  }

  refreshPins(subflow) {
    if(!subflow)
      return;

    subflow.extras = { inputSlots: [], outputSlots: [] };
    const { inputSlots } = subflow.extras;
    const { outputSlots } = subflow.extras;

    // remove all existing pins
    const numInputs = this.inputs?.length ?? 0;
    const numOutputs = this.outputs?.length ?? 0;
    for(let i = numInputs-1; i > -1; i--) {
      this.removeInput(i);
    }
    for(let i = numOutputs-1; i > -1; i--) {
      this.removeOutput(i);
    }

    const subflowNodes = subflow.nodes;
    // add the new pins and keep track of where the exported vars go to within the inner nodes
    for (const subflowNode of subflowNodes) {
      const exports = subflowNode.properties?.exports;
      if (exports) {
        let pinNum = 0;
        for (const inputRef of exports.inputs) {
          const input = subflowNode.inputs.find(q => q.name === inputRef);
          if (!input) continue;
          this.addInput(input.name, input.type);
          inputSlots.push([subflowNode.id, pinNum]);
          pinNum++;
        }
        pinNum = 0;
        for (const outputRef of exports.outputs) {
          const output = subflowNode.outputs.find(q => q.name === outputRef);
          if (!output) continue;
          this.addOutput(output.name, output.type);
          outputSlots.push([subflowNode.id, pinNum]);
          pinNum++;
        }
      }
    }

    this.size[0] = 180;
  };

  updateSubflowPrompt(subflow) {
    this.subflow = subflow;
  };
}

app.registerExtension({
  name: "Comfy.Subflow",
  async nodeCreated(node) {
    if (!node.widgets) return;
    if (node.widgets[0].name !== "subflow_name") return;

    let outputSlots = []; // (int (node), int (slot))
    let inputSlots = [];

    const refreshNode = async (subflowName) => {
      const subflowData = await api.getSubflow(subflowName);
      if (!subflowData.subflow) return;

      const subflowNodes = subflowData.subflow.nodes
      updateSubflowPrompt(subflowData.subflow);
      refreshPins(subflowNodes);

      node.size[0] = computeSizeX(subflowName);
    };

    const refreshPins = (subflowNodes) => {
      inputSlots = [];
      outputSlots = [];

      // remove all existing pins
      const numInputs = node.inputs?.length ?? 0;
      const numOutputs = node.outputs?.length ?? 0;
      for(let i = numInputs-1; i > -1; i--) {
        node.removeInput(i);
      }
      for(let i = numOutputs-1; i > -1; i--) {
        node.removeOutput(i);
      }

      // add the new pins and keep track of where the exported vars go to within the inner nodes
      for (const subflowNode of subflowNodes) {
        const exports = subflowNode.properties.exports;
        if (exports) {
          let pinNum = 0;
          for (const inputRef of exports.inputs) {
            const input = subflowNode.inputs.find(q => q.name === inputRef);
            if (!input) continue;
            node.addInput(input.name, input.type);
            inputSlots.push([subflowNode.id, pinNum]);
            pinNum++;
          }
          pinNum = 0;
          for (const outputRef of exports.outputs) {
            const output = subflowNode.outputs.find(q => q.name === outputRef);
            if (!output) continue;
            node.addOutput(output.name, output.type);
            outputSlots.push([subflowNode.id, pinNum]);
            pinNum++;
          }
        }
      }
    };

    const updateSubflowPrompt = (subflow) => {
      node.subflow = subflow;
    };

    const computeSizeX = (subflowName) => {
      return Math.max(100, LiteGraph.NODE_TEXT_SIZE * subflowName.length * 0.45 + 160);
    };

    node.widgets[0].callback = (subflowName) => refreshNode(subflowName);

    node.getExportedOutput = (slot) => {
      return outputSlots[slot];
    };

    node.getExportedInput = (slot) => {
      return inputSlots[slot];
    };
  },
  registerCustomNodes() {
		LiteGraph.registerNodeType(
			"InMemorySubflow",
			Object.assign(InMemorySubflow, {
				title: "InMemorySubflow",
			})
		);
	},
});
