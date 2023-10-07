import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
  name: "Comfy.Subflow",
  async nodeCreated(node) {

    if (!node.widgets) return;
    if (node.widgets[0].name !== "subflow_name") return;

    const refreshPins = (subflowNodes) => {
      // remove all existing pins
      const numInputs = node.inputs.length;
      const numOutputs = node.outputs.length;
      for(let i = numInputs-1; i > -1; i--) {
        node.removeInput(i);
      }
      for(let i = numOutputs-1; i > -1; i--) {
        node.removeOutput(i);
      }

      for (const subflowNode of subflowNodes) {
        const exports = subflowNode.properties.exports;
        if (exports) {
          for (const inputRef of exports.inputs) {
            const input = subflowNode.inputs.find(q => q.name === inputRef);
            if (!input) continue;
            node.addInput(input.name, input.type);
          }
          for (const outputRef of exports.outputs) {
            const output = subflowNode.outputs.find(q => q.name === outputRef);
            if (!output) continue;
            node.addOutput(output.name, output.type);
          }
        }
      }
    };

    node.onConfigure = async function () {
      const subflowData = await api.getSubflow(node.widgets[0].value);
      if (subflowData.subflow) {
        refreshPins(subflowData.subflow.nodes);
      }
		};

    node.widgets[0].callback = async function (subflowName) {
      const subflowData = await api.getSubflow(subflowName);
      if (subflowData.subflow) {
        refreshPins(subflowData.subflow.nodes);
      }
		};

  }
});
