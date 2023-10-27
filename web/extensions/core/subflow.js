import { app } from "../../scripts/app.js";

app.registerExtension({
  name: "Comfy.Subflow",
  beforeRegisterNodeDef(nodeType, nodeData, app) {
    const refreshPins = (node, subflow) => {
      if(!subflow)
        return;
  
      subflow.extras = { inputSlots: [], outputSlots: [] };
      const { inputSlots } = subflow.extras;
      const { outputSlots } = subflow.extras;
  
      // remove all existing pins
      const numInputs = node.inputs?.length ?? 0;
      const numOutputs = node.outputs?.length ?? 0;
      for(let i = numInputs-1; i > -1; i--) {
        node.removeInput(i);
      }
      for(let i = numOutputs-1; i > -1; i--) {
        node.removeOutput(i);
      }
  
      const subflowNodes = subflow.nodes;
      // add the new pins and keep track of where the exported vars go to within the inner nodes
      for (const subflowNode of subflowNodes) {
        const exports = subflowNode.properties?.exports;
        if (exports) {
          let pinNum = 0;
          for (const exportedInput of exports.inputs) {
            const input = subflowNode.inputs.find(q => q.name === exportedInput.name);
            if (!input) continue;
            const { name, type, link, slot_index, ...extras } = input;

            node.addInput(input.name, input.type, extras);
            inputSlots.push([subflowNode, pinNum]);
            pinNum++;
          }

          pinNum = 0;
          for (const exportedOutput of exports.outputs) {
            const output = subflowNode.outputs.find(q => q.name === exportedOutput.name);
            if (!output) continue;
            node.addOutput(output.name, output.type);
            outputSlots.push([subflowNode, pinNum]);
            pinNum++;
          }
        }
      }
  
      node.size[0] = 180;
    };

    const refreshWidgets = (node, subflow, recoverValues) => {
      if (!subflow)
        return;

      // Allow widgets to cleanup
      for (let i = 1; i < node.widgets.length; ++i) {
        if (node.widgets[i].onRemove) {
          node.widgets[i].onRemove();
        }
      }
      node.widgets = [node.widgets[0]];

      // Map widgets
      subflow.extras.widgetSlots = {};
      const subflowNodes = subflow.nodes;

      let widgetIndex = 1;
      for (const subflowNode of subflowNodes) {
        const exports = subflowNode.properties?.exports;
        if (exports) {

          for (const exportedWidget of exports.widgets) {
            subflow.extras.widgetSlots[exportedWidget.name] = subflowNode;

            let type = exportedWidget.config[0];
            let options = type;
            if (type instanceof Array) {
              options = { values: type };
              type = "combo";
            } else {
              options = exportedWidget.config[1];
            }
            if (type === "INT" || type === "FLOAT") {
              type = "number";
            }
            const getWidgetCallback = (widgetIndex) => {
              return (v) => {
                if (v !== null && node.widgets_values) {
                  node.widgets_values[widgetIndex] = v;
                }
              }
            };
            let value = exportedWidget.value;
            if (recoverValues) {
              value = node.widgets_values[widgetIndex] ?? value;
            }
            node.addWidget(type, exportedWidget.name, value, getWidgetCallback(widgetIndex), options);
            widgetIndex++;
          }

        }
      }
    };

    const refreshNode = (node, subflow, filename) => {
      if (!subflow) return;
  
      node.subflow = subflow;
      node.title = `File Subflow (Loaded: ${filename})`;
      refreshPins(node, subflow);
      refreshWidgets(node, subflow, false);
      
  
      node.size[0] = Math.max(100, LiteGraph.NODE_TEXT_SIZE * node.title.length * 0.45 + 100);
    };

    if (nodeData.name == "FileSubflow") {
      nodeType.prototype.onConfigure = function() { refreshWidgets(this, this.subflow, true); };
      nodeType.prototype.refreshNode = function(subflow, filename) { refreshNode(this, subflow, filename); };

      nodeData.input.required = { subflow: ["SUBFLOWUPLOAD"] };
    }
	}
});
