import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { ComfyWidgets } from "/scripts/widgets.js";
const CONFIG = Symbol();
// const GET_CONFIG = Symbol();
import { GET_CONFIG } from "./widgetInputs.js";

function getConfig(widgetName) {
	const { nodeData } = this.constructor;
	return nodeData?.input?.required[widgetName] ?? nodeData?.input?.optional?.[widgetName];
}

// class InMemorySubflow extends LGraphNode {
//   constructor(title) {
//     super(title ?? "InMemorySubflow");
//   }

//   onConfigure() {
//     console.log(this);
//   }

//   getExportedOutput(slot) {
//     return this.subflow.extras.outputSlots[slot];
//   };

//   getExportedInput(slot) {
//     return this.subflow.extras.inputSlots[slot];
//   };

//   async refreshNode(subflow) {
//     if (!subflow) return;
//     this.updateSubflowPrompt(subflow);
//     this.refreshPins(subflow);
//   }

//   refreshPins(subflow) {
//     if(!subflow)
//       return;

//     subflow.extras = { inputSlots: [], outputSlots: [] };
//     const { inputSlots } = subflow.extras;
//     const { outputSlots } = subflow.extras;

//     // remove all existing pins
//     const numInputs = this.inputs?.length ?? 0;
//     const numOutputs = this.outputs?.length ?? 0;
//     for(let i = numInputs-1; i > -1; i--) {
//       this.removeInput(i);
//     }
//     for(let i = numOutputs-1; i > -1; i--) {
//       this.removeOutput(i);
//     }

//     const subflowNodes = subflow.nodes;
//     // add the new pins and keep track of where the exported vars go to within the inner nodes
//     for (const subflowNode of subflowNodes) {
//       const exports = subflowNode.properties?.exports;
//       if (exports) {
//         let pinNum = 0;
//         for (const inputRef of exports.inputs) {
//           console.log(subflowNode.inputs);
//           const input = subflowNode.inputs.find(q => q.name === inputRef);
//           if (!input) continue;
//           const { name, type, link, slot_index, ...extras } = input;
//           console.log("Input");
//           console.log(input);
//           console.log(extras);
//           if (extras.widget) {
//             extras.widget[GET_CONFIG] = () => config;
//           }
//           this.addInput(input.name, input.type, extras );
//           inputSlots.push([subflowNode.id, pinNum]);
//           pinNum++;
//         }
//         pinNum = 0;
//         for (const outputRef of exports.outputs) {
//           const output = subflowNode.outputs.find(q => q.name === outputRef);
//           if (!output) continue;
//           this.addOutput(output.name, output.type);
//           outputSlots.push([subflowNode.id, pinNum]);
//           pinNum++;
//         }
//       }
//     }

//     this.size[0] = 180;
//   };

//   updateSubflowPrompt(subflow) {
//     this.subflow = subflow;
//   };
// }

// class FileSubflow extends InMemorySubflow {
//   constructor() {
//     super("FileSubflow");

//     console.log("constructor called");
//     // ComfyWidgets.SUBFLOWUPLOAD(this, null, null, app);
//   }

//   async refreshNode(subflow) {
//     if (!subflow) return;

//     this.updateSubflowPrompt(subflow);
//     this.refreshPins(subflow);

//     this.size[0] = this.computeSizeX();
//   }

//   computeSizeX() {
//     return Math.max(100, LiteGraph.NODE_TEXT_SIZE * this.title.length * 0.45 + 160);
//   }
// }

app.registerExtension({
  name: "Comfy.Subflow",
  beforeRegisterNodeDef(nodeType, nodeData, app) {

    // console.log("in init")
		// LiteGraph.registerNodeType(
		// 	"InMemorySubflow",
		// 	Object.assign(InMemorySubflow, {
		// 		title: "InMemorySubflow",
		// 	})
		// );

    // LiteGraph.registerNodeType(
		// 	"FileSubflow",
		// 	Object.assign(FileSubflow, {
		// 		title: "FileSubflow",
		// 	})
    // );

    // FileSubflow.category = "utils";
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
          for (const inputRef of exports.inputs) {
            console.log(subflowNode.inputs);
            const input = subflowNode.inputs.find(q => q.name === inputRef);
            if (!input) continue;
            const { name, type, link, slot_index, ...extras } = input;
            console.log("Input");
            console.log(input);
            console.log(extras);
            if (extras.widget) {
              const w = extras.widget;
              const config = getConfig.call(this, input.name) ?? [input.type, w.options || {}];
              extras.widget[GET_CONFIG] = () => config;
              console.log(extras);
              console.log(input.type);
            }
            node.addInput(input.name, input.type, extras );
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
  
      node.size[0] = 180;
    };

    const refreshNode = async (node, subflow) => {
      if (!subflow) return;
  
      node.subflow = subflow;
      refreshPins(node, subflow);
  
      node.size[0] = Math.max(100, LiteGraph.NODE_TEXT_SIZE * node.title.length * 0.45 + 160);
    };

    // if (nodeData.name == "InMemorySubflow") {
    //   Object.assign(nodeData, new InMemorySubflow());
    // }

    if (nodeData.name == "FileSubflow") {
      nodeType.prototype.refreshNode = function(subflow) {  refreshNode(this, subflow); };
      nodeType.prototype.getExportedOutput = function(slot) { return this.subflow.extras.outputSlots[slot]; }
      nodeType.prototype.getExportedInput =  function(slot) { return this.subflow.extras.inputSlots[slot]; }

      nodeData.input.required = { subflow: ["SUBFLOWUPLOAD"] };
    }
	}
});
