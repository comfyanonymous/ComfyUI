import type {LGraphNode, LGraphNodeConstructor} from "@comfyorg/frontend";
import type {ComfyNodeDef} from "typings/comfy.js";

import {app} from "scripts/app.js";
import {addConnectionLayoutSupport} from "./utils.js";
import {PowerPrompt} from "./base_power_prompt.js";
import {NodeTypesString} from "./constants.js";

let nodeData: ComfyNodeDef | null = null;
app.registerExtension({
  name: "rgthree.PowerPrompt",
  async beforeRegisterNodeDef(nodeType: typeof LGraphNode, passedNodeData: ComfyNodeDef) {
    if (passedNodeData.name.includes("Power Prompt") && passedNodeData.name.includes("rgthree")) {
      nodeData = passedNodeData;
      const onNodeCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        onNodeCreated ? onNodeCreated.apply(this, []) : undefined;
        (this as any).powerPrompt = new PowerPrompt(this as LGraphNode, passedNodeData);
      };
      addConnectionLayoutSupport(nodeType as LGraphNodeConstructor, app, [
        ["Left", "Right"],
        ["Right", "Left"],
      ]);
    }
  },
  async loadedGraphNode(node: LGraphNode) {
    if (node.type === NodeTypesString.POWER_PROMPT) {
      setTimeout(() => {
        // If the first output is STRING, then it's the text output from the initial launch.
        // Let's port it to the new
        if (node.outputs[0]!.type === "STRING") {
          if (node.outputs[0]!.links) {
            node.outputs[3]!.links = node.outputs[3]!.links || [];
            for (const link of node.outputs[0]!.links) {
              node.outputs[3]!.links.push(link);
              (node.graph || app.graph).links[link]!.origin_slot = 3;
            }
            node.outputs[0]!.links = null;
          }
          node.outputs[0]!.type = nodeData!.output![0] as string;
          node.outputs[0]!.name = nodeData!.output_name![0] || (node.outputs[0]!.type as string);
          node.outputs[0]!.color_on = undefined;
          node.outputs[0]!.color_off = undefined;
        }
      }, 50);
    }
  },
});
