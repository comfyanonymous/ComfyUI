import type {ISerialisedNode} from "@comfyorg/frontend";
import type {ComfyNodeDef} from "typings/comfy";

import {app} from "scripts/app.js";
import {RgthreeBaseServerNode} from "./base_node.js";
import {NodeTypesString} from "./constants.js";

class RgthreeImageOrLatentSize extends RgthreeBaseServerNode {
  static override title = NodeTypesString.IMAGE_OR_LATENT_SIZE;
  static override type = NodeTypesString.IMAGE_OR_LATENT_SIZE;
  static comfyClass = NodeTypesString.IMAGE_OR_LATENT_SIZE;

  static override setUp(comfyClass: typeof LGraphNode, nodeData: ComfyNodeDef) {
    RgthreeBaseServerNode.registerForOverride(comfyClass, nodeData, NODE_CLASS);
  }

  constructor(title = NODE_CLASS.title) {
    super(title);
  }

  override onNodeCreated() {
    super.onNodeCreated?.();

    // Litegraph uses an array of acceptable input types, even though ComfyUI's types don't type
    // it out that way.
    this.addInput("input", ["IMAGE", "LATENT", "MASK"] as any);
  }

  override configure(info: ISerialisedNode): void {
    super.configure(info);

    if (this.inputs?.length) {
      // Litegraph uses an array of acceptable input types, even though ComfyUI's types don't type
      // it out that way.
      this.inputs[0]!.type = ["IMAGE", "LATENT", "MASK"] as any;
    }
  }
}

/** An uniformed name reference to the node class. */
const NODE_CLASS = RgthreeImageOrLatentSize;

/** Register the node. */
app.registerExtension({
  name: "rgthree.ImageOrLatentSize",
  async beforeRegisterNodeDef(nodeType: typeof LGraphNode, nodeData: ComfyNodeDef) {
    if (nodeData.name === NODE_CLASS.type) {
      NODE_CLASS.setUp(nodeType, nodeData);
    }
  },
});
