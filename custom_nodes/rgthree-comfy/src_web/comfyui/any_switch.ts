import type {
  ComfyApp,
  INodeInputSlot,
  INodeOutputSlot,
  LGraphNode,
  LLink,
} from "@comfyorg/frontend";
import type {ComfyNodeDef} from "typings/comfy.js";

import {app} from "scripts/app.js";
import {IoDirection, addConnectionLayoutSupport, followConnectionUntilType} from "./utils.js";
import {RgthreeBaseServerNode} from "./base_node.js";
import {NodeTypesString} from "./constants.js";
import {removeUnusedInputsFromEnd} from "./utils_inputs_outputs.js";
import {debounce} from "rgthree/common/shared_utils.js";

class RgthreeAnySwitch extends RgthreeBaseServerNode {
  static override title = NodeTypesString.ANY_SWITCH;
  static override type = NodeTypesString.ANY_SWITCH;
  static comfyClass = NodeTypesString.ANY_SWITCH;

  private stabilizeBound = this.stabilize.bind(this);
  private nodeType: string | string[] | null = null;

  constructor(title = RgthreeAnySwitch.title) {
    super(title);
    // Adding five. Note, configure will add as many as was in the stored workflow automatically.
    this.addAnyInput(5);
  }

  override onConnectionsChange(
    type: number,
    slotIndex: number,
    isConnected: boolean,
    linkInfo: LLink,
    ioSlot: INodeOutputSlot | INodeInputSlot,
  ) {
    super.onConnectionsChange?.(type, slotIndex, isConnected, linkInfo, ioSlot);
    this.scheduleStabilize();
  }

  onConnectionsChainChange() {
    this.scheduleStabilize();
  }

  scheduleStabilize(ms = 64) {
    return debounce(this.stabilizeBound, ms);
  }

  private addAnyInput(num = 1) {
    for (let i = 0; i < num; i++) {
      this.addInput(
        `any_${String(this.inputs.length + 1).padStart(2, "0")}`,
        (this.nodeType || "*") as string,
      );
    }
  }

  stabilize() {
    // First, clean up the dynamic number of inputs.
    removeUnusedInputsFromEnd(this, 4);
    this.addAnyInput();

    // We prefer the inputs, then the output.
    let connectedType = followConnectionUntilType(this, IoDirection.INPUT, undefined, true);
    if (!connectedType) {
      connectedType = followConnectionUntilType(this, IoDirection.OUTPUT, undefined, true);
    }
    // TODO: What this doesn't do is broadcast to other nodes when its type changes. Reroute node
    // does, but, for now, if this was connected to another Any Switch, say, the second one wouldn't
    // change its type when the first does. The user would need to change the connections.
    this.nodeType = connectedType?.type || "*";
    for (const input of this.inputs) {
      input.type = this.nodeType as string; // So, types can indeed be arrays,,
    }
    for (const output of this.outputs) {
      output.type = this.nodeType as string; // So, types can indeed be arrays,,
      output.label =
        output.type === "RGTHREE_CONTEXT"
          ? "CONTEXT"
          : Array.isArray(this.nodeType) || this.nodeType.includes(",")
            ? connectedType?.label || String(this.nodeType)
            : String(this.nodeType);
    }
  }

  static override setUp(comfyClass: typeof LGraphNode, nodeData: ComfyNodeDef) {
    RgthreeBaseServerNode.registerForOverride(comfyClass, nodeData, RgthreeAnySwitch);
    addConnectionLayoutSupport(RgthreeAnySwitch, app, [
      ["Left", "Right"],
      ["Right", "Left"],
    ]);
  }
}

app.registerExtension({
  name: "rgthree.AnySwitch",
  async beforeRegisterNodeDef(nodeType: typeof LGraphNode, nodeData: any, app: ComfyApp) {
    if (nodeData.name === "Any Switch (rgthree)") {
      RgthreeAnySwitch.setUp(nodeType, nodeData);
    }
  },
});
