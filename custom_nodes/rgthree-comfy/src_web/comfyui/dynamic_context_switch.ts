import type {
  LGraphNode,
  LLink,
  LGraphCanvas,
  INodeInputSlot,
  INodeOutputSlot,
  ISlotType,
} from "@comfyorg/frontend";
import type {ComfyNodeDef} from "typings/comfy.js";

import {app} from "scripts/app.js";
import {DynamicContextNodeBase, InputLike} from "./dynamic_context_base.js";
import {NodeTypesString} from "./constants.js";
import {
  InputMutation,
  SERVICE as CONTEXT_SERVICE,
  getContextOutputName,
} from "./services/context_service.js";
import {getConnectedInputNodesAndFilterPassThroughs} from "./utils.js";
import {debounce, moveArrayItem} from "rgthree/common/shared_utils.js";
import {measureText} from "./utils_canvas.js";
import {SERVICE as CONFIG_SERVICE} from "./services/config_service.js";

type ShadowInputData = {
  node: LGraphNode;
  slot: number;
  shadowIndex: number;
  shadowIndexIfShownSingularly: number;
  shadowIndexFull: number;
  nodeIndex: number;
  type: string | -1;
  name: string;
  key: string;
  // isDuplicatedBefore: boolean,
  duplicatesBefore: number[];
  duplicatesAfter: number[];
};

/**
 * The Context Switch  node.
 */
class DynamicContextSwitchNode extends DynamicContextNodeBase {
  static override title = NodeTypesString.DYNAMIC_CONTEXT_SWITCH;
  static override type = NodeTypesString.DYNAMIC_CONTEXT_SWITCH;
  static comfyClass = NodeTypesString.DYNAMIC_CONTEXT_SWITCH;

  protected override readonly hasShadowInputs = true;

  // override hasShadowInputs = true;

  /**
   * We should be able to assume that `lastInputsList` is the input list after the last, major
   * synchronous change. Which should mean, if we're handling a change that is currently live, but
   * not represented in our node (like, an upstream node has already removed an input), then we
   * should be able to compar the current InputList to this `lastInputsList`.
   */
  lastInputsList: ShadowInputData[] = [];

  private shadowInputs: (InputLike & {count: number})[] = [
    {name: "base_ctx", type: "RGTHREE_DYNAMIC_CONTEXT", link: null, count: 0, boundingRect: null},
  ];

  constructor(title = DynamicContextSwitchNode.title) {
    super(title);
  }

  override getContextInputsList() {
    return this.shadowInputs;
  }
  override handleUpstreamMutation(mutation: InputMutation) {
    this.scheduleHardRefresh();
  }

  override onConnectionsChange(
    type: ISlotType,
    slotIndex: number,
    isConnected: boolean,
    link: LLink | null | undefined,
    inputOrOutput: INodeInputSlot | INodeOutputSlot,
  ): void {
    super.onConnectionsChange?.call(this, type, slotIndex, isConnected, link, inputOrOutput);
    if (this.configuring) {
      return;
    }
    if (type === LiteGraph.INPUT) {
      this.scheduleHardRefresh();
    }
  }

  scheduleHardRefresh(ms = 64) {
    return debounce(() => {
      this.refreshInputsAndOutputs();
    }, ms);
  }

  override onNodeCreated() {
    this.addInput("ctx_1", "RGTHREE_DYNAMIC_CONTEXT");
    this.addInput("ctx_2", "RGTHREE_DYNAMIC_CONTEXT");
    this.addInput("ctx_3", "RGTHREE_DYNAMIC_CONTEXT");
    this.addInput("ctx_4", "RGTHREE_DYNAMIC_CONTEXT");
    this.addInput("ctx_5", "RGTHREE_DYNAMIC_CONTEXT");
    super.onNodeCreated();
  }

  override addContextInput(name: string, type: string, slot?: number): void {}

  /**
   * This is a "hard" refresh of the list, but looping over the actual context inputs, and
   * recompiling the shadowInputs and outputs.
   */
  private refreshInputsAndOutputs() {
    const inputs: (InputLike & {count: number})[] = [
      {name: "base_ctx", type: "RGTHREE_DYNAMIC_CONTEXT", link: null, count: 0, boundingRect: null},
    ];
    let numConnected = 0;
    for (let i = 0; i < this.inputs.length; i++) {
      const childCtxs = getConnectedInputNodesAndFilterPassThroughs(
        this,
        this,
        i,
      ) as DynamicContextNodeBase[];
      if (childCtxs.length > 1) {
        throw new Error("How is there more than one input?");
      }
      const ctx = childCtxs[0];
      if (!ctx) continue;
      numConnected++;
      const slotsData = CONTEXT_SERVICE.getDynamicContextInputsData(ctx);
      console.log(slotsData);
      for (const slotData of slotsData) {
        const found = inputs.find(
          (n) => getContextOutputName(slotData.name) === getContextOutputName(n.name),
        );
        if (found) {
          found.count += 1;
          continue;
        }
        inputs.push({
          name: slotData.name,
          type: slotData.type,
          link: null,
          count: 1,
          boundingRect: null,
        });
      }
    }
    this.shadowInputs = inputs;
    // First output is always CONTEXT, so "p" is the offset.
    let i = 0;
    for (i; i < this.shadowInputs.length; i++) {
      const data = this.shadowInputs[i]!;
      let existing = this.outputs.find(
        (o) => getContextOutputName(o.name) === getContextOutputName(data.name),
      );
      if (!existing) {
        existing = this.addOutput(getContextOutputName(data.name), data.type);
      }
      moveArrayItem(this.outputs, existing, i);
      delete existing.rgthree_status;
      if (data.count !== numConnected) {
        existing.rgthree_status = "WARN";
      }
    }
    while (this.outputs[i]) {
      const output = this.outputs[i];
      if (output?.links?.length) {
        output.rgthree_status = "ERROR";
        i++;
      } else {
        this.removeOutput(i);
      }
    }
    this.fixInputsOutputsLinkSlots();
  }

  override onDrawForeground(ctx: CanvasRenderingContext2D, canvas: LGraphCanvas): void {
    const low_quality = (canvas?.ds?.scale ?? 1) < 0.6;
    if (low_quality || this.size[0] <= 10) {
      return;
    }
    let y = LiteGraph.NODE_SLOT_HEIGHT - 1;
    const w = this.size[0];
    ctx.save();
    ctx.font = "normal " + LiteGraph.NODE_SUBTEXT_SIZE + "px Arial";
    ctx.textAlign = "right";

    for (const output of this.outputs) {
      if (!output.rgthree_status) {
        y += LiteGraph.NODE_SLOT_HEIGHT;
        continue;
      }
      const x = w - 20 - measureText(ctx, output.name);
      if (output.rgthree_status === "ERROR") {
        ctx.fillText("🛑", x, y);
      } else if (output.rgthree_status === "WARN") {
        ctx.fillText("⚠️", x, y);
      }
      y += LiteGraph.NODE_SLOT_HEIGHT;
    }
    ctx.restore();
  }
}

app.registerExtension({
  name: "rgthree.DynamicContextSwitch",
  async beforeRegisterNodeDef(nodeType: typeof LGraphNode, nodeData: ComfyNodeDef) {
    if (!CONFIG_SERVICE.getConfigValue("unreleased.dynamic_context.enabled")) {
      return;
    }
    if (nodeData.name === DynamicContextSwitchNode.type) {
      DynamicContextSwitchNode.setUp(nodeType, nodeData);
    }
  },
});
