import type {
  INodeInputSlot,
  INodeOutputSlot,
  LGraphCanvas as TLGraphCanvas,
  LGraphNode as TLGraphNode,
  LLink,
  ISlotType,
  ConnectByTypeOptions,
} from "@comfyorg/frontend";
import type {ComfyNodeDef} from "typings/comfy.js";

import {app} from "scripts/app.js";
import {
  IoDirection,
  addConnectionLayoutSupport,
  addMenuItem,
  matchLocalSlotsToServer,
  replaceNode,
} from "./utils.js";
import {RgthreeBaseServerNode} from "./base_node.js";
import {SERVICE as KEY_EVENT_SERVICE} from "./services/key_events_services.js";
import {RgthreeBaseServerNodeConstructor} from "typings/rgthree.js";
import {debounce, wait} from "rgthree/common/shared_utils.js";
import {removeUnusedInputsFromEnd} from "./utils_inputs_outputs.js";
import {NodeTypesString} from "./constants.js";

/**
 * Takes a non-context node and determins for its input or output slot, if there is a valid
 * connection for an opposite context output or input slot.
 */
function findMatchingIndexByTypeOrName(
  otherNode: TLGraphNode,
  otherSlot: INodeInputSlot | INodeOutputSlot,
  ctxSlots: INodeInputSlot[] | INodeOutputSlot[],
) {
  const otherNodeType = (otherNode.type || "").toUpperCase();
  const otherNodeName = (otherNode.title || "").toUpperCase();
  let otherSlotType = otherSlot.type as string;
  if (Array.isArray(otherSlotType) || otherSlotType.includes(",")) {
    otherSlotType = "COMBO";
  }
  const otherSlotName = otherSlot.name.toUpperCase().replace("OPT_", "").replace("_NAME", "");
  let ctxSlotIndex = -1;
  if (["CONDITIONING", "INT", "STRING", "FLOAT", "COMBO"].includes(otherSlotType)) {
    ctxSlotIndex = ctxSlots.findIndex((ctxSlot) => {
      const ctxSlotName = ctxSlot.name.toUpperCase().replace("OPT_", "").replace("_NAME", "");
      let ctxSlotType = ctxSlot.type as string;
      if (Array.isArray(ctxSlotType) || ctxSlotType.includes(",")) {
        ctxSlotType = "COMBO";
      }
      if (ctxSlotType !== otherSlotType) {
        return false;
      }
      // Straightforward matches.
      if (
        ctxSlotName === otherSlotName ||
        (ctxSlotName === "SEED" && otherSlotName.includes("SEED")) ||
        (ctxSlotName === "STEP_REFINER" && otherSlotName.includes("AT_STEP")) ||
        (ctxSlotName === "STEP_REFINER" && otherSlotName.includes("REFINER_STEP"))
      ) {
        return true;
      }
      // If postive other node, try to match conditining and text.
      if (
        (otherNodeType.includes("POSITIVE") || otherNodeName.includes("POSITIVE")) &&
        ((ctxSlotName === "POSITIVE" && otherSlotType === "CONDITIONING") ||
          (ctxSlotName === "TEXT_POS_G" && otherSlotName.includes("TEXT_G")) ||
          (ctxSlotName === "TEXT_POS_L" && otherSlotName.includes("TEXT_L")))
      ) {
        return true;
      }
      if (
        (otherNodeType.includes("NEGATIVE") || otherNodeName.includes("NEGATIVE")) &&
        ((ctxSlotName === "NEGATIVE" && otherSlotType === "CONDITIONING") ||
          (ctxSlotName === "TEXT_NEG_G" && otherSlotName.includes("TEXT_G")) ||
          (ctxSlotName === "TEXT_NEG_L" && otherSlotName.includes("TEXT_L")))
      ) {
        return true;
      }
      return false;
    });
  } else {
    ctxSlotIndex = ctxSlots.map((s) => s.type).indexOf(otherSlotType);
  }
  return ctxSlotIndex;
}

/**
 * A Base Context node for other context based nodes to extend.
 */
export class BaseContextNode extends RgthreeBaseServerNode {
  constructor(title: string) {
    super(title);
  }

  // LiteGraph adds more spacing than we want when calculating a nodes' `_collapsed_width`, so we'll
  // override it with a setter and re-set it measured exactly as we want.
  ___collapsed_width: number = 0;

  //@ts-ignore - TS Doesn't like us overriding a property with accessors but, too bad.
  override get _collapsed_width() {
    return this.___collapsed_width;
  }

  override set _collapsed_width(width: number) {
    const canvas = app.canvas as TLGraphCanvas;
    const ctx = canvas.canvas.getContext("2d")!;
    const oldFont = ctx.font;
    ctx.font = canvas.title_text_font;
    let title = this.title.trim();
    this.___collapsed_width = 30 + (title ? 10 + ctx.measureText(title).width : 0);
    ctx.font = oldFont;
  }

  override connectByType(
    slot: number | string,
    targetNode: TLGraphNode,
    targetSlotType: ISlotType,
    optsIn?: ConnectByTypeOptions,
  ): LLink | null {
    let canConnect = super.connectByType?.call(this, slot, targetNode, targetSlotType, optsIn);
    if (!super.connectByType) {
      canConnect = LGraphNode.prototype.connectByType.call(
        this,
        slot,
        targetNode,
        targetSlotType,
        optsIn,
      );
    }
    if (!canConnect && slot === 0) {
      const ctrlKey = KEY_EVENT_SERVICE.ctrlKey;
      // Okay, we've dragged a context and it can't connect.. let's connect all the other nodes.
      // Unfortunately, we don't know which are null now, so we'll just connect any that are
      // not already connected.
      for (const [index, input] of (targetNode.inputs || []).entries()) {
        if (input.link && !ctrlKey) {
          continue;
        }
        const thisOutputSlot = findMatchingIndexByTypeOrName(targetNode, input, this.outputs);
        if (thisOutputSlot > -1) {
          this.connect(thisOutputSlot, targetNode, index);
        }
      }
    }
    return null;
  }

  override connectByTypeOutput(
    slot: number | string,
    sourceNode: TLGraphNode,
    sourceSlotType: ISlotType,
    optsIn?: ConnectByTypeOptions,
  ): LLink | null {
    let canConnect = super.connectByTypeOutput?.call(
      this,
      slot,
      sourceNode,
      sourceSlotType,
      optsIn,
    );
    if (!super.connectByType) {
      canConnect = LGraphNode.prototype.connectByTypeOutput.call(
        this,
        slot,
        sourceNode,
        sourceSlotType,
        optsIn,
      );
    }
    if (!canConnect && slot === 0) {
      const ctrlKey = KEY_EVENT_SERVICE.ctrlKey;
      // Okay, we've dragged a context and it can't connect.. let's connect all the other nodes.
      // Unfortunately, we don't know which are null now, so we'll just connect any that are
      // not already connected.
      for (const [index, output] of (sourceNode.outputs || []).entries()) {
        if (output.links?.length && !ctrlKey) {
          continue;
        }
        const thisInputSlot = findMatchingIndexByTypeOrName(sourceNode, output, this.inputs);
        if (thisInputSlot > -1) {
          sourceNode.connect(index, this, thisInputSlot);
        }
      }
    }
    return null;
  }

  static override setUp(
    comfyClass: typeof LGraphNode,
    nodeData: ComfyNodeDef,
    ctxClass: RgthreeBaseServerNodeConstructor,
  ) {
    RgthreeBaseServerNode.registerForOverride(comfyClass, nodeData, ctxClass);
    // [🤮] ComfyUI only adds "required" inputs to the outputs list when dragging an output to
    // empty space, but since RGTHREE_CONTEXT is optional, it doesn't get added to the menu because
    // ...of course. So, we'll manually add it. Of course, we also have to do this in a timeout
    // because ComfyUI clears out `LiteGraph.slot_types_default_out` in its own 'Comfy.SlotDefaults'
    // extension and we need to wait for that to happen.
    wait(500).then(() => {
      LiteGraph.slot_types_default_out["RGTHREE_CONTEXT"] =
        LiteGraph.slot_types_default_out["RGTHREE_CONTEXT"] || [];
      LiteGraph.slot_types_default_out["RGTHREE_CONTEXT"].push((comfyClass as any).comfyClass);
    });
  }

  static override onRegisteredForOverride(comfyClass: any, ctxClass: any) {
    addConnectionLayoutSupport(ctxClass, app, [
      ["Left", "Right"],
      ["Right", "Left"],
    ]);
    setTimeout(() => {
      ctxClass.category = comfyClass.category;
    });
  }
}

/**
 * The original Context node.
 */
class ContextNode extends BaseContextNode {
  static override title = NodeTypesString.CONTEXT;
  static override type = NodeTypesString.CONTEXT;
  static comfyClass = NodeTypesString.CONTEXT;

  constructor(title = ContextNode.title) {
    super(title);
  }

  static override setUp(comfyClass: typeof LGraphNode, nodeData: ComfyNodeDef) {
    BaseContextNode.setUp(comfyClass, nodeData, ContextNode);
  }

  static override onRegisteredForOverride(comfyClass: any, ctxClass: any) {
    BaseContextNode.onRegisteredForOverride(comfyClass, ctxClass);
    addMenuItem(ContextNode, app, {
      name: "Convert To Context Big",
      callback: (node) => {
        replaceNode(node, ContextBigNode.type);
      },
    });
  }
}

/**
 * The Context Big node.
 */
class ContextBigNode extends BaseContextNode {
  static override title = NodeTypesString.CONTEXT_BIG;
  static override type = NodeTypesString.CONTEXT_BIG;
  static comfyClass = NodeTypesString.CONTEXT_BIG;

  constructor(title = ContextBigNode.title) {
    super(title);
  }

  static override setUp(comfyClass: typeof LGraphNode, nodeData: ComfyNodeDef) {
    BaseContextNode.setUp(comfyClass, nodeData, ContextBigNode);
  }

  static override onRegisteredForOverride(comfyClass: any, ctxClass: any) {
    BaseContextNode.onRegisteredForOverride(comfyClass, ctxClass);
    addMenuItem(ContextBigNode, app, {
      name: "Convert To Context (Original)",
      callback: (node) => {
        replaceNode(node, ContextNode.type);
      },
    });
  }
}

/**
 * A base node for Context Switche nodes and Context Merges nodes that will always add another empty
 * ctx input, no less than five.
 */
class BaseContextMultiCtxInputNode extends BaseContextNode {
  private stabilizeBound = this.stabilize.bind(this);

  constructor(title: string) {
    super(title);
    // Adding five. Note, configure will add as many as was in the stored workflow automatically.
    this.addContextInput(5);
  }

  private addContextInput(num = 1) {
    for (let i = 0; i < num; i++) {
      this.addInput(`ctx_${String(this.inputs.length + 1).padStart(2, "0")}`, "RGTHREE_CONTEXT");
    }
  }

  override onConnectionsChange(
    type: number,
    slotIndex: number,
    isConnected: boolean,
    link: LLink,
    ioSlot: INodeInputSlot | INodeOutputSlot,
  ): void {
    super.onConnectionsChange?.apply(this, [...arguments] as any);
    if (type === LiteGraph.INPUT) {
      this.scheduleStabilize();
    }
  }

  private scheduleStabilize(ms = 64) {
    return debounce(this.stabilizeBound, 64);
  }

  /**
   * Stabilizes the inputs; removing any disconnected ones from the bottom, then adding an empty
   * one to the end so we always have one empty one to expand.
   */
  private stabilize() {
    removeUnusedInputsFromEnd(this, 4);
    this.addContextInput();
  }
}

/**
 * The Context Switch (original) node.
 */
class ContextSwitchNode extends BaseContextMultiCtxInputNode {
  static override title = NodeTypesString.CONTEXT_SWITCH;
  static override type = NodeTypesString.CONTEXT_SWITCH;
  static comfyClass = NodeTypesString.CONTEXT_SWITCH;

  constructor(title = ContextSwitchNode.title) {
    super(title);
  }

  static override setUp(comfyClass: typeof LGraphNode, nodeData: ComfyNodeDef) {
    BaseContextNode.setUp(comfyClass, nodeData, ContextSwitchNode);
  }

  static override onRegisteredForOverride(comfyClass: any, ctxClass: any) {
    BaseContextNode.onRegisteredForOverride(comfyClass, ctxClass);
    addMenuItem(ContextSwitchNode, app, {
      name: "Convert To Context Switch Big",
      callback: (node) => {
        replaceNode(node, ContextSwitchBigNode.type);
      },
    });
  }
}

/**
 * The Context Switch Big node.
 */
class ContextSwitchBigNode extends BaseContextMultiCtxInputNode {
  static override title = NodeTypesString.CONTEXT_SWITCH_BIG;
  static override type = NodeTypesString.CONTEXT_SWITCH_BIG;
  static comfyClass = NodeTypesString.CONTEXT_SWITCH_BIG;

  constructor(title = ContextSwitchBigNode.title) {
    super(title);
  }

  static override setUp(comfyClass: typeof LGraphNode, nodeData: ComfyNodeDef) {
    BaseContextNode.setUp(comfyClass, nodeData, ContextSwitchBigNode);
  }

  static override onRegisteredForOverride(comfyClass: any, ctxClass: any) {
    BaseContextNode.onRegisteredForOverride(comfyClass, ctxClass);
    addMenuItem(ContextSwitchBigNode, app, {
      name: "Convert To Context Switch",
      callback: (node) => {
        replaceNode(node, ContextSwitchNode.type);
      },
    });
  }
}

/**
 * The Context Merge (original) node.
 */
class ContextMergeNode extends BaseContextMultiCtxInputNode {
  static override title = NodeTypesString.CONTEXT_MERGE;
  static override type = NodeTypesString.CONTEXT_MERGE;
  static comfyClass = NodeTypesString.CONTEXT_MERGE;

  constructor(title = ContextMergeNode.title) {
    super(title);
  }

  static override setUp(comfyClass: typeof LGraphNode, nodeData: ComfyNodeDef) {
    BaseContextNode.setUp(comfyClass, nodeData, ContextMergeNode);
  }

  static override onRegisteredForOverride(comfyClass: any, ctxClass: any) {
    BaseContextNode.onRegisteredForOverride(comfyClass, ctxClass);
    addMenuItem(ContextMergeNode, app, {
      name: "Convert To Context Merge Big",
      callback: (node) => {
        replaceNode(node, ContextMergeBigNode.type);
      },
    });
  }
}

/**
 * The Context Switch Big node.
 */
class ContextMergeBigNode extends BaseContextMultiCtxInputNode {
  static override title = NodeTypesString.CONTEXT_MERGE_BIG;
  static override type = NodeTypesString.CONTEXT_MERGE_BIG;
  static comfyClass = NodeTypesString.CONTEXT_MERGE_BIG;

  constructor(title = ContextMergeBigNode.title) {
    super(title);
  }

  static override setUp(comfyClass: typeof LGraphNode, nodeData: ComfyNodeDef) {
    BaseContextNode.setUp(comfyClass, nodeData, ContextMergeBigNode);
  }

  static override onRegisteredForOverride(comfyClass: any, ctxClass: any) {
    BaseContextNode.onRegisteredForOverride(comfyClass, ctxClass);
    addMenuItem(ContextMergeBigNode, app, {
      name: "Convert To Context Switch",
      callback: (node) => {
        replaceNode(node, ContextMergeNode.type);
      },
    });
  }
}

const contextNodes = [
  ContextNode,
  ContextBigNode,
  ContextSwitchNode,
  ContextSwitchBigNode,
  ContextMergeNode,
  ContextMergeBigNode,
];
const contextTypeToServerDef: {[type: string]: ComfyNodeDef} = {};

function fixBadConfigs(node: ContextNode) {
  // Dumb mistake, but let's fix our mispelling. This will probably need to stay in perpetuity to
  // keep any old workflows operating.
  const wrongName = node.outputs.find((o, i) => o.name === "CLIP_HEIGTH");
  if (wrongName) {
    wrongName.name = "CLIP_HEIGHT";
  }
}

app.registerExtension({
  name: "rgthree.Context",
  async beforeRegisterNodeDef(nodeType: typeof LGraphNode, nodeData: ComfyNodeDef) {
    // Loop over out context nodes and see if any match the server data.
    for (const ctxClass of contextNodes) {
      if (nodeData.name === ctxClass.type) {
        contextTypeToServerDef[ctxClass.type] = nodeData;
        ctxClass.setUp(nodeType, nodeData);
        break;
      }
    }
  },

  async nodeCreated(node: TLGraphNode) {
    const type = node.type || (node.constructor as any).type;
    const serverDef = type && contextTypeToServerDef[type];
    if (serverDef) {
      fixBadConfigs(node as ContextNode);
      matchLocalSlotsToServer(node, IoDirection.OUTPUT, serverDef);
      // Switches don't need to change inputs, only context outputs
      if (!type!.includes("Switch") && !type!.includes("Merge")) {
        matchLocalSlotsToServer(node, IoDirection.INPUT, serverDef);
      }
      // }, 100);
    }
  },

  /**
   * When we're loaded from the server, check if we're using an out of date version and update our
   * inputs / outputs to match.
   */
  async loadedGraphNode(node: TLGraphNode) {
    const type = node.type || (node.constructor as any).type;
    const serverDef = type && contextTypeToServerDef[type];
    if (serverDef) {
      fixBadConfigs(node as ContextNode);
      matchLocalSlotsToServer(node, IoDirection.OUTPUT, serverDef);
      // Switches don't need to change inputs, only context outputs
      if (!type!.includes("Switch") && !type!.includes("Merge")) {
        matchLocalSlotsToServer(node, IoDirection.INPUT, serverDef);
      }
    }
  },
});
