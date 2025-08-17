import type {
  Vector2,
  LLink,
  INodeInputSlot,
  INodeOutputSlot,
  LGraphNode as TLGraphNode,
  ISlotType,
  ConnectByTypeOptions,
  TWidgetType,
  IWidgetOptions,
  IWidget,
  IBaseWidget,
  WidgetTypeMap,
} from "@comfyorg/frontend";

import {app} from "scripts/app.js";
import {RgthreeBaseVirtualNode} from "./base_node.js";
import {rgthree} from "./rgthree.js";
import {
  PassThroughFollowing,
  addConnectionLayoutSupport,
  addMenuItem,
  getConnectedInputNodes,
  getConnectedInputNodesAndFilterPassThroughs,
  getConnectedOutputNodes,
  getConnectedOutputNodesAndFilterPassThroughs,
} from "./utils.js";

/**
 * A Virtual Node that allows any node's output to connect to it.
 */
export class BaseAnyInputConnectedNode extends RgthreeBaseVirtualNode {
  override isVirtualNode = true;

  /**
   * Whether inputs show the immediate nodes, or follow and show connected nodes through
   * passthrough nodes.
   */
  readonly inputsPassThroughFollowing: PassThroughFollowing = PassThroughFollowing.NONE;

  debouncerTempWidth: number = 0;
  schedulePromise: Promise<void> | null = null;

  constructor(title = BaseAnyInputConnectedNode.title) {
    super(title);
  }

  override onConstructed() {
    this.addInput("", "*");
    return super.onConstructed();
  }

  override clone() {
    const cloned = super.clone()!;
    // Copying to clipboard (and also, creating node templates) work by cloning nodes and, for some
    // reason, it manually manipulates the cloned data. So, we want to keep the present input slots
    // so if it's pasted/templatized the data is correct. Otherwise, clear the inputs and so the new
    // node is ready to go, fresh.
    if (!rgthree.canvasCurrentlyCopyingToClipboardWithMultipleNodes) {
      while (cloned.inputs.length > 1) {
        cloned.removeInput(cloned.inputs.length - 1);
      }
      if (cloned.inputs[0]) {
        cloned.inputs[0].label = "";
      }
    }
    return cloned;
  }

  /**
   * Schedules a promise to run a stabilization, debouncing duplicate requests.
   */
  scheduleStabilizeWidgets(ms = 100) {
    if (!this.schedulePromise) {
      this.schedulePromise = new Promise((resolve) => {
        setTimeout(() => {
          this.schedulePromise = null;
          this.doStablization();
          resolve();
        }, ms);
      });
    }
    return this.schedulePromise;
  }

  /**
   * Ensures we have at least one empty input at the end, returns true if changes were made, or false
   * if no changes were needed.
   */
  private stabilizeInputsOutputs(): boolean {
    let changed = false;
    const hasEmptyInput = !this.inputs[this.inputs.length - 1]?.link;
    if (!hasEmptyInput) {
      this.addInput("", "*");
      changed = true;
    }
    for (let index = this.inputs.length - 2; index >= 0; index--) {
      const input = this.inputs[index]!;
      if (!input.link) {
        this.removeInput(index);
        changed = true;
      } else {
        const node = getConnectedInputNodesAndFilterPassThroughs(
          this,
          this,
          index,
          this.inputsPassThroughFollowing,
        )[0];
        const newName = node?.title || "";
        if (input.name !== newName) {
          input.name = node?.title || "";
          changed = true;
        }
      }
    }
    return changed;
  }

  /**
   * Stabilizes the node's inputs and widgets.
   */
  private doStablization() {
    if (!this.graph) {
      return;
    }
    let dirty = false;

    // When we add/remove widgets, litegraph is going to mess up the size, so we
    // store it so we can retrieve it in computeSize. Hacky..
    (this as any)._tempWidth = this.size[0];

    dirty = this.stabilizeInputsOutputs();
    const linkedNodes = getConnectedInputNodesAndFilterPassThroughs(this);
    dirty = this.handleLinkedNodesStabilization(linkedNodes) || dirty;

    // Only mark dirty if something's changed.
    if (dirty) {
      this.graph.setDirtyCanvas(true, true);
    }

    // Schedule another stabilization in the future.
    this.scheduleStabilizeWidgets(500);
  }

  /**
   * Handles stabilization of linked nodes. To be overridden. Should return true if changes were
   * made, or false if no changes were needed.
   */
  handleLinkedNodesStabilization(linkedNodes: TLGraphNode[]): boolean {
    linkedNodes; // No-op, but makes overridding in VSCode cleaner.
    throw new Error("handleLinkedNodesStabilization should be overridden.");
  }

  onConnectionsChainChange() {
    this.scheduleStabilizeWidgets();
  }

  override onConnectionsChange(
    type: number,
    index: number,
    connected: boolean,
    linkInfo: LLink,
    ioSlot: INodeOutputSlot | INodeInputSlot,
  ) {
    super.onConnectionsChange &&
      super.onConnectionsChange(type, index, connected, linkInfo, ioSlot);
    if (!linkInfo) return;
    // Follow outputs to see if we need to trigger an onConnectionChange.
    const connectedNodes = getConnectedOutputNodesAndFilterPassThroughs(this);
    for (const node of connectedNodes) {
      if ((node as BaseAnyInputConnectedNode).onConnectionsChainChange) {
        (node as BaseAnyInputConnectedNode).onConnectionsChainChange();
      }
    }
    this.scheduleStabilizeWidgets();
  }

  override removeInput(slot: number) {
    (this as any)._tempWidth = this.size[0];
    return super.removeInput(slot);
  }

  override addInput<TProperties extends Partial<INodeInputSlot>>(
    name: string,
    type: ISlotType,
    extra_info?: TProperties | undefined,
  ): INodeInputSlot & TProperties {
    (this as any)._tempWidth = this.size[0];
    return super.addInput(name, type, extra_info);
  }

  override addWidget<Type extends TWidgetType, TValue extends WidgetTypeMap[Type]["value"]>(
    type: Type,
    name: string,
    value: TValue,
    callback: IBaseWidget["callback"] | string | null,
    options?: IWidgetOptions | string,
  ):
    | IBaseWidget<string | number | boolean | object | undefined, string, IWidgetOptions<unknown>>
    | WidgetTypeMap[Type] {
    (this as any)._tempWidth = this.size[0];
    return super.addWidget(type, name, value, callback, options);
  }

  override removeWidget(widget: IBaseWidget | IWidget | number | undefined): void {
    (this as any)._tempWidth = this.size[0];
    super.removeWidget(widget);
  }

  override computeSize(out: Vector2) {
    let size = super.computeSize(out);
    if ((this as any)._tempWidth) {
      size[0] = (this as any)._tempWidth;
      // We sometimes get repeated calls to compute size, so debounce before clearing.
      this.debouncerTempWidth && clearTimeout(this.debouncerTempWidth);
      this.debouncerTempWidth = setTimeout(() => {
        (this as any)._tempWidth = null;
      }, 32);
    }
    // If we're collapsed, then subtract the total calculated height of the other input slots.
    if (this.properties["collapse_connections"]) {
      const rows = Math.max(this.inputs?.length || 0, this.outputs?.length || 0, 1) - 1;
      size[1] = size[1] - rows * LiteGraph.NODE_SLOT_HEIGHT;
    }
    setTimeout(() => {
      this.graph?.setDirtyCanvas(true, true);
    }, 16);
    return size;
  }

  /**
   * When we connect our output, check our inputs and make sure we're not trying to connect a loop.
   */
  override onConnectOutput(
    outputIndex: number,
    inputType: string | -1,
    inputSlot: INodeInputSlot,
    inputNode: TLGraphNode,
    inputIndex: number,
  ): boolean {
    let canConnect = true;
    if (super.onConnectOutput) {
      canConnect = super.onConnectOutput(outputIndex, inputType, inputSlot, inputNode, inputIndex);
    }
    if (canConnect) {
      const nodes = getConnectedInputNodes(this); // We want passthrough nodes, since they will loop.
      if (nodes.includes(inputNode)) {
        alert(
          `Whoa, whoa, whoa. You've just tried to create a connection that loops back on itself, ` +
            `a situation that could create a time paradox, the results of which could cause a ` +
            `chain reaction that would unravel the very fabric of the space time continuum, ` +
            `and destroy the entire universe!`,
        );
        canConnect = false;
      }
    }
    return canConnect;
  }

  override onConnectInput(
    inputIndex: number,
    outputType: string | -1,
    outputSlot: INodeOutputSlot,
    outputNode: TLGraphNode,
    outputIndex: number,
  ): boolean {
    let canConnect = true;
    if (super.onConnectInput) {
      canConnect = super.onConnectInput(
        inputIndex,
        outputType,
        outputSlot,
        outputNode,
        outputIndex,
      );
    }
    if (canConnect) {
      const nodes = getConnectedOutputNodes(this); // We want passthrough nodes, since they will loop.
      if (nodes.includes(outputNode)) {
        alert(
          `Whoa, whoa, whoa. You've just tried to create a connection that loops back on itself, ` +
            `a situation that could create a time paradox, the results of which could cause a ` +
            `chain reaction that would unravel the very fabric of the space time continuum, ` +
            `and destroy the entire universe!`,
        );
        canConnect = false;
      }
    }
    return canConnect;
  }

  /**
   * If something is dropped on us, just add it to the bottom. onConnectInput should already cancel
   * if it's disallowed.
   */
  override connectByTypeOutput(
    slot: number | string,
    sourceNode: TLGraphNode,
    sourceSlotType: ISlotType,
    optsIn?: ConnectByTypeOptions,
  ): LLink | null {
    const lastInput = this.inputs[this.inputs.length - 1];
    if (!lastInput?.link && lastInput?.type === "*") {
      var sourceSlot = sourceNode.findOutputSlotByType(sourceSlotType, false, true);
      return sourceNode.connect(sourceSlot, this, slot);
    }
    return super.connectByTypeOutput(slot, sourceNode, sourceSlotType, optsIn);
  }

  static override setUp() {
    super.setUp();
    addConnectionLayoutSupport(this, app, [
      ["Left", "Right"],
      ["Right", "Left"],
    ]);
    addMenuItem(this, app, {
      name: (node) =>
        `${node.properties?.["collapse_connections"] ? "Show" : "Collapse"} Connections`,
      property: "collapse_connections",
      prepareValue: (_value, node) => !node.properties?.["collapse_connections"],
      callback: (_node) => {
        app.canvas.getCurrentGraph()?.setDirtyCanvas(true, true);
      },
    });
  }
}

// Ok, hack time! LGraphNode's connectByType is powerful, but for our nodes, that have multiple "*"
// input types, it seems it just takes the first one, and disconnects it. I'd rather we don't do
// that and instead take the next free one. If that doesn't work, then we'll give it to the old
// method.
const oldLGraphNodeConnectByType = LGraphNode.prototype.connectByType;
LGraphNode.prototype.connectByType = function connectByType(
  slot: number | string,
  targetNode: TLGraphNode,
  targetSlotType: ISlotType,
  optsIn?: ConnectByTypeOptions,
): LLink | null {
  // If we're dropping on a node, and the last input is free and an "*" type, then connect there
  // first...
  if (targetNode.inputs) {
    for (const [index, input] of targetNode.inputs.entries()) {
      if (!input.link && input.type === "*") {
        this.connect(slot, targetNode, index);
        return null;
      }
    }
  }
  return (
    (oldLGraphNodeConnectByType &&
      oldLGraphNodeConnectByType.call(this, slot, targetNode, targetSlotType, optsIn)) ||
    null
  );
};
