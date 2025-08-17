import type {
  IContextMenuValue,
  IFoundSlot,
  INodeInputSlot,
  INodeOutputSlot,
  ISlotType,
  LGraphNode,
  LLink,
} from "@comfyorg/frontend";
import type {ComfyNodeDef} from "typings/comfy.js";

import {app} from "scripts/app.js";
import {
  IoDirection,
  followConnectionUntilType,
  getConnectedInputInfosAndFilterPassThroughs,
} from "./utils.js";
import {rgthree} from "./rgthree.js";
import {
  SERVICE as CONTEXT_SERVICE,
  InputMutation,
  InputMutationOperation,
} from "./services/context_service.js";
import {NodeTypesString} from "./constants.js";
import {removeUnusedInputsFromEnd} from "./utils_inputs_outputs.js";
import {DynamicContextNodeBase} from "./dynamic_context_base.js";
import {SERVICE as CONFIG_SERVICE} from "./services/config_service.js";

const OWNED_PREFIX = "+";
const REGEX_OWNED_PREFIX = /^\+\s*/;
const REGEX_EMPTY_INPUT = /^\+\s*$/;

/**
 * The Dynamic Context node.
 */
export class DynamicContextNode extends DynamicContextNodeBase {
  static override title = NodeTypesString.DYNAMIC_CONTEXT;
  static override type = NodeTypesString.DYNAMIC_CONTEXT;
  static comfyClass = NodeTypesString.DYNAMIC_CONTEXT;

  constructor(title = DynamicContextNode.title) {
    super(title);
  }

  override onNodeCreated() {
    this.addInput("base_ctx", "RGTHREE_DYNAMIC_CONTEXT");
    this.ensureOneRemainingNewInputSlot();
    super.onNodeCreated();
  }

  override onConnectionsChange(
    type: ISlotType,
    slotIndex: number,
    isConnected: boolean,
    link: LLink | null | undefined,
    ioSlot: INodeInputSlot | INodeOutputSlot,
  ): void {
    super.onConnectionsChange?.call(this, type, slotIndex, isConnected, link, ioSlot);
    if (this.configuring) {
      return;
    }
    if (type === LiteGraph.INPUT) {
      if (isConnected) {
        this.handleInputConnected(slotIndex);
      } else {
        this.handleInputDisconnected(slotIndex);
      }
    }
  }

  override onConnectInput(
    inputIndex: number,
    outputType: INodeOutputSlot["type"],
    outputSlot: INodeOutputSlot,
    outputNode: LGraphNode,
    outputIndex: number,
  ): boolean {
    let canConnect = true;
    if (super.onConnectInput) {
      canConnect = super.onConnectInput.apply(this, [...arguments] as any);
    }
    if (
      canConnect &&
      outputNode instanceof DynamicContextNode &&
      outputIndex === 0 &&
      inputIndex !== 0
    ) {
      const [n, v] = rgthree.logger.warnParts(
        "Currently, you can only connect a context node in the first slot.",
      );
      console[n]?.call(console, ...v);
      canConnect = false;
    }
    return canConnect;
  }

  handleInputConnected(slotIndex: number) {
    const ioSlot = this.inputs[slotIndex];
    const connectedIndexes = [];
    if (slotIndex === 0) {
      let baseNodeInfos = getConnectedInputInfosAndFilterPassThroughs(this, this, 0);
      const baseNodes = baseNodeInfos.map((n) => n.node)!;
      const baseNodesDynamicCtx = baseNodes[0] as DynamicContextNodeBase;
      if (baseNodesDynamicCtx?.provideInputsData) {
        const inputsData = CONTEXT_SERVICE.getDynamicContextInputsData(baseNodesDynamicCtx);
        console.log("inputsData", inputsData);
        for (const input of baseNodesDynamicCtx.provideInputsData()) {
          if (input.name === "base_ctx" || input.name === "+") {
            continue;
          }
          this.addContextInput(input.name, input.type, input.index);
          this.stabilizeNames();
        }
      }
    } else if (this.isInputSlotForNewInput(slotIndex)) {
      this.handleNewInputConnected(slotIndex);
    }
  }

  isInputSlotForNewInput(slotIndex: number) {
    const ioSlot = this.inputs[slotIndex];
    return ioSlot && ioSlot.name === "+" && ioSlot.type === "*";
  }

  handleNewInputConnected(slotIndex: number) {
    if (!this.isInputSlotForNewInput(slotIndex)) {
      throw new Error('Expected the incoming slot index to be the "new input" input.');
    }
    const ioSlot = this.inputs[slotIndex]!;
    let cxn = null;
    if (ioSlot.link != null) {
      cxn = followConnectionUntilType(this, IoDirection.INPUT, slotIndex, true);
    }
    if (cxn?.type && cxn?.name) {
      let name = this.addOwnedPrefix(this.getNextUniqueNameForThisNode(cxn.name));
      if (name.match(/^\+\s*[A-Z_]+(\.\d+)?$/)) {
        name = name.toLowerCase();
      }
      ioSlot.name = name;
      ioSlot.type = cxn.type as string;
      ioSlot.removable = true;
      while (!this.outputs[slotIndex]) {
        this.addOutput("*", "*");
      }
      this.outputs[slotIndex]!.type = cxn.type as string;
      this.outputs[slotIndex]!.name = this.stripOwnedPrefix(name).toLocaleUpperCase();
      // This is a dumb override for ComfyUI's widgetinputs issues.
      if (cxn.type === "COMBO" || cxn.type.includes(",") || Array.isArray(cxn.type)) {
        (this.outputs[slotIndex] as any).widget = true;
      }
      this.inputsMutated({
        operation: InputMutationOperation.ADDED,
        node: this,
        slotIndex,
        slot: ioSlot,
      });
      this.stabilizeNames();
      this.ensureOneRemainingNewInputSlot();
    }
  }

  handleInputDisconnected(slotIndex: number) {
    const inputs = this.getContextInputsList();
    if (slotIndex === 0) {
      for (let index = inputs.length - 1; index > 0; index--) {
        if (index === 0 || index === inputs.length - 1) {
          continue;
        }
        const input = inputs[index]!;
        if (!this.isOwnedInput(input.name)) {
          if (input.link || this.outputs[index]?.links?.length) {
            this.renameContextInput(index, input.name, true);
          } else {
            this.removeContextInput(index);
          }
        }
      }
      this.setSize(this.computeSize());
      this.setDirtyCanvas(true, true);
    }
  }

  ensureOneRemainingNewInputSlot() {
    removeUnusedInputsFromEnd(this, 1, REGEX_EMPTY_INPUT);
    this.addInput(OWNED_PREFIX, "*");
  }

  getNextUniqueNameForThisNode(desiredName: string) {
    const inputs = this.getContextInputsList();
    const allExistingKeys = inputs.map((i) => this.stripOwnedPrefix(i.name).toLocaleUpperCase());
    desiredName = this.stripOwnedPrefix(desiredName);
    let newName = desiredName;
    let n = 0;
    while (allExistingKeys.includes(newName.toLocaleUpperCase())) {
      newName = `${desiredName}.${++n}`;
    }
    return newName;
  }

  override removeInput(slotIndex: number) {
    const slot = this.inputs[slotIndex]!;
    super.removeInput(slotIndex);
    if (this.outputs[slotIndex]) {
      this.removeOutput(slotIndex);
    }
    this.inputsMutated({operation: InputMutationOperation.REMOVED, node: this, slotIndex, slot});
    this.stabilizeNames();
  }

  stabilizeNames() {
    const inputs = this.getContextInputsList();
    const names: string[] = [];
    for (const [index, input] of inputs.entries()) {
      if (index === 0 || index === inputs.length - 1) {
        continue;
      }
      input.label = undefined;
      this.outputs[index]!.label = undefined;
      let origName = this.stripOwnedPrefix(input.name).replace(/\.\d+$/, "");
      let name = input.name;
      if (!this.isOwnedInput(name)) {
        names.push(name.toLocaleUpperCase());
      } else {
        let n = 0;
        name = this.addOwnedPrefix(origName);
        while (names.includes(this.stripOwnedPrefix(name).toLocaleUpperCase())) {
          name = `${this.addOwnedPrefix(origName)}.${++n}`;
        }
        names.push(this.stripOwnedPrefix(name).toLocaleUpperCase());
        if (input.name !== name) {
          this.renameContextInput(index, name);
        }
      }
    }
  }

  override getSlotMenuOptions(slot: IFoundSlot): IContextMenuValue[] {
    const editable = this.isOwnedInput(slot.input!.name) && this.type !== "*";
    return [
      {
        content: "✏️ Rename Input",
        disabled: !editable,
        callback: () => {
          var dialog = app.canvas.createDialog(
            "<span class='name'>Name</span><input autofocus type='text'/><button>OK</button>",
            {},
          );
          var dialogInput = dialog.querySelector("input")!;
          if (dialogInput) {
            dialogInput.value = this.stripOwnedPrefix(slot.input!.name || "");
          }
          var inner = () => {
            this.handleContextMenuRenameInputDialog(slot.slot, dialogInput.value);
            dialog.close();
          };
          dialog.querySelector("button")!.addEventListener("click", inner);
          dialogInput.addEventListener("keydown", (e) => {
            dialog.is_modified = true;
            if (e.keyCode == 27) {
              dialog.close();
            } else if (e.keyCode == 13) {
              inner();
            } else if (e.keyCode != 13 && (e.target as HTMLElement)?.localName != "textarea") {
              return;
            }
            e.preventDefault();
            e.stopPropagation();
          });
          dialogInput.focus();
        },
      },
      {
        content: "🗑️ Delete Input",
        disabled: !editable,
        callback: () => {
          this.removeInput(slot.slot);
        },
      },
    ];
  }

  handleContextMenuRenameInputDialog(slotIndex: number, value: string) {
    app.graph.beforeChange();
    this.renameContextInput(slotIndex, value);
    this.stabilizeNames();
    this.setDirtyCanvas(true, true);
    app.graph.afterChange();
  }
}

const contextDynamicNodes = [DynamicContextNode];
app.registerExtension({
  name: "rgthree.DynamicContext",
  async beforeRegisterNodeDef(nodeType: typeof LGraphNode, nodeData: ComfyNodeDef) {
    if (!CONFIG_SERVICE.getConfigValue("unreleased.dynamic_context.enabled")) {
      return;
    }
    if (nodeData.name === DynamicContextNode.type) {
      DynamicContextNode.setUp(nodeType, nodeData);
    }
  },
});
