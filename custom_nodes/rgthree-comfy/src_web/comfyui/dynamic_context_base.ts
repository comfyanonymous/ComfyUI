import type {INodeInputSlot, LGraphNodeConstructor} from "@comfyorg/frontend";
import type {ComfyNodeDef} from "typings/comfy.js";

import {app} from "scripts/app.js";
import {BaseContextNode} from "./context.js";
import {RgthreeBaseServerNode} from "./base_node.js";
import {moveArrayItem, wait} from "rgthree/common/shared_utils.js";
import {RgthreeInvisibleWidget} from "./utils_widgets.js";
import {
  getContextOutputName,
  InputMutation,
  InputMutationOperation,
} from "./services/context_service.js";
import {SERVICE as CONTEXT_SERVICE} from "./services/context_service.js";

const OWNED_PREFIX = "+";
const REGEX_OWNED_PREFIX = /^\+\s*/;
const REGEX_EMPTY_INPUT = /^\+\s*$/;

export type InputLike = {
  name: string;
  type: number | string;
  label?: string;
  link: number | null;
  removable?: boolean;
  boundingRect: any;
};

/**
 * The base context node that contains some shared between DynamicContext nodes. Not labels
 * `abstract` so we can reference `this` in static methods.
 */
export class DynamicContextNodeBase extends BaseContextNode {
  protected readonly hasShadowInputs: boolean = false;

  getContextInputsList(): InputLike[] {
    return this.inputs;
  }

  provideInputsData() {
    const inputs = this.getContextInputsList();
    return inputs
      .map((input, index) => ({
        name: this.stripOwnedPrefix(input.name),
        type: String(input.type),
        index,
      }))
      .filter((i) => i.type !== "*");
  }

  addOwnedPrefix(name: string) {
    return `+ ${this.stripOwnedPrefix(name)}`;
  }

  isOwnedInput(inputOrName: string | null | INodeInputSlot) {
    const name = typeof inputOrName == "string" ? inputOrName : inputOrName?.name || "";
    return REGEX_OWNED_PREFIX.test(name);
  }

  stripOwnedPrefix(name: string) {
    return name.replace(REGEX_OWNED_PREFIX, "");
  }

  // handleUpstreamMutation(mutation: InputMutation) {
  //   throw new Error('handleUpstreamMutation not overridden!')
  // }

  handleUpstreamMutation(mutation: InputMutation) {
    console.log(`[node ${this.id}] handleUpstreamMutation`, mutation);
    if (mutation.operation === InputMutationOperation.ADDED) {
      const slot = mutation.slot;
      if (!slot) {
        throw new Error("Cannot have an ADDED mutation without a provided slot data.");
      }
      this.addContextInput(
        this.stripOwnedPrefix(slot.name),
        slot.type as string,
        mutation.slotIndex,
      );
      return;
    }
    if (mutation.operation === InputMutationOperation.REMOVED) {
      const slot = mutation.slot;
      if (!slot) {
        throw new Error("Cannot have an REMOVED mutation without a provided slot data.");
      }
      this.removeContextInput(mutation.slotIndex);
      return;
    }
    if (mutation.operation === InputMutationOperation.RENAMED) {
      const slot = mutation.slot;
      if (!slot) {
        throw new Error("Cannot have an RENAMED mutation without a provided slot data.");
      }
      this.renameContextInput(mutation.slotIndex, slot.name);
      return;
    }
  }
  override clone() {
    const cloned = super.clone()! as DynamicContextNodeBase;
    while (cloned.inputs.length > 1) {
      cloned.removeInput(cloned.inputs.length - 1);
    }
    while (cloned.widgets.length > 1) {
      cloned.removeWidget(cloned.widgets.length - 1);
    }
    while (cloned.outputs.length > 1) {
      cloned.removeOutput(cloned.outputs.length - 1);
    }
    return cloned;
  }

  /**
   * Adds the basic output_keys widget. Should be called _after_ specific nodes setup their inputs
   * or widgets.
   */
  override onNodeCreated() {
    const node = this;
    this.addCustomWidget(
      new RgthreeInvisibleWidget("output_keys", "RGTHREE_DYNAMIC_CONTEXT_OUTPUTS", "", () => {
        return (node.outputs || [])
          .map((o, i) => i > 0 && o.name)
          .filter((n) => n !== false)
          .join(",");
      }),
    );
  }

  addContextInput(name: string, type: string, slot = -1) {
    const inputs = this.getContextInputsList();
    if (this.hasShadowInputs) {
      inputs.push({name, type, link: null, boundingRect: null});
    } else {
      this.addInput(name, type);
    }
    if (slot > -1) {
      moveArrayItem(inputs, inputs.length - 1, slot);
    } else {
      slot = inputs.length - 1;
    }
    if (type !== "*") {
      const output = this.addOutput(getContextOutputName(name), type);
      if (type === "COMBO" || String(type).includes(",") || Array.isArray(type)) {
        (output as any).widget = true;
      }
      if (slot > -1) {
        moveArrayItem(this.outputs, this.outputs.length - 1, slot);
      }
    }
    this.fixInputsOutputsLinkSlots();
    this.inputsMutated({
      operation: InputMutationOperation.ADDED,
      node: this,
      slotIndex: slot,
      slot: inputs[slot]!,
    });
  }

  removeContextInput(slotIndex: number) {
    if (this.hasShadowInputs) {
      const inputs = this.getContextInputsList();
      const input = inputs.splice(slotIndex, 1)[0];
      if (this.outputs[slotIndex]) {
        this.removeOutput(slotIndex);
      }
    } else {
      this.removeInput(slotIndex);
    }
  }

  renameContextInput(index: number, newName: string, forceOwnBool: boolean | null = null) {
    const inputs = this.getContextInputsList();
    const input = inputs[index]!;
    const oldName = input.name;
    newName = this.stripOwnedPrefix(newName.trim() || this.getSlotDefaultInputLabel(index));
    if (forceOwnBool === true || (this.isOwnedInput(oldName) && forceOwnBool !== false)) {
      newName = this.addOwnedPrefix(newName);
    }
    if (oldName !== newName) {
      input.name = newName;
      input.removable = this.isOwnedInput(newName);
      this.outputs[index]!.name = getContextOutputName(inputs[index]!.name);
      this.inputsMutated({
        node: this,
        operation: InputMutationOperation.RENAMED,
        slotIndex: index,
        slot: input,
      });
    }
  }

  getSlotDefaultInputLabel(slotIndex: number) {
    const inputs = this.getContextInputsList();
    const input = inputs[slotIndex]!;
    let defaultLabel = this.stripOwnedPrefix(input.name).toLowerCase();
    return defaultLabel.toLocaleLowerCase();
  }

  inputsMutated(mutation: InputMutation) {
    CONTEXT_SERVICE.onInputChanges(this, mutation);
  }

  fixInputsOutputsLinkSlots() {
    if (!this.hasShadowInputs) {
      const inputs = this.getContextInputsList();
      for (let index = inputs.length - 1; index > 0; index--) {
        const input = inputs[index]!;
        if ((input === null || input === void 0 ? void 0 : input.link) != null) {
          app.graph.links[input.link!]!.target_slot = index;
        }
      }
    }
    const outputs = this.outputs;
    for (let index = outputs.length - 1; index > 0; index--) {
      const output = outputs[index];
      if (output) {
        output.nameLocked = true;
        for (const link of output.links || []) {
          app.graph.links[link!]!.origin_slot = index;
        }
      }
    }
  }

  static override setUp(comfyClass: typeof LGraphNode, nodeData: ComfyNodeDef) {
    RgthreeBaseServerNode.registerForOverride(comfyClass, nodeData, this);
    // [🤮] ComfyUI only adds "required" inputs to the outputs list when dragging an output to
    // empty space, but since RGTHREE_CONTEXT is optional, it doesn't get added to the menu because
    // ...of course. So, we'll manually add it. Of course, we also have to do this in a timeout
    // because ComfyUI clears out `LiteGraph.slot_types_default_out` in its own 'Comfy.SlotDefaults'
    // extension and we need to wait for that to happen.
    wait(500).then(() => {
      LiteGraph.slot_types_default_out["RGTHREE_DYNAMIC_CONTEXT"] =
        LiteGraph.slot_types_default_out["RGTHREE_DYNAMIC_CONTEXT"] || [];
      LiteGraph.slot_types_default_out["RGTHREE_DYNAMIC_CONTEXT"].push(
        (comfyClass as LGraphNodeConstructor).comfyClass,
      );
    });
  }
}
