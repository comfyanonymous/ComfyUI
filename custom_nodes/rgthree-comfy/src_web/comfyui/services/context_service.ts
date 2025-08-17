import type {DynamicContextNodeBase} from "../dynamic_context_base.js";

import {NodeTypesString} from "../constants.js";
import {getConnectedOutputNodesAndFilterPassThroughs} from "../utils.js";
import {INodeInputSlot, INodeOutputSlot, INodeSlot, LGraphNode} from "@comfyorg/frontend";

export let SERVICE: ContextService;

const OWNED_PREFIX = "+";
const REGEX_PREFIX = /^[\+⚠️]\s*/;
const REGEX_EMPTY_INPUT = /^\+\s*$/;

export function stripContextInputPrefixes(name: string) {
  return name.replace(REGEX_PREFIX, "");
}

export function getContextOutputName(inputName: string) {
  if (inputName === "base_ctx") return "CONTEXT";
  return stripContextInputPrefixes(inputName).toUpperCase();
}

export enum InputMutationOperation {
  "UNKNOWN",
  "ADDED",
  "REMOVED",
  "RENAMED",
}

export type InputMutation = {
  operation: InputMutationOperation;
  node: DynamicContextNodeBase;
  slotIndex: number;
  slot: INodeSlot;
};

export class ContextService {
  constructor() {
    if (SERVICE) {
      throw new Error("ContextService was already instantiated.");
    }
  }

  onInputChanges(node: any, mutation: InputMutation) {
    const childCtxs = getConnectedOutputNodesAndFilterPassThroughs(
      node,
      node,
      0,
    ) as DynamicContextNodeBase[];
    for (const childCtx of childCtxs) {
      childCtx.handleUpstreamMutation(mutation);
    }
  }

  getDynamicContextInputsData(node: DynamicContextNodeBase) {
    return node
      .getContextInputsList()
      .map((input: INodeInputSlot, index: number) => ({
        name: stripContextInputPrefixes(input.name),
        type: String(input.type),
        index,
      }))
      .filter((i) => i.type !== "*");
  }

  getDynamicContextOutputsData(node: LGraphNode) {
    return node.outputs.map((output: INodeOutputSlot, index: number) => ({
      name: stripContextInputPrefixes(output.name),
      type: String(output.type),
      index,
    }));
  }
}

SERVICE = new ContextService();
