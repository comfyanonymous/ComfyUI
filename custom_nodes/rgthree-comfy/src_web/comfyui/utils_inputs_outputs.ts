import type {LGraphNode} from "@comfyorg/frontend";
import { RgthreeBaseNode } from "./base_node";

/** Removes all inputs from the end. */
export function removeUnusedInputsFromEnd(node: LGraphNode, minNumber = 1, nameMatch?: RegExp) {
  // No need to remove inputs from nodes that have been removed. This can happen because we may
  // have debounced cleanup tasks.
  if ((node as RgthreeBaseNode).removed) return;
  for (let i = node.inputs.length - 1; i >= minNumber; i--) {
    if (!node.inputs[i]?.link) {
      if (!nameMatch || nameMatch.test(node.inputs[i]!.name)) {
        node.removeInput(i);
      }
      continue;
    }
    break;
  }
}
