import type {INodeOutputSlot, LGraphNode} from "@comfyorg/frontend";

import {rgthree} from "./rgthree.js";
import {BaseAnyInputConnectedNode} from "./base_any_input_connected_node.js";
import {
  PassThroughFollowing,
  getConnectedInputNodes,
  getConnectedInputNodesAndFilterPassThroughs,
  shouldPassThrough,
} from "./utils.js";

/**
 * Base collector node that monitors changing inputs and outputs.
 */
export class BaseCollectorNode extends BaseAnyInputConnectedNode {
  /**
   * We only want to show nodes through re_route nodes, other pass through nodes show each input.
   */
  override readonly inputsPassThroughFollowing: PassThroughFollowing =
    PassThroughFollowing.REROUTE_ONLY;

  readonly logger = rgthree.newLogSession("[BaseCollectorNode]");

  constructor(title?: string) {
    super(title);
  }

  override clone() {
    const cloned = super.clone()!;
    return cloned;
  }

  override handleLinkedNodesStabilization(linkedNodes: LGraphNode[]) {
    return false; // No-op, no widgets.
  }

  /**
   * When we connect an input, check to see if it's already connected and cancel it.
   */
  override onConnectInput(
    inputIndex: number,
    outputType: string | -1,
    outputSlot: INodeOutputSlot,
    outputNode: LGraphNode,
    outputIndex: number,
  ): boolean {
    let canConnect = super.onConnectInput(
      inputIndex,
      outputType,
      outputSlot,
      outputNode,
      outputIndex,
    );
    if (canConnect) {
      const allConnectedNodes = getConnectedInputNodes(this); // We want passthrough nodes, since they will loop.
      const nodesAlreadyInSlot = getConnectedInputNodes(this, undefined, inputIndex);
      if (allConnectedNodes.includes(outputNode)) {
        // If we're connecting to the same slot, then allow it by replacing the one we have.
        // const slotsOriginNode = getOriginNodeByLink(this.inputs[inputIndex]?.link);
        const [n, v] = this.logger.debugParts(
          `${outputNode.title} is already connected to ${this.title}.`,
        );
        console[n]?.(...v);
        if (nodesAlreadyInSlot.includes(outputNode)) {
          const [n, v] = this.logger.debugParts(
            `... but letting it slide since it's for the same slot.`,
          );
          console[n]?.(...v);
        } else {
          canConnect = false;
        }
      }
      if (canConnect && shouldPassThrough(outputNode, PassThroughFollowing.REROUTE_ONLY)) {
        const connectedNode = getConnectedInputNodesAndFilterPassThroughs(
          outputNode,
          undefined,
          undefined,
          PassThroughFollowing.REROUTE_ONLY,
        )[0];
        if (connectedNode && allConnectedNodes.includes(connectedNode)) {
          // If we're connecting to the same slot, then allow it by replacing the one we have.
          const [n, v] = this.logger.debugParts(
            `${connectedNode.title} is already connected to ${this.title}.`,
          );
          console[n]?.(...v);
          if (nodesAlreadyInSlot.includes(connectedNode)) {
            const [n, v] = this.logger.debugParts(
              `... but letting it slide since it's for the same slot.`,
            );
            console[n]?.(...v);
          } else {
            canConnect = false;
          }
        }
      }
    }
    return canConnect;
  }
}
