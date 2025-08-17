import type {LGraphNode} from "@comfyorg/frontend";

import {app} from "scripts/app.js";
import {BaseAnyInputConnectedNode} from "./base_any_input_connected_node.js";
import {NodeTypesString} from "./constants.js";
import {rgthree} from "./rgthree.js";
import {changeModeOfNodes, getConnectedInputNodesAndFilterPassThroughs} from "./utils.js";

const MODE_MUTE = 2;
const MODE_ALWAYS = 0;

class RandomUnmuterNode extends BaseAnyInputConnectedNode {
  static override exposedActions = ["Mute all", "Enable all"];

  static override type = NodeTypesString.RANDOM_UNMUTER;
  override comfyClass = NodeTypesString.RANDOM_UNMUTER;
  static override title = RandomUnmuterNode.type;
  readonly modeOn = MODE_ALWAYS;
  readonly modeOff = MODE_MUTE;

  tempEnabledNode: LGraphNode | null = null;
  processingQueue: boolean = false;

  onQueueBound = this.onQueue.bind(this);
  onQueueEndBound = this.onQueueEnd.bind(this);
  onGraphtoPromptBound = this.onGraphtoPrompt.bind(this);
  onGraphtoPromptEndBound = this.onGraphtoPromptEnd.bind(this);

  constructor(title = RandomUnmuterNode.title) {
    super(title);

    rgthree.addEventListener("queue", this.onQueueBound);
    rgthree.addEventListener("queue-end", this.onQueueEndBound);
    rgthree.addEventListener("graph-to-prompt", this.onGraphtoPromptBound);
    rgthree.addEventListener("graph-to-prompt-end", this.onGraphtoPromptEndBound);
    this.onConstructed();
  }

  override onRemoved() {
    rgthree.removeEventListener("queue", this.onQueueBound);
    rgthree.removeEventListener("queue-end", this.onQueueEndBound);
    rgthree.removeEventListener("graph-to-prompt", this.onGraphtoPromptBound);
    rgthree.removeEventListener("graph-to-prompt-end", this.onGraphtoPromptEndBound);
  }

  onQueue(event: Event) {
    this.processingQueue = true;
  }
  onQueueEnd(event: Event) {
    this.processingQueue = false;
  }
  onGraphtoPrompt(event: Event) {
    if (!this.processingQueue) {
      return;
    }
    this.tempEnabledNode = null;
    // Check that all are muted and, if so, choose one to unmute.
    const linkedNodes = getConnectedInputNodesAndFilterPassThroughs(this);
    let allMuted = true;
    if (linkedNodes.length) {
      for (const node of linkedNodes) {
        if (node.mode !== this.modeOff) {
          allMuted = false;
          break;
        }
      }
      if (allMuted) {
        this.tempEnabledNode = linkedNodes[Math.floor(Math.random() * linkedNodes.length)] || null;
        if (this.tempEnabledNode) {
          changeModeOfNodes(this.tempEnabledNode, this.modeOn);
        }
      }
    }
  }
  onGraphtoPromptEnd(event: Event) {
    if (this.tempEnabledNode) {
      changeModeOfNodes(this.tempEnabledNode, this.modeOff);
      this.tempEnabledNode = null;
    }
  }

  override handleLinkedNodesStabilization(linkedNodes: LGraphNode[]) {
    return false; // No-op, no widgets.
  }

  override getHelp(): string {
    return `
      <p>
        Use this node to unmute on of its inputs randomly when the graph is queued (and, immediately
        mute it back).
      </p>
      <ul>
        <li><p>
          NOTE: All input nodes MUST be muted to start; if not this node will not randomly unmute
          another. (This is powerful, as the generated image can be dragged in and the chosen input
          will already by unmuted and work w/o any further action.)
        </p></li>
        <li><p>
          TIP: Connect a Repeater's output to this nodes input and place that Repeater on a group
          without any other inputs, and it will mute/unmute the entire group.
        </p></li>
      </ul>
    `;
  }
}

app.registerExtension({
  name: "rgthree.RandomUnmuter",
  registerCustomNodes() {
    RandomUnmuterNode.setUp();
  },
  loadedGraphNode(node: LGraphNode) {
    if (node.type == RandomUnmuterNode.title) {
      (node as any)._tempWidth = node.size[0];
    }
  },
});
