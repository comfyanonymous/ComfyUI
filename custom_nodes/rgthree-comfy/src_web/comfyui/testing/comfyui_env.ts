import type {LGraphNode} from "@comfyorg/frontend";

import {app} from "scripts/app.js";
import {wait} from "rgthree/common/shared_utils.js";
import {NodeTypesString} from "../constants.js";

type addNodeOptions = {
  placement?: string;
};

/**
 * A testing environment to make setting up, clearing, and queuing more predictable in an
 * integration test environment.
 */
export class ComfyUITestEnvironment {
  private lastNode: LGraphNode | null = null;
  private maxY = 0;

  constructor() {}

  wait = wait;

  async addNode(nodeString: string, options: addNodeOptions = {}) {
    const [canvas, graph] = [app.canvas, app.graph];
    const node = LiteGraph.createNode(nodeString)!;
    let x = 0;
    let y = 30;
    if (this.lastNode) {
      const placement = options.placement || "right";
      if (placement === "under") {
        x = this.lastNode.pos[0];
        y = this.lastNode.pos[1] + this.lastNode.size[1] + 30;
      } else if (placement === "right") {
        x = this.lastNode.pos[0] + this.lastNode.size[0] + 100;
        y = this.lastNode.pos[1];
      } else if (placement === "start") {
        x = 0;
        y = this.maxY + 50;
      }
    }
    canvas.graph!.add(node);
    node.pos = [x, y];
    canvas.selectNode(node);
    app.graph.setDirtyCanvas(true, true);
    await wait();
    this.lastNode = node;
    this.maxY = Math.max(this.maxY, y + this.lastNode.size[1]);
    return (this.lastNode = node);
  }

  async clear() {
    app.clean();
    app.graph.clear();
    const nodeConfig = await this.addNode(NodeTypesString.KSAMPLER_CONFIG);
    const displayAny = await this.addNode(NodeTypesString.DISPLAY_ANY);
    nodeConfig.widgets = nodeConfig.widgets || [];
    nodeConfig.widgets[0]!.value = Math.round(Math.random() * 100);
    nodeConfig.connect(0, displayAny, 0);
    await this.queuePrompt();
    app.clean();
    app.graph.clear();
    this.lastNode = null;
    this.maxY = 0;
    await wait();
  }

  async queuePrompt() {
    await app.queuePrompt(0);
    await wait(150);
  }
}
