import type {
  IContextMenuValue,
  LGraphCanvas as TLGraphCanvas,
  LGraphNode,
} from "@comfyorg/frontend";
import type {ComfyNodeDef} from "typings/comfy.js";

import {app} from "scripts/app.js";
import {rgthree} from "./rgthree.js";
import {getGroupNodes, getOutputNodes} from "./utils.js";
import {SERVICE as CONFIG_SERVICE} from "./services/config_service.js";

function showQueueNodesMenuIfOutputNodesAreSelected(
  existingOptions: (IContextMenuValue<unknown> | null)[],
) {
  if (CONFIG_SERVICE.getConfigValue("features.menu_queue_selected_nodes") === false) {
    return;
  }
  const outputNodes = getOutputNodes(Object.values(app.canvas.selected_nodes));
  const menuItem = {
    content: `Queue Selected Output Nodes (rgthree) &nbsp;`,
    className: "rgthree-contextmenu-item",
    callback: () => {
      rgthree.queueOutputNodes(outputNodes.map((n) => n.id));
    },
    disabled: !outputNodes.length,
  };

  let idx = existingOptions.findIndex((o) => o?.content === "Outputs") + 1;
  idx = idx || existingOptions.findIndex((o) => o?.content === "Align") + 1;
  idx = idx || 3;
  existingOptions.splice(idx, 0, menuItem);
}

function showQueueGroupNodesMenuIfGroupIsSelected(
  existingOptions: (IContextMenuValue<unknown> | null)[],
) {
  if (CONFIG_SERVICE.getConfigValue("features.menu_queue_selected_nodes") === false) {
    return;
  }
  const group =
    rgthree.lastCanvasMouseEvent &&
    (app.canvas.getCurrentGraph() || app.graph).getGroupOnPos(
      rgthree.lastCanvasMouseEvent.canvasX,
      rgthree.lastCanvasMouseEvent.canvasY,
    );

  const outputNodes: LGraphNode[] | null = (group && getOutputNodes(getGroupNodes(group))) || null;
  const menuItem = {
    content: `Queue Group Output Nodes (rgthree) &nbsp;`,
    className: "rgthree-contextmenu-item",
    callback: () => {
      outputNodes && rgthree.queueOutputNodes(outputNodes.map((n) => n.id));
    },
    disabled: !outputNodes?.length,
  };

  let idx = existingOptions.findIndex((o) => o?.content?.startsWith("Queue Selected ")) + 1;
  idx = idx || existingOptions.findIndex((o) => o?.content === "Outputs") + 1;
  idx = idx || existingOptions.findIndex((o) => o?.content === "Align") + 1;
  idx = idx || 3;
  existingOptions.splice(idx, 0, menuItem);
}

/**
 * Adds a "Queue Node" menu item to all output nodes, working with `rgthree.queueOutputNode` to
 * execute only a single node's path.
 */
app.registerExtension({
  name: "rgthree.QueueNode",
  async beforeRegisterNodeDef(nodeType: typeof LGraphNode, nodeData: ComfyNodeDef) {
    const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
    nodeType.prototype.getExtraMenuOptions = function (
      canvas: TLGraphCanvas,
      options: (IContextMenuValue<unknown> | null)[],
    ): (IContextMenuValue<unknown> | null)[] {
      const extraOptions = getExtraMenuOptions?.call(this, canvas, options) ?? [];
      showQueueNodesMenuIfOutputNodesAreSelected(options);
      showQueueGroupNodesMenuIfGroupIsSelected(options);
      return extraOptions;
    };
  },

  async setup() {
    const getCanvasMenuOptions = LGraphCanvas.prototype.getCanvasMenuOptions;
    LGraphCanvas.prototype.getCanvasMenuOptions = function (...args: any[]) {
      const options = getCanvasMenuOptions.apply(this, [...args] as any);
      showQueueNodesMenuIfOutputNodesAreSelected(options);
      showQueueGroupNodesMenuIfGroupIsSelected(options);
      return options;
    };
  },
});
