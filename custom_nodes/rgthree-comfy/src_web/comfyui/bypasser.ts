import type {LGraphNode} from "@comfyorg/frontend";

import {app} from "scripts/app.js";
import {BaseNodeModeChanger} from "./base_node_mode_changer.js";
import {NodeTypesString} from "./constants.js";

const MODE_BYPASS = 4;
const MODE_ALWAYS = 0;

class BypasserNode extends BaseNodeModeChanger {
  static override exposedActions = ["Bypass all", "Enable all", "Toggle all"];

  static override type = NodeTypesString.FAST_BYPASSER;
  static override title = NodeTypesString.FAST_BYPASSER;
  override comfyClass = NodeTypesString.FAST_BYPASSER;

  override readonly modeOn = MODE_ALWAYS;
  override readonly modeOff = MODE_BYPASS;

  constructor(title = BypasserNode.title) {
    super(title);
    this.onConstructed();
  }

  override async handleAction(action: string) {
    if (action === "Bypass all") {
      for (const widget of this.widgets || []) {
        this.forceWidgetOff(widget, true);
      }
    } else if (action === "Enable all") {
      for (const widget of this.widgets || []) {
        this.forceWidgetOn(widget, true);
      }
    } else if (action === "Toggle all") {
      for (const widget of this.widgets || []) {
        this.forceWidgetToggle(widget, true);
      }
    }
  }
}

app.registerExtension({
  name: "rgthree.Bypasser",
  registerCustomNodes() {
    BypasserNode.setUp();
  },
  loadedGraphNode(node: LGraphNode) {
    if (node.type == BypasserNode.title) {
      (node as any)._tempWidth = node.size[0];
    }
  },
});
