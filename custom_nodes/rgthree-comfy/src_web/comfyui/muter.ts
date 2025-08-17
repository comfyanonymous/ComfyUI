import type {LGraphNode} from "@comfyorg/frontend";

import {app} from "scripts/app.js";
import {BaseNodeModeChanger} from "./base_node_mode_changer.js";
import {NodeTypesString} from "./constants.js";

const MODE_MUTE = 2;
const MODE_ALWAYS = 0;

class MuterNode extends BaseNodeModeChanger {
  static override exposedActions = ["Mute all", "Enable all", "Toggle all"];

  static override type = NodeTypesString.FAST_MUTER;
  static override title = NodeTypesString.FAST_MUTER;
  override comfyClass = NodeTypesString.FAST_MUTER;
  override readonly modeOn = MODE_ALWAYS;
  override readonly modeOff = MODE_MUTE;

  constructor(title = MuterNode.title) {
    super(title);
    this.onConstructed();
  }

  override async handleAction(action: string) {
    if (action === "Mute all") {
      for (const widget of this.widgets) {
        this.forceWidgetOff(widget, true);
      }
    } else if (action === "Enable all") {
      for (const widget of this.widgets) {
        this.forceWidgetOn(widget, true);
      }
    } else if (action === "Toggle all") {
      for (const widget of this.widgets) {
        this.forceWidgetToggle(widget, true);
      }
    }
  }
}

app.registerExtension({
  name: "rgthree.Muter",
  registerCustomNodes() {
    MuterNode.setUp();
  },
  loadedGraphNode(node: LGraphNode) {
    if (node.type == MuterNode.title) {
      (node as any)._tempWidth = node.size[0];
    }
  },
});
