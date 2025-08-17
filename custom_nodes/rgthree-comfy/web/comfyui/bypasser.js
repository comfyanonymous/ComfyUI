import { app } from "../../scripts/app.js";
import { BaseNodeModeChanger } from "./base_node_mode_changer.js";
import { NodeTypesString } from "./constants.js";
const MODE_BYPASS = 4;
const MODE_ALWAYS = 0;
class BypasserNode extends BaseNodeModeChanger {
    constructor(title = BypasserNode.title) {
        super(title);
        this.comfyClass = NodeTypesString.FAST_BYPASSER;
        this.modeOn = MODE_ALWAYS;
        this.modeOff = MODE_BYPASS;
        this.onConstructed();
    }
    async handleAction(action) {
        if (action === "Bypass all") {
            for (const widget of this.widgets || []) {
                this.forceWidgetOff(widget, true);
            }
        }
        else if (action === "Enable all") {
            for (const widget of this.widgets || []) {
                this.forceWidgetOn(widget, true);
            }
        }
        else if (action === "Toggle all") {
            for (const widget of this.widgets || []) {
                this.forceWidgetToggle(widget, true);
            }
        }
    }
}
BypasserNode.exposedActions = ["Bypass all", "Enable all", "Toggle all"];
BypasserNode.type = NodeTypesString.FAST_BYPASSER;
BypasserNode.title = NodeTypesString.FAST_BYPASSER;
app.registerExtension({
    name: "rgthree.Bypasser",
    registerCustomNodes() {
        BypasserNode.setUp();
    },
    loadedGraphNode(node) {
        if (node.type == BypasserNode.title) {
            node._tempWidth = node.size[0];
        }
    },
});
