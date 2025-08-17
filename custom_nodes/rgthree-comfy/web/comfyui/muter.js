import { app } from "../../scripts/app.js";
import { BaseNodeModeChanger } from "./base_node_mode_changer.js";
import { NodeTypesString } from "./constants.js";
const MODE_MUTE = 2;
const MODE_ALWAYS = 0;
class MuterNode extends BaseNodeModeChanger {
    constructor(title = MuterNode.title) {
        super(title);
        this.comfyClass = NodeTypesString.FAST_MUTER;
        this.modeOn = MODE_ALWAYS;
        this.modeOff = MODE_MUTE;
        this.onConstructed();
    }
    async handleAction(action) {
        if (action === "Mute all") {
            for (const widget of this.widgets) {
                this.forceWidgetOff(widget, true);
            }
        }
        else if (action === "Enable all") {
            for (const widget of this.widgets) {
                this.forceWidgetOn(widget, true);
            }
        }
        else if (action === "Toggle all") {
            for (const widget of this.widgets) {
                this.forceWidgetToggle(widget, true);
            }
        }
    }
}
MuterNode.exposedActions = ["Mute all", "Enable all", "Toggle all"];
MuterNode.type = NodeTypesString.FAST_MUTER;
MuterNode.title = NodeTypesString.FAST_MUTER;
app.registerExtension({
    name: "rgthree.Muter",
    registerCustomNodes() {
        MuterNode.setUp();
    },
    loadedGraphNode(node) {
        if (node.type == MuterNode.title) {
            node._tempWidth = node.size[0];
        }
    },
});
