import { app } from "../../scripts/app.js";
import { NodeTypesString } from "./constants.js";
import { BaseFastGroupsModeChanger } from "./fast_groups_muter.js";
export class FastGroupsBypasser extends BaseFastGroupsModeChanger {
    constructor(title = FastGroupsBypasser.title) {
        super(title);
        this.comfyClass = NodeTypesString.FAST_GROUPS_BYPASSER;
        this.helpActions = "bypass and enable";
        this.modeOn = LiteGraph.ALWAYS;
        this.modeOff = 4;
        this.onConstructed();
    }
}
FastGroupsBypasser.type = NodeTypesString.FAST_GROUPS_BYPASSER;
FastGroupsBypasser.title = NodeTypesString.FAST_GROUPS_BYPASSER;
FastGroupsBypasser.exposedActions = ["Bypass all", "Enable all", "Toggle all"];
app.registerExtension({
    name: "rgthree.FastGroupsBypasser",
    registerCustomNodes() {
        FastGroupsBypasser.setUp();
    },
    loadedGraphNode(node) {
        if (node.type == FastGroupsBypasser.title) {
            node.tempSize = [...node.size];
        }
    },
});
