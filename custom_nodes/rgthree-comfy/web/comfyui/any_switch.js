import { app } from "../../scripts/app.js";
import { IoDirection, addConnectionLayoutSupport, followConnectionUntilType } from "./utils.js";
import { RgthreeBaseServerNode } from "./base_node.js";
import { NodeTypesString } from "./constants.js";
import { removeUnusedInputsFromEnd } from "./utils_inputs_outputs.js";
import { debounce } from "../../rgthree/common/shared_utils.js";
class RgthreeAnySwitch extends RgthreeBaseServerNode {
    constructor(title = RgthreeAnySwitch.title) {
        super(title);
        this.stabilizeBound = this.stabilize.bind(this);
        this.nodeType = null;
        this.addAnyInput(5);
    }
    onConnectionsChange(type, slotIndex, isConnected, linkInfo, ioSlot) {
        var _a;
        (_a = super.onConnectionsChange) === null || _a === void 0 ? void 0 : _a.call(this, type, slotIndex, isConnected, linkInfo, ioSlot);
        this.scheduleStabilize();
    }
    onConnectionsChainChange() {
        this.scheduleStabilize();
    }
    scheduleStabilize(ms = 64) {
        return debounce(this.stabilizeBound, ms);
    }
    addAnyInput(num = 1) {
        for (let i = 0; i < num; i++) {
            this.addInput(`any_${String(this.inputs.length + 1).padStart(2, "0")}`, (this.nodeType || "*"));
        }
    }
    stabilize() {
        removeUnusedInputsFromEnd(this, 4);
        this.addAnyInput();
        let connectedType = followConnectionUntilType(this, IoDirection.INPUT, undefined, true);
        if (!connectedType) {
            connectedType = followConnectionUntilType(this, IoDirection.OUTPUT, undefined, true);
        }
        this.nodeType = (connectedType === null || connectedType === void 0 ? void 0 : connectedType.type) || "*";
        for (const input of this.inputs) {
            input.type = this.nodeType;
        }
        for (const output of this.outputs) {
            output.type = this.nodeType;
            output.label =
                output.type === "RGTHREE_CONTEXT"
                    ? "CONTEXT"
                    : Array.isArray(this.nodeType) || this.nodeType.includes(",")
                        ? (connectedType === null || connectedType === void 0 ? void 0 : connectedType.label) || String(this.nodeType)
                        : String(this.nodeType);
        }
    }
    static setUp(comfyClass, nodeData) {
        RgthreeBaseServerNode.registerForOverride(comfyClass, nodeData, RgthreeAnySwitch);
        addConnectionLayoutSupport(RgthreeAnySwitch, app, [
            ["Left", "Right"],
            ["Right", "Left"],
        ]);
    }
}
RgthreeAnySwitch.title = NodeTypesString.ANY_SWITCH;
RgthreeAnySwitch.type = NodeTypesString.ANY_SWITCH;
RgthreeAnySwitch.comfyClass = NodeTypesString.ANY_SWITCH;
app.registerExtension({
    name: "rgthree.AnySwitch",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "Any Switch (rgthree)") {
            RgthreeAnySwitch.setUp(nodeType, nodeData);
        }
    },
});
