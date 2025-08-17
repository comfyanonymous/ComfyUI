import { app } from "../../scripts/app.js";
import { DynamicContextNodeBase } from "./dynamic_context_base.js";
import { NodeTypesString } from "./constants.js";
import { SERVICE as CONTEXT_SERVICE, getContextOutputName, } from "./services/context_service.js";
import { getConnectedInputNodesAndFilterPassThroughs } from "./utils.js";
import { debounce, moveArrayItem } from "../../rgthree/common/shared_utils.js";
import { measureText } from "./utils_canvas.js";
import { SERVICE as CONFIG_SERVICE } from "./services/config_service.js";
class DynamicContextSwitchNode extends DynamicContextNodeBase {
    constructor(title = DynamicContextSwitchNode.title) {
        super(title);
        this.hasShadowInputs = true;
        this.lastInputsList = [];
        this.shadowInputs = [
            { name: "base_ctx", type: "RGTHREE_DYNAMIC_CONTEXT", link: null, count: 0, boundingRect: null },
        ];
    }
    getContextInputsList() {
        return this.shadowInputs;
    }
    handleUpstreamMutation(mutation) {
        this.scheduleHardRefresh();
    }
    onConnectionsChange(type, slotIndex, isConnected, link, inputOrOutput) {
        var _a;
        (_a = super.onConnectionsChange) === null || _a === void 0 ? void 0 : _a.call(this, type, slotIndex, isConnected, link, inputOrOutput);
        if (this.configuring) {
            return;
        }
        if (type === LiteGraph.INPUT) {
            this.scheduleHardRefresh();
        }
    }
    scheduleHardRefresh(ms = 64) {
        return debounce(() => {
            this.refreshInputsAndOutputs();
        }, ms);
    }
    onNodeCreated() {
        this.addInput("ctx_1", "RGTHREE_DYNAMIC_CONTEXT");
        this.addInput("ctx_2", "RGTHREE_DYNAMIC_CONTEXT");
        this.addInput("ctx_3", "RGTHREE_DYNAMIC_CONTEXT");
        this.addInput("ctx_4", "RGTHREE_DYNAMIC_CONTEXT");
        this.addInput("ctx_5", "RGTHREE_DYNAMIC_CONTEXT");
        super.onNodeCreated();
    }
    addContextInput(name, type, slot) { }
    refreshInputsAndOutputs() {
        var _a;
        const inputs = [
            { name: "base_ctx", type: "RGTHREE_DYNAMIC_CONTEXT", link: null, count: 0, boundingRect: null },
        ];
        let numConnected = 0;
        for (let i = 0; i < this.inputs.length; i++) {
            const childCtxs = getConnectedInputNodesAndFilterPassThroughs(this, this, i);
            if (childCtxs.length > 1) {
                throw new Error("How is there more than one input?");
            }
            const ctx = childCtxs[0];
            if (!ctx)
                continue;
            numConnected++;
            const slotsData = CONTEXT_SERVICE.getDynamicContextInputsData(ctx);
            console.log(slotsData);
            for (const slotData of slotsData) {
                const found = inputs.find((n) => getContextOutputName(slotData.name) === getContextOutputName(n.name));
                if (found) {
                    found.count += 1;
                    continue;
                }
                inputs.push({
                    name: slotData.name,
                    type: slotData.type,
                    link: null,
                    count: 1,
                    boundingRect: null,
                });
            }
        }
        this.shadowInputs = inputs;
        let i = 0;
        for (i; i < this.shadowInputs.length; i++) {
            const data = this.shadowInputs[i];
            let existing = this.outputs.find((o) => getContextOutputName(o.name) === getContextOutputName(data.name));
            if (!existing) {
                existing = this.addOutput(getContextOutputName(data.name), data.type);
            }
            moveArrayItem(this.outputs, existing, i);
            delete existing.rgthree_status;
            if (data.count !== numConnected) {
                existing.rgthree_status = "WARN";
            }
        }
        while (this.outputs[i]) {
            const output = this.outputs[i];
            if ((_a = output === null || output === void 0 ? void 0 : output.links) === null || _a === void 0 ? void 0 : _a.length) {
                output.rgthree_status = "ERROR";
                i++;
            }
            else {
                this.removeOutput(i);
            }
        }
        this.fixInputsOutputsLinkSlots();
    }
    onDrawForeground(ctx, canvas) {
        var _a, _b;
        const low_quality = ((_b = (_a = canvas === null || canvas === void 0 ? void 0 : canvas.ds) === null || _a === void 0 ? void 0 : _a.scale) !== null && _b !== void 0 ? _b : 1) < 0.6;
        if (low_quality || this.size[0] <= 10) {
            return;
        }
        let y = LiteGraph.NODE_SLOT_HEIGHT - 1;
        const w = this.size[0];
        ctx.save();
        ctx.font = "normal " + LiteGraph.NODE_SUBTEXT_SIZE + "px Arial";
        ctx.textAlign = "right";
        for (const output of this.outputs) {
            if (!output.rgthree_status) {
                y += LiteGraph.NODE_SLOT_HEIGHT;
                continue;
            }
            const x = w - 20 - measureText(ctx, output.name);
            if (output.rgthree_status === "ERROR") {
                ctx.fillText("🛑", x, y);
            }
            else if (output.rgthree_status === "WARN") {
                ctx.fillText("⚠️", x, y);
            }
            y += LiteGraph.NODE_SLOT_HEIGHT;
        }
        ctx.restore();
    }
}
DynamicContextSwitchNode.title = NodeTypesString.DYNAMIC_CONTEXT_SWITCH;
DynamicContextSwitchNode.type = NodeTypesString.DYNAMIC_CONTEXT_SWITCH;
DynamicContextSwitchNode.comfyClass = NodeTypesString.DYNAMIC_CONTEXT_SWITCH;
app.registerExtension({
    name: "rgthree.DynamicContextSwitch",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!CONFIG_SERVICE.getConfigValue("unreleased.dynamic_context.enabled")) {
            return;
        }
        if (nodeData.name === DynamicContextSwitchNode.type) {
            DynamicContextSwitchNode.setUp(nodeType, nodeData);
        }
    },
});
