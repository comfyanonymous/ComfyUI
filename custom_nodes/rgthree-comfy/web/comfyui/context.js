import { app } from "../../scripts/app.js";
import { IoDirection, addConnectionLayoutSupport, addMenuItem, matchLocalSlotsToServer, replaceNode, } from "./utils.js";
import { RgthreeBaseServerNode } from "./base_node.js";
import { SERVICE as KEY_EVENT_SERVICE } from "./services/key_events_services.js";
import { debounce, wait } from "../../rgthree/common/shared_utils.js";
import { removeUnusedInputsFromEnd } from "./utils_inputs_outputs.js";
import { NodeTypesString } from "./constants.js";
function findMatchingIndexByTypeOrName(otherNode, otherSlot, ctxSlots) {
    const otherNodeType = (otherNode.type || "").toUpperCase();
    const otherNodeName = (otherNode.title || "").toUpperCase();
    let otherSlotType = otherSlot.type;
    if (Array.isArray(otherSlotType) || otherSlotType.includes(",")) {
        otherSlotType = "COMBO";
    }
    const otherSlotName = otherSlot.name.toUpperCase().replace("OPT_", "").replace("_NAME", "");
    let ctxSlotIndex = -1;
    if (["CONDITIONING", "INT", "STRING", "FLOAT", "COMBO"].includes(otherSlotType)) {
        ctxSlotIndex = ctxSlots.findIndex((ctxSlot) => {
            const ctxSlotName = ctxSlot.name.toUpperCase().replace("OPT_", "").replace("_NAME", "");
            let ctxSlotType = ctxSlot.type;
            if (Array.isArray(ctxSlotType) || ctxSlotType.includes(",")) {
                ctxSlotType = "COMBO";
            }
            if (ctxSlotType !== otherSlotType) {
                return false;
            }
            if (ctxSlotName === otherSlotName ||
                (ctxSlotName === "SEED" && otherSlotName.includes("SEED")) ||
                (ctxSlotName === "STEP_REFINER" && otherSlotName.includes("AT_STEP")) ||
                (ctxSlotName === "STEP_REFINER" && otherSlotName.includes("REFINER_STEP"))) {
                return true;
            }
            if ((otherNodeType.includes("POSITIVE") || otherNodeName.includes("POSITIVE")) &&
                ((ctxSlotName === "POSITIVE" && otherSlotType === "CONDITIONING") ||
                    (ctxSlotName === "TEXT_POS_G" && otherSlotName.includes("TEXT_G")) ||
                    (ctxSlotName === "TEXT_POS_L" && otherSlotName.includes("TEXT_L")))) {
                return true;
            }
            if ((otherNodeType.includes("NEGATIVE") || otherNodeName.includes("NEGATIVE")) &&
                ((ctxSlotName === "NEGATIVE" && otherSlotType === "CONDITIONING") ||
                    (ctxSlotName === "TEXT_NEG_G" && otherSlotName.includes("TEXT_G")) ||
                    (ctxSlotName === "TEXT_NEG_L" && otherSlotName.includes("TEXT_L")))) {
                return true;
            }
            return false;
        });
    }
    else {
        ctxSlotIndex = ctxSlots.map((s) => s.type).indexOf(otherSlotType);
    }
    return ctxSlotIndex;
}
export class BaseContextNode extends RgthreeBaseServerNode {
    constructor(title) {
        super(title);
        this.___collapsed_width = 0;
    }
    get _collapsed_width() {
        return this.___collapsed_width;
    }
    set _collapsed_width(width) {
        const canvas = app.canvas;
        const ctx = canvas.canvas.getContext("2d");
        const oldFont = ctx.font;
        ctx.font = canvas.title_text_font;
        let title = this.title.trim();
        this.___collapsed_width = 30 + (title ? 10 + ctx.measureText(title).width : 0);
        ctx.font = oldFont;
    }
    connectByType(slot, targetNode, targetSlotType, optsIn) {
        var _a;
        let canConnect = (_a = super.connectByType) === null || _a === void 0 ? void 0 : _a.call(this, slot, targetNode, targetSlotType, optsIn);
        if (!super.connectByType) {
            canConnect = LGraphNode.prototype.connectByType.call(this, slot, targetNode, targetSlotType, optsIn);
        }
        if (!canConnect && slot === 0) {
            const ctrlKey = KEY_EVENT_SERVICE.ctrlKey;
            for (const [index, input] of (targetNode.inputs || []).entries()) {
                if (input.link && !ctrlKey) {
                    continue;
                }
                const thisOutputSlot = findMatchingIndexByTypeOrName(targetNode, input, this.outputs);
                if (thisOutputSlot > -1) {
                    this.connect(thisOutputSlot, targetNode, index);
                }
            }
        }
        return null;
    }
    connectByTypeOutput(slot, sourceNode, sourceSlotType, optsIn) {
        var _a, _b;
        let canConnect = (_a = super.connectByTypeOutput) === null || _a === void 0 ? void 0 : _a.call(this, slot, sourceNode, sourceSlotType, optsIn);
        if (!super.connectByType) {
            canConnect = LGraphNode.prototype.connectByTypeOutput.call(this, slot, sourceNode, sourceSlotType, optsIn);
        }
        if (!canConnect && slot === 0) {
            const ctrlKey = KEY_EVENT_SERVICE.ctrlKey;
            for (const [index, output] of (sourceNode.outputs || []).entries()) {
                if (((_b = output.links) === null || _b === void 0 ? void 0 : _b.length) && !ctrlKey) {
                    continue;
                }
                const thisInputSlot = findMatchingIndexByTypeOrName(sourceNode, output, this.inputs);
                if (thisInputSlot > -1) {
                    sourceNode.connect(index, this, thisInputSlot);
                }
            }
        }
        return null;
    }
    static setUp(comfyClass, nodeData, ctxClass) {
        RgthreeBaseServerNode.registerForOverride(comfyClass, nodeData, ctxClass);
        wait(500).then(() => {
            LiteGraph.slot_types_default_out["RGTHREE_CONTEXT"] =
                LiteGraph.slot_types_default_out["RGTHREE_CONTEXT"] || [];
            LiteGraph.slot_types_default_out["RGTHREE_CONTEXT"].push(comfyClass.comfyClass);
        });
    }
    static onRegisteredForOverride(comfyClass, ctxClass) {
        addConnectionLayoutSupport(ctxClass, app, [
            ["Left", "Right"],
            ["Right", "Left"],
        ]);
        setTimeout(() => {
            ctxClass.category = comfyClass.category;
        });
    }
}
class ContextNode extends BaseContextNode {
    constructor(title = ContextNode.title) {
        super(title);
    }
    static setUp(comfyClass, nodeData) {
        BaseContextNode.setUp(comfyClass, nodeData, ContextNode);
    }
    static onRegisteredForOverride(comfyClass, ctxClass) {
        BaseContextNode.onRegisteredForOverride(comfyClass, ctxClass);
        addMenuItem(ContextNode, app, {
            name: "Convert To Context Big",
            callback: (node) => {
                replaceNode(node, ContextBigNode.type);
            },
        });
    }
}
ContextNode.title = NodeTypesString.CONTEXT;
ContextNode.type = NodeTypesString.CONTEXT;
ContextNode.comfyClass = NodeTypesString.CONTEXT;
class ContextBigNode extends BaseContextNode {
    constructor(title = ContextBigNode.title) {
        super(title);
    }
    static setUp(comfyClass, nodeData) {
        BaseContextNode.setUp(comfyClass, nodeData, ContextBigNode);
    }
    static onRegisteredForOverride(comfyClass, ctxClass) {
        BaseContextNode.onRegisteredForOverride(comfyClass, ctxClass);
        addMenuItem(ContextBigNode, app, {
            name: "Convert To Context (Original)",
            callback: (node) => {
                replaceNode(node, ContextNode.type);
            },
        });
    }
}
ContextBigNode.title = NodeTypesString.CONTEXT_BIG;
ContextBigNode.type = NodeTypesString.CONTEXT_BIG;
ContextBigNode.comfyClass = NodeTypesString.CONTEXT_BIG;
class BaseContextMultiCtxInputNode extends BaseContextNode {
    constructor(title) {
        super(title);
        this.stabilizeBound = this.stabilize.bind(this);
        this.addContextInput(5);
    }
    addContextInput(num = 1) {
        for (let i = 0; i < num; i++) {
            this.addInput(`ctx_${String(this.inputs.length + 1).padStart(2, "0")}`, "RGTHREE_CONTEXT");
        }
    }
    onConnectionsChange(type, slotIndex, isConnected, link, ioSlot) {
        var _a;
        (_a = super.onConnectionsChange) === null || _a === void 0 ? void 0 : _a.apply(this, [...arguments]);
        if (type === LiteGraph.INPUT) {
            this.scheduleStabilize();
        }
    }
    scheduleStabilize(ms = 64) {
        return debounce(this.stabilizeBound, 64);
    }
    stabilize() {
        removeUnusedInputsFromEnd(this, 4);
        this.addContextInput();
    }
}
class ContextSwitchNode extends BaseContextMultiCtxInputNode {
    constructor(title = ContextSwitchNode.title) {
        super(title);
    }
    static setUp(comfyClass, nodeData) {
        BaseContextNode.setUp(comfyClass, nodeData, ContextSwitchNode);
    }
    static onRegisteredForOverride(comfyClass, ctxClass) {
        BaseContextNode.onRegisteredForOverride(comfyClass, ctxClass);
        addMenuItem(ContextSwitchNode, app, {
            name: "Convert To Context Switch Big",
            callback: (node) => {
                replaceNode(node, ContextSwitchBigNode.type);
            },
        });
    }
}
ContextSwitchNode.title = NodeTypesString.CONTEXT_SWITCH;
ContextSwitchNode.type = NodeTypesString.CONTEXT_SWITCH;
ContextSwitchNode.comfyClass = NodeTypesString.CONTEXT_SWITCH;
class ContextSwitchBigNode extends BaseContextMultiCtxInputNode {
    constructor(title = ContextSwitchBigNode.title) {
        super(title);
    }
    static setUp(comfyClass, nodeData) {
        BaseContextNode.setUp(comfyClass, nodeData, ContextSwitchBigNode);
    }
    static onRegisteredForOverride(comfyClass, ctxClass) {
        BaseContextNode.onRegisteredForOverride(comfyClass, ctxClass);
        addMenuItem(ContextSwitchBigNode, app, {
            name: "Convert To Context Switch",
            callback: (node) => {
                replaceNode(node, ContextSwitchNode.type);
            },
        });
    }
}
ContextSwitchBigNode.title = NodeTypesString.CONTEXT_SWITCH_BIG;
ContextSwitchBigNode.type = NodeTypesString.CONTEXT_SWITCH_BIG;
ContextSwitchBigNode.comfyClass = NodeTypesString.CONTEXT_SWITCH_BIG;
class ContextMergeNode extends BaseContextMultiCtxInputNode {
    constructor(title = ContextMergeNode.title) {
        super(title);
    }
    static setUp(comfyClass, nodeData) {
        BaseContextNode.setUp(comfyClass, nodeData, ContextMergeNode);
    }
    static onRegisteredForOverride(comfyClass, ctxClass) {
        BaseContextNode.onRegisteredForOverride(comfyClass, ctxClass);
        addMenuItem(ContextMergeNode, app, {
            name: "Convert To Context Merge Big",
            callback: (node) => {
                replaceNode(node, ContextMergeBigNode.type);
            },
        });
    }
}
ContextMergeNode.title = NodeTypesString.CONTEXT_MERGE;
ContextMergeNode.type = NodeTypesString.CONTEXT_MERGE;
ContextMergeNode.comfyClass = NodeTypesString.CONTEXT_MERGE;
class ContextMergeBigNode extends BaseContextMultiCtxInputNode {
    constructor(title = ContextMergeBigNode.title) {
        super(title);
    }
    static setUp(comfyClass, nodeData) {
        BaseContextNode.setUp(comfyClass, nodeData, ContextMergeBigNode);
    }
    static onRegisteredForOverride(comfyClass, ctxClass) {
        BaseContextNode.onRegisteredForOverride(comfyClass, ctxClass);
        addMenuItem(ContextMergeBigNode, app, {
            name: "Convert To Context Switch",
            callback: (node) => {
                replaceNode(node, ContextMergeNode.type);
            },
        });
    }
}
ContextMergeBigNode.title = NodeTypesString.CONTEXT_MERGE_BIG;
ContextMergeBigNode.type = NodeTypesString.CONTEXT_MERGE_BIG;
ContextMergeBigNode.comfyClass = NodeTypesString.CONTEXT_MERGE_BIG;
const contextNodes = [
    ContextNode,
    ContextBigNode,
    ContextSwitchNode,
    ContextSwitchBigNode,
    ContextMergeNode,
    ContextMergeBigNode,
];
const contextTypeToServerDef = {};
function fixBadConfigs(node) {
    const wrongName = node.outputs.find((o, i) => o.name === "CLIP_HEIGTH");
    if (wrongName) {
        wrongName.name = "CLIP_HEIGHT";
    }
}
app.registerExtension({
    name: "rgthree.Context",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        for (const ctxClass of contextNodes) {
            if (nodeData.name === ctxClass.type) {
                contextTypeToServerDef[ctxClass.type] = nodeData;
                ctxClass.setUp(nodeType, nodeData);
                break;
            }
        }
    },
    async nodeCreated(node) {
        const type = node.type || node.constructor.type;
        const serverDef = type && contextTypeToServerDef[type];
        if (serverDef) {
            fixBadConfigs(node);
            matchLocalSlotsToServer(node, IoDirection.OUTPUT, serverDef);
            if (!type.includes("Switch") && !type.includes("Merge")) {
                matchLocalSlotsToServer(node, IoDirection.INPUT, serverDef);
            }
        }
    },
    async loadedGraphNode(node) {
        const type = node.type || node.constructor.type;
        const serverDef = type && contextTypeToServerDef[type];
        if (serverDef) {
            fixBadConfigs(node);
            matchLocalSlotsToServer(node, IoDirection.OUTPUT, serverDef);
            if (!type.includes("Switch") && !type.includes("Merge")) {
                matchLocalSlotsToServer(node, IoDirection.INPUT, serverDef);
            }
        }
    },
});
