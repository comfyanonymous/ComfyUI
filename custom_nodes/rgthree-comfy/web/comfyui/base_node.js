import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";
import { SERVICE as KEY_EVENT_SERVICE } from "./services/key_events_services.js";
import { LogLevel, rgthree } from "./rgthree.js";
import { addHelpMenuItem } from "./utils.js";
import { RgthreeHelpDialog } from "../../rgthree/common/dialog.js";
import { importIndividualNodesInnerOnDragDrop, importIndividualNodesInnerOnDragOver, } from "./feature_import_individual_nodes.js";
import { defineProperty, moveArrayItem } from "../../rgthree/common/shared_utils.js";
export class RgthreeBaseNode extends LGraphNode {
    constructor(title = RgthreeBaseNode.title, skipOnConstructedCall = true) {
        super(title);
        this.comfyClass = "__NEED_COMFY_CLASS__";
        this.nickname = "rgthree";
        this.isVirtualNode = false;
        this.isDropEnabled = false;
        this.removed = false;
        this.configuring = false;
        this._tempWidth = 0;
        this.__constructed__ = false;
        this.helpDialog = null;
        if (title == "__NEED_CLASS_TITLE__") {
            throw new Error("RgthreeBaseNode needs overrides.");
        }
        this.widgets = this.widgets || [];
        this.properties = this.properties || {};
        setTimeout(() => {
            if (this.comfyClass == "__NEED_COMFY_CLASS__") {
                throw new Error("RgthreeBaseNode needs a comfy class override.");
            }
            if (this.constructor.type == "__NEED_CLASS_TYPE__") {
                throw new Error("RgthreeBaseNode needs overrides.");
            }
            this.checkAndRunOnConstructed();
        });
        defineProperty(this, "mode", {
            get: () => {
                return this.rgthree_mode;
            },
            set: (mode) => {
                if (this.rgthree_mode != mode) {
                    const oldMode = this.rgthree_mode;
                    this.rgthree_mode = mode;
                    this.onModeChange(oldMode, mode);
                }
            },
        });
    }
    checkAndRunOnConstructed() {
        var _a;
        if (!this.__constructed__) {
            this.onConstructed();
            const [n, v] = rgthree.logger.logParts(LogLevel.DEV, `[RgthreeBaseNode] Child class did not call onConstructed for "${this.type}.`);
            (_a = console[n]) === null || _a === void 0 ? void 0 : _a.call(console, ...v);
        }
        return this.__constructed__;
    }
    onDragOver(e) {
        if (!this.isDropEnabled)
            return false;
        return importIndividualNodesInnerOnDragOver(this, e);
    }
    async onDragDrop(e) {
        if (!this.isDropEnabled)
            return false;
        return importIndividualNodesInnerOnDragDrop(this, e);
    }
    onConstructed() {
        var _a;
        if (this.__constructed__)
            return false;
        this.type = (_a = this.type) !== null && _a !== void 0 ? _a : undefined;
        this.__constructed__ = true;
        rgthree.invokeExtensionsAsync("nodeCreated", this);
        return this.__constructed__;
    }
    configure(info) {
        this.configuring = true;
        super.configure(info);
        for (const w of this.widgets || []) {
            w.last_y = w.last_y || 0;
        }
        this.configuring = false;
    }
    clone() {
        const cloned = super.clone();
        if ((cloned === null || cloned === void 0 ? void 0 : cloned.properties) && !!window.structuredClone) {
            cloned.properties = structuredClone(cloned.properties);
        }
        cloned.graph = this.graph;
        return cloned;
    }
    onModeChange(from, to) {
    }
    async handleAction(action) {
        action;
    }
    removeWidget(widget) {
        var _a;
        if (typeof widget === "number") {
            widget = this.widgets[widget];
        }
        if (!widget)
            return;
        if (typeof super.removeWidget === 'function') {
            super.removeWidget(widget);
        }
        else {
            const index = this.widgets.indexOf(widget);
            if (index > -1) {
                this.widgets.splice(index, 1);
            }
            (_a = widget.onRemove) === null || _a === void 0 ? void 0 : _a.call(widget);
        }
    }
    replaceWidget(widgetOrSlot, newWidget) {
        let index = null;
        if (widgetOrSlot) {
            index = typeof widgetOrSlot === "number" ? widgetOrSlot : this.widgets.indexOf(widgetOrSlot);
            this.removeWidget(this.widgets[index]);
        }
        index = index != null ? index : this.widgets.length - 1;
        if (this.widgets.includes(newWidget)) {
            moveArrayItem(this.widgets, newWidget, index);
        }
        else {
            this.widgets.splice(index, 0, newWidget);
        }
    }
    defaultGetSlotMenuOptions(slot) {
        var _a, _b;
        const menu_info = [];
        if ((_b = (_a = slot === null || slot === void 0 ? void 0 : slot.output) === null || _a === void 0 ? void 0 : _a.links) === null || _b === void 0 ? void 0 : _b.length) {
            menu_info.push({ content: "Disconnect Links", slot });
        }
        let inputOrOutput = slot.input || slot.output;
        if (inputOrOutput) {
            if (inputOrOutput.removable) {
                menu_info.push(inputOrOutput.locked ? { content: "Cannot remove" } : { content: "Remove Slot", slot });
            }
            if (!inputOrOutput.nameLocked) {
                menu_info.push({ content: "Rename Slot", slot });
            }
        }
        return menu_info;
    }
    onRemoved() {
        var _a;
        (_a = super.onRemoved) === null || _a === void 0 ? void 0 : _a.call(this);
        this.removed = true;
    }
    static setUp(...args) {
    }
    getHelp() {
        return "";
    }
    showHelp() {
        const help = this.getHelp() || this.constructor.help;
        if (help) {
            this.helpDialog = new RgthreeHelpDialog(this, help).show();
            this.helpDialog.addEventListener("close", (e) => {
                this.helpDialog = null;
            });
        }
    }
    onKeyDown(event) {
        KEY_EVENT_SERVICE.handleKeyDownOrUp(event);
        if (event.key == "?" && !this.helpDialog) {
            this.showHelp();
        }
    }
    onKeyUp(event) {
        KEY_EVENT_SERVICE.handleKeyDownOrUp(event);
    }
    getExtraMenuOptions(canvas, options) {
        var _a, _b, _c, _d, _e, _f;
        if (super.getExtraMenuOptions) {
            (_a = super.getExtraMenuOptions) === null || _a === void 0 ? void 0 : _a.apply(this, [canvas, options]);
        }
        else if ((_c = (_b = this.constructor.nodeType) === null || _b === void 0 ? void 0 : _b.prototype) === null || _c === void 0 ? void 0 : _c.getExtraMenuOptions) {
            (_f = (_e = (_d = this.constructor.nodeType) === null || _d === void 0 ? void 0 : _d.prototype) === null || _e === void 0 ? void 0 : _e.getExtraMenuOptions) === null || _f === void 0 ? void 0 : _f.apply(this, [canvas, options]);
        }
        const help = this.getHelp() || this.constructor.help;
        if (help) {
            addHelpMenuItem(this, help, options);
        }
        return [];
    }
}
RgthreeBaseNode.exposedActions = [];
RgthreeBaseNode.title = "__NEED_CLASS_TITLE__";
RgthreeBaseNode.type = "__NEED_CLASS_TYPE__";
RgthreeBaseNode.category = "rgthree";
RgthreeBaseNode._category = "rgthree";
export class RgthreeBaseVirtualNode extends RgthreeBaseNode {
    constructor(title = RgthreeBaseNode.title) {
        super(title, false);
        this.isVirtualNode = true;
    }
    static setUp() {
        if (!this.type) {
            throw new Error(`Missing type for RgthreeBaseVirtualNode: ${this.title}`);
        }
        LiteGraph.registerNodeType(this.type, this);
        if (this._category) {
            this.category = this._category;
        }
    }
}
export class RgthreeBaseServerNode extends RgthreeBaseNode {
    constructor(title) {
        super(title, true);
        this.isDropEnabled = true;
        this.serialize_widgets = true;
        this.setupFromServerNodeData();
        this.onConstructed();
    }
    getWidgets() {
        return ComfyWidgets;
    }
    async setupFromServerNodeData() {
        var _a, _b, _c, _d, _e;
        const nodeData = this.constructor.nodeData;
        if (!nodeData) {
            throw Error("No node data");
        }
        this.comfyClass = nodeData.name;
        let inputs = nodeData["input"]["required"];
        if (nodeData["input"]["optional"] != undefined) {
            inputs = Object.assign({}, inputs, nodeData["input"]["optional"]);
        }
        const WIDGETS = this.getWidgets();
        const config = {
            minWidth: 1,
            minHeight: 1,
            widget: null,
        };
        for (const inputName in inputs) {
            const inputData = inputs[inputName];
            const type = inputData[0];
            if ((_a = inputData[1]) === null || _a === void 0 ? void 0 : _a.forceInput) {
                this.addInput(inputName, type);
            }
            else {
                let widgetCreated = true;
                if (Array.isArray(type)) {
                    Object.assign(config, WIDGETS.COMBO(this, inputName, inputData, app) || {});
                }
                else if (`${type}:${inputName}` in WIDGETS) {
                    Object.assign(config, WIDGETS[`${type}:${inputName}`](this, inputName, inputData, app) || {});
                }
                else if (type in WIDGETS) {
                    Object.assign(config, WIDGETS[type](this, inputName, inputData, app) || {});
                }
                else {
                    this.addInput(inputName, type);
                    widgetCreated = false;
                }
                if (widgetCreated && ((_b = inputData[1]) === null || _b === void 0 ? void 0 : _b.forceInput) && (config === null || config === void 0 ? void 0 : config.widget)) {
                    if (!config.widget.options)
                        config.widget.options = {};
                    config.widget.options.forceInput = inputData[1].forceInput;
                }
                if (widgetCreated && ((_c = inputData[1]) === null || _c === void 0 ? void 0 : _c.defaultInput) && (config === null || config === void 0 ? void 0 : config.widget)) {
                    if (!config.widget.options)
                        config.widget.options = {};
                    config.widget.options.defaultInput = inputData[1].defaultInput;
                }
            }
        }
        for (const o in nodeData["output"]) {
            let output = nodeData["output"][o];
            if (output instanceof Array)
                output = "COMBO";
            const outputName = nodeData["output_name"][o] || output;
            const outputShape = nodeData["output_is_list"][o]
                ? LiteGraph.GRID_SHAPE
                : LiteGraph.CIRCLE_SHAPE;
            this.addOutput(outputName, output, { shape: outputShape });
        }
        const s = this.computeSize();
        s[0] = Math.max((_d = config.minWidth) !== null && _d !== void 0 ? _d : 1, s[0] * 1.5);
        s[1] = Math.max((_e = config.minHeight) !== null && _e !== void 0 ? _e : 1, s[1]);
        this.size = s;
        this.serialize_widgets = true;
    }
    static registerForOverride(comfyClass, nodeData, rgthreeClass) {
        if (OVERRIDDEN_SERVER_NODES.has(comfyClass)) {
            throw Error(`Already have a class to override ${comfyClass.type || comfyClass.name || comfyClass.title}`);
        }
        OVERRIDDEN_SERVER_NODES.set(comfyClass, rgthreeClass);
        if (!rgthreeClass.__registeredForOverride__) {
            rgthreeClass.__registeredForOverride__ = true;
            rgthreeClass.nodeType = comfyClass;
            rgthreeClass.nodeData = nodeData;
            rgthreeClass.onRegisteredForOverride(comfyClass, rgthreeClass);
        }
    }
    static onRegisteredForOverride(comfyClass, rgthreeClass) {
    }
}
RgthreeBaseServerNode.nodeType = null;
RgthreeBaseServerNode.nodeData = null;
RgthreeBaseServerNode.__registeredForOverride__ = false;
const OVERRIDDEN_SERVER_NODES = new Map();
const oldregisterNodeType = LiteGraph.registerNodeType;
LiteGraph.registerNodeType = async function (nodeId, baseClass) {
    var _a;
    const clazz = OVERRIDDEN_SERVER_NODES.get(baseClass) || baseClass;
    if (clazz !== baseClass) {
        const classLabel = clazz.type || clazz.name || clazz.title;
        const [n, v] = rgthree.logger.logParts(LogLevel.DEBUG, `${nodeId}: replacing default ComfyNode implementation with custom ${classLabel} class.`);
        (_a = console[n]) === null || _a === void 0 ? void 0 : _a.call(console, ...v);
    }
    return oldregisterNodeType.call(LiteGraph, nodeId, clazz);
};
