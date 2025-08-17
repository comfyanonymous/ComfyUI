import { app } from "../../scripts/app.js";
import { BaseAnyInputConnectedNode } from "./base_any_input_connected_node.js";
import { NodeTypesString } from "./constants.js";
import { addMenuItem, changeModeOfNodes } from "./utils.js";
import { rgthree } from "./rgthree.js";
const MODE_ALWAYS = 0;
const MODE_MUTE = 2;
const MODE_BYPASS = 4;
class FastActionsButton extends BaseAnyInputConnectedNode {
    constructor(title) {
        super(title);
        this.comfyClass = NodeTypesString.FAST_ACTIONS_BUTTON;
        this.logger = rgthree.newLogSession("[FastActionsButton]");
        this.isVirtualNode = true;
        this.serialize_widgets = true;
        this.widgetToData = new Map();
        this.nodeIdtoFunctionCache = new Map();
        this.executingFromShortcut = false;
        this.properties["buttonText"] = "🎬 Action!";
        this.properties["shortcutModifier"] = "alt";
        this.properties["shortcutKey"] = "";
        this.buttonWidget = this.addWidget("button", this.properties["buttonText"], "", () => {
            this.executeConnectedNodes();
        }, { serialize: false });
        this.keypressBound = this.onKeypress.bind(this);
        this.keyupBound = this.onKeyup.bind(this);
        this.onConstructed();
    }
    configure(info) {
        super.configure(info);
        setTimeout(() => {
            if (info.widgets_values) {
                for (let [index, value] of info.widgets_values.entries()) {
                    if (index > 0) {
                        if (typeof value === "string" && value.startsWith("comfy_action:")) {
                            value = value.replace("comfy_action:", "");
                            this.addComfyActionWidget(index, value);
                        }
                        if (this.widgets[index]) {
                            this.widgets[index].value = value;
                        }
                    }
                }
            }
        }, 100);
    }
    clone() {
        const cloned = super.clone();
        cloned.properties["buttonText"] = "🎬 Action!";
        cloned.properties["shortcutKey"] = "";
        return cloned;
    }
    onAdded(graph) {
        window.addEventListener("keydown", this.keypressBound);
        window.addEventListener("keyup", this.keyupBound);
    }
    onRemoved() {
        window.removeEventListener("keydown", this.keypressBound);
        window.removeEventListener("keyup", this.keyupBound);
    }
    async onKeypress(event) {
        const target = event.target;
        if (this.executingFromShortcut ||
            target.localName == "input" ||
            target.localName == "textarea") {
            return;
        }
        if (this.properties["shortcutKey"].trim() &&
            this.properties["shortcutKey"].toLowerCase() === event.key.toLowerCase()) {
            const shortcutModifier = this.properties["shortcutModifier"];
            let good = shortcutModifier === "ctrl" && event.ctrlKey;
            good = good || (shortcutModifier === "alt" && event.altKey);
            good = good || (shortcutModifier === "shift" && event.shiftKey);
            good = good || (shortcutModifier === "meta" && event.metaKey);
            if (good) {
                setTimeout(() => {
                    this.executeConnectedNodes();
                }, 20);
                this.executingFromShortcut = true;
                event.preventDefault();
                event.stopImmediatePropagation();
                app.canvas.dirty_canvas = true;
                return false;
            }
        }
        return;
    }
    onKeyup(event) {
        const target = event.target;
        if (target.localName == "input" || target.localName == "textarea") {
            return;
        }
        this.executingFromShortcut = false;
    }
    onPropertyChanged(property, value, prevValue) {
        var _a, _b;
        if (property == "buttonText" && typeof value === "string") {
            this.buttonWidget.name = value;
        }
        if (property == "shortcutKey" && typeof value === "string") {
            this.properties["shortcutKey"] = (_b = (_a = value.trim()[0]) === null || _a === void 0 ? void 0 : _a.toLowerCase()) !== null && _b !== void 0 ? _b : "";
        }
        return true;
    }
    handleLinkedNodesStabilization(linkedNodes) {
        var _a, _b, _c, _d, _e, _f, _g, _h;
        let changed = false;
        for (const [widget, data] of this.widgetToData.entries()) {
            if (!data.node) {
                continue;
            }
            if (!linkedNodes.includes(data.node)) {
                const index = this.widgets.indexOf(widget);
                if (index > -1) {
                    this.widgetToData.delete(widget);
                    this.removeWidget(widget);
                    changed = true;
                }
                else {
                    const [m, a] = this.logger.debugParts("Connected widget is not in widgets... weird.");
                    (_a = console[m]) === null || _a === void 0 ? void 0 : _a.call(console, ...a);
                }
            }
        }
        const badNodes = [];
        let indexOffset = 1;
        for (const [index, node] of linkedNodes.entries()) {
            if (!node) {
                const [m, a] = this.logger.debugParts("linkedNode provided that does not exist. ");
                (_b = console[m]) === null || _b === void 0 ? void 0 : _b.call(console, ...a);
                badNodes.push(node);
                continue;
            }
            let widgetAtSlot = this.widgets[index + indexOffset];
            if (widgetAtSlot && ((_c = this.widgetToData.get(widgetAtSlot)) === null || _c === void 0 ? void 0 : _c.comfy)) {
                indexOffset++;
                widgetAtSlot = this.widgets[index + indexOffset];
            }
            if (!widgetAtSlot || ((_e = (_d = this.widgetToData.get(widgetAtSlot)) === null || _d === void 0 ? void 0 : _d.node) === null || _e === void 0 ? void 0 : _e.id) !== node.id) {
                let widget = null;
                for (let i = index + indexOffset; i < this.widgets.length; i++) {
                    if (((_g = (_f = this.widgetToData.get(this.widgets[i])) === null || _f === void 0 ? void 0 : _f.node) === null || _g === void 0 ? void 0 : _g.id) === node.id) {
                        widget = this.widgets.splice(i, 1)[0];
                        this.widgets.splice(index + indexOffset, 0, widget);
                        changed = true;
                        break;
                    }
                }
                if (!widget) {
                    const exposedActions = node.constructor.exposedActions || [];
                    widget = this.addWidget("combo", node.title, "None", "", {
                        values: ["None", "Mute", "Bypass", "Enable", ...exposedActions],
                    });
                    widget.serializeValue = async (_node, _index) => {
                        return widget === null || widget === void 0 ? void 0 : widget.value;
                    };
                    this.widgetToData.set(widget, { node });
                    changed = true;
                }
            }
        }
        for (let i = this.widgets.length - 1; i > linkedNodes.length + indexOffset - 1; i--) {
            const widgetAtSlot = this.widgets[i];
            if (widgetAtSlot && ((_h = this.widgetToData.get(widgetAtSlot)) === null || _h === void 0 ? void 0 : _h.comfy)) {
                continue;
            }
            this.removeWidget(widgetAtSlot);
            changed = true;
        }
        return changed;
    }
    removeWidget(widget) {
        widget = typeof widget === "number" ? this.widgets[widget] : widget;
        if (widget && this.widgetToData.has(widget)) {
            this.widgetToData.delete(widget);
        }
        super.removeWidget(widget);
    }
    async executeConnectedNodes() {
        var _a, _b;
        for (const widget of this.widgets) {
            if (widget == this.buttonWidget) {
                continue;
            }
            const action = widget.value;
            const { comfy, node } = (_a = this.widgetToData.get(widget)) !== null && _a !== void 0 ? _a : {};
            if (comfy) {
                if (action === "Queue Prompt") {
                    await comfy.queuePrompt(0);
                }
                continue;
            }
            if (node) {
                if (action === "Mute") {
                    changeModeOfNodes(node, MODE_MUTE);
                }
                else if (action === "Bypass") {
                    changeModeOfNodes(node, MODE_BYPASS);
                }
                else if (action === "Enable") {
                    changeModeOfNodes(node, MODE_ALWAYS);
                }
                if (node.handleAction) {
                    if (typeof action !== "string") {
                        throw new Error("Fast Actions Button action should be a string: " + action);
                    }
                    await node.handleAction(action);
                }
                (_b = this.graph) === null || _b === void 0 ? void 0 : _b.change();
                continue;
            }
            console.warn("Fast Actions Button has a widget without correct data.");
        }
    }
    addComfyActionWidget(slot, value) {
        let widget = this.addWidget("combo", "Comfy Action", "None", () => {
            if (String(widget.value).startsWith("MOVE ")) {
                this.widgets.push(this.widgets.splice(this.widgets.indexOf(widget), 1)[0]);
                widget.value = String(widget.rgthree_lastValue);
            }
            else if (String(widget.value).startsWith("REMOVE ")) {
                this.removeWidget(widget);
            }
            widget.rgthree_lastValue = widget.value;
        }, {
            values: ["None", "Queue Prompt", "REMOVE Comfy Action", "MOVE to end"],
        });
        widget.rgthree_lastValue = value;
        widget.serializeValue = async (_node, _index) => {
            return `comfy_app:${widget === null || widget === void 0 ? void 0 : widget.value}`;
        };
        this.widgetToData.set(widget, { comfy: app });
        if (slot != null) {
            this.widgets.splice(slot, 0, this.widgets.splice(this.widgets.indexOf(widget), 1)[0]);
        }
        return widget;
    }
    onSerialize(serialised) {
        var _a, _b;
        (_a = super.onSerialize) === null || _a === void 0 ? void 0 : _a.call(this, serialised);
        for (let [index, value] of (serialised.widgets_values || []).entries()) {
            if (((_b = this.widgets[index]) === null || _b === void 0 ? void 0 : _b.name) === "Comfy Action") {
                serialised.widgets_values[index] = `comfy_action:${value}`;
            }
        }
    }
    static setUp() {
        super.setUp();
        addMenuItem(this, app, {
            name: "➕ Append a Comfy Action",
            callback: (nodeArg) => {
                nodeArg.addComfyActionWidget();
            },
        });
    }
}
FastActionsButton.type = NodeTypesString.FAST_ACTIONS_BUTTON;
FastActionsButton.title = NodeTypesString.FAST_ACTIONS_BUTTON;
FastActionsButton["@buttonText"] = { type: "string" };
FastActionsButton["@shortcutModifier"] = {
    type: "combo",
    values: ["ctrl", "alt", "shift"],
};
FastActionsButton["@shortcutKey"] = { type: "string" };
FastActionsButton.collapsible = false;
app.registerExtension({
    name: "rgthree.FastActionsButton",
    registerCustomNodes() {
        FastActionsButton.setUp();
    },
    loadedGraphNode(node) {
        if (node.type == FastActionsButton.title) {
            node._tempWidth = node.size[0];
        }
    },
});
