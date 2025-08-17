import { app } from "../../scripts/app.js";
import { RgthreeBaseServerNode } from "./base_node.js";
import { NodeTypesString } from "./constants.js";
import { ComfyWidgets } from "../../scripts/widgets.js";
import { moveArrayItem } from "../../rgthree/common/shared_utils.js";
const PROPERTY_HIDE_TYPE_SELECTOR = "hideTypeSelector";
const PRIMITIVES = {
    STRING: "STRING",
    INT: "INT",
    FLOAT: "FLOAT",
    BOOLEAN: "BOOLEAN",
};
class RgthreePowerPrimitive extends RgthreeBaseServerNode {
    constructor(title = NODE_CLASS.title) {
        super(title);
        this.typeState = '';
        this.properties[PROPERTY_HIDE_TYPE_SELECTOR] = false;
    }
    static setUp(comfyClass, nodeData) {
        RgthreeBaseServerNode.registerForOverride(comfyClass, nodeData, NODE_CLASS);
    }
    onNodeCreated() {
        var _a;
        (_a = super.onNodeCreated) === null || _a === void 0 ? void 0 : _a.call(this);
        this.addInitialWidgets();
    }
    configure(info) {
        super.configure(info);
        if (this.outputTypeWidget.value === 'BOOL') {
            this.outputTypeWidget.value = 'BOOLEAN';
        }
        setTimeout(() => {
            this.setTypedData();
        });
    }
    getExtraMenuOptions(canvas, options) {
        const that = this;
        super.getExtraMenuOptions(canvas, options);
        const isHidden = !!this.properties[PROPERTY_HIDE_TYPE_SELECTOR];
        const menuItems = [
            {
                content: `${isHidden ? "Show" : "Hide"} Type Selector Widget`,
                callback: (...args) => {
                    this.setProperty(PROPERTY_HIDE_TYPE_SELECTOR, !this.properties[PROPERTY_HIDE_TYPE_SELECTOR]);
                },
            },
            {
                content: `Set type`,
                submenu: {
                    options: Object.keys(PRIMITIVES),
                    callback(value, ...args) {
                        that.outputTypeWidget.value = value;
                        that.setTypedData();
                    },
                },
            },
        ];
        options.splice(0, 0, ...menuItems, null);
        return [];
    }
    addInitialWidgets() {
        if (!this.outputTypeWidget) {
            this.outputTypeWidget = this.addWidget("combo", "type", "STRING", (...args) => {
                this.setTypedData();
            }, {
                values: Object.keys(PRIMITIVES),
            });
            this.outputTypeWidget.hidden = this.properties[PROPERTY_HIDE_TYPE_SELECTOR];
        }
        this.setTypedData();
    }
    setTypedData() {
        var _a, _b, _c, _d, _e;
        const name = "value";
        const type = this.outputTypeWidget.value;
        const linked = !!((_b = (_a = this.inputs) === null || _a === void 0 ? void 0 : _a[0]) === null || _b === void 0 ? void 0 : _b.link);
        const newTypeState = `${type}|${linked}`;
        if (this.typeState == newTypeState)
            return;
        this.typeState = newTypeState;
        let value = (_d = (_c = this.valueWidget) === null || _c === void 0 ? void 0 : _c.value) !== null && _d !== void 0 ? _d : null;
        let newWidget = null;
        if (linked) {
            newWidget = ComfyWidgets["STRING"](this, name, ["STRING"], app).widget;
            newWidget.value = "";
        }
        else if (type == "STRING") {
            newWidget = ComfyWidgets["STRING"](this, name, ["STRING", { multiline: true }], app).widget;
            newWidget.value = value ? "" : String(value);
        }
        else if (type === "INT" || type === "FLOAT") {
            const isFloat = type === "FLOAT";
            newWidget = this.addWidget("number", name, value !== null && value !== void 0 ? value : 1, undefined, {
                precision: isFloat ? 1 : 0,
                step2: isFloat ? 0.1 : 0,
            });
            value = Number(value);
            value = value == null || isNaN(value) ? 0 : value;
            newWidget.value = value;
        }
        else if (type === "BOOLEAN") {
            newWidget = this.addWidget("toggle", name, !!(value !== null && value !== void 0 ? value : true), undefined, {
                on: "true",
                off: "false",
            });
            if (typeof value === "string") {
                value = !["false", "null", "None", "", "0"].includes(value.toLowerCase());
            }
            newWidget.value = !!value;
        }
        if (newWidget == null) {
            throw new Error(`Unsupported type "${type}".`);
        }
        if (this.valueWidget) {
            this.replaceWidget(this.valueWidget, newWidget);
        }
        else {
            if (!this.widgets.includes(newWidget)) {
                this.addCustomWidget(newWidget);
            }
            moveArrayItem(this.widgets, newWidget, 1);
        }
        this.valueWidget = newWidget;
        if (!((_e = this.inputs) === null || _e === void 0 ? void 0 : _e.length)) {
            this.addInput("value", "*", { widget: this.valueWidget });
        }
        else {
            this.inputs[0].widget = this.valueWidget;
        }
        const output = this.outputs[0];
        const outputLabel = output.label === "*" || output.label === output.type ? null : output.label;
        output.type = type;
        output.label = outputLabel || output.type;
    }
    onConnectionsChange(type, index, isConnected, link_info, inputOrOutput) {
        var _a;
        (_a = super.onConnectionsChange) === null || _a === void 0 ? void 0 : _a.apply(this, [...arguments]);
        if (this.inputs.includes(inputOrOutput)) {
            this.setTypedData();
        }
    }
    onPropertyChanged(name, value, prev_value) {
        if (name === PROPERTY_HIDE_TYPE_SELECTOR) {
            if (!this.outputTypeWidget) {
                return true;
            }
            this.outputTypeWidget.hidden = this.properties[PROPERTY_HIDE_TYPE_SELECTOR];
            if (this.outputTypeWidget.hidden) {
                this.outputTypeWidget.computeLayoutSize = () => ({
                    minHeight: 0,
                    minWidth: 0,
                    maxHeight: 0,
                    maxWidth: 0,
                });
            }
            else {
                this.outputTypeWidget.computeLayoutSize = undefined;
            }
        }
        return true;
    }
}
RgthreePowerPrimitive.title = NodeTypesString.POWER_PRIMITIVE;
RgthreePowerPrimitive.type = NodeTypesString.POWER_PRIMITIVE;
RgthreePowerPrimitive.comfyClass = NodeTypesString.POWER_PRIMITIVE;
RgthreePowerPrimitive["@hideTypeSelector"] = { type: "boolean" };
const NODE_CLASS = RgthreePowerPrimitive;
app.registerExtension({
    name: "rgthree.PowerPrimitive",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === NODE_CLASS.type) {
            NODE_CLASS.setUp(nodeType, nodeData);
        }
    },
});
