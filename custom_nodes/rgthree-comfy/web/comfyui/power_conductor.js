var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
var __classPrivateFieldSet = (this && this.__classPrivateFieldSet) || function (receiver, state, value, kind, f) {
    if (kind === "m") throw new TypeError("Private method is not writable");
    if (kind === "a" && !f) throw new TypeError("Private accessor was defined without a setter");
    if (typeof state === "function" ? receiver !== state || !f : !state.has(receiver)) throw new TypeError("Cannot write private member to an object whose class did not declare it");
    return (kind === "a" ? f.call(receiver, value) : f ? f.value = value : state.set(receiver, value)), value;
};
var __classPrivateFieldGet = (this && this.__classPrivateFieldGet) || function (receiver, state, kind, f) {
    if (kind === "a" && !f) throw new TypeError("Private accessor was defined without a getter");
    if (typeof state === "function" ? receiver !== state || !f : !state.has(receiver)) throw new TypeError("Cannot read private member from an object whose class did not declare it");
    return kind === "m" ? f : kind === "a" ? f.call(receiver) : f ? f.value : state.get(receiver);
};
var _ComfyNodeWrapper_id, _ComfyWidgetWrapper_widget;
import { app } from "../../scripts/app.js";
import { Exposed, execute, PyTuple } from "../../rgthree/common/py_parser.js";
import { RgthreeBaseVirtualNode } from "./base_node.js";
import { RgthreeBetterButtonWidget } from "./utils_widgets.js";
import { NodeTypesString } from "./constants.js";
import { ComfyWidgets } from "../../scripts/widgets.js";
import { SERVICE as CONFIG_SERVICE } from "./services/config_service.js";
import { changeModeOfNodes, getNodeById } from "./utils.js";
const BUILT_INS = {
    node: {
        fn: (query) => {
            if (typeof query === "number" || /^\d+(\.\d+)?/.exec(query)) {
                return new ComfyNodeWrapper(Number(query));
            }
            return null;
        },
    },
};
class RgthreePowerConductor extends RgthreeBaseVirtualNode {
    constructor(title = RgthreePowerConductor.title) {
        super(title);
        this.comfyClass = NodeTypesString.POWER_CONDUCTOR;
        this.serialize_widgets = true;
        this.codeWidget = ComfyWidgets.STRING(this, "", ["STRING", { multiline: true }], app).widget;
        this.addCustomWidget(this.codeWidget);
        (this.buttonWidget = new RgthreeBetterButtonWidget("Run", (...args) => {
            this.execute();
        })),
            this.addCustomWidget(this.buttonWidget);
        this.onConstructed();
    }
    execute() {
        execute(this.codeWidget.value, {}, BUILT_INS);
    }
}
RgthreePowerConductor.title = NodeTypesString.POWER_CONDUCTOR;
RgthreePowerConductor.type = NodeTypesString.POWER_CONDUCTOR;
const NODE_CLASS = RgthreePowerConductor;
class ComfyNodeWrapper {
    constructor(id) {
        _ComfyNodeWrapper_id.set(this, void 0);
        __classPrivateFieldSet(this, _ComfyNodeWrapper_id, id, "f");
    }
    getNode() {
        return getNodeById(__classPrivateFieldGet(this, _ComfyNodeWrapper_id, "f"));
    }
    get id() {
        return this.getNode().id;
    }
    get title() {
        return this.getNode().title;
    }
    set title(value) {
        this.getNode().title = value;
    }
    get widgets() {
        var _a;
        return new PyTuple((_a = this.getNode().widgets) === null || _a === void 0 ? void 0 : _a.map((w) => new ComfyWidgetWrapper(w)));
    }
    get mode() {
        return this.getNode().mode;
    }
    mute() {
        changeModeOfNodes(this.getNode(), 2);
    }
    bypass() {
        changeModeOfNodes(this.getNode(), 4);
    }
    enable() {
        changeModeOfNodes(this.getNode(), 0);
    }
}
_ComfyNodeWrapper_id = new WeakMap();
__decorate([
    Exposed
], ComfyNodeWrapper.prototype, "id", null);
__decorate([
    Exposed
], ComfyNodeWrapper.prototype, "title", null);
__decorate([
    Exposed
], ComfyNodeWrapper.prototype, "widgets", null);
__decorate([
    Exposed
], ComfyNodeWrapper.prototype, "mode", null);
__decorate([
    Exposed
], ComfyNodeWrapper.prototype, "mute", null);
__decorate([
    Exposed
], ComfyNodeWrapper.prototype, "bypass", null);
__decorate([
    Exposed
], ComfyNodeWrapper.prototype, "enable", null);
class ComfyWidgetWrapper {
    constructor(widget) {
        _ComfyWidgetWrapper_widget.set(this, void 0);
        __classPrivateFieldSet(this, _ComfyWidgetWrapper_widget, widget, "f");
    }
    get value() {
        return __classPrivateFieldGet(this, _ComfyWidgetWrapper_widget, "f").value;
    }
    get label() {
        return __classPrivateFieldGet(this, _ComfyWidgetWrapper_widget, "f").label;
    }
    toggle(value) {
        if (typeof __classPrivateFieldGet(this, _ComfyWidgetWrapper_widget, "f")["toggle"] === "function") {
            __classPrivateFieldGet(this, _ComfyWidgetWrapper_widget, "f")["toggle"](value);
        }
        else {
        }
    }
}
_ComfyWidgetWrapper_widget = new WeakMap();
__decorate([
    Exposed
], ComfyWidgetWrapper.prototype, "value", null);
__decorate([
    Exposed
], ComfyWidgetWrapper.prototype, "label", null);
__decorate([
    Exposed
], ComfyWidgetWrapper.prototype, "toggle", null);
app.registerExtension({
    name: "rgthree.PowerConductor",
    registerCustomNodes() {
        if (CONFIG_SERVICE.getConfigValue("unreleased.power_conductor.enabled")) {
            NODE_CLASS.setUp();
        }
    },
});
