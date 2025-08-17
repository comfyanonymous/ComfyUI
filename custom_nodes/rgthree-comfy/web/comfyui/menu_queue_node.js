import { app } from "../../scripts/app.js";
import { rgthree } from "./rgthree.js";
import { getGroupNodes, getOutputNodes } from "./utils.js";
import { SERVICE as CONFIG_SERVICE } from "./services/config_service.js";
function showQueueNodesMenuIfOutputNodesAreSelected(existingOptions) {
    if (CONFIG_SERVICE.getConfigValue("features.menu_queue_selected_nodes") === false) {
        return;
    }
    const outputNodes = getOutputNodes(Object.values(app.canvas.selected_nodes));
    const menuItem = {
        content: `Queue Selected Output Nodes (rgthree) &nbsp;`,
        className: "rgthree-contextmenu-item",
        callback: () => {
            rgthree.queueOutputNodes(outputNodes.map((n) => n.id));
        },
        disabled: !outputNodes.length,
    };
    let idx = existingOptions.findIndex((o) => (o === null || o === void 0 ? void 0 : o.content) === "Outputs") + 1;
    idx = idx || existingOptions.findIndex((o) => (o === null || o === void 0 ? void 0 : o.content) === "Align") + 1;
    idx = idx || 3;
    existingOptions.splice(idx, 0, menuItem);
}
function showQueueGroupNodesMenuIfGroupIsSelected(existingOptions) {
    if (CONFIG_SERVICE.getConfigValue("features.menu_queue_selected_nodes") === false) {
        return;
    }
    const group = rgthree.lastCanvasMouseEvent &&
        (app.canvas.getCurrentGraph() || app.graph).getGroupOnPos(rgthree.lastCanvasMouseEvent.canvasX, rgthree.lastCanvasMouseEvent.canvasY);
    const outputNodes = (group && getOutputNodes(getGroupNodes(group))) || null;
    const menuItem = {
        content: `Queue Group Output Nodes (rgthree) &nbsp;`,
        className: "rgthree-contextmenu-item",
        callback: () => {
            outputNodes && rgthree.queueOutputNodes(outputNodes.map((n) => n.id));
        },
        disabled: !(outputNodes === null || outputNodes === void 0 ? void 0 : outputNodes.length),
    };
    let idx = existingOptions.findIndex((o) => { var _a; return (_a = o === null || o === void 0 ? void 0 : o.content) === null || _a === void 0 ? void 0 : _a.startsWith("Queue Selected "); }) + 1;
    idx = idx || existingOptions.findIndex((o) => (o === null || o === void 0 ? void 0 : o.content) === "Outputs") + 1;
    idx = idx || existingOptions.findIndex((o) => (o === null || o === void 0 ? void 0 : o.content) === "Align") + 1;
    idx = idx || 3;
    existingOptions.splice(idx, 0, menuItem);
}
app.registerExtension({
    name: "rgthree.QueueNode",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function (canvas, options) {
            var _a;
            const extraOptions = (_a = getExtraMenuOptions === null || getExtraMenuOptions === void 0 ? void 0 : getExtraMenuOptions.call(this, canvas, options)) !== null && _a !== void 0 ? _a : [];
            showQueueNodesMenuIfOutputNodesAreSelected(options);
            showQueueGroupNodesMenuIfGroupIsSelected(options);
            return extraOptions;
        };
    },
    async setup() {
        const getCanvasMenuOptions = LGraphCanvas.prototype.getCanvasMenuOptions;
        LGraphCanvas.prototype.getCanvasMenuOptions = function (...args) {
            const options = getCanvasMenuOptions.apply(this, [...args]);
            showQueueNodesMenuIfOutputNodesAreSelected(options);
            showQueueGroupNodesMenuIfGroupIsSelected(options);
            return options;
        };
    },
});
