import { app } from "../../scripts/app.js";
import { rgthreeApi } from "../../rgthree/common/rgthree_api.js";
const PASS_THROUGH = function (item) {
    return item;
};
export async function showLoraChooser(event, callback, parentMenu, loras) {
    var _a, _b;
    const canvas = app.canvas;
    if (!loras) {
        loras = ["None", ...(await rgthreeApi.getLoras().then((loras) => loras.map((l) => l.file)))];
    }
    new LiteGraph.ContextMenu(loras, {
        event: event,
        parentMenu: parentMenu != null ? parentMenu : undefined,
        title: "Choose a lora",
        scale: Math.max(1, (_b = (_a = canvas.ds) === null || _a === void 0 ? void 0 : _a.scale) !== null && _b !== void 0 ? _b : 1),
        className: "dark",
        callback,
    });
}
export function showNodesChooser(event, mapFn, callback, parentMenu) {
    var _a, _b;
    const canvas = app.canvas;
    const nodesOptions = app.graph._nodes
        .map(mapFn)
        .filter((e) => e != null);
    nodesOptions.sort((a, b) => {
        return a.value - b.value;
    });
    new LiteGraph.ContextMenu(nodesOptions, {
        event: event,
        parentMenu,
        title: "Choose a node id",
        scale: Math.max(1, (_b = (_a = canvas.ds) === null || _a === void 0 ? void 0 : _a.scale) !== null && _b !== void 0 ? _b : 1),
        className: "dark",
        callback,
    });
}
export function showWidgetsChooser(event, node, mapFn, callback, parentMenu) {
    var _a, _b;
    const options = (node.widgets || [])
        .map(mapFn)
        .filter((e) => e != null);
    if (options.length) {
        const canvas = app.canvas;
        new LiteGraph.ContextMenu(options, {
            event,
            parentMenu,
            title: "Choose an input/widget",
            scale: Math.max(1, (_b = (_a = canvas.ds) === null || _a === void 0 ? void 0 : _a.scale) !== null && _b !== void 0 ? _b : 1),
            className: "dark",
            callback,
        });
    }
}
