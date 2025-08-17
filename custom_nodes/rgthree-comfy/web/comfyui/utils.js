import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { getResolver, wait } from "../../rgthree/common/shared_utils.js";
import { RgthreeHelpDialog } from "../../rgthree/common/dialog.js";
const oldApiGetNodeDefs = api.getNodeDefs;
api.getNodeDefs = async function () {
    const defs = await oldApiGetNodeDefs.call(api);
    this.dispatchEvent(new CustomEvent("fresh-node-defs", { detail: defs }));
    return defs;
};
export var IoDirection;
(function (IoDirection) {
    IoDirection[IoDirection["INPUT"] = 0] = "INPUT";
    IoDirection[IoDirection["OUTPUT"] = 1] = "OUTPUT";
})(IoDirection || (IoDirection = {}));
const PADDING = 0;
export const LAYOUT_LABEL_TO_DATA = {
    Left: [LiteGraph.LEFT, [0, 0.5], [PADDING, 0]],
    Right: [LiteGraph.RIGHT, [1, 0.5], [-PADDING, 0]],
    Top: [LiteGraph.UP, [0.5, 0], [0, PADDING]],
    Bottom: [LiteGraph.DOWN, [0.5, 1], [0, -PADDING]],
};
export const LAYOUT_LABEL_OPPOSITES = {
    Left: "Right",
    Right: "Left",
    Top: "Bottom",
    Bottom: "Top",
};
export const LAYOUT_CLOCKWISE = ["Top", "Right", "Bottom", "Left"];
export function addMenuItem(node, _app, config, after = "Shape") {
    const oldGetExtraMenuOptions = node.prototype.getExtraMenuOptions;
    node.prototype.getExtraMenuOptions = function (canvas, menuOptions) {
        oldGetExtraMenuOptions && oldGetExtraMenuOptions.apply(this, [canvas, menuOptions]);
        addMenuItemOnExtraMenuOptions(this, config, menuOptions, after);
    };
}
let canvasResolver = null;
export function waitForCanvas() {
    if (canvasResolver === null) {
        canvasResolver = getResolver();
        function _waitForCanvas() {
            if (!canvasResolver.completed) {
                if (app === null || app === void 0 ? void 0 : app.canvas) {
                    canvasResolver.resolve(app.canvas);
                }
                else {
                    requestAnimationFrame(_waitForCanvas);
                }
            }
        }
        _waitForCanvas();
    }
    return canvasResolver.promise;
}
let graphResolver = null;
export function waitForGraph() {
    if (graphResolver === null) {
        graphResolver = getResolver();
        function _wait() {
            if (!graphResolver.completed) {
                if (app === null || app === void 0 ? void 0 : app.graph) {
                    graphResolver.resolve(app.graph);
                }
                else {
                    requestAnimationFrame(_wait);
                }
            }
        }
        _wait();
    }
    return graphResolver.promise;
}
export function addMenuItemOnExtraMenuOptions(node, config, menuOptions, after = "Shape") {
    let idx = menuOptions
        .slice()
        .reverse()
        .findIndex((option) => option === null || option === void 0 ? void 0 : option.isRgthree);
    if (idx == -1) {
        idx = menuOptions.findIndex((option) => { var _a; return (_a = option === null || option === void 0 ? void 0 : option.content) === null || _a === void 0 ? void 0 : _a.includes(after); }) + 1;
        if (!idx) {
            idx = menuOptions.length - 1;
        }
        menuOptions.splice(idx, 0, null);
        idx++;
    }
    else {
        idx = menuOptions.length - idx;
    }
    const subMenuOptions = typeof config.subMenuOptions === "function"
        ? config.subMenuOptions(node)
        : config.subMenuOptions;
    menuOptions.splice(idx, 0, {
        content: typeof config.name == "function" ? config.name(node) : config.name,
        has_submenu: !!(subMenuOptions === null || subMenuOptions === void 0 ? void 0 : subMenuOptions.length),
        isRgthree: true,
        callback: (value, _options, event, parentMenu, _node) => {
            if (!!(subMenuOptions === null || subMenuOptions === void 0 ? void 0 : subMenuOptions.length)) {
                new LiteGraph.ContextMenu(subMenuOptions.map((option) => (option ? { content: option } : null)), {
                    event,
                    parentMenu,
                    callback: (subValue, _options, _event, _parentMenu, _node) => {
                        if (config.property) {
                            node.properties = node.properties || {};
                            node.properties[config.property] = config.prepareValue
                                ? config.prepareValue(subValue.content || "", node)
                                : subValue.content || "";
                        }
                        config.callback && config.callback(node, subValue === null || subValue === void 0 ? void 0 : subValue.content);
                    },
                });
                return;
            }
            if (config.property) {
                node.properties = node.properties || {};
                node.properties[config.property] = config.prepareValue
                    ? config.prepareValue(node.properties[config.property], node)
                    : !node.properties[config.property];
            }
            config.callback && config.callback(node, value === null || value === void 0 ? void 0 : value.content);
        },
    });
}
export function addConnectionLayoutSupport(node, app, options = [
    ["Left", "Right"],
    ["Right", "Left"],
], callback) {
    addMenuItem(node, app, {
        name: "Connections Layout",
        property: "connections_layout",
        subMenuOptions: options.map((option) => option[0] + (option[1] ? " -> " + option[1] : "")),
        prepareValue: (value, node) => {
            var _a;
            const values = String(value).split(" -> ");
            if (!values[1] && !((_a = node.outputs) === null || _a === void 0 ? void 0 : _a.length)) {
                values[1] = LAYOUT_LABEL_OPPOSITES[values[0]];
            }
            if (!LAYOUT_LABEL_TO_DATA[values[0]] || !LAYOUT_LABEL_TO_DATA[values[1]]) {
                throw new Error(`New Layout invalid: [${values[0]}, ${values[1]}]`);
            }
            return values;
        },
        callback: (node) => {
            var _a;
            callback && callback(node);
            (_a = node.graph) === null || _a === void 0 ? void 0 : _a.setDirtyCanvas(true, true);
        },
    });
    node.prototype.getConnectionPos = function (isInput, slotNumber, out) {
        return getConnectionPosForLayout(this, isInput, slotNumber, out);
    };
    node.prototype.getInputPos = function (slotNumber) {
        return getConnectionPosForLayout(this, true, slotNumber, [0, 0]);
    };
    node.prototype.getOutputPos = function (slotNumber) {
        return getConnectionPosForLayout(this, false, slotNumber, [0, 0]);
    };
}
export function setConnectionsLayout(node, newLayout) {
    var _a;
    newLayout = newLayout || node.defaultConnectionsLayout || ["Left", "Right"];
    if (!newLayout[1] && !((_a = node.outputs) === null || _a === void 0 ? void 0 : _a.length)) {
        newLayout[1] = LAYOUT_LABEL_OPPOSITES[newLayout[0]];
    }
    if (!LAYOUT_LABEL_TO_DATA[newLayout[0]] || !LAYOUT_LABEL_TO_DATA[newLayout[1]]) {
        throw new Error(`New Layout invalid: [${newLayout[0]}, ${newLayout[1]}]`);
    }
    node.properties = node.properties || {};
    node.properties["connections_layout"] = newLayout;
}
export function setConnectionsCollapse(node, collapseConnections = null) {
    node.properties = node.properties || {};
    collapseConnections =
        collapseConnections !== null ? collapseConnections : !node.properties["collapse_connections"];
    node.properties["collapse_connections"] = collapseConnections;
}
export function getConnectionPosForLayout(node, isInput, slotNumber, out) {
    var _a, _b, _c;
    out = out || new Float32Array(2);
    node.properties = node.properties || {};
    const layout = node.properties["connections_layout"] ||
        node.defaultConnectionsLayout || ["Left", "Right"];
    const collapseConnections = node.properties["collapse_connections"] || false;
    const offset = (_a = node.constructor.layout_slot_offset) !== null && _a !== void 0 ? _a : LiteGraph.NODE_SLOT_HEIGHT * 0.5;
    let side = isInput ? layout[0] : layout[1];
    const otherSide = isInput ? layout[1] : layout[0];
    let data = LAYOUT_LABEL_TO_DATA[side];
    const slotList = node[isInput ? "inputs" : "outputs"];
    const cxn = slotList[slotNumber];
    if (!cxn) {
        console.log("No connection found.. weird", isInput, slotNumber);
        return out;
    }
    if (cxn.disabled) {
        if (cxn.color_on !== "#666665") {
            cxn._color_on_org = cxn._color_on_org || cxn.color_on;
            cxn._color_off_org = cxn._color_off_org || cxn.color_off;
        }
        cxn.color_on = "#666665";
        cxn.color_off = "#666665";
    }
    else if (cxn.color_on === "#666665") {
        cxn.color_on = cxn._color_on_org || undefined;
        cxn.color_off = cxn._color_off_org || undefined;
    }
    const displaySlot = collapseConnections
        ? 0
        : slotNumber -
            slotList.reduce((count, ioput, index) => {
                count += index < slotNumber && ioput.hidden ? 1 : 0;
                return count;
            }, 0);
    cxn.dir = data[0];
    const connections_dir = node.properties["connections_dir"];
    if ((node.size[0] == 10 || node.size[1] == 10) && connections_dir) {
        cxn.dir = connections_dir[isInput ? 0 : 1];
    }
    if (side === "Left") {
        if (node.flags.collapsed) {
            var w = node._collapsed_width || LiteGraph.NODE_COLLAPSED_WIDTH;
            out[0] = node.pos[0];
            out[1] = node.pos[1] - LiteGraph.NODE_TITLE_HEIGHT * 0.5;
        }
        else {
            toggleConnectionLabel(cxn, !isInput || collapseConnections || !!node.hideSlotLabels);
            out[0] = node.pos[0] + offset;
            if ((_b = node.constructor) === null || _b === void 0 ? void 0 : _b.type.includes("Reroute")) {
                out[1] = node.pos[1] + node.size[1] * 0.5;
            }
            else {
                out[1] =
                    node.pos[1] +
                        (displaySlot + 0.7) * LiteGraph.NODE_SLOT_HEIGHT +
                        (node.constructor.slot_start_y || 0);
            }
        }
    }
    else if (side === "Right") {
        if (node.flags.collapsed) {
            var w = node._collapsed_width || LiteGraph.NODE_COLLAPSED_WIDTH;
            out[0] = node.pos[0] + w;
            out[1] = node.pos[1] - LiteGraph.NODE_TITLE_HEIGHT * 0.5;
        }
        else {
            toggleConnectionLabel(cxn, isInput || collapseConnections || !!node.hideSlotLabels);
            out[0] = node.pos[0] + node.size[0] + 1 - offset;
            if ((_c = node.constructor) === null || _c === void 0 ? void 0 : _c.type.includes("Reroute")) {
                out[1] = node.pos[1] + node.size[1] * 0.5;
            }
            else {
                out[1] =
                    node.pos[1] +
                        (displaySlot + 0.7) * LiteGraph.NODE_SLOT_HEIGHT +
                        (node.constructor.slot_start_y || 0);
            }
        }
    }
    else if (side === "Top") {
        if (!cxn.has_old_label) {
            cxn.has_old_label = true;
            cxn.old_label = cxn.label;
            cxn.label = " ";
        }
        out[0] = node.pos[0] + node.size[0] * 0.5;
        out[1] = node.pos[1] + offset;
    }
    else if (side === "Bottom") {
        if (!cxn.has_old_label) {
            cxn.has_old_label = true;
            cxn.old_label = cxn.label;
            cxn.label = " ";
        }
        out[0] = node.pos[0] + node.size[0] * 0.5;
        out[1] = node.pos[1] + node.size[1] - offset;
    }
    return out;
}
function toggleConnectionLabel(cxn, hide = true) {
    if (hide) {
        if (!cxn.has_old_label) {
            cxn.has_old_label = true;
            cxn.old_label = cxn.label;
        }
        cxn.label = " ";
    }
    else if (!hide && cxn.has_old_label) {
        cxn.has_old_label = false;
        cxn.label = cxn.old_label;
        cxn.old_label = undefined;
    }
    return cxn;
}
export function addHelpMenuItem(node, content, menuOptions) {
    addMenuItemOnExtraMenuOptions(node, {
        name: "🛟 Node Help",
        callback: (node) => {
            if (node.showHelp) {
                node.showHelp();
            }
            else {
                new RgthreeHelpDialog(node, content).show();
            }
        },
    }, menuOptions, "Properties Panel");
}
export var PassThroughFollowing;
(function (PassThroughFollowing) {
    PassThroughFollowing[PassThroughFollowing["ALL"] = 0] = "ALL";
    PassThroughFollowing[PassThroughFollowing["NONE"] = 1] = "NONE";
    PassThroughFollowing[PassThroughFollowing["REROUTE_ONLY"] = 2] = "REROUTE_ONLY";
})(PassThroughFollowing || (PassThroughFollowing = {}));
export function shouldPassThrough(node, passThroughFollowing = PassThroughFollowing.ALL) {
    var _a;
    const type = (_a = node === null || node === void 0 ? void 0 : node.constructor) === null || _a === void 0 ? void 0 : _a.type;
    if (!type || passThroughFollowing === PassThroughFollowing.NONE) {
        return false;
    }
    if (passThroughFollowing === PassThroughFollowing.REROUTE_ONLY) {
        return type.includes("Reroute");
    }
    return (type.includes("Reroute") || type.includes("Node Combiner") || type.includes("Node Collector"));
}
function filterOutPassthroughNodes(infos, passThroughFollowing = PassThroughFollowing.ALL) {
    return infos.filter((i) => !shouldPassThrough(i.node, passThroughFollowing));
}
export function getConnectedInputNodes(startNode, currentNode, slot, passThroughFollowing = PassThroughFollowing.ALL) {
    return getConnectedNodesInfo(startNode, IoDirection.INPUT, currentNode, slot, passThroughFollowing).map((n) => n.node);
}
export function getConnectedInputInfosAndFilterPassThroughs(startNode, currentNode, slot, passThroughFollowing = PassThroughFollowing.ALL) {
    return filterOutPassthroughNodes(getConnectedNodesInfo(startNode, IoDirection.INPUT, currentNode, slot, passThroughFollowing), passThroughFollowing);
}
export function getConnectedInputNodesAndFilterPassThroughs(startNode, currentNode, slot, passThroughFollowing = PassThroughFollowing.ALL) {
    return getConnectedInputInfosAndFilterPassThroughs(startNode, currentNode, slot, passThroughFollowing).map((n) => n.node);
}
export function getConnectedOutputNodes(startNode, currentNode, slot, passThroughFollowing = PassThroughFollowing.ALL) {
    return getConnectedNodesInfo(startNode, IoDirection.OUTPUT, currentNode, slot, passThroughFollowing).map((n) => n.node);
}
export function getConnectedOutputNodesAndFilterPassThroughs(startNode, currentNode, slot, passThroughFollowing = PassThroughFollowing.ALL) {
    return filterOutPassthroughNodes(getConnectedNodesInfo(startNode, IoDirection.OUTPUT, currentNode, slot, passThroughFollowing), passThroughFollowing).map((n) => n.node);
}
export function getConnectedNodesInfo(startNode, dir = IoDirection.INPUT, currentNode, slot, passThroughFollowing = PassThroughFollowing.ALL, originTravelFromSlot) {
    var _a, _b, _c, _d, _e, _f, _g, _h;
    currentNode = currentNode || startNode;
    let rootNodes = [];
    if (startNode === currentNode || shouldPassThrough(currentNode, passThroughFollowing)) {
        let linkIds;
        slot = slot != null && slot > -1 ? slot : undefined;
        if (dir == IoDirection.OUTPUT) {
            if (slot != null) {
                linkIds = [...(((_b = (_a = currentNode.outputs) === null || _a === void 0 ? void 0 : _a[slot]) === null || _b === void 0 ? void 0 : _b.links) || [])];
            }
            else {
                linkIds = ((_c = currentNode.outputs) === null || _c === void 0 ? void 0 : _c.flatMap((i) => i.links)) || [];
            }
        }
        else {
            if (slot != null) {
                linkIds = [(_e = (_d = currentNode.inputs) === null || _d === void 0 ? void 0 : _d[slot]) === null || _e === void 0 ? void 0 : _e.link];
            }
            else {
                linkIds = ((_f = currentNode.inputs) === null || _f === void 0 ? void 0 : _f.map((i) => i.link)) || [];
            }
        }
        const graph = (_g = currentNode.graph) !== null && _g !== void 0 ? _g : app.graph;
        for (const linkId of linkIds) {
            let link = null;
            if (typeof linkId == "number") {
                link = (_h = graph.links[linkId]) !== null && _h !== void 0 ? _h : null;
            }
            if (!link) {
                continue;
            }
            const travelFromSlot = dir == IoDirection.OUTPUT ? link.origin_slot : link.target_slot;
            const connectedId = dir == IoDirection.OUTPUT ? link.target_id : link.origin_id;
            const travelToSlot = dir == IoDirection.OUTPUT ? link.target_slot : link.origin_slot;
            originTravelFromSlot = originTravelFromSlot != null ? originTravelFromSlot : travelFromSlot;
            const originNode = graph.getNodeById(connectedId);
            if (!link) {
                console.error("No connected node found... weird");
                continue;
            }
            if (rootNodes.some((n) => n.node == originNode)) {
                console.log(`${startNode.title} (${startNode.id}) seems to have two links to ${originNode.title} (${originNode.id}). One may be stale: ${linkIds.join(", ")}`);
            }
            else {
                rootNodes.push({ node: originNode, travelFromSlot, travelToSlot, originTravelFromSlot });
                if (shouldPassThrough(originNode, passThroughFollowing)) {
                    for (const foundNode of getConnectedNodesInfo(startNode, dir, originNode, undefined, undefined, originTravelFromSlot)) {
                        if (!rootNodes.map((n) => n.node).includes(foundNode.node)) {
                            rootNodes.push(foundNode);
                        }
                    }
                }
            }
        }
    }
    return rootNodes;
}
export function followConnectionUntilType(node, dir, slotNum, skipSelf = false) {
    const slots = dir === IoDirection.OUTPUT ? node.outputs : node.inputs;
    if (!slots || !slots.length) {
        return null;
    }
    let type = null;
    if (slotNum) {
        if (!slots[slotNum]) {
            return null;
        }
        type = getTypeFromSlot(slots[slotNum], dir, skipSelf);
    }
    else {
        for (const slot of slots) {
            type = getTypeFromSlot(slot, dir, skipSelf);
            if (type) {
                break;
            }
        }
    }
    return type;
}
function getTypeFromSlot(slot, dir, skipSelf = false) {
    let graph = app.canvas.getCurrentGraph();
    let type = slot === null || slot === void 0 ? void 0 : slot.type;
    if (!skipSelf && type != null && type != "*") {
        return { type: type, label: slot === null || slot === void 0 ? void 0 : slot.label, name: slot === null || slot === void 0 ? void 0 : slot.name };
    }
    const links = getSlotLinks(slot);
    for (const link of links) {
        const connectedId = dir == IoDirection.OUTPUT ? link.link.target_id : link.link.origin_id;
        const connectedSlotNum = dir == IoDirection.OUTPUT ? link.link.target_slot : link.link.origin_slot;
        const connectedNode = graph.getNodeById(connectedId);
        const connectedSlots = dir === IoDirection.OUTPUT ? connectedNode.inputs : connectedNode.outputs;
        let connectedSlot = connectedSlots[connectedSlotNum];
        if ((connectedSlot === null || connectedSlot === void 0 ? void 0 : connectedSlot.type) != null && (connectedSlot === null || connectedSlot === void 0 ? void 0 : connectedSlot.type) != "*") {
            return {
                type: connectedSlot.type,
                label: connectedSlot === null || connectedSlot === void 0 ? void 0 : connectedSlot.label,
                name: connectedSlot === null || connectedSlot === void 0 ? void 0 : connectedSlot.name,
            };
        }
        else if ((connectedSlot === null || connectedSlot === void 0 ? void 0 : connectedSlot.type) == "*") {
            return followConnectionUntilType(connectedNode, dir);
        }
    }
    return null;
}
export async function replaceNode(existingNode, typeOrNewNode, inputNameMap) {
    const existingCtor = existingNode.constructor;
    const newNode = typeof typeOrNewNode === "string" ? LiteGraph.createNode(typeOrNewNode) : typeOrNewNode;
    if (existingNode.title != existingCtor.title) {
        newNode.title = existingNode.title;
    }
    newNode.pos = [...existingNode.pos];
    newNode.properties = { ...existingNode.properties };
    const oldComputeSize = [...existingNode.computeSize()];
    const oldSize = [
        existingNode.size[0] === oldComputeSize[0] ? null : existingNode.size[0],
        existingNode.size[1] === oldComputeSize[1] ? null : existingNode.size[1],
    ];
    let setSizeIters = 0;
    const setSizeFn = () => {
        const newComputesize = newNode.computeSize();
        newNode.size[0] = Math.max(oldSize[0] || 0, newComputesize[0]);
        newNode.size[1] = Math.max(oldSize[1] || 0, newComputesize[1]);
        setSizeIters++;
        if (setSizeIters > 10) {
            requestAnimationFrame(setSizeFn);
        }
    };
    setSizeFn();
    const links = [];
    const graph = existingNode.graph || app.graph;
    for (const [index, output] of existingNode.outputs.entries()) {
        for (const linkId of output.links || []) {
            const link = graph.links[linkId];
            if (!link)
                continue;
            const targetNode = graph.getNodeById(link.target_id);
            links.push({ node: newNode, slot: output.name, targetNode, targetSlot: link.target_slot });
        }
    }
    for (const [index, input] of existingNode.inputs.entries()) {
        const linkId = input.link;
        if (linkId) {
            const link = graph.links[linkId];
            const originNode = graph.getNodeById(link.origin_id);
            links.push({
                node: originNode,
                slot: link.origin_slot,
                targetNode: newNode,
                targetSlot: (inputNameMap === null || inputNameMap === void 0 ? void 0 : inputNameMap.has(input.name))
                    ? inputNameMap.get(input.name)
                    : input.name || index,
            });
        }
    }
    graph.add(newNode);
    await wait();
    for (const link of links) {
        link.node.connect(link.slot, link.targetNode, link.targetSlot);
    }
    await wait();
    graph.remove(existingNode);
    newNode.size = newNode.computeSize();
    newNode.setDirtyCanvas(true, true);
    return newNode;
}
export function getOriginNodeByLink(linkId) {
    let node = null;
    if (linkId != null) {
        const link = getLinkById(linkId);
        node = (link != null && getNodeById(link.origin_id)) || null;
    }
    return node;
}
export function getLinkById(linkId) {
    var _a, _b, _c, _d;
    if (linkId == null)
        return null;
    let link = (_a = app.graph.links[linkId]) !== null && _a !== void 0 ? _a : null;
    link = (_c = link !== null && link !== void 0 ? link : (_b = app.canvas.getCurrentGraph()) === null || _b === void 0 ? void 0 : _b.links[linkId]) !== null && _c !== void 0 ? _c : null;
    const subgraphs = app.graph.rootGraph.subgraphs.values();
    let subgraph;
    while (!link && (subgraph = subgraphs.next().value)) {
        link = (_d = subgraph === null || subgraph === void 0 ? void 0 : subgraph.links[linkId]) !== null && _d !== void 0 ? _d : null;
    }
    return link;
}
export function getNodeById(id) {
    var _a, _b, _c;
    if (id == null)
        return null;
    let node = app.graph.getNodeById(id);
    node = (_b = node !== null && node !== void 0 ? node : (_a = app.canvas.getCurrentGraph()) === null || _a === void 0 ? void 0 : _a.getNodeById(id)) !== null && _b !== void 0 ? _b : null;
    const subgraphs = app.graph.rootGraph.subgraphs.values();
    let subgraph;
    while (!node && (subgraph = subgraphs.next().value)) {
        node = (_c = subgraph === null || subgraph === void 0 ? void 0 : subgraph.getNodeById(id)) !== null && _c !== void 0 ? _c : null;
    }
    return node;
}
export function applyMixins(original, constructors) {
    constructors.forEach((baseCtor) => {
        Object.getOwnPropertyNames(baseCtor.prototype).forEach((name) => {
            Object.defineProperty(original.prototype, name, Object.getOwnPropertyDescriptor(baseCtor.prototype, name) || Object.create(null));
        });
    });
}
export function getSlotLinks(inputOrOutput) {
    var _a;
    const links = [];
    if (!inputOrOutput) {
        return links;
    }
    if ((_a = inputOrOutput.links) === null || _a === void 0 ? void 0 : _a.length) {
        const output = inputOrOutput;
        for (const linkId of output.links || []) {
            const link = app.graph.links[linkId];
            if (link) {
                links.push({ id: linkId, link: link });
            }
        }
    }
    if (inputOrOutput.link) {
        const input = inputOrOutput;
        const link = app.graph.links[input.link];
        if (link) {
            links.push({ id: input.link, link: link });
        }
    }
    return links;
}
export async function matchLocalSlotsToServer(node, direction, serverNodeData) {
    var _a, _b, _c;
    const serverSlotNames = direction == IoDirection.INPUT
        ? Object.keys(((_a = serverNodeData.input) === null || _a === void 0 ? void 0 : _a.optional) || {})
        : serverNodeData.output_name;
    const serverSlotTypes = direction == IoDirection.INPUT
        ? Object.values(((_b = serverNodeData.input) === null || _b === void 0 ? void 0 : _b.optional) || {}).map((i) => i[0])
        : serverNodeData.output;
    const slots = direction == IoDirection.INPUT ? node.inputs : node.outputs;
    let firstIndex = slots.findIndex((o, i) => i !== serverSlotNames.indexOf(o.name));
    if (firstIndex > -1) {
        const links = {};
        slots.map((slot) => {
            var _a;
            links[slot.name] = links[slot.name] || [];
            (_a = links[slot.name]) === null || _a === void 0 ? void 0 : _a.push(...getSlotLinks(slot));
        });
        for (const [index, serverSlotName] of serverSlotNames.entries()) {
            const currentNodeSlot = slots.map((s) => s.name).indexOf(serverSlotName);
            if (currentNodeSlot > -1) {
                if (currentNodeSlot != index) {
                    const splicedItem = slots.splice(currentNodeSlot, 1)[0];
                    slots.splice(index, 0, splicedItem);
                }
            }
            else if (currentNodeSlot === -1) {
                const splicedItem = {
                    name: serverSlotName,
                    type: serverSlotTypes[index],
                    links: [],
                };
                slots.splice(index, 0, splicedItem);
            }
        }
        if (slots.length > serverSlotNames.length) {
            for (let i = slots.length - 1; i > serverSlotNames.length - 1; i--) {
                if (direction == IoDirection.INPUT) {
                    node.disconnectInput(i);
                    node.removeInput(i);
                }
                else {
                    node.disconnectOutput(i);
                    node.removeOutput(i);
                }
            }
        }
        for (const [name, slotLinks] of Object.entries(links)) {
            let currentNodeSlot = slots.map((s) => s.name).indexOf(name);
            if (currentNodeSlot > -1) {
                for (const linkData of slotLinks) {
                    if (direction == IoDirection.INPUT) {
                        linkData.link.target_slot = currentNodeSlot;
                    }
                    else {
                        linkData.link.origin_slot = currentNodeSlot;
                        const nextNode = app.graph.getNodeById(linkData.link.target_id);
                        if (nextNode && ((_c = nextNode.constructor) === null || _c === void 0 ? void 0 : _c.type.includes("Reroute"))) {
                            nextNode.stabilize && nextNode.stabilize();
                        }
                    }
                }
            }
        }
    }
}
export function isValidConnection(ioA, ioB) {
    if (!ioA || !ioB) {
        return false;
    }
    const typeA = String(ioA.type);
    const typeB = String(ioB.type);
    let isValid = LiteGraph.isValidConnection(typeA, typeB);
    if (!isValid) {
        let areCombos = (typeA.includes(",") && typeB === "COMBO") || (typeA === "COMBO" && typeB.includes(","));
        if (areCombos) {
            const nameA = ioA.name.toUpperCase().replace("_NAME", "").replace("CKPT", "MODEL");
            const nameB = ioB.name.toUpperCase().replace("_NAME", "").replace("CKPT", "MODEL");
            isValid = nameA.includes(nameB) || nameB.includes(nameA);
        }
    }
    return isValid;
}
const oldIsValidConnection = LiteGraph.isValidConnection;
LiteGraph.isValidConnection = function (typeA, typeB) {
    let isValid = oldIsValidConnection.call(LiteGraph, typeA, typeB);
    if (!isValid) {
        typeA = String(typeA);
        typeB = String(typeB);
        let areCombos = (typeA.includes(",") && typeB === "COMBO") || (typeA === "COMBO" && typeB.includes(","));
        isValid = areCombos;
    }
    return isValid;
};
export function getOutputNodes(nodes) {
    return ((nodes === null || nodes === void 0 ? void 0 : nodes.filter((n) => {
        var _a;
        return (n.mode != LiteGraph.NEVER && ((_a = n.constructor.nodeData) === null || _a === void 0 ? void 0 : _a.output_node));
    })) || []);
}
export function changeModeOfNodes(nodeOrNodes, mode) {
    reduceNodesDepthFirst(nodeOrNodes, (n) => {
        n.mode = mode;
    });
}
export function reduceNodesDepthFirst(nodeOrNodes, reduceFn, reduceTo) {
    var _a;
    const nodes = Array.isArray(nodeOrNodes) ? nodeOrNodes : [nodeOrNodes];
    const stack = nodes.map((node) => ({ node }));
    while (stack.length > 0) {
        const { node } = stack.pop();
        const result = reduceFn(node, reduceTo);
        if (result !== undefined && result !== reduceTo) {
            reduceTo = result;
        }
        if (((_a = node.isSubgraphNode) === null || _a === void 0 ? void 0 : _a.call(node)) && node.subgraph) {
            const children = node.subgraph.nodes;
            for (let i = children.length - 1; i >= 0; i--) {
                stack.push({ node: children[i] });
            }
        }
    }
    return reduceTo;
}
export function getGroupNodes(group) {
    return Array.from(group._children).filter((c) => c instanceof LGraphNode);
}
export function getFullColor(color, liteGraphKey = "color") {
    if (!color) {
        return "";
    }
    if (LGraphCanvas.node_colors[color]) {
        color = LGraphCanvas.node_colors[color][liteGraphKey];
    }
    color = color.replace("#", "").toLocaleLowerCase();
    if (color.length === 3) {
        color = color.replace(/(.)(.)(.)/, "$1$1$2$2$3$3");
    }
    return `#${color}`;
}
