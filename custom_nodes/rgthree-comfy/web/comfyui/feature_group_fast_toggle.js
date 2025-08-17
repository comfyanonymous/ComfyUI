import { app } from "../../scripts/app.js";
import { rgthree } from "./rgthree.js";
import { changeModeOfNodes, getGroupNodes, getOutputNodes } from "./utils.js";
import { SERVICE as CONFIG_SERVICE } from "./services/config_service.js";
const BTN_SIZE = 20;
const BTN_MARGIN = [6, 6];
const BTN_SPACING = 8;
const BTN_GRID = BTN_SIZE / 8;
const TOGGLE_TO_MODE = new Map([
    ["MUTE", LiteGraph.NEVER],
    ["BYPASS", 4],
]);
function getToggles() {
    return [...CONFIG_SERVICE.getFeatureValue("group_header_fast_toggle.toggles", [])].reverse();
}
function clickedOnToggleButton(e, group) {
    const toggles = getToggles();
    const pos = group.pos;
    const size = group.size;
    for (let i = 0; i < toggles.length; i++) {
        const toggle = toggles[i];
        if (LiteGraph.isInsideRectangle(e.canvasX, e.canvasY, pos[0] + size[0] - (BTN_SIZE + BTN_MARGIN[0]) * (i + 1), pos[1] + BTN_MARGIN[1], BTN_SIZE, BTN_SIZE)) {
            return toggle;
        }
    }
    return null;
}
app.registerExtension({
    name: "rgthree.GroupHeaderToggles",
    async setup() {
        setInterval(() => {
            if (CONFIG_SERVICE.getFeatureValue("group_header_fast_toggle.enabled") &&
                CONFIG_SERVICE.getFeatureValue("group_header_fast_toggle.show") !== "always") {
                app.canvas.setDirty(true, true);
            }
        }, 250);
        rgthree.addEventListener("on-process-mouse-down", ((e) => {
            if (!CONFIG_SERVICE.getFeatureValue("group_header_fast_toggle.enabled"))
                return;
            const canvas = app.canvas;
            if (canvas.selected_group) {
                const originalEvent = e.detail.originalEvent;
                const group = canvas.selected_group;
                const clickedOnToggle = clickedOnToggleButton(originalEvent, group) || "";
                const toggleAction = clickedOnToggle === null || clickedOnToggle === void 0 ? void 0 : clickedOnToggle.toLocaleUpperCase();
                if (toggleAction) {
                    console.log(toggleAction);
                    const nodes = getGroupNodes(group);
                    if (toggleAction === "QUEUE") {
                        const outputNodes = getOutputNodes(nodes);
                        if (!(outputNodes === null || outputNodes === void 0 ? void 0 : outputNodes.length)) {
                            rgthree.showMessage({
                                id: "no-output-in-group",
                                type: "warn",
                                timeout: 4000,
                                message: "No output nodes for group!",
                            });
                        }
                        else {
                            rgthree.queueOutputNodes(outputNodes.map((n) => n.id));
                        }
                    }
                    else {
                        const toggleMode = TOGGLE_TO_MODE.get(toggleAction);
                        if (toggleMode) {
                            group.recomputeInsideNodes();
                            const hasAnyActiveNodes = nodes.some((n) => n.mode === LiteGraph.ALWAYS);
                            const isAllMuted = !hasAnyActiveNodes && nodes.every((n) => n.mode === LiteGraph.NEVER);
                            const isAllBypassed = !hasAnyActiveNodes && !isAllMuted && nodes.every((n) => n.mode === 4);
                            let newMode = LiteGraph.ALWAYS;
                            if (toggleMode === LiteGraph.NEVER) {
                                newMode = isAllMuted ? LiteGraph.ALWAYS : LiteGraph.NEVER;
                            }
                            else {
                                newMode = isAllBypassed ? LiteGraph.ALWAYS : 4;
                            }
                            changeModeOfNodes(nodes, newMode);
                        }
                    }
                    canvas.selected_group = null;
                    canvas.dragging_canvas = false;
                }
            }
        }));
        const drawGroups = LGraphCanvas.prototype.drawGroups;
        LGraphCanvas.prototype.drawGroups = function (canvasEl, ctx) {
            drawGroups.apply(this, [...arguments]);
            if (!CONFIG_SERVICE.getFeatureValue("group_header_fast_toggle.enabled") ||
                !rgthree.lastCanvasMouseEvent) {
                return;
            }
            const graph = app.canvas.graph;
            let groups;
            if (CONFIG_SERVICE.getFeatureValue("group_header_fast_toggle.show") !== "always") {
                const hoverGroup = graph.getGroupOnPos(rgthree.lastCanvasMouseEvent.canvasX, rgthree.lastCanvasMouseEvent.canvasY);
                groups = hoverGroup ? [hoverGroup] : [];
            }
            else {
                groups = graph._groups || [];
            }
            if (!groups.length) {
                return;
            }
            const toggles = getToggles();
            ctx.save();
            for (const group of groups || []) {
                const nodes = getGroupNodes(group);
                let anyActive = false;
                let allMuted = !!nodes.length;
                let allBypassed = allMuted;
                for (const node of nodes) {
                    if (!(node instanceof LGraphNode))
                        continue;
                    anyActive = anyActive || node.mode === LiteGraph.ALWAYS;
                    allMuted = allMuted && node.mode === LiteGraph.NEVER;
                    allBypassed = allBypassed && node.mode === 4;
                    if (anyActive || (!allMuted && !allBypassed)) {
                        break;
                    }
                }
                for (let i = 0; i < toggles.length; i++) {
                    const toggle = toggles[i];
                    const pos = group._pos;
                    const size = group._size;
                    ctx.fillStyle = ctx.strokeStyle = group.color || "#335";
                    const x = pos[0] + size[0] - BTN_MARGIN[0] - BTN_SIZE - (BTN_SPACING + BTN_SIZE) * i;
                    const y = pos[1] + BTN_MARGIN[1];
                    const midX = x + BTN_SIZE / 2;
                    const midY = y + BTN_SIZE / 2;
                    if (toggle === "queue") {
                        const outputNodes = getOutputNodes(nodes);
                        const oldGlobalAlpha = ctx.globalAlpha;
                        if (!(outputNodes === null || outputNodes === void 0 ? void 0 : outputNodes.length)) {
                            ctx.globalAlpha = 0.5;
                        }
                        ctx.lineJoin = "round";
                        ctx.lineCap = "round";
                        const arrowSizeX = BTN_SIZE * 0.6;
                        const arrowSizeY = BTN_SIZE * 0.7;
                        const arrow = new Path2D(`M ${x + arrowSizeX / 2} ${midY} l 0 -${arrowSizeY / 2} l ${arrowSizeX} ${arrowSizeY / 2} l -${arrowSizeX} ${arrowSizeY / 2} z`);
                        ctx.stroke(arrow);
                        if (outputNodes === null || outputNodes === void 0 ? void 0 : outputNodes.length) {
                            ctx.fill(arrow);
                        }
                        ctx.globalAlpha = oldGlobalAlpha;
                    }
                    else {
                        const on = toggle === "bypass" ? allBypassed : allMuted;
                        ctx.beginPath();
                        ctx.lineJoin = "round";
                        ctx.rect(x, y, BTN_SIZE, BTN_SIZE);
                        ctx.lineWidth = 2;
                        if (toggle === "mute") {
                            ctx.lineJoin = "round";
                            ctx.lineCap = "round";
                            if (on) {
                                ctx.stroke(new Path2D(`
                    ${eyeFrame(midX, midY)}
                    ${eyeLashes(midX, midY)}
                `));
                            }
                            else {
                                const radius = BTN_GRID * 1.5;
                                ctx.fill(new Path2D(`
                    ${eyeFrame(midX, midY)}
                    ${eyeFrame(midX, midY, -1)}
                    ${circlePath(midX, midY, radius)}
                    ${circlePath(midX + BTN_GRID / 2, midY - BTN_GRID / 2, BTN_GRID * 0.375)}
                  `), "evenodd");
                                ctx.stroke(new Path2D(`${eyeFrame(midX, midY)} ${eyeFrame(midX, midY, -1)}`));
                                ctx.globalAlpha = this.editor_alpha * 0.5;
                                ctx.stroke(new Path2D(`${eyeLashes(midX, midY)} ${eyeLashes(midX, midY, -1)}`));
                                ctx.globalAlpha = this.editor_alpha;
                            }
                        }
                        else {
                            const lineChanges = on
                                ? `a ${BTN_GRID * 3}, ${BTN_GRID * 3} 0 1, 1 ${BTN_GRID * 3 * 2},0
                  l ${BTN_GRID * 2.0} 0`
                                : `l ${BTN_GRID * 8} 0`;
                            ctx.stroke(new Path2D(`
                  M ${x} ${midY}
                  ${lineChanges}
                  M ${x + BTN_SIZE} ${midY} l -2  2
                  M ${x + BTN_SIZE} ${midY} l -2 -2
                `));
                            ctx.fill(new Path2D(`${circlePath(x + BTN_GRID * 3, midY, BTN_GRID * 1.8)}`));
                        }
                    }
                }
            }
            ctx.restore();
        };
    },
});
function eyeFrame(midX, midY, yFlip = 1) {
    return `
      M ${midX - BTN_SIZE / 2} ${midY}
      c ${BTN_GRID * 1.5} ${yFlip * BTN_GRID * 2.5}, ${BTN_GRID * (8 - 1.5)} ${yFlip * BTN_GRID * 2.5}, ${BTN_GRID * 8} 0
  `;
}
function eyeLashes(midX, midY, yFlip = 1) {
    return `
    M ${midX - BTN_GRID * 3.46} ${midY + yFlip * BTN_GRID * 0.9} l -1.15  ${1.25 * yFlip}
    M ${midX - BTN_GRID * 2.38} ${midY + yFlip * BTN_GRID * 1.6} l -0.90  ${1.5 * yFlip}
    M ${midX - BTN_GRID * 1.15} ${midY + yFlip * BTN_GRID * 1.95} l -0.50  ${1.75 * yFlip}
    M ${midX + BTN_GRID * 0.0} ${midY + yFlip * BTN_GRID * 2.0} l  0.00  ${2.0 * yFlip}
    M ${midX + BTN_GRID * 1.15} ${midY + yFlip * BTN_GRID * 1.95} l  0.50  ${1.75 * yFlip}
    M ${midX + BTN_GRID * 2.38} ${midY + yFlip * BTN_GRID * 1.6} l  0.90  ${1.5 * yFlip}
    M ${midX + BTN_GRID * 3.46} ${midY + yFlip * BTN_GRID * 0.9} l  1.15  ${1.25 * yFlip}
`;
}
function circlePath(cx, cy, radius) {
    return `
      M ${cx} ${cy}
      m ${radius}, 0
      a ${radius},${radius} 0 1, 1 -${radius * 2},0
      a ${radius},${radius} 0 1, 1  ${radius * 2},0
  `;
}
