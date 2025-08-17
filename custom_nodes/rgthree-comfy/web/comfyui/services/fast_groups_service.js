import { app } from "../../../scripts/app.js";
import { getGroupNodes, reduceNodesDepthFirst } from "../utils.js";
class FastGroupsService {
    constructor() {
        this.msThreshold = 400;
        this.msLastUnsorted = 0;
        this.msLastAlpha = 0;
        this.msLastPosition = 0;
        this.groupsUnsorted = [];
        this.groupsSortedAlpha = [];
        this.groupsSortedPosition = [];
        this.fastGroupNodes = [];
        this.runScheduledForMs = null;
        this.runScheduleTimeout = null;
        this.runScheduleAnimation = null;
        this.cachedNodeBoundings = null;
    }
    addFastGroupNode(node) {
        this.fastGroupNodes.push(node);
        this.scheduleRun(8);
    }
    removeFastGroupNode(node) {
        var _a;
        const index = this.fastGroupNodes.indexOf(node);
        if (index > -1) {
            this.fastGroupNodes.splice(index, 1);
        }
        if (!((_a = this.fastGroupNodes) === null || _a === void 0 ? void 0 : _a.length)) {
            this.clearScheduledRun();
            this.groupsUnsorted = [];
            this.groupsSortedAlpha = [];
            this.groupsSortedPosition = [];
        }
    }
    run() {
        if (!this.runScheduledForMs) {
            return;
        }
        for (const node of this.fastGroupNodes) {
            node.refreshWidgets();
        }
        this.clearScheduledRun();
        this.scheduleRun();
    }
    scheduleRun(ms = 500) {
        if (this.runScheduledForMs && ms < this.runScheduledForMs) {
            this.clearScheduledRun();
        }
        if (!this.runScheduledForMs && this.fastGroupNodes.length) {
            this.runScheduledForMs = ms;
            this.runScheduleTimeout = setTimeout(() => {
                this.runScheduleAnimation = requestAnimationFrame(() => this.run());
            }, ms);
        }
    }
    clearScheduledRun() {
        this.runScheduleTimeout && clearTimeout(this.runScheduleTimeout);
        this.runScheduleAnimation && cancelAnimationFrame(this.runScheduleAnimation);
        this.runScheduleTimeout = null;
        this.runScheduleAnimation = null;
        this.runScheduledForMs = null;
    }
    getBoundingsForAllNodes() {
        if (!this.cachedNodeBoundings) {
            this.cachedNodeBoundings = reduceNodesDepthFirst(app.graph._nodes, (node, acc) => {
                var _a, _b;
                let bounds = node.getBounding();
                if (bounds[0] === 0 && bounds[1] === 0 && bounds[2] === 0 && bounds[3] === 0) {
                    const ctx = (_b = (_a = node.graph) === null || _a === void 0 ? void 0 : _a.primaryCanvas) === null || _b === void 0 ? void 0 : _b.canvas.getContext('2d');
                    if (ctx) {
                        node.updateArea(ctx);
                        bounds = node.getBounding();
                    }
                }
                acc[String(node.id)] = bounds;
            }, {});
            setTimeout(() => {
                this.cachedNodeBoundings = null;
            }, 50);
        }
        return this.cachedNodeBoundings;
    }
    recomputeInsideNodesForGroup(group) {
        const cachedBoundings = this.getBoundingsForAllNodes();
        const nodes = group.graph.nodes;
        group._children.clear();
        group.nodes.length = 0;
        for (const node of nodes) {
            const nodeBounding = cachedBoundings[String(node.id)];
            const nodeCenter = nodeBounding &&
                [nodeBounding[0] + nodeBounding[2] * 0.5, nodeBounding[1] + nodeBounding[3] * 0.5];
            if (nodeCenter) {
                const grouBounds = group._bounding;
                if (nodeCenter[0] >= grouBounds[0] &&
                    nodeCenter[0] < grouBounds[0] + grouBounds[2] &&
                    nodeCenter[1] >= grouBounds[1] &&
                    nodeCenter[1] < grouBounds[1] + grouBounds[3]) {
                    group._children.add(node);
                    group.nodes.push(node);
                }
            }
        }
    }
    getGroupsUnsorted(now) {
        var _a, _b;
        const canvas = app.canvas;
        const graph = (_a = canvas.getCurrentGraph()) !== null && _a !== void 0 ? _a : app.graph;
        if (!canvas.selected_group_moving &&
            (!this.groupsUnsorted.length || now - this.msLastUnsorted > this.msThreshold)) {
            this.groupsUnsorted = [...graph._groups];
            const subgraphs = graph.subgraphs.values();
            let s;
            while ((s = subgraphs.next().value))
                this.groupsUnsorted.push(...((_b = s.groups) !== null && _b !== void 0 ? _b : []));
            for (const group of this.groupsUnsorted) {
                this.recomputeInsideNodesForGroup(group);
                group.rgthree_hasAnyActiveNode = getGroupNodes(group).some((n) => n.mode === LiteGraph.ALWAYS);
            }
            this.msLastUnsorted = now;
        }
        return this.groupsUnsorted;
    }
    getGroupsAlpha(now) {
        if (!this.groupsSortedAlpha.length || now - this.msLastAlpha > this.msThreshold) {
            this.groupsSortedAlpha = [...this.getGroupsUnsorted(now)].sort((a, b) => {
                return a.title.localeCompare(b.title);
            });
            this.msLastAlpha = now;
        }
        return this.groupsSortedAlpha;
    }
    getGroupsPosition(now) {
        if (!this.groupsSortedPosition.length || now - this.msLastPosition > this.msThreshold) {
            this.groupsSortedPosition = [...this.getGroupsUnsorted(now)].sort((a, b) => {
                const aY = Math.floor(a._pos[1] / 30);
                const bY = Math.floor(b._pos[1] / 30);
                if (aY == bY) {
                    const aX = Math.floor(a._pos[0] / 30);
                    const bX = Math.floor(b._pos[0] / 30);
                    return aX - bX;
                }
                return aY - bY;
            });
            this.msLastPosition = now;
        }
        return this.groupsSortedPosition;
    }
    getGroups(sort) {
        const now = +new Date();
        if (sort === "alphanumeric") {
            return this.getGroupsAlpha(now);
        }
        if (sort === "position") {
            return this.getGroupsPosition(now);
        }
        return this.getGroupsUnsorted(now);
    }
}
export const SERVICE = new FastGroupsService();
