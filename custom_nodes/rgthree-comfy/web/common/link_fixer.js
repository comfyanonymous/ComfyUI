var IoDirection;
(function (IoDirection) {
    IoDirection[IoDirection["INPUT"] = 0] = "INPUT";
    IoDirection[IoDirection["OUTPUT"] = 1] = "OUTPUT";
})(IoDirection || (IoDirection = {}));
function getLinksData(links) {
    if (links instanceof Map) {
        const data = [];
        for (const [key, llink] of links.entries()) {
            if (!llink)
                continue;
            data.push(llink);
        }
        return data;
    }
    if (!Array.isArray(links)) {
        const data = [];
        for (const key in links) {
            const llink = (links.hasOwnProperty(key) && links[key]) || null;
            if (!llink)
                continue;
            data.push(llink);
        }
        return data;
    }
    return links.map((link) => ({
        id: link[0],
        origin_id: link[1],
        origin_slot: link[2],
        target_id: link[3],
        target_slot: link[4],
        type: link[5],
    }));
}
export class WorkflowLinkFixer {
    static create(graph) {
        if (typeof graph.getNodeById === "function") {
            return new WorkflowLinkFixerGraph(graph);
        }
        return new WorkflowLinkFixerSerialized(graph);
    }
    constructor(graph) {
        this.silent = false;
        this.checkedData = null;
        this.logger = console;
        this.patchedNodeSlots = {};
        this.instructions = [];
        this.graph = graph;
    }
    check(force = false) {
        var _a, _b;
        if (this.checkedData && !force) {
            return { ...this.checkedData };
        }
        this.instructions = [];
        this.patchedNodeSlots = {};
        const instructions = [];
        const links = getLinksData(this.graph.links);
        links.reverse();
        for (const link of links) {
            if (!link)
                continue;
            const originNode = this.getNodeById(link.origin_id);
            const originHasLink = () => this.nodeHasLinkId(originNode, IoDirection.OUTPUT, link.origin_slot, link.id);
            const patchOrigin = (op, id = link.id) => this.getNodePatchInstruction(originNode, IoDirection.OUTPUT, link.origin_slot, id, op);
            const targetNode = this.getNodeById(link.target_id);
            const targetHasLink = () => this.nodeHasLinkId(targetNode, IoDirection.INPUT, link.target_slot, link.id);
            const targetHasAnyLink = () => this.nodeHasAnyLink(targetNode, IoDirection.INPUT, link.target_slot);
            const patchTarget = (op, id = link.id) => this.getNodePatchInstruction(targetNode, IoDirection.INPUT, link.target_slot, id, op);
            const originLog = `origin(${link.origin_id}).outputs[${link.origin_slot}].links`;
            const targetLog = `target(${link.target_id}).inputs[${link.target_slot}].link`;
            if (!originNode || !targetNode) {
                if (!originNode && !targetNode) {
                }
                else if (!originNode && targetNode) {
                    this.log(`Link ${link.id} is funky... ` +
                        `origin ${link.origin_id} does not exist, but target ${link.target_id} does.`);
                    if (targetHasLink()) {
                        this.log(` > [PATCH] ${targetLog} does have link, will remove the inputs' link first.`);
                        instructions.push(patchTarget("REMOVE", -1));
                    }
                }
                else if (!targetNode && originNode) {
                    this.log(`Link ${link.id} is funky... ` +
                        `target ${link.target_id} does not exist, but origin ${link.origin_id} does.`);
                    if (originHasLink()) {
                        this.log(` > [PATCH] Origin's links' has ${link.id}; will remove the link first.`);
                        instructions.push(patchOrigin("REMOVE"));
                    }
                }
                continue;
            }
            if (targetHasLink() || originHasLink()) {
                if (!originHasLink()) {
                    this.log(`${link.id} is funky... ${originLog} does NOT contain it, but ${targetLog} does.`);
                    this.log(` > [PATCH] Attempt a fix by adding this ${link.id} to ${originLog}.`);
                    instructions.push(patchOrigin("ADD"));
                }
                else if (!targetHasLink()) {
                    this.log(`${link.id} is funky... ${targetLog} is NOT correct (is ${(_b = (_a = targetNode.inputs) === null || _a === void 0 ? void 0 : _a[link.target_slot]) === null || _b === void 0 ? void 0 : _b.link}), but ${originLog} contains it`);
                    if (!targetHasAnyLink()) {
                        this.log(` > [PATCH] ${targetLog} is not defined, will set to ${link.id}.`);
                        let instruction = patchTarget("ADD");
                        if (!instruction) {
                            this.log(` > [PATCH] Nvm, ${targetLog} already patched. Removing ${link.id} from ${originLog}.`);
                            instruction = patchOrigin("REMOVE");
                        }
                        instructions.push(instruction);
                    }
                    else {
                        this.log(` > [PATCH] ${targetLog} is defined, removing ${link.id} from ${originLog}.`);
                        instructions.push(patchOrigin("REMOVE"));
                    }
                }
            }
        }
        for (let link of links) {
            if (!link)
                continue;
            const originNode = this.getNodeById(link.origin_id);
            const targetNode = this.getNodeById(link.target_id);
            if (!originNode && !targetNode) {
                instructions.push({
                    op: "DELETE",
                    linkId: link.id,
                    reason: `Both nodes #${link.origin_id} & #${link.target_id} are removed`,
                });
            }
            if ((!originNode ||
                !this.nodeHasLinkId(originNode, IoDirection.OUTPUT, link.origin_slot, link.id)) &&
                (!targetNode ||
                    !this.nodeHasLinkId(targetNode, IoDirection.INPUT, link.target_slot, link.id))) {
                instructions.push({
                    op: "DELETE",
                    linkId: link.id,
                    reason: `both origin node #${link.origin_id} ` +
                        `${!originNode ? "is removed" : `is missing link id output slot ${link.origin_slot}`}` +
                        `and target node #${link.target_id} ` +
                        `${!targetNode ? "is removed" : `is missing link id input slot ${link.target_slot}`}.`,
                });
                continue;
            }
        }
        this.instructions = instructions.filter((i) => !!i);
        this.checkedData = {
            hasBadLinks: !!this.instructions.length,
            graph: this.graph,
            patches: this.instructions.filter((i) => !!i.node)
                .length,
            deletes: this.instructions.filter((i) => i.op === "DELETE").length,
        };
        return { ...this.checkedData };
    }
    fix(force = false, times) {
        var _a, _b, _c, _d, _e, _f, _g;
        if (!this.checkedData || force) {
            this.check(force);
        }
        let patches = 0;
        let deletes = 0;
        for (const instruction of this.instructions) {
            if (instruction.node) {
                let { node, slot, linkIdToUse, dir, op } = instruction;
                if (dir == IoDirection.INPUT) {
                    node.inputs = node.inputs || [];
                    const old = (_a = node.inputs[slot]) === null || _a === void 0 ? void 0 : _a.link;
                    node.inputs[slot] = node.inputs[slot] || {};
                    node.inputs[slot].link = linkIdToUse;
                    this.log(`Node #${node.id}: Set link ${linkIdToUse} to input slot ${slot} (was ${old})`);
                }
                else if (op === "ADD" && linkIdToUse != null) {
                    node.outputs = node.outputs || [];
                    node.outputs[slot] = node.outputs[slot] || {};
                    node.outputs[slot].links = node.outputs[slot].links || [];
                    node.outputs[slot].links.push(linkIdToUse);
                    this.log(`Node #${node.id}: Add link ${linkIdToUse} to output slot #${slot}`);
                }
                else if (op === "REMOVE" && linkIdToUse != null) {
                    if (((_d = (_c = (_b = node.outputs) === null || _b === void 0 ? void 0 : _b[slot]) === null || _c === void 0 ? void 0 : _c.links) === null || _d === void 0 ? void 0 : _d.length) === undefined) {
                        this.log(`Node #${node.id}: Couldn't remove link ${linkIdToUse} from output slot #${slot}` +
                            ` because it didn't exist.`);
                    }
                    else {
                        let linkIdIndex = node.outputs[slot].links.indexOf(linkIdToUse);
                        node.outputs[slot].links.splice(linkIdIndex, 1);
                        this.log(`Node #${node.id}: Remove link ${linkIdToUse} from output slot #${slot}`);
                    }
                }
                else {
                    throw new Error("Unhandled Node Instruction");
                }
                patches++;
            }
            else if (instruction.op === "DELETE") {
                const wasDeleted = this.deleteGraphLink(instruction.linkId);
                if (wasDeleted === true) {
                    this.log(`Link #${instruction.linkId}: Removed workflow link b/c ${instruction.reason}`);
                }
                else {
                    this.log(`Error Link #${instruction.linkId} was not removed!`);
                }
                deletes += wasDeleted ? 1 : 0;
            }
            else {
                throw new Error("Unhandled Instruction");
            }
        }
        const newCheck = this.check(force);
        times = times == null ? 5 : times;
        let newFix = null;
        if (newCheck.hasBadLinks && times > 0) {
            newFix = this.fix(true, times - 1);
        }
        return {
            hasBadLinks: (_e = newFix === null || newFix === void 0 ? void 0 : newFix.hasBadLinks) !== null && _e !== void 0 ? _e : newCheck.hasBadLinks,
            graph: this.graph,
            patches: patches + ((_f = newFix === null || newFix === void 0 ? void 0 : newFix.patches) !== null && _f !== void 0 ? _f : 0),
            deletes: deletes + ((_g = newFix === null || newFix === void 0 ? void 0 : newFix.deletes) !== null && _g !== void 0 ? _g : 0),
        };
    }
    log(...args) {
        if (this.silent)
            return;
        this.logger.log(...args);
    }
    getNodePatchInstruction(node, ioDir, slot, linkId, op) {
        var _a, _b;
        const nodeId = node.id;
        this.patchedNodeSlots[nodeId] = this.patchedNodeSlots[nodeId] || {};
        const patchedNode = this.patchedNodeSlots[nodeId];
        if (ioDir == IoDirection.INPUT) {
            patchedNode["inputs"] = patchedNode["inputs"] || {};
            if (patchedNode["inputs"][slot] !== undefined) {
                this.log(` > Already set ${nodeId}.inputs[${slot}] to ${patchedNode["inputs"][slot]} Skipping.`);
                return null;
            }
            let linkIdToUse = op === "REMOVE" ? null : linkId;
            patchedNode["inputs"][slot] = linkIdToUse;
            return { node, dir: ioDir, op, slot, linkId, linkIdToUse };
        }
        patchedNode["outputs"] = patchedNode["outputs"] || {};
        patchedNode["outputs"][slot] = patchedNode["outputs"][slot] || {
            links: [...(((_b = (_a = node.outputs) === null || _a === void 0 ? void 0 : _a[slot]) === null || _b === void 0 ? void 0 : _b.links) || [])],
            changes: {},
        };
        if (patchedNode["outputs"][slot]["changes"][linkId] !== undefined) {
            this.log(` > Already set ${nodeId}.outputs[${slot}] to ${patchedNode["outputs"][slot]}! Skipping.`);
            return null;
        }
        patchedNode["outputs"][slot]["changes"][linkId] = op;
        if (op === "ADD") {
            let linkIdIndex = patchedNode["outputs"][slot]["links"].indexOf(linkId);
            if (linkIdIndex !== -1) {
                this.log(` > Hmmm.. asked to add ${linkId} but it is already in list...`);
                return null;
            }
            patchedNode["outputs"][slot]["links"].push(linkId);
            return { node, dir: ioDir, op, slot, linkId, linkIdToUse: linkId };
        }
        let linkIdIndex = patchedNode["outputs"][slot]["links"].indexOf(linkId);
        if (linkIdIndex === -1) {
            this.log(` > Hmmm.. asked to remove ${linkId} but it doesn't exist...`);
            return null;
        }
        patchedNode["outputs"][slot]["links"].splice(linkIdIndex, 1);
        return { node, dir: ioDir, op, slot, linkId, linkIdToUse: linkId };
    }
    nodeHasLinkId(node, ioDir, slot, linkId) {
        var _a, _b, _c, _d, _e, _f, _g, _h, _j;
        const nodeId = node.id;
        let has = false;
        if (ioDir === IoDirection.INPUT) {
            let nodeHasIt = ((_b = (_a = node.inputs) === null || _a === void 0 ? void 0 : _a[slot]) === null || _b === void 0 ? void 0 : _b.link) === linkId;
            if ((_c = this.patchedNodeSlots[nodeId]) === null || _c === void 0 ? void 0 : _c["inputs"]) {
                let patchedHasIt = this.patchedNodeSlots[nodeId]["inputs"][slot] === linkId;
                has = patchedHasIt;
            }
            else {
                has = nodeHasIt;
            }
        }
        else {
            let nodeHasIt = (_f = (_e = (_d = node.outputs) === null || _d === void 0 ? void 0 : _d[slot]) === null || _e === void 0 ? void 0 : _e.links) === null || _f === void 0 ? void 0 : _f.includes(linkId);
            if ((_j = (_h = (_g = this.patchedNodeSlots[nodeId]) === null || _g === void 0 ? void 0 : _g["outputs"]) === null || _h === void 0 ? void 0 : _h[slot]) === null || _j === void 0 ? void 0 : _j["changes"][linkId]) {
                let patchedHasIt = this.patchedNodeSlots[nodeId]["outputs"][slot].links.includes(linkId);
                has = !!patchedHasIt;
            }
            else {
                has = !!nodeHasIt;
            }
        }
        return has;
    }
    nodeHasAnyLink(node, ioDir, slot) {
        var _a, _b, _c, _d, _e, _f, _g, _h, _j, _k;
        const nodeId = node.id;
        let hasAny = false;
        if (ioDir === IoDirection.INPUT) {
            let nodeHasAny = ((_b = (_a = node.inputs) === null || _a === void 0 ? void 0 : _a[slot]) === null || _b === void 0 ? void 0 : _b.link) != null;
            if ((_c = this.patchedNodeSlots[nodeId]) === null || _c === void 0 ? void 0 : _c["inputs"]) {
                let patchedHasAny = this.patchedNodeSlots[nodeId]["inputs"][slot] != null;
                hasAny = patchedHasAny;
            }
            else {
                hasAny = !!nodeHasAny;
            }
        }
        else {
            let nodeHasAny = (_f = (_e = (_d = node.outputs) === null || _d === void 0 ? void 0 : _d[slot]) === null || _e === void 0 ? void 0 : _e.links) === null || _f === void 0 ? void 0 : _f.length;
            if ((_j = (_h = (_g = this.patchedNodeSlots[nodeId]) === null || _g === void 0 ? void 0 : _g["outputs"]) === null || _h === void 0 ? void 0 : _h[slot]) === null || _j === void 0 ? void 0 : _j["changes"]) {
                let patchedHasAny = (_k = this.patchedNodeSlots[nodeId]["outputs"][slot].links) === null || _k === void 0 ? void 0 : _k.length;
                hasAny = !!patchedHasAny;
            }
            else {
                hasAny = !!nodeHasAny;
            }
        }
        return hasAny;
    }
}
class WorkflowLinkFixerSerialized extends WorkflowLinkFixer {
    constructor(graph) {
        super(graph);
    }
    getNodeById(id) {
        var _a;
        return (_a = this.graph.nodes.find((node) => Number(node.id) === id)) !== null && _a !== void 0 ? _a : null;
    }
    fix(force = false, times) {
        const ret = super.fix(force, times);
        this.graph.links = this.graph.links.filter((l) => !!l);
        return ret;
    }
    deleteGraphLink(id) {
        const idx = this.graph.links.findIndex((l) => l && (l[0] === id || l.id === id));
        if (idx === -1) {
            return `Link #${id} not found in workflow links.`;
        }
        this.graph.links.splice(idx, 1);
        return true;
    }
}
class WorkflowLinkFixerGraph extends WorkflowLinkFixer {
    constructor(graph) {
        super(graph);
    }
    getNodeById(id) {
        var _a;
        return (_a = this.graph.getNodeById(id)) !== null && _a !== void 0 ? _a : null;
    }
    deleteGraphLink(id) {
        if (this.graph.links instanceof Map) {
            if (!this.graph.links.has(id)) {
                return `Link #${id} not found in workflow links.`;
            }
            this.graph.links.delete(id);
            return true;
        }
        if (this.graph.links[id] == null) {
            return `Link #${id} not found in workflow links.`;
        }
        delete this.graph.links[id];
        return true;
    }
}
