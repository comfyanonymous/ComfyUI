import { rgthree } from "./rgthree.js";
import { BaseAnyInputConnectedNode } from "./base_any_input_connected_node.js";
import { PassThroughFollowing, getConnectedInputNodes, getConnectedInputNodesAndFilterPassThroughs, shouldPassThrough, } from "./utils.js";
export class BaseCollectorNode extends BaseAnyInputConnectedNode {
    constructor(title) {
        super(title);
        this.inputsPassThroughFollowing = PassThroughFollowing.REROUTE_ONLY;
        this.logger = rgthree.newLogSession("[BaseCollectorNode]");
    }
    clone() {
        const cloned = super.clone();
        return cloned;
    }
    handleLinkedNodesStabilization(linkedNodes) {
        return false;
    }
    onConnectInput(inputIndex, outputType, outputSlot, outputNode, outputIndex) {
        var _a, _b, _c, _d;
        let canConnect = super.onConnectInput(inputIndex, outputType, outputSlot, outputNode, outputIndex);
        if (canConnect) {
            const allConnectedNodes = getConnectedInputNodes(this);
            const nodesAlreadyInSlot = getConnectedInputNodes(this, undefined, inputIndex);
            if (allConnectedNodes.includes(outputNode)) {
                const [n, v] = this.logger.debugParts(`${outputNode.title} is already connected to ${this.title}.`);
                (_a = console[n]) === null || _a === void 0 ? void 0 : _a.call(console, ...v);
                if (nodesAlreadyInSlot.includes(outputNode)) {
                    const [n, v] = this.logger.debugParts(`... but letting it slide since it's for the same slot.`);
                    (_b = console[n]) === null || _b === void 0 ? void 0 : _b.call(console, ...v);
                }
                else {
                    canConnect = false;
                }
            }
            if (canConnect && shouldPassThrough(outputNode, PassThroughFollowing.REROUTE_ONLY)) {
                const connectedNode = getConnectedInputNodesAndFilterPassThroughs(outputNode, undefined, undefined, PassThroughFollowing.REROUTE_ONLY)[0];
                if (connectedNode && allConnectedNodes.includes(connectedNode)) {
                    const [n, v] = this.logger.debugParts(`${connectedNode.title} is already connected to ${this.title}.`);
                    (_c = console[n]) === null || _c === void 0 ? void 0 : _c.call(console, ...v);
                    if (nodesAlreadyInSlot.includes(connectedNode)) {
                        const [n, v] = this.logger.debugParts(`... but letting it slide since it's for the same slot.`);
                        (_d = console[n]) === null || _d === void 0 ? void 0 : _d.call(console, ...v);
                    }
                    else {
                        canConnect = false;
                    }
                }
            }
        }
        return canConnect;
    }
}
