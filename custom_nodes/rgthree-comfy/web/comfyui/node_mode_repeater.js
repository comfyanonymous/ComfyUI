import { app } from "../../scripts/app.js";
import { BaseCollectorNode } from "./base_node_collector.js";
import { NodeTypesString, stripRgthree } from "./constants.js";
import { PassThroughFollowing, addConnectionLayoutSupport, changeModeOfNodes, getConnectedInputNodesAndFilterPassThroughs, getConnectedOutputNodesAndFilterPassThroughs, getGroupNodes, } from "./utils.js";
class NodeModeRepeater extends BaseCollectorNode {
    constructor(title) {
        super(title);
        this.inputsPassThroughFollowing = PassThroughFollowing.ALL;
        this.comfyClass = NodeTypesString.NODE_MODE_REPEATER;
        this.hasRelayInput = false;
        this.hasTogglerOutput = false;
        this.onConstructed();
    }
    onConstructed() {
        this.addOutput("OPT_CONNECTION", "*", {
            color_on: "#Fc0",
            color_off: "#a80",
        });
        return super.onConstructed();
    }
    onConnectOutput(outputIndex, inputType, inputSlot, inputNode, inputIndex) {
        let canConnect = !this.hasRelayInput;
        canConnect =
            canConnect && super.onConnectOutput(outputIndex, inputType, inputSlot, inputNode, inputIndex);
        let nextNode = getConnectedOutputNodesAndFilterPassThroughs(this, inputNode)[0] || inputNode;
        return (canConnect &&
            [
                NodeTypesString.FAST_MUTER,
                NodeTypesString.FAST_BYPASSER,
                NodeTypesString.NODE_COLLECTOR,
                NodeTypesString.FAST_ACTIONS_BUTTON,
                NodeTypesString.REROUTE,
                NodeTypesString.RANDOM_UNMUTER,
            ].includes(nextNode.type || ""));
    }
    onConnectInput(inputIndex, outputType, outputSlot, outputNode, outputIndex) {
        var _a;
        let canConnect = (_a = super.onConnectInput) === null || _a === void 0 ? void 0 : _a.call(this, inputIndex, outputType, outputSlot, outputNode, outputIndex);
        let nextNode = getConnectedOutputNodesAndFilterPassThroughs(this, outputNode)[0] || outputNode;
        const isNextNodeRelay = nextNode.type === NodeTypesString.NODE_MODE_RELAY;
        return canConnect && (!isNextNodeRelay || !this.hasTogglerOutput);
    }
    onConnectionsChange(type, slotIndex, isConnected, linkInfo, ioSlot) {
        super.onConnectionsChange(type, slotIndex, isConnected, linkInfo, ioSlot);
        let hasTogglerOutput = false;
        let hasRelayInput = false;
        const outputNodes = getConnectedOutputNodesAndFilterPassThroughs(this);
        for (const outputNode of outputNodes) {
            if ((outputNode === null || outputNode === void 0 ? void 0 : outputNode.type) === NodeTypesString.FAST_MUTER ||
                (outputNode === null || outputNode === void 0 ? void 0 : outputNode.type) === NodeTypesString.FAST_BYPASSER) {
                hasTogglerOutput = true;
                break;
            }
        }
        const inputNodes = getConnectedInputNodesAndFilterPassThroughs(this);
        for (const [index, inputNode] of inputNodes.entries()) {
            if ((inputNode === null || inputNode === void 0 ? void 0 : inputNode.type) === NodeTypesString.NODE_MODE_RELAY) {
                if (hasTogglerOutput) {
                    console.log(`Can't be connected to a Relay if also output to a toggler.`);
                    this.disconnectInput(index);
                }
                else {
                    hasRelayInput = true;
                    if (this.inputs[index]) {
                        this.inputs[index].color_on = "#FC0";
                        this.inputs[index].color_off = "#a80";
                    }
                }
            }
            else {
                changeModeOfNodes(inputNode, this.mode);
            }
        }
        this.hasTogglerOutput = hasTogglerOutput;
        this.hasRelayInput = hasRelayInput;
        if (this.hasRelayInput) {
            if (this.outputs[0]) {
                this.disconnectOutput(0);
                this.removeOutput(0);
            }
        }
        else if (!this.outputs[0]) {
            this.addOutput("OPT_CONNECTION", "*", {
                color_on: "#Fc0",
                color_off: "#a80",
            });
        }
    }
    onModeChange(from, to) {
        var _a, _b;
        super.onModeChange(from, to);
        const linkedNodes = getConnectedInputNodesAndFilterPassThroughs(this).filter((node) => node.type !== NodeTypesString.NODE_MODE_RELAY);
        if (linkedNodes.length) {
            for (const node of linkedNodes) {
                if (node.type !== NodeTypesString.NODE_MODE_RELAY) {
                    changeModeOfNodes(node, to);
                }
            }
        }
        else if ((_b = (_a = this.graph) === null || _a === void 0 ? void 0 : _a._groups) === null || _b === void 0 ? void 0 : _b.length) {
            for (const group of this.graph._groups) {
                group.recomputeInsideNodes();
                const groupNodes = getGroupNodes(group);
                if (groupNodes === null || groupNodes === void 0 ? void 0 : groupNodes.includes(this)) {
                    for (const node of groupNodes) {
                        if (node !== this) {
                            changeModeOfNodes(node, to);
                        }
                    }
                }
            }
        }
    }
    getHelp() {
        return `
      <p>
        When this node's mode (Mute, Bypass, Active) changes, it will "repeat" that mode to all
        connected input nodes, or, if there are no connected nodes AND it is overlapping a group,
        "repeat" it's mode to all nodes in that group.
      </p>
      <ul>
        <li><p>
          Optionally, connect this mode's output to a ${stripRgthree(NodeTypesString.FAST_MUTER)}
          or ${stripRgthree(NodeTypesString.FAST_BYPASSER)} for a single toggle to quickly
          mute/bypass all its connected nodes.
        </p></li>
        <li><p>
          Optionally, connect a ${stripRgthree(NodeTypesString.NODE_MODE_RELAY)} to this nodes
          inputs to have it automatically toggle its mode. If connected, this will always take
          precedence (and disconnect any connected fast togglers).
        </p></li>
      </ul>
    `;
    }
}
NodeModeRepeater.type = NodeTypesString.NODE_MODE_REPEATER;
NodeModeRepeater.title = NodeTypesString.NODE_MODE_REPEATER;
app.registerExtension({
    name: "rgthree.NodeModeRepeater",
    registerCustomNodes() {
        addConnectionLayoutSupport(NodeModeRepeater, app, [
            ["Left", "Right"],
            ["Right", "Left"],
        ]);
        LiteGraph.registerNodeType(NodeModeRepeater.type, NodeModeRepeater);
        NodeModeRepeater.category = NodeModeRepeater._category;
    },
});
