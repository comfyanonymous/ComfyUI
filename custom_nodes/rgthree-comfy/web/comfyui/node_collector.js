import { app } from "../../scripts/app.js";
import { addConnectionLayoutSupport } from "./utils.js";
import { wait } from "../../rgthree/common/shared_utils.js";
import { ComfyWidgets } from "../../scripts/widgets.js";
import { BaseCollectorNode } from "./base_node_collector.js";
import { NodeTypesString } from "./constants.js";
class CollectorNode extends BaseCollectorNode {
    constructor(title = CollectorNode.title) {
        super(title);
        this.comfyClass = NodeTypesString.NODE_COLLECTOR;
        this.onConstructed();
    }
    onConstructed() {
        this.addOutput("Output", "*");
        return super.onConstructed();
    }
}
CollectorNode.type = NodeTypesString.NODE_COLLECTOR;
CollectorNode.title = NodeTypesString.NODE_COLLECTOR;
class CombinerNode extends CollectorNode {
    constructor(title = CombinerNode.title) {
        super(title);
        const note = ComfyWidgets["STRING"](this, "last_seed", ["STRING", { multiline: true }], app).widget;
        note.inputEl.value =
            'The Node Combiner has been renamed to Node Collector. You can right-click and select "Update to Node Collector" to attempt to automatically update.';
        note.inputEl.readOnly = true;
        note.inputEl.style.backgroundColor = "#332222";
        note.inputEl.style.fontWeight = "bold";
        note.inputEl.style.fontStyle = "italic";
        note.inputEl.style.opacity = "0.8";
        this.getExtraMenuOptions = (canvas, options) => {
            options.splice(options.length - 1, 0, {
                content: "‼️ Update to Node Collector",
                callback: (_value, _options, _event, _parentMenu, _node) => {
                    updateCombinerToCollector(this);
                },
            });
            return [];
        };
    }
    configure(info) {
        super.configure(info);
        if (this.title != CombinerNode.title && !this.title.startsWith("‼️")) {
            this.title = "‼️ " + this.title;
        }
    }
}
CombinerNode.legacyType = "Node Combiner (rgthree)";
CombinerNode.title = "‼️ Node Combiner [DEPRECATED]";
async function updateCombinerToCollector(node) {
    if (node.type === CombinerNode.legacyType) {
        const newNode = new CollectorNode();
        if (node.title != CombinerNode.title) {
            newNode.title = node.title.replace("‼️ ", "");
        }
        newNode.pos = [...node.pos];
        newNode.size = [...node.size];
        newNode.properties = { ...node.properties };
        const links = [];
        const graph = (node.graph || app.graph);
        for (const [index, output] of node.outputs.entries()) {
            for (const linkId of output.links || []) {
                const link = graph.links[linkId];
                if (!link)
                    continue;
                const targetNode = graph.getNodeById(link.target_id);
                links.push({ node: newNode, slot: index, targetNode, targetSlot: link.target_slot });
            }
        }
        for (const [index, input] of node.inputs.entries()) {
            const linkId = input.link;
            if (linkId) {
                const link = graph.links[linkId];
                const originNode = graph.getNodeById(link.origin_id);
                links.push({
                    node: originNode,
                    slot: link.origin_slot,
                    targetNode: newNode,
                    targetSlot: index,
                });
            }
        }
        graph.add(newNode);
        await wait();
        for (const link of links) {
            link.node.connect(link.slot, link.targetNode, link.targetSlot);
        }
        await wait();
        graph.remove(node);
    }
}
app.registerExtension({
    name: "rgthree.NodeCollector",
    registerCustomNodes() {
        addConnectionLayoutSupport(CollectorNode, app, [
            ["Left", "Right"],
            ["Right", "Left"],
        ]);
        LiteGraph.registerNodeType(CollectorNode.title, CollectorNode);
        CollectorNode.category = CollectorNode._category;
    },
});
app.registerExtension({
    name: "rgthree.NodeCombiner",
    registerCustomNodes() {
        addConnectionLayoutSupport(CombinerNode, app, [
            ["Left", "Right"],
            ["Right", "Left"],
        ]);
        LiteGraph.registerNodeType(CombinerNode.legacyType, CombinerNode);
        CombinerNode.category = CombinerNode._category;
    },
});
