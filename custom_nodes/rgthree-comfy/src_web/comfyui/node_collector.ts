import type {
  LLink,
  LGraph,
  LGraphCanvas,
  LGraphNode as TLGraphNode,
  IContextMenuOptions,
  ContextMenu,
  IContextMenuValue,
  Size,
  ISerialisedNode,
  Point,
} from "@comfyorg/frontend";

import {app} from "scripts/app.js";
import {addConnectionLayoutSupport} from "./utils.js";
import {wait} from "rgthree/common/shared_utils.js";
import {ComfyWidgets} from "scripts/widgets.js";
import {BaseCollectorNode} from "./base_node_collector.js";
import {NodeTypesString} from "./constants.js";

/**
 * The Collector Node. Takes any number of inputs as connections for nodes and collects them into
 * one outputs. The next node will decide what to do with them.
 *
 * Currently only works with the Fast Muter, Fast Bypasser, and Fast Actions Button.
 */
class CollectorNode extends BaseCollectorNode {
  static override type = NodeTypesString.NODE_COLLECTOR;
  static override title = NodeTypesString.NODE_COLLECTOR;
  override comfyClass = NodeTypesString.NODE_COLLECTOR;

  constructor(title = CollectorNode.title) {
    super(title);
    this.onConstructed();
  }

  override onConstructed(): boolean {
    this.addOutput("Output", "*");
    return super.onConstructed();
  }
}

/** Legacy "Combiner" */
class CombinerNode extends CollectorNode {
  static legacyType = "Node Combiner (rgthree)";
  static override title = "‼️ Node Combiner [DEPRECATED]";

  constructor(title = CombinerNode.title) {
    super(title);

    const note = ComfyWidgets["STRING"](
      this,
      "last_seed",
      ["STRING", {multiline: true}],
      app,
    ).widget;
    note.inputEl!.value =
      'The Node Combiner has been renamed to Node Collector. You can right-click and select "Update to Node Collector" to attempt to automatically update.';
    note.inputEl!.readOnly = true;
    note.inputEl!.style.backgroundColor = "#332222";
    note.inputEl!.style.fontWeight = "bold";
    note.inputEl!.style.fontStyle = "italic";
    note.inputEl!.style.opacity = "0.8";

    this.getExtraMenuOptions = (
      canvas: LGraphCanvas,
      options: (IContextMenuValue<unknown> | null)[],
    ): (IContextMenuValue<unknown> | null)[] => {
      options.splice(options.length - 1, 0, {
        content: "‼️ Update to Node Collector",
        callback: (
          _value: IContextMenuValue,
          _options: IContextMenuOptions,
          _event: MouseEvent,
          _parentMenu: ContextMenu | undefined,
          _node: TLGraphNode,
        ) => {
          updateCombinerToCollector(this);
        },
      });
      return [];
    };
  }

  override configure(info: ISerialisedNode) {
    super.configure(info);
    if (this.title != CombinerNode.title && !this.title.startsWith("‼️")) {
      this.title = "‼️ " + this.title;
    }
  }
}

/**
 * Updates a Node Combiner to a Node Collector.
 */
async function updateCombinerToCollector(node: TLGraphNode) {
  if (node.type === CombinerNode.legacyType) {
    // Create a new CollectorNode.
    const newNode = new CollectorNode();
    if (node.title != CombinerNode.title) {
      newNode.title = node.title.replace("‼️ ", "");
    }
    // Port the position, size, and properties from the old node.
    newNode.pos = [...node.pos] as Point;
    newNode.size = [...node.size] as Size;
    newNode.properties = {...node.properties};
    // We now collect the links data, inputs and outputs, of the old node since these will be
    // lost when we remove it.
    const links: any[] = [];
    const graph = (node.graph || app.graph);
    for (const [index, output] of node.outputs.entries()) {
      for (const linkId of output.links || []) {
        const link: LLink = graph.links[linkId]!;
        if (!link) continue;
        const targetNode = graph.getNodeById(link.target_id);
        links.push({node: newNode, slot: index, targetNode, targetSlot: link.target_slot});
      }
    }
    for (const [index, input] of node.inputs.entries()) {
      const linkId = input.link;
      if (linkId) {
        const link: LLink = graph.links[linkId]!;
        const originNode = graph.getNodeById(link.origin_id);
        links.push({
          node: originNode,
          slot: link.origin_slot,
          targetNode: newNode,
          targetSlot: index,
        });
      }
    }
    // Add the new node, remove the old node.
    graph.add(newNode);
    await wait();
    // Now go through and connect the other nodes up as they were.
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
