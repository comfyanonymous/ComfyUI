import { app } from "../../scripts/app.js";

export function master_graph()    { return app.graph }
export function master_graph_id() { return master_graph().id }

export function visible_graph()    { return app.canvas.graph }
export function visible_graph_id() { return visible_graph().id }

export function node_graph(node)    { return node.graph }
export function node_graph_id(node) { return node_graph(node).id }

export function in_visible_graph(node) { 
    try {
        return node_graph_id(node) == visible_graph_id() 
    } catch {
        return false
    }
}

export function get_subgraph_input_type(graph, slot) { return graph.inputNode.slots[slot].type }
export function link_is_from_subgraph_input(link) { return link.origin_id==-10 }

class WrappedInputNode {
    constructor(subgraph_input_node) {
        this.subgraph_input_node = subgraph_input_node;
        this.graph = subgraph_input_node.subgraph;
    }

    connect(output_index, input_node, input_index) {
        this.graph.last_link_id += 1
        this.graph.links[this.graph.last_link_id] = new LLink(this.graph.last_link_id, this.subgraph_input_node.slots[output_index].type, -10, output_index, input_node.id, input_index) 
        input_node.inputs[input_index].link = this.graph.last_link_id;
        return this.graph.links[this.graph.last_link_id]
    }

}
export function wrap_input(subgraph_input_node) {
    return new WrappedInputNode(subgraph_input_node);
}