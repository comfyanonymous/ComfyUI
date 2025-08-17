import { app } from "../../scripts/app.js";
import { display_name } from "./use_everywhere_classes.js";
import { master_graph, node_graph } from "./use_everywhere_subgraph_utils.js";
import { is_UEnode, get_real_node, Logger } from "./use_everywhere_utilities.js";


function _convert_to_links(ue, added_links, removed_links) {
    const output_node_id = ue.output[0];
    const output_index = ue.output[1];
    const output_node = get_real_node(output_node_id, ue.graph);
    Logger.log_info("Adding links for " + ue.description);
    ue.sending_to.forEach((st) => {
        const input_node_id = st.node.id;
        const input_node = get_real_node(input_node_id, ue.graph);
        const input_index = st.input_index;
        if (input_node.inputs[input_index].link) { // why would this be happening?
            const llink = ue.graph.links[input_node.inputs[input_index].link]
            removed_links.push( {...llink} )
        }
        const new_link = output_node.connect(output_index, input_node, input_index);
        if (!new_link)
            console.error("Failed to connect nodes: " +
                          `${output_node_id}[${output_index}] -> ` +
                          `${input_node_id}[${input_index}].`);
        else { // Memorize the links we are adding to remove them later
            if (added_links) {
                new_link.control_node = ue.controller;
                added_links.push(new_link);
            }
            Logger.log_info(`  -> ${display_name(st.node)}, ${st.input.name} ` +
                                           `(${st.node.id}.${st.input_index}) (ID: ${new_link.id})`);
        }
    });
}

function convert_to_links(ues, control_node, graph) {
    if (control_node) {
        if (!graph) graph = node_graph(control_node)
        return _convert_graph_to_links(graph, ues, control_node.id );
    } else {
        if (!graph) graph = master_graph();
        return _convert_graph_to_links(graph, ues, undefined);
    }
}

function _convert_graph_to_links(graph, ues, control_node_id) {
    if (!ues?.ues) return {restorer:function(){}, added_links:[]}
    const added_links = []
    const removed_links = []
    ues.ues.forEach((ue)=> {
        if (control_node_id==undefined || ue.controller.id == control_node_id) _convert_to_links(ue, added_links, removed_links);
    });

    const restorer = function() {
        const links_within_graph_ids = new Set()
        const links_within_subgraph_ids = new Set()
        const links_into_subgraph_ids = new Set()
        const links_out_of_subgraph = new Set()
        const problem_links = new Set()

        const new_subgraph_node = graph._nodes_by_id[graph.last_node_id];
        const new_subgraph = new_subgraph_node?.subgraph;

        added_links.forEach(added_link => { 
            var id = added_link.id
            try {
                if (graph.links[id]) {
                    links_within_graph_ids.add(id);  
                } else { 
                    const link_to_subgraph_node = new_subgraph_node?.inputs.find(input => input.linkIds.includes(id))?.link
                    if (link_to_subgraph_node) {
                        links_into_subgraph_ids.add(link_to_subgraph_node)
                    } else {
                        if (new_subgraph.links[id]) {
                            if (new_subgraph.links[id].target_id==-20) {
                                links_out_of_subgraph.add(added_link)
                            } else {
                                links_within_subgraph_ids.add(id)
                            }
                        } else problem_links.add(added_link)
                    }
                }
            } catch (e) {
                problem_links.add(added_link)
                Logger.log_error(e);
            }
        });

        links_out_of_subgraph.forEach((llink)=>{
            const control_node_id = llink.control_node?.id
            const control_node_in_graph = graph._nodes_by_id[control_node_id]
            if (control_node_in_graph) {
                // the control node is outside, so we should disconnect anything else that is connected to the same output of the subgraph node
                new_subgraph_node.outputs[new_subgraph.links[llink.id].target_slot].links.forEach((link_id)=>{
                    if (graph.links[link_id].target_id==control_node_id) {
                        // leave the link to the UE node
                    } else {
                        // remove others
                        links_within_graph_ids.add(link_id)
                    }
                })
            } else {
                // control node is inside, leave the link, since we can't UE connect to the output panel
            }
        })

        links_within_graph_ids.forEach((lid)=>{graph.removeLink(lid)})
        links_into_subgraph_ids.forEach((lid)=>{graph.removeLink(lid)})
        links_within_subgraph_ids.forEach((lid)=>{new_subgraph.removeLink(lid)})

        if (problem_links.size>0) {
            Logger.log_problem("Failed to work out how to remove some temporary links:", problem_links)
        }

        removed_links.forEach(llink => {
            graph._nodes_by_id[llink.origin_id].connect(llink.origin_slot, graph._nodes_by_id[llink.target_id], llink.target_slot)
        })
    };

    return {restorer:restorer, added_links:added_links}
}

function remove_this(node, keep_seed_everywhere) {
    return  (is_UEnode(node) && !(keep_seed_everywhere && node.comfyClass=="Seed Everywhere") ) 
}

function remove_all_ues(keep_seed_everywhere) {
    app.graph._nodes.filter((node)=>remove_this(node, keep_seed_everywhere)).forEach((node)=>{app.graph.remove(node)})
}

export {convert_to_links, remove_all_ues}