import { UseEverywhereList } from "./use_everywhere_classes.js";
import { node_in_loop, node_is_live, is_connected, is_UEnode, Logger, get_real_node, Pausable } from "./use_everywhere_utilities.js";
import { convert_to_links } from "./use_everywhere_apply.js";
import { app } from "../../scripts/app.js";
import { settingsCache } from "./use_everywhere_cache.js";
import { master_graph, node_graph, visible_graph } from "./use_everywhere_subgraph_utils.js";

class GraphAnalyser extends Pausable {
    static _instance;
    static instance() {
        if (!this._instance) this._instance = new GraphAnalyser();
        return this._instance;
    }

    constructor() {
        super('GraphAnalyser')
        this.original_graphToPrompt = app.graphToPrompt;
        this.ambiguity_messages = [];
        this.latest_ues = null
    }

    modify_graphs_recursively(graph, mods) {
        const modifications = convert_to_links( this.analyse_graph(graph), null, graph )
        mods.push( modifications );
        if (!graph.extra) graph.extra = {}
        graph.extra['links_added_by_ue'] = modifications.added_links.map(x=>x.id)
        graph.nodes.filter((node)=>(node.subgraph)).forEach((node) => {this.modify_graphs_recursively(node.subgraph, mods);});
    }

    async graph_to_prompt() {
        var p;
        this.pause('graph_to_prompt')
        try { 
            const mods = []
            this.modify_graphs_recursively(master_graph(), mods);

            // Now create the prompt using the ComfyUI original functionality and the patched graph
            p = await this.original_graphToPrompt.apply(app);
            // Remove the added virtual links
            mods.forEach((mod)=>{mod.restorer()})

        } catch (e) { 
            Logger.log_error(e)
        } finally { 
            this.unpause()
        }

        if (!p) {
            Logger.log_problem("graph_to_prompt_fallback")
            p = await this.original_graphToPrompt.apply(app);
        }
        
        return p;
    }

    analyse_visible_graph() { return this.analyse_graph(visible_graph()); }

    analyse_master_graph() { return this.analyse_graph(master_graph()); }

    wait_to_analyse_visible_graph() { return this.wait_to_analyse_graph(visible_graph()); }

    wait_to_analyse_master_graph() { return this.wait_to_analyse_graph(master_graph()); }

    wait_to_analyse_graph(graph) {
        if (this.paused()) { 
            Logger.log_problem("Don't know how to wait", null, true);
        }
        return this.analyse_graph(graph);
    }

    maybe_check_for_loops() {
        if (settingsCache.getSettingValue('Use Everywhere.Options.checkloops')) {
            try {
                node_in_loop(live_nodes, links_added);
            } catch (e) {
                if (!e.stack) throw e;
                if (e.ues && e.ues.length > 0){
                    alert(`Loop (${e.stack}) with broadcast (${e.ues}) - not submitting workflow`);
                } else {
                    alert(`Loop (${e.stack}) - not submitting workflow`);
                }
                throw new Error(`Loop Detected ${e.stack}, ${e.ues}`, {"cause":e});
            }
        }
    }

    analyse_graph(graph) {
        this.ambiguity_messages = [];
        const treat_bypassed_as_live = settingsCache.getSettingValue("Use Everywhere.Options.connect_to_bypassed") || this.connect_to_bypassed
        const live_nodes = graph.nodes.filter((node) => node_is_live(node, treat_bypassed_as_live))
                
        // Create a UseEverywhereList and populate it from all live (not bypassed) UE nodes
        const ues = new UseEverywhereList();
        live_nodes.filter((node) => is_UEnode(node)).filter((node)=>node_is_live(node,false)).forEach(node => { ues.add_ue_from_node(node); })

        // List all unconnected inputs on non-UE nodes which are connectable
        const connectable = []
        live_nodes.filter((node) => !is_UEnode(node)).forEach(node => {
            if (node && !node.properties.rejects_ue_links) {
                //if (!real_node._widget_name_map) real_node._widget_name_map =  real_node.widgets?.map(w => w.name) || [];
                node.inputs?.forEach((input,index) => {
                    if (is_connected(input, treat_bypassed_as_live, node_graph(node))) return;  
                    if (node.reject_ue_connection && node.reject_ue_connection(input)) return;
                    if (node._getWidgetByName(input.name) && !(node.properties?.ue_properties?.widget_ue_connectable && node.properties.ue_properties.widget_ue_connectable[input.name])) return;
                    connectable.push({node, input, index});
                })
            }
        })

        // see if we can connect them
        const links_added = new Set();
        connectable.forEach(({node, input, index}) => {
            var ue = ues.find_best_match(node, input, this.ambiguity_messages);
            if (ue) {
                links_added.add({
                    "downstream":node.id, "downstream_slot":index,
                    "upstream":ue.output[0], "upstream_slot":ue.output[1], 
                    "controller":ue.controller.id,
                    "type":ue.type
                });
            }
        });

        graph.extra['ue_links'] = Array.from(links_added)
    
        if (this.ambiguity_messages.length) Logger.log_problem("Ambiguous connections", this.ambiguity_messages, true);
 
        this.latest_ues = ues;
        return this.latest_ues;
    }
}

export { GraphAnalyser }
