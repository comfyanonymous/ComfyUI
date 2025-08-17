import { app } from "../../scripts/app.js";
import { settingsCache } from "./use_everywhere_cache.js";
import { link_is_from_subgraph_input, node_graph, visible_graph, wrap_input } from "./use_everywhere_subgraph_utils.js";

export const VERSION = "7.0.1"

export function create( tag, clss, parent, properties ) {
    const nd = document.createElement(tag);
    if (clss)       clss.split(" ").forEach((s) => nd.classList.add(s))
    if (parent)     parent.appendChild(nd);
    if (properties) Object.assign(nd, properties);
    return nd;
}

/*
Return  1 if x is  a   later version than y (or y is not defined)
Return -1 if x is an earlier version than y (or x is not defined)
Return  0 if they are the same version 
*/
function version_compare(x,y) {
    if (x==y) return  0
    if (!y)   return  1
    if (!x)   return -1
    const xbits = x.split('.')
    const ybits = y.split('.')
    var result = 0
    for (var i=0; result!=0 && i<Math.min(xbits.length, ybits.length); i++) {
        if (parseInt(xbits[i]) < parseInt(ybits[i])) result = -1
        if (parseInt(xbits[i]) > parseInt(ybits[i])) result = 1
    }
    if (result==0) {
        if (xbits.length < ybits.length) result = -1
        if (xbits.length > ybits.length) result = 1
    }
    return result
}

export function version_at_least(x,y) {
    return (version_compare(x,y) >= 0)
}

/*
Return the node object for this node_id. 
*/
function get_real_node(node_id, graph) {
    if (!graph) graph = visible_graph()
    const nid = node_id.toString();
    if (nid==-10) return wrap_input(graph.inputNode); // special case for subgraph input
    return graph._nodes_by_id[nid];
}

class Logger {
    static LIMITED_LOG_BLOCKED = false;
    static LIMITED_LOG_MS      = 5000;
    static level;  // 0 for errors only, 1 activates 'log_problem', 2 activates 'log_info', 3 activates 'log_detail'

    static log_error(message) { console.error(message) }

    static log(message, foreachable, limited) {    
        if (limited && Logger.check_limited()) return
        console.log(message);
        try {
            foreachable?.forEach((x)=>{console.log(x)})
        } catch {
            let a;
        }
    }

    static check_limited() {
        if (Logger.LIMITED_LOG_BLOCKED) return true
        Logger.LIMITED_LOG_BLOCKED = true
        setTimeout( ()=>{Logger.LIMITED_LOG_BLOCKED = false}, Logger.LIMITED_LOG_MS )
        return false
    }

    static null() {}

    static level_changed(new_level) {
        Logger.level = new_level    
        Logger.log_detail  = (Logger.level>=3) ? Logger.log : Logger.null
        Logger.log_info    = (Logger.level>=2) ? Logger.log : Logger.null
        Logger.log_problem = (Logger.level>=1) ? Logger.log : Logger.null
    }

    static log_detail(){}
    static log_info(){}
    static log_problem(){}
}

Logger.level_changed(settingsCache.getSettingValue('Use Everywhere.Options.logging'))
settingsCache.addCallback('Use Everywhere.Options.logging', Logger.level_changed)

class GraphConverter {
    static _instance;
    static instance() {
        if (!GraphConverter._instance) GraphConverter._instance = new GraphConverter();
        return GraphConverter._instance;
    }

    constructor() { 
        this.node_input_map = {};
        this.given_message = false;
        this.did_conversion = false;
        this.graph_being_configured = false;
     }

    running_116_plus() {
        const version = __COMFYUI_FRONTEND_VERSION__.split('.')
        return (parseInt(version[0])>=1 && (parseInt(version[0])>1 || parseInt(version[1])>=16))
    }

    store_node_input_map(data) { 
        this.node_input_map = {};
        data?.nodes.filter((node)=>(node.inputs)).forEach((node) => { this.node_input_map[node.id] = node.inputs.map((input) => input.name); })
        Logger.log_detail("stored node_input_map", this.node_input_map);
    }


    clean_ue_node(node) {
        var expected_inputs = 1
        if (node.type == "Seed Everywhere") expected_inputs = 0
        if (node.type == "Prompts Everywhere") expected_inputs = 2
        if (node.type == "Anything Everywhere3") expected_inputs = 3
        if (node.type == "Anything Everywhere?") expected_inputs = 4

        // remove all the 'anything' inputs (because they may be duplicated)
        const removed = node.inputs.filter(i=>(i.label=='anything' || i.label=='*'))
        node.inputs   = node.inputs.filter(i=>(i.label!='anything' && i.label!='*')) 
        // add them back as required
        while (node.inputs.length < expected_inputs) { node.inputs.push(removed.pop()) }
        // the input comes before the regex widgets in UE?
        if (expected_inputs==4) {
            while(node.inputs[0].name.includes('regex')) {
                node.inputs.unshift(node.inputs.pop()) 
            }
        }
        // fix the localized names
        node.inputs = node.inputs.map((input) => {
            if (!input.localized_name || input.localized_name.startsWith('anything')) input.localized_name = input.name
            return input;
        })

        // set types to match
        node.inputs.forEach((input) => {
            if (input.type=='*') {
                const graph = node_graph(node);
                if (input.link) {
                    const llink = graph.links[input.link];
                    if (link_is_from_subgraph_input(llink)) {
                        input.type = get_subgraph_input_type(graph, llink.origin_slot);
                    } else {
                        input.type = llink.type;
                    }
                } else {
                    input.type = (input.label && input.label!='anything') ? input.label : input.name
                }
            }
        });

        Logger.log_detail(`clean_ue_node ${node.id} (${node.type})`, node.inputs);
    }

    convert_if_pre_116(node) {
        if (!node) return;

        if (node.IS_UE) this.clean_ue_node(node)
        
        if (node.properties?.ue_properties?.widget_ue_connectable) return
        if (node.properties?.widget_ue_connectable) return  // pre 7.0 node which will be converted

        if (!this.given_message) {
            Logger.log_info(`Graph was saved with a version of ComfyUI before 1.16, so Anything Everywhere will try to work out which widgets are connectable`);
            this.given_message = true;
        }

        if (!node.properties.ue_properties) node.properties.ue_properties = {}
        node.properties.ue_properties['widget_ue_connectable'] = {}
        const widget_names = node.widgets?.map(w => w.name) || [];

        if (!(this.node_input_map[node.id])) {
            Logger.log_detail(`node ${node.id} (${node.type} has no node_input_map`);
        } else {
            this.node_input_map[node.id].filter((input_name)=>widget_names.includes(input_name)).forEach((input_name) => {
                node.properties.ue_properties['widget_ue_connectable'][input_name] = true;
                this.did_conversion = true;
                Logger.log_info(`node ${node.id} widget ${input_name} marked as accepting UE because it was an input when saved`);
            });
        }
    }

    remove_saved_ue_links_recursively(graph) {
        if (graph.extra?.links_added_by_ue) {
            graph.extra.links_added_by_ue.forEach((link_id) => { app.graph.links.delete(link_id); })
        }
        graph.nodes.filter((node)=>(node.subgraph)).forEach((node) => {this.remove_saved_ue_links_recursively(node.subgraph);});
    }

}

export const graphConverter = GraphConverter.instance();

class LoopError extends Error {
    constructor(id, stack, ues) {
        super("Loop detected");
        this.id = id;
        this.stack = [...stack];
        this.ues = [...ues];
    }
}

function find_all_upstream(node, links_added) {
    const all_upstream = [];
    node?.inputs?.forEach((input) => { // normal links
        const link_id = input.link;
        if (link_id) {
            const link = app.graph.links[link_id];
            if (link) all_upstream.push({id:link.origin_id, slot:link.origin_slot});
        }
    });
    links_added.forEach((la)=>{ // UE links
        if (la.downstream==node.id) {
            all_upstream.push({id:la.upstream, slot:la.upstream_slot, ue:la.controller.toString()})
        }
    });

    return all_upstream;
}

function recursive_follow(node, links_added, stack, nodes_cleared, ues, count, slot) {
    count += 1;
    if (stack.includes(node.id.toString())) throw new LoopError(node.id, new Set(stack), new Set(ues));
    if (nodes_cleared.has(node.id.toString())) return;
    stack.push(node.id.toString());

    find_all_upstream(node, links_added).forEach((upstream) => {
        if (upstream.ue) ues.push(upstream.ue);
        count = recursive_follow(upstream, links_added, stack, nodes_cleared, ues, count, upstream.slot);
        if (upstream.ue) ues.pop();
    })

    nodes_cleared.add(node.id.toString());
    stack.pop();
    return count;
}

/*
Throw a LoopError if there is a loop.
live_nodes is a list of all live (ie not bypassed) nodes in the graph
links_added is a list of the UE virtuals links 
*/
function node_in_loop(live_nodes, links_added) {
    var nodes_to_check = [];
    const nodes_cleared = new Set();
    live_nodes.forEach((n)=>nodes_to_check.push(n));
    var count = 0;
    while (nodes_to_check.length>0) {
        const node = nodes_to_check.pop();
        count += recursive_follow(node, links_added, [], nodes_cleared, [], 0, -1);
        nodes_to_check = nodes_to_check.filter((nid)=>!nodes_cleared.has(nid.toString()));
    }
    console.log(`node_in_loop made ${count} checks`)
}

/*
Is a node alive (ie not bypassed or set to never)
*/
function node_is_live(node, treat_bypassed_as_live){
    if (!node) return false;
    if (node.mode===0) return true;
    if (node.mode===2 || node.mode===4) return !!treat_bypassed_as_live;
    Logger.log_error(`node ${node.id} has mode ${node.mode} - I only understand modes 0, 2 and 4`);
    return true;
}

function node_is_bypassed(node) {
    return (node.mode===4);
}

/*
Given a link object, and the type of the link,
go upstream, following links with the same type, until you find a parent node which isn't bypassed.
If either type or original link is null, or if the upstream thread ends, return null
*/
function handle_bypass(original_link, type, graph) {
    if (!type || !original_link) return null;
    var link = original_link;
    if (link_is_from_subgraph_input(link)) return link
    var parent = get_real_node(link.origin_id, graph);
    if (!parent) return null;
    while (node_is_bypassed(parent)) {
        if (!parent.inputs) return null;
        var link_id;
        if (parent?.inputs[link.origin_slot]?.type == type) link_id = parent.inputs[link.origin_slot].link; // try matching number first
        else link_id = parent.inputs.find((input)=>input.type==type)?.link;
        if (!link_id) { return null; }
        link = graph.links[link_id];
        parent = get_real_node(link.origin_id, graph);
    }
    return link;
}




/*
Does this input connect upstream to a live node?
*/
function is_connected(input, treat_bypassed_as_live, graph) {
    const link_id = input.link;
    if (link_id === null) return false;                                    // no connection
    var the_link = graph.links[link_id];
    if (!the_link) return false; 
    if (treat_bypassed_as_live) return true;
    the_link = handle_bypass(the_link, the_link.type, graph);              // find the link upstream of bypasses
    if (!the_link) return false;                                           // no source for data.
    return true;
}

/*
Is this a UE node?
*/
function is_UEnode(node_or_nodeType) {
    const title = node_or_nodeType.type || node_or_nodeType.comfyClass;
    return ((title) && (title.startsWith("Anything Everywhere") || title==="Seed Everywhere" || title==="Prompts Everywhere"))
}

function is_helper(node_or_nodeType) {
    const title = node_or_nodeType.type || node_or_nodeType.comfyClass;
    return ((title) && (title.startsWith("Simple String")))
}

/*
Inject a call into a method on object with name methodname.
The injection is added at the end of the existing method (if the method didn't exist, it is created)
injectionthis and injectionarguments are passed into the apply call (as the this and the arguments)
*/
function inject(object, methodname, tracetext, injection, injectionthis, injectionarguments) {
    const original = object[methodname];
    object[methodname] = function() {
        original?.apply(this, arguments);
        injection.apply(injectionthis, injectionarguments);
    }
}


export { node_in_loop, handle_bypass, node_is_live, is_connected, is_UEnode, is_helper, inject, Logger, get_real_node }

export function defineProperty(instance, property, desc) {
    const existingDesc = Object.getOwnPropertyDescriptor(instance, property);
    if (existingDesc?.configurable === false) {
      throw new Error(`Error: Cannot define un-configurable property "${property}"`);
    }
    if (existingDesc?.get && desc.get) {
      const descGet = desc.get;
      desc.get = () => {
        existingDesc.get.apply(instance, []);
        return descGet.apply(instance, []);
      };
    }
    if (existingDesc?.set && desc.set) {
      const descSet = desc.set;
      desc.set = (v) => {
        existingDesc.set.apply(instance, [v]);
        return descSet.apply(instance, [v]);
      };
    }
    desc.enumerable = desc.enumerable ?? existingDesc?.enumerable ?? true;
    desc.configurable = desc.configurable ?? existingDesc?.configurable ?? true;
    if (!desc.get && !desc.set) {
      desc.writable = desc.writable ?? existingDesc?.writable ?? true;
    }
    return Object.defineProperty(instance, property, desc);
  }

export class Pausable {
    constructor(name) {
        this.name = name
        this.pause_depth = 0
    }
    pause(note, ms) {
        this.pause_depth += 1;
        if (this.pause_depth>10) {
            Logger.log_error(`${this.name} Over pausing`)
        }
        Logger.log_detail(`${this.name} pause ${note} with ${ms}`)
        if (ms) setTimeout( this.unpause.bind(this), ms );
    }
    unpause() { 
        this.pause_depth -= 1
        Logger.log_detail(`${this.name} unpause`)
        if (this.pause_depth<0) {
            Logger.log_error(`${this.name} Over unpausing`)
            this.pause_depth = 0
        }
    this.on_unpause()
    }
    paused() {
        return (this.pause_depth>0)
    }
    on_unpause(){}
}

export function get_connection(node, i, override_type) {
    const graph = node_graph(node)
    const in_link = node?.inputs[i]?.link;
    var type = override_type;
    var link = undefined;
    if (in_link) {
        if (!override_type) type = node.inputs[i].type;
        link = handle_bypass(graph.links[in_link], type, graph);
    } 
    return { link:link, type:type }
}


/*
This is called in various places (node load, creation, link change) to ensure there is exactly one empty input 
*/
export function fix_inputs(node) {
    if (!node.graph) return // node has been deleted prior to the fix
    if (node.properties.ue_properties.fixed_inputs) return

    const empty_inputs = node.inputs.filter((inputslot)=>(inputslot.type=='*'))
    var excess_inputs = empty_inputs.length - 1
    
    if (excess_inputs<0) {
        try {
            node.properties.ue_properties.next_input_index = (node.properties.ue_properties.next_input_index || 10) + 1
            node.addInput(`anything${node.properties.ue_properties.next_input_index}`, "*", {label:"anything"})
            fix_inputs(node)
        } catch (e) {
            Logger.log_error(e)
        }
    } else if (excess_inputs>0) {
        const idx = node.inputs.findIndex((inputslot)=>(inputslot.type=='*'))
        if (idx>=0) {
            try {
                node.removeInput(idx)
                fix_inputs(node)
            } catch (e) {
                Logger.log_error(e)
            }
        } else {
            Logger.log_problem(`Something very odd happened in fix_inputs for ${node.id}`)
        }
    }
}