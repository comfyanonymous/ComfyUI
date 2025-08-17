import { app } from "../../../../scripts/app.js";

function links_with(p, node_id, down, up) {
    const links_with = [];
    p.workflow.links.forEach((l) => {
        if (down && l[1]===node_id && !links_with.includes(l[3])) links_with.push(l[3])
        if (up && l[3]===node_id && !links_with.includes(l[1])) links_with.push(l[1])
    });
    return links_with;
}

function _all_v_nodes(p, here_id) {
    /*
    Make a list of all downstream nodes.
    */
    const downstream = [];
    const to_process = [here_id]
    while(to_process.length>0) {
        const id = to_process.pop();
        downstream.push(id);
        to_process.push(
            ...links_with(p,id,true,false).filter((nid)=>{
                return !(downstream.includes(nid) || to_process.includes(nid))
            })
        )
    }

    /*
    Now all upstream nodes from any of the downstream nodes (except us).
    Put us on the result list so we don't flow up through us
    */
    to_process.push(...downstream.filter((n)=>{ return n!=here_id}));
    const back_upstream = [here_id];
    while(to_process.length>0) {
        const id = to_process.pop();
        back_upstream.push(id);
        to_process.push(
            ...links_with(p,id,false,true).filter((nid)=>{
                return !(back_upstream.includes(nid) || to_process.includes(nid))
            })
        )
    }

    const keep = [];
    keep.push(...downstream);
    keep.push(...back_upstream.filter((n)=>{return !keep.includes(n)}));

    console.log(`Nodes to keep: ${keep}`);
    return keep;
}

async function all_v_nodes(here_id) {
    const p = structuredClone(await app.graphToPrompt());
    const all_nodes = [];
    p.workflow.nodes.forEach((node)=>{all_nodes.push(node.id)})
    p.workflow.links = p.workflow.links.filter((l)=>{ return (all_nodes.includes(l[1]) && all_nodes.includes(l[3]))} )
    return _all_v_nodes(p,here_id);
}

async function restart_from_here(here_id, go_down_to_chooser=false) {
    const p = structuredClone(await app.graphToPrompt());
    /*
    Make a list of all nodes, and filter out links that are no longer valid
    */
    const all_nodes = [];
    p.workflow.nodes.forEach((node)=>{all_nodes.push(node.id)})
    p.workflow.links = p.workflow.links.filter((l)=>{ return (all_nodes.includes(l[1]) && all_nodes.includes(l[3]))} )

    /* Move downstream to a chooser */
    if (go_down_to_chooser) {
        while (!app.graph._nodes_by_id[here_id].isChooser) {
            here_id = links_with(p, here_id, true, false)[0];
        }
    }

    const keep = _all_v_nodes(p, here_id);

    /*
    Filter p.workflow.nodes and p.workflow.links
    */
    p.workflow.nodes = p.workflow.nodes.filter((node) => {
        if (node.id===here_id) node.inputs.forEach((i)=>{i.link=null})  // remove our upstream links
        return (keep.includes(node.id))                                 // only keep keepers
    })
    p.workflow.links = p.workflow.links.filter((l) => {return (keep.includes(l[1]) && keep.includes(l[3]))})

    /*
    Filter the p.output object to only include nodes we're keeping
    */
    const new_output = {}
    for (const [key, value] of Object.entries(p.output)) {
        if (keep.includes(parseInt(key))) new_output[key] = value;
    }
    /*
    Filter the p.output entry for the start node to remove any list (ie link) inputs
    */
    const new_inputs = {};
    for (const [key, value] of Object.entries(new_output[here_id.toString()].inputs)) {
        if (!Array.isArray(value)) new_inputs[key] = value;
    }
    new_output[here_id.toString()].inputs = new_inputs;

    p.output = new_output;

    // temporarily hijack graph_to_prompt with a version that restores the old one but returns this prompt
    const gtp_was = app.graphToPrompt;
    app.graphToPrompt = () => {
        app.graphToPrompt = gtp_was;
        return p;
    }
    app.queuePrompt(0);
}

export { restart_from_here, all_v_nodes }