function workflowToCworkflow(_workflow)
{

    let workflow = JSON.parse(JSON.stringify(_workflow));

    //
    var flow_links = null;
    if ("links" in workflow)
    {
        flow_links = workflow.links.filter(link => link[5] == "FLOW");
        workflow.links = workflow.links.filter(link => link[5] != "FLOW");
    }
    workflow.flow_links = flow_links;

    // id-links map
    let links_map = {};
    workflow.links?.forEach(link => {
        if(link!=null) links_map[link[0]] = link;
    });

    // input & output
    for (const node of workflow.nodes)
    {
        // flow_inputs data
        let flow_inputs = node.inputs.filter(inp => inp.type == "FLOW");
        flow_inputs.forEach(inp => {
            inp.links = [inp.link];
            delete inp.link;
        });
        node.flow_inputs = flow_inputs;

        // inputs data and related links
        let node_inputs = node.inputs;
        for (let idx = node_inputs.length -1; idx >=0; idx--)
        {
            // input is flow
            if (node_inputs[idx].type == 'FLOW')
            {
                // update slot index in the links
                for (let j = idx + 1; j < node_inputs.length; j++)
                {
                    if (node_inputs[j].link != null && node_inputs[j].type != "FLOW")
                    {
                        --links_map[node_inputs[j].link][4];
                    }
                }

                // delete input
                node_inputs.splice(idx, 1);
            }
        }

        // flow_outputs data
        let flow_outputs = node.outputs?.filter(op => op.type == "FLOW");
        flow_outputs?.forEach(op => {
            op.link = op.links == null? null : op.links[0];					// for FLOW output, only one connection is allowed
            delete op.links;
        });
        node.flow_outputs = flow_outputs?flow_outputs:[];

        // output data and related links
        let node_outputs = node.outputs? node.outputs : [];
        for (let idx = node_outputs.length - 1; idx >=0; idx--)
        {
            if (node_outputs[idx].type == 'FLOW')
            {
                // update slot index in the links
                for (let j = idx + 1; j < node_outputs.length; j++)
                {
                    if (node_outputs[j].links != null && node_outputs[j].links.length > 0 
                        && node_outputs[j].type != "FLOW")
                    {
                        for(const link_id of node_outputs[j].links)
                        {
                            --links_map[link_id][2];
                        }
                    }
                }
                // delete input
                node_outputs.splice(idx, 1);
            }
        }
    }

    // support flow control tag
    setWorkflowCompatiblity(workflow, true);

    return workflow;
}



function cworkflowToWorkflow(graphData)
{

    // No nodes exist, just return
    if(graphData.nodes.length==0)
    {
        return graphData;
    }

    if(!isWorkflowCompatible(graphData))
    {
        return graphData;
    }


    // id-link map, id-node map
    let links = {};
    for(const link of graphData.links)
    {
        links[link[0]] = link;
    }
    let nodes = {};
    for(const node of graphData.nodes)
    {
        nodes[node.id] = node;
    }

    // add flow inputs & outputs
    for(const cur_node of graphData.nodes)
    {
        if(!('inputs' in cur_node))
        {
            cur_node.inputs = [];
        }

        // add flow inputs to the front of all the normal inputs
        if(cur_node.inputs == undefined || cur_node.inputs == null)
            cur_node.inputs = [];
        let nb_flow_inputs = cur_node["flow_inputs"].length;
        for (let i = nb_flow_inputs -1; i >= 0; i--)
        {
            let flowin = cur_node["flow_inputs"][i];
            cur_node.inputs.unshift({
                name: flowin.name,
                type: "FLOW",
                shape: LiteGraph.ARROW_SHAPE,
                link: flowin.links==null ? null : (flowin.links.length>0? flowin.links[0]: null)	// multi-inputs not supported
            });
        }
        // add slot_idx in 'links'
        for(let i = nb_flow_inputs; i < cur_node.inputs.length; i++)
        {
            let link_id = cur_node.inputs[i].link;
            if(link_id != null)
            {
                links[link_id][4] += nb_flow_inputs;
            }
        }

        // add flow outputs to the front of all the normal outputs
        if(cur_node.outputs == undefined || cur_node.outputs == null)
            cur_node.outputs = [];
        let nb_flow_outputs = cur_node["flow_outputs"].length;
        for (let i = nb_flow_outputs - 1; i >= 0; i--)
        {
            let flowout = cur_node["flow_outputs"][i];
            cur_node.outputs.unshift({
                name: flowout.name,
                type: "FLOW",
                slot_index: i,
                shape: LiteGraph.ARROW_SHAPE,
                links: flowout.link 
            });
        }
        // add slot_idx in 'links'
        for(let i = nb_flow_outputs; i < cur_node.outputs.length; i++)
        {
            let link_ids = cur_node.outputs[i].links;
            if(link_ids != null && link_ids.length>0)
            {
                link_ids.forEach(linkid => {
                    links[linkid][2] += nb_flow_outputs;
                });
            }
            cur_node.outputs[i].slot_index += nb_flow_outputs;
        }
    }

    if (graphData.flow_links == null)
    {
        return graphData;
    }

    let last_link_id = graphData.last_link_id;
    for(let flow_link of graphData.flow_links)
    {
        let new_link_id = ++last_link_id;
        graphData.links.push([new_link_id, flow_link[1], flow_link[2], flow_link[3], flow_link[4], "FLOW"]);

        // update link id in nodes
        nodes[flow_link[1]].outputs[flow_link[2]].links = [new_link_id];
        nodes[flow_link[3]].inputs[flow_link[4]].link = new_link_id;
    }

    graphData.last_link_id = last_link_id;

    workflowSupportFlowControl(graphData, true);
    setWorkflowCompatiblity(graphData, false);

    return graphData;
}


// is connected by "Anything Everywhere"
function isConnectingUEnode(_node, _prompt_node, _inp_name)
{
    if (
        // inp in inputs not widgets
        "inputs" in _node 
        && _node.inputs.filter((_inp)=> _inp.name == _inp_name && _inp.link == null).length > 0 
        // inp data in prompt is valid
        && "inputs" in _prompt_node && _inp_name in _prompt_node.inputs && _prompt_node.inputs[_inp_name]
    )
    {
        return true;
    }
    else
    {   return false;}
}

/**
 * 
 * @param {*} cworkflow, compatible workflow
 * @param {*} prompt 
 */
function prompt2cprompt(cworkflow, prompt)
{
    let nodes = {};
    for(const node of cworkflow.nodes)
    {
        nodes[node.id] = node;
    }

    // remove 'FLOW's in all inputs
    for (let node_id in prompt)
    {
        let prompt_inps = prompt[node_id].inputs ? prompt[node_id].inputs : {};
        let node = nodes[node_id];
        for(let inp_name in prompt_inps)
        {
            let is_flow_type = node.flow_inputs? 
                                node.flow_inputs.filter((flow_inp)=>flow_inp.name == inp_name).length > 0 : false;

            // remove flow input
            if (is_flow_type)
            {
                delete prompt_inps[inp_name];
            }
            // update slot index
            else{
                let inp = prompt_inps[inp_name];
                if ((prompt[node_id].is_input_linked[inp_name] && inp)
                    || isConnectingUEnode(node, prompt[node_id], inp_name))
                {
                    let ori_id = inp[0];
                    let ori_node = nodes[ori_id];
                    inp[1] -= ori_node.flow_outputs? ori_node.flow_outputs.length : 0;

                    prompt[node_id].is_input_linked[inp_name] = true; // for UEnodes
                }
            }
        }
    }

    return prompt;
}



function getPromptFlow(cworkflow)
{
    // prompt flow datas
    var prompt_flows = {};
    for (const link of cworkflow.flow_links)
    {
        let all_goto = [];
        if (link[1] in prompt_flows)
        {
            all_goto = prompt_flows[link[1]];
        }
        
        let new_goto = link[3].toString();
        let origin_slot_int = link[2];				// output slot in the original node.
        if (all_goto.length <= origin_slot_int)
        {
            all_goto = all_goto.concat(Array(origin_slot_int + 1 - all_goto.length).fill(null));
        }
        all_goto[origin_slot_int] = [new_goto, link[4]];
        prompt_flows[link[1]] = all_goto;
    }
    cworkflow.nodes.forEach(node => {
        if (!(node.id in prompt_flows))
        {
            prompt_flows[node.id] = null;
        }
    });
    return prompt_flows;
}


function isWorkflowCompatible(workflow)
{
    return workflowSupportFlowControl(workflow) && workflow.is_compatible;
}

function setWorkflowCompatiblity(workflow, compatibility)
{
    if (compatibility){
        workflow.support_flow_control = true;
    }
    workflow.is_compatible = compatibility;
}


function workflowSupportFlowControl(workflow)
{
    return workflow.support_flow_control;
}

function setworkflowSupportFlowControl(workflow, support)
{
    workflow.support_flow_control = support;
}



export{workflowToCworkflow, cworkflowToWorkflow, prompt2cprompt, getPromptFlow,
    isWorkflowCompatible, setWorkflowCompatiblity, setworkflowSupportFlowControl, workflowSupportFlowControl};