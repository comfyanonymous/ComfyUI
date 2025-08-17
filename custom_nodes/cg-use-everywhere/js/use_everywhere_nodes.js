import { handle_bypass, get_real_node, get_group_node } from "./use_everywhere_utilities.js";
import { app } from "../../scripts/app.js";

const CONVERTED_TYPE = "converted-widget";
// import {CONVERTED_TYPE} from "../../extensions/core/widgetInputs.js"

/*
If a widget hasn't been converted, just get it's value
If it has, *try* to go upstream
*/
function get_widget_or_input_values(node_obj, widget_id) {
    if (node_obj.widgets[widget_id]?.type.startsWith(CONVERTED_TYPE)) {
        try {
            const name = node_obj.widgets[widget_id].name;
            const input_id = node_obj.inputs.findIndex((input) => input.name==name);
            const connection = get_connection(node_obj, input_id, "STRING");
            const upstream_node_obj = get_real_node(connection.link.origin_id.toString());
            const widget = upstream_node_obj.widgets.find((w) => w.name.toLowerCase() == upstream_node_obj.outputs[connection.link.origin_slot].name.toLowerCase());
            return widget.value;
        } catch (error) {
            return "NOT CONNECTED DONT MATCH";
        }
    }
    return node_obj.widgets[widget_id].value;
}

function add_ue_from_node_in_group(ues, node, group_node_id, group_data) {
    const group_node = get_real_node(group_node_id);
    const ue_node = group_node.getInnerNodes()[node.index];
    ue_node.in_group_with_data = group_data;
    ue_node.getInnerNodesOfGroup = group_node.getInnerNodes;
    add_ue_from_node(ues, ue_node)
}

function get_available_input_name(inputs, the_input, type) {
    const used_names = [];
    inputs.forEach((input) => { if (input!=the_input) used_names.push(input.name); });
    const base = `UE ${type.toLowerCase()}`;
    if (!used_names.includes(base)) return base;
    for (var i=2; ;i++) {
        if (!used_names.includes(`${base}${i}`)) return `${base}${i}`;
    }
}

function get_connection(node, i, override_type) {
    const in_link = node?.inputs[i]?.link;
    var type = override_type;
    var link = undefined;
    if (in_link) {
        if (!override_type) type = get_real_node(node.id.toString())?.input_type[i];
        link = handle_bypass(app.graph.links[in_link],type);
    } else if (node.in_group_with_data) {
        if (node.in_group_with_data.linksTo[node.index] && node.in_group_with_data.linksTo[node.index][i]) {
            const group_style_link = node.in_group_with_data.linksTo[node.index][i];
            link = { "origin_id":node.getInnerNodesOfGroup()[group_style_link[0]].id, "origin_slot" : group_style_link[1] };
            if (!override_type) type = group_style_link[5];
        } else { // group external input
            const group_node = get_group_node(node.id);
            const group_node_input = group_node.inputs[node.in_group_with_data.oldToNewInputMap[node.index][i]];
            const link_n = group_node_input.link;
            if (link_n) {
                link = app.graph.links[link_n];
                if (!override_type) type = app.graph._nodes_by_id[link.origin_id].outputs[link.origin_slot].type;
                // update the group input node... and the link type
                group_node_input.type = type;
                group_node_input.name = get_available_input_name(group_node.inputs, group_node_input, type);
                link.type = type;
            }
        }            
    }
    return { link:link, type:type }
}

/*
Add UseEverywhere broadcasts from this node to the list
*/
function add_ue_from_node(ues, node) {
    if (node.type === "Seed Everywhere") ues.add_ue(node, -1, "INT", [node.id.toString(),0], 
                                                    undefined, new RegExp("seed|随机种"), undefined, 5);

    if (node.type === "Anything Everywhere?") {
        const connection = get_connection(node, 0);
        if (connection.link) {
            const node_obj = get_real_node(node.id.toString());
            const w0 = get_widget_or_input_values(node_obj,0);
            const r0 = new RegExp(w0);
            const w1 = get_widget_or_input_values(node_obj,1);
            const r1 = (w1.startsWith('+')) ? w1 : new RegExp(w1);
            const w2 = get_widget_or_input_values(node_obj,2);
            const r2 = (w2 && w2!=".*") ? new RegExp(w2) : null;
            ues.add_ue(node, 0, connection.type, [connection.link.origin_id.toString(), connection.link.origin_slot], r0, r1, r2, 10);
        }
    }
    if (node.type === "Prompts Everywhere") {
        for (var i=0; i<2; i++) {
            const connection = get_connection(node, i);
            if (connection.link) ues.add_ue(node, i, connection.type, [connection.link.origin_id.toString(), connection.link.origin_slot], 
                undefined, new RegExp(["(_|\\b)pos(itive|_|\\b)|^prompt|正面","(_|\\b)neg(ative|_|\\b)|负面"][i]), undefined, 5);
        }
    }
    if (node.type === "Anything Everywhere") {
        const connection = get_connection(node, 0);
        if (connection.link) ues.add_ue(node, 0, connection.type, [connection.link.origin_id.toString(),connection. link.origin_slot], undefined, undefined, undefined, 2);
    }
    if (node.type === "Anything Everywhere3") {
        for (var i=0; i<3; i++) {
            const connection = get_connection(node, i);
            if (connection.link) ues.add_ue(node, i, connection.type, [connection.link.origin_id.toString(), connection.link.origin_slot]);
        }
    }
}

export {add_ue_from_node, add_ue_from_node_in_group}
