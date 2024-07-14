import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

let double_click_policy = "copy-all";

api.fetchApi('/manager/dbl_click/policy')
	.then(response => response.text())
	.then(data => set_double_click_policy(data));

export function set_double_click_policy(mode) {
	double_click_policy = mode;
}

function addMenuHandler(nodeType, cb) {
	const getOpts = nodeType.prototype.getExtraMenuOptions;
	nodeType.prototype.getExtraMenuOptions = function () {
		const r = getOpts.apply(this, arguments);
		cb.apply(this, arguments);
		return r;
	};
}

function distance(node1, node2) {
	let dx = (node1.pos[0] + node1.size[0]/2) - (node2.pos[0] + node2.size[0]/2);
	let dy = (node1.pos[1] + node1.size[1]/2) - (node2.pos[1] + node2.size[1]/2);
	return Math.sqrt(dx * dx + dy * dy);
}

function lookup_nearest_nodes(node) {
	let nearest_distance = Infinity;
	let nearest_node = null;
	for(let other of app.graph._nodes) {
		if(other === node)
			continue;

		let dist = distance(node, other);
		if (dist < nearest_distance && dist < 1000) {
			nearest_distance = dist;
			nearest_node = other;
		}
	}

	return nearest_node;
}

function lookup_nearest_inputs(node) {
	let input_map = {};

	for(let i in node.inputs) {
		let input = node.inputs[i];

		if(input.link || input_map[input.type])
			continue;

		input_map[input.type] = {distance: Infinity, input_name: input.name, node: null, slot: null};
	}

	let x = node.pos[0];
	let y = node.pos[1] + node.size[1]/2;

	for(let other of app.graph._nodes) {
		if(other === node || !other.outputs)
			continue;

		let dx = x - (other.pos[0] + other.size[0]);
		let dy = y - (other.pos[1] + other.size[1]/2);

		if(dx < 0)
			continue;

		let dist = Math.sqrt(dx * dx + dy * dy);

		for(let input_type in input_map) {
			for(let j in other.outputs) {
				let output = other.outputs[j];
				if(output.type == input_type) {
					if(input_map[input_type].distance > dist) {
						input_map[input_type].distance = dist;
						input_map[input_type].node = other;
						input_map[input_type].slot = parseInt(j);
					}
				}
			}
		}
	}

	let res = {};
	for (let i in input_map) {
		if (input_map[i].node) {
			res[i] = input_map[i];
		}
	}

	return res;
}

function connect_inputs(nearest_inputs, node) {
	for(let i in nearest_inputs) {
		let info = nearest_inputs[i];
		info.node.connect(info.slot, node.id, info.input_name);
	}
}

function node_info_copy(src, dest, connect_both) {
	// copy input connections
	for(let i in src.inputs) {
		let input = src.inputs[i];
		if(input.link) {
			let link = app.graph.links[input.link];
			let src_node = app.graph.getNodeById(link.origin_id);
			src_node.connect(link.origin_slot, dest.id, input.name);
		}
	}

	// copy output connections
	if(connect_both) {
		let output_links = {};
		for(let i in src.outputs) {
			let output = src.outputs[i];
			if(output.links) {
				let links = [];
				for(let j in output.links) {
					links.push(app.graph.links[output.links[j]]);
				}
				output_links[output.name] = links;
			}
		}

		for(let i in dest.outputs) {
			let links = output_links[dest.outputs[i].name];
			if(links) {
				for(let j in links) {
					let link = links[j];
					let target_node = app.graph.getNodeById(link.target_id);
					dest.connect(parseInt(i), target_node, link.target_slot);
				}
			}
		}
	}

	app.graph.afterChange();
}

app.registerExtension({
	name: "Comfy.Manager.NodeFixer",

	async nodeCreated(node, app) {
		let orig_dblClick = node.onDblClick;
		node.onDblClick = function (e, pos, self) {
			orig_dblClick?.apply?.(this, arguments);

			if((!node.inputs && !node.outputs) || pos[1] > 0)
				return;

			switch(double_click_policy) {
				case "copy-all":
				case "copy-input":
					{
						if(node.inputs?.some(x => x.link != null) || node.outputs?.some(x => x.links != null && x.links.length > 0) )
							return;

						let src_node = lookup_nearest_nodes(node);
						if(src_node)
							node_info_copy(src_node, node, double_click_policy == "copy-all");
					}
					break;
				case "possible-input":
					{
						let nearest_inputs = lookup_nearest_inputs(node);
						if(nearest_inputs)
							connect_inputs(nearest_inputs, node);
					}
					break;
				case "dual":
					{
						if(pos[0] < node.size[0]/2) {
							// left: possible-input
							let nearest_inputs = lookup_nearest_inputs(node);
							if(nearest_inputs)
								connect_inputs(nearest_inputs, node);
						}
						else {
							// right: copy-all
							if(node.inputs?.some(x => x.link != null) || node.outputs?.some(x => x.links != null && x.links.length > 0) )
								return;

							let src_node = lookup_nearest_nodes(node);
							if(src_node)
								node_info_copy(src_node, node, true);
						}
					}
					break;
			}
		}
	},

	beforeRegisterNodeDef(nodeType, nodeData, app) {
		addMenuHandler(nodeType, function (_, options) {
			options.push({
				content: "Fix node (recreate)",
				callback: () => {
					let new_node = LiteGraph.createNode(nodeType.comfyClass);
					new_node.pos = [this.pos[0], this.pos[1]];
					app.canvas.graph.add(new_node, false);
					node_info_copy(this, new_node);
					app.canvas.graph.remove(this);
				},
			});
		});
	}
});
