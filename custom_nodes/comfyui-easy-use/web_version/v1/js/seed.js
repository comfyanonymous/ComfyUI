import { api } from "../../../scripts/api.js";

// 全局Seed
function globalSeedHandler(event) {
	let nodes = app.graph._nodes_by_id;
	for(let i in nodes) {
	    let node = nodes[i];
	    if(node.type == 'easy globalSeed') {
	        if(node.widgets) {
			    const w = node.widgets.find((w) => w.name == 'value');
			    const last_w = node.widgets.find((w) => w.name == 'last_seed');
			    last_w.value = w.value;
			    w.value = event.detail.value;
	        }
	    }
        else{
			if(node.widgets) {
                const w = node.widgets.find((w) => w.name == 'seed_num' || w.name == 'seed' || w.name == 'noise_seed');
				if(w && event.detail.seed_map[node.id] != undefined) {
                   w.value = event.detail.seed_map[node.id];
                }
            }
		}

	}
}

api.addEventListener("easyuse-global-seed", globalSeedHandler);

const original_queuePrompt = api.queuePrompt;
async function queuePrompt_with_seed(number, { output, workflow }) {
	workflow.seed_widgets = {};

	for(let i in app.graph._nodes_by_id) {
		let widgets = app.graph._nodes_by_id[i].widgets;
		if(widgets) {
		    for(let j in widgets) {
		        if((widgets[j].name == 'seed_num' || widgets[j].name == 'seed' || widgets[j].name == 'noise_seed') && widgets[j].type != 'converted-widget')
		            workflow.seed_widgets[i] = parseInt(j);
		    }
        }
	}

	return await original_queuePrompt.call(api, number, { output, workflow });
}

api.queuePrompt = queuePrompt_with_seed;
