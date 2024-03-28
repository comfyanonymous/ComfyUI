import { api } from "../../scripts/api.js";

function widgetFeedbackHandler(event) {
	try {
		let nodes = app.graph._nodes_by_id;
		let node = nodes[event.detail.node_id];

		if(!node) {
			console.log(`[widgetFeedback] invalid node id '${event.detail.node_id}'`);
			return;
		}

		const w = node.widgets?.find((w) => event.detail.widget_name === w.name);
		if(w) {
			w.value = event.detail.value;
		}
		else {
			console.log(`[widgetFeedback] invalid widget name '${event.detail.widget_name}'`);
		}
	}
	catch(e) {
		console.log(`[widgetFeedback] exception occurs\n${e}`);
	}
}

api.addEventListener("widget-feedback", widgetFeedbackHandler);