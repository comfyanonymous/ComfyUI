import { ComfyWidgets } from "/scripts/widgets.js";
import { app } from "/scripts/app.js";

class BatchInfo {
    constructor() {
        this.addOutput("iteration", "INT");
        
        let widget = (ComfyWidgets["INT:batch_index"](this, "iteration", ["INT",{}], app) || {}).widget;
        
        this.serialize_widgets = true;
        this.isVirtualNode = true;
    }

    applyToGraph() {
        if (!this.outputs[0].links?.length) return;

        // For each output link copy our value over the original widget value
        for (const l of this.outputs[0].links) {
            const linkInfo = app.graph.links[l];
            const node = this.graph.getNodeById(linkInfo.target_id);
            const input = node.inputs[linkInfo.target_slot];
            const widgetName = input.widget.name;
            if (widgetName) {
                const widget = node.widgets.find((w) => w.name === widgetName);
                if (widget) {
                    widget.value = this.widgets[0].value;
                    if (widget.callback) {
                        widget.callback(widget.value, app.canvas, node, app.canvas.graph_mouse, {});
                    }
                }
            }
        }
    }
}

app.registerExtension({
	name: "Comfy.BatchInfo",
    
    registerCustomNodes() {
        LiteGraph.registerNodeType(
			"BatchInfo",
			Object.assign(BatchInfo, {
				title: "BatchInfo",
			})
		);
		BatchInfo.category = "utils";
    },
});