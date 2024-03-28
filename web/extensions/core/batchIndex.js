import { ComfyWidgets } from "/scripts/widgets.js";
import { app } from "/scripts/app.js";

class BatchInfo {
    constructor() {
        this.addOutput("iteration", "FLOAT");
        this.addOutput("batchCount", "FLOAT");
        
        const batchIteration = (ComfyWidgets["INT"](this, "iteration", ["INT",{}], app) || {}).widget;
        batchIteration.disabled = true;

        const batchCount = (ComfyWidgets["INT"](this, "batchCount", ["INT",{}], app) || {}).widget;
        batchCount.disabled = true;

        batchIteration.onInitBatch = (batchSize) => {
            batchIteration.value = 0;
            batchCount.value = batchSize;
        };

        batchIteration.afterQueued = () => {
            batchIteration.value += 1;
        };
    
        this.serialize_widgets = true;
        this.isVirtualNode = true;
    }

    applyToGraph() {
        for (const idx in this.outputs) {
            // For each output link copy our value over the original widget value
            if (this.outputs[idx].links?.length) {
                for (const l of this.outputs[idx].links) {
                    const linkInfo = app.graph.links[l];
                    const node = this.graph.getNodeById(linkInfo.target_id);
                    const input = node.inputs[linkInfo.target_slot];
                    const widgetName = input.widget.name;
                    if (widgetName) {
                        const widget = node.widgets.find((w) => w.name === widgetName);
                        if (widget) {
                            widget.value = this.widgets[idx].value;
                            if (widget.callback) {
                                widget.callback(widget.value, app.canvas, node, app.canvas.graph_mouse, {});
                            }
                        }
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