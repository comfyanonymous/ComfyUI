import { app } from "../../scripts/app.js";

function node_id_to_show() {
    const nodes_to_show = Object.keys(app.lastNodeErrors ?? {})
    if (nodes_to_show.length > 0)             return nodes_to_show[0]
    else if (app.lastExecutionError?.node_id) return app.lastExecutionError.node_id
    else                                      return null
}

function show_error_node() { app.canvas.centerOnNode(app.graph._nodes_by_id[node_id_to_show()]) }

app.registerExtension({
	name: "comfy.error_zoom",
	setup() {
		const orig = LGraphCanvas.prototype.getCanvasMenuOptions;
		LGraphCanvas.prototype.getCanvasMenuOptions = function () {
            const options = orig.apply(this, arguments);
            if (node_id_to_show()) {
                options.push(null);
                options.push({
                    content: `Show error node`,
                    callback: show_error_node,
                })
            }
            return options;
        }
    }
})