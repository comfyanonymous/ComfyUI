import { ComfyApp, app } from "../../scripts/app.js";

app.registerExtension({
	name: "Comfy.ToggleMigration",

	nodeCreated(node, app) {
		for(let i in node.widgets) {
			let widget = node.widgets[i];

			if(widget.type == "toggle") {
				Object.defineProperty(widget, "value", {
					set: (value) => {
							delete widget.value;
							widget.value = value == true || value == widget.options.on;
						},
					get: () => { return false; }
				});
			}
		}
	}
});
