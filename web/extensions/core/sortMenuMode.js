import { app } from "../../scripts/app.js";

const id = "Comfy.SortMenuMode";
const ext = {
	name: id,
	async setup(app) {
		app.ui.settings.addSetting({
			id,
			name: "Sort Menu",
			type: "combo",
			defaultValue: 0,
			options: [
				{ text: "LiteGraph Default", value: "litegraph" },
				{ text: "True", value: "true" },
				{ text: "False", value: "false" },
			],
			onChange(value) {
				switch (value) {
					case "litegraph": // Default
						if (localStorage.getItem("Comfy.Settings.Comfy.SortMenuMode.defAutoSort") != null) {
							LiteGraph.auto_sort_node_types = JSON.parse(localStorage.getItem("Comfy.Settings.Comfy.SortMenuMode.defAutoSort")); // reset to original;
							localStorage.removeItem("Comfy.Settings.Comfy.SortMenuMode.defAutoSort");
						}
						break;
					
					default:
						if (localStorage.getItem("Comfy.Settings.Comfy.SortMenuMode.defAutoSort") == null) {
							localStorage.setItem(["Comfy.Settings.Comfy.SortMenuMode.defAutoSort"], LiteGraph.auto_sort_node_types);
						}
						LiteGraph.auto_sort_node_types = JSON.parse(value);
				}
			},
		});
	},
};

app.registerExtension(ext);
