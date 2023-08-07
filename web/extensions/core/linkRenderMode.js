import { app } from "../../scripts/app.js";

const id = "Comfy.LinkRenderMode";
const ext = {
	name: id,
	async setup(app) {
		app.ui.settings.addSetting({
			id,
			name: "Link Render Mode",
			defaultValue: 2,
			type: "combo",
			options: LiteGraph.LINK_RENDER_MODES.map((m, i) => ({
				value: i,
				text: m,
				selected: i == app.canvas.links_render_mode,
			})),
			onChange(value) {
				app.canvas.links_render_mode = +value;
				app.graph.setDirtyCanvas(true);
			},
		});
	},
};

app.registerExtension(ext);
