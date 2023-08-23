import { app } from "../../scripts/app.js";
import { LiteGraph, LINK_RENDER_MODE_NAMES } from "../../lib/litegraph.core.js"

const id = "Comfy.LinkRenderMode";
const ext = {
	name: id,
	async setup(app) {
		app.ui.settings.addSetting({
			id,
			name: "Link Render Mode",
			defaultValue: 2,
			type: "combo",
			options: LINK_RENDER_MODE_NAMES.map((m, i) => ({
				value: i,
				text: m,
				selected: i == app.canvas.links_render_mode,
			})),
			onChange(value) {
				app.canvas.links_render_mode = +value;
				app.canvas.draw(true, true);
			},
		});
	},
};

app.registerExtension(ext);
