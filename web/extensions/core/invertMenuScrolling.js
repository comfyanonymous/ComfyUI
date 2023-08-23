import { app } from "../../scripts/app.js";
import { LiteGraph } from "../../lib/litegraph.core.js"
import { hook } from "../../scripts/utils.js";

// Inverts the scrolling of context menus

const id = "Comfy.InvertMenuScrolling";
app.registerExtension({
	name: id,
	init() {
		let invert = false;
		hook(LiteGraph, "onContextMenuCreated", (orig, contextMenu) => {
			orig?.(contextMenu);
			contextMenu.invert_scrolling = invert;
		})
		app.ui.settings.addSetting({
			id,
			name: "Invert Menu Scrolling",
			type: "boolean",
			defaultValue: false,
			onChange(value) {
				invert = value;
			},
		});
	},
});
