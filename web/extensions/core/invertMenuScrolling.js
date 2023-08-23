import { app } from "../../scripts/app.js";
import { LiteGraph } from "../../lib/litegraph.core.js"
import { hook } from "../../scripts/utils.js";

// Inverts the scrolling of context menus

const id = "Comfy.InvertMenuScrolling";
app.registerExtension({
	name: id,
	init() {
		hook(LiteGraph, "onContextMenuCreated", (orig, args) => {
			orig?.(...args)
			const contextMenu = args[0];
			contextMenu.options.invert_scrolling = localStorage[`Comfy.Settings.${id}`] === "true";
		})
		app.ui.settings.addSetting({
			id,
			name: "Invert Menu Scrolling",
			type: "boolean",
			defaultValue: false
		});
	},
});
