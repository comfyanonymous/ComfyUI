import { app } from "../../scripts/app.js";

// Inverts the scrolling of context menus

const id = "Comfy.InvertMenuScrolling";
app.registerExtension({
	name: id,
	init() {
		const ctxMenu = LiteGraph.ContextMenu;
		const replace = () => {
			LiteGraph.ContextMenu = function (values, options) {
				options = options || {};
				if (options.scroll_speed) {
					options.scroll_speed *= -1;
				} else {
					options.scroll_speed = -0.1;
				}
				return ctxMenu.call(this, values, options);
			};
			LiteGraph.ContextMenu.prototype = ctxMenu.prototype;
		};
		app.ui.settings.addSetting({
			id,
			name: "Invert Menu Scrolling",
			type: "boolean",
			defaultValue: false,
			onChange(value) {
				if (value) {
					replace();
				} else {
					LiteGraph.ContextMenu = ctxMenu;
				}
			},
		});
	},
});
