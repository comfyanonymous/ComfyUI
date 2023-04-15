import { $el } from "./helpers.js";

export class ComfySplitButton {
	constructor(text, default_action, { get_options }) {
		this.button = $el("button", {
			textContent: text,
			onclick: default_action,
		});
		this.arrow = $el("div.comfy-split-arrow", {
			textContent: "▼",
			onclick: (e) => {
				LiteGraph.closeAllContextMenus();
				var menu = new LiteGraph.ContextMenu(
					get_options(),
					{
						event: e,
						scale: 1.3,
					},
					window
				);
				menu.root.classList.add("comfy-split-popup");
			},
		});

		this.element = $el("div.comfy-menu-button.comfy-split-button", {}, [this.button, this.arrow]);
	}
}