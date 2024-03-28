// @ts-check

import { $el } from "../../ui.js";
import { ComfyButton } from "./button.js";
import { prop } from "../../utils.js";
import { ComfyPopup } from "./popup.js";

export class ComfySplitButton {
	/**
	 *  @param {{
	 * 		primary: ComfyButton,
	 * 		mode?: "hover" | "click",
	 * 		horizontal?: "left" | "right",
	 * 		position?: "relative" | "absolute"
	 *  }} param0
	 *  @param {Array<ComfyButton> | Array<HTMLElement>} items
	 */
	constructor({ primary, mode, horizontal = "left", position = "relative" }, ...items) {
		this.arrow = new ComfyButton({
			icon: "chevron-down",
		});
		this.element = $el("div.comfyui-split-button" + (mode === "hover" ? ".hover" : ""), [
			$el("div.comfyui-split-primary", primary.element),
			$el("div.comfyui-split-arrow", this.arrow.element),
		]);
		this.popup = new ComfyPopup({
			target: this.element,
			container: position === "relative" ? this.element : document.body,
			classList: "comfyui-split-button-popup" + (mode === "hover" ? " hover" : ""),
			closeOnEscape: mode === "click",
			position,
			horizontal,
		});

		this.arrow.withPopup(this.popup, mode);

		this.items = prop(this, "items", items, () => this.update());
	}

	update() {
		this.popup.element.replaceChildren(...this.items.map((b) => b.element ?? b));
	}
}
