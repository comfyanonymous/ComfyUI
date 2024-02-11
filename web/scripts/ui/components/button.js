// @ts-check

import { $el } from "../../ui.js";
import { applyClasses, toggleElement } from "../utils.js";
import { prop } from "../../utils.js";

/**
 * @typedef {{
 *    icon?: string;
 *    iconSize?: number;
 *    content?: string | HTMLElement;
 *    tooltip?: string;
 *    action?: (e: Event, btn: ComfyButton) => void,
 *    classList?: import("../utils.js").ClassList,
 * 	  visibilitySetting?: { id: string, showValue: any },
 * 	  app?: import("../../app.js").ComfyApp
 * }} ComfyButtonProps
 */
export class ComfyButton {
	#over = 0;
	iconElement = $el("i.mdi");
	contentElement = $el("span");
	/**
	 * @type {import("./popup.js").ComfyPopup}
	 */
	popup;

	/**
	 * @param {ComfyButtonProps} opts
	 */
	constructor({ icon, iconSize, content, tooltip, action, classList = "comfyui-button", visibilitySetting, app }) {
		this.element = $el("button", [this.iconElement, this.contentElement]);

		this.icon = prop(this, "icon", icon, toggleElement(this.iconElement, { onShow: this.updateIcon }));
		this.iconSize = prop(this, "iconSize", iconSize, this.updateIcon);
		this.content = prop(
			this,
			"content",
			content,
			toggleElement(this.contentElement, {
				onShow: (el, v) => {
					if (typeof v === "string") {
						el.textContent = v;
					} else {
						el.replaceChildren(v);
					}
				},
			})
		);

		this.tooltip = prop(this, "tooltip", tooltip, (v) => (this.element.title = v ?? ""));
		this.classList = prop(this, "classList", classList, this.updateClasses);
		this.hidden = prop(this, "hidden", false, this.updateClasses);
		this.action = prop(this, "action", action);
		this.element.addEventListener("click", (e) => {
			if (this.popup) {
				// we are either a touch device or triggered by click not hover
				if (!this.#over) {
					this.popup.toggle();
				}
			}
			this.action?.(e, this);
		});

		if (visibilitySetting?.id) {
			const settingUpdated = () => {
				this.hidden = app.ui.settings.getSettingValue(visibilitySetting.id) !== visibilitySetting.showValue;
			};
			app.ui.settings.addEventListener(visibilitySetting.id + ".change", settingUpdated);
			settingUpdated();
		}
	}

	updateIcon = () => (this.iconElement.className = `mdi mdi-${this.icon}${this.iconSize ? " mdi-" + this.iconSize : ""}`);
	updateClasses = () => applyClasses(this.element, this.classList, ...(this.hidden ? ["hidden"] : []));

	/**
	 *
	 * @param { import("./popup.js").ComfyPopup } popup
	 * @param { "click" | "hover" } mode
	 */
	withPopup(popup, mode = "click") {
		this.popup = popup;

		if (mode === "hover") {
			for (const el of [this.element, this.popup.element]) {
				el.addEventListener("mouseenter", () => {
					this.popup.open = !!++this.#over;
				});
				el.addEventListener("mouseleave", () => {
					this.popup.open = !!--this.#over;
				});
			}
		}

		return this;
	}
}
