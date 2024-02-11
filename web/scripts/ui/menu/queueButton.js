// @ts-check

import { ComfyButton } from "../components/button.js";
import { $el } from "../../ui.js";
import { ComfySplitButton } from "../components/splitButton.js";

export class ComfyQueueButton {
	element = $el("div.comfyui-queue-action");

	constructor() {
		const btn = new ComfySplitButton(
			{
				primary: new ComfyButton({
					content: $el("div", [
						$el("span", {
							textContent: "Queue",
						}),
						$el("span.comfyui-queue-count", {
							textContent: "99+",
							title: "186 prompts queued"
						}),
					]),
					icon: "play",
					classList: "comfyui-button",
				}),
				mode: "click",
				position: "absolute",
				horizontal: "right",
			},
			new ComfyButton({
				content: $el("button", { style: { height: "300px", width: "300px" } }),
			})
		);
		btn.element.classList.add("primary");
		this.element.append(btn.element);
	}
}
