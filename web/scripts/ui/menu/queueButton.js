// @ts-check

import { ComfyButton } from "../components/button.js";
import { $el } from "../../ui.js";
import { api } from "../../api.js";
import { ComfySplitButton } from "../components/splitButton.js";

export class ComfyQueueButton {
	element = $el("div.comfyui-queue-button");

	constructor(app) {
		this.queueSizeElement = $el("span.comfyui-queue-count", {
			textContent: "?",
		});

		const btn = new ComfySplitButton(
			{
				primary: new ComfyButton({
					content: $el("div", [
						$el("span", {
							textContent: "Queue",
						}),
						this.queueSizeElement,
					]),
					icon: "play",
					classList: "comfyui-button",
					action: () => {
						app.queuePrompt(0, 1);
					},
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

		api.addEventListener("status", ({ detail }) => {
			const sz = detail?.exec_info?.queue_remaining;
			if (sz != null) {
				this.queueSizeElement.textContent = sz > 99 ? "99+" : sz;
				this.queueSizeElement.title = `${sz} promps in queue`;
			}
		});
	}
}
