// @ts-check

import { ComfyButton } from "../components/button.js";
import { $el } from "../../ui.js";
import { api } from "../../api.js";
import { ComfySplitButton } from "../components/splitButton.js";
import { ComfyQueueOptions } from "./queueOptions.js";
import { prop } from "../../utils.js";

export class ComfyQueueButton {
	element = $el("div.comfyui-queue-button");
	#internalQueueSize = 0;

	queuePrompt = async () => {
		this.#internalQueueSize += this.queueOptions.batchCount;
		await this.app.queuePrompt(-1, this.queueOptions.batchCount);
	};

	constructor(app) {
		this.app = app;
		this.queueSizeElement = $el("span.comfyui-queue-count", {
			textContent: "?",
		});

		const queue = new ComfyButton({
			content: $el("div", [
				$el("span", {
					textContent: "Queue",
				}),
				this.queueSizeElement,
			]),
			icon: "play",
			classList: "comfyui-button",
			action: this.queuePrompt,
		});

		this.queueOptions = new ComfyQueueOptions(app);

		const btn = new ComfySplitButton(
			{
				primary: queue,
				mode: "click",
				position: "absolute",
				horizontal: "right",
			},
			this.queueOptions.element
		);
		btn.element.classList.add("primary");
		this.element.append(btn.element);

		this.autoQueueMode = prop(this, "autoQueueMode", "", () => {
			switch (this.autoQueueMode) {
				case "instant":
					queue.icon = "infinity";
					break;
				case "change":
					queue.icon = "auto-mode";
					break;
				default:
					queue.icon = "play";
					break;
			}
		});

		this.queueOptions.addEventListener("autoQueueMode", (e) => (this.autoQueueMode = e["detail"]));

		api.addEventListener("graphChanged", () => {
			if (this.autoQueueMode === "change") {
				if (this.#internalQueueSize) {
					this.graphHasChanged = true;
				} else {
					this.graphHasChanged = false;
					this.queuePrompt();
				}
			}
		});

		api.addEventListener("status", ({ detail }) => {
			this.#internalQueueSize = detail?.exec_info?.queue_remaining;
			if (this.#internalQueueSize != null) {
				this.queueSizeElement.textContent = this.#internalQueueSize > 99 ? "99+" : this.#internalQueueSize + "";
				this.queueSizeElement.title = `${this.#internalQueueSize} prompts in queue`;
				if (!this.#internalQueueSize && !app.lastExecutionError) {
					if (this.autoQueueMode === "instant" || (this.autoQueueMode === "change" && this.graphHasChanged)) {
						this.graphHasChanged = false;
						this.queuePrompt();
					}
				}
			}
		});
	}
}
