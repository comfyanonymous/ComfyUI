// @ts-check

import { $el } from "../../ui.js";
import { prop } from "../../utils.js";

export class ComfyQueueOptions extends EventTarget {
	element = $el("div.comfyui-queue-options");

	constructor(app) {
		super();
		this.app = app;

		this.batchCountInput = $el("input", {
			className: "comfyui-queue-batch-value",
			type: "number",
			min: "1",
			value: "1",
			oninput: () => (this.batchCount = +this.batchCountInput.value),
		});

		this.batchCountRange = $el("input", {
			type: "range",
			min: "1",
			max: "100",
			value: "1",
			oninput: () => (this.batchCount = +this.batchCountRange.value),
		});

		this.element.append(
			$el("div.comfyui-queue-batch", [
				$el(
					"label",
					{
						textContent: "Batch count: ",
					},
					this.batchCountInput
				),
				this.batchCountRange,
			])
		);

		const createOption = (text, value, checked = false) =>
			$el(
				"label",
				{ textContent: text },
				$el("input", {
					type: "radio",
					name: "AutoQueueMode",
					checked,
					value,
					oninput: (e) => (this.autoQueueMode = e.target["value"]),
				})
			);

		this.autoQueueEl = $el("div.comfyui-queue-mode", [
			$el("span", "Auto Queue:"),
			createOption("Disabled", "", true),
			createOption("Instant", "instant"),
			createOption("On Change", "change"),
		]);

		this.element.append(this.autoQueueEl);

		this.batchCount = prop(this, "batchCount", 1, () => {
			this.batchCountInput.value = this.batchCount + "";
			this.batchCountRange.value = this.batchCount + "";
		});

		this.autoQueueMode = prop(this, "autoQueueMode", "Disabled", () => {
			this.dispatchEvent(
				new CustomEvent("autoQueueMode", {
					detail: this.autoQueueMode,
				})
			);
		});
	}
}
