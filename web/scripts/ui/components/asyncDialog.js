import { ComfyDialog } from "../dialog.js";
import { $el } from "../../ui.js";

export class ComfyAsyncDialog extends ComfyDialog {
	#resolve;

	constructor(actions) {
		super(
			"dialog.comfy-dialog.comfyui-dialog",
			actions?.map((opt) => {
				if (typeof opt === "string") {
					opt = { text: opt };
				}
				return $el("button.comfyui-button", {
					type: "button",
					textContent: opt.text,
					onclick: () => this.close(opt.value ?? opt.text),
				});
			})
		);
	}

	show(html) {
		this.element.addEventListener("close", () => {
			this.close();
		});

		super.show(html);

		return new Promise((resolve) => {
			this.#resolve = resolve;
		});
	}

	showModal(html) {
		this.element.addEventListener("close", () => {
			this.close();
		});

		super.show(html);
		this.element.showModal();

		return new Promise((resolve) => {
			this.#resolve = resolve;
		});
	}

	close(result = null) {
		this.#resolve(result);
		this.element.close();
		super.close();
	}

	static async prompt({ title = null, message, actions }) {
		const dialog = new ComfyAsyncDialog(actions);
		const content = [$el("span", message)];
		if (title) {
			content.unshift($el("h3", title));
		}
		const res = await dialog.showModal(content);
		dialog.element.remove();
		return res;
	}
}
