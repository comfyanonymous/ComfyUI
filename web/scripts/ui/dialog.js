import { $el } from "../ui.js";

export class ComfyDialog extends EventTarget {
	constructor(type = "div") {
		super();
		this.element = $el(type + ".comfy-modal", { parent: document.body }, [
			$el("div.comfy-modal-content", [$el("p", { $: (p) => (this.textElement = p) }), ...this.createButtons()]),
		]);
	}

	createButtons() {
		return [
			$el("button", {
				type: "button",
				textContent: "Close",
				onclick: () => this.close(),
			}),
		];
	}

	close() {
		this.element.style.display = "none";
	}

	show(html) {
		if (typeof html === "string") {
			this.textElement.innerHTML = html;
		} else {
			this.textElement.replaceChildren(...(html instanceof Array ? html : [html]));
		}
		this.element.style.display = "flex";
	}
}
