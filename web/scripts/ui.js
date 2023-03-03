import { api } from "./api.js";

function $el(tag, propsOrChildren, children) {
	const split = tag.split(".");
	const element = document.createElement(split.shift());
	element.classList.add(...split);
	if (propsOrChildren) {
		if (Array.isArray(propsOrChildren)) {
			element.append(...propsOrChildren);
		} else {
			const parent = propsOrChildren.parent;
			delete propsOrChildren.parent;
			const cb = propsOrChildren.$;
			delete propsOrChildren.$;

			Object.assign(element, propsOrChildren);
			if (children) {
				element.append(...children);
			}

			if (parent) {
				parent.append(element);
			}

			if (cb) {
				cb(element);
			}
		}
	}
	return element;
}

class ComfyDialog {
	constructor() {
		this.element = $el("div.comfy-modal", { parent: document.body }, [
			$el("div.comfy-modal-content", [
				$el("p", { $: (p) => (this.textElement = p) }),
				$el("button", {
					type: "button",
					textContent: "CLOSE",
					onclick: () => this.close(),
				}),
			]),
		]);
	}

	close() {
		this.element.style.display = "none";
	}

	show(html) {
		this.textElement.innerHTML = html;
		this.element.style.display = "flex";
	}
}

class ComfyList {
	#type;
	#text;

	constructor(text, type) {
		this.#text = text;
		this.#type = type || text.toLowerCase();
		this.element = $el("div.comfy-list");
		this.element.style.display = "none";
	}

	get visible() {
		return this.element.style.display !== "none";
	}

	async load() {
		const items = await api.getItems(this.#type);
		this.element.replaceChildren(
			...Object.keys(items).flatMap((section) => [
				$el("h4", {
					textContent: section,
				}),
				$el("div.comfy-list-items", [
					...items[section].map((item) => {
						// Allow items to specify a custom remove action (e.g. for interrupt current prompt)
						const removeAction = item.remove || {
							name: "Delete",
							cb: () => api.deleteItem(this.#type, item.prompt[1]),
						};
						return $el("div", { textContent: item.prompt[0] + ": " }, [
							$el("button", {
								textContent: "Load",
								onclick: () => {
									if (item.outputs) {
										app.nodeOutputs = item.outputs;
									}
									app.loadGraphData(item.prompt[3].extra_pnginfo.workflow);
								},
							}),
							$el("button", {
								textContent: removeAction.name,
								onclick: async () => {
									await removeAction.cb();
									await this.update();
								},
							}),
						]);
					}),
				]),
			]),
			$el("div.comfy-list-actions", [
				$el("button", {
					textContent: "Clear " + this.#text,
					onclick: async () => {
						await api.clearItems(this.#type);
						await this.load();
					},
				}),
				$el("button", { textContent: "Refresh", onclick: () => this.load() }),
			])
		);
	}

	async update() {
		if (this.visible) {
			await this.load();
		}
	}

	async show() {
		this.element.style.display = "block";
		this.button.textContent = "Close";

		await this.load();
	}

	hide() {
		this.element.style.display = "none";
		this.button.textContent = "See " + this.#text;
	}

	toggle() {
		if (this.visible) {
			this.hide();
			return false;
		} else {
			this.show();
			return true;
		}
	}
}

export class ComfyUI {
	constructor(app) {
		this.app = app;
		this.dialog = new ComfyDialog();

		this.queue = new ComfyList("Queue");
		this.history = new ComfyList("History");

		api.addEventListener("status", () => {
			this.queue.update();
			this.history.update();
		});

		var input = document.createElement("input");
		input.setAttribute("type", "file");
		input.setAttribute("accept", ".json,image/png");
		input.style.display = "none";
		document.body.appendChild(input);

		input.addEventListener("change", function () {
			var file = input.files[0];
			prompt_file_load(file);
		});


		this.menuContainer = $el("div.comfy-menu", { parent: document.body }, [
			$el("span", { $: (q) => (this.queueSize = q) }),
			$el("button.comfy-queue-btn", { textContent: "Queue Prompt", onclick: () => app.queuePrompt(0) }),
			$el("div.comfy-menu-btns", [
				$el("button", { textContent: "Queue Front", onclick: () => app.queuePrompt(-1) }),
				$el("button", {
					$: (b) => (this.queue.button = b),
					textContent: "View Queue",
					onclick: () => {
						this.history.hide();
						this.queue.toggle();
					},
				}),
				$el("button", {
					$: (b) => (this.history.button = b),
					textContent: "View History",
					onclick: () => {
						this.queue.hide();
						this.history.toggle();
					},
				}),
			]),
			this.queue.element,
			this.history.element,
			$el("button", {
				textContent: "Save",
				onclick: () => {
					const json = JSON.stringify(app.graph.serialize()); // convert the data to a JSON string
					const blob = new Blob([json], { type: "application/json" });
					const url = URL.createObjectURL(blob);
					const a = $el("a", {
						href: url,
						download: "workflow.json",
						style: "display: none",
						parent: document.body,
					});
					a.click();
					setTimeout(function () {
						a.remove();
						window.URL.revokeObjectURL(url);
					}, 0);
				},
			}),
			$el("button", { textContent: "Load", onclick: () => {} }),
			$el("button", { textContent: "Clear", onclick: () => app.graph.clear() }),
			$el("button", { textContent: "Load Default", onclick: () => app.loadGraphData() }),
		]);

		this.setStatus({ exec_info: { queue_remaining: "X" } });
	}

	setStatus(status) {
		this.queueSize.textContent = "Queue size: " + (status ? status.exec_info.queue_remaining : "ERR");
	}
}
