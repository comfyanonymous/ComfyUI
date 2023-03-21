import { api } from "./api.js";

function $el(tag, propsOrChildren, children) {
	const split = tag.split(".");
	const element = document.createElement(split.shift());
	element.classList.add(...split);
	if (!propsOrChildren) return element;
	if (Array.isArray(propsOrChildren)) {
		element.append(...propsOrChildren);
		return element;
	}

	const { parent, style } = propsOrChildren;
	const cb = propsOrChildren.$;
	delete propsOrChildren.parent;
	delete propsOrChildren.$;

	if (style) {
		Object.assign(element.style, style);
		delete propsOrChildren.style;
	}

	Object.assign(element, propsOrChildren);
	if (children) {
		element.append(...children);
	}

	parent?.append(element);
	cb?.(element);

	return element;
}

class ComfyDialog {
	constructor() {
		const p = $el("p", {
			$: (p) => (this.textElement = p)
		});

		const button = $el("button", {
			type: "button",
			textContent: "CLOSE",
			onclick: this.close.bind(this)
		});

		const modalContent = $el("div.comfy-modal-content", [p, button]);
		this.element = $el("div.comfy-modal",
			{ parent: document.body }, [modalContent]
		);
	}

	close() {
		this.element.style.display = "none";
	}

	show(html) {
		this.textElement.innerHTML = html;
		this.element.style.display = "flex";
	}
}

class ComfySettingsDialog extends ComfyDialog {
	constructor() {
		super();
		this.element.classList.add("comfy-settings");
		this.settings = [];
	}

	addSetting({ id, name, type, defaultValue, onChange }) {
		if (!id) {
			throw new Error("Settings must have an ID");
		}
		if (this.settings.find((s) => s.id === id)) {
			throw new Error("Setting IDs must be unique");
		}

		const settingId = "Comfy.Settings." + id;
		const v = localStorage[settingId];

		// JSON.parse(null) -> null
		// If v can be undefined -> JSON.parse(v ?? null)
		let value = JSON.parse(v) ?? defaultValue;

		// Trigger initial setting of value
		onChange?.(value, undefined);

		const setter = (v) => {
			onChange?.(v, value);
			localStorage[settingId] = JSON.stringify(v);
			value = v;
		};

		const createLable = (input) => (
			$el("div", [
				$el("label", { textContent: name || id }, [input])
			])
		);

		const render = () => {
			if (typeof type === "function") {
				return type(name, setter);
			}

			if (type == 'boolean') {
				return createLable(
					$el("input", {
						type: "checkbox",
						checked: !!value,
						oninput: (e) => setter(e.target.checked),
					})
				);
			}

			console.warn("Unsupported setting type, defaulting to text");
			return createLable(
				$el("input", {
					value,
					oninput: (e) => setter(e.target.value),
				})
			);

		};

		this.settings.push({ render });
	}

	show() {
		super.show();
		this.textElement.replaceChildren(...this.settings.map((s) => s.render()));
	}
}

class ComfyList {
	#type;
	#text;
	element;

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

		const processItem = (item) => {
			// Allow items to specify a custom remove action (e.g. for interrupt current prompt)
			const removeAction = item.remove || {
				name: "Delete",
				cb: () => api.deleteItem(this.#type, item.prompt[1]),
			};

			const loadButton = $el("button", {
				textContent: "Load",
				onclick: () => {
					if (item.outputs) {
						this.app.nodeOutputs = item.outputs;
					}
					this.app.loadGraphData(item.prompt[3].extra_pnginfo.workflow);
				}
			});
			const removeButton = $el("button", {
				textContent: removeAction.name,
				onclick: async () => {
					await removeAction.cb();
					await this.update();
				}
			});

			return $el("div", { textContent: item.prompt[0] + ": " }, [loadButton, removeButton]);
		};
		const processSection = (section) => [
			$el("h4", { textContent: section }),
			$el("div.comfy-list-items", [...items[section].map(processItem)]),
		];

		this.element.replaceChildren(
			...Object.keys(items).flatMap(processSection),
			$el("div.comfy-list-actions", [
				$el("button", {
					textContent: "Clear " + this.#text,
					onclick: async () => {
						await api.clearItems(this.#type);
						await this.load();
					},
				}),
				$el("button", { textContent: "Refresh", onclick: this.load.bind(this) }),
			])
		);
	}

	async update() {
		if (!this.visible) return;
		await this.load();
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
		(this.visible ? this.hide : this.show).call(this);
		return !this.visible;
	}
}

export class ComfyUI {
	constructor(app) {
		this.app = app;
		this.dialog = new ComfyDialog();
		this.settings = new ComfySettingsDialog();

		this.batchCount = 1;
		this.lastQueueSize = 0;
		this.queue = new ComfyList("Queue");
		this.history = new ComfyList("History");

		api.addEventListener("status", () => {
			this.queue.update();
			this.history.update();
		});

		this.createMenuContainer();
		this.setStatus({ exec_info: { queue_remaining: "X" } });
	}

	createMenuContainer() {
		const fileInput = $el("input", {
			type: "file",
			accept: ".json,image/png",
			style: { display: "none" },
			parent: document.body,
			onchange: () => this.app.handleFile(fileInput.files[0]),
		});

		this.menuContainer = $el("div.comfy-menu", { parent: document.body }, [
			$el("div", { style: { overflow: "hidden", position: "relative", width: "100%" } }, [
				$el("span", { $: (q) => (this.queueSize = q) }),
				$el("button.comfy-settings-btn", { textContent: "⚙️", onclick: this.settings.show.bind(this.settings) }),
			]),
			$el("button.comfy-queue-btn", { textContent: "Queue Prompt", onclick: () => this.app.queuePrompt(0, this.batchCount) }),
			$el("div", {}, [
				$el("label", { innerHTML: "Extra options" }, [
					$el("input", {
						type: "checkbox",
						onchange: (i) => {
							document.getElementById('extraOptions').style.display = i.srcElement.checked ? "block" : "none";
							this.batchCount = i.srcElement.checked ? document.getElementById('batchCountInputRange').value : 1;
							document.getElementById('autoQueueCheckbox').checked = false;
						}
					})
				])
			]),
			$el("div", { id: "extraOptions", style: { width: "100%", display: "none" } }, [
				$el("label", { innerHTML: "Batch count" }, [
					$el("input", {
						id: "batchCountInputNumber", type: "number", value: this.batchCount, min: "1", style: { width: "35%", "margin-left": "0.4em" },
						oninput: (i) => {
							this.batchCount = i.target.value;
							document.getElementById('batchCountInputRange').value = this.batchCount;
						}
					}),
					$el("input", {
						id: "batchCountInputRange", type: "range", min: "1", max: "100", value: this.batchCount,
						oninput: (i) => {
							this.batchCount = i.srcElement.value;
							document.getElementById('batchCountInputNumber').value = i.srcElement.value;
						}
					}),
					$el("input", {
						id: "autoQueueCheckbox", type: "checkbox", checked: false, title: "automatically queue prompt when the queue size hits 0",
					})
				]),
			]),
			$el("div.comfy-menu-btns", [
				$el("button", { textContent: "Queue Front", onclick: () => this.app.queuePrompt(-1, this.batchCount) }),
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
					const json = JSON.stringify(this.app.graph.serialize(), null, 2); // convert the data to a JSON string
					const blob = new Blob([json], { type: "application/json" });
					const url = URL.createObjectURL(blob);
					const a = $el("a", {
						href: url,
						download: "workflow.json",
						style: { display: "none" },
						parent: document.body,
					});
					a.click();
					setTimeout(function () {
						a.remove();
						window.URL.revokeObjectURL(url);
					}, 0);
				},
			}),
			$el("button", { textContent: "Load", onclick: fileInput.click.bind(fileInput) }),
			$el("button", { textContent: "Clear", onclick: () => this.app.graph.clear() }),
			$el("button", { textContent: "Load Default", onclick: () => this.app.loadGraphData() })
		]);
	}

	setStatus(status) {
		this.queueSize.textContent = "Queue size: " + status?.exec_info?.queue_remaining ?? "ERR";
		if (!status) return;
		if (this.lastQueueSize != 0 && status.exec_info.queue_remaining == 0 && document.getElementById('autoQueueCheckbox').checked) {
			this.app.queuePrompt(0, this.batchCount);
		}
		this.lastQueueSize = status.exec_info.queue_remaining
	}
}
