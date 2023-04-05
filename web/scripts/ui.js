import { api } from "./api.js";

export function $el(tag, propsOrChildren, children) {
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

			if (propsOrChildren.style) {
				Object.assign(element.style, propsOrChildren.style);
				delete propsOrChildren.style;
			}

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

function dragElement(dragEl, settings) {
	var posDiffX = 0,
		posDiffY = 0,
		posStartX = 0,
		posStartY = 0,
		newPosX = 0,
		newPosY = 0;
	if (dragEl.getElementsByClassName("drag-handle")[0]) {
		// if present, the handle is where you move the DIV from:
		dragEl.getElementsByClassName("drag-handle")[0].onmousedown = dragMouseDown;
	} else {
		// otherwise, move the DIV from anywhere inside the DIV:
		dragEl.onmousedown = dragMouseDown;
	}

	// When the element resizes (e.g. view queue) ensure it is still in the windows bounds
	const resizeObserver = new ResizeObserver(() => {
		ensureInBounds();
	}).observe(dragEl);

	function ensureInBounds() {
		if (dragEl.classList.contains("comfy-menu-manual-pos")) {
			newPosX = Math.min(document.body.clientWidth - dragEl.clientWidth, Math.max(0, dragEl.offsetLeft));
			newPosY = Math.min(document.body.clientHeight - dragEl.clientHeight, Math.max(0, dragEl.offsetTop));

			positionElement();
		}
	}

	function positionElement() {
		const halfWidth = document.body.clientWidth / 2;
		const anchorRight = newPosX + dragEl.clientWidth / 2 > halfWidth;

		// set the element's new position:
		if (anchorRight) {
			dragEl.style.left = "unset";
			dragEl.style.right = document.body.clientWidth - newPosX - dragEl.clientWidth + "px";
		} else {
			dragEl.style.left = newPosX + "px";
			dragEl.style.right = "unset";
		}
		
		dragEl.style.top = newPosY + "px";
		dragEl.style.bottom = "unset";

		if (savePos) {
			localStorage.setItem(
				"Comfy.MenuPosition",
				JSON.stringify({
					x: dragEl.offsetLeft,
					y: dragEl.offsetTop,
				})
			);
		}
	}

	function restorePos() {
		let pos = localStorage.getItem("Comfy.MenuPosition");
		if (pos) {
			pos = JSON.parse(pos);
			newPosX = pos.x;
			newPosY = pos.y;
			positionElement();
			ensureInBounds();
		}
	}

	let savePos = undefined;
	settings.addSetting({
		id: "Comfy.MenuPosition",
		name: "Save menu position",
		type: "boolean",
		defaultValue: savePos,
		onChange(value) {
			if (savePos === undefined && value) {
				restorePos();
			}
			savePos = value;
		},
	});

	function dragMouseDown(e) {
		e = e || window.event;
		e.preventDefault();
		// get the mouse cursor position at startup:
		posStartX = e.clientX;
		posStartY = e.clientY;
		document.onmouseup = closeDragElement;
		// call a function whenever the cursor moves:
		document.onmousemove = elementDrag;
	}

	function elementDrag(e) {
		e = e || window.event;
		e.preventDefault();

		dragEl.classList.add("comfy-menu-manual-pos");

		// calculate the new cursor position:
		posDiffX = e.clientX - posStartX;
		posDiffY = e.clientY - posStartY;
		posStartX = e.clientX;
		posStartY = e.clientY;

		newPosX = Math.min(document.body.clientWidth - dragEl.clientWidth, Math.max(0, dragEl.offsetLeft + posDiffX));
		newPosY = Math.min(document.body.clientHeight - dragEl.clientHeight, Math.max(0, dragEl.offsetTop + posDiffY));

		positionElement();
	}

	window.addEventListener("resize", () => {
			ensureInBounds();
	});

	function closeDragElement() {
		// stop moving when mouse button is released:
		document.onmouseup = null;
		document.onmousemove = null;
	}
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

class ComfySettingsDialog extends ComfyDialog {
	constructor() {
		super();
		this.element.classList.add("comfy-settings");
		this.settings = [];
	}

	getSettingValue(id, defaultValue) {
		const settingId = "Comfy.Settings." + id;
		const v = localStorage[settingId];
		return v == null ? defaultValue : JSON.parse(v);
	}

	setSettingValue(id, value) {
		const settingId = "Comfy.Settings." + id;
		localStorage[settingId] = JSON.stringify(value);
	}

	addSetting({ id, name, type, defaultValue, onChange, attrs = {}, tooltip = "", }) {
		if (!id) {
			throw new Error("Settings must have an ID");
		}
		if (this.settings.find((s) => s.id === id)) {
			throw new Error("Setting IDs must be unique");
		}

		const settingId = "Comfy.Settings." + id;
		const v = localStorage[settingId];
		let value = v == null ? defaultValue : JSON.parse(v);

		// Trigger initial setting of value
		if (onChange) {
			onChange(value, undefined);
		}

		this.settings.push({
			render: () => {
				const setter = (v) => {
					if (onChange) {
						onChange(v, value);
					}
					localStorage[settingId] = JSON.stringify(v);
					value = v;
				};

				let element;

				if (typeof type === "function") {
					element = type(name, setter, value, attrs);
				} else {
					switch (type) {
						case "boolean":
							element = $el("div", [
								$el("label", { textContent: name || id }, [
									$el("input", {
										type: "checkbox",
										checked: !!value,
										oninput: (e) => {
											setter(e.target.checked);
										},
										...attrs
									}),
								]),
							]);
							break;
						case "number":
							element = $el("div", [
								$el("label", { textContent: name || id }, [
									$el("input", {
										type,
										value,
										oninput: (e) => {
											setter(e.target.value);
										},
										...attrs
									}),
								]),
							]);
							break;
						default:
							console.warn("Unsupported setting type, defaulting to text");
							element = $el("div", [
								$el("label", { textContent: name || id }, [
									$el("input", {
										value,
										oninput: (e) => {
											setter(e.target.value);
										},
										...attrs
									}),
								]),
							]);
							break;
					}
				}
				if(tooltip) {
					element.title = tooltip;
				}

				return element;
			},
		});
	}

	show() {
		super.show();
		Object.assign(this.textElement.style, {
			display: "flex",
			flexDirection: "column",
			gap: "10px"
		});
		this.textElement.replaceChildren(...this.settings.map((s) => s.render()));
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
									app.loadGraphData(item.prompt[3].extra_pnginfo.workflow);
									if (item.outputs) {
										app.nodeOutputs = item.outputs;
									}
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
		this.settings = new ComfySettingsDialog();

		this.batchCount = 1;
		this.lastQueueSize = 0;
		this.queue = new ComfyList("Queue");
		this.history = new ComfyList("History");

		api.addEventListener("status", () => {
			this.queue.update();
			this.history.update();
		});

		const fileInput = $el("input", {
			type: "file",
			accept: ".json,image/png",
			style: { display: "none" },
			parent: document.body,
			onchange: () => {
				app.handleFile(fileInput.files[0]);
			},
		});

		this.menuContainer = $el("div.comfy-menu", { parent: document.body }, [
			$el("div.drag-handle", { style: { overflow: "hidden", position: "relative", width: "100%", cursor: "default" } }, [
				$el("span.drag-handle"),
				$el("span", { $: (q) => (this.queueSize = q) }),
				$el("button.comfy-settings-btn", { textContent: "⚙️", onclick: () => this.settings.show() }),
			]),
			$el("button.comfy-queue-btn", {
				textContent: "Queue Prompt",
				onclick: () => app.queuePrompt(0, this.batchCount),
			}),
			$el("div", {}, [
				$el("label", { innerHTML: "Extra options" }, [
					$el("input", {
						type: "checkbox",
						onchange: (i) => {
							document.getElementById("extraOptions").style.display = i.srcElement.checked ? "block" : "none";
							this.batchCount = i.srcElement.checked ? document.getElementById("batchCountInputRange").value : 1;
							document.getElementById("autoQueueCheckbox").checked = false;
						},
					}),
				]),
			]),
			$el("div", { id: "extraOptions", style: { width: "100%", display: "none" } }, [
				$el("label", { innerHTML: "Batch count" }, [
					$el("input", {
						id: "batchCountInputNumber",
						type: "number",
						value: this.batchCount,
						min: "1",
						style: { width: "35%", "margin-left": "0.4em" },
						oninput: (i) => {
							this.batchCount = i.target.value;
							document.getElementById("batchCountInputRange").value = this.batchCount;
						},
					}),
					$el("input", {
						id: "batchCountInputRange",
						type: "range",
						min: "1",
						max: "100",
						value: this.batchCount,
						oninput: (i) => {
							this.batchCount = i.srcElement.value;
							document.getElementById("batchCountInputNumber").value = i.srcElement.value;
						},
					}),
					$el("input", {
						id: "autoQueueCheckbox",
						type: "checkbox",
						checked: false,
						title: "automatically queue prompt when the queue size hits 0",
					}),
				]),
			]),
			$el("div.comfy-menu-btns", [
				$el("button", { textContent: "Queue Front", onclick: () => app.queuePrompt(-1, this.batchCount) }),
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
					const json = JSON.stringify(app.graph.serialize(), null, 2); // convert the data to a JSON string
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
			$el("button", { textContent: "Load", onclick: () => fileInput.click() }),
			$el("button", { textContent: "Refresh", onclick: () => app.refreshComboInNodes() }),
			$el("button", { textContent: "Clear", onclick: () => {
				if (!confirm("Are you sure you want to remove all nodes?")) return;
				app.clean();
				app.graph.clear();
			}}),
			$el("button", { textContent: "Load Default", onclick: () => app.loadGraphData() }),
		]);

		dragElement(this.menuContainer, this.settings);

		this.setStatus({ exec_info: { queue_remaining: "X" } });
	}

	setStatus(status) {
		this.queueSize.textContent = "Queue size: " + (status ? status.exec_info.queue_remaining : "ERR");
		if (status) {
			if (
				this.lastQueueSize != 0 &&
				status.exec_info.queue_remaining == 0 &&
				document.getElementById("autoQueueCheckbox").checked
			) {
				app.queuePrompt(0, this.batchCount);
			}
			this.lastQueueSize = status.exec_info.queue_remaining;
		}
	}
}
