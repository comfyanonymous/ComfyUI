import {api} from "./api.js";

export function $el(tag, propsOrChildren, children) {
	const split = tag.split(".");
	const element = document.createElement(split.shift());
	if (split.length > 0) {
		element.classList.add(...split);
	}

	if (propsOrChildren) {
		if (Array.isArray(propsOrChildren)) {
			element.append(...propsOrChildren);
		} else {
			const {parent, $: cb, dataset, style} = propsOrChildren;
			delete propsOrChildren.parent;
			delete propsOrChildren.$;
			delete propsOrChildren.dataset;
			delete propsOrChildren.style;

			if (Object.hasOwn(propsOrChildren, "for")) {
				element.setAttribute("for", propsOrChildren.for)
			}

			if (style) {
				Object.assign(element.style, style);
			}

			if (dataset) {
				Object.assign(element.dataset, dataset);
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

export class ComfyDialog {
	constructor() {
		this.element = $el("div.comfy-modal", {parent: document.body}, [
			$el("div.comfy-modal-content", [$el("p", {$: (p) => (this.textElement = p)}), ...this.createButtons()]),
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
			this.textElement.replaceChildren(html);
		}
		this.element.style.display = "flex";
	}
}

class ComfySettingsDialog extends ComfyDialog {
	constructor() {
		super();
		this.element = $el("dialog", {
			id: "comfy-settings-dialog",
			parent: document.body,
		}, [
			$el("table.comfy-modal-content.comfy-table", [
				$el("caption", {textContent: "Settings"}),
				$el("tbody", {$: (tbody) => (this.textElement = tbody)}),
				$el("button", {
					type: "button",
					textContent: "Close",
					style: {
						cursor: "pointer",
					},
					onclick: () => {
						this.element.close();
					},
				}),
			]),
		]);
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

	addSetting({id, name, type, defaultValue, onChange, attrs = {}, tooltip = "", options = undefined}) {
		if (!id) {
			throw new Error("Settings must have an ID");
		}

		if (this.settings.find((s) => s.id === id)) {
			throw new Error(`Setting ${id} of type ${type} must have a unique ID.`);
		}

		const settingId = `Comfy.Settings.${id}`;
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
				value = this.getSettingValue(id, defaultValue);

				let element;
				const htmlID = id.replaceAll(".", "-");

				const labelCell = $el("td", [
					$el("label", {
						for: htmlID,
						classList: [tooltip !== "" ? "comfy-tooltip-indicator" : ""],
						textContent: name,
					})
				]);

				if (typeof type === "function") {
					element = type(name, setter, value, attrs);
				} else {
					switch (type) {
						case "boolean":
							element = $el("tr", [
								labelCell,
								$el("td", [
									$el("input", {
										id: htmlID,
										type: "checkbox",
										checked: value,
										onchange: (event) => {
											const isChecked = event.target.checked;
											if (onChange !== undefined) {
												onChange(isChecked)
											}
											this.setSettingValue(id, isChecked);
										},
									}),
								]),
							])
							break;
						case "number":
							element = $el("tr", [
								labelCell,
								$el("td", [
									$el("input", {
										type,
										value,
										id: htmlID,
										oninput: (e) => {
											setter(e.target.value);
										},
										...attrs
									}),
								]),
							]);
							break;
						case "slider":
							element = $el("tr", [
								labelCell,
								$el("td", [
									$el("div", {
										style: {
											display: "grid",
											gridAutoFlow: "column",
										},
									}, [
										$el("input", {
											...attrs,
											value,
											type: "range",
											oninput: (e) => {
												setter(e.target.value);
												e.target.nextElementSibling.value = e.target.value;
											},
										}),
										$el("input", {
											...attrs,
											value,
											id: htmlID,
											type: "number",
											style: {maxWidth: "4rem"},
											oninput: (e) => {
												setter(e.target.value);
												e.target.previousElementSibling.value = e.target.value;
											},
										}),
									]),
								]),
							]);
							break;
						case "combo":
							element = $el("tr", [
								labelCell,
								$el("td", [
									$el(
										"select",
										{
											oninput: (e) => {
												setter(e.target.value);
											},
										},
										(typeof options === "function" ? options(value) : options || []).map((opt) => {
											if (typeof opt === "string") {
												opt = { text: opt };
											}
											const v = opt.value ?? opt.text;
											return $el("option", {
												value: v,
												textContent: opt.text,
												selected: value + "" === v + "",
											});
										})
									),
								]),
							]);
							break;
						case "text":
						default:
							if (type !== "text") {
								console.warn(`Unsupported setting type '${type}, defaulting to text`);
							}

							element = $el("tr", [
								labelCell,
								$el("td", [
									$el("input", {
										value,
										id: htmlID,
										oninput: (e) => {
											setter(e.target.value);
										},
										...attrs,
									}),
								]),
							]);
							break;
					}
				}
				if (tooltip) {
					element.title = tooltip;
				}

				return element;
			},
		});

		const self = this;
		return {
			get value() {
				return self.getSettingValue(id, defaultValue);
			},
			set value(v) {
				self.setSettingValue(id, v);
			},
		};
	}

	show() {
		this.textElement.replaceChildren(
			$el("tr", {
				style: {display: "none"},
			}, [
				$el("th"),
				$el("th", {style: {width: "33%"}})
			]),
			...this.settings.map((s) => s.render()),
		)
		this.element.showModal();
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
						return $el("div", {textContent: item.prompt[0] + ": "}, [
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
				$el("button", {textContent: "Refresh", onclick: () => this.load()}),
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
		this.button.textContent = "View " + this.#text;
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

		const confirmClear = this.settings.addSetting({
			id: "Comfy.ConfirmClear",
			name: "Require confirmation when clearing workflow",
			type: "boolean",
			defaultValue: true,
		});

		const promptFilename = this.settings.addSetting({
			id: "Comfy.PromptFilename",
			name: "Prompt for filename when saving workflow",
			type: "boolean",
			defaultValue: true,
		});

		/**
		 * file format for preview
		 *
		 * format;quality
		 *
		 * ex)
		 * webp;50 -> webp, quality 50
		 * jpeg;80 -> rgb, jpeg, quality 80
		 *
		 * @type {string}
		 */
		const previewImage = this.settings.addSetting({
			id: "Comfy.PreviewFormat",
			name: "When displaying a preview in the image widget, convert it to a lightweight image, e.g. webp, jpeg, webp;50, etc.",
			type: "text",
			defaultValue: "",
		});

		this.settings.addSetting({
			id: "Comfy.DisableSliders",
			name: "Disable sliders.",
			type: "boolean",
			defaultValue: false,
		});

		const fileInput = $el("input", {
			id: "comfy-file-input",
			type: "file",
			accept: ".json,image/png,.latent,.safetensors",
			style: {display: "none"},
			parent: document.body,
			onchange: () => {
				app.handleFile(fileInput.files[0]);
			},
		});

		this.menuContainer = $el("div.comfy-menu", {parent: document.body}, [
			$el("div.drag-handle", {
				style: {
					overflow: "hidden",
					position: "relative",
					width: "100%",
					cursor: "default"
				}
			}, [
				$el("span.drag-handle"),
				$el("span", {$: (q) => (this.queueSize = q)}),
				$el("button.comfy-settings-btn", {textContent: "⚙️", onclick: () => this.settings.show()}),
			]),
			$el("button.comfy-queue-btn", {
				id: "queue-button",
				textContent: "Queue Prompt",
				onclick: () => app.queuePrompt(0, this.batchCount),
			}),
			$el("div", {}, [
				$el("label", {innerHTML: "Extra options"}, [
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
			$el("div", {id: "extraOptions", style: {width: "100%", display: "none"}}, [
				$el("label", {innerHTML: "Batch count"}, [
					$el("input", {
						id: "batchCountInputNumber",
						type: "number",
						value: this.batchCount,
						min: "1",
						style: {width: "35%", "margin-left": "0.4em"},
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
				$el("button", {
					id: "queue-front-button",
					textContent: "Queue Front",
					onclick: () => app.queuePrompt(-1, this.batchCount)
				}),
				$el("button", {
					$: (b) => (this.queue.button = b),
					id: "comfy-view-queue-button",
					textContent: "View Queue",
					onclick: () => {
						this.history.hide();
						this.queue.toggle();
					},
				}),
				$el("button", {
					$: (b) => (this.history.button = b),
					id: "comfy-view-history-button",
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
				id: "comfy-save-button",
				textContent: "Save",
				onclick: () => {
					let filename = "workflow.json";
					if (promptFilename.value) {
						filename = prompt("Save workflow as:", filename);
						if (!filename) return;
						if (!filename.toLowerCase().endsWith(".json")) {
							filename += ".json";
						}
					}
					const json = JSON.stringify(app.graph.serialize(), null, 2); // convert the data to a JSON string
					const blob = new Blob([json], {type: "application/json"});
					const url = URL.createObjectURL(blob);
					const a = $el("a", {
						href: url,
						download: filename,
						style: {display: "none"},
						parent: document.body,
					});
					a.click();
					setTimeout(function () {
						a.remove();
						window.URL.revokeObjectURL(url);
					}, 0);
				},
			}),
			$el("button", {
				id: "comfy-dev-save-api-button",
				textContent: "Save (API Format)",
				style: {width: "100%", display: "none"},
				onclick: () => {
					let filename = "workflow_api.json";
					if (promptFilename.value) {
						filename = prompt("Save workflow (API) as:", filename);
						if (!filename) return;
						if (!filename.toLowerCase().endsWith(".json")) {
							filename += ".json";
						}
					}
					app.graphToPrompt().then(p=>{
						const json = JSON.stringify(p.output, null, 2); // convert the data to a JSON string
						const blob = new Blob([json], {type: "application/json"});
						const url = URL.createObjectURL(blob);
						const a = $el("a", {
							href: url,
							download: filename,
							style: {display: "none"},
							parent: document.body,
						});
						a.click();
						setTimeout(function () {
							a.remove();
							window.URL.revokeObjectURL(url);
						}, 0);
					});
				},
			}),
			$el("button", {id: "comfy-load-button", textContent: "Load", onclick: () => fileInput.click()}),
			$el("button", {
				id: "comfy-refresh-button",
				textContent: "Refresh",
				onclick: () => app.refreshComboInNodes()
			}),
			$el("button", {id: "comfy-clipspace-button", textContent: "Clipspace", onclick: () => app.openClipspace()}),
			$el("button", {
				id: "comfy-clear-button", textContent: "Clear", onclick: () => {
					if (!confirmClear.value || confirm("Clear workflow?")) {
						app.clean();
						app.graph.clear();
					}
				}
			}),
			$el("button", {
				id: "comfy-load-default-button", textContent: "Load Default", onclick: () => {
					if (!confirmClear.value || confirm("Load default workflow?")) {
						app.loadGraphData()
					}
				}
			}),
		]);

		const devMode = this.settings.addSetting({
			id: "Comfy.DevMode",
			name: "Enable Dev mode Options",
			type: "boolean",
			defaultValue: false,
			onChange: function(value) { document.getElementById("comfy-dev-save-api-button").style.display = value ? "block" : "none"},
		});

		dragElement(this.menuContainer, this.settings);

		this.setStatus({exec_info: {queue_remaining: "X"}});
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
