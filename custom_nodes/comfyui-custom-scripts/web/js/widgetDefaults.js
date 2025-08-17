import { app } from "../../../scripts/app.js";
import { $el, ComfyDialog } from "../../../scripts/ui.js";

// Allows you to specify custom default values for any widget on any node

const id = "pysssss.WidgetDefaults";
const nodeDataKey = Symbol();

app.registerExtension({
	name: id,
	beforeRegisterNodeDef(nodeType, nodeData) {
		nodeType[nodeDataKey] = nodeData;
	},
	setup() {
		let defaults;
		let regexDefaults;
		let setting;

		const getNodeDefaults = (node, defaults) => {
			const nodeDefaults = defaults[node.type] ?? {};
			const propSetBy = {};

			Object.keys(regexDefaults)
				.filter((r) => new RegExp(r).test(node.type))
				.reduce((p, n) => {
					const props = regexDefaults[n];
					for (const k in props) {
						// Use the longest matching key as its probably the most specific
						if (!(k in nodeDefaults) || (k in propSetBy && n.length > propSetBy[k].length)) {
							propSetBy[k] = n;
							nodeDefaults[k] = props[k];
						}
					}
					return p;
				}, nodeDefaults);

			return nodeDefaults;
		};

		const applyDefaults = (defaults) => {
			for (const node of Object.values(LiteGraph.registered_node_types)) {
				const nodeData = node[nodeDataKey];
				if (!nodeData) continue;
				const nodeDefaults = getNodeDefaults(node, defaults);
				if (!nodeDefaults) continue;
				const inputs = { ...(nodeData.input?.required || {}), ...(nodeData.input?.optional || {}) };

				for (const w in nodeDefaults) {
					const widgetDef = inputs[w];
					if (widgetDef) {
						let v = nodeDefaults[w];
						if (widgetDef[0] === "INT" || widgetDef[0] === "FLOAT") {
							v = +v;
						}
						if (widgetDef[1]) {
							widgetDef[1].default = v;
						} else {
							widgetDef[1] = { default: v };
						}
					}
				}
			}
		};

		const getDefaults = () => {
			let items;
			regexDefaults = {};
			try {
				items = JSON.parse(setting.value);
				items = items.reduce((p, n) => {
					if (n.node.startsWith("/") && n.node.endsWith("/")) {
						const name = n.node.substring(1, n.node.length - 1);
						try {
							// Validate regex
							new RegExp(name);

							if (!regexDefaults[name]) regexDefaults[name] = {};
							regexDefaults[name][n.widget] = n.value;
						} catch (error) {}
					}

					if (!p[n.node]) p[n.node] = {};
					p[n.node][n.widget] = n.value;
					return p;
				}, {});
			} catch (error) {}
			if (!items) {
				items = {};
			}
			applyDefaults(items);
			return items;
		};

		const onNodeAdded = app.graph.onNodeAdded;
		app.graph.onNodeAdded = function (node) {
			onNodeAdded?.apply?.(this, arguments);

			// See if we have any defaults for this type of node
			const nodeDefaults = getNodeDefaults(node.constructor, defaults);
			if (!nodeDefaults) return;

			// Dont run if they are pre-configured nodes from load/pastes
			const stack = new Error().stack;
			if (stack.includes("pasteFromClipboard") || stack.includes("loadGraphData")) {
				return;
			}

			for (const k in nodeDefaults) {
				if (k.startsWith("property.")) {
					const name = k.substring(9);
					let v = nodeDefaults[k];
					// Special handling for some built in values
					if (name in node || ["color", "bgcolor", "title"].includes(name)) {
						node[name] = v;
					} else {
						// Try using the correct type
						if (!node.properties) node.properties = {};
						if (typeof node.properties[name] === "number") v = +v;
						else if (typeof node.properties[name] === "boolean") v = v === "true";
						else if (v === "true") v = true;

						node.properties[name] = v;
					}
				}
			}
		};

		class WidgetDefaultsDialog extends ComfyDialog {
			constructor() {
				super();
				this.element.classList.add("comfy-manage-templates");
				this.grid = $el(
					"div",
					{
						style: {
							display: "grid",
							gridTemplateColumns: "1fr auto auto auto",
							gap: "5px",
						},
						className: "pysssss-widget-defaults",
					},
					[
						$el("label", {
							textContent: "Node Class",
						}),
						$el("label", {
							textContent: "Widget Name",
						}),
						$el("label", {
							textContent: "Default Value",
						}),
						$el("label"),
						(this.rows = $el("div", {
							style: {
								display: "contents",
							},
						})),
					]
				);
			}

			createButtons() {
				const btns = super.createButtons();
				btns[0].textContent = "Cancel";
				btns.unshift(
					$el("button", {
						type: "button",
						textContent: "Add New",
						onclick: () => this.addRow(),
					}),
					$el("button", {
						type: "button",
						textContent: "Save",
						onclick: () => this.save(),
					})
				);
				return btns;
			}

			addRow(node = "", widget = "", value = "") {
				let nameInput;
				this.rows.append(
					$el(
						"div",
						{
							style: {
								display: "contents",
							},
							className: "pysssss-widget-defaults-row",
						},
						[
							$el("input", {
								placeholder: "e.g. CheckpointLoaderSimple",
								value: node,
							}),
							$el("input", {
								placeholder: "e.g. ckpt_name",
								value: widget,
								$: (el) => (nameInput = el),
							}),
							$el("input", {
								placeholder: "e.g. myBestModel.safetensors",
								value,
							}),
							$el("button", {
								textContent: "Delete",
								style: {
									fontSize: "12px",
									color: "red",
									fontWeight: "normal",
								},
								onclick: (e) => {
									nameInput.value = "";
									e.target.parentElement.style.display = "none";
								},
							}),
						]
					)
				);
			}

			save() {
				const rows = this.rows.children;
				const items = [];

				for (const row of rows) {
					const inputs = row.querySelectorAll("input");
					const node = inputs[0].value.trim();
					const widget = inputs[1].value.trim();
					const value = inputs[2].value;
					if (node && widget) {
						items.push({ node, widget, value });
					}
				}

				setting.value = JSON.stringify(items);
				defaults = getDefaults();

				this.close();
			}

			show() {
				this.rows.replaceChildren();
				for (const nodeName in defaults) {
					const node = defaults[nodeName];
					for (const widgetName in node) {
						this.addRow(nodeName, widgetName, node[widgetName]);
					}
				}

				this.addRow();
				super.show(this.grid);
			}
		}

		setting = app.ui.settings.addSetting({
			id,
			name: "🐍 Widget Defaults",
			type: () => {
				return $el("tr", [
					$el("td", [
						$el("label", {
							for: id.replaceAll(".", "-"),
							textContent: "🐍 Widget & Property Defaults:",
						}),
					]),
					$el("td", [
						$el("button", {
							textContent: "Manage",
							onclick: () => {
								try {
									// Try closing old settings window
									if (typeof app.ui.settings.element?.close === "function") {
										app.ui.settings.element.close();
									}
								} catch (error) {}
								try {
									// Try closing new vue dialog
									document.querySelector(".p-dialog-close-button").click();
								} catch (error) {
									// Fallback to just hiding the element
									app.ui.settings.element.style.display = "none";
								}
								const dialog = new WidgetDefaultsDialog();
								dialog.show();
							},
							style: {
								fontSize: "14px",
							},
						}),
					]),
				]);
			},
		});
		defaults = getDefaults();
	},
});
