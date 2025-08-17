import { app } from "../../../scripts/app.js";

// Allows you to manage preset tags for e.g. common negative prompt
// Also performs replacements on any text field e.g. allowing you to use preset text in CLIP Text encode fields

let replaceRegex;
const id = "pysssss.PresetText.Presets";
const MISSING = Symbol();

const getPresets = () => {
	let items;
	try {
		items = JSON.parse(localStorage.getItem(id));
	} catch (error) {}
	if (!items || !items.length) {
		items = [{ name: "default negative", value: "worst quality" }];
	}
	return items;
};

let presets = getPresets();

app.registerExtension({
	name: "pysssss.PresetText",
	setup() {
		app.ui.settings.addSetting({
			id: "pysssss.PresetText.ReplacementRegex",
			name: "🐍 Preset Text Replacement Regex",
			type: "text",
			defaultValue: "(?:^|[^\\w])(?<replace>@(?<id>[\\w-]+))",
			tooltip:
				"The regex should return two named capture groups: id (the name of the preset text to use), replace (the matched text to replace)",
			attrs: {
				style: {
					fontFamily: "monospace",
				},
			},
			onChange(value) {
				if (!value) {
					replaceRegex = null;
					return;
				}
				try {
					replaceRegex = new RegExp(value, "g");
				} catch (error) {
					alert("Error creating regex for preset text replacement, no replacements will be performed.");
					replaceRegex = null;
				}
			},
		});

		const drawNodeWidgets = LGraphCanvas.prototype.drawNodeWidgets
		LGraphCanvas.prototype.drawNodeWidgets = function(node) {
			const c = LiteGraph.WIDGET_BGCOLOR;
			try {
				if(node[MISSING]) {
					LiteGraph.WIDGET_BGCOLOR = "red"
				}
				return drawNodeWidgets.apply(this, arguments);
			} finally {
				LiteGraph.WIDGET_BGCOLOR = c;
			}
		}
	},
	registerCustomNodes() {
		class PresetTextNode extends LiteGraph.LGraphNode {
			constructor() {
				super();
				this.title = "Preset Text 🐍";
				this.isVirtualNode = true;
				this.serialize_widgets = true;
				this.addOutput("text", "STRING");

				const widget = this.addWidget("combo", "value", presets[0].name, () => {}, {
					values: presets.map((p) => p.name),
				});
				this.addWidget("button", "Manage", "Manage", () => {
					const container = document.createElement("div");
					Object.assign(container.style, {
						display: "grid",
						gridTemplateColumns: "1fr 1fr",
						gap: "10px",
					});

					const addNew = document.createElement("button");
					addNew.textContent = "Add New";
					addNew.classList.add("pysssss-presettext-addnew");
					Object.assign(addNew.style, {
						fontSize: "13px",
						gridColumn: "1 / 3",
						color: "dodgerblue",
						width: "auto",
						textAlign: "center",
					});
					addNew.onclick = () => {
						addRow({ name: "", value: "" });
					};
					container.append(addNew);

					function addRow(p) {
						const name = document.createElement("input");
						const nameLbl = document.createElement("label");
						name.value = p.name;
						nameLbl.textContent = "Name:";
						nameLbl.append(name);

						const value = document.createElement("input");
						const valueLbl = document.createElement("label");
						value.value = p.value;
						valueLbl.textContent = "Value:";
						valueLbl.append(value);

						addNew.before(nameLbl, valueLbl);
					}
					for (const p of presets) {
						addRow(p);
					}

					const help = document.createElement("span");
					help.textContent = "To remove a preset set the name or value to blank";
					help.style.gridColumn = "1 / 3";
					container.append(help);

					dialog.show("");
					dialog.textElement.append(container);
				});

				const dialog = new app.ui.dialog.constructor();
				dialog.element.classList.add("comfy-settings");

				const closeButton = dialog.element.querySelector("button");
				closeButton.textContent = "CANCEL";
				const saveButton = document.createElement("button");
				saveButton.textContent = "SAVE";
				saveButton.onclick = function () {
					const inputs = dialog.element.querySelectorAll("input");
					const p = [];
					for (let i = 0; i < inputs.length; i += 2) {
						const n = inputs[i];
						const v = inputs[i + 1];
						if (!n.value.trim() || !v.value.trim()) {
							continue;
						}
						p.push({ name: n.value, value: v.value });
					}

					widget.options.values = p.map((p) => p.name);
					if (!widget.options.values.includes(widget.value)) {
						widget.value = widget.options.values[0];
					}

					presets = p;
					localStorage.setItem(id, JSON.stringify(presets));

					dialog.close();
				};

				closeButton.before(saveButton);

				this.applyToGraph = function (workflow) {
					// For each output link copy our value over the original widget value
					if (this.outputs[0].links && this.outputs[0].links.length) {
						for (const l of this.outputs[0].links) {
							const link_info = app.graph.links[l];
							const outNode = app.graph.getNodeById(link_info.target_id);
							const outIn = outNode && outNode.inputs && outNode.inputs[link_info.target_slot];
							if (outIn.widget) {
								const w = outNode.widgets.find((w) => w.name === outIn.widget.name);
								if (!w) continue;
								const preset = presets.find((p) => p.name === widget.value);
								if (!preset) {
									this[MISSING] = true;
									app.graph.setDirtyCanvas(true, true);
									const msg = `Preset text '${widget.value}' not found. Please fix this and queue again.`;
									throw new Error(msg);
								}
								delete this[MISSING];
								w.value = preset.value;
							}
						}
					}
				};
			}
		}

		LiteGraph.registerNodeType(
			"PresetText|pysssss",
			Object.assign(PresetTextNode, {
				title: "Preset Text 🐍",
			})
		);

		PresetTextNode.category = "utils";
	},
	nodeCreated(node) {
		if (node.widgets) {
			// Locate dynamic prompt text widgets
			const widgets = node.widgets.filter((n) => n.type === "customtext" || n.type === "text");
			for (const widget of widgets) {
				const callbacks = [
					() => {
						let prompt = widget.value;
						if (replaceRegex && typeof prompt.replace !== 'undefined') {
							prompt = prompt.replace(replaceRegex, (match, p1, p2, index, text, groups) => {
								if (!groups.replace || !groups.id) return match; // No match, bad regex?

								const preset = presets.find((p) => p.name.replaceAll(/\s/g, "-") === groups.id);
								if (!preset) return match; // Invalid name

								const pos = match.indexOf(groups.replace);
								return match.substring(0, pos) + preset.value;
							});
						}
						return prompt;
					},
				];
				let inheritedSerializeValue = widget.serializeValue || null;

				let called = false;
				const serializeValue = async (workflowNode, widgetIndex) => {
					const origWidgetValue = widget.value;
					if (called) return origWidgetValue;
					called = true;

					let allCallbacks = [...callbacks];
					if (inheritedSerializeValue) {
						allCallbacks.push(inheritedSerializeValue)
					}
					let valueIsUndefined = false;

					for (const cb of allCallbacks) {
						let value = await cb(workflowNode, widgetIndex);
						// Need to check the callback return value before it is set on widget.value as it coerces it to a string (even for undefined)
						if (value === undefined) valueIsUndefined = true;
						widget.value = value;
					}

					const prompt = valueIsUndefined ? undefined : widget.value;
					widget.value = origWidgetValue;

					called = false;

					return prompt;
				};

				Object.defineProperty(widget, "serializeValue", {
					get() {
						return serializeValue;
					},
					set(cb) {
						inheritedSerializeValue = cb;
					},
				});
			}
		}
	},
});
