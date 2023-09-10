import { ComfyWidgets, addValueControlWidget } from "../../scripts/widgets.js";
import { app } from "../../scripts/app.js";
import { getWidgetType, forwardOutputValues, applyInputWidgetConversionMenu } from "./utilities.js"

app.registerExtension({
	name: "Comfy.WidgetInputs",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		applyInputWidgetConversionMenu(nodeType, nodeData, app);
	},

	registerCustomNodes() {
		class PrimitiveNode {
			constructor() {
				this.addOutput("connect to widget input", "*");
				this.serialize_widgets = true;
				this.isVirtualNode = true;
			}

			applyToGraph() {
				forwardOutputValues(this, (output) => this.widgets[0].value);
			}

			onConnectionsChange(_, index, connected) {
				if (connected) {
					if (this.outputs[0].links?.length) {
						if (!this.widgets?.length) {
							this.#onFirstConnection();
						}
						if (!this.widgets?.length && this.outputs[0].widget) {
							// On first load it often cant recreate the widget as the other node doesnt exist yet
							// Manually recreate it from the output info
							this.#createWidget(this.outputs[0].widget.config);
						}
					}
				} else if (!this.outputs[0].links?.length) {
					this.#onLastDisconnect();
				}
			}

			onConnectOutput(slot, type, input, target_node, target_slot) {
				// Fires before the link is made allowing us to reject it if it isn't valid

				// No widget, we cant connect
				if (!input.widget) {
					if (!(input.type in ComfyWidgets)) return false;
				}

				if (this.outputs[slot].links?.length) {
					return this.#isValidConnection(input);
				}
			}

			#onFirstConnection() {
				// First connection can fire before the graph is ready on initial load so random things can be missing
				const linkId = this.outputs[0].links[0];
				const link = this.graph.links[linkId];
				if (!link) return;

				const theirNode = this.graph.getNodeById(link.target_id);
				if (!theirNode || !theirNode.inputs) return;

				const input = theirNode.inputs[link.target_slot];
				if (!input) return;


				var _widget;
				if (!input.widget) {
					if (!(input.type in ComfyWidgets)) return;
					_widget = { "name": input.name, "config": [input.type, {}] }//fake widget
				} else {
					_widget = input.widget;
				}

				const widget = _widget;
				const { type, linkType } = getWidgetType(widget.config);
				// Update our output to restrict to the widget type
				this.outputs[0].type = linkType;
				this.outputs[0].name = type;
				this.outputs[0].widget = widget;

				this.#createWidget(widget.config, theirNode, widget.name);
			}

			#createWidget(inputData, node, widgetName) {
				let type = inputData[0];

				if (type instanceof Array) {
					type = "COMBO";
				}

				let widget;
				if (type in ComfyWidgets) {
					widget = (ComfyWidgets[type](this, "value", inputData, app) || {}).widget;
				} else {
					if (type == "combo")
						widget = this.addWidget("combo", "value", null, () => {}, { values: inputData[1]?.values || [] });
					else
						widget = this.addWidget(type, "value", null, () => {}, {});
				}

				if (node?.widgets && widget) {
					const theirWidget = node.widgets.find((w) => w.name === widgetName);
					if (theirWidget) {
						widget.value = theirWidget.value;
					}
				}

				if (widget.type === "number" || widget.type === "combo") {
					addValueControlWidget(this, widget, "fixed");
				}

				// When our value changes, update other widgets to reflect our changes
				// e.g. so LoadImage shows correct image
				const callback = widget.callback;
				const self = this;
				widget.callback = function () {
					const r = callback ? callback.apply(this, arguments) : undefined;
					self.applyToGraph();
					return r;
				};

				// Grow our node if required
				const sz = this.computeSize();
				if (this.size[0] < sz[0]) {
					this.size[0] = sz[0];
				}
				if (this.size[1] < sz[1]) {
					this.size[1] = sz[1];
				}

				requestAnimationFrame(() => {
					if (this.onResize) {
						this.onResize(this.size);
					}
				});
			}

			#isValidConnection(input) {
				// Only allow connections where the configs match
				const config1 = this.outputs[0].widget.config;
				const config2 = input.widget.config;

				if (config1[0] instanceof Array) {
					// These checks shouldnt actually be necessary as the types should match
					// but double checking doesn't hurt

					// New input isnt a combo
					if (!(config2[0] instanceof Array)) return false;
					// New imput combo has a different size
					if (config1[0].length !== config2[0].length) return false;
					// New input combo has different elements
					if (config1[0].find((v, i) => config2[0][i] !== v)) return false;
				} else if (config1[0] !== config2[0]) {
					// Configs dont match
					return false;
				}

				for (const k in config1[1]) {
					if (k !== "default" && k !== 'forceInput') {
						const lhs = config1[1][k];
						const rhs = config2[1][k];
						if (lhs instanceof Array && rhs instanceof Array) {
							if (lhs.length != rhs.length || !lhs.reduce((state, value, index) => state && value === rhs[index], true))
								return false;
						} else if (lhs !== rhs) {
							return false;
						}
					}
				}

				return true;
			}

			#onLastDisconnect() {
				// We cant remove + re-add the output here as if you drag a link over the same link
				// it removes, then re-adds, causing it to break
				this.outputs[0].type = "*";
				this.outputs[0].name = "connect to widget input";
				delete this.outputs[0].widget;

				if (this.widgets) {
					// Allow widgets to cleanup
					for (const w of this.widgets) {
						if (w.onRemove) {
							w.onRemove();
						}
					}
					this.widgets.length = 0;
				}
			}
		}

		LiteGraph.registerNodeType(
			"PrimitiveNode",
			Object.assign(PrimitiveNode, {
				title: "Primitive",
			})
		);
		PrimitiveNode.category = "utils";
	},
});
