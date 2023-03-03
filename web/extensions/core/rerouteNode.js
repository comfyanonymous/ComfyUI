import { app } from "../../scripts/app.js";

// Node that allows you to redirect connections for cleaner graphs

app.registerExtension({
	name: "Comfy.RerouteNode",
	registerCustomNodes() {
		class RerouteNode {
			constructor() {
				if (!this.properties) {
					this.properties = {};
				}
				this.properties.showOutputText = RerouteNode.defaultVisibility;

				this.addInput("", "*");
				this.addOutput(this.properties.showOutputText ? "*" : "", "*");
				this.onConnectInput = function (_, type) {
					if (type !== this.outputs[0].type) {
						this.removeOutput(0);
						this.addOutput(this.properties.showOutputText ? type : "", type);
						this.size = this.computeSize();
					}
				};

				this.clone = function () {
					const cloned = RerouteNode.prototype.clone.apply(this);
					cloned.removeOutput(0);
					cloned.addOutput(this.properties.showOutputText ? "*" : "", "*");
					cloned.size = cloned.computeSize();
					return cloned;
				};

				// This node is purely frontend and does not impact the resulting prompt so should not be serialized
				this.isVirtualNode = true;
			}

			getExtraMenuOptions(_, options) {
				options.unshift(
					{
						content: (this.properties.showOutputText ? "Hide" : "Show") + " Type",
						callback: () => {
							this.properties.showOutputText = !this.properties.showOutputText;
							if (this.properties.showOutputText) {
								this.outputs[0].name = this.outputs[0].type;
							} else {
								this.outputs[0].name = "";
							}
							this.size = this.computeSize();
						},
					},
					{
						content: (RerouteNode.defaultVisibility ? "Hide" : "Show") + " Type By Default",
						callback: () => {
							RerouteNode.setDefaultTextVisibility(!RerouteNode.defaultVisibility);
						},
					}
				);
			}

			computeSize() {
				return [
					this.properties.showOutputText && this.outputs && this.outputs.length
						? Math.max(55, LiteGraph.NODE_TEXT_SIZE * this.outputs[0].name.length * 0.6 + 40)
						: 55,
					26,
				];
			}

			static setDefaultTextVisibility(visible) {
				RerouteNode.defaultVisibility = visible;
				if (visible) {
					localStorage["Comfy.RerouteNode.DefaultVisibility"] = "true";
				} else {
					delete localStorage["Comfy.RerouteNode.DefaultVisibility"];
				}
			}
		}

		// Load default visibility
		RerouteNode.setDefaultTextVisibility(!!localStorage["Comfy.RerouteNode.DefaultVisibility"]);

		LiteGraph.registerNodeType(
			"Reroute",
			Object.assign(RerouteNode, {
				title_mode: LiteGraph.NO_TITLE,
				title: "Reroute",
			})
		);

		RerouteNode.category = "utils";
	},
});
