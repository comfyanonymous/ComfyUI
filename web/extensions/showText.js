import { app } from "/scripts/app.js";
import { ComfyWidgets } from "/scripts/widgets.js";

// Displays input text on a node

app.registerExtension({
	name: "pysssss.ShowText",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeData.name === "ShowText") {
			// When the node is created we want to add a readonly text widget to display the text
			const onNodeCreated = nodeType.prototype.onNodeCreated;
			nodeType.prototype.onNodeCreated = function () {
				const r = onNodeCreated?.apply(this, arguments);

				const w = ComfyWidgets["STRING"](this, "text", ["STRING", { multiline: true }], app).widget;
				w.inputEl.readOnly = true;
				w.inputEl.style.opacity = 0.6;

				return r;
			};

			// When the node is executed we will be sent the input text, display this in the widget
			const onExecuted = nodeType.prototype.onExecuted;
			nodeType.prototype.onExecuted = function (message) {
				onExecuted?.apply(this, arguments);

				this.widgets[0].value = message.text;

				if (this.size[1] < 180) {
					this.setSize([this.size[0], 180]);
				}
			};
		}
	},
});