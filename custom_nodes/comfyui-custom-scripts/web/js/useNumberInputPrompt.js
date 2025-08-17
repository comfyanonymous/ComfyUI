import { app } from "../../../scripts/app.js";

const id = "pysssss.UseNumberInputPrompt";
const ext = {
	name: id,
	async setup(app) {
		const prompt = LGraphCanvas.prototype.prompt;

		const setting = app.ui.settings.addSetting({
			id,
			name: "🐍 Use number input on value entry",
			defaultValue: false,
			type: "boolean",
		});

		LGraphCanvas.prototype.prompt = function () {
			const dialog = prompt.apply(this, arguments);
			if (setting.value && typeof arguments[1] === "number") {
				// If this should be a number then update the imput
				const input = dialog.querySelector("input");
				input.type = "number";

				// Add constraints
				const widget = app.canvas.node_widget?.[1];
				if (widget?.options) {
					for (const prop of ["min", "max", "step"]) {
						if (widget.options[prop]) input[prop] = widget.options[prop];
					}
				}
			}
			return dialog;
		};
	},
};

app.registerExtension(ext);
