import { app } from "../../../scripts/app.js";
app.registerExtension({
	name: "pysssss.KSamplerAdvDenoise",
	async beforeRegisterNodeDef(nodeType) {
		// Add menu options to conver to/from widgets
		const origGetExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
		nodeType.prototype.getExtraMenuOptions = function (_, options) {
			const r = origGetExtraMenuOptions?.apply?.(this, arguments);

			let stepsWidget = null;
			let startAtWidget = null;
			let endAtWidget = null;
			for (const w of this.widgets || []) {
				if (w.name === "steps") {
					stepsWidget = w;
				} else if (w.name === "start_at_step") {
					startAtWidget = w;
				} else if (w.name === "end_at_step") {
					endAtWidget = w;
				}
			}

			if (stepsWidget && startAtWidget && endAtWidget) {
				options.push(
					{
						content: "Set Denoise",
						callback: () => {
							const steps = +prompt("How many steps do you want?", 15);
							if (isNaN(steps)) {
								return;
							}
							const denoise = +prompt("How much denoise? (0-1)", 0.5);
							if (isNaN(denoise)) {
								return;
							}

							stepsWidget.value = Math.floor(steps / Math.max(0, Math.min(1, denoise)));
							stepsWidget.callback?.(stepsWidget.value);

							startAtWidget.value = stepsWidget.value - steps;
							startAtWidget.callback?.(startAtWidget.value);

							endAtWidget.value = stepsWidget.value;
							endAtWidget.callback?.(endAtWidget.value);
						},
					},
					null
				);
			}

			return r;
		};
	},
});
