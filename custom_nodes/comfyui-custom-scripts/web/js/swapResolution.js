import { app } from "../../../scripts/app.js";
app.registerExtension({
	name: "pysssss.SwapResolution",
	async beforeRegisterNodeDef(nodeType, nodeData) {
		const inputs = { ...nodeData.input?.required, ...nodeData.input?.optional };
		if (inputs.width && inputs.height) {
			const origGetExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
			nodeType.prototype.getExtraMenuOptions = function (_, options) {
				const r = origGetExtraMenuOptions?.apply?.(this, arguments);

				options.push(
					{
						content: "Swap width/height",
						callback: () => {
							const w = this.widgets.find((w) => w.name === "width");
							const h = this.widgets.find((w) => w.name === "height");
							const a = w.value;
							w.value = h.value;
							h.value = a;
							app.graph.setDirtyCanvas(true);
						},
					},
					null
				);

				return r;
			};
		}
	},
});
