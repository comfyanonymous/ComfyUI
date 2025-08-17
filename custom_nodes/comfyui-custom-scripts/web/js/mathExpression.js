import { app } from "../../../scripts/app.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";

app.registerExtension({
	name: "pysssss.MathExpression",
	init() {
		const STRING = ComfyWidgets.STRING;
		ComfyWidgets.STRING = function (node, inputName, inputData) {
			const r = STRING.apply(this, arguments);
			r.widget.dynamicPrompts = inputData?.[1].dynamicPrompts;
			return r;
		};
	},
	beforeRegisterNodeDef(nodeType) {
		if (nodeType.comfyClass === "MathExpression|pysssss") {
			const onDrawForeground = nodeType.prototype.onDrawForeground;

			nodeType.prototype.onNodeCreated = function() {
				// These are typed as any to bypass backend validation
				// update frontend to restrict types
				for(const input of this.inputs) {
					input.type = "INT,FLOAT,IMAGE,LATENT";
				}
			}

			nodeType.prototype.onDrawForeground = function (ctx) {
				const r = onDrawForeground?.apply?.(this, arguments);

				const v = app.nodeOutputs?.[this.id + ""];
				if (!this.flags.collapsed && v) {
					const text = v.value[0] + "";
					ctx.save();
					ctx.font = "bold 12px sans-serif";
					ctx.fillStyle = "dodgerblue";
					const sz = ctx.measureText(text);
					ctx.fillText(text, this.size[0] - sz.width - 5, LiteGraph.NODE_SLOT_HEIGHT * 3);
					ctx.restore();
				}

				return r;
			};
		}
	},
});
