import { app } from "../../scripts/app.js";
import { $el } from "../../scripts/ui.js";

// Adds support for tooltips

function getHoveredWidget() {
	if (!app) {
        return;
    }

	const node = app.canvas.node_over;
	if (!node.widgets) return;

	const graphPos = app.canvas.graph_mouse;

	const x = graphPos[0] - node.pos[0];
	const y = graphPos[1] - node.pos[1];

	for (const w of node.widgets) {
		let widgetWidth, widgetHeight;
		if (w.computeSize) {
			const sz = w.computeSize();
			widgetWidth = sz[0];
			widgetHeight = sz[1];
		} else {
			widgetWidth = w.width || node.size[0];
			widgetHeight = LiteGraph.NODE_WIDGET_HEIGHT;
		}

		if (w.last_y !== undefined && x >= 6 && x <= widgetWidth - 12 && y >= w.last_y && y <= w.last_y + widgetHeight) {
			return w;
		}
	}
}

app.registerExtension({
	name: "Comfy.Tooltips",
	setup() {
		const tooltipEl = $el("div.comfy-graph-tooltip", {
			parent: document.body,
		});

		let tooltipTimeout;
		const hideTooltip = () => {
			if (tooltipTimeout) {
				clearTimeout(tooltipTimeout);
			}
			tooltipEl.style.display = "none";
		};
		const showTooltip = (tooltip) => {
			if (tooltipTimeout) {
				clearTimeout(tooltipTimeout);
			}
			if (tooltip) {
				tooltipTimeout = setTimeout(() => {
					tooltipEl.textContent = tooltip;
					tooltipEl.style.display = "block";
					tooltipEl.style.left = app.canvas.mouse[0] + "px";
					tooltipEl.style.top = app.canvas.mouse[1] + "px";
					const rect = tooltipEl.getBoundingClientRect();
					if(rect.right > window.innerWidth) {
						tooltipEl.style.left = (app.canvas.mouse[0] - rect.width) + "px";
					}

					if(rect.top < 0) {
						tooltipEl.style.top = (app.canvas.mouse[1] + rect.height) + "px";
					}
				}, 500);
			}
		};

		const onCanvasPointerMove = function () {
			hideTooltip();
			const node = this.node_over;
			if (!node) return;

			const tooltips = node.constructor.nodeData?.tooltips;
			if (!tooltips) return;

			const inputSlot = this.isOverNodeInput(node, this.graph_mouse[0], this.graph_mouse[1], [0, 0]);
			if (inputSlot !== -1) {
				return showTooltip(tooltips.input?.[node.inputs[inputSlot].name]);
			}

			const outputSlot = this.isOverNodeOutput(node, this.graph_mouse[0], this.graph_mouse[1], [0, 0]);
			if (outputSlot !== -1) {
				return showTooltip(tooltips.output?.[outputSlot]);
			}

			const widget = getHoveredWidget();
			// Dont show for DOM widgets, these use native browser tooltips as we dont get proper mouse events on these
			if (widget && !widget.element) {
				return showTooltip(tooltips.input?.[widget.name]);
			}
		}.bind(app.canvas);

		app.ui.settings.addSetting({
			id: "Comfy.EnableTooltips",
			name: "Enable Tooltips",
			type: "boolean",
			defaultValue: true,
			onChange(value) {
				if (value) {
					LiteGraph.pointerListenerAdd(app.canvasEl, "move", onCanvasPointerMove);
					window.addEventListener("click", hideTooltip);
				} else {
					LiteGraph.pointerListenerRemove(app.canvasEl, "move", onCanvasPointerMove);
					window.removeEventListener("click", hideTooltip);
				}
			},
		});
	},
});
