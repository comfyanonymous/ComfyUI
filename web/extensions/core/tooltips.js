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
		let idleTimeout;

		const hideTooltip = () => {
			tooltipEl.style.display = "none";
		};
		const showTooltip = (tooltip) => {
			if (!tooltip) return;

			tooltipEl.textContent = tooltip;
			tooltipEl.style.display = "block";
			tooltipEl.style.left = app.canvas.mouse[0] + "px";
			tooltipEl.style.top = app.canvas.mouse[1] + "px";
			const rect = tooltipEl.getBoundingClientRect();
			if (rect.right > window.innerWidth) {
				tooltipEl.style.left = app.canvas.mouse[0] - rect.width + "px";
			}

			if (rect.top < 0) {
				tooltipEl.style.top = app.canvas.mouse[1] + rect.height + "px";
			}
		};
		const getInputTooltip = (nodeData, name) => {
			const inputDef = nodeData.input?.required?.[name] ?? nodeData.input?.optional?.[name];
			return inputDef?.[1]?.tooltip;
		};
		const onIdle = () => {
			const { canvas } = app;
			const node = canvas.node_over;
			if (!node) return;

			const nodeData = node.constructor.nodeData ?? {};

			if (node.constructor.title_mode !== LiteGraph.NO_TITLE && canvas.graph_mouse[1] < node.pos[1]) {
				return showTooltip(nodeData.description);
			}

			if (node.flags?.collapsed) return;

			const inputSlot = canvas.isOverNodeInput(node, canvas.graph_mouse[0], canvas.graph_mouse[1], [0, 0]);
			if (inputSlot !== -1) {
				const inputName = node.inputs[inputSlot].name;
				return showTooltip(getInputTooltip(nodeData, inputName));
			}

			const outputSlot = canvas.isOverNodeOutput(node, canvas.graph_mouse[0], canvas.graph_mouse[1], [0, 0]);
			if (outputSlot !== -1) {
				return showTooltip(nodeData.output_tooltips?.[outputSlot]);
			}

			const widget = getHoveredWidget();
			// Dont show for DOM widgets, these use native browser tooltips as we dont get proper mouse events on these
			if (widget && !widget.element) {
				return showTooltip(widget.tooltip ?? getInputTooltip(nodeData, widget.name));
			}
		};

		const onMouseMove = (e) => {
			hideTooltip();
			clearTimeout(idleTimeout);
			
			if(e.target.nodeName !== "CANVAS") return
			idleTimeout = setTimeout(onIdle, 500);
		};

		app.ui.settings.addSetting({
			id: "Comfy.EnableTooltips",
			name: "Enable Tooltips",
			type: "boolean",
			defaultValue: true,
			onChange(value) {
				if (value) {
					window.addEventListener("mousemove", onMouseMove);
					window.addEventListener("click", hideTooltip);
				} else {
					window.removeEventListener("mousemove", onMouseMove);
					window.removeEventListener("click", hideTooltip);
				}
			},
		});
	},
});
