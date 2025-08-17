import { app } from "../../../scripts/app.js";
import { $el } from "../../../scripts/ui.js";

let guide_config;
const id = "pysssss.SnapToGrid.Guide";
const guide_config_default = {
	lines: {
		enabled: false,
		fillStyle: "rgba(255, 0, 0, 0.5)",
	},
	block: {
		enabled: false,
		fillStyle: "rgba(0, 0, 255, 0.5)",
	},
}

const ext = {
	name: id,
	init() {
		if (localStorage.getItem(id) === null) {
			localStorage.setItem(id, JSON.stringify(guide_config_default));
		}
		guide_config = JSON.parse(localStorage.getItem(id));

		app.ui.settings.addSetting({
			id,
			name: "🐍 Display drag-and-drop guides",
			type: (name, setter, value) => {
				return $el("tr", [
					$el("td", [
						$el("label", {
							for: id.replaceAll(".", "-"),
							textContent: name,
						}),
					]),
					$el("td", [
						$el(
							"label",
							{
								textContent: "Lines: ",
								style: {
									display: "inline-block",
								},
							},
							[
								$el("input", {
									id: id.replaceAll(".", "-") + "-line-text",
									type: "text",
									value: guide_config.lines.fillStyle,
									onchange: (event) => {
										guide_config.lines.fillStyle = event.target.value;
										localStorage.setItem(id, JSON.stringify(guide_config));
									}
								}),
								$el("input", {
									id: id.replaceAll(".", "-") + "-line-checkbox",
									type: "checkbox",
									checked: guide_config.lines.enabled,
									onchange: (event) => {
										guide_config.lines.enabled = !!event.target.checked;
										localStorage.setItem(id, JSON.stringify(guide_config));
									},
								}),
							]
						),
						$el(
							"label",
							{
								textContent: "Block: ",
								style: {
									display: "inline-block",
								},
							},
							[
								$el("input", {
									id: id.replaceAll(".", "-") + "-block-text",
									type: "text",
									value: guide_config.block.fillStyle,
									onchange: (event) => {
										guide_config.block.fillStyle = event.target.value;
										localStorage.setItem(id, JSON.stringify(guide_config));
									}
								}),
								$el("input", {
									id: id.replaceAll(".", "-") + '-block-checkbox',
									type: "checkbox",
									checked: guide_config.block.enabled,
									onchange: (event) => {
										guide_config.block.enabled = !!event.target.checked;
										localStorage.setItem(id, JSON.stringify(guide_config));
									},
								}),
							]
						),
					]),
				]);
			}
		});

		const alwaysSnapToGrid = () =>
			app.ui.settings.getSettingValue("pysssss.SnapToGrid", /* default=*/ false);
		const snapToGridEnabled = () =>
			app.shiftDown || alwaysSnapToGrid();

		// Override drag-and-drop behavior to show orthogonal guide lines around selected node(s) and preview of where the node(s) will be placed
		const origDrawNode = LGraphCanvas.prototype.drawNode;
		LGraphCanvas.prototype.drawNode = function (node, ctx) {
			const enabled = guide_config.lines.enabled || guide_config.block.enabled;
			if (enabled && this.node_dragged && node.id in this.selected_nodes && snapToGridEnabled()) {
				// discretize the canvas into grid
				let x = LiteGraph.CANVAS_GRID_SIZE * Math.round(node.pos[0] / LiteGraph.CANVAS_GRID_SIZE);
				let y = LiteGraph.CANVAS_GRID_SIZE * Math.round(node.pos[1] / LiteGraph.CANVAS_GRID_SIZE);

				// calculate the width and height of the node
				// (also need to shift the y position of the node, depending on whether the title is visible)
				x -= node.pos[0];
				y -= node.pos[1];
				let w, h;
				if (node.flags.collapsed) {
					w = node._collapsed_width;
					h = LiteGraph.NODE_TITLE_HEIGHT;
					y -= LiteGraph.NODE_TITLE_HEIGHT;
				} else {
					w = node.size[0];
					h = node.size[1];
					let titleMode = node.constructor.title_mode;
					if (titleMode !== LiteGraph.TRANSPARENT_TITLE && titleMode !== LiteGraph.NO_TITLE) {
						h += LiteGraph.NODE_TITLE_HEIGHT;
						y -= LiteGraph.NODE_TITLE_HEIGHT;
					}
				}

				// save the original fill style
				const f = ctx.fillStyle;

				// draw preview for drag-and-drop (rectangle to show where the node will be placed)
				if (guide_config.block.enabled) {
					ctx.fillStyle = guide_config.block.fillStyle;
					ctx.fillRect(x, y, w, h);
				}

				// add guide lines around node (arbitrarily long enough to span most workflows)
				if (guide_config.lines.enabled) {
					const xd = 10000;
					const yd = 10000;
					const thickness = 3;
					ctx.fillStyle = guide_config.lines.fillStyle;
					ctx.fillRect(x - xd, y, 2*xd, thickness);
					ctx.fillRect(x, y - yd, thickness, 2*yd);
					ctx.fillRect(x - xd, y + h, 2*xd, thickness);
					ctx.fillRect(x + w, y - yd, thickness, 2*yd);
				}

				// restore the original fill style
				ctx.fillStyle = f;
			}

			return origDrawNode.apply(this, arguments);
		};
	},
};

app.registerExtension(ext);
