import { app } from "../../scripts/app.js";
import { LiteGraph, LGraphCanvas } from "../../lib/litegraph.core.js"

// Shift + drag/resize to snap to grid

app.registerExtension({
	name: "Comfy.SnapToGrid",
	init() {
		// Add setting to control grid size
		app.ui.settings.addSetting({
			id: "Comfy.SnapToGrid.GridSize",
			name: "Grid Size",
			type: "slider",
			attrs: {
				min: 1,
				max: 500,
			},
			tooltip:
				"When dragging and resizing nodes while holding shift they will be aligned to the grid, this controls the size of that grid.",
			defaultValue: LiteGraph.CANVAS_GRID_SIZE,
			onChange(value) {
				LiteGraph.CANVAS_GRID_SIZE = +value;
			},
		});
	},
});
