import { app } from "../../../scripts/app.js";

let setting;
const id = "pysssss.SnapToGrid";

/** Wraps the provided function call to set/reset shiftDown when setting is enabled. */
function wrapCallInSettingCheck(fn) {
	if (setting?.value) {
		const shift = app.shiftDown;
		app.shiftDown = true;
		const r = fn();
		app.shiftDown = shift;
		return r;
	}
	return fn();
}

const ext = {
	name: id,
	init() {
		setting = app.ui.settings.addSetting({
			id,
			name: "🐍 Always snap to grid",
			defaultValue: false,
			type: "boolean",
			onChange(value) {
				app.canvas.align_to_grid = value;
			},
		});

		// We need to register our hooks after the core snap to grid extension runs
		// Do this from the graph configure function so we still get onNodeAdded calls
		const configure = LGraph.prototype.configure;
		LGraph.prototype.configure = function () {
			// Override drawNode to draw the drop position
			const drawNode = LGraphCanvas.prototype.drawNode;
			LGraphCanvas.prototype.drawNode = function () {
				wrapCallInSettingCheck(() => drawNode.apply(this, arguments));
			};

			// Override node added to add a resize handler to force grid alignment
			const onNodeAdded = app.graph.onNodeAdded;
			app.graph.onNodeAdded = function (node) {
				const r = onNodeAdded?.apply(this, arguments);
				const onResize = node.onResize;
				node.onResize = function () {
					wrapCallInSettingCheck(() => onResize?.apply(this, arguments));
				};
				return r;
			};


			const groupMove = LGraphGroup.prototype.move;
			LGraphGroup.prototype.move = function(deltax, deltay, ignore_nodes) {
				wrapCallInSettingCheck(() => groupMove.apply(this, arguments));
			}

			const canvasDrawGroups = LGraphCanvas.prototype.drawGroups;
			LGraphCanvas.prototype.drawGroups = function (canvas, ctx) {
				wrapCallInSettingCheck(() => canvasDrawGroups.apply(this, arguments));
			}

			const canvasOnGroupAdd = LGraphCanvas.onGroupAdd;
			LGraphCanvas.onGroupAdd = function() {
				wrapCallInSettingCheck(() => canvasOnGroupAdd.apply(this, arguments));
			}

			return configure.apply(this, arguments);
		};
	},
};

app.registerExtension(ext);
