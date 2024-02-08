import { app } from "../../scripts/app.js";

let touchZooming;
let touchCount = 0;

app.registerExtension({
	name: "Comfy.SimpleTouchSupport",
	setup() {
		let zoomPos;
		let touchTime;
		let lastTouch;

		function getMultiTouchPos(e) {
			return Math.hypot(e.touches[0].clientX - e.touches[1].clientX, e.touches[0].clientY - e.touches[1].clientY);
		}

		app.canvasEl.addEventListener(
			"touchstart",
			(e) => {
				touchCount++;
				lastTouch = null;
				if (e.touches?.length === 1) {
					// Store start time for press+hold for context menu
					touchTime = new Date();
					lastTouch = e.touches[0];
				} else {
					touchTime = null;
					if (e.touches?.length === 2) {
						// Store center pos for zoom
						zoomPos = getMultiTouchPos(e);
						app.canvas.pointer_is_down = false;
					}
				}
			},
			true
		);

		app.canvasEl.addEventListener("touchend", (e) => {
			touchZooming = false;
			touchCount = e.touches?.length ?? touchCount - 1;
			if (touchTime && !e.touches?.length) {
				if (new Date() - touchTime > 600) {
					try {
						// hack to get litegraph to use this event
						e.constructor = CustomEvent;
					} catch (error) {}
					e.clientX = lastTouch.clientX;
					e.clientY = lastTouch.clientY;

					app.canvas.pointer_is_down = true;
					app.canvas._mousedown_callback(e);
				}
				touchTime = null;
			}
		});

		app.canvasEl.addEventListener(
			"touchmove",
			(e) => {
				touchTime = null;
				if (e.touches?.length === 2) {
					app.canvas.pointer_is_down = false;
					touchZooming = true;
					LiteGraph.closeAllContextMenus();
					app.canvas.search_box?.close();
					const newZoomPos = getMultiTouchPos(e);

					const midX = (e.touches[0].clientX + e.touches[1].clientX) / 2;
					const midY = (e.touches[0].clientY + e.touches[1].clientY) / 2;

					let scale = app.canvas.ds.scale;
					const diff = zoomPos - newZoomPos;
					if (diff > 0.5) {
						scale *= 1 / 1.07;
					} else if (diff < -0.5) {
						scale *= 1.07;
					}
					app.canvas.ds.changeScale(scale, [midX, midY]);
					app.canvas.setDirty(true, true);
					zoomPos = newZoomPos;
				}
			},
			true
		);
	},
});

const processMouseDown = LGraphCanvas.prototype.processMouseDown;
LGraphCanvas.prototype.processMouseDown = function (e) {
	if (touchZooming || touchCount) {
		return;
	}
	return processMouseDown.apply(this, arguments);
};

const processMouseMove = LGraphCanvas.prototype.processMouseMove;
LGraphCanvas.prototype.processMouseMove = function (e) {
	if (touchZooming || touchCount > 1) {
		return;
	}
	return processMouseMove.apply(this, arguments);
};
