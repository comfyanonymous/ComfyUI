import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js"

const MAX_HISTORY = 50;

let undo = [];
let redo = [];
let activeState = null;
let isOurLoad = false;
function checkState() {
	const currentState = app.graph.serialize();
	if (!graphEqual(activeState, currentState)) {
		undo.push(activeState);
		if (undo.length > MAX_HISTORY) {
			undo.shift();
		}
		activeState = clone(currentState);
		redo.length = 0;
		api.dispatchEvent(new CustomEvent("graphChanged", { detail: activeState }));
	}
}

const loadGraphData = app.loadGraphData;
app.loadGraphData = async function () {
	const v = await loadGraphData.apply(this, arguments);
	if (isOurLoad) {
		isOurLoad = false;
	} else {
		checkState();
	}
	return v;
};

function clone(obj) {
	try {
		if (typeof structuredClone !== "undefined") {
			return structuredClone(obj);
		}
	} catch (error) {
		// structuredClone is stricter than using JSON.parse/stringify so fallback to that
	}

	return JSON.parse(JSON.stringify(obj));
}

function graphEqual(a, b, root = true) {
	if (a === b) return true;

	if (typeof a == "object" && a && typeof b == "object" && b) {
		const keys = Object.getOwnPropertyNames(a);

		if (keys.length != Object.getOwnPropertyNames(b).length) {
			return false;
		}

		for (const key of keys) {
			let av = a[key];
			let bv = b[key];
			if (root && key === "nodes") {
				// Nodes need to be sorted as the order changes when selecting nodes
				av = [...av].sort((a, b) => a.id - b.id);
				bv = [...bv].sort((a, b) => a.id - b.id);
			}
			if (!graphEqual(av, bv, false)) {
				return false;
			}
		}

		return true;
	}

	return false;
}

const undoRedo = async (e) => {
	const updateState = async (source, target) => {
		const prevState = source.pop();
		if (prevState) {
			target.push(activeState);
			isOurLoad = true;
			await app.loadGraphData(prevState, false);
			activeState = prevState;
		}
	}
	if (e.ctrlKey || e.metaKey) {
		if (e.key === "y") {
			updateState(redo, undo);
			return true;
		} else if (e.key === "z") {
			updateState(undo, redo);
			return true;
		}
	}
};

const bindInput = (activeEl) => {
	if (activeEl && activeEl.tagName !== "CANVAS" && activeEl.tagName !== "BODY") {
		for (const evt of ["change", "input", "blur"]) {
			if (`on${evt}` in activeEl) {
				const listener = () => {
					checkState();
					activeEl.removeEventListener(evt, listener);
				};
				activeEl.addEventListener(evt, listener);
				return true;
			}
		}
	}
};

let keyIgnored = false;
window.addEventListener(
	"keydown",
	(e) => {
		requestAnimationFrame(async () => {
			let activeEl;
			// If we are auto queue in change mode then we do want to trigger on inputs
			if (!app.ui.autoQueueEnabled || app.ui.autoQueueMode === "instant") {
				activeEl = document.activeElement;
				if (activeEl?.tagName === "INPUT" || activeEl?.type === "textarea") {
					// Ignore events on inputs, they have their native history
					return;
				}
			}
		
			keyIgnored = e.key === "Control" || e.key === "Shift" || e.key === "Alt" || e.key === "Meta";
			if (keyIgnored) return;

			// Check if this is a ctrl+z ctrl+y
			if (await undoRedo(e)) return;

			// If our active element is some type of input then handle changes after they're done
			if (bindInput(activeEl)) return;
			checkState();
		});
	},
	true
);

window.addEventListener("keyup", (e) => {
	if (keyIgnored) {
		keyIgnored = false;
		checkState();
	}
});

// Handle clicking DOM elements (e.g. widgets)
window.addEventListener("mouseup", () => {
	checkState();
});

// Handle prompt queue event for dynamic widget changes
api.addEventListener("promptQueued", () => {
	checkState();
});

// Handle litegraph clicks
const processMouseUp = LGraphCanvas.prototype.processMouseUp;
LGraphCanvas.prototype.processMouseUp = function (e) {
	const v = processMouseUp.apply(this, arguments);
	checkState();
	return v;
};
const processMouseDown = LGraphCanvas.prototype.processMouseDown;
LGraphCanvas.prototype.processMouseDown = function (e) {
	const v = processMouseDown.apply(this, arguments);
	checkState();
	return v;
};

// Handle litegraph context menu for COMBO widgets
const close = LiteGraph.ContextMenu.prototype.close;
LiteGraph.ContextMenu.prototype.close = function(e) {
	const v = close.apply(this, arguments);
	checkState();
	return v;
}