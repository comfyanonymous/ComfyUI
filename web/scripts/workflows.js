// @ts-check

import { api } from "./api.js";
import { clone } from "./utils.js";

export class ComfyWorkflow {
	/** @type { Array<ComfyWorkflow> } */
	static openWorkflows = [];
	/** @type {import("./app.js").ComfyApp} */
	static app;
	static unsavedCount = 0;

	unsaved = false;
	/** @type { string | null } */
	name = null;
	/** @type { ChangeTracker } */
	changeTracker;
	/** @type { string | null } */
	tempName = null;

	get displayName() {
		return this.name ?? this.tempName;
	}

	static get activeWorkflow() {
		return this.openWorkflows[0];
	}

	static get changeTracker() {
		return this.activeWorkflow.changeTracker;
	}

	/**
	 * @param {string} [name]
	 */
	constructor(name) {
		this.name = name;
		if (!name) {
			this.tempName = "Unsaved workflow";
			if (ComfyWorkflow.unsavedCount++) {
				this.tempName += ` (${ComfyWorkflow.unsavedCount})`;
			}
		}
		this.changeTracker = new ChangeTracker(this);
	}

	/**
	 * @param {import("./app.js").ComfyApp} app
	 */
	static init(app) {
		ComfyWorkflow.app = app;
		ChangeTracker.init(app);
	}

	/**
	 * 
	 * @param { Partial<ComfyWorkflow> } opts 
	 * @returns 
	 */
	static newWorkflow(opts = {}) {
		const workflow = new ComfyWorkflow(opts?.name);
		Object.assign(workflow, opts);
		ComfyWorkflow.openWorkflows.unshift(workflow);
		ComfyWorkflow.#workflowsChanged();
		return workflow;
	}

	/**
	 * @param { ComfyWorkflow | string } workflow
	 */
	static changeWorkflow(workflow) {
		let index;
		if (typeof workflow === "string") {
			index = this.openWorkflows.findIndex((w) => w.name === workflow);
			if (index === -1) {
				const r = ComfyWorkflow.newWorkflow({ name: workflow });
				return r;
			} else {
				workflow = this.openWorkflows[index];
			}
		} else {
			index = this.openWorkflows.indexOf(workflow);
		}
		if (index !== -1) {
			this.openWorkflows.splice(index, 1);
		}
		this.openWorkflows.unshift(workflow);
		this.#workflowsChanged();
		return workflow;
	}

	/**
	 * @param { ComfyWorkflow } workflow
	 */
	static closeWorkflow(workflow) {
		const index = this.openWorkflows.indexOf(workflow);
		if (index !== -1) {
			this.openWorkflows.splice(index, 1);
		}
		ComfyWorkflow.#workflowsChanged();
	}

	static #workflowsChanged() {
		console.log("workflowsChanged", this.activeWorkflow?.name, this.openWorkflows.length);
		api.dispatchEvent(new CustomEvent("workflowsChanged", { detail: this.activeWorkflow }));
	}
}

export class ChangeTracker {
	static MAX_HISTORY = 50;

	undo = [];
	redo = [];
	activeState = null;
	isOurLoad = false;
	/** @type { ComfyWorkflow } */
	workflow;

	constructor(workflow) {
		this.workflow = workflow;
	}

	checkState() {
		const currentState = ComfyWorkflow.app.graph.serialize();
		if (!ChangeTracker.graphEqual(this.activeState, currentState)) {
			this.undo.push(this.activeState);
			if (this.undo.length > ChangeTracker.MAX_HISTORY) {
				this.undo.shift();
			}
			this.activeState = clone(currentState);
			this.redo.length = 0;
			this.workflow.unsaved = true;
			api.dispatchEvent(new CustomEvent("graphChanged", { detail: this.activeState }));
		}
	}

	async updateState(source, target) {
		const prevState = source.pop();
		if (prevState) {
			target.push(this.activeState);
			this.isOurLoad = true;
			await ComfyWorkflow.app.loadGraphData(prevState, false, this.workflow);
			this.activeState = prevState;
		}
	}

	async undoRedo(e) {
		if (e.ctrlKey || e.metaKey) {
			if (e.key === "y") {
				this.updateState(this.redo, this.undo);
				return true;
			} else if (e.key === "z") {
				this.updateState(this.undo, this.redo);
				return true;
			}
		}
	}

	static init(app) {
		const loadGraphData = app.loadGraphData;
		app.loadGraphData = async function () {
			const v = await loadGraphData.apply(this, arguments);
			if (ComfyWorkflow.changeTracker.isOurLoad) {
				ComfyWorkflow.changeTracker.isOurLoad = false;
			} else {
				ComfyWorkflow.changeTracker.checkState();
			}
			return v;
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
						if (activeEl?.tagName === "INPUT" || activeEl?.["type"] === "textarea") {
							// Ignore events on inputs, they have their native history
							return;
						}
					}

					keyIgnored = e.key === "Control" || e.key === "Shift" || e.key === "Alt" || e.key === "Meta";
					if (keyIgnored) return;

					// Check if this is a ctrl+z ctrl+y
					if (await ComfyWorkflow.changeTracker.undoRedo(e)) return;

					// If our active element is some type of input then handle changes after they're done
					if (ChangeTracker.bindInput(activeEl)) return;
					ComfyWorkflow.changeTracker.checkState();
				});
			},
			true
		);

		window.addEventListener("keyup", (e) => {
			if (keyIgnored) {
				keyIgnored = false;
				ComfyWorkflow.changeTracker.checkState();
			}
		});

		// Handle clicking DOM elements (e.g. widgets)
		window.addEventListener("mouseup", () => {
			ComfyWorkflow.changeTracker.checkState();
		});

		// Handle prompt queue event for dynamic widget changes
		api.addEventListener("promptQueued", () => {
			ComfyWorkflow.changeTracker.checkState();
		});

		// Handle litegraph clicks
		const processMouseUp = LGraphCanvas.prototype.processMouseUp;
		LGraphCanvas.prototype.processMouseUp = function (e) {
			const v = processMouseUp.apply(this, arguments);
			ComfyWorkflow.changeTracker.checkState();
			return v;
		};
		const processMouseDown = LGraphCanvas.prototype.processMouseDown;
		LGraphCanvas.prototype.processMouseDown = function (e) {
			const v = processMouseDown.apply(this, arguments);
			ComfyWorkflow.changeTracker.checkState();
			return v;
		};

		// Handle litegraph context menu for COMBO widgets
		const close = LiteGraph.ContextMenu.prototype.close;
		LiteGraph.ContextMenu.prototype.close = function (e) {
			const v = close.apply(this, arguments);
			ComfyWorkflow.changeTracker.checkState();
			return v;
		};
	}

	static bindInput(activeEl) {
		if (activeEl && activeEl.tagName !== "CANVAS" && activeEl.tagName !== "BODY") {
			for (const evt of ["change", "input", "blur"]) {
				if (`on${evt}` in activeEl) {
					const listener = () => {
						ComfyWorkflow.changeTracker.checkState();
						activeEl.removeEventListener(evt, listener);
					};
					activeEl.addEventListener(evt, listener);
					return true;
				}
			}
		}
	}

	static graphEqual(a, b, root = true) {
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
				if (!ChangeTracker.graphEqual(av, bv, false)) {
					return false;
				}
			}

			return true;
		}

		return false;
	}
}
