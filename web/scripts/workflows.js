// @ts-check

import { api } from "./api.js";
import { ChangeTracker } from "./changeTracker.js";
import { ComfyAsyncDialog } from "./ui/components/asyncDialog.js";
import { getStorageValue, setStorageValue } from "./utils.js";

function appendJsonExt(path) {
	if (!path.toLowerCase().endsWith(".json")) {
		path += ".json";
	}
	return path;
}

export function trimJsonExt(path) {
	return path?.replace(/\.json$/, "");
}

export class ComfyWorkflowManager extends EventTarget {
	/** @type {string | null} */
	#activePromptId = null;
	#unsavedCount = 0;
	#activeWorkflow;

	/** @type {Record<string, ComfyWorkflow>} */
	workflowLookup = {};
	/** @type {Array<ComfyWorkflow>} */
	workflows = [];
	/** @type {Array<ComfyWorkflow>} */
	openWorkflows = [];
	/** @type {Record<string, {workflow?: ComfyWorkflow, nodes?: Record<string, boolean>}>} */
	queuedPrompts = {};

	get activeWorkflow() {
		return this.#activeWorkflow ?? this.openWorkflows[0];
	}

	get activePromptId() {
		return this.#activePromptId;
	}

	get activePrompt() {
		return this.queuedPrompts[this.#activePromptId];
	}

	/**
	 * @param {import("./app.js").ComfyApp} app
	 */
	constructor(app) {
		super();
		this.app = app;
		ChangeTracker.init(app);

		this.#bindExecutionEvents();
	}

	#bindExecutionEvents() {
		// TODO: on reload, set active prompt based on the latest ws message

		const emit = () => this.dispatchEvent(new CustomEvent("execute", { detail: this.activePrompt }));
		let executing = null;
		api.addEventListener("execution_start", (e) => {
			this.#activePromptId = e.detail.prompt_id;

			// This event can fire before the event is stored, so put a placeholder
			this.queuedPrompts[this.#activePromptId] ??= { nodes: {} };
			emit();
		});
		api.addEventListener("execution_cached", (e) => {
			if (!this.activePrompt) return;
			for (const n of e.detail.nodes) {
				this.activePrompt.nodes[n] = true;
			}
			emit();
		});
		api.addEventListener("executed", (e) => {
			if (!this.activePrompt) return;
			this.activePrompt.nodes[e.detail.node] = true;
			emit();
		});
		api.addEventListener("executing", (e) => {
			if (!this.activePrompt) return;

			if (executing) {
				// Seems sometimes nodes that are cached fire executing but not executed
				this.activePrompt.nodes[executing] = true;
			}
			executing = e.detail;
			if (!executing) {
				delete this.queuedPrompts[this.#activePromptId];
				this.#activePromptId = null;
			}
			emit();
		});
	}

	async loadWorkflows() {
		try {
			let favorites;
			const resp = await api.getUserData("workflows/.index.json");
			let info;
			if (resp.status === 200) {
				info = await resp.json();
				favorites = new Set(info?.favorites ?? []);
			} else {
				favorites = new Set();
			}

			const workflows = (await api.listUserData("workflows", true, true)).map((w) => {
				let workflow = this.workflowLookup[w[0]];
				if (!workflow) {
					workflow = new ComfyWorkflow(this, w[0], w.slice(1), favorites.has(w[0]));
					this.workflowLookup[workflow.path] = workflow;
				}
				return workflow;
			});

			this.workflows = workflows;
		} catch (error) {
			alert("Error loading workflows: " + (error.message ?? error));
			this.workflows = [];
		}
	}

	async saveWorkflowMetadata() {
		await api.storeUserData("workflows/.index.json", {
			favorites: [...this.workflows.filter((w) => w.isFavorite).map((w) => w.path)],
		});
	}

	/**
	 * @param {string | ComfyWorkflow | null} workflow
	 */
	setWorkflow(workflow) {
		if (workflow && typeof workflow === "string") {
			// Selected by path, i.e. on reload of last workflow
			const found = this.workflows.find((w) => w.path === workflow);
			if (found) {
				workflow = found;
				workflow.unsaved = !workflow || getStorageValue("Comfy.PreviousWorkflowUnsaved") === "true";
			}
		}

		if (!(workflow instanceof ComfyWorkflow)) {
			// Still not found, either reloading a deleted workflow or blank
			workflow = new ComfyWorkflow(this, workflow || "Unsaved Workflow" + (this.#unsavedCount++ ? ` (${this.#unsavedCount})` : ""));
		}

		const index = this.openWorkflows.indexOf(workflow);
		if (index === -1) {
			// Opening a new workflow
			this.openWorkflows.push(workflow);
		}

		this.#activeWorkflow = workflow;

		setStorageValue("Comfy.PreviousWorkflow", this.activeWorkflow.path ?? "");
		this.dispatchEvent(new CustomEvent("changeWorkflow"));
	}

	storePrompt({ nodes, id }) {
		this.queuedPrompts[id] ??= {};
		this.queuedPrompts[id].nodes = {
			...nodes.reduce((p, n) => {
				p[n] = false;
				return p;
			}, {}),
			...this.queuedPrompts[id].nodes,
		};
		this.queuedPrompts[id].workflow = this.activeWorkflow;
	}

	/**
	 * @param {ComfyWorkflow} workflow
	 */
	async closeWorkflow(workflow, warnIfUnsaved = true) {
		if (!workflow.isOpen) {
			return true;
		}
		if (workflow.unsaved && warnIfUnsaved) {
			const res = await ComfyAsyncDialog.prompt({
				title: "Save Changes?",
				message: `Do you want to save changes to "${workflow.path ?? workflow.name}" before closing?`,
				actions: ["Yes", "No", "Cancel"],
			});
			if (res === "Yes") {
				const active = this.activeWorkflow;
				if (active !== workflow) {
					// We need to switch to the workflow to save it
					await workflow.load();
				}

				if (!(await workflow.save())) {
					// Save was canceled, restore the previous workflow
					if (active !== workflow) {
						await active.load();
					}
					return;
				}
			} else if (res === "Cancel") {
				return;
			}
		}
		workflow.changeTracker = null;
		this.openWorkflows.splice(this.openWorkflows.indexOf(workflow), 1);
		if (this.openWorkflows.length) {
			this.#activeWorkflow = this.openWorkflows[0];
			await this.#activeWorkflow.load();
		} else {
			// Load default
			await this.app.loadGraphData();
		}
	}
}

export class ComfyWorkflow {
	#name;
	#path;
	#pathParts;
	#isFavorite = false;
	/** @type {ChangeTracker | null} */
	changeTracker = null;
	unsaved = false;

	get name() {
		return this.#name;
	}

	get path() {
		return this.#path;
	}

	get pathParts() {
		return this.#pathParts;
	}

	get isFavorite() {
		return this.#isFavorite;
	}

	get isOpen() {
		return !!this.changeTracker;
	}

	/**
	 * @overload
	 * @param {ComfyWorkflowManager} manager
	 * @param {string} path
	 */
	/**
	 * @overload
	 * @param {ComfyWorkflowManager} manager
	 * @param {string} path
	 * @param {string[]} pathParts
	 * @param {boolean} isFavorite
	 */
	/**
	 * @param {ComfyWorkflowManager} manager
	 * @param {string} path
	 * @param {string[]} [pathParts]
	 * @param {boolean} [isFavorite]
	 */
	constructor(manager, path, pathParts, isFavorite) {
		this.manager = manager;
		if (pathParts) {
			this.#updatePath(path, pathParts);
			this.#isFavorite = isFavorite;
		} else {
			this.#name = path;
			this.unsaved = true;
		}
	}

	/**
	 * @param {string} path
	 * @param {string[]} [pathParts]
	 */
	#updatePath(path, pathParts) {
		this.#path = path;

		if (!pathParts) {
			if (!path.includes("\\")) {
				pathParts = path.split("/");
			} else {
				pathParts = path.split("\\");
			}
		}

		this.#pathParts = pathParts;
		this.#name = trimJsonExt(pathParts[pathParts.length - 1]);
	}

	async getWorkflowData() {
		const resp = await api.getUserData("workflows/" + this.path);
		if (resp.status !== 200) {
			alert(`Error loading workflow file '${this.path}': ${resp.status} ${resp.statusText}`);
			return;
		}
		return await resp.json();
	}

	load = async () => {
		if (this.isOpen) {
			await this.manager.app.loadGraphData(this.changeTracker.activeState, true, this);
		} else {
			const data = await this.getWorkflowData();
			if (!data) return;
			await this.manager.app.loadGraphData(data, true, this);
		}
	};

	async save(saveAs = false) {
		if (!this.path || saveAs) {
			return !!(await this.#save(null, false));
		} else {
			return !!(await this.#save(this.path, true));
		}
	}

	/**
	 * @param {boolean} value
	 */
	async favorite(value) {
		try {
			if (this.#isFavorite === value) return;
			this.#isFavorite = value;
			await this.manager.saveWorkflowMetadata();
			this.manager.dispatchEvent(new CustomEvent("favorite", { detail: this }));
		} catch (error) {
			alert("Error favoriting workflow " + this.path + "\n" + (error.message ?? error));
		}
	}

	/**
	 * @param {string} path
	 */
	async rename(path) {
		path = appendJsonExt(path);
		let resp = await api.moveUserData("workflows/" + this.path, "workflows/" + path);

		if (resp.status === 409) {
			if (!confirm(`Workflow '${path}' already exists, do you want to overwrite it?`)) return resp;
			resp = await api.moveUserData("workflows/" + this.path, "workflows/" + path, { overwrite: true });
		}

		if (resp.status !== 200) {
			alert(`Error renaming workflow file '${this.path}': ${resp.status} ${resp.statusText}`);
			return;
		}

		const isFav = this.isFavorite;
		if (isFav) {
			await this.favorite(false);
		}
		path = (await resp.json()).substring("workflows/".length);
		this.#updatePath(path, null);
		if (isFav) {
			await this.favorite(true);
		}
		this.manager.dispatchEvent(new CustomEvent("rename", { detail: this }));
		setStorageValue("Comfy.PreviousWorkflow", this.path ?? "");
	}

	async insert() {
		const data = await this.getWorkflowData();
		if (!data) return;

		const old = localStorage.getItem("litegrapheditor_clipboard");
		const graph = new LGraph(data);
		const canvas = new LGraphCanvas(null, graph, { skip_events: true, skip_render: true });
		canvas.selectNodes();
		canvas.copyToClipboard();
		this.manager.app.canvas.pasteFromClipboard();
		localStorage.setItem("litegrapheditor_clipboard", old);
	}

	async delete() {
		// TODO: fix delete of current workflow - should mark workflow as unsaved and when saving use old name by default

		try {
			if (this.isFavorite) {
				await this.favorite(false);
			}
			await api.deleteUserData("workflows/" + this.path);
			this.unsaved = true;
			this.#path = null;
			this.#pathParts = null;
			this.manager.workflows.splice(this.manager.workflows.indexOf(this), 1);
			this.manager.dispatchEvent(new CustomEvent("delete", { detail: this }));
		} catch (error) {
			alert(`Error deleting workflow: ${error.message || error}`);
		}
	}

	track() {
		if (this.changeTracker) {
			this.changeTracker.restore();
		} else {
			this.changeTracker = new ChangeTracker(this);
		}
	}

	/**
	 * @param {string|null} path
	 * @param {boolean} overwrite
	 */
	async #save(path, overwrite) {
		if (!path) {
			path = prompt("Save workflow as:", trimJsonExt(this.path) ?? this.name ?? "workflow");
			if (!path) return;
		}

		path = appendJsonExt(path);

		const p = await this.manager.app.graphToPrompt();
		const json = JSON.stringify(p.workflow, null, 2);
		let resp = await api.storeUserData("workflows/" + path, json, { stringify: false, throwOnError: false, overwrite });
		if (resp.status === 409) {
			if (!confirm(`Workflow '${path}' already exists, do you want to overwrite it?`)) return;
			resp = await api.storeUserData("workflows/" + path, json, { stringify: false });
		}

		if (resp.status !== 200) {
			alert(`Error saving workflow '${this.path}': ${resp.status} ${resp.statusText}`);
			return;
		}

		path = (await resp.json()).substring("workflows/".length);

		if (!this.path) {
			// Saved new workflow, patch this instance
			this.#updatePath(path, null);
			await this.manager.loadWorkflows();
			this.unsaved = false;
			this.manager.dispatchEvent(new CustomEvent("rename", { detail: this }));
			setStorageValue("Comfy.PreviousWorkflow", this.path ?? "");
		} else if (path !== this.path) {
			// Saved as, open the new copy
			await this.manager.loadWorkflows();
			const workflow = this.manager.workflowLookup[path];
			await workflow.load();
		} else {
			// Normal save
			this.unsaved = false;
			this.manager.dispatchEvent(new CustomEvent("save", { detail: this }));
		}

		return true;
	}
}
