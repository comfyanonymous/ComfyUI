// @ts-check

import { ComfyButton } from "../components/button.js";
import { prop, getStorageValue, setStorageValue } from "../../utils.js";
import { $el } from "../../ui.js";
import { api } from "../../api.js";
import { ComfyPopup } from "../components/popup.js";
import { createSpinner } from "../spinner.js";
import { ComfyWorkflow } from "../../workflows.js";

export class ComfyWorkflowsMenu {
	#first = true;
	element = $el("div.comfyui-workflows");

	get open() {
		return this.popup.open;
	}

	set open(open) {
		this.popup.open = open;
	}

	/**
	 * @param {import("../../app.js").ComfyApp} app
	 */
	constructor(app) {
		this.app = app;
		this.#bindEvents();

		const classList = {
			"comfyui-workflows-button": true,
			"comfyui-button": true,
			unsaved: getStorageValue("Comfy.PreviousWorkflowUnsaved") === "true",
			running: false,
		};
		this.buttonProgress = $el("div.comfyui-workflows-button-progress");
		this.workflowLabel = $el("span.comfyui-workflows-label", "");
		this.button = new ComfyButton({
			content: $el("div.comfyui-workflows-button-inner", [$el("i.mdi.mdi-graph"), this.workflowLabel, this.buttonProgress]),
			icon: "chevron-down",
			classList,
		});

		this.element.append(this.button.element);

		this.popup = new ComfyPopup({ target: this.element, classList: "comfyui-workflows-popup" });
		this.content = new ComfyWorkflowsContent(app, this.popup);
		this.popup.children = [this.content.element];
		this.popup.addEventListener("change", () => {
			this.button.icon = "chevron-" + (this.popup.open ? "up" : "down");
		});
		this.button.withPopup(this.popup);

		this.unsaved = prop(this, "unsaved", classList.unsaved, (v) => {
			classList.unsaved = v;
			this.button.classList = classList;
			setStorageValue("Comfy.PreviousWorkflowUnsaved", v);
		});
	}

	#updateProgress = () => {
		const prompt = this.app.workflowManager.activePrompt;
		let percent = 0;
		if (this.app.workflowManager.activeWorkflow === prompt?.workflow) {
			const total = Object.values(prompt.nodes);
			const done = total.filter(Boolean);
			percent = (done.length / total.length) * 100;
		}
		this.buttonProgress.style.width = percent + "%";
	};

	#updateActive = () => {
		const active = this.app.workflowManager.activeWorkflow;
		this.button.tooltip = active.path;
		this.workflowLabel.textContent = active.name;
		this.unsaved = active.unsaved;

		if (this.#first) {
			this.#first = false;
			this.content.load();
		}

		this.#updateProgress();
	};

	#bindEvents() {
		this.app.workflowManager.addEventListener("changeWorkflow", this.#updateActive);
		this.app.workflowManager.addEventListener("rename", this.#updateActive);
		this.app.workflowManager.addEventListener("delete", this.#updateActive);

		this.app.workflowManager.addEventListener("save", () => {
			this.unsaved = this.app.workflowManager.activeWorkflow.unsaved;
		});

		this.app.workflowManager.addEventListener("execute", (e) => {
			this.#updateProgress();
		});

		api.addEventListener("graphChanged", () => {
			this.unsaved = true;
		});
	}
}

export class ComfyWorkflowsContent {
	element = $el("div.comfyui-workflows-panel");
	treeState = {};
	treeFiles = {};
	/** @type { Map<ComfyWorkflow, WorkflowElement> } */
	openFiles = new Map();
	/** @type {WorkflowElement} */
	activeElement = null;

	/**
	 * @param {import("../../app.js").ComfyApp} app
	 * @param {ComfyPopup} popup
	 */
	constructor(app, popup) {
		this.app = app;
		this.popup = popup;
		this.actions = $el("div.comfyui-workflows-actions", [
			new ComfyButton({
				content: "Default",
				icon: "file-code",
				iconSize: 18,
				classList: "comfyui-button primary",
				action: () => {
					popup.open = false;
					app.loadGraphData();
				},
			}).element,
			new ComfyButton({
				content: "Browse",
				icon: "folder",
				iconSize: 18,
				action: () => {
					popup.open = false;
					app.ui.loadFile();
				},
			}).element,
			new ComfyButton({
				content: "Blank",
				icon: "plus-thick",
				iconSize: 18,
				action: () => {
					app.workflowManager.setWorkflow(null);
					app.clean();
					app.graph.clear();
					app.workflowManager.activeWorkflow.track();
					popup.open = false;
				},
			}).element,
		]);

		this.spinner = createSpinner();
		this.element.replaceChildren(this.actions, this.spinner);

		this.popup.addEventListener("open", () => this.load());
		this.popup.addEventListener("close", () => this.element.replaceChildren(this.actions, this.spinner));

		this.app.workflowManager.addEventListener("favorite", (e) => {
			const workflow = e["detail"];
			const button = this.treeFiles[workflow.path]?.primary;
			if (!button) return; // Can happen when a workflow is renamed
			button.icon = this.#getFavoriteIcon(workflow);
			button.overIcon = this.#getFavoriteOverIcon(workflow);
			this.updateFavorites();
		});

		for (const e of ["save", "open", "close", "changeWorkflow"]) {
			// TODO: dont be lazy and just update the specific element
			app.workflowManager.addEventListener(e, () => this.updateOpen());
		}
		this.app.workflowManager.addEventListener("rename", () => this.load());
		this.app.workflowManager.addEventListener("execute", (e) => this.#updateActive());
	}

	async load() {
		await this.app.workflowManager.loadWorkflows();
		this.updateTree();
		this.updateFavorites();
		this.updateOpen();
		this.element.replaceChildren(this.actions, this.openElement, this.favoritesElement, this.treeElement);
	}

	updateOpen() {
		const current = this.openElement;
		this.openFiles.clear();

		this.openElement = $el("div.comfyui-workflows-open", [
			$el("h3", "Open"),
			...this.app.workflowManager.openWorkflows.map((w) => {
				const wrapper = new WorkflowElement(this, w, {
					primary: { element: $el("i.mdi.mdi-18px.mdi-circle-small") },
					buttons: [
						new ComfyButton({
							icon: "content-save",
							iconSize: 18,
							classList: "comfyui-button comfyui-workflows-file-action",
							enabled: w.unsaved,
							action: (e) => {
								e.stopImmediatePropagation();
								w.save();
							},
						}),
						new ComfyButton({
							icon: "close",
							iconSize: 18,
							classList: "comfyui-button comfyui-workflows-file-action",
							action: (e) => {
								e.stopImmediatePropagation();
								this.app.workflowManager.closeWorkflow(w);
							},
						}),
					],
				});
				if (w.unsaved) {
					wrapper.element.classList.add("unsaved");
				}

				this.openFiles.set(w, wrapper);
				return wrapper.element;
			}),
		]);

		this.#updateActive();
		current?.replaceWith(this.openElement);
	}

	updateFavorites() {
		const current = this.favoritesElement;
		const favorites = [...this.app.workflowManager.workflows.filter((w) => w.isFavorite)];

		this.favoritesElement = $el("div.comfyui-workflows-favorites", [
			$el("h3", "Favorites"),
			...favorites
				.map((w) => {
					return this.#getWorkflowElement(w).element;
				})
				.filter(Boolean),
		]);

		current?.replaceWith(this.favoritesElement);
	}

	updateTree() {
		const current = this.treeElement;
		const nodes = {};

		this.treeFiles = {};
		const tree = $el("ul.comfyui-workflows-tree");
		this.treeElement = $el("section", [$el("h3", "Browse"), tree]);

		for (const workflow of this.app.workflowManager.workflows) {
			if (!workflow.pathParts) continue;

			let currentPath = "";
			let currentRoot = tree;

			for (let i = 0; i < workflow.pathParts.length; i++) {
				currentPath += (currentPath ? "\\" : "") + workflow.pathParts[i];
				const parentNode = nodes[currentPath] ?? this.#createNode(currentPath, workflow, i, currentRoot);

				nodes[currentPath] = parentNode;
				currentRoot = parentNode;
			}
		}

		current?.replaceWith(this.treeElement);
	}

	#expandNode(el, workflow, thisPath, i) {
		const expanded = !el.classList.toggle("closed");
		if (expanded) {
			let c = "";
			for (let j = 0; j <= i; j++) {
				c += (c ? "\\" : "") + workflow.pathParts[j];
				this.treeState[c] = true;
			}
		} else {
			let c = thisPath;
			for (let j = i + 1; j < workflow.pathParts.length; j++) {
				c += (c ? "\\" : "") + workflow.pathParts[j];
				delete this.treeState[c];
			}
			delete this.treeState[thisPath];
		}
	}

	#updateActive() {
		this.#removeActive();

		const active = this.app.workflowManager.activePrompt;
		if (!active?.workflow) return;

		const open = this.openFiles.get(active.workflow);
		if (!open) return;

		this.activeElement = open;

		const total = Object.values(active.nodes);
		const done = total.filter(Boolean);
		const percent = done.length / total.length;
		open.element.classList.add("running");
		open.element.style.setProperty("--progress", percent * 100 + "%");
		open.primary.element.classList.remove("mdi-circle-small");
		open.primary.element.classList.add("mdi-play");
	}

	#removeActive() {
		if (!this.activeElement) return;
		this.activeElement.element.classList.remove("running");
		this.activeElement.element.style.removeProperty("--progress");
		this.activeElement.primary.element.classList.add("mdi-circle-small");
		this.activeElement.primary.element.classList.remove("mdi-play");
	}

	/** @param {ComfyWorkflow} workflow */
	#getFavoriteIcon(workflow) {
		return workflow.isFavorite ? "star" : "file-outline";
	}

	/** @param {ComfyWorkflow} workflow */
	#getFavoriteOverIcon(workflow) {
		return workflow.isFavorite ? "star-off" : "star-outline";
	}

	/** @param {ComfyWorkflow} workflow */
	#getFavoriteButton(workflow, primary) {
		return new ComfyButton({
			icon: this.#getFavoriteIcon(workflow),
			overIcon: this.#getFavoriteOverIcon(workflow),
			iconSize: 18,
			classList: "comfyui-button comfyui-workflows-file-action-favorite" + (primary ? " comfyui-workflows-file-action-primary" : ""),
			action: (e) => {
				e.stopImmediatePropagation();
				workflow.favorite(!workflow.isFavorite);
			},
		});
	}

	/** @param {ComfyWorkflow} workflow */
	#getDeleteButton(workflow) {
		const deleteButton = new ComfyButton({
			icon: "delete",
			tooltip: "Delete this workflow",
			classList: "comfyui-button comfyui-workflows-file-action",
			iconSize: 18,
			action: async (e, btn) => {
				e.stopImmediatePropagation();

				if (btn.icon === "delete-empty") {
					btn.enabled = false;
					await workflow.delete();
					await this.load();
				} else {
					btn.icon = "delete-empty";
					btn.element.style.background = "red";
				}
			},
		});
		deleteButton.element.addEventListener("mouseleave", () => {
			deleteButton.icon = "delete";
			deleteButton.element.style.removeProperty("background");
		});
		return deleteButton;
	}

	/** @param {ComfyWorkflow} workflow */
	#getInsertButton(workflow) {
		return new ComfyButton({
			icon: "file-move-outline",
			iconSize: 18,
			tooltip: "Insert this workflow into the current workflow",
			classList: "comfyui-button comfyui-workflows-file-action",
			action: (e) => {
				if (!this.app.shiftDown) {
					this.popup.open = false;
				}
				e.stopImmediatePropagation();
				if (!this.app.shiftDown) {
					this.popup.open = false;
				}
				workflow.insert();
			},
		});
	}

	/** @param {ComfyWorkflow} workflow */
	#getRenameButton(workflow) {
		return new ComfyButton({
			icon: "pencil",
			tooltip: "Rename this workflow",
			classList: "comfyui-button comfyui-workflows-file-action",
			iconSize: 18,
			action: async (e) => {
				e.stopImmediatePropagation();
				const newName = prompt("Enter new name", workflow.path);
				if (newName) {
					await workflow.rename(newName);
				}
			},
		});
	}

	/** @param {ComfyWorkflow} workflow */
	#getWorkflowElement(workflow) {
		return new WorkflowElement(this, workflow, {
			primary: this.#getFavoriteButton(workflow, true),
			buttons: [this.#getInsertButton(workflow), this.#getRenameButton(workflow), this.#getDeleteButton(workflow)],
		});
	}

	/** @param {ComfyWorkflow} workflow */
	#createLeafNode(workflow) {
		const fileNode = this.#getWorkflowElement(workflow);
		this.treeFiles[workflow.path] = fileNode;
		return fileNode;
	}

	#createNode(currentPath, workflow, i, currentRoot) {
		const part = workflow.pathParts[i];

		const parentNode = $el("ul" + (this.treeState[currentPath] ? "" : ".closed"), {
			$: (el) => {
				el.onclick = (e) => {
					this.#expandNode(el, workflow, currentPath, i);
					e.stopImmediatePropagation();
				};
			},
		});
		currentRoot.append(parentNode);

		// Create a node for the current part and an inner UL for its children if it isnt a leaf node
		const leaf = i === workflow.pathParts.length - 1;
		let nodeElement;
		if (leaf) {
			nodeElement = this.#createLeafNode(workflow).element;
		} else {
			nodeElement = $el("li", [$el("i.mdi.mdi-18px.mdi-folder"), $el("span", part)]);
		}
		parentNode.append(nodeElement);
		return parentNode;
	}
}

class WorkflowElement {
	/**
	 * @param { ComfyWorkflowsContent } parent
	 * @param { ComfyWorkflow } workflow
	 */
	constructor(parent, workflow, { tagName = "li", primary, buttons }) {
		this.parent = parent;
		this.workflow = workflow;
		this.primary = primary;
		this.buttons = buttons;

		this.element = $el(
			tagName + ".comfyui-workflows-tree-file",
			{
				onclick: () => {
					workflow.load();
					this.parent.popup.open = false;
				},
				title: this.workflow.path,
			},
			[this.primary?.element, $el("span", workflow.name), ...buttons.map((b) => b.element)]
		);
	}
}
