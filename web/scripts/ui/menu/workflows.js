// @ts-check

import { ComfyButton } from "../components/button.js";
import { prop, getStorageValue, setStorageValue } from "../../utils.js";
import { $el } from "../../ui.js";
import { api } from "../../api.js";
import { ComfyPopup } from "../components/popup.js";
import { createSpinner } from "../spinner.js";
import { ComfyWorkflow } from "../../workflows.js";

export class ComfyWorkflowsMenu {
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

		app.workflowManager.addEventListener("change", () => {
			const active = app.workflowManager.activeWorkflow;
			this.button.tooltip = active.path;
			this.workflowLabel.textContent = active.name;
			this.unsaved = active.unsaved;
		});

		api.addEventListener("graphChanged", () => {
			this.unsaved = true;
		});

		const classList = {
			"comfyui-workflows-button": true,
			"comfyui-button": true,
			unsaved: getStorageValue("Comfy.PreviousWorkflowUnsaved") === "true",
		};
		this.workflowLabel = $el("span.comfyui-workflows-label", "Unsaved workflow");
		this.button = new ComfyButton({
			content: $el("div.comfyui-workflows-button-inner", [$el("i.mdi.mdi-graph"), this.workflowLabel]),
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
}

export class ComfyWorkflowsContent {
	element = $el("div.comfyui-workflows-panel");

	/**
	 * @param {import("../../app.js").ComfyApp} app
	 * @param {ComfyPopup} popup
	 */
	constructor(app, popup) {
		this.app = app;
		this.popup = popup;
		this.actions = $el("div.comfyui-workflows-actions", [
			new ComfyButton({
				content: "New Workflow",
				icon: "plus",
				action: () => {
					app.workflowManager.setWorkflow(null);
					app.clean();
					app.graph.clear();
					popup.open = false;
				},
			}).element,
			new ComfyButton({
				content: "Load JSON",
				action: () => {
					popup.open = false;
					app.loadGraphData();
				},
			}).element,
			new ComfyButton({
				content: "Load Default",
				action: () => {
					popup.open = false;
					app.loadGraphData();
				},
			}).element,
		]);

		this.spinner = createSpinner();
		this.element.replaceChildren(this.actions, this.spinner);

		this.popup.addEventListener("open", () => this.load());
		this.popup.addEventListener("close", () => this.element.replaceChildren(this.actions, this.spinner));
	}

	async load() {
		this.updateTree();
		this.updateFavorites();
		this.updateOpenWorkflows();
		this.element.replaceChildren(this.actions, this.openElement, this.favoritesElement, this.treeElement);
	}

	updateOpenWorkflows() {
		const current = this.openElement;

		this.openElement = $el("div.comfyui-workflows-open", [
			$el("h3", "Open Workflows"),
			...this.app.workflowManager.openWorkflows.map((w) => {
				const wrapper = new WorkflowElement(w, {
					buttons: [],
				});

				return wrapper.element;
			}),
		]);

		current?.replaceWith(this.openElement);
	}

	updateFavorites() {
		const current = this.favoritesElement;
		const favorites = [...this.app.workflowManager.workflows.filter((w) => w.isFavorite)];

		this.favoritesElement = $el(
			"div.comfyui-workflows-favorites",
			favorites
				.map((f) => {
					return $el("span", f.path);
					// const entry = this.#fileLookup[f];
					// if (!entry) return null;
					// const res = new FileNode(
					// 	this,
					// 	entry.path,
					// 	entry.part,
					// 	"div",
					// 	(f) => {
					// 		entry.isFavorite = f;
					// 		entry.updateFavorite();
					// 		this.updateFavoritesElement();
					// 	},
					// 	() => {
					// 		res.element.remove();
					// 		entry.remove();
					// 		return false;
					// 	}
					// );
					// return res.element;
				})
				.filter(Boolean)
		);

		current?.replaceWith(this.favoritesElement);
	}

	updateTree() {
		const current = this.treeElement;
		const flat = {};
		const nodes = {};

		this.treeElement = $el("ul.comfyui-workflows-tree");

		for (const workflow of this.app.workflowManager.workflows) {
			if (!workflow.pathParts) continue;

			let currentPath = "";
			let currentRoot = this.treeElement;

			for (let i = 0; i < workflow.pathParts.length; i++) {
				const part = workflow.pathParts[i];
				currentPath += (currentPath ? "\\" : "") + part;
				let parentNode = nodes[currentPath];

				// Create a new parent node if it doesn't exist
				if (!parentNode) {
					parentNode = $el("ul.closed", {
						$: (el) => {
							el.onclick = (e) => {
								el.classList.toggle("closed");
								e.stopImmediatePropagation();
							};
						},
					});
					currentRoot.append(parentNode);

					// Create a node for the current part and an inner UL for its children if it isnt a leaf node
					const leaf = i === workflow.pathParts.length - 1;
					let nodeElement;
					if (leaf) {
						const fileNode = new WorkflowElement(workflow, { buttons: [] });
						nodeElement = fileNode.element;
					} else {
						nodeElement = $el("li", [$el("i.mdi.mdi-18px.mdi-folder"), $el("span", part)]);
					}
					parentNode.append(nodeElement);
				}

				nodes[currentPath] = parentNode;
				currentRoot = parentNode;
			}
		}

		current?.replaceWith(this.treeElement);
	}
}

class WorkflowElement {
	/**
	 * @param { ComfyWorkflow } workflow
	 */
	constructor(workflow, { tagName = "li", buttons }) {
		this.workflow = workflow;
		this.buttons = buttons;

		this.element = $el(
			tagName + ".comfyui-workflows-tree-file",
			{
				onclick: workflow.load,
				title: this.workflow.path,
			},
			[$el("span", workflow.name)]
		);
	}
}

// class FileNode {
// 	constructor(parent, path, part, tagName = "li", onFavorite = null, onRemove = null) {
// 		this.parent = parent;
// 		this.path = path;
// 		this.part = part;
// 		this.createFileElement(tagName);
// 		this.onFavorite = onFavorite;
// 		this.onRemove = onRemove;
// 	}

// 	updateFavorite() {
// 		if (this.onFavorite?.(this.isFavorite) === false) return;
// 		if (this.isFavorite) {
// 			this.parent.favorites.add(this.path);
// 			this.favoriteButton.icon = "star";
// 			this.favoriteButton.tooltip = "Unfavorite this workflow";
// 			this.nodeIcon.classList.remove("mdi-file-outline");
// 			this.nodeIcon.classList.add("mdi-star");
// 		} else {
// 			this.parent.favorites.delete(this.path);
// 			this.favoriteButton.icon = "star-outline";
// 			this.favoriteButton.tooltip = "Favorite this workflow";
// 			this.nodeIcon.classList.add("mdi-file-outline");
// 			this.nodeIcon.classList.remove("mdi-star");
// 		}
// 	}

// 	remove() {
// 		if (this.onRemove?.() === false) return;

// 		const folderNode = this.element.parentElement.parentElement;
// 		if (folderNode.querySelectorAll("ul").length === 1) {
// 			folderNode.remove();
// 		} else {
// 			this.element.parentElement.remove();
// 		}
// 	}

// 	createFileElement(tagName) {
// 		const fileName = `workflows\\${this.path}`;

// 		this.isFavorite = this.parent.favorites.has(this.path);
// 		this.nodeIcon = $el("i.mdi.mdi-18px");
// 		this.favoriteButton = new ComfyButton({
// 			icon: this.isFavorite ? "star" : "star-outline",
// 			classList: "comfyui-button comfyui-workflows-file-action comfyui-workflows-file-action-favorite",
// 			iconSize: 18,
// 			action: async (e) => {
// 				e.stopImmediatePropagation();
// 				this.isFavorite = !this.isFavorite;
// 				this.updateFavorite();
// 				this.parent.updateFavoritesElement();
// 				await this.parent.storeWorkflowsInfo();
// 			},
// 		});

// 		const deleteButton = new ComfyButton({
// 			icon: "delete",
// 			tooltip: "Delete this workflow",
// 			classList: "comfyui-button comfyui-workflows-file-action",
// 			iconSize: 18,
// 			action: async (e, btn) => {
// 				e.stopImmediatePropagation();

// 				if (btn.icon === "delete-empty") {
// 					btn.enabled = false;
// 					if (this.isFavorite) {
// 						this.parent.favorites.delete(this.path);
// 						await this.parent.storeWorkflowsInfo();
// 					}
// 					if (this.parent.app.currentWorkflow === this.path) {
// 						this.parent.app.currentWorkflow = null;
// 					}
// 					await api.deleteUserData(fileName);
// 					this.remove();
// 				} else {
// 					btn.icon = "delete-empty";
// 					btn.element.style.background = "red";
// 				}
// 			},
// 		});
// 		deleteButton.element.addEventListener("mouseleave", () => {
// 			deleteButton.icon = "delete";
// 			deleteButton.element.style.removeProperty("background");
// 		});

// 		const getWorkflow = async (e) => {
// 			e.stopImmediatePropagation();
// 			const resp = await api.getUserData(fileName);
// 			if (resp.status !== 200) {
// 				alert(`Error loading user data file '${fileName}': ${resp.status} ${resp.statusText}`);
// 				return;
// 			}
// 			return await resp.json();
// 		};

// 		const loadWorkflow = async (e) => {
// 			const data = await getWorkflow(e);
// 			await this.parent.app.loadGraphData(data, true, this.path);
// 			this.parent.popup.open = false;
// 		};

// 		const insertWorkflow = async (e) => {
// 			const data = await getWorkflow(e);
// 			const old = localStorage.getItem("litegrapheditor_clipboard");
// 			const graph = new LGraph(data);
// 			const canvas = new LGraphCanvas(null, graph, { skip_events: true, skip_render: true });
// 			canvas.selectNodes();
// 			canvas.copyToClipboard();
// 			this.parent.app.canvas.pasteFromClipboard();
// 			localStorage.setItem("litegrapheditor_clipboard", old);
// 		};

// 		const name = trimJsonExt(this.part);
// 		this.element = $el(
// 			tagName + ".comfyui-workflows-tree-file",
// 			{
// 				onclick: loadWorkflow,
// 				title: this.path,
// 			},
// 			[
// 				this.nodeIcon,
// 				$el("span", name),
// 				new ComfyButton({
// 					icon: "file-move-outline",
// 					tooltip: "Insert this workflow into the current view",
// 					classList: "comfyui-button comfyui-workflows-file-action",
// 					iconSize: 18,
// 					action: insertWorkflow,
// 				}).element,
// 				new ComfyButton({
// 					icon: "pencil",
// 					tooltip: "Rename this workflow",
// 					classList: "comfyui-button comfyui-workflows-file-action",
// 					iconSize: 18,
// 					action: async (e) => {
// 						e.stopImmediatePropagation();
// 						const newName = prompt("Enter new name", trimJsonExt(this.path));
// 						if (newName) {
// 							// Rename file
// 							try {
// 								const res = await api.moveUserData("workflows/" + this.path, "workflows/" + newName);

// 								if (res.status === 409) {
// 									if (!confirm(`Workflow '${newName}' already exists, do you want to overwrite it?`)) return;
// 									await api.moveUserData("workflows/" + this.path, "workflows/" + newName, { overwrite: true });
// 								}

// 								if (this.parent.app.currentWorkflow === this.path) {
// 									this.parent.app.currentWorkflow = newName;
// 								}

// 								await this.parent.load();
// 							} catch (error) {
// 								alert(error.message ?? error);
// 							}
// 						}
// 					},
// 				}).element,
// 				this.favoriteButton.element,
// 				deleteButton.element,
// 			]
// 		);
// 		this.updateFavorite();
// 	}
// }
