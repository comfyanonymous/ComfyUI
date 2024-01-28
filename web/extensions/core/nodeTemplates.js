import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { ComfyDialog, $el } from "../../scripts/ui.js";
import { GroupNodeConfig, GroupNodeHandler } from "./groupNode.js";

// Adds the ability to save and add multiple nodes as a template
// To save:
// Select multiple nodes (ctrl + drag to select a region or ctrl+click individual nodes)
// Right click the canvas
// Save Node Template -> give it a name
//
// To add:
// Right click the canvas
// Node templates -> click the one to add
//
// To delete/rename:
// Right click the canvas
// Node templates -> Manage
//
// To rearrange:
// Open the manage dialog and Drag and drop elements using the "Name:" label as handle

const id = "Comfy.NodeTemplates";
const file = "comfy.templates.json";

class ManageTemplates extends ComfyDialog {
	constructor() {
		super();
		this.load().then((v) => {
			this.templates = v;
		});

		this.element.classList.add("comfy-manage-templates");
		this.draggedEl = null;
		this.saveVisualCue = null;
		this.emptyImg = new Image();
		this.emptyImg.src = "data:image/gif;base64,R0lGODlhAQABAIAAAAUEBAAAACwAAAAAAQABAAACAkQBADs=";

		this.importInput = $el("input", {
			type: "file",
			accept: ".json",
			multiple: true,
			style: { display: "none" },
			parent: document.body,
			onchange: () => this.importAll(),
		});
	}

	createButtons() {
		const btns = super.createButtons();
		btns[0].textContent = "Close";
		btns[0].onclick = (e) => {
			clearTimeout(this.saveVisualCue);
			this.close();
		};
		btns.unshift(
			$el("button", {
				type: "button",
				textContent: "Export",
				onclick: () => this.exportAll(),
			})
		);
		btns.unshift(
			$el("button", {
				type: "button",
				textContent: "Import",
				onclick: () => {
					this.importInput.click();
				},
			})
		);
		return btns;
	}

	async load() {
		let templates = [];
		if (app.storageLocation === "server") {
			if (app.isNewUserSession) {
				// New user so migrate existing templates
				const json = localStorage.getItem(id);
				if (json) {
					templates = JSON.parse(json);
				}
				await api.storeUserData(file, json, { stringify: false });
			} else {
				const res = await api.getUserData(file);
				if (res.status === 200) {
					try {
						templates = await res.json();
					} catch (error) {
					}
				} else if (res.status !== 404) {
					console.error(res.status + " " + res.statusText);
				}
			}
		} else {
			const json = localStorage.getItem(id);
			if (json) {
				templates = JSON.parse(json);
			}
		}

		return templates ?? [];
	}

	async store() {
		if(app.storageLocation === "server") {
			const templates = JSON.stringify(this.templates, undefined, 4);
			localStorage.setItem(id, templates); // Backwards compatibility
			try {
				await api.storeUserData(file, templates, { stringify: false });
			} catch (error) {
				console.error(error);
				alert(error.message);
			}
		} else {
			localStorage.setItem(id, JSON.stringify(this.templates));
		}
	}

	async importAll() {
		for (const file of this.importInput.files) {
			if (file.type === "application/json" || file.name.endsWith(".json")) {
				const reader = new FileReader();
				reader.onload = async () => {
					const importFile = JSON.parse(reader.result);
					if (importFile?.templates) {
						for (const template of importFile.templates) {
							if (template?.name && template?.data) {
								this.templates.push(template);
							}
						}
						await this.store();
					}
				};
				await reader.readAsText(file);
			}
		}

		this.importInput.value = null;

		this.close();
	}

	exportAll() {
		if (this.templates.length == 0) {
			alert("No templates to export.");
			return;
		}

		const json = JSON.stringify({ templates: this.templates }, null, 2); // convert the data to a JSON string
		const blob = new Blob([json], { type: "application/json" });
		const url = URL.createObjectURL(blob);
		const a = $el("a", {
			href: url,
			download: "node_templates.json",
			style: { display: "none" },
			parent: document.body,
		});
		a.click();
		setTimeout(function () {
			a.remove();
			window.URL.revokeObjectURL(url);
		}, 0);
	}

	show() {
		// Show list of template names + delete button
		super.show(
			$el(
				"div",
				{},
				this.templates.flatMap((t,i) => {
					let nameInput;
					return [
						$el(
							"div",
							{
								dataset: { id: i },
								className: "tempateManagerRow",
								style: {
									display: "grid",
									gridTemplateColumns: "1fr auto",
									border: "1px dashed transparent",
									gap: "5px",
									backgroundColor: "var(--comfy-menu-bg)"
								},
								ondragstart: (e) => {
									this.draggedEl = e.currentTarget;
									e.currentTarget.style.opacity = "0.6";
									e.currentTarget.style.border = "1px dashed yellow";
									e.dataTransfer.effectAllowed = 'move';
									e.dataTransfer.setDragImage(this.emptyImg, 0, 0);
								},
								ondragend: (e) => {
									e.target.style.opacity = "1";
									e.currentTarget.style.border = "1px dashed transparent";
									e.currentTarget.removeAttribute("draggable");

									// rearrange the elements
									this.element.querySelectorAll('.tempateManagerRow').forEach((el,i) => {
										var prev_i = el.dataset.id;

										if ( el == this.draggedEl && prev_i != i ) {
											this.templates.splice(i, 0, this.templates.splice(prev_i, 1)[0]);
										}
										el.dataset.id = i;
									});
									this.store();
								},
								ondragover: (e) => {
									e.preventDefault();
									if ( e.currentTarget == this.draggedEl )
										return;

									let rect = e.currentTarget.getBoundingClientRect();
									if (e.clientY > rect.top + rect.height / 2) {
										e.currentTarget.parentNode.insertBefore(this.draggedEl, e.currentTarget.nextSibling);
									} else {
										e.currentTarget.parentNode.insertBefore(this.draggedEl, e.currentTarget);
									}
								}
							},
							[
								$el(
									"label",
									{
										textContent: "Name: ",
										style: {
											cursor: "grab",
										},
										onmousedown: (e) => {
											// enable dragging only from the label
											if (e.target.localName == 'label')
												e.currentTarget.parentNode.draggable = 'true';
										}
									},
									[
										$el("input", {
											value: t.name,
											dataset: { name: t.name },
											style: {
												transitionProperty: 'background-color',
												transitionDuration: '0s',
											},
											onchange: (e) => {
												clearTimeout(this.saveVisualCue);
												var el = e.target;
												var row = el.parentNode.parentNode;
												this.templates[row.dataset.id].name = el.value.trim() || 'untitled';
												this.store();
												el.style.backgroundColor = 'rgb(40, 95, 40)';
												el.style.transitionDuration = '0s';
												this.saveVisualCue = setTimeout(function () {
													el.style.transitionDuration = '.7s';
													el.style.backgroundColor = 'var(--comfy-input-bg)';
												}, 15);
											},
											onkeypress: (e) => {
												var el = e.target;
												clearTimeout(this.saveVisualCue);
												el.style.transitionDuration = '0s';
												el.style.backgroundColor = 'var(--comfy-input-bg)';
											},
											$: (el) => (nameInput = el),
										})
									]
								),
								$el(
									"div",
									{},
									[
										$el("button", {
											textContent: "Export",
											style: {
												fontSize: "12px",
												fontWeight: "normal",
											},
											onclick: (e) => {
												const json = JSON.stringify({templates: [t]}, null, 2); // convert the data to a JSON string
												const blob = new Blob([json], {type: "application/json"});
												const url = URL.createObjectURL(blob);
												const a = $el("a", {
													href: url,
													download: (nameInput.value || t.name) + ".json",
													style: {display: "none"},
													parent: document.body,
												});
												a.click();
												setTimeout(function () {
													a.remove();
													window.URL.revokeObjectURL(url);
												}, 0);
											},
										}),
										$el("button", {
											textContent: "Delete",
											style: {
												fontSize: "12px",
												color: "red",
												fontWeight: "normal",
											},
											onclick: (e) => {
												const item = e.target.parentNode.parentNode;
												item.parentNode.removeChild(item);
												this.templates.splice(item.dataset.id*1, 1);
												this.store();
												// update the rows index, setTimeout ensures that the list is updated
												var that = this;
												setTimeout(function (){
													that.element.querySelectorAll('.tempateManagerRow').forEach((el,i) => {
														el.dataset.id = i;
													});
												}, 0);
											},
										}),
									]
								),
							]
						)
					];
				})
			)
		);
	}
}

app.registerExtension({
	name: id,
	setup() {
		const manage = new ManageTemplates();

		const clipboardAction = async (cb) => {
			// We use the clipboard functions but dont want to overwrite the current user clipboard
			// Restore it after we've run our callback
			const old = localStorage.getItem("litegrapheditor_clipboard");
			await cb();
			localStorage.setItem("litegrapheditor_clipboard", old);
		};

		const orig = LGraphCanvas.prototype.getCanvasMenuOptions;
		LGraphCanvas.prototype.getCanvasMenuOptions = function () {
			const options = orig.apply(this, arguments);

			options.push(null);
			options.push({
				content: `Save Selected as Template`,
				disabled: !Object.keys(app.canvas.selected_nodes || {}).length,
				callback: () => {
					const name = prompt("Enter name");
					if (!name?.trim()) return;

					clipboardAction(() => {
						app.canvas.copyToClipboard();
						let data = localStorage.getItem("litegrapheditor_clipboard");
						data = JSON.parse(data);
						const nodeIds = Object.keys(app.canvas.selected_nodes);
						for (let i = 0; i < nodeIds.length; i++) {
							const node = app.graph.getNodeById(nodeIds[i]);
							const nodeData = node?.constructor.nodeData;
							
							let groupData = GroupNodeHandler.getGroupData(node);
							if (groupData) {
								groupData = groupData.nodeData;
								if (!data.groupNodes) {
									data.groupNodes = {};
								}
								data.groupNodes[nodeData.name] = groupData;
								data.nodes[i].type = nodeData.name;
							}
						}

						manage.templates.push({
							name,
							data: JSON.stringify(data),
						});
						manage.store();
					});
				},
			});

			// Map each template to a menu item
			const subItems = manage.templates.map((t) => {
				return {
					content: t.name,
					callback: () => {
						clipboardAction(async () => {
							const data = JSON.parse(t.data);
							await GroupNodeConfig.registerFromWorkflow(data.groupNodes, {});
							localStorage.setItem("litegrapheditor_clipboard", t.data);
							app.canvas.pasteFromClipboard();
						});
					},
				};
			});

			subItems.push(null, {
				content: "Manage",
				callback: () => manage.show(),
			});

			options.push({
				content: "Node Templates",
				submenu: {
					options: subItems,
				},
			});

			return options;
		};
	},
});
