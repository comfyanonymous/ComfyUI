import { app } from "../../scripts/app.js";
import { ComfyDialog, $el } from "../../scripts/ui.js";

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

const id = "Comfy.NodeTemplates";

class ManageTemplates extends ComfyDialog {
	constructor() {
		super();
		this.element.classList.add("comfy-manage-templates");
		this.templates = this.load();

		this.importInput = $el("input", {
			type: "file",
			accept: ".json",
			multiple: true,
			style: {display: "none"},
			parent: document.body,
			onchange: () => this.importAll(),
		});
	}

	createButtons() {
		const btns = super.createButtons();
		btns[0].textContent = "Cancel";
		btns.unshift(
			$el("button", {
				type: "button",
				textContent: "Save",
				onclick: () => this.save(),
			})
		);
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

	load() {
		const templates = localStorage.getItem(id);
		if (templates) {
			return JSON.parse(templates);
		} else {
			return [];
		}
	}

	save() {
		// Find all visible inputs and save them as our new list
		const inputs = this.element.querySelectorAll("input");
		const updated = [];

		for (let i = 0; i < inputs.length; i++) {
			const input = inputs[i];
			if (input.parentElement.style.display !== "none") {
				const t = this.templates[i];
				t.name = input.value.trim() || input.getAttribute("data-name");
				updated.push(t);
			}
		}

		this.templates = updated;
		this.store();
		this.close();
	}

	store() {
		localStorage.setItem(id, JSON.stringify(this.templates));
	}

	async importAll() {
		for (const file of this.importInput.files) {
			if (file.type === "application/json" || file.name.endsWith(".json")) {
				const reader = new FileReader();
				reader.onload = async () => {
					var importFile = JSON.parse(reader.result);
					if (importFile && importFile?.templates) {
						for (const template of importFile.templates) {
							if (template?.name && template?.data) {
								this.templates.push(template);
							}
						}
						this.store();
					}
				};
				await reader.readAsText(file);
			}
		}

		this.close();
	}

	exportAll() {
		if (this.templates.length == 0) {
			alert("No templates to export.");
			return;
		}

		const json = JSON.stringify({templates: this.templates}, null, 2); // convert the data to a JSON string
		const blob = new Blob([json], {type: "application/json"});
		const url = URL.createObjectURL(blob);
		const a = $el("a", {
			href: url,
			download: "node_templates.json",
			style: {display: "none"},
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
				{
					style: {
						display: "grid",
						gridTemplateColumns: "1fr auto",
						gap: "5px",
					},
				},
				this.templates.flatMap((t) => {
					let nameInput;
					return [
						$el(
							"label",
							{
								textContent: "Name: ",
							},
							[
								$el("input", {
									value: t.name,
									dataset: { name: t.name },
									$: (el) => (nameInput = el),
								}),
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
											download: t.name + ".json",
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
										nameInput.value = "";
										e.target.parentElement.style.display = "none";
										e.target.parentElement.previousElementSibling.style.display = "none";
									},
								}),
							]
						),
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

		const clipboardAction = (cb) => {
			// We use the clipboard functions but dont want to overwrite the current user clipboard
			// Restore it after we've run our callback
			const old = localStorage.getItem("litegrapheditor_clipboard");
			cb();
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
					if (!name || !name.trim()) return;

					clipboardAction(() => {
						app.canvas.copyToClipboard();
						manage.templates.push({
							name,
							data: localStorage.getItem("litegrapheditor_clipboard"),
						});
						manage.store();
					});
				},
			});

			// Map each template to a menu item
			const subItems = manage.templates.map((t) => ({
				content: t.name,
				callback: () => {
					clipboardAction(() => {
						localStorage.setItem("litegrapheditor_clipboard", t.data);
						app.canvas.pasteFromClipboard();
					});
				},
			}));

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
