import { $el } from "./helpers.js";
import { ComfySplitButton } from "./comfySplitButton.js";
import { api } from "../api.js";

export class ComfyWorkflows {
	async load() {
		this.workflows = await api.get_workflows();
		this.load_button.arrow.style.display = this.workflows.length ? "flex" : "none";
	}

	constructor() {
		function build_menu(workflows) {
			var menu = [];
			var directories = new Map();
			for (const workflow of workflows) {
				var path = workflow.split("/");
				var parent = menu;
				var current_path = "";
				for (var i = 0; i < path.length - 1; i++) {
					current_path += "/" + path[i];
					var new_parent = directories.get(current_path);
					if (!new_parent) {
						new_parent = {
							title: path[i],
							has_submenu: true,
							submenu: {
								options: [],
							},
						};
						parent.push(new_parent);
						new_parent = new_parent.submenu.options;
						directories.set(current_path, new_parent);
					}
					parent = new_parent;
				}
				parent.push({
					title: path[path.length - 1],
					callback: async () => {
						const json = await api.get_workflow(workflow);
						app.loadGraphData(json);
					},
				});
			}
			return menu;
		}

		const fileInput = $el("input", {
			type: "file",
			accept: ".json,image/png",
			style: { display: "none" },
			parent: document.body,
			onchange: () => {
				app.handleFile(fileInput.files[0]);
			},
		});

		this.load_button = new ComfySplitButton("Load", () => fileInput.click(), {
			get_options: () => {
				return build_menu(this.workflows || []);
			},
		});

		function save(name) {
			const json = JSON.stringify(app.graph.serialize(), null, 2); // convert the data to a JSON string
			const blob = new Blob([json], { type: "application/json" });
			const url = URL.createObjectURL(blob);
			const a = $el("a", {
				href: url,
				download: name + ".json",
				style: { display: "none" },
				parent: document.body,
			});
			a.click();
			setTimeout(function () {
				a.remove();
				window.URL.revokeObjectURL(url);
			}, 0);
		}

		this.save_button = new ComfySplitButton("Save", () => save("workflow"), {
			get_options: () => {
				return [
					{
						title: "Save as",
						callback() {
							var name = prompt("Enter filename", "workflow");
							if (name) {
								save(name);
							}
						},
					},
					{
						title: "Save to workflows",
						callback: async () => {
							var name = prompt("Enter filename", "workflow");
							if (name) {
								var data = app.graph.serialize();
								if (!(await api.save_workflow(name, data))) {
									if (confirm("A workspace with this name already exists, do you want to overwrite it?")) {
										await api.save_workflow(name, app.graph.serialize(), true);
									} else {
										return;
									}
								}
								await this.load();
							}
						},
					},
				];
			},
		});

		this.load();
	}
}
