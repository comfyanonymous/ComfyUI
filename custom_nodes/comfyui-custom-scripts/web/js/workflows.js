import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import { $el } from "../../../scripts/ui.js";

// Adds workflow management
// Original implementation by https://github.com/i-h4x
// Thanks for permission to reimplement as an extension

const style = `
#comfy-save-button, #comfy-load-button {
   position: relative;
   overflow: hidden;
}
.pysssss-workflow-arrow {
   position: absolute;
   top: 0;
   bottom: 0;
   right: 0;
   font-size: 12px;
   display: flex;
   align-items: center;
   width: 24px;
   justify-content: center;
   background: rgba(255,255,255,0.1);
}
.pysssss-workflow-arrow:after {
   content: "▼";
}
.pysssss-workflow-arrow:hover {
   filter: brightness(1.6);
   background-color: var(--comfy-menu-bg);
}
.pysssss-workflow-load .litemenu-entry:not(.has_submenu):before,
.pysssss-workflow-load ~ .litecontextmenu .litemenu-entry:not(.has_submenu):before {
	content: "🎛️";
	padding-right: 5px;
}
.pysssss-workflow-load .litemenu-entry.has_submenu:before,
.pysssss-workflow-load ~ .litecontextmenu .litemenu-entry.has_submenu:before {
	content: "📂";
	padding-right: 5px;
	position: relative;
	top: -1px;
}
.pysssss-workflow-popup ~ .litecontextmenu {
	transform: scale(1.3);
}
`;

async function getWorkflows() {
	const response = await api.fetchApi("/pysssss/workflows", { cache: "no-store" });
	return await response.json();
}

async function getWorkflow(name) {
	const response = await api.fetchApi(`/pysssss/workflows/${encodeURIComponent(name)}`, { cache: "no-store" });
	return await response.json();
}

async function saveWorkflow(name, workflow, overwrite) {
	try {
		const response = await api.fetchApi("/pysssss/workflows", {
			method: "POST",
			headers: {
				"Content-Type": "application/json",
			},
			body: JSON.stringify({ name, workflow, overwrite }),
		});
		if (response.status === 201) {
			return true;
		}
		if (response.status === 409) {
			return false;
		}
		throw new Error(response.statusText);
	} catch (error) {
		console.error(error);
	}
}

class PysssssWorkflows {
	async load() {
		this.workflows = await getWorkflows();
		if(this.workflows.length) {
			this.workflows.sort();
		}
		this.loadMenu.style.display = this.workflows.length ? "flex" : "none";
	}

	getMenuOptions(callback) {
		const menu = [];
		const directories = new Map();
		for (const workflow of this.workflows || []) {
			const path = workflow.split("/");
			let parent = menu;
			let currentPath = "";
			for (let i = 0; i < path.length - 1; i++) {
				currentPath += "/" + path[i];
				let newParent = directories.get(currentPath);
				if (!newParent) {
					newParent = {
						title: path[i],
						has_submenu: true,
						submenu: {
							options: [],
						},
					};
					parent.push(newParent);
					newParent = newParent.submenu.options;
					directories.set(currentPath, newParent);
				}
				parent = newParent;
			}
			parent.push({
				title: path[path.length - 1],
				callback: () => callback(workflow),
			});
		}
		return menu;
	}

	constructor() {
		function addWorkflowMenu(type, getOptions) {
			return $el("div.pysssss-workflow-arrow", {
				parent: document.getElementById(`comfy-${type}-button`),
				onclick: (e) => {
					e.preventDefault();
					e.stopPropagation();

					LiteGraph.closeAllContextMenus();
					const menu = new LiteGraph.ContextMenu(
						getOptions(),
						{
							event: e,
							scale: 1.3,
						},
						window
					);
					menu.root.classList.add("pysssss-workflow-popup");
					menu.root.classList.add(`pysssss-workflow-${type}`);
				},
			});
		}

		this.loadMenu = addWorkflowMenu("load", () =>
			this.getMenuOptions(async (workflow) => {
				const json = await getWorkflow(workflow);
				app.loadGraphData(json);
			})
		);
		addWorkflowMenu("save", () => {
			return [
				{
					title: "Save as",
					callback: () => {
						let filename = prompt("Enter filename", this.workflowName || "workflow");
						if (filename) {
							if (!filename.toLowerCase().endsWith(".json")) {
								filename += ".json";
							}

							this.workflowName = filename;

							const json = JSON.stringify(app.graph.serialize(), null, 2); // convert the data to a JSON string
							const blob = new Blob([json], { type: "application/json" });
							const url = URL.createObjectURL(blob);
							const a = $el("a", {
								href: url,
								download: filename,
								style: { display: "none" },
								parent: document.body,
							});
							a.click();
							setTimeout(function () {
								a.remove();
								window.URL.revokeObjectURL(url);
							}, 0);
						}
					},
				},
				{
					title: "Save to workflows",
					callback: async () => {
						const name = prompt("Enter filename", this.workflowName || "workflow");
						if (name) {
							this.workflowName = name;

							const data = app.graph.serialize();
							if (!(await saveWorkflow(name, data))) {
								if (confirm("A workspace with this name already exists, do you want to overwrite it?")) {
									await saveWorkflow(name, app.graph.serialize(), true);
								} else {
									return;
								}
							}
							await this.load();
						}
					},
				},
			];
		});
		this.load();

		const handleFile = app.handleFile;
		const self = this;
		app.handleFile = function (file) {
			if (file?.name?.endsWith(".json")) {
				self.workflowName = file.name;
			} else {
				self.workflowName = null;
			}
			return handleFile.apply(this, arguments);
		};
	}
}

const refreshComboInNodes = app.refreshComboInNodes;
let workflows;

async function sendToWorkflow(img, workflow) {
	const graph = !workflow ? app.graph.serialize() : await getWorkflow(workflow);
	const nodes = graph.nodes.filter((n) => n.type === "LoadImage");
	let targetNode;
	if (nodes.length === 0) {
		alert("To send the image to another workflow, that workflow must have a LoadImage node.");
		return;
	} else if (nodes.length > 1) {
		targetNode = nodes.find((n) => n.title?.toLowerCase().includes("input"));
		if (!targetNode) {
			targetNode = nodes[0];
			alert(
				"The target workflow has multiple LoadImage nodes, include 'input' in the name of the one you want to use. The first one will be used here."
			);
		}
	} else {
		targetNode = nodes[0];
	}

	const blob = await (await fetch(img.src)).blob();
	const name =
		(workflow || "sendtoworkflow").replace(/\//g, "_") +
		"-" +
		+new Date() +
		new URLSearchParams(img.src.split("?")[1]).get("filename");
	const body = new FormData();
	body.append("image", new File([blob], name));

	const resp = await api.fetchApi("/upload/image", {
		method: "POST",
		body,
	});

	if (resp.status === 200) {
		await refreshComboInNodes.call(app);
		targetNode.widgets_values[0] = name;
		app.loadGraphData(graph);
		app.graph.getNodeById(targetNode.id);
	} else {
		alert(resp.status + " - " + resp.statusText);
	}
}

app.registerExtension({
	name: "pysssss.Workflows",
	init() {
		$el("style", {
			textContent: style,
			parent: document.head,
		});
	},

	async refreshComboInNodes() {
		workflows.load()
	},
	
	async setup() {
		workflows = new PysssssWorkflows();

		const comfyDefault = "[ComfyUI Default]";
		const defaultWorkflow = app.ui.settings.addSetting({
			id: "pysssss.Workflows.Default",
			name: "🐍 Default Workflow",
			defaultValue: comfyDefault,
			type: "combo",
			options: (value) =>
				[comfyDefault, ...workflows.workflows].map((m) => ({
					value: m,
					text: m,
					selected: m === value,
				})),
		});

		document.getElementById("comfy-load-default-button").onclick = async function () {
			if (
				localStorage["Comfy.Settings.Comfy.ConfirmClear"] === "false" ||
				confirm(`Load default workflow (${defaultWorkflow.value})?`)
			) {
				if (defaultWorkflow.value === comfyDefault) {
					app.loadGraphData();
				} else {
					const json = await getWorkflow(defaultWorkflow.value);
					app.loadGraphData(json);
				}
			}
		};
	},
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
		nodeType.prototype.getExtraMenuOptions = function (_, options) {
			const r = getExtraMenuOptions?.apply?.(this, arguments);
			let img;
			if (this.imageIndex != null) {
				// An image is selected so select that
				img = this.imgs[this.imageIndex];
			} else if (this.overIndex != null) {
				// No image is selected but one is hovered
				img = this.imgs[this.overIndex];
			}

			if (img) {
				let pos = options.findIndex((o) => o.content === "Save Image");
				if (pos === -1) {
					pos = 0;
				} else {
					pos++;
				}

				options.splice(pos, 0, {
					content: "Send to workflow",
					has_submenu: true,
					submenu: {
						options: [
							{ callback: () => sendToWorkflow(img), title: "[Current workflow]" },
							...workflows.getMenuOptions(sendToWorkflow.bind(null, img)),
						],
					},
				});
			}

			return r;
		};
	},
});
