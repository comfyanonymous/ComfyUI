import { $el, ComfyDialog } from "../../scripts/ui.js";
import { DraggableList } from "../../scripts/ui/draggableList.js";
import { addStylesheet } from "../../scripts/utils.js";
import { GroupNodeConfig, GroupNodeHandler } from "./groupNode.js";

addStylesheet(import.meta.url);

const ORDER = Symbol();

function merge(target, source) {
	if (typeof target === "object" && typeof source === "object") {
		for (const key in source) {
			const sv = source[key];
			if (typeof sv === "object") {
				let tv = target[key];
				if (!tv) tv = target[key] = {};
				merge(tv, source[key]);
			} else {
				target[key] = sv;
			}
		}
	}

	return target;
}

export class ManageGroupDialog extends ComfyDialog {
	/** @type { Record<"Inputs" | "Outputs" | "Widgets", {tab: HTMLAnchorElement, page: HTMLElement}> } */
	tabs = {};
	/** @type { number | null | undefined } */
	selectedNodeIndex;
	/** @type { keyof ManageGroupDialog["tabs"] } */
	selectedTab = "Inputs";
	/** @type { string | undefined } */
	selectedGroup;

	/** @type { Record<string, Record<string, Record<string, { name?: string | undefined, visible?: boolean | undefined }>>> } */
	modifications = {};

	get selectedNodeInnerIndex() {
		return +this.nodeItems[this.selectedNodeIndex].dataset.nodeindex;
	}

	constructor(app) {
		super();
		this.app = app;
		this.element = $el("dialog.comfy-group-manage", {
			parent: document.body,
		});
	}

	changeTab(tab) {
		this.tabs[this.selectedTab].tab.classList.remove("active");
		this.tabs[this.selectedTab].page.classList.remove("active");
		this.tabs[tab].tab.classList.add("active");
		this.tabs[tab].page.classList.add("active");
		this.selectedTab = tab;
	}

	changeNode(index, force) {
		if (!force && this.selectedNodeIndex === index) return;

		if (this.selectedNodeIndex != null) {
			this.nodeItems[this.selectedNodeIndex].classList.remove("selected");
		}
		this.nodeItems[index].classList.add("selected");
		this.selectedNodeIndex = index;

		if (!this.buildInputsPage() && this.selectedTab === "Inputs") {
			this.changeTab("Widgets");
		}
		if (!this.buildWidgetsPage() && this.selectedTab === "Widgets") {
			this.changeTab("Outputs");
		}
		if (!this.buildOutputsPage() && this.selectedTab === "Outputs") {
			this.changeTab("Inputs");
		}

		this.changeTab(this.selectedTab);
	}

	getGroupData() {
		this.groupNodeType = LiteGraph.registered_node_types["workflow/" + this.selectedGroup];
		this.groupNodeDef = this.groupNodeType.nodeData;
		this.groupData = GroupNodeHandler.getGroupData(this.groupNodeType);
	}

	changeGroup(group, reset = true) {
		this.selectedGroup = group;
		this.getGroupData();

		const nodes = this.groupData.nodeData.nodes;
		this.nodeItems = nodes.map((n, i) =>
			$el(
				"li.draggable-item",
				{
					dataset: {
						nodeindex: n.index + "",
					},
					onclick: () => {
						this.changeNode(i);
					},
				},
				[
					$el("span.drag-handle"),
					$el(
						"div",
						{
							textContent: n.title ?? n.type,
						},
						n.title
							? $el("span", {
									textContent: n.type,
							  })
							: []
					),
				]
			)
		);

		this.innerNodesList.replaceChildren(...this.nodeItems);

		if (reset) {
			this.selectedNodeIndex = null;
			this.changeNode(0);
		} else {
			const items = this.draggable.getAllItems();
			let index = items.findIndex(item => item.classList.contains("selected"));
			if(index === -1) index = this.selectedNodeIndex;
			this.changeNode(index, true);
		}

		const ordered = [...nodes];
		this.draggable?.dispose();
		this.draggable = new DraggableList(this.innerNodesList, "li");
		this.draggable.addEventListener("dragend", ({ detail: { oldPosition, newPosition } }) => {
			if (oldPosition === newPosition) return;
			ordered.splice(newPosition, 0, ordered.splice(oldPosition, 1)[0]);
			for (let i = 0; i < ordered.length; i++) {
				this.storeModification({ nodeIndex: ordered[i].index, section: ORDER, prop: "order", value: i });
			}
		});
	}

	storeModification({ nodeIndex, section, prop, value }) {
		const groupMod = (this.modifications[this.selectedGroup] ??= {});
		const nodesMod = (groupMod.nodes ??= {});
		const nodeMod = (nodesMod[nodeIndex ?? this.selectedNodeInnerIndex] ??= {});
		const typeMod = (nodeMod[section] ??= {});
		if (typeof value === "object") {
			const objMod = (typeMod[prop] ??= {});
			Object.assign(objMod, value);
		} else {
			typeMod[prop] = value;
		}
	}

	getEditElement(section, prop, value, placeholder, checked, checkable = true) {
		if (value === placeholder) value = "";

		const mods = this.modifications[this.selectedGroup]?.nodes?.[this.selectedNodeInnerIndex]?.[section]?.[prop];
		if (mods) {
			if (mods.name != null) {
				value = mods.name;
			}
			if (mods.visible != null) {
				checked = mods.visible;
			}
		}

		return $el("div", [
			$el("input", {
				value,
				placeholder,
				type: "text",
				onchange: (e) => {
					this.storeModification({ section, prop, value: { name: e.target.value } });
				},
			}),
			$el("label", { textContent: "Visible" }, [
				$el("input", {
					type: "checkbox",
					checked,
					disabled: !checkable,
					onchange: (e) => {
						this.storeModification({ section, prop, value: { visible: !!e.target.checked } });
					},
				}),
			]),
		]);
	}

	buildWidgetsPage() {
		const widgets = this.groupData.oldToNewWidgetMap[this.selectedNodeInnerIndex];
		const items = Object.keys(widgets ?? {});
		const type = app.graph.extra.groupNodes[this.selectedGroup];
		const config = type.config?.[this.selectedNodeInnerIndex]?.input;
		this.widgetsPage.replaceChildren(
			...items.map((oldName) => {
				return this.getEditElement("input", oldName, widgets[oldName], oldName, config?.[oldName]?.visible !== false);
			})
		);
		return !!items.length;
	}

	buildInputsPage() {
		const inputs = this.groupData.nodeInputs[this.selectedNodeInnerIndex];
		const items = Object.keys(inputs ?? {});
		const type = app.graph.extra.groupNodes[this.selectedGroup];
		const config = type.config?.[this.selectedNodeInnerIndex]?.input;
		this.inputsPage.replaceChildren(
			...items
				.map((oldName) => {
					let value = inputs[oldName];
					if (!value) {
						return;
					}

					return this.getEditElement("input", oldName, value, oldName, config?.[oldName]?.visible !== false);
				})
				.filter(Boolean)
		);
		return !!items.length;
	}

	buildOutputsPage() {
		const nodes = this.groupData.nodeData.nodes;
		const innerNodeDef = this.groupData.getNodeDef(nodes[this.selectedNodeInnerIndex]);
		const outputs = innerNodeDef?.output ?? [];
		const groupOutputs = this.groupData.oldToNewOutputMap[this.selectedNodeInnerIndex];

		const type = app.graph.extra.groupNodes[this.selectedGroup];
		const config = type.config?.[this.selectedNodeInnerIndex]?.output;
		const node = this.groupData.nodeData.nodes[this.selectedNodeInnerIndex];
		const checkable = node.type !== "PrimitiveNode";
		this.outputsPage.replaceChildren(
			...outputs
				.map((type, slot) => {
					const groupOutputIndex = groupOutputs?.[slot];
					const oldName = innerNodeDef.output_name?.[slot] ?? type;
					let value = config?.[slot]?.name;
					const visible = config?.[slot]?.visible || groupOutputIndex != null;
					if (!value || value === oldName) {
						value = "";
					}
					return this.getEditElement("output", slot, value, oldName, visible, checkable);
				})
				.filter(Boolean)
		);
		return !!outputs.length;
	}

	show(type) {
		const groupNodes = Object.keys(app.graph.extra?.groupNodes ?? {}).sort((a, b) => a.localeCompare(b));

		this.innerNodesList = $el("ul.comfy-group-manage-list-items");
		this.widgetsPage = $el("section.comfy-group-manage-node-page");
		this.inputsPage = $el("section.comfy-group-manage-node-page");
		this.outputsPage = $el("section.comfy-group-manage-node-page");
		const pages = $el("div", [this.widgetsPage, this.inputsPage, this.outputsPage]);

		this.tabs = [
			["Inputs", this.inputsPage],
			["Widgets", this.widgetsPage],
			["Outputs", this.outputsPage],
		].reduce((p, [name, page]) => {
			p[name] = {
				tab: $el("a", {
					onclick: () => {
						this.changeTab(name);
					},
					textContent: name,
				}),
				page,
			};
			return p;
		}, {});

		const outer = $el("div.comfy-group-manage-outer", [
			$el("header", [
				$el("h2", "Group Nodes"),
				$el(
					"select",
					{
						onchange: (e) => {
							this.changeGroup(e.target.value);
						},
					},
					groupNodes.map((g) =>
						$el("option", {
							textContent: g,
							selected: "workflow/" + g === type,
							value: g,
						})
					)
				),
			]),
			$el("main", [
				$el("section.comfy-group-manage-list", this.innerNodesList),
				$el("section.comfy-group-manage-node", [
					$el(
						"header",
						Object.values(this.tabs).map((t) => t.tab)
					),
					pages,
				]),
			]),
			$el("footer", [
				$el(
					"button.comfy-btn",
					{
						onclick: (e) => {
							const node = app.graph._nodes.find((n) => n.type === "workflow/" + this.selectedGroup);
							if (node) {
								alert("This group node is in use in the current workflow, please first remove these.");
								return;
							}
							if (confirm(`Are you sure you want to remove the node: "${this.selectedGroup}"`)) {
								delete app.graph.extra.groupNodes[this.selectedGroup];
								LiteGraph.unregisterNodeType("workflow/" + this.selectedGroup);
							}
							this.show();
						},
					},
					"Delete Group Node"
				),
				$el(
					"button.comfy-btn",
					{
						onclick: async () => {
							let nodesByType;
							let recreateNodes = [];
							const types = {};
							for (const g in this.modifications) {
								const type = app.graph.extra.groupNodes[g];
								let config = (type.config ??= {});

								let nodeMods = this.modifications[g]?.nodes;
								if (nodeMods) {
									const keys = Object.keys(nodeMods);
									if (nodeMods[keys[0]][ORDER]) {
										// If any node is reordered, they will all need sequencing
										const orderedNodes = [];
										const orderedMods = {};
										const orderedConfig = {};

										for (const n of keys) {
											const order = nodeMods[n][ORDER].order;
											orderedNodes[order] = type.nodes[+n];
											orderedMods[order] = nodeMods[n];
											orderedNodes[order].index = order;
										}

										// Rewrite links
										for (const l of type.links) {
											if (l[0] != null) l[0] = type.nodes[l[0]].index;
											if (l[2] != null) l[2] = type.nodes[l[2]].index;
										}

										// Rewrite externals
										if (type.external) {
											for (const ext of type.external) {
												ext[0] = type.nodes[ext[0]];
											}
										}

										// Rewrite modifications
										for (const id of keys) {
											if (config[id]) {
												orderedConfig[type.nodes[id].index] = config[id];
											}
											delete config[id];
										}

										type.nodes = orderedNodes;
										nodeMods = orderedMods;
										type.config = config = orderedConfig;
									}

									merge(config, nodeMods);
								}

								types[g] = type;

								if (!nodesByType) {
									nodesByType = app.graph._nodes.reduce((p, n) => {
										p[n.type] ??= [];
										p[n.type].push(n);
										return p;
									}, {});
								}

								const nodes = nodesByType["workflow/" + g];
								if (nodes) recreateNodes.push(...nodes);
							}

							await GroupNodeConfig.registerFromWorkflow(types, {});

							for (const node of recreateNodes) {
								node.recreate();
							}

							this.modifications = {};
							this.app.graph.setDirtyCanvas(true, true);
							this.changeGroup(this.selectedGroup, false);
						},
					},
					"Save"
				),
				$el("button.comfy-btn", { onclick: () => this.element.close() }, "Close"),
			]),
		]);

		this.element.replaceChildren(outer);
		this.changeGroup(type ? groupNodes.find((g) => "workflow/" + g === type) : groupNodes[0]);
		this.element.showModal();

		this.element.addEventListener("close", () => {
			this.draggable?.dispose();
		});
	}
}