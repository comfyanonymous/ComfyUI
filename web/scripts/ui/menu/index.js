// @ts-check

import { $el } from "../../ui.js";
import { addStylesheet, downloadBlob } from "../../utils.js";
import { ComfyButton } from "../components/button.js";
import { ComfyButtonGroup } from "../components/buttonGroup.js";
import { ComfySplitButton } from "../components/splitButton.js";
import { ComfyViewHistoryButton } from "./viewHistory.js";
import { ComfyQueueButton } from "./queueButton.js";
import { ComfyWorkflowsMenu } from "./workflows.js";
import { ComfyViewQueueButton } from "./viewQueue.js";
import { getInteruptButton } from "./interruptButton.js";

addStylesheet("menu.css", import.meta.url);

const collapseOnMobile = (t) => {
	(t.element ?? t).classList.add("comfyui-menu-mobile-collapse");
	return t;
};
const showOnMobile = (t) => {
	(t.element ?? t).classList.add("sm-show");
	return t;
};

export class ComfyAppMenu {
	/**
	 * @param { import("../../app.js").ComfyApp } app
	 */
	constructor(app) {
		this.app = app;

		const getSaveButton = (t) =>
			new ComfyButton({
				icon: "content-save",
				tooltip: "Save the current workflow",
				action: () => this.exportWorkflow("workflow", "workflow"),
				content: t,
			});

		this.logo = $el("h1.comfyui-logo.sm-hide", { title: "ComfyUI" }, "ComfyUI");
		this.workflows = new ComfyWorkflowsMenu();
		this.element = $el("nav.comfyui-menu", { parent: document.body }, [
			this.logo,
			this.workflows.element,
			new ComfySplitButton(
				{
					primary: getSaveButton(),
					mode: "hover",
				},
				getSaveButton("Save"),
				new ComfyButton({
					icon: "content-save-edit",
					content: "Save As",
					tooltip: "Save the current graph as a new workflow",
					action: () => {
						let filename = prompt("Save workflow as:", "workflow");
						if (!filename) return;
						if (!filename.toLowerCase().endsWith(".json")) {
							filename += ".json";
						}

						this.exportWorkflow("workflow", "workflow");
					},
				}),
				new ComfyButton({
					icon: "download",
					content: "Export",
					tooltip: "Export the current workflow as JSON",
					action: () => this.exportWorkflow("workflow", "workflow"),
				}),
				new ComfyButton({
					icon: "api",
					content: "Export (API Format)",
					tooltip: "Export the current workflow as JSON for use with the ComfyUI API",
					action: () => this.exportWorkflow("workflow_api", "output"),
					visibilitySetting: { id: "Comfy.DevMode", showValue: true },
					app,
				})
			).element,
			collapseOnMobile(
				new ComfyButtonGroup(
					new ComfyButton({
						icon: "refresh",
						content: "Refresh",
						tooltip: "Refresh widgets in nodes to find new models or files",
						action: () => app.refreshComboInNodes(),
					}),
					new ComfyButton({
						icon: "clipboard-edit-outline",
						content: "Clipspace",
						tooltip: "Open Clipspace window",
						action: () => app["openClipspace"](),
					}),
					new ComfyButton({
						icon: "cancel",
						content: "Clear",
						tooltip: "Clears current workflow",
						action: () => {
							if (!app.ui.settings.getSettingValue("Comfy.ConfirmClear", true) || confirm("Clear workflow?")) {
								app.clean();
								app.graph.clear();
							}
						},
					})
				)
			).element,
			$el("section.comfyui-menu-push"),
			collapseOnMobile(
				new ComfyButton({
					icon: "cog",
					content: "Settings",
					tooltip: "Open settings",
					action: () => {
						app.ui.settings.show();
					},
				})
			).element,
			collapseOnMobile(
				new ComfyButtonGroup(
					new ComfyViewHistoryButton(app).element,
					new ComfyViewQueueButton(app).element,
					getInteruptButton("sm-hide").element
				)
			).element,

			getInteruptButton("sm-show").element,
			new ComfyQueueButton(app).element,
			showOnMobile(
				new ComfyButton({
					icon: "menu",
					action: (_, btn) => {
						btn.icon = this.element.classList.toggle("expanded") ? "menu-open" : "menu";
					},
					classList: "comfyui-button comfyui-menu-button",
				})
			).element,
		]);
	}

	/**
	 * @param {string} defaultName
	 */
	getFilename(defaultName) {
		if (this.app.ui.settings.getSettingValue("Comfy.PromptFilename", true)) {
			defaultName = prompt("Save workflow as:", defaultName);
			if (!defaultName) return;
			if (!defaultName.toLowerCase().endsWith(".json")) {
				defaultName += ".json";
			}
		}
		return defaultName;
	}

	/**
	 * @param {string} [filename]
	 * @param { "workflow" | "output" } [promptProperty]
	 */
	async exportWorkflow(filename, promptProperty) {
		const p = await this.app.graphToPrompt();
		const json = JSON.stringify(p[promptProperty], null, 2);
		const blob = new Blob([json], { type: "application/json" });
		const file = this.getFilename(filename);
		downloadBlob(file, blob);
	}
}
