// @ts-check

import { $el } from "../../ui.js";
import { addStylesheet, downloadBlob } from "../../utils.js";
import { ComfyButton } from "../components/button.js";
import { ComfyButtonGroup } from "../components/buttonGroup.js";
import { ComfySplitButton } from "../components/splitButton.js";
import { ComfyQueueButton } from "./queueButton.js";
import { ComfyWorkflows } from "./workflows.js";

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
		this.workflows = new ComfyWorkflows();
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
					visibilitySetting: { id: "Comfy.DevMode", showValue: "true" },
					app
				})
			).element,
			collapseOnMobile(
				new ComfyButtonGroup(
					new ComfyButton({
						icon: "refresh",
						content: "Refresh",
						action: () => {
							console.log("refresh");
						},
					}),
					new ComfyButton({
						icon: "clipboard-edit-outline",
						content: "Clipspace",
						action: () => {
							console.log("clipboard-edit-outline");
						},
					}),
					new ComfyButton({
						icon: "cancel",
						content: "Clear",
						action: () => {
							console.log("cancel");
						},
					})
				)
			).element,
			$el("section.comfyui-menu-push"),
			collapseOnMobile(
				new ComfyButton({
					icon: "cog",
					content: "Settings",
					action: () => {
						app.ui.settings.show();
					},
				})
			).element,
			collapseOnMobile(
				new ComfyButtonGroup(
					new ComfyButton({
						icon: "history",
						content: "View History",
						action: () => {
							console.log("history");
						},
					}),
					new ComfyButton({
						icon: "format-list-numbered",
						content: "View Queue",
						action: () => {
							console.log("format-list-numbered");
						},
					})
				)
			).element,
			new ComfyQueueButton().element,
			showOnMobile(
				new ComfyButton({
					icon: "menu",
					action: (_, btn) => {
						btn.icon = this.element.classList.toggle("expanded") ? "close" : "menu";
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
		if (this.app.ui.settings.getSettingValue("Comfy.PromptFilename")) {
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
