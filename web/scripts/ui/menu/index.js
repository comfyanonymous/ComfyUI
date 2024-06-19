// @ts-check

import { $el } from "../../ui.js";
import { downloadBlob } from "../../utils.js";
import { ComfyButton } from "../components/button.js";
import { ComfyButtonGroup } from "../components/buttonGroup.js";
import { ComfySplitButton } from "../components/splitButton.js";
import { ComfyViewHistoryButton } from "./viewHistory.js";
import { ComfyQueueButton } from "./queueButton.js";
import { ComfyWorkflowsMenu } from "./workflows.js";
import { ComfyViewQueueButton } from "./viewQueue.js";
import { getInteruptButton } from "./interruptButton.js";

const collapseOnMobile = (t) => {
	(t.element ?? t).classList.add("comfyui-menu-mobile-collapse");
	return t;
};
const showOnMobile = (t) => {
	(t.element ?? t).classList.add("lt-lg-show");
	return t;
};

export class ComfyAppMenu {
	#sizeBreak = "lg";
	#lastSizeBreaks = {
		lg: null,
		md: null,
		sm: null,
		xs: null,
	};
	#sizeBreaks = Object.keys(this.#lastSizeBreaks);
	#cachedInnerSize = null;
	#cacheTimeout = null;

	/**
	 * @param { import("../../app.js").ComfyApp } app
	 */
	constructor(app) {
		this.app = app;

		this.workflows = new ComfyWorkflowsMenu(app);
		const getSaveButton = (t) =>
			new ComfyButton({
				icon: "content-save",
				tooltip: "Save the current workflow",
				action: () => app.workflowManager.activeWorkflow.save(),
				content: t,
			});

		this.logo = $el("h1.comfyui-logo.nlg-hide", { title: "ComfyUI" }, "ComfyUI");
		this.saveButton = new ComfySplitButton(
			{
				primary: getSaveButton(),
				mode: "hover",
				position: "absolute",
			},
			getSaveButton("Save"),
			new ComfyButton({
				icon: "content-save-edit",
				content: "Save As",
				tooltip: "Save the current graph as a new workflow",
				action: () => app.workflowManager.activeWorkflow.save(true),
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
		);
		this.actionsGroup = new ComfyButtonGroup(
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
				icon: "fit-to-page-outline",
				content: "Reset View",
				tooltip: "Reset the canvas view",
				action: () => app.resetView(),
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
		);
		this.settingsGroup = new ComfyButtonGroup(
			new ComfyButton({
				icon: "cog",
				content: "Settings",
				tooltip: "Open settings",
				action: () => {
					app.ui.settings.show();
				},
			})
		);
		this.viewGroup = new ComfyButtonGroup(
			new ComfyViewHistoryButton(app).element,
			new ComfyViewQueueButton(app).element,
			getInteruptButton("nlg-hide").element
		);
		this.mobileMenuButton = new ComfyButton({
			icon: "menu",
			action: (_, btn) => {
				btn.icon = this.element.classList.toggle("expanded") ? "menu-open" : "menu";
				window.dispatchEvent(new Event("resize"));
			},
			classList: "comfyui-button comfyui-menu-button",
		});

		this.element = $el("nav.comfyui-menu.lg", { style: { display: "none" } }, [
			this.logo,
			this.workflows.element,
			this.saveButton.element,
			collapseOnMobile(this.actionsGroup).element,
			$el("section.comfyui-menu-push"),
			collapseOnMobile(this.settingsGroup).element,
			collapseOnMobile(this.viewGroup).element,

			getInteruptButton("lt-lg-show").element,
			new ComfyQueueButton(app).element,
			showOnMobile(this.mobileMenuButton).element,
		]);

		let resizeHandler;
		this.menuPositionSetting = app.ui.settings.addSetting({
			id: "Comfy.UseNewMenu",
			defaultValue: "Disabled",
			name: "[Beta] Use new menu and workflow management. Note: On small screens the menu will always be at the top.",
			type: "combo",
			options: ["Disabled", "Top", "Bottom"],
			onChange: async (v) => {
				if (v && v !== "Disabled") {
					if (!resizeHandler) {
						resizeHandler = () => {
							this.calculateSizeBreak();
						};
						window.addEventListener("resize", resizeHandler);
					}
					this.updatePosition(v);
				} else {
					if (resizeHandler) {
						window.removeEventListener("resize", resizeHandler);
						resizeHandler = null;
					}
					document.body.style.removeProperty("display");
					app.ui.menuContainer.style.removeProperty("display");
					this.element.style.display = "none";
					app.ui.restoreMenuPosition();
				}
				window.dispatchEvent(new Event("resize"));
			},
		});
	}

	updatePosition(v) {
		document.body.style.display = "grid";
		this.app.ui.menuContainer.style.display = "none";
		this.element.style.removeProperty("display");
		this.position = v;
		if (v === "Bottom") {
			this.app.bodyBottom.append(this.element);
		} else {
			this.app.bodyTop.prepend(this.element);
		}
		this.calculateSizeBreak();
	}

	updateSizeBreak(idx, prevIdx, direction) {
		const newSize = this.#sizeBreaks[idx];
		if (newSize === this.#sizeBreak) return;
		this.#cachedInnerSize = null;
		clearTimeout(this.#cacheTimeout);

		this.#sizeBreak = this.#sizeBreaks[idx];
		for (let i = 0; i < this.#sizeBreaks.length; i++) {
			const sz = this.#sizeBreaks[i];
			if (sz === this.#sizeBreak) {
				this.element.classList.add(sz);
			} else {
				this.element.classList.remove(sz);
			}
			if (i < idx) {
				this.element.classList.add("lt-" + sz);
			} else {
				this.element.classList.remove("lt-" + sz);
			}
		}

		if (idx) {
			// We're on a small screen, force the menu at the top
			if (this.position !== "Top") {
				this.updatePosition("Top");
			}
		} else if (this.position != this.menuPositionSetting.value) {
			// Restore user position
			this.updatePosition(this.menuPositionSetting.value);
		}

		// Allow multiple updates, but prevent bouncing
		if (!direction) {
			direction = prevIdx - idx;
		} else if (direction != prevIdx - idx) {
			return;
		}
		this.calculateSizeBreak(direction);
	}

	calculateSizeBreak(direction = 0) {
		let idx = this.#sizeBreaks.indexOf(this.#sizeBreak);
		const currIdx = idx;
		const innerSize = this.calculateInnerSize(idx);
		if (window.innerWidth >= this.#lastSizeBreaks[this.#sizeBreaks[idx - 1]]) {
			if (idx > 0) {
				idx--;
			}
		} else if (innerSize > this.element.clientWidth) {
			this.#lastSizeBreaks[this.#sizeBreak] = Math.max(window.innerWidth, innerSize);
			// We need to shrink
			if (idx < this.#sizeBreaks.length - 1) {
				idx++;
			}
		}

		this.updateSizeBreak(idx, currIdx, direction);
	}

	calculateInnerSize(idx) {
		// Cache the inner size to prevent too much calculation when resizing the window
		clearTimeout(this.#cacheTimeout);
		if (this.#cachedInnerSize) {
			// Extend cache time
			this.#cacheTimeout = setTimeout(() => (this.#cachedInnerSize = null), 100);
		} else {
			let innerSize = 0;
			let count = 1;
			for (const c of this.element.children) {
				if (c.classList.contains("comfyui-menu-push")) continue; // ignore right push
				if (idx && c.classList.contains("comfyui-menu-mobile-collapse")) continue; // ignore collapse items
				innerSize += c.clientWidth;
				count++;
			}
			innerSize += 8 * count;
			this.#cachedInnerSize = innerSize;
			this.#cacheTimeout = setTimeout(() => (this.#cachedInnerSize = null), 100);
		}
		return this.#cachedInnerSize;
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
		if (this.app.workflowManager.activeWorkflow?.path) {
			filename = this.app.workflowManager.activeWorkflow.name;
		}
		const p = await this.app.graphToPrompt();
		const json = JSON.stringify(p[promptProperty], null, 2);
		const blob = new Blob([json], { type: "application/json" });
		const file = this.getFilename(filename);
		if (!file) return;
		downloadBlob(file, blob);
	}
}
