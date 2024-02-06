// @ts-check

import { $el } from "../ui.js";

/**
 * @typedef { {
 * 	id: string,
 *  icon?: string,
 *  text?: string,
 *  tooltip?: string,
 *  callback?: (e: Event) => void,
 *  $?: (el: HTMLElement) => void,
 * } | HTMLElement } ComfyMenuButton
 */

$el("style", {
	parent: document.body,
	textContent: `
	.comfy-app-menu {
		position: absolute;
		top: 0;
		left: 0;
		right: 0;
		z-index: 99;
		background: var(--comfy-menu-bg);
		color: var(--fg-color);
		font-family: Arial;
		padding: 4px 8px;
		display: flex;
		gap: 0.8em;
		align-items: center;
		font-size: 0.8em;
	}
	.comfy-app-logo {
		font-size: 1.2em;
		margin: 0;
	}
	.comfy-app-menu .pull-right {
		margin-left: auto;
	}
	.comfy-app-menu button {
		background: transparent;
		border: none;
		color: var(--fg-color);
		height: 100%;
		cursor: pointer;
		padding: 0 10px;
		display: flex;
		align-items: center;
		gap: 6px
		font-size: 1em;
	}
	.comfy-app-menu button:hover, .comfy-split-visible .comfy-split-open {
		background: var(--border-color);
	}
	.comfy-app-menu button:first-child {
		border-top-left-radius: 4px;
		border-bottom-left-radius: 4px;
	}
	.comfy-app-menu button:last-child {
		border-top-right-radius: 4px;
		border-bottom-right-radius: 4px;
	}
	.comfy-app-menu button:not(:last-child) {
		border-right: 1px solid var(--comfy-menu-bg);
	}
	.comfy-app-menu button .mdi {
		font-size: 1.7em;
	}

	.comfy-menu-item {
		height: 2.5em;
		border-radius: 4px;
		display: flex;
		align-items: center;
		background: var(--comfy-input-bg);
	}

	.comfy-split-button {
		position: relative;
	}
	.comfy-split-button button:hover, .comfy-split-visible .comfy-split-open {
		background: var(--border-color);
	}
	.comfy-split-button .comfy-split-primary {
		border-top-left-radius: 4px;
		border-bottom-left-radius: 4px;
	}
	.comfy-split-button .comfy-split-primary .mdi {
		font-size: 1.4em;
	}
	.comfy-split-button .comfy-split-open {
		border-top-right-radius: 4px;
		border-bottom-right-radius: 4px;
		position: relative;
		padding: 0 3px;
	}
	.comfy-split-button .comfy-split-open:hover .comfy-split-menu, 
	.comfy-split-visible .comfy-split-menu {
		display: block;
	}
	.comfy-split-menu {
		display: none;
		position: absolute;
	}
	.comfy-split-button ul {
		margin: 0;
		padding: 0;
		list-style: none;
		text-align: left;
		white-space: nowrap;
		position: absolute;
		right: 0;
		top: calc(100% - 2px);
		background: var(--border-color);
		border-radius: 4px;
		border-top-right-radius: 0;
		overflow: hidden;
	}
	.comfy-split-button li {
		padding: 5px 10px;
		display: flex;
		gap: 8px;
		align-items: center;
	}
	.comfy-split-button li:hover {
		background: var(--comfy-menu-bg);
	}
	.comfy-menu-item-text-hidden {
		display: none;
	}

	.mobile-show {
		display: none;
	}

	@media only screen and (max-width: 750px) {
		.comfy-app-menu {
			flex-wrap: wrap;
			font-size: 0.9em;
		}
		.comfy-app-menu.open {
			border-bottom: 2px solid var(--border-color)
		}
		.comfy-app-menu:not(.open) .comfy-menu-mobile-collapse {
			display: none;
		}
		.comfy-app-menu > * {
			order: 1;
		}
		.comfy-app-menu .mobile-hide {
			display: none;
		}
		.comfy-app-menu .comfy-menu-mobile-collapse {
			order: 9999;
			width: 100%;
		}
		.comfy-menu-mobile-collapse .comfy-menu-item-text-hidden {
			display: unset;
		}
		.comfy-menu-mobile-collapse.comfy-menu-group {
			flex-wrap: wrap;
			height: auto;
		}
		.comfy-menu-mobile-collapse .comfy-menu-item {
			width: 100%
		}
		.comfy-menu-mobile-collapse.comfy-menu-group button {
			width: 100%;
			padding: 10px;
			gap: 10px;
		}
		.mobile-pull-right {
			margin-left: auto;
		}
		.mobile-show {
			display: block;
		}
		.comfy-menu-item {
			background: var(--comfy-input-bg) !important;
		}

		.comfy-menu-mobile-collapse .comfy-split-button {
			height: auto;
			flex-wrap: wrap;
		}
		.comfy-menu-mobile-collapse .comfy-split-menu {
			position: static;
			width: 100%;
			border-radius: 0;
		}
	}

	@media only screen and (max-width: 420px) {
		.comfy-workflow-menu {
			width: auto !important;
		}
		.comfy-workflow-menu span {
			display: none;
		}
	}
`,
});

/**
 *
 * @param { string } tag
 * @param { ComfyMenuButton } config
 * @param { boolean } showText
 * @returns
 */
function getMenuButtonContent(tag, config, showText = true) {
	const children = [];
	const props = {};
	if (config.id) {
		props.id = config.id;
	}
	if ("onclick" in config) {
		children.push(config);
	} else {
		if (config.icon) {
			children.push($el(`i.mdi.mdi-${config.icon}`));
		}
		if (config.text) {
			children.push($el("span" + (!showText ? ".comfy-menu-item-text-hidden" : ""), config.text));
		}
		if (config.tooltip) {
			props.title = config.tooltip;
		}
		if (config.callback) {
			props.onclick = (e) => config.callback(e);
		}
		props.$ = config.$;
	}
	return $el(tag, props, children);
}

/**
 *
 * @param { { event: string, eventSource: HTMLElement, className: string, classTarget?: HTMLElement, captureRoot?: HTMLElement } } param0
 */
function toggleOnEvent({ event, eventSource, className, classTarget, captureRoot }) {
	const toggle = () => {
		classTarget ??= eventSource;
		captureRoot ??= classTarget;

		if (classTarget.classList.contains(className)) {
			classTarget.classList.remove(className);
			console.log("removeEventListener", event);
			window.removeEventListener(event, closeHandler, { capture: true });
		} else {
			classTarget.classList.add(className);
			window.addEventListener(event, closeHandler, { capture: true });
			console.log("addEventListener", event);
		}
	};
	const closeHandler = (e) => {
		console.log(captureRoot, e.target);
		if (!captureRoot.contains(e.target)) {
			console.log("bye")
			toggle();
		}
	};

	eventSource.addEventListener(event, toggle);
}

export class ComfyMenuSplitButton {
	/**
	 * @param { {
	 *  primary?: Element | string,
	 *  items: Array<{ action?: "toggle" } & ComfyMenuButton >,
	 *  showPrimaryInList?: boolean,
	 *  showPrimaryText?: boolean,
	 *  showArrow?: boolean
	 * } } config
	 */
	constructor({ primary, items, showPrimaryInList, showPrimaryText = false, showArrow = true }) {
		let primaryItem;
		if (typeof primary === "string") {
			primaryItem = items.find((item) => item.id === primary);
			if (!primaryItem) {
				console.warn("Invalid primary item for split button: " + primary);
			}
		} else {
			primaryItem = primary;
		}
		if (!primaryItem) {
			primaryItem = items[0];
		}

		this.element = $el("div.comfy-split-button.comfy-menu-item");

		const menu = $el(
			"ul.comfy-split-menu",
			items
				.map((item) => {
					if (item === primaryItem && !showPrimaryInList) return;
					return getMenuButtonContent("li", item);
				})
				.filter(Boolean)
		);

		const toggle = "action" in primaryItem && primaryItem.action === "toggle";
		const primaryEl = getMenuButtonContent("button.comfy-split-primary", primaryItem, showPrimaryText);

		if (toggle) {
			toggleOnEvent({
				event: "click",
				eventSource: primaryEl,
				className: "comfy-split-visible",
				classTarget: this.element,
			});
		}

		const children = [
			primaryEl,
			showArrow
				? $el(
						"button.comfy-split-open",
						{
							$: (el) => {
								toggleOnEvent({
									event: "touchstart",
									eventSource: el,
									className: "comfy-split-visible",
									classTarget: this.element,
								});
							},
						},
						[$el("i.mdi.mdi-chevron-down"), menu]
				  )
				: menu,
		];

		this.element.append(...children);
	}
}

export class ComfyMenuGroup {
	/**
	 * @param { { className?: string, items: Array<ComfyMenuButton>, showText?: boolean } } config
	 */
	constructor({ className, items, showText = false }) {
		this.element = $el(
			"div.comfy-menu-group.comfy-menu-item",
			items.map((item) => ("onclick" in item ? item : getMenuButtonContent("button", item, showText)))
		);
	}
}

export class ComfyWorkflows {
	constructor() {
		this.element = $el(
			"div.comfy-menu-item.comfy-workflow-menu",
			{
				style: { width: "150px", padding: "0 10px", display: "flex", gap: "5px" },
			},
			[
				$el("i.mdi.mdi-graph.mdi-18px", {
					style: {
						transform: "rotate(-90deg)",
					},
				}),
				$el("span", {
					style: {
						whiteSpace: "nowrap",
						textOverflow: "ellipsis",
						overflow: "hidden",
						userSelect: "none"
					}
				}, "Some complex outpainting workflow"),
				$el("i.mdi.mdi-chevron-down.mdi-18px"),
			]
		);
	}
}

export class ComfyQueueButton {
	constructor() {
		this.element = $el("div", "qqqqqqq");
	}
}

export class ComfyExtraOptions {
	constructor() {
		this.element = $el("div");
	}
}

export class ComfyMenu {
	constructor() {
		const mobileCollapse = (t) => {
			if (t.element) t = t.element;
			t.classList.add("comfy-menu-mobile-collapse");
			return t;
		};

		this.el = $el("nav.comfy-app-menu", { parent: document.body });
		this.logo = $el("h1.comfy-app-logo.mobile-hide", "ComfyUI");
		this.workflows = new ComfyWorkflows();
		this.extraOptions = new ComfyExtraOptions();
		this.queueButton = new ComfyQueueButton();
		this.saveActions = new ComfyMenuSplitButton({
			items: [
				{
					id: "comfy-save-workflow-button",
					icon: "content-save",
					tooltip: "Save your current workflow",
					text: "Save",
					callback: () => {
						alert("swave");
					},
				},
				{
					id: "comfy-save-workflow-as-button",
					icon: "content-save-edit",
					tooltip: "Save your current workflow with a new name",
					text: "Save as",
					callback: () => {
						alert("swaveas");
					},
				},
				{
					id: "comfy-save-button",
					icon: "download",
					text: "Export",
					tooltip: "Download a JSON file of your workflow for sharing",
					callback: () => {},
				},
				{
					id: "comfy-dev-save-api-button",
					icon: "api",
					tooltip: "Download a JSON file in API format for use with the ComfyUI API",
					text: "Export (API format)",
					callback: () => {},
				},
			],
		});
		this.commonActions = new ComfyMenuGroup({
			items: [
				{
					id: "comfy-refresh-button",
					icon: "refresh",
					tooltip: "Refresh widgets to use newly added models or files",
					text: "Refresh",
					callback: () => {},
				},
				{
					id: "comfy-clipspace-button",
					icon: "clipboard-edit-outline",
					tooltip: "Open the Clipspace manager",
					text: "Clipspace",
					callback: () => {},
				},
				{
					id: "comfy-clear-button",
					icon: "cancel",
					tooltip: "Clear the current workflow",
					text: "Clear",
					callback: () => {},
				},
			],
		});

		this.viewHistory = new ComfyMenuSplitButton({
			showArrow: false,
			items: [
				{
					id: "comfy-view-queue-button",
					icon: "history",
					tooltip: "View and manage the prompt queue",
					text: "View History",
					action: "toggle",
				},
				new ComfyQueueButton().element,
			],
		});

		this.promptList = new ComfyMenuGroup({
			items: [
				this.viewHistory.element,
				{
					id: "comfy-view-history-button",
					icon: "format-list-numbered",
					tooltip: "View prompt history to reload previous generations",
					text: "View Queue",
					callback: () => {},
				},
			],
		});
		this.promptList.element.classList.add("mobile-pull-right");

		this.queueActions = new ComfyMenuSplitButton({
			primary: this.queueButton.element,
			items: [
				{
					id: "queue-front-button",
					icon: "numeric-1-box-outline",
					tooltip: "Queue this prompt at the front of the queue",
					text: "Queue Front",
					callback: () => {},
				},
				this.extraOptions.element,
			],
		});
		this.queueActions.element.classList.add("pull-right");

		this.settingsButton = new ComfyMenuGroup({
			items: [
				{
					id: "comfy-settings-btn",
					icon: "cog",
					tooltip: "Open ComfyUI settings",
					text: "Settings",
					callback: () => {},
				},
			],
		});
		this.settingsButton.element.style.background = "transparent";

		this.mobileMenuButton = new ComfyMenuGroup({
			items: [
				{
					id: "comfy-settings-btn",
					icon: "menu",
					tooltip: "Show Menu",
					text: "Menu",
					$: (el) => {
						toggleOnEvent({
							className: "open",
							event: "click",
							eventSource: el,
							classTarget: this.el,
						});
					},
				},
			],
		});
		this.mobileMenuButton.element.style.background = "transparent";
		this.mobileMenuButton.element.classList.add("mobile-show");

		this.el.append(
			this.logo,
			this.workflows.element,
			this.saveActions.element,
			mobileCollapse(this.commonActions.element),
			mobileCollapse(this.promptList.element),
			this.queueActions.element,
			mobileCollapse(this.settingsButton.element),
			this.mobileMenuButton.element
		);
	}
}

console.log(new ComfyMenu().el);
