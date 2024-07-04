// @ts-check

import { ComfyButton } from "../components/button.js";
import { $el } from "../../ui.js";
import { api } from "../../api.js";
import { ComfyPopup } from "../components/popup.js";

export class ComfyViewListButton {
	get open() {
		return this.popup.open;
	}

	set open(open) {
		this.popup.open = open;
	}

	constructor(app, { button, list, mode }) {
		this.app = app;
		this.button = button;
		this.element = $el("div.comfyui-button-wrapper", this.button.element);
		this.popup = new ComfyPopup({
			target: this.element,
			container: this.element,
			horizontal: "right",
		});
		this.list = new (list ?? ComfyViewList)(app, mode, this.popup);
		this.popup.children = [this.list.element];
		this.popup.addEventListener("open", () => {
			this.list.update();
		});
		this.popup.addEventListener("close", () => {
			this.list.close();
		});
		this.button.withPopup(this.popup);

		api.addEventListener("status", () => {
			if (this.popup.open) {
				this.popup.update();
			}
		});
	}
}

export class ComfyViewList {
	popup;

	constructor(app, mode, popup) {
		this.app = app;
		this.mode = mode;
		this.popup = popup;
		this.type = mode.toLowerCase();

		this.items = $el(`div.comfyui-${this.type}-items.comfyui-view-list-items`);
		this.clear = new ComfyButton({
			icon: "cancel",
			content: "Clear",
			action: async () => {
				this.showSpinner(false);
				await api.clearItems(this.type);
				await this.update();
			},
		});

		this.refresh = new ComfyButton({
			icon: "refresh",
			content: "Refresh",
			action: async () => {
				await this.update(false);
			},
		});

		this.element = $el(`div.comfyui-${this.type}-popup.comfyui-view-list-popup`, [
			$el("h3", mode),
			$el("header", [this.clear.element, this.refresh.element]),
			this.items,
		]);

		api.addEventListener("status", () => {
			if (this.popup.open) {
				this.update();
			}
		});
	}

	async close() {
		this.items.replaceChildren();
	}

	async update(resize = true) {
		this.showSpinner(resize);
		const res = await this.loadItems();
		let any = false;

		const names = Object.keys(res);
		const sections = names
			.map((section) => {
				const items = res[section];
				if (items?.length) {
					any = true;
				} else {
					return;
				}

				const rows = [];
				if (names.length > 1) {
					rows.push($el("h5", section));
				}
				rows.push(...items.flatMap((item) => this.createRow(item, section)));
				return $el("section", rows);
			})
			.filter(Boolean);

		if (any) {
			this.items.replaceChildren(...sections);
		} else {
			this.items.replaceChildren($el("h5", "None"));
		}

		this.popup.update();
		this.clear.enabled = this.refresh.enabled = true;
		this.element.style.removeProperty("height");
	}

	showSpinner(resize = true) {
		// if (!this.spinner) {
		// 	this.spinner = createSpinner();
		// }
		// if (!resize) {
		// 	this.element.style.height = this.element.clientHeight + "px";
		// }
		// this.clear.enabled = this.refresh.enabled = false;
		// this.items.replaceChildren(
		// 	$el(
		// 		"div",
		// 		{
		// 			style: {
		// 				fontSize: "18px",
		// 			},
		// 		},
		// 		this.spinner
		// 	)
		// );
		// this.popup.update();
	}

	async loadItems() {
		return await api.getItems(this.type);
	}

	getRow(item, section) {
		return {
			text: item.prompt[0] + "",
			actions: [
				{
					text: "Load",
					action: async () => {
						try {
							await this.app.loadGraphData(item.prompt[3].extra_pnginfo.workflow);
							if (item.outputs) {
								this.app.nodeOutputs = item.outputs;
							}
						} catch (error) {
							alert("Error loading workflow: " + error.message);
							console.error(error);
						}
					},
				},
				{
					text: "Delete",
					action: async () => {
						try {
							await api.deleteItem(this.type, item.prompt[1]);
							this.update();
						} catch (error) {}
					},
				},
			],
		};
	}

	createRow = (item, section) => {
		const row = this.getRow(item, section);
		return [
			$el("span", row.text),
			...row.actions.map(
				(a) =>
					new ComfyButton({
						content: a.text,
						action: async (e, btn) => {
							btn.enabled = false;
							try {
								await a.action();
							} catch (error) {
								throw error;
							} finally {
								btn.enabled = true;
							}
						},
					}).element
			),
		];
	};
}
