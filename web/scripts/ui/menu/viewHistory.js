// @ts-check

import { ComfyButton } from "../components/button.js";
import { ComfyViewList, ComfyViewListButton } from "./viewList.js";

export class ComfyViewHistoryButton extends ComfyViewListButton {
	constructor(app) {
		super(app, {
			button: new ComfyButton({
				content: "View History",
				icon: "history",
				tooltip: "View history",
				classList: "comfyui-button comfyui-history-button",
			}),
			list: ComfyViewHistoryList,
			mode: "History",
		});
	}
}

export class ComfyViewHistoryList extends ComfyViewList {
	async loadItems() {
		const items = await super.loadItems();
		items["History"].reverse();
		return items;
	}
}
