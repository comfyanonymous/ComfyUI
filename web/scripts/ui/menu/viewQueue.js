// @ts-check

import { ComfyButton } from "../components/button.js";
import { ComfyViewList, ComfyViewListButton } from "./viewList.js";
import { api } from "../../api.js";

export class ComfyViewQueueButton extends ComfyViewListButton {
	constructor(app) {
		super(app, {
			button: new ComfyButton({
				content: "View Queue",
				icon: "format-list-numbered",
				tooltip: "View queue",
				classList: "comfyui-button comfyui-queue-button",
			}),
			list: ComfyViewQueueList,
			mode: "Queue",
		});
	}
}

export class ComfyViewQueueList extends ComfyViewList {
	getRow = (item, section) => {
		if (section !== "Running") {
			return super.getRow(item, section);
		}
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
					text: "Cancel",
					action: async () => {
						try {
							await api.interrupt();
						} catch (error) {}
					},
				},
			],
		};
	}
}
