import { app } from "/scripts/app.js";
import { $el } from "/scripts/ui.js";

const id = "Comfy.Keybinds";
app.registerExtension({
	name: id,
	init() {
		const keybindListener = function(event) {
			const target = event.composedPath()[0];

			if (target.tagName === "INPUT" || target.tagName === "TEXTAREA") {
				return;
			}

			const modifierPressed = event.ctrlKey || event.metaKey;

			// Queue prompt using ctrl or command + enter
			if (modifierPressed && (event.key === "Enter" || event.keyCode === 13 || event.keyCode === 10)) {
				app.queuePrompt(event.shiftKey ? -1 : 0);
			}

			// Save workflow using ctrl or command + s
			if (modifierPressed && (event.key === "s" || event.keyCode === 83)) {
				event.preventDefault();

				const saveButton = document.querySelector("#comfy-save-button");
				saveButton.click()
			}

			// Load workflow using ctrl or command + o
			if (modifierPressed && (event.key === "o" || event.keyCode === 79)) {
				event.preventDefault();

				const fileInput = document.querySelector("#comfy-file-input");
				fileInput.click()
			}

			// Delete all nodes using ctrl or command + backspace or delete
			if (modifierPressed && ((event.key === "Backspace" || event.keyCode === 8) || (event.key === "Delete" || event.keyCode === 46))) {
				const clearButton = document.querySelector("#comfy-clear-button");
				clearButton.click()
			}

			// Load default workflow using ctrl or command + d
			if (modifierPressed && (event.key === "d" || event.keyCode === 68)) {
				event.preventDefault();
				const loadDefaultButton = document.querySelector("#comfy-load-default-button");
				loadDefaultButton.click()
			}

			// Finished Handling all modifier keybinds, now handle the rest
			if (event.ctrlKey || event.altKey || event.metaKey) {
				return;
			}

			const keyToButtonIdMap = {
				"q": "comfy-view-queue-button",
				81: "comfy-view-queue-button",
				"h": "comfy-view-history-button",
				72: "comfy-view-history-button",
				"r": "comfy-refresh-button",
				82: "comfy-refresh-button",
			};

			const buttonId = keyToButtonIdMap[event.key] || keyToButtonIdMap[event.keyCode];
			if (buttonId) {
				const button = document.querySelector(`#${buttonId}`);
				button.click();
			}
		}

		window.addEventListener("keydown", keybindListener, true);
	}
});
