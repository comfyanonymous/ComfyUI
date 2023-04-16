import { app } from "/scripts/app.js";

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
				return;
			}

			const modifierKeyIdMap = {
				"s": "#comfy-save-button",
				83: "#comfy-save-button",
				"o": "#comfy-file-input",
				79: "#comfy-file-input",
				"Backspace": "#comfy-clear-button",
				8: "#comfy-clear-button",
				"Delete": "#comfy-clear-button",
				46: "#comfy-clear-button",
				"d": "#comfy-load-default-button",
				68: "#comfy-load-default-button",
			};

			const modifierKeybindId = modifierKeyIdMap[event.key] || modifierKeyIdMap[event.keyCode];
			if (modifierPressed && modifierKeybindId) {
				event.preventDefault();

				const elem = document.querySelector(modifierKeybindId);
				elem.click();
				return;
			}

			// Finished Handling all modifier keybinds, now handle the rest
			if (event.ctrlKey || event.altKey || event.metaKey) {
				return;
			}

			// Close out of modals using escape
			if (event.key === "Escape" || event.keyCode === 27) {
				const modals = document.querySelectorAll(".comfy-modal");
				const modal = Array.from(modals).find(modal => window.getComputedStyle(modal).getPropertyValue("display") !== "none");
				if (modal) {
					modal.style.display = "none";
				}
			}

			const keyIdMap = {
				"q": "#comfy-view-queue-button",
				81: "#comfy-view-queue-button",
				"h": "#comfy-view-history-button",
				72: "#comfy-view-history-button",
				"r": "#comfy-refresh-button",
				82: "#comfy-refresh-button",
			};

			const buttonId = keyIdMap[event.key] || keyIdMap[event.keyCode];
			if (buttonId) {
				const button = document.querySelector(buttonId);
				button.click();
			}
		}

		window.addEventListener("keydown", keybindListener, true);
	}
});
