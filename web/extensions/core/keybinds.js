import {app} from "../../scripts/app.js";

app.registerExtension({
	name: "ccniy.Keybinds",
	init() {
		const keybindListener = function (event) {
			const modifierPressed = event.ctrlKey || event.metaKey;

			// Queue prompt using ctrl or command + enter
			if (modifierPressed && event.key === "Enter") {
				app.queuePrompt(event.shiftKey ? -1 : 0).then();
				return;
			}

			const target = event.composedPath()[0];
			if (["INPUT", "TEXTAREA"].includes(target.tagName)) {
				return;
			}

			const modifierKeyIdMap = {
				s: "#ccniy-save-button",
				o: "#ccniy-file-input",
				Backspace: "#ccniy-clear-button",
				Delete: "#ccniy-clear-button",
				d: "#ccniy-load-default-button",
			};

			const modifierKeybindId = modifierKeyIdMap[event.key];
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
			if (event.key === "Escape") {
				const modals = document.querySelectorAll(".ccniy-modal");
				const modal = Array.from(modals).find(modal => window.getComputedStyle(modal).getPropertyValue("display") !== "none");
				if (modal) {
					modal.style.display = "none";
				}

				[...document.querySelectorAll("dialog")].forEach(d => {
					d.close();
				});
			}

			const keyIdMap = {
				q: "#ccniy-view-queue-button",
				h: "#ccniy-view-history-button",
				r: "#ccniy-refresh-button",
			};

			const buttonId = keyIdMap[event.key];
			if (buttonId) {
				const button = document.querySelector(buttonId);
				button.click();
			}
		}

		window.addEventListener("keydown", keybindListener, true);
	}
});
