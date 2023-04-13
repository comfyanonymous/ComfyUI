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

				const json = JSON.stringify(app.graph.serialize(), null, 2); // convert the data to a JSON string
				const blob = new Blob([json], { type: "application/json" });
				const url = URL.createObjectURL(blob);

				const defaultFileName = "workflow.json";
				const userFileName = prompt("Save workflow as:", defaultFileName);
				if (userFileName === null) return;
				const fileName = userFileName ? userFileName : defaultFileName;

				if (!fileName.endsWith(".json")) {
					fileName += ".json";
				}

				const a = $el("a", {
					href: url,
					download: fileName,
					style: { display: "none" },
					parent: document.body,
				});
				a.click();
				setTimeout(() => {
					a.remove();
					window.URL.revokeObjectURL(url);
				}, 0);
			}

			// Load workflow using ctrl or command + o
			if (modifierPressed && (event.key === "o" || event.keyCode === 79)) {
				event.preventDefault();

				const fileInput = document.querySelector("#comfy-file-input");
				fileInput.click()
			}

			// Delete all nodes using ctrl or command + backspace or delete
			if (modifierPressed && ((event.key === "Backspace" || event.keyCode === 8) || (event.key === "Delete" || event.keyCode === 46))) {
				const clearButton = document.querySelector("#clear-button");
				clearButton.click()
			}

            // Finished Handling all modifier keybinds, now handle the rest
            if (event.ctrlKey || event.altKey || event.metaKey) {
                return;
            }

            const keyToButtonIdMap = {
                "q": "view-queue-button",
                "h": "view-history-button",
                "r": "refresh-button",
                "d": "load-default-button",
            };

            const buttonId = keyToButtonIdMap[event.key];
            if (buttonId) {
                const button = document.querySelector(`#${buttonId}`);
                button.dispatchEvent(new Event("click"));
            }
        }

        window.addEventListener("keydown", keybindListener, true);
    }
});
