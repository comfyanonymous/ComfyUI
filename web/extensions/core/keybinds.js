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

            const keyToButtonIdMap = {
                "Enter": "queue-button",
                "f": "queue-front-button",
                "q": "view-queue-button",
                "h": "view-history-button",
                "s": "save-button",
                "l": "load-button",
                "r": "refresh-button",
                "c": "clear-button",
                "d": "load-default-button",
            };

            const buttonId = keyToButtonIdMap[event.key];
            console.log(event.key, buttonId);
            if (buttonId) {
                const button = document.querySelector(`#${buttonId}`);
                button.dispatchEvent(new Event("click"));
            }
        }

        document.addEventListener("keyup", keybindListener);
    }
});