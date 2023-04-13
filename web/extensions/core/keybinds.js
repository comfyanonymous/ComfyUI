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
                "s": "save-button",
                "l": "load-button",
                "r": "refresh-button",
                "c": "clear-button",
                "d": "load-default-button"
            };

            const buttonId = keyToButtonIdMap[event.key];
            if (buttonId) {
                const button = document.querySelector(`#${buttonId}`);
                button.dispatchEvent(new Event("click"));
            }
        }

        document.addEventListener("keyup", keybindListener);
    }
});