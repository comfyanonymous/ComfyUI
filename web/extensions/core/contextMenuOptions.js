import {app} from "/scripts/app.js";

app.registerExtension({
    name: "Comfy.ContextMenuOptions",
    init() {
        const ctxMenu = LiteGraph.ContextMenu;

        LiteGraph.ContextMenu = function (values, options) {
            options.autoopen = true;
			const ctx = ctxMenu.call(this, values, options);

            return ctx;
        }

        LiteGraph.ContextMenu.prototype = ctxMenu.prototype;
    }
});