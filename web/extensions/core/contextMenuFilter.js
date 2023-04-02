import { app } from "/scripts/app.js";

// Adds filtering to context menus

const id = "Comfy.ContextMenuFilter";
app.registerExtension({
	name: id,
	init() {
		const ctxMenu = LiteGraph.ContextMenu;
		LiteGraph.ContextMenu = function (values, options) {
			const ctx = ctxMenu.call(this, values, options);

			// If we are a dark menu (only used for combo boxes) then add a filter input
			if (options?.className === "dark" && values?.length > 10) {
				const filter = document.createElement("input");
				Object.assign(filter.style, {
					width: "calc(100% - 10px)",
					border: "0",
					boxSizing: "border-box",
					background: "#333",
					border: "1px solid #999",
					margin: "0 0 5px 5px",
					color: "#fff",
				});
				filter.placeholder = "Filter list";
				this.root.prepend(filter);

				filter.addEventListener("input", () => {
					// Hide all items that dont match our filter
					const term = filter.value.toLocaleLowerCase();
					const items = this.root.querySelectorAll(".litemenu-entry");
					for (const item of items) {
						item.style.display = !term || item.textContent.toLocaleLowerCase().includes(term) ? "block" : "none";
					}

					// If we have an event then we can try and position the list under the source
					if (options.event) {
						let top = options.event.clientY - 10;

						const bodyRect = document.body.getBoundingClientRect();
						const rootRect = this.root.getBoundingClientRect();
						if (bodyRect.height && top > bodyRect.height - rootRect.height - 10) {
							top = Math.max(0, bodyRect.height - rootRect.height - 10);
						}

						this.root.style.top = top + "px";
					}
				});

				requestAnimationFrame(() => {
					// Focus the filter box when opening
					filter.focus();

					// If the top is off screen then shift the element
					if (parseInt(this.root.style.top) < 0) {
						this.root.style.top = 0;
					}
				});
			}

			return ctx;
		};

		LiteGraph.ContextMenu.prototype = ctxMenu.prototype;
	},
});
