import { app } from "/scripts/app.js";

// Adds filtering to combo context menus

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

				let selectedIndex = 0;
				let items = this.root.querySelectorAll(".litemenu-entry");
				let itemCount = items.length;
				let selectedItem;

				// Apply highlighting to the selected item
				function updateSelected() {
					if (selectedItem) {
						selectedItem.style.setProperty("background-color", "");
						selectedItem.style.setProperty("color", "");
					}
					selectedItem = items[selectedIndex];
					if (selectedItem) {
						selectedItem.style.setProperty("background-color", "#ccc", "important");
						selectedItem.style.setProperty("color", "#000", "important");
					}
				}

				updateSelected();

				// Arrow up/down to select items
				filter.addEventListener("keydown", (e) => {
					if (e.key === "ArrowUp") {
						if (selectedIndex === 0) {
							selectedIndex = itemCount - 1;
						} else {
							selectedIndex--;
						}
						updateSelected();
						e.preventDefault();
					} else if (e.key === "ArrowDown") {
						if (selectedIndex === itemCount - 1) {
							selectedIndex = 0;
						} else {
							selectedIndex++;
						}
						updateSelected();
						e.preventDefault();
					} else if ((selectedItem && e.key === "Enter") || e.keyCode === 13 || e.keyCode === 10) {
						selectedItem.click();
					}
				});

				filter.addEventListener("input", () => {
					// Hide all items that dont match our filter
					const term = filter.value.toLocaleLowerCase();
					items = this.root.querySelectorAll(".litemenu-entry");
					// When filtering recompute which items are visible for arrow up/down
					// Try and maintain selection
					let visibleItems = [];
					for (const item of items) {
						const visible = !term || item.textContent.toLocaleLowerCase().includes(term);
						if (visible) {
							item.style.display = "block";
							if (item === selectedItem) {
								selectedIndex = visibleItems.length;
							}
							visibleItems.push(item);
						} else {
							item.style.display = "none";
							if (item === selectedItem) {
								selectedIndex = 0;
							}
						}
					}
					items = visibleItems;
					updateSelected();

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

					const rect = this.root.getBoundingClientRect();

					// If the top is off screen then shift the element with scaling applied
					if (rect.top < 0) {
						const scale = 1 - this.root.getBoundingClientRect().height / this.root.clientHeight;
						const shift = (this.root.clientHeight * scale) / 2;
						this.root.style.top = -shift + "px";
					}
				});
			}

			return ctx;
		};

		LiteGraph.ContextMenu.prototype = ctxMenu.prototype;
	},
});
