import { app } from "../../scripts/app.js";
import { LiteGraph, LGraphCanvas } from "../../lib/litegraph.core.js"
import { hook } from "../../scripts/utils.js";

// Adds filtering to combo context menus

const ext = {
	name: "Comfy.ContextMenuFilter",
	init() {
		hook(LiteGraph, "onContextMenuCreated", (orig, args) => {
			orig?.(...args);
			const contextMenu = args[0];

			// If we are a dark menu (only used for combo boxes) then add a filter input
			if (contextMenu.options?.className === "dark" && contextMenu.values?.length > 10) {
				const filter = document.createElement("input");
				filter.classList.add("comfy-context-menu-filter");
				filter.placeholder = "Filter list";
				contextMenu.root.prepend(filter);

				const items = Array.from(contextMenu.root.querySelectorAll(".litemenu-entry"));
				let displayedItems = [...items];
				let itemCount = displayedItems.length;

				// We must request an animation frame for the current node of the active canvas to update.
				requestAnimationFrame(() => {
					const currentNode = LGraphCanvas.active_canvas.current_node;
					const clickedComboValue = currentNode.widgets
						.filter(w => w.type === "combo" && w.options.values.length === contextMenu.values.length)
						.find(w => w.options.values.every((v, i) => v === contextMenu.values[i]))
						?.value;

					let selectedIndex = clickedComboValue ? contextMenu.values.findIndex(v => v === clickedComboValue) : 0;
					if (selectedIndex < 0) {
						selectedIndex = 0;
					}
					let selectedItem = displayedItems[selectedIndex];
					updateSelected();

					// Apply highlighting to the selected item
					function updateSelected() {
						selectedItem?.style.setProperty("background-color", "");
						selectedItem?.style.setProperty("color", "");
						selectedItem = displayedItems[selectedIndex];
						selectedItem?.style.setProperty("background-color", "#ccc", "important");
						selectedItem?.style.setProperty("color", "#000", "important");
					}

					const positionList = () => {
						const rect = contextMenu.root.getBoundingClientRect();

						// If the top is off-screen then shift the element with scaling applied
						if (rect.top < 0) {
							const scale = 1 - contextMenu.root.getBoundingClientRect().height / contextMenu.root.clientHeight;
							const shift = (contextMenu.root.clientHeight * scale) / 2;
							contextMenu.root.style.top = -shift + "px";
						}
					}

					// Arrow up/down to select items
					filter.addEventListener("keydown", (event) => {
						switch (event.key) {
							case "ArrowUp":
								event.preventDefault();
								if (selectedIndex === 0) {
									selectedIndex = itemCount - 1;
								} else {
									selectedIndex--;
								}
								updateSelected();
								break;
							case "ArrowRight":
								event.preventDefault();
								selectedIndex = itemCount - 1;
								updateSelected();
								break;
							case "ArrowDown":
								event.preventDefault();
								if (selectedIndex === itemCount - 1) {
									selectedIndex = 0;
								} else {
									selectedIndex++;
								}
								updateSelected();
								break;
							case "ArrowLeft":
								event.preventDefault();
								selectedIndex = 0;
								updateSelected();
								break;
							case "Enter":
								selectedItem?.click();
								break;
							case "Escape":
								contextMenu.close();
								break;
						}
					});

					filter.addEventListener("input", () => {
						// Hide all items that don't match our filter
						const term = filter.value.toLocaleLowerCase();
						// When filtering, recompute which items are visible for arrow up/down and maintain selection.
						displayedItems = items.filter(item => {
							const isVisible = !term || item.textContent.toLocaleLowerCase().includes(term);
							item.style.display = isVisible ? "block" : "none";
							return isVisible;
						});

						selectedIndex = 0;
						if (displayedItems.includes(selectedItem)) {
							selectedIndex = displayedItems.findIndex(d => d === selectedItem);
						}
						itemCount = displayedItems.length;

						updateSelected();

						// If we have an event then we can try and position the list under the source
						if (contextMenu.options.event) {
							let top = contextMenu.options.event.clientY - 10;

							const bodyRect = document.body.getBoundingClientRect();
							const rootRect = contextMenu.root.getBoundingClientRect();
							if (bodyRect.height && top > bodyRect.height - rootRect.height - 10) {
								top = Math.max(0, bodyRect.height - rootRect.height - 10);
							}

							contextMenu.root.style.top = top + "px";
							positionList();
						}
					});

					requestAnimationFrame(() => {
						// Focus the filter box when opening
						filter.focus();

						positionList();
					});
				})
			}
		});
	},
}

app.registerExtension(ext);
