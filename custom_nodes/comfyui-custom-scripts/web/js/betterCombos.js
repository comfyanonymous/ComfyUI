import { app } from "../../../scripts/app.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";
import { $el } from "../../../scripts/ui.js";
import { api } from "../../../scripts/api.js";

const CHECKPOINT_LOADER = "CheckpointLoader|pysssss";
const LORA_LOADER = "LoraLoader|pysssss";
const IMAGE_WIDTH = 384;
const IMAGE_HEIGHT = 384;

function getType(node) {
	if (node.comfyClass === CHECKPOINT_LOADER) {
		return "checkpoints";
	}
	return "loras";
}

function getWidgetName(type) {
	return type === "checkpoints" ? "ckpt_name" : "lora_name";
}

function encodeRFC3986URIComponent(str) {
	return encodeURIComponent(str).replace(/[!'()*]/g, (c) => `%${c.charCodeAt(0).toString(16).toUpperCase()}`);
}

const calculateImagePosition = (el, bodyRect) => {
	let { top, left, right } = el.getBoundingClientRect();
	const { width: bodyWidth, height: bodyHeight } = bodyRect;

	const isSpaceRight = right + IMAGE_WIDTH <= bodyWidth;
	if (isSpaceRight) {
		left = right;
	} else {
		left -= IMAGE_WIDTH;
	}

	top = top - IMAGE_HEIGHT / 2;
	if (top + IMAGE_HEIGHT > bodyHeight) {
		top = bodyHeight - IMAGE_HEIGHT;
	}
	if (top < 0) {
		top = 0;
	}

	return { left: Math.round(left), top: Math.round(top), isLeft: !isSpaceRight };
};

function showImage(relativeToEl, imageEl) {
	const bodyRect = document.body.getBoundingClientRect();
	if (!bodyRect) return;

	const { left, top, isLeft } = calculateImagePosition(relativeToEl, bodyRect);

	imageEl.style.left = `${left}px`;
	imageEl.style.top = `${top}px`;

	if (isLeft) {
		imageEl.classList.add("left");
	} else {
		imageEl.classList.remove("left");
	}

	document.body.appendChild(imageEl);
}

let imagesByType = {};
const loadImageList = async (type) => {
	imagesByType[type] = await (await api.fetchApi(`/pysssss/images/${type}`)).json();
};

app.registerExtension({
	name: "pysssss.Combo++",
	init() {
		const displayOptions = { "List (normal)": 0, "Tree (subfolders)": 1, "Thumbnails (grid)": 2 };
		const displaySetting = app.ui.settings.addSetting({
			id: "pysssss.Combo++.Submenu",
			name: "🐍 Lora & Checkpoint loader display mode",
			defaultValue: 1,
			type: "combo",
			options: (value) => {
				value = +value;

				return Object.entries(displayOptions).map(([k, v]) => ({
					value: v,
					text: k,
					selected: k === value,
				}));
			},
		});

		$el("style", {
			textContent: `
				.pysssss-combo-image {
					position: absolute;
					left: 0;
					top: 0;
					width: ${IMAGE_WIDTH}px;
					height: ${IMAGE_HEIGHT}px;
					object-fit: contain;
					object-position: top left;
					z-index: 9999;
				}
				.pysssss-combo-image.left {
					object-position: top right;
				}
				.pysssss-combo-folder { opacity: 0.7 }
				.pysssss-combo-folder-arrow { display: inline-block; width: 15px; }
				.pysssss-combo-folder:hover { background-color: rgba(255, 255, 255, 0.1); }
				.pysssss-combo-prefix { display: none }

				/* Special handling for when the filter input is populated to revert to normal */
				.litecontextmenu:has(input:not(:placeholder-shown)) .pysssss-combo-folder-contents {
					display: block !important;
				}
				.litecontextmenu:has(input:not(:placeholder-shown)) .pysssss-combo-folder { 
					display: none;
				}
				.litecontextmenu:has(input:not(:placeholder-shown)) .pysssss-combo-prefix { 
					display: inline;
				}
				.litecontextmenu:has(input:not(:placeholder-shown)) .litemenu-entry { 
					padding-left: 2px !important;
				}

				/* Grid mode */
				.pysssss-combo-grid {
					display: grid;
					grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
					gap: 10px;
					overflow-x: hidden;
					max-width: 60vw;
				}
				.pysssss-combo-grid .comfy-context-menu-filter {
					grid-column: 1 / -1;
					position: sticky;
					top: 0;
				}
				.pysssss-combo-grid .litemenu-entry {
					word-break: break-word;
					display: flex;
					flex-direction: column;
					justify-content: space-between;
					align-items: center;
				}
				.pysssss-combo-grid .litemenu-entry:before {
					content: "";
					display: block;
					width: 100%;
					height: 250px;
					background-size: contain;
					background-position: center;
					background-repeat: no-repeat;
					/* No-image image attribution: Picture icons created by Pixel perfect - Flaticon */
					background-image: var(--background-image, url(extensions/ComfyUI-Custom-Scripts/js/assets/no-image.png));
				}

			`,
			parent: document.body,
		});
		const p1 = loadImageList("checkpoints");
		const p2 = loadImageList("loras");

		const refreshComboInNodes = app.refreshComboInNodes;
		app.refreshComboInNodes = async function () {
			const r = await Promise.all([
				refreshComboInNodes.apply(this, arguments),
				loadImageList("checkpoints").catch(() => {}),
				loadImageList("loras").catch(() => {}),
			]);
			return r[0];
		};

		const imageHost = $el("img.pysssss-combo-image");

		const positionMenu = (menu, fillWidth) => {
			// compute best position
			let left = app.canvas.last_mouse[0] - 10;
			let top = app.canvas.last_mouse[1] - 10;

			const body_rect = document.body.getBoundingClientRect();
			const root_rect = menu.getBoundingClientRect();

			if (body_rect.width && left > body_rect.width - root_rect.width - 10) left = body_rect.width - root_rect.width - 10;
			if (body_rect.height && top > body_rect.height - root_rect.height - 10) top = body_rect.height - root_rect.height - 10;

			menu.style.left = `${left}px`;
			menu.style.top = `${top}px`;
			if (fillWidth) {
				menu.style.right = "10px";
			}
		};

		const updateMenu = async (menu, type) => {
			try {
				await p1;
				await p2;
			} catch (error) {
				console.error(error);
				console.error("Error loading pysssss.betterCombos data");
			}

			// Clamp max height so it doesn't overflow the screen
			const position = menu.getBoundingClientRect();
			const maxHeight = window.innerHeight - position.top - 20;
			menu.style.maxHeight = `${maxHeight}px`;

			const images = imagesByType[type];
			const items = menu.querySelectorAll(".litemenu-entry");

			// Add image handler to items
			const addImageHandler = (item) => {
				const text = item.getAttribute("data-value").trim();
				if (images[text]) {
					const textNode = document.createTextNode("*");
					item.appendChild(textNode);

					item.addEventListener(
						"mouseover",
						() => {
							imageHost.src = `/pysssss/view/${encodeRFC3986URIComponent(images[text])}?${+new Date()}`;
							document.body.appendChild(imageHost);
							showImage(item, imageHost);
						},
						{ passive: true }
					);
					item.addEventListener(
						"mouseout",
						() => {
							imageHost.remove();
						},
						{ passive: true }
					);
					item.addEventListener(
						"click",
						() => {
							imageHost.remove();
						},
						{ passive: true }
					);
				}
			};

			const createTree = () => {
				// Create a map to store folder structures
				const folderMap = new Map();
				const rootItems = [];
				const splitBy = (navigator.platform || navigator.userAgent).includes("Win") ? /\/|\\/ : /\//;
				const itemsSymbol = Symbol("items");

				// First pass - organize items into folder structure
				for (const item of items) {
					const path = item.getAttribute("data-value").split(splitBy);

					// Remove path from visible text
					item.textContent = path[path.length - 1];
					if (path.length > 1) {
						// Add the prefix path back in so it can be filtered on
						const prefix = $el("span.pysssss-combo-prefix", {
							textContent: path.slice(0, -1).join("/") + "/",
						});
						item.prepend(prefix);
					}

					addImageHandler(item);

					if (path.length === 1) {
						rootItems.push(item);
						continue;
					}

					// Temporarily remove the item from current position
					item.remove();

					// Create folder hierarchy
					let currentLevel = folderMap;
					for (let i = 0; i < path.length - 1; i++) {
						const folder = path[i];
						if (!currentLevel.has(folder)) {
							currentLevel.set(folder, new Map());
						}
						currentLevel = currentLevel.get(folder);
					}

					// Store the actual item in the deepest folder
					if (!currentLevel.has(itemsSymbol)) {
						currentLevel.set(itemsSymbol, []);
					}
					currentLevel.get(itemsSymbol).push(item);
				}

				const createFolderElement = (name) => {
					const folder = $el("div.litemenu-entry.pysssss-combo-folder", {
						innerHTML: `<span class="pysssss-combo-folder-arrow">▶</span> ${name}`,
						style: { paddingLeft: "5px" },
					});
					return folder;
				};

				const insertFolderStructure = (parentElement, map, level = 0) => {
					for (const [folderName, content] of map.entries()) {
						if (folderName === itemsSymbol) continue;

						const folderElement = createFolderElement(folderName);
						folderElement.style.paddingLeft = `${level * 10 + 5}px`;
						parentElement.appendChild(folderElement);

						const childContainer = $el("div.pysssss-combo-folder-contents", {
							style: { display: "none" },
						});

						// Add items in this folder
						const items = content.get(itemsSymbol) || [];
						for (const item of items) {
							item.style.paddingLeft = `${(level + 1) * 10 + 14}px`;
							childContainer.appendChild(item);
						}

						// Recursively add subfolders
						insertFolderStructure(childContainer, content, level + 1);
						parentElement.appendChild(childContainer);

						// Add click handler for folder
						folderElement.addEventListener("click", (e) => {
							e.stopPropagation();
							const arrow = folderElement.querySelector(".pysssss-combo-folder-arrow");
							const contents = folderElement.nextElementSibling;
							if (contents.style.display === "none") {
								contents.style.display = "block";
								arrow.textContent = "▼";
							} else {
								contents.style.display = "none";
								arrow.textContent = "▶";
							}
						});
					}
				};

				insertFolderStructure(items[0]?.parentElement || menu, folderMap);
				positionMenu(menu);
			};

			const addImageData = (item) => {
				const text = item.getAttribute("data-value").trim();
				if (images[text]) {
					item.style.setProperty("--background-image", `url(/pysssss/view/${encodeRFC3986URIComponent(images[text])})`);
				}
			};

			if (displaySetting.value === 1 || displaySetting.value === true) {
				createTree();
			} else if (displaySetting.value === 2) {
				menu.classList.add("pysssss-combo-grid");

				for (const item of items) {
					addImageData(item);
				}
				positionMenu(menu, true);
			} else {
				for (const item of items) {
					addImageHandler(item);
				}
			}
		};

		const mutationObserver = new MutationObserver((mutations) => {
			const node = app.canvas.current_node;

			if (!node || (node.comfyClass !== LORA_LOADER && node.comfyClass !== CHECKPOINT_LOADER)) {
				return;
			}

			for (const mutation of mutations) {
				for (const removed of mutation.removedNodes) {
					if (removed.classList?.contains("litecontextmenu")) {
						imageHost.remove();
					}
				}

				for (const added of mutation.addedNodes) {
					if (added.classList?.contains("litecontextmenu")) {
						const overWidget = app.canvas.getWidgetAtCursor();
						const type = getType(node);
						if (overWidget?.name === getWidgetName(type)) {
							requestAnimationFrame(() => {
								// Bad hack to prevent showing on right click menu by checking for the filter input
								if (!added.querySelector(".comfy-context-menu-filter")) return;
								updateMenu(added, type);
							});
						}
						return;
					}
				}
			}
		});
		mutationObserver.observe(document.body, { childList: true, subtree: false });
	},
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		const isCkpt = nodeData.name === CHECKPOINT_LOADER;
		const isLora = nodeData.name === LORA_LOADER;
		if (isCkpt || isLora) {
			const onAdded = nodeType.prototype.onAdded;
			nodeType.prototype.onAdded = function () {
				onAdded?.apply(this, arguments);
				const { widget: exampleList } = ComfyWidgets["COMBO"](this, "example", [[""], {}], app);
				this.widgets.find((w) => w.name === "prompt").computeSize = () => [0, -4];
				let exampleWidget;

				const get = async (route, suffix) => {
					const url = encodeRFC3986URIComponent(`${getType(nodeType)}${suffix || ""}`);
					return await api.fetchApi(`/pysssss/${route}/${url}`);
				};

				const getExample = async () => {
					if (exampleList.value === "[none]") {
						if (exampleWidget) {
							exampleWidget.inputEl.remove();
							exampleWidget = null;
							this.widgets.length -= 1;
						}
						return;
					}

					const v = this.widgets[0].value;
					const pos = v.lastIndexOf(".");
					const name = v.substr(0, pos);
					let exampleName = exampleList.value;
					let viewPath = `/${name}`;
					if (exampleName === "notes") {
						viewPath += ".txt";
					} else {
						viewPath += `/${exampleName}`;
					}
					const example = await (await get("view", viewPath)).text();
					if (!exampleWidget) {
						exampleWidget = ComfyWidgets["STRING"](this, "prompt", ["STRING", { multiline: true }], app).widget;
						exampleWidget.inputEl.readOnly = true;
						exampleWidget.inputEl.style.opacity = 0.6;
					}
					exampleWidget.value = example;
				};

				const exampleCb = exampleList.callback;
				exampleList.callback = function () {
					getExample();
					return exampleCb?.apply(this, arguments) ?? exampleList.value;
				};

				const listExamples = async () => {
					exampleList.disabled = true;
					exampleList.options.values = ["[none]"];
					exampleList.value = "[none]";
					let examples = [];
					if (this.widgets[0].value) {
						try {
							examples = await (await get("examples", `/${this.widgets[0].value}`)).json();
						} catch (error) {}
					}
					exampleList.options.values = ["[none]", ...examples];
					exampleList.value = exampleList.options.values[+!!examples.length];
					exampleList.callback();
					exampleList.disabled = !examples.length;
					app.graph.setDirtyCanvas(true, true);
				};

				// Expose function to update examples
				nodeType.prototype["pysssss.updateExamples"] = listExamples;

				const modelWidget = this.widgets[0];
				const modelCb = modelWidget.callback;
				let prev = undefined;
				modelWidget.callback = function () {
					let ret = modelCb?.apply(this, arguments) ?? modelWidget.value;
					if (typeof ret === "object" && "content" in ret) {
						ret = ret.content;
						modelWidget.value = ret;
					}
					let v = ret;
					if (prev !== v) {
						listExamples();
						prev = v;
					}
					return ret;
				};
				setTimeout(() => {
					modelWidget.callback();
				}, 30);
			};
		}

		const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
		nodeType.prototype.getExtraMenuOptions = function (_, options) {
			if (this.imgs) {
				// If this node has images then we add an open in new tab item
				let img;
				if (this.imageIndex != null) {
					// An image is selected so select that
					img = this.imgs[this.imageIndex];
				} else if (this.overIndex != null) {
					// No image is selected but one is hovered
					img = this.imgs[this.overIndex];
				}
				if (img) {
					const nodes = app.graph._nodes.filter((n) => n.comfyClass === LORA_LOADER || n.comfyClass === CHECKPOINT_LOADER);
					if (nodes.length) {
						options.unshift({
							content: "Save as Preview",
							submenu: {
								options: nodes.map((n) => ({
									content: n.widgets[0].value,
									callback: async () => {
										const url = new URL(img.src);
										await api.fetchApi("/pysssss/save/" + encodeRFC3986URIComponent(`${getType(n)}/${n.widgets[0].value}`), {
											method: "POST",
											body: JSON.stringify({
												filename: url.searchParams.get("filename"),
												subfolder: url.searchParams.get("subfolder"),
												type: url.searchParams.get("type"),
											}),
											headers: {
												"content-type": "application/json",
											},
										});
										loadImageList(getType(n));
									},
								})),
							},
						});
					}
				}
			}
			return getExtraMenuOptions?.apply(this, arguments);
		};
	},
});
