import { api } from "../../../scripts/api.js";
import { app } from "../../../scripts/app.js";
import { $el } from "../../../scripts/ui.js";
import { lightbox } from "./common/lightbox.js";

$el("style", {
	textContent: `
	.pysssss-image-feed {
		position: absolute;
		background: var(--comfy-menu-bg);
		color: var(--fg-color);
		z-index: 99;
		font-family: sans-serif;
		font-size: 12px;
		display: flex;
		flex-direction: column;
	}
	div > .pysssss-image-feed {
		position: static;
	}
	.pysssss-image-feed--top, .pysssss-image-feed--bottom {
		width: 100vw;
		min-height: 30px;
		max-height: calc(var(--max-size, 20) * 1vh);
	}
	.pysssss-image-feed--top {
		top: 0;
	}
	.pysssss-image-feed--bottom {
		bottom: 0;
		flex-direction: column-reverse;
		padding-top: 5px;
	}
	.pysssss-image-feed--left, .pysssss-image-feed--right {
		top: 0;
		height: 100vh;
		min-width: 200px;
		max-width: calc(var(--max-size, 10) * 1vw);
	}
	.comfyui-body-left .pysssss-image-feed--left, .comfyui-body-right .pysssss-image-feed--right {
		height: 100%;
	}
	.pysssss-image-feed--left {
		left: 0;
	}
	.pysssss-image-feed--right {
		right: 0;
	}

	.pysssss-image-feed--left .pysssss-image-feed-menu, .pysssss-image-feed--right .pysssss-image-feed-menu {
		flex-direction: column;
	}

	.pysssss-image-feed-menu {
		position: relative;
		flex: 0 1 min-content;
		display: flex;
		gap: 5px;
		padding: 5px;
		justify-content: space-between;
	}
	.pysssss-image-feed-btn-group {
		align-items: stretch;
		display: flex;
		gap: .5rem;
		flex: 0 1 fit-content;
		justify-content: flex-end;
	}
	.pysssss-image-feed-btn {
		background-color:var(--comfy-input-bg);
		border-radius:5px;
		border:2px solid var(--border-color);
		color: var(--fg-color);
		cursor:pointer;
		display:inline-block;
		flex: 0 1 fit-content;
		text-decoration:none;
	}
	.pysssss-image-feed-btn.sizing-btn:checked {
		filter: invert();
	}
	.pysssss-image-feed-btn.clear-btn {
		padding: 5px 20px;
	}
	.pysssss-image-feed-btn.hide-btn {
		padding: 5px;
		aspect-ratio: 1 / 1;
	}
	.pysssss-image-feed-btn:hover {
		filter: brightness(1.2);
	}
	.pysssss-image-feed-btn:active {
		position:relative;
		top:1px;
	}
	
	.pysssss-image-feed-menu section {
		border-radius: 5px;
		background: rgba(0,0,0,0.6);
		padding: 0 5px;
		display: flex;
		gap: 5px;
		align-items: center;
		position: relative;
	}
	.pysssss-image-feed-menu section span {
		white-space: nowrap;
	}
	.pysssss-image-feed-menu section input {
		flex: 1 1 100%;
		background: rgba(0,0,0,0.6);
		border-radius: 5px;
		overflow: hidden;
		z-index: 100;
	}

	.sizing-menu {
		position: relative;
	}

	.size-controls-flyout {
		position: absolute;
		transform: scaleX(0%);
		transition: 200ms ease-out;
		transition-delay: 500ms;
		z-index: 101;
		width: 300px;
	}

	.sizing-menu:hover .size-controls-flyout {
		transform: scale(1, 1);
		transition: 200ms linear;
		transition-delay: 0;
	}
	.pysssss-image-feed--bottom .size-controls-flyout  {
		transform: scale(1,0);
		transform-origin: bottom;
		bottom: 0;
		left: 0;
	}
	.pysssss-image-feed--top .size-controls-flyout  {
		transform: scale(1,0);
		transform-origin: top;
		top: 0;
		left: 0;
	}
	.pysssss-image-feed--left .size-controls-flyout  {
		transform: scale(0, 1);
		transform-origin: left;
		top: 0;
		left: 0;
	}
	.pysssss-image-feed--right .size-controls-flyout  {
		transform: scale(0, 1);
		transform-origin: right;
		top: 0;
		right: 0;
	}
	
	.pysssss-image-feed-menu > * {
		min-height: 24px;
	}
	.pysssss-image-feed-list {
		flex: 1 1 auto;
		overflow-y: auto;
		display: grid;
		align-items: center;
		justify-content: center;
		gap: 4px;
		grid-auto-rows: min-content;
		grid-template-columns: repeat(var(--img-sz, 3), 1fr);
		transition: 100ms linear;
		scrollbar-gutter: stable both-edges;
		padding: 5px;
		background: var(--comfy-input-bg);
		border-radius: 5px;
		margin: 5px;
		margin-top: 0px;
	}
	.pysssss-image-feed-list:empty {
		display: none;
	}
	.pysssss-image-feed-list div {
		height: 100%;
		text-align: center;
	}
	.pysssss-image-feed-list::-webkit-scrollbar {
		background: var(--comfy-input-bg);
		border-radius: 5px;
	}
	.pysssss-image-feed-list::-webkit-scrollbar-thumb {
		background:var(--comfy-menu-bg);
		border: 5px solid transparent;
		border-radius: 8px;
		background-clip: content-box;
	}
	.pysssss-image-feed-list::-webkit-scrollbar-thumb:hover {
		background: var(--border-color);
		background-clip: content-box;
	}
	.pysssss-image-feed-list img {
		object-fit: var(--img-fit, contain);
		max-width: 100%;
		max-height: calc(var(--max-size) * 1vh);
		border-radius: 4px;
	}
	.pysssss-image-feed-list img:hover {
		filter: brightness(1.2);
	}`,
	parent: document.body,
});

app.registerExtension({
	name: "pysssss.ImageFeed",
	async setup() {
		let visible = true;
		const seenImages = new Map();
		const showButton = $el("button.comfy-settings-btn", {
			textContent: "🖼️",
			style: {
				right: "16px",
				cursor: "pointer",
				display: "none",
			},
		});
		let showMenuButton;
		if (!app.menu?.element.style.display && app.menu?.settingsGroup) {
			showMenuButton = new (await import("../../../scripts/ui/components/button.js")).ComfyButton({
				icon: "image-multiple",
				action: () => showButton.click(),
				tooltip: "Show Image Feed 🐍",
				content: "Show Image Feed 🐍",
			});
			showMenuButton.enabled = false;
			showMenuButton.element.style.display = "none";
			app.menu.settingsGroup.append(showMenuButton);
		}

		const getVal = (n, d) => {
			const v = localStorage.getItem("pysssss.ImageFeed." + n);
			if (v && !isNaN(+v)) {
				return v;
			}
			return d;
		};

		const saveVal = (n, v) => {
			localStorage.setItem("pysssss.ImageFeed." + n, v);
		};

		const imageFeed = $el("div.pysssss-image-feed");
		const imageList = $el("div.pysssss-image-feed-list");

		function updateMenuParent(location) {
			if (showMenuButton) {
				const el = document.querySelector(".comfyui-body-" + location);
				if (!el) return;
				el.append(imageFeed);
			} else {
				if (!imageFeed.parent) {
					document.body.append(imageFeed);
				}
			}
		}

		const feedLocation = app.ui.settings.addSetting({
			id: "pysssss.ImageFeed.Location",
			name: "🐍 Image Feed Location",
			defaultValue: "bottom",
			type: () => {
				return $el("tr", [
					$el("td", [
						$el("label", {
							textContent: "🐍 Image Feed Location:",
						}),
					]),
					$el("td", [
						$el(
							"select",
							{
								style: {
									fontSize: "14px",
								},
								oninput: (e) => {
									feedLocation.value = e.target.value;
									imageFeed.className = `pysssss-image-feed pysssss-image-feed--${feedLocation.value}`;
									updateMenuParent(feedLocation.value);
									saveVal("Location", feedLocation.value);
									window.dispatchEvent(new Event("resize"));
								},
							},
							["left", "top", "right", "bottom", "hidden"].map((m) =>
								$el("option", {
									value: m,
									textContent: m,
									selected: feedLocation.value === m,
								})
							)
						),
					]),
				]);
			},
			onChange(value) {
				if (value === "hidden") {
					imageFeed.remove();
					if (showMenuButton) {
						requestAnimationFrame(() => {
							showMenuButton.element.style.display = "none";
						});
					}
					showButton.style.display = "none";
				} else {
					showMenuButton.element.style.display = "unset";
					showButton.style.display = visible ? "none" : "unset";
					imageFeed.className = `pysssss-image-feed pysssss-image-feed--${value}`;
					updateMenuParent(value);
				}
			},
		});

		const feedDirection = app.ui.settings.addSetting({
			id: "pysssss.ImageFeed.Direction",
			name: "🐍 Image Feed Direction",
			defaultValue: "newest first",
			type: () => {
				return $el("tr", [
					$el("td", [
						$el("label", {
							textContent: "🐍 Image Feed Direction:",
						}),
					]),
					$el("td", [
						$el(
							"select",
							{
								style: {
									fontSize: "14px",
								},
								oninput: (e) => {
									feedDirection.value = e.target.value;
									imageList.replaceChildren(...[...imageList.childNodes].reverse());
								},
							},
							["newest first", "oldest first"].map((m) =>
								$el("option", {
									value: m,
									textContent: m,
									selected: feedDirection.value === m,
								})
							)
						),
					]),
				]);
			},
		});

		const deduplicateFeed = app.ui.settings.addSetting({
			id: "pysssss.ImageFeed.Deduplication",
			name: "🐍 Image Feed Deduplication",
			tooltip: `Ensures unique images in the image feed but at the cost of CPU-bound performance impact \
(from hundreds of milliseconds to seconds per image, depending on byte size). For workflows that produce duplicate images, turning this setting on may yield overall client-side performance improvements \
by reducing the number of images in the feed.

Recommended: "enabled (max performance)" uness images are erroneously deduplicated.`,
			defaultValue: 0,
			type: "combo",
			options: (value) => {
				let dedupeOptions = { disabled: 0, "enabled (slow)": 1, "enabled (performance)": 0.5, "enabled (max performance)": 0.25 };
				return Object.entries(dedupeOptions).map(([k, v]) => ({
					value: v,
					text: k,
					selected: k === value,
				}));
			},
		});

		const maxImages = app.ui.settings.addSetting({
			id: "pysssss.ImageFeed.MaxImages",
			name: "🐍 Image Feed Max Images",
			tooltip: `Limits the number of images in the feed to a maximum, removing the oldest images as new ones are added.`,
			defaultValue: 0,
			type: "number",
		});

		const saveNodeOnly = app.ui.settings.addSetting({
			id: "pysssss.ImageFeed.SaveNodeOnly",
			name: "🐍 Image Feed Display 'SaveImage' Only",
			tooltip: `Only show images from 'SaveImage' nodes. This prevents 'PreviewImage' node outputs from appearing in the feed.`,
			defaultValue: false,
			type: "boolean",
		});

		const clearButton = $el("button.pysssss-image-feed-btn.clear-btn", {
			textContent: "Clear",
			onclick: () => {
				imageList.replaceChildren();
				window.dispatchEvent(new Event("resize"));
			},
		});

		const hideButton = $el("button.pysssss-image-feed-btn.hide-btn", {
			textContent: "❌",
			onclick: () => {
				imageFeed.style.display = "none";
				showButton.style.display = feedLocation.value === "hidden" ? "none" : "unset";
				if (showMenuButton) {
					showMenuButton.enabled = true;
					showMenuButton.element.style.display = "";
				}
				saveVal("Visible", 0);
				visible = false;
				window.dispatchEvent(new Event("resize"));
			},
		});

		let columnInput;
		function updateColumnCount(v) {
			columnInput.parentElement.title = `Controls the number of columns in the feed (${v} columns).\nClick label to set custom value.`;
			imageFeed.style.setProperty("--img-sz", v);
			saveVal("ImageSize", v);
			columnInput.max = Math.max(10, v, columnInput.max);
			columnInput.value = v;
			window.dispatchEvent(new Event("resize"));
		}

		function addImageToFeed(href) {
			const method = feedDirection.value === "newest first" ? "prepend" : "append";

			if (maxImages.value > 0 && imageList.children.length >= maxImages.value) {
				imageList.children[method === "prepend" ? imageList.children.length - 1 : 0].remove();
			}

			imageList[method](
				$el("div", [
					$el(
						"a",
						{
							target: "_blank",
							href,
							onclick: (e) => {
								const imgs = [...imageList.querySelectorAll("img")].map((img) => img.getAttribute("src"));
								lightbox.show(imgs, imgs.indexOf(href));
								e.preventDefault();
							},
						},
						[$el("img", { src: href })]
					),
				])
			);
			// If lightbox is open, update it with new image
			lightbox.updateWithNewImage(href, feedDirection.value);
		}

		imageFeed.append(
			$el("div.pysssss-image-feed-menu", [
				$el("section.sizing-menu", {}, [
					$el("label.size-control-handle", { textContent: "↹ Resize Feed" }),
					$el("div.size-controls-flyout", {}, [
						$el("section.size-control.feed-size-control", {}, [
							$el("span", {
								textContent: "Feed Size...",
							}),
							$el("input", {
								type: "range",
								min: 10,
								max: 80,
								oninput: (e) => {
									e.target.parentElement.title = `Controls the maximum size of the image feed panel (${e.target.value}vh)`;
									imageFeed.style.setProperty("--max-size", e.target.value);
									saveVal("FeedSize", e.target.value);
									window.dispatchEvent(new Event("resize"));
								},
								$: (el) => {
									requestAnimationFrame(() => {
										el.value = getVal("FeedSize", 25);
										el.oninput({ target: el });
									});
								},
							}),
						]),
						$el("section.size-control.image-size-control", {}, [
							$el("a", {
								textContent: "Column count...",
								style: {
									cursor: "pointer",
									textDecoration: "underline",
								},
								onclick: () => {
									const v = +prompt("Enter custom column count", 20);
									if (!isNaN(v)) {
										updateColumnCount(v);
									}
								},
							}),
							$el("input", {
								type: "range",
								min: 1,
								max: 10,
								step: 1,
								oninput: (e) => {
									updateColumnCount(e.target.value);
								},
								$: (el) => {
									columnInput = el;
									requestAnimationFrame(() => {
										updateColumnCount(getVal("ImageSize", 4));
									});
								},
							}),
						]),
					]),
				]),
				$el("div.pysssss-image-feed-btn-group", {}, [clearButton, hideButton]),
			]),
			imageList
		);
		showButton.onclick = () => {
			imageFeed.style.display = "flex";
			showButton.style.display = "none";
			if (showMenuButton) {
				showMenuButton.enabled = false;
				showMenuButton.element.style.display = "none";
			}

			saveVal("Visible", 1);
			visible = true;
			window.dispatchEvent(new Event("resize"));
		};
		document.querySelector(".comfy-settings-btn").after(showButton);
		window.dispatchEvent(new Event("resize"));

		if (!+getVal("Visible", 1)) {
			hideButton.onclick();
		}

		api.addEventListener("executed", ({ detail }) => {
			if (visible && detail?.output?.images) {
				if (detail.node?.includes?.(":")) {
					// Ignore group nodes
					const n = app.graph.getNodeById(detail.node.split(":")[0]);
					if (n?.getInnerNodes) return;
				}

				// Apply "Display Save Image Node Only" filter if setting is enabled
				const nodeName = detail.node?.split(":")?.[0];
				if (nodeName) {
					const node = app.graph.getNodeById(nodeName);

					if (saveNodeOnly.value && node?.type !== "SaveImage") return;
				}

				for (const src of detail.output.images) {
					const href = `./view?filename=${encodeURIComponent(src.filename)}&type=${src.type}&
					subfolder=${encodeURIComponent(src.subfolder)}&t=${+new Date()}`;

					// deduplicateFeed.value is essentially the scaling factor used for image hashing
					// but when deduplication is disabled, this value is "0"
					if (deduplicateFeed.value > 0) {
						// deduplicate by ignoring images with the same filename/type/subfolder
						const fingerprint = JSON.stringify({ filename: src.filename, type: src.type, subfolder: src.subfolder });
						if (seenImages.has(fingerprint)) {
							// NOOP: image is a duplicate
						} else {
							seenImages.set(fingerprint, true);
							let img = $el("img", { src: href });
							img.onerror = () => {
								// fall back to default behavior
								addImageToFeed(href);
							};
							img.onload = () => {
								// redraw the image onto a canvas to strip metadata (resize if performance mode)
								let imgCanvas = document.createElement("canvas");
								let imgScalar = deduplicateFeed.value;
								imgCanvas.width = imgScalar * img.width;
								imgCanvas.height = imgScalar * img.height;

								let imgContext = imgCanvas.getContext("2d");
								imgContext.drawImage(img, 0, 0, imgCanvas.width, imgCanvas.height);
								const data = imgContext.getImageData(0, 0, imgCanvas.width, imgCanvas.height);

								// calculate fast hash of the image data
								let hash = 0;
								for (const b of data.data) {
									hash = (hash << 5) - hash + b;
								}

								// add image to feed if we've never seen the hash before
								if (seenImages.has(hash)) {
									// NOOP: image is a duplicate
								} else {
									// if we got to here, then the image is unique--so add to feed
									seenImages.set(hash, true);
									addImageToFeed(href);
								}
							};
						}
					} else {
						addImageToFeed(href);
					}
				}
			}
		});
	},
});
