import { api } from "/scripts/api.js";
import { app } from "/scripts/app.js";

// Adds a list of images that are generated to the bottom of the page
// This script was created by pythongosssss
// https://github.com/pythongosssss/ComfyUI-Custom-Scripts
app.registerExtension({
	name: "Comfy.ImageFeed",
	setup() {
		const imageList = document.createElement("div");
		Object.assign(imageList.style, {
			minHeight: "30px",
			maxHeight: "300px",
			width: "100vw",
			position: "absolute",
			bottom: 0,
			background: "#333",
			overflow: "auto",
		});
		document.body.append(imageList);

		function makeButton(text, style) {
			const btn = document.createElement("button");
			btn.type = "button";
			btn.textContent = text;
			Object.assign(btn.style, {
				...style,
				height: "20px",
				cursor: "pointer",
				position: "absolute",
				top: "5px",
				fontSize: "12px",
				lineHeight: "12px",
			});
			imageList.append(btn);
			return btn;
		}

		const showButton = document.createElement("button");
		const closeButton = makeButton("❌", {
			width: "20px",
			textIndent: "-4px",
			right: "5px",
		});
		closeButton.onclick = () => {
			imageList.style.display = "none";
			showButton.style.display = "unset";
		};

		const clearButton = makeButton("Clear", {
			right: "30px",
		});
		clearButton.onclick = () => {
			imageList.replaceChildren(closeButton, clearButton);
		};

		showButton.classList.add("comfy-settings-btn");
		showButton.style.right = "16px";
		showButton.style.cursor = "pointer";
		showButton.style.display = "none";
		showButton.textContent = "🖼️";
		showButton.onclick = () => {
			imageList.style.display = "block";
			showButton.style.display = "none";
		};
		document.querySelector(".comfy-settings-btn").after(showButton);

		api.addEventListener("executed", ({ detail }) => {
			if (detail?.output?.images) {
				for (const src of detail.output.images) {
					const img = document.createElement("img");
					const a = document.createElement("a");
					a.href = `/view?filename=${encodeURIComponent(src.filename)}&type=${src.type}&subfolder=${encodeURIComponent(
						src.subfolder
					)}`;
					a.target = "_blank";
					Object.assign(img.style, {
						height: "120px",
						width: "120px",
						objectFit: "cover",
					});

					img.src = a.href;
					a.append(img);
					imageList.prepend(a);
				}
			}
		});
	},
});