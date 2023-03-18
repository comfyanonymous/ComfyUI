import { app } from "../scripts/app.js";

const ext = {
	name: "Comfy.MenuMove",

	async init(app) {
		let menuContainerRect = { f: false, x: 0, y: 0 };

		document.addEventListener("mousemove", (event) => {
			if (menuContainerRect.f) {
				localStorage[`${this.name}.X`] = menu.style.left = `${((event.x - menuContainerRect.x) / window.innerWidth) * 100}%`;
				localStorage[`${this.name}.Y`] = menu.style.top = `${((event.y - menuContainerRect.y) / window.innerHeight) * 100}%`;
			}
		});

		document.addEventListener("mouseup", _ => {
			menuContainerRect.f = false;
		}, true);

		const menu = document.querySelector("div.comfy-menu");

		menu.onmousedown = event => {
			if (menu === event.toElement) {
				menuContainerRect.f = true;
				menuContainerRect.x = event.offsetX;
				menuContainerRect.y = event.offsetY;
			}
		}

		menu.style.left = localStorage[`${this.name}.X`] ?? "";
		menu.style.top = localStorage[`${this.name}.Y`] ?? "50%";

		console.log("start Menu Moving!");
	},
};

app.registerExtension(ext);
