import { api } from "./api.js";

class ComfyDialog {
	constructor() {
		this.element = document.createElement("div");
		this.element.classList.add("comfy-modal");

		const content = document.createElement("div");
		content.classList.add("comfy-modal-content");
		this.textElement = document.createElement("p");
		content.append(this.textElement);

		const closeBtn = document.createElement("button");
		closeBtn.type = "button";
		closeBtn.textContent = "CLOSE";
		content.append(closeBtn);
		closeBtn.onclick = () => this.close();

		this.element.append(content);
		document.body.append(this.element);
	}

	close() {
		this.element.style.display = "none";
	}

	show(html) {
		this.textElement.innerHTML = html;
		this.element.style.display = "flex";
	}
}

class ComfyList {
	constructor() {
		this.element = document.createElement("div");
		this.element.style.display = "none";
		this.element.textContent = "hello";
	}

	get visible() {
		return this.element.style.display !== "none";
	}

	async load() {
		// const queue = await api.getQueue();
	}

	async update() {
		if (this.visible) {
			await this.load();
		}
	}

	async show() {
		this.element.style.display = "block";
		await this.load();
	}

	hide() {
		this.element.style.display = "none";
	}

	toggle() {
		if (this.visible) {
			this.hide();
			return false;
		} else {
			this.show();
			return true;
		}
	}
}

export class ComfyUI {
	constructor(app) {
		this.app = app;
		this.dialog = new ComfyDialog();
		this.queue = new ComfyList();
		this.history = new ComfyList();

		this.menuContainer = document.createElement("div");
		this.menuContainer.classList.add("comfy-menu");

		this.queueSize = document.createElement("span");
		this.menuContainer.append(this.queueSize);

		this.addAction("Queue Prompt", () => {
			app.queuePrompt(0);
		}, "queue");

		this.btnContainer = document.createElement("div");
		this.btnContainer.classList.add("comfy-menu-btns");
		this.menuContainer.append(this.btnContainer);

		this.addAction(
			"Queue Front",
			() => {
				app.queuePrompt(-1);
			},
			"sm"
		);

		this.addAction(
			"See Queue",
			(btn) => {
				btn.textContent = this.queue.toggle() ? "Close" : "See Queue";
			},
			"sm"
		);

		this.addAction(
			"See History",
			(btn) => {
				btn.textContent = this.history.toggle() ? "Close" : "See History";
			},
			"sm"
		);

		this.menuContainer.append(this.queue.element);
		this.menuContainer.append(this.history.element);

		this.addAction("Save", () => {
			app.queuePrompt(-1);
		});
		this.addAction("Load", () => {
			app.queuePrompt(-1);
		});
		this.addAction("Clear", () => {
			app.queuePrompt(-1);
		});
		this.addAction("Load Default", () => {
			app.queuePrompt(-1);
		});

		document.body.append(this.menuContainer);
		this.setStatus({ exec_info: { queue_remaining: "X" } });
	}

	addAction(text, cb, cls) {
		const btn = document.createElement("button");
		btn.classList.add("comfy-menu-btn-" + (cls || "lg"));
		btn.textContent = text;
		btn.onclick = () => {
			cb(btn);
		};
		(cls === "sm" ? this.btnContainer : this.menuContainer).append(btn);
	}

	setStatus(status) {
		this.queueSize.textContent = "Queue size: " + (status ? status.exec_info.queue_remaining : "ERR");
	}
}
