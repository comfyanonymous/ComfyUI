class ComfyApi extends EventTarget {
	constructor() {
		super();
	}

	#pollQueue() {
		setInterval(async () => {
			try {
				const resp = await fetch("/prompt");
				const status = await resp.json();
				this.dispatchEvent(new CustomEvent("status", { detail: status }));
			} catch (error) {
				this.dispatchEvent(new CustomEvent("status", { detail: null }));
			}
		}, 1000);
	}

	#createSocket(isReconnect) {
		if (this.socket) {
			return;
		}

		let opened = false;
		this.socket = new WebSocket(`ws${window.location.protocol === "https:" ? "s" : ""}://${location.host}/ws`);

		this.socket.addEventListener("open", () => {
			opened = true;
			if (isReconnect) {
				this.dispatchEvent(new CustomEvent("reconnected"));
			}
		});

		this.socket.addEventListener("error", () => {
			if (this.socket) this.socket.close();
			this.#pollQueue();
		});

		this.socket.addEventListener("close", () => {
			setTimeout(() => {
				this.socket = null;
				this.#createSocket(true);
			}, 300);
			if (opened) {
				this.dispatchEvent(new CustomEvent("status", { detail: null }));
				this.dispatchEvent(new CustomEvent("reconnecting"));
			}
		});

		this.socket.addEventListener("message", (event) => {
			try {
				const msg = JSON.parse(event.data);
				switch (msg.type) {
					case "status":
						if (msg.data.sid) {
							this.clientId = msg.data.sid;
						}
						this.dispatchEvent(new CustomEvent("status", { detail: msg.data.status }));
						break;
					case "progress":
						this.dispatchEvent(new CustomEvent("progress", { detail: msg.data }));
						break;
					case "executing":
						this.dispatchEvent(new CustomEvent("executing", { detail: msg.data.node }));
						break;
					case "executed":
						this.dispatchEvent(new CustomEvent("executed", { detail: msg.data }));
						break;
					default:
						throw new Error("Unknown message type");
				}
			} catch (error) {
				console.warn("Unhandled message:", event.data);
			}
		});
	}

	init() {
		this.#createSocket();
	}

	async getNodeDefs() {
		const resp = await fetch("object_info", { cache: "no-store" });
		return await resp.json();
	}

	async queuePrompt(number, { output, workflow }) {
		const body = {
			client_id: this.clientId,
			prompt: output,
			extra_data: { extra_pnginfo: { workflow } },
		};

		if (number === -1) {
			body.front = true;
		} else if (number != 0) {
			body.number = number;
		}

		const res = await fetch("/prompt", {
			method: "POST",
			headers: {
				"Content-Type": "application/json",
			},
			body: JSON.stringify(body),
		});

		if (res.status !== 200) {
			throw {
				response: await res.text(),
			};
		}
	}

	async getItems(type) {
		if (type === "queue") {
			return this.getQueue();
		}
		return this.getHistory();
	}

	async getQueue() {
		try {
			const res = await fetch("/queue");
			const data = await res.json();
			return {
				// Running action uses a different endpoint for cancelling
				Running: data.queue_running.map((prompt) => ({ prompt, remove: { name: "Cancel", cb: () => api.interrupt() } })),
				Pending: data.queue_pending.map((prompt) => ({ prompt })),
			};
		} catch (error) {
			console.error(error);
			return { Running: [], Pending: [] };
		}
	}

	async getHistory() {
		try {
			const res = await fetch("/history");
			return { History: Object.values(await res.json()) };
		} catch (error) {
			console.error(error);
			return { History: [] };
		}
	}

	async #postItem(type, body) {
		try {
			await fetch("/" + type, {
				method: "POST",
				headers: {
					"Content-Type": "application/json",
				},
				body: body ? JSON.stringify(body) : undefined,
			});
		} catch (error) {
			console.error(error);
		}
	}

	async deleteItem(type, id) {
		await this.#postItem(type, { delete: [id] });
	}

	async clearItems(type) {
		await this.#postItem(type, { clear: true });
	}

	async interrupt() {
		await this.#postItem("interrupt", null);
	}
}

export const api = new ComfyApi();
