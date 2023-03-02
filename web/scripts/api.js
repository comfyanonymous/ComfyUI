class ComfyApi {
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
}

export const api = new ComfyApi();
