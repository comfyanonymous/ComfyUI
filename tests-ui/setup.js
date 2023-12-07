const { spawn } = require("child_process");
const { resolve } = require("path");
const { existsSync, mkdirSync, writeFileSync } = require("fs");
const http = require("http");

async function setup() {
	// Wait up to 30s for it to start
	let success = false;
	let child;
	for (let i = 0; i < 30; i++) {
		try {
			await new Promise((res, rej) => {
				http
					.get("http://127.0.0.1:8188/object_info", (resp) => {
						let data = "";
						resp.on("data", (chunk) => {
							data += chunk;
						});
						resp.on("end", () => {
							// Modify the response data to add some checkpoints
							const objectInfo = JSON.parse(data);
							objectInfo.CheckpointLoaderSimple.input.required.ckpt_name[0] = ["model1.safetensors", "model2.ckpt"];
							objectInfo.VAELoader.input.required.vae_name[0] = ["vae1.safetensors", "vae2.ckpt"];

							data = JSON.stringify(objectInfo, undefined, "\t");

							const outDir = resolve("./data");
							if (!existsSync(outDir)) {
								mkdirSync(outDir);
							}

							const outPath = resolve(outDir, "object_info.json");
							console.log(`Writing ${Object.keys(objectInfo).length} nodes to ${outPath}`);
							writeFileSync(outPath, data, {
								encoding: "utf8",
							});
							res();
						});
					})
					.on("error", rej);
			});
			success = true;
			break;
		} catch (error) {
			console.log(i + "/30", error);
			if (i === 0) {
				// Start the server on first iteration if it fails to connect
				console.log("Starting ComfyUI server...");

				let python = resolve("../../python_embeded/python.exe");
				let args;
				let cwd;
				if (existsSync(python)) {
					args = ["-s", "ComfyUI/main.py"];
					cwd = "../..";
				} else {
					python = "python";
					args = ["main.py"];
					cwd = "..";
				}
				args.push("--cpu");
				console.log(python, ...args);
				child = spawn(python, args, { cwd });
				child.on("error", (err) => {
					console.log(`Server error (${err})`);
					i = 30;
				});
				child.on("exit", (code) => {
					if (!success) {
						console.log(`Server exited (${code})`);
						i = 30;
					}
				});
			}
			await new Promise((r) => {
				setTimeout(r, 1000);
			});
		}
	}

	child?.kill();

	if (!success) {
		throw new Error("Waiting for server failed...");
	}
}

 setup();