import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const WEBCAM_READY = Symbol();

app.registerExtension({
	name: "Comfy.WebcamCapture",
	getCustomWidgets(app) {
		return {
			WEBCAM(node, inputName) {
				let res;
				node[WEBCAM_READY] = new Promise((resolve) => (res = resolve));

				const container = document.createElement("div");
				container.style.background = "rgba(0,0,0,0.25)";
				container.style.textAlign = "center";

				const video = document.createElement("video");
				video.style.height = video.style.width = "100%";

				const loadVideo = async () => {
					try {
						const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
						container.replaceChildren(video);

						setTimeout(() => res(video), 500); // Fallback as loadedmetadata doesnt fire sometimes?
						video.addEventListener("loadedmetadata", () => res(video), false);
						video.srcObject = stream;
						video.play();
					} catch (error) {
						const label = document.createElement("div");
						label.style.color = "red";
						label.style.overflow = "auto";
						label.style.maxHeight = "100%";
						label.style.whiteSpace = "pre-wrap";

						if (window.isSecureContext) {
							label.textContent = "Unable to load webcam, please ensure access is granted:\n" + error.message;
						} else {
							label.textContent = "Unable to load webcam. A secure context is required, if you are not accessing ComfyUI on localhost (127.0.0.1) you will have to enable TLS (https)\n\n" + error.message;
						}

						container.replaceChildren(label);
					}
				};

				loadVideo();

				return { widget: node.addDOMWidget(inputName, "WEBCAM", container) };
			},
		};
	},
	nodeCreated(node) {
		if ((node.type, node.constructor.comfyClass !== "WebcamCapture")) return;

		let video;
		const camera = node.widgets.find((w) => w.name === "image");
		const w = node.widgets.find((w) => w.name === "width");
		const h = node.widgets.find((w) => w.name === "height");
		const captureOnQueue = node.widgets.find((w) => w.name === "capture_on_queue");

		const canvas = document.createElement("canvas");

		const capture = () => {
			canvas.width = w.value;
			canvas.height = h.value;
			const ctx = canvas.getContext("2d");
			ctx.drawImage(video, 0, 0, w.value, h.value);
			const data = canvas.toDataURL("image/png");

			const img = new Image();
			img.onload = () => {
				node.imgs = [img];
				app.graph.setDirtyCanvas(true);
				requestAnimationFrame(() => {
					node.setSizeForImage?.();
				});
			};
			img.src = data;
		};

		const btn = node.addWidget("button", "waiting for camera...", "capture", capture);
		btn.disabled = true;
		btn.serializeValue = () => undefined;

		camera.serializeValue = async () => {
			if (captureOnQueue.value) {
				capture();
			} else if (!node.imgs?.length) {
				const err = `No webcam image captured`;
				alert(err);
				throw new Error(err);
			}

			// Upload image to temp storage
			const blob = await new Promise((r) => canvas.toBlob(r));
			const name = `${+new Date()}.png`;
			const file = new File([blob], name);
			const body = new FormData();
			body.append("image", file);
			body.append("subfolder", "webcam");
			body.append("type", "temp");
			const resp = await api.fetchApi("/upload/image", {
				method: "POST",
				body,
			});
			if (resp.status !== 200) {
				const err = `Error uploading camera image: ${resp.status} - ${resp.statusText}`;
				alert(err);
				throw new Error(err);
			}
			return `webcam/${name} [temp]`;
		};

		node[WEBCAM_READY].then((v) => {
			video = v;
			// If width isnt specified then use video output resolution
			if (!w.value) {
				w.value = video.videoWidth || 640;
				h.value = video.videoHeight || 480; 
			}
			btn.disabled = false;
			btn.label = "capture";
		});
	},
});
