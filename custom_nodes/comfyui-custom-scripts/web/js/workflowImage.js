import { app } from "../../../scripts/app.js";
import { importA1111 } from "../../../scripts/pnginfo.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";

let getDrawTextConfig = null;
let fileInput;

class WorkflowImage {
	static accept = "";

	getBounds() {
		// Calculate the min max bounds for the nodes on the graph
		const bounds = app.graph._nodes.reduce(
			(p, n) => {
				if (n.pos[0] < p[0]) p[0] = n.pos[0];
				if (n.pos[1] < p[1]) p[1] = n.pos[1];
				const bounds = n.getBounding();
				const r = n.pos[0] + bounds[2];
				const b = n.pos[1] + bounds[3];
				if (r > p[2]) p[2] = r;
				if (b > p[3]) p[3] = b;
				return p;
			},
			[99999, 99999, -99999, -99999]
		);

		bounds[0] -= 100;
		bounds[1] -= 100;
		bounds[2] += 100;
		bounds[3] += 100;
		return bounds;
	}

	saveState() {
		this.state = {
			scale: app.canvas.ds.scale,
			width: app.canvas.canvas.width,
			height: app.canvas.canvas.height,
			offset: app.canvas.ds.offset,
			transform: app.canvas.canvas.getContext("2d").getTransform(), // Save the original transformation matrix
		};
	}

	restoreState() {
		app.canvas.ds.scale = this.state.scale;
		app.canvas.canvas.width = this.state.width;
		app.canvas.canvas.height = this.state.height;
		app.canvas.ds.offset = this.state.offset;
		app.canvas.canvas.getContext("2d").setTransform(this.state.transform); // Reapply the original transformation matrix
	}

	updateView(bounds) {
		const scale = window.devicePixelRatio || 1;
		app.canvas.ds.scale = 1;
		app.canvas.canvas.width = (bounds[2] - bounds[0]) * scale;
		app.canvas.canvas.height = (bounds[3] - bounds[1]) * scale;
		app.canvas.ds.offset = [-bounds[0], -bounds[1]];
		app.canvas.canvas.getContext("2d").setTransform(scale, 0, 0, scale, 0, 0);
	}

	getDrawTextConfig(_, widget) {
		return {
			x: 10,
			y: widget.last_y + 10,
			resetTransform: false,
		};
	}

	async export(includeWorkflow) {
		// Save the current state of the canvas
		this.saveState();
		// Update to render the whole workflow
		this.updateView(this.getBounds());

		// Flag that we are saving and render the canvas
		getDrawTextConfig = this.getDrawTextConfig;
		app.canvas.draw(true, true);
		getDrawTextConfig = null;

		// Generate a blob of the image containing the workflow
		const blob = await this.getBlob(includeWorkflow ? JSON.stringify(app.graph.serialize()) : undefined);

		// Restore initial state and redraw
		this.restoreState();
		app.canvas.draw(true, true);

		// Download the generated image
		this.download(blob);
	}

	download(blob) {
		const url = URL.createObjectURL(blob);
		const a = document.createElement("a");
		Object.assign(a, {
			href: url,
			download: "workflow." + this.extension,
			style: "display: none",
		});
		document.body.append(a);
		a.click();
		setTimeout(function () {
			a.remove();
			window.URL.revokeObjectURL(url);
		}, 0);
	}

	static import() {
		if (!fileInput) {
			fileInput = document.createElement("input");
			Object.assign(fileInput, {
				type: "file",
				style: "display: none",
				onchange: () => {
					app.handleFile(fileInput.files[0]);
				},
			});
			document.body.append(fileInput);
		}
		fileInput.accept = WorkflowImage.accept;
		fileInput.click();
	}
}

class PngWorkflowImage extends WorkflowImage {
	static accept = ".png,image/png";
	extension = "png";

	n2b(n) {
		return new Uint8Array([(n >> 24) & 0xff, (n >> 16) & 0xff, (n >> 8) & 0xff, n & 0xff]);
	}

	joinArrayBuffer(...bufs) {
		const result = new Uint8Array(bufs.reduce((totalSize, buf) => totalSize + buf.byteLength, 0));
		bufs.reduce((offset, buf) => {
			result.set(buf, offset);
			return offset + buf.byteLength;
		}, 0);
		return result;
	}

	crc32(data) {
		const crcTable =
			PngWorkflowImage.crcTable ||
			(PngWorkflowImage.crcTable = (() => {
				let c;
				const crcTable = [];
				for (let n = 0; n < 256; n++) {
					c = n;
					for (let k = 0; k < 8; k++) {
						c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
					}
					crcTable[n] = c;
				}
				return crcTable;
			})());
		let crc = 0 ^ -1;
		for (let i = 0; i < data.byteLength; i++) {
			crc = (crc >>> 8) ^ crcTable[(crc ^ data[i]) & 0xff];
		}
		return (crc ^ -1) >>> 0;
	}

	async getBlob(workflow) {
		return new Promise((r) => {
			app.canvasEl.toBlob(async (blob) => {
				if (workflow) {
					// If we have a workflow embed it in the PNG
					const buffer = await blob.arrayBuffer();
					const typedArr = new Uint8Array(buffer);
					const view = new DataView(buffer);

					const data = new TextEncoder().encode(`tEXtworkflow\0${workflow}`);
					const chunk = this.joinArrayBuffer(this.n2b(data.byteLength - 4), data, this.n2b(this.crc32(data)));

					const sz = view.getUint32(8) + 20;
					const result = this.joinArrayBuffer(typedArr.subarray(0, sz), chunk, typedArr.subarray(sz));

					blob = new Blob([result], { type: "image/png" });
				}

				r(blob);
			});
		});
	}
}

class DataReader {
	/**	@type {DataView} */
	view;
	/** @type {boolean | undefined} */
	littleEndian;
	offset = 0;

	/**
	 * @param {DataView} view
	 */
	constructor(view) {
		this.view = view;
	}

	/**
	 * Reads N bytes and increments the offset
	 * @param {1 | 2 | 4 | 8} size
	 */
	read(size, signed = false, littleEndian = undefined) {
		const v = this.peek(size, signed, littleEndian);
		this.offset += size;
		return v;
	}

	/**
	 * Reads N bytes
	 * @param {1 | 2 | 4 | 8} size
	 */
	peek(size, signed = false, littleEndian = undefined) {
		this.view.getBigInt64;
		let m = "";
		if (size === 8) m += "Big";
		m += signed ? "Int" : "Uint";
		m += size * 8;
		m = "get" + m;
		if (!this.view[m]) {
			throw new Error("Method not found: " + m);
		}

		return this.view[m](this.offset, littleEndian == null ? this.littleEndian : littleEndian);
	}

	/**
	 * Seeks to the specified position or by the number of bytes specified relative to the current offset
	 * @param {number} pos
	 * @param {boolean} relative
	 */
	seek(pos, relative = true) {
		if (relative) {
			this.offset += pos;
		} else {
			this.offset = pos;
		}
	}
}

class Tiff {
	/** @type {DataReader} */
	#reader;
	#start;

	readExif(reader) {
		const TIFF_MARKER = 0x2a;
		const EXIF_IFD = 0x8769;

		this.#reader = reader;
		this.#start = this.#reader.offset;
		this.#readEndianness();

		if (!this.#reader.read(2) === TIFF_MARKER) {
			throw new Error("Invalid TIFF: Marker not found.");
		}

		const dirOffset = this.#reader.read(4);
		this.#reader.seek(this.#start + dirOffset, false);

		for (const t of this.#readTags()) {
			if (t.id === EXIF_IFD) {
				return this.#readExifTag(t);
			}
		}
		throw new Error("No EXIF: TIFF Exif IFD tag not found");
	}

	#readUserComment(tag) {
		this.#reader.seek(this.#start + tag.offset, false);
		const encoding = this.#reader.read(8);
		if (encoding !== 0x45444f43494e55n) {
			throw new Error("Unable to read non-Unicode data");
		}
		const decoder = new TextDecoder("utf-16be");
		return decoder.decode(new DataView(this.#reader.view.buffer, this.#reader.offset, tag.count - 8));
	}

	#readExifTag(exifTag) {
		const EXIF_USER_COMMENT = 0x9286;

		this.#reader.seek(this.#start + exifTag.offset, false);
		for (const t of this.#readTags()) {
			if (t.id === EXIF_USER_COMMENT) {
				return this.#readUserComment(t);
			}
		}
		throw new Error("No embedded data: UserComment Exif tag not found");
	}

	*#readTags() {
		const count = this.#reader.read(2);
		for (let i = 0; i < count; i++) {
			yield {
				id: this.#reader.read(2),
				type: this.#reader.read(2),
				count: this.#reader.read(4),
				offset: this.#reader.read(4),
			};
		}
	}

	#readEndianness() {
		const II = 0x4949;
		const MM = 0x4d4d;
		const endianness = this.#reader.read(2);
		if (endianness === II) {
			this.#reader.littleEndian = true;
		} else if (endianness === MM) {
			this.#reader.littleEndian = false;
		} else {
			throw new Error("Invalid JPEG: Endianness marker not found.");
		}
	}
}

class Jpeg {
	/** @type {DataReader} */
	#reader;

	/**
	 * @param {ArrayBuffer} buffer
	 */
	readExif(buffer) {
		const JPEG_MARKER = 0xffd8;
		const EXIF_SIG = 0x45786966;

		this.#reader = new DataReader(new DataView(buffer));
		if (!this.#reader.read(2) === JPEG_MARKER) {
			throw new Error("Invalid JPEG: SOI not found.");
		}

		const app0 = this.#readAppMarkerId();
		if (app0 !== 0) {
			throw new Error(`Invalid JPEG: APP0 not found [found: ${app0}].`);
		}

		this.#consumeAppSegment();
		const app1 = this.#readAppMarkerId();
		if (app1 !== 1) {
			throw new Error(`No EXIF: APP1 not found [found: ${app0}].`);
		}

		// Skip size
		this.#reader.seek(2);

		if (this.#reader.read(4) !== EXIF_SIG) {
			throw new Error(`No EXIF: Invalid EXIF header signature.`);
		}
		if (this.#reader.read(2) !== 0) {
			throw new Error(`No EXIF: Invalid EXIF header.`);
		}

		return new Tiff().readExif(this.#reader);
	}

	#readAppMarkerId() {
		const APP0_MARKER = 0xffe0;
		return this.#reader.read(2) - APP0_MARKER;
	}

	#consumeAppSegment() {
		this.#reader.seek(this.#reader.read(2) - 2);
	}
}

class SvgWorkflowImage extends WorkflowImage {
	static accept = ".svg,image/svg+xml";
	extension = "svg";

	static init() {
		// Override file handling to allow drag & drop of SVG
		const handleFile = app.handleFile;
		app.handleFile = async function (file) {
			if (file && (file.type === "image/svg+xml" || file.name?.endsWith(".svg"))) {
				const reader = new FileReader();
				reader.onload = () => {
					// Extract embedded workflow from desc tags
					const descEnd = reader.result.lastIndexOf("</desc>");
					if (descEnd !== -1) {
						const descStart = reader.result.lastIndexOf("<desc>", descEnd);
						if (descStart !== -1) {
							const json = reader.result.substring(descStart + 6, descEnd);
							this.loadGraphData(JSON.parse(SvgWorkflowImage.unescapeXml(json)));
						}
					}
				};
				reader.readAsText(file);
				return;
			} else if (file && (file.type === "image/jpeg" || file.name?.endsWith(".jpg") || file.name?.endsWith(".jpeg"))) {
				if (
					await new Promise((resolve) => {
						try {
							// This shouldnt go in here but it's easier than refactoring handleFile
							const reader = new FileReader();
							reader.onload = async () => {
								try {
									const value = new Jpeg().readExif(reader.result);
									importA1111(app.graph, value);
									resolve(true);
								} catch (error) {
									resolve(false);
								}
							};
							reader.onerror = () => resolve(false);
							reader.readAsArrayBuffer(file);
						} catch (error) {
							resolve(false);
						}
					})
				) {
					return;
				}
			}
			return handleFile.apply(this, arguments);
		};
	}

	static escapeXml(unsafe) {
		return unsafe.replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
	}

	static unescapeXml(safe) {
		return safe.replaceAll("&amp;", "&").replaceAll("&lt;", "<").replaceAll("&gt;", ">");
	}

	getDrawTextConfig(_, widget) {
		const domWrapper = widget.inputEl.closest(".dom-widget") ?? widget.inputEl;
		return {
			x: parseInt(domWrapper.style.left),
			y: parseInt(domWrapper.style.top),
			resetTransform: true,
		};
	}

	saveState() {
		super.saveState();
		this.state.ctx = app.canvas.ctx;
	}

	restoreState() {
		super.restoreState();
		app.canvas.ctx = this.state.ctx;
	}

	updateView(bounds) {
		super.updateView(bounds);
		this.createSvgCtx(bounds);
	}

	createSvgCtx(bounds) {
		const ctx = this.state.ctx;
		const svgCtx = (this.svgCtx = new C2S(bounds[2] - bounds[0], bounds[3] - bounds[1]));
		svgCtx.canvas.getBoundingClientRect = function () {
			return { width: svgCtx.width, height: svgCtx.height };
		};

		// Override the c2s handling of images to draw images as canvases
		const drawImage = svgCtx.drawImage;
		svgCtx.drawImage = function (...args) {
			const image = args[0];
			// If we are an image node and not a datauri then we need to replace with a canvas
			// we cant convert to data uri here as it is an async process
			if (image.nodeName === "IMG" && !image.src.startsWith("data:image/")) {
				const canvas = document.createElement("canvas");
				canvas.width = image.width;
				canvas.height = image.height;
				const imgCtx = canvas.getContext("2d");
				imgCtx.drawImage(image, 0, 0);
				args[0] = canvas;
			}

			return drawImage.apply(this, args);
		};

		// Implement missing required functions
		svgCtx.getTransform = function () {
			return ctx.getTransform();
		};
		svgCtx.resetTransform = function () {
			return ctx.resetTransform();
		};
		svgCtx.roundRect = svgCtx.rect;
		app.canvas.ctx = svgCtx;
	}

	getBlob(workflow) {
		let svg = this.svgCtx.getSerializedSvg(true).replace("<svg ", `<svg style="background: ${app.canvas.clear_background_color}" `);

		if (workflow) {
			svg = svg.replace("</svg>", `<desc>${SvgWorkflowImage.escapeXml(workflow)}</desc></svg>`);
		}

		return new Blob([svg], { type: "image/svg+xml" });
	}
}

app.registerExtension({
	name: "pysssss.WorkflowImage",
	init() {
		// https://codepen.io/peterhry/pen/nbMaYg
		function wrapText(context, text, x, y, maxWidth, lineHeight) {
			var words = text.split(" "),
				line = "",
				i,
				test,
				metrics;

			for (i = 0; i < words.length; i++) {
				test = words[i];
				metrics = context.measureText(test);
				while (metrics.width > maxWidth) {
					// Determine how much of the word will fit
					test = test.substring(0, test.length - 1);
					metrics = context.measureText(test);
				}
				if (words[i] != test) {
					words.splice(i + 1, 0, words[i].substr(test.length));
					words[i] = test;
				}

				test = line + words[i] + " ";
				metrics = context.measureText(test);

				if (metrics.width > maxWidth && i > 0) {
					context.fillText(line, x, y);
					line = words[i] + " ";
					y += lineHeight;
				} else {
					line = test;
				}
			}

			context.fillText(line, x, y);
		}

		const stringWidget = ComfyWidgets.STRING;
		// Override multiline string widgets to draw text using canvas while saving as svg
		ComfyWidgets.STRING = function () {
			const w = stringWidget.apply(this, arguments);
			if (w.widget && w.widget.type === "customtext") {
				const draw = w.widget.draw;
				w.widget.draw = function (ctx) {
					draw.apply(this, arguments);
					if (this.inputEl.hidden) return;

					if (getDrawTextConfig) {
						const config = getDrawTextConfig(ctx, this);
						const t = ctx.getTransform();
						ctx.save();
						if (config.resetTransform) {
							ctx.resetTransform();
						}

						const style = document.defaultView.getComputedStyle(this.inputEl, null);
						const x = config.x;
						const y = config.y;
						const domWrapper = this.inputEl.closest(".dom-widget") ?? widget.inputEl;
						let w = parseInt(domWrapper.style.width);
						if (w === 0) {
							w = this.node.size[0] - 20;
						}
						const h = parseInt(domWrapper.style.height);
						ctx.fillStyle = style.getPropertyValue("background-color");
						ctx.fillRect(x, y, w, h);

						ctx.fillStyle = style.getPropertyValue("color");
						ctx.font = style.getPropertyValue("font");

						const line = t.d * 12;
						const split = this.inputEl.value.split("\n");
						let start = y;
						for (const l of split) {
							start += line;
							wrapText(ctx, l, x + 4, start, w, line);
						}

						ctx.restore();
					}
				};
			}
			return w;
		};
	},
	setup() {
		const script = document.createElement("script");
		script.onload = function () {
			const formats = [SvgWorkflowImage, PngWorkflowImage];
			for (const f of formats) {
				f.init?.call();
				WorkflowImage.accept += (WorkflowImage.accept ? "," : "") + f.accept;
			}

			// Add canvas menu options
			const orig = LGraphCanvas.prototype.getCanvasMenuOptions;
			LGraphCanvas.prototype.getCanvasMenuOptions = function () {
				const options = orig.apply(this, arguments);

				options.push(null, {
					content: "Workflow Image",
					submenu: {
						options: [
							{
								content: "Import",
								callback: () => {
									WorkflowImage.import();
								},
							},
							{
								content: "Export",
								submenu: {
									options: formats.flatMap((f) => [
										{
											content: f.name.replace("WorkflowImage", "").toLocaleLowerCase(),
											callback: () => {
												new f().export(true);
											},
										},
										{
											content: f.name.replace("WorkflowImage", "").toLocaleLowerCase() + " (no embedded workflow)",
											callback: () => {
												new f().export();
											},
										},
									]),
								},
							},
						],
					},
				});
				return options;
			};
		};

		script.src = new URL(`assets/canvas2svg.js`, import.meta.url);
		document.body.append(script);
	},
});
