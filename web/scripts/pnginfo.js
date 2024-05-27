import { api } from "./api.js";

export function getPngMetadata(file) {
	return new Promise((r) => {
		const reader = new FileReader();
		reader.onload = (event) => {
			// Get the PNG data as a Uint8Array
			const pngData = new Uint8Array(event.target.result);
			const dataView = new DataView(pngData.buffer);

			// Check that the PNG signature is present
			if (dataView.getUint32(0) !== 0x89504e47) {
				console.error("Not a valid PNG file");
				r();
				return;
			}

			// Start searching for chunks after the PNG signature
			let offset = 8;
			let txt_chunks = {};
			// Loop through the chunks in the PNG file
			while (offset < pngData.length) {
				// Get the length of the chunk
				const length = dataView.getUint32(offset);
				// Get the chunk type
				const type = String.fromCharCode(...pngData.slice(offset + 4, offset + 8));
				if (type === "tEXt" || type == "comf" || type === "iTXt") {
					// Get the keyword
					let keyword_end = offset + 8;
					while (pngData[keyword_end] !== 0) {
						keyword_end++;
					}
					const keyword = String.fromCharCode(...pngData.slice(offset + 8, keyword_end));
					// Get the text
					const contentArraySegment = pngData.slice(keyword_end + 1, offset + 8 + length);
					const contentJson = new TextDecoder("utf-8").decode(contentArraySegment);
					txt_chunks[keyword] = contentJson;
				}

				offset += 12 + length;
			}

			r(txt_chunks);
		};

		reader.readAsArrayBuffer(file);
	});
}

function parseExifData(exifData) {
	// Check for the correct TIFF header (0x4949 for little-endian or 0x4D4D for big-endian)
	const isLittleEndian = new Uint16Array(exifData.slice(0, 2))[0] === 0x4949;

	// Function to read 16-bit and 32-bit integers from binary data
	function readInt(offset, isLittleEndian, length) {
		let arr = exifData.slice(offset, offset + length)
		if (length === 2) {
			return new DataView(arr.buffer, arr.byteOffset, arr.byteLength).getUint16(0, isLittleEndian);
		} else if (length === 4) {
			return new DataView(arr.buffer, arr.byteOffset, arr.byteLength).getUint32(0, isLittleEndian);
		}
	}

	// Read the offset to the first IFD (Image File Directory)
	const ifdOffset = readInt(4, isLittleEndian, 4);

	function parseIFD(offset) {
		const numEntries = readInt(offset, isLittleEndian, 2);
		const result = {};

		for (let i = 0; i < numEntries; i++) {
			const entryOffset = offset + 2 + i * 12;
			const tag = readInt(entryOffset, isLittleEndian, 2);
			const type = readInt(entryOffset + 2, isLittleEndian, 2);
			const numValues = readInt(entryOffset + 4, isLittleEndian, 4);
			const valueOffset = readInt(entryOffset + 8, isLittleEndian, 4);

			// Read the value(s) based on the data type
			let value;
			if (type === 2) {
				// ASCII string
				value = String.fromCharCode(...exifData.slice(valueOffset, valueOffset + numValues - 1));
			}

			result[tag] = value;
		}

		return result;
	}

	// Parse the first IFD
	const ifdData = parseIFD(ifdOffset);
	return ifdData;
}

function splitValues(input) {
    var output = {};
    for (var key in input) {
		var value = input[key];
		var splitValues = value.split(':', 2);
		output[splitValues[0]] = splitValues[1];
    }
    return output;
}

export function getWebpMetadata(file) {
	return new Promise((r) => {
		const reader = new FileReader();
		reader.onload = (event) => {
			const webp = new Uint8Array(event.target.result);
			const dataView = new DataView(webp.buffer);

			// Check that the WEBP signature is present
			if (dataView.getUint32(0) !== 0x52494646 || dataView.getUint32(8) !== 0x57454250) {
				console.error("Not a valid WEBP file");
				r();
				return;
			}

			// Start searching for chunks after the WEBP signature
			let offset = 12;
			let txt_chunks = {};
			// Loop through the chunks in the WEBP file
			while (offset < webp.length) {
				const chunk_length = dataView.getUint32(offset + 4, true);
				const chunk_type = String.fromCharCode(...webp.slice(offset, offset + 4));
				if (chunk_type === "EXIF") {
					if (String.fromCharCode(...webp.slice(offset + 8, offset + 8 + 6)) == "Exif\0\0") {
						offset += 6;
					}
					let data = parseExifData(webp.slice(offset + 8, offset + 8 + chunk_length));
					for (var key in data) {
						if (data[key]) {
							var value = data[key];
							let index = value.indexOf(':');
							txt_chunks[value.slice(0, index)] = value.slice(index + 1);
						}
					}
				}

				offset += 8 + chunk_length;
			}

			r(txt_chunks);
		};

		reader.readAsArrayBuffer(file);
	});
}


export function getAvifMetadata(file) {
	return new Promise((r) => {
		const reader = new FileReader();
		reader.onload = (event) => {
			const avif = new Uint8Array(event.target.result);
			const dataView = new DataView(avif.buffer);

			// https://aomediacodec.github.io/av1-avif/#brands-overview
			// https://www.garykessler.net/library/file_sigs.html
			// Check that the AVIF signature is present: [4 byte offset] ftypheic or ftypavif
			// console.log('avif dataView.getUint32(4)',dataView.getUint32(4));  // 1718909296 = 0x66747970 = ftyp
			// console.log('avif dataView.getUint32(8)',dataView.getUint32(8));  // 1635150182 = 0x61766966 = avif
			// console.log('avif dataView.getUint32(8)',dataView.getUint32(8));  // 1751476579 = 0x68656963 = heic
			//                1718909296 = 0x66747970 = ftyp          1635150182 = 0x61766966 = avif         1751476579 = 0x68656963 = heic
			if (!(dataView.getUint32(4) == 0x66747970 && (dataView.getUint32(8) == 0x61766966 || dataView.getUint32(8) == 0x68656963))) {
				console.error("Not a valid AVIF file");
				r();
				return;
			}

			// Start searching for Exif chunks after the AVIF signature
			/*
			we have in this order:
			- signature that starts at 0x4 and is 8-byte long: ftypavif most likely
			- 4-byte 00 space
			- multiple 4-byte words defining how the avif was build, ending with 00 00
			- examples above have avifmiflmiaf or avifmiflmiafMA1B
			- Then the core image definition length on 2-bytes: examples above are 0x133 nd 0x12B long, ending just before the 8-byte mdat section
			- mdat section is 8-byte long
			- then we finally have the 6-byte Exif\0\0 followed by the IFD definitions, and then the Exif chunks

			we start at offset 4 + 8 + 4 = 0x10
			then we slice n 4-byte words until the last one starts with 00 00: `0x10 + 0x4n` = offset for core metadata size (meta)
			then we read core metadata size offset as a 4-byte word (meta)
			then we add `0x10 + 0x4n + 4 + meta + 8` = offset for Exif\0\0 section!
			*/
			let metaOffset = 0x10
			while (metaOffset < avif.length / 2) {
				let word = String.fromCharCode(...avif.slice(metaOffset, metaOffset+4));
				if (word.slice(0,2) != "\0\0") {
					metaOffset += 4
				} else break;
			}
			let metaSize = dataView.getUint32(metaOffset);
			let offset = metaOffset + 4 + metaSize + 8;
			
			// Now we calculate offsetChunk_length = offset for the Exif chunk size
			/*
			We start from metaOffset + 4 = offset for the whole meta section. As seen above, each section length in meta is defined by the last 1 or 2-byte, more or less.
			we set slice = 0xC
			as long as the current 4-byte word is not == iloc:
			we slice and get the last 2 as length for the next section;
			we offset + slice and
			if next 4-byte word is not iloc, loop: slice = the last 2
			if next 4-byte word is iloc, we offset + 0x2C and this is the offsetChunk_length!
			read chunk_length and move on
			*/
			let offsetChunk_length = metaOffset + 4;
			let slice = 0xC;
			while (offsetChunk_length < avif.length / 2) {
				let word = String.fromCharCode(...avif.slice(offsetChunk_length, offsetChunk_length+4));
				if (word != "iloc") {
					offsetChunk_length += slice;												// next offset to read from
					slice = dataView.getUint16(offsetChunk_length - 2); // get new slice length
				} else break;
			}
			offsetChunk_length += 0x2C;
			let chunk_length = dataView.getUint32(offsetChunk_length);
			
			/*
			dataView.getUint32() reads 4 bytes starting at the specified byte offset of DataView
			dataView.getUint16() reads 2 bytes starting at the specified byte offset of DataView
			dataView.getUint8()	reads 1 bytes starting at the specified byte offset of DataView
			https://stackoverflow.com/questions/7555842/questions-about-exif-in-hexadecimal-form
			The Exif APP1 Section is actually Exif\0\0 then followed by TIFF Header of 10-bytes length
			therefore, we have at 0x15F: 6-bytes Exif\0\0 then 10-bytes Tiff Header
			
			Some Exif color data starts at 0xCA offset, but since we (we = save image custom nodes) add more Exif metadata, 
			we have our own block starting at 0x15F. I generated many avif and they all start at 0x15F. 
			We can assume valid avif should come from those custom nodes anyways. Don't you think?
			45 78 69 66 00 00						 starts at offset = metaOffset + 4 + metaSize + 8: Exif + \0\0 = 6-bytes
			E	x	i	f	00 00
			Tiff header after is 0xA long (10 bytes) and contains information about byte-order of IFD sections and a pointer to the 0th IFD
			Tiff header:
			4D 4D 00 2A 00 00 00 08 00 02
			-----|		 |
			49 49|		 |									(II for Intel) if the byte-order is little-endian
			4D 4D|		 |									(MM for Motorola) for big-endian
					 |00 2A|									magic bytes 0x002A (=42, haha...)
								 |				 08|			following 4-byte will tell the offset to the 0th IFD from the start of the TIFF header.
														 |	 02 last 2-byte seem like the number of extra Exif IFD, we have indeed only 2 in this example
			
			IFD Fields are 12-byte subsections of IFD sections
			For example, if we are looking for only 2 IFD fields with 0x9286/UserComment/prompt and 0x010e/ImageDescription/workflow:
			4D 4D 00 2A 00 00 00 08 00 02				 0th IFD, 10-byte long, last 2 bytes give how many IFD fields there are
			
			01 0E 00 02 00 00 2F 15 00 00 00 26	 010e/ImageDescription IFD1
			92 86 00 02 00 00 0F 7C 00 00 2F 3C	 9286/UserComment			IFD2
			-----|																tag ID
					 |-----|													type of the field data. 1 for byte, 2 for ascii, 3 for short (uint16), 4 for long (uint32), etc
								 |-----------|							length in 4 bytes, only for ascii, which we only care about; max is 4GB of data
														 |-----------|	4-byte field value, no idea what that's for
																						
			00 00 00 00													 then 4-byte offset from the end of the TIFF header to the start of the first IFD value
			Workflow: {"...0.4} 00								W starts at offset 0x18B	(don't care) and is length 0x2F15 including 1 last \0
			00																		00 separator
			Prompt: {"...		 } 00								P starts at offset 0x30A1 (don't care) and is length 0x0F7C including 1 last \0
			
			*/
			let txt_chunks = {}
			// Loop through the chunks in the AVIF file
			// avif clearly are different beasts than webp, there is only one chunck of Exif data at the beginning.
			// If we ever come across one that is different, surely it's not been produced by a custom node and surely, the metadata is invalid
			// while (offset < (offset + chunk_length)) { // no need to scan the whole avif file
				const chunk_type = String.fromCharCode(...avif.slice(offset, offset + 6));
				if (chunk_type === "Exif\0\0") {
					offset += 6;
					
					// parseExifData must start at the Tiff Header: 0x4949 or 0x4D4D for Big-Endian
					let data = parseExifData(avif.slice(offset, offset + chunk_length));
					for (var key in data) {
						if (data[key]) {
							var value = data[key];
							let index = value.indexOf(':');
							txt_chunks[value.slice(0, index)] = value.slice(index + 1);
						}
					}
				}
			
			// offset += chunk_length;
			// }

			r(txt_chunks);
		};

		reader.readAsArrayBuffer(file);
	});
}

export function getLatentMetadata(file) {
	return new Promise((r) => {
		const reader = new FileReader();
		reader.onload = (event) => {
			const safetensorsData = new Uint8Array(event.target.result);
			const dataView = new DataView(safetensorsData.buffer);
			let header_size = dataView.getUint32(0, true);
			let offset = 8;
			let header = JSON.parse(new TextDecoder().decode(safetensorsData.slice(offset, offset + header_size)));
			r(header.__metadata__);
		};

		var slice = file.slice(0, 1024 * 1024 * 4);
		reader.readAsArrayBuffer(slice);
	});
}

export async function importA1111(graph, parameters) {
	const p = parameters.lastIndexOf("\nSteps:");
	if (p > -1) {
		const embeddings = await api.getEmbeddings();
		const opts = parameters
			.substr(p)
			.split("\n")[1]
			.match(new RegExp("\\s*([^:]+:\\s*([^\"\\{].*?|\".*?\"|\\{.*?\\}))\\s*(,|$)", "g"))
			.reduce((p, n) => {
				const s = n.split(":");
				if (s[1].endsWith(',')) {
					s[1] = s[1].substr(0, s[1].length -1);
				}
				p[s[0].trim().toLowerCase()] = s[1].trim();
				return p;
			}, {});
		const p2 = parameters.lastIndexOf("\nNegative prompt:", p);
		if (p2 > -1) {
			let positive = parameters.substr(0, p2).trim();
			let negative = parameters.substring(p2 + 18, p).trim();

			const ckptNode = LiteGraph.createNode("CheckpointLoaderSimple");
			const clipSkipNode = LiteGraph.createNode("CLIPSetLastLayer");
			const positiveNode = LiteGraph.createNode("CLIPTextEncode");
			const negativeNode = LiteGraph.createNode("CLIPTextEncode");
			const samplerNode = LiteGraph.createNode("KSampler");
			const imageNode = LiteGraph.createNode("EmptyLatentImage");
			const vaeNode = LiteGraph.createNode("VAEDecode");
			const vaeLoaderNode = LiteGraph.createNode("VAELoader");
			const saveNode = LiteGraph.createNode("SaveImage");
			let hrSamplerNode = null;
			let hrSteps = null;

			const ceil64 = (v) => Math.ceil(v / 64) * 64;

			function getWidget(node, name) {
				return node.widgets.find((w) => w.name === name);
			}

			function setWidgetValue(node, name, value, isOptionPrefix) {
				const w = getWidget(node, name);
				if (isOptionPrefix) {
					const o = w.options.values.find((w) => w.startsWith(value));
					if (o) {
						w.value = o;
					} else {
						console.warn(`Unknown value '${value}' for widget '${name}'`, node);
						w.value = value;
					}
				} else {
					w.value = value;
				}
			}

			function createLoraNodes(clipNode, text, prevClip, prevModel) {
				const loras = [];
				text = text.replace(/<lora:([^:]+:[^>]+)>/g, function (m, c) {
					const s = c.split(":");
					const weight = parseFloat(s[1]);
					if (isNaN(weight)) {
						console.warn("Invalid LORA", m);
					} else {
						loras.push({ name: s[0], weight });
					}
					return "";
				});

				for (const l of loras) {
					const loraNode = LiteGraph.createNode("LoraLoader");
					graph.add(loraNode);
					setWidgetValue(loraNode, "lora_name", l.name, true);
					setWidgetValue(loraNode, "strength_model", l.weight);
					setWidgetValue(loraNode, "strength_clip", l.weight);
					prevModel.node.connect(prevModel.index, loraNode, 0);
					prevClip.node.connect(prevClip.index, loraNode, 1);
					prevModel = { node: loraNode, index: 0 };
					prevClip = { node: loraNode, index: 1 };
				}

				prevClip.node.connect(1, clipNode, 0);
				prevModel.node.connect(0, samplerNode, 0);
				if (hrSamplerNode) {
					prevModel.node.connect(0, hrSamplerNode, 0);
				}

				return { text, prevModel, prevClip };
			}

			function replaceEmbeddings(text) {
				if(!embeddings.length) return text;
				return text.replaceAll(
					new RegExp(
						"\\b(" + embeddings.map((e) => e.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("\\b|\\b") + ")\\b",
						"ig"
					),
					"embedding:$1"
				);
			}

			function popOpt(name) {
				const v = opts[name];
				delete opts[name];
				return v;
			}

			graph.clear();
			graph.add(ckptNode);
			graph.add(clipSkipNode);
			graph.add(positiveNode);
			graph.add(negativeNode);
			graph.add(samplerNode);
			graph.add(imageNode);
			graph.add(vaeNode);
			graph.add(vaeLoaderNode);
			graph.add(saveNode);

			ckptNode.connect(1, clipSkipNode, 0);
			clipSkipNode.connect(0, positiveNode, 0);
			clipSkipNode.connect(0, negativeNode, 0);
			ckptNode.connect(0, samplerNode, 0);
			positiveNode.connect(0, samplerNode, 1);
			negativeNode.connect(0, samplerNode, 2);
			imageNode.connect(0, samplerNode, 3);
			vaeNode.connect(0, saveNode, 0);
			samplerNode.connect(0, vaeNode, 0);
			vaeLoaderNode.connect(0, vaeNode, 1);

			const handlers = {
				model(v) {
					setWidgetValue(ckptNode, "ckpt_name", v, true);
				},
				"vae"(v) {
					setWidgetValue(vaeLoaderNode, "vae_name", v, true);
				},
				"cfg scale"(v) {
					setWidgetValue(samplerNode, "cfg", +v);
				},
				"clip skip"(v) {
					setWidgetValue(clipSkipNode, "stop_at_clip_layer", -v);
				},
				sampler(v) {
					let name = v.toLowerCase().replace("++", "pp").replaceAll(" ", "_");
					if (name.includes("karras")) {
						name = name.replace("karras", "").replace(/_+$/, "");
						setWidgetValue(samplerNode, "scheduler", "karras");
					} else {
						setWidgetValue(samplerNode, "scheduler", "normal");
					}
					const w = getWidget(samplerNode, "sampler_name");
					const o = w.options.values.find((w) => w === name || w === "sample_" + name);
					if (o) {
						setWidgetValue(samplerNode, "sampler_name", o);
					}
				},
				size(v) {
					const wxh = v.split("x");
					const w = ceil64(+wxh[0]);
					const h = ceil64(+wxh[1]);
					const hrUp = popOpt("hires upscale");
					const hrSz = popOpt("hires resize");
					hrSteps = popOpt("hires steps");
					let hrMethod = popOpt("hires upscaler");

					setWidgetValue(imageNode, "width", w);
					setWidgetValue(imageNode, "height", h);

					if (hrUp || hrSz) {
						let uw, uh;
						if (hrUp) {
							uw = w * hrUp;
							uh = h * hrUp;
						} else {
							const s = hrSz.split("x");
							uw = +s[0];
							uh = +s[1];
						}

						let upscaleNode;
						let latentNode;

						if (hrMethod.startsWith("Latent")) {
							latentNode = upscaleNode = LiteGraph.createNode("LatentUpscale");
							graph.add(upscaleNode);
							samplerNode.connect(0, upscaleNode, 0);

							switch (hrMethod) {
								case "Latent (nearest-exact)":
									hrMethod = "nearest-exact";
									break;
							}
							setWidgetValue(upscaleNode, "upscale_method", hrMethod, true);
						} else {
							const decode = LiteGraph.createNode("VAEDecodeTiled");
							graph.add(decode);
							samplerNode.connect(0, decode, 0);
							vaeLoaderNode.connect(0, decode, 1);

							const upscaleLoaderNode = LiteGraph.createNode("UpscaleModelLoader");
							graph.add(upscaleLoaderNode);
							setWidgetValue(upscaleLoaderNode, "model_name", hrMethod, true);

							const modelUpscaleNode = LiteGraph.createNode("ImageUpscaleWithModel");
							graph.add(modelUpscaleNode);
							decode.connect(0, modelUpscaleNode, 1);
							upscaleLoaderNode.connect(0, modelUpscaleNode, 0);

							upscaleNode = LiteGraph.createNode("ImageScale");
							graph.add(upscaleNode);
							modelUpscaleNode.connect(0, upscaleNode, 0);

							const vaeEncodeNode = (latentNode = LiteGraph.createNode("VAEEncodeTiled"));
							graph.add(vaeEncodeNode);
							upscaleNode.connect(0, vaeEncodeNode, 0);
							vaeLoaderNode.connect(0, vaeEncodeNode, 1);
						}

						setWidgetValue(upscaleNode, "width", ceil64(uw));
						setWidgetValue(upscaleNode, "height", ceil64(uh));

						hrSamplerNode = LiteGraph.createNode("KSampler");
						graph.add(hrSamplerNode);
						ckptNode.connect(0, hrSamplerNode, 0);
						positiveNode.connect(0, hrSamplerNode, 1);
						negativeNode.connect(0, hrSamplerNode, 2);
						latentNode.connect(0, hrSamplerNode, 3);
						hrSamplerNode.connect(0, vaeNode, 0);
					}
				},
				steps(v) {
					setWidgetValue(samplerNode, "steps", +v);
				},
				seed(v) {
					setWidgetValue(samplerNode, "seed", +v);
				},
			};

			for (const opt in opts) {
				if (opt in handlers) {
					handlers[opt](popOpt(opt));
				}
			}

			if (hrSamplerNode) {
				setWidgetValue(hrSamplerNode, "steps", hrSteps? +hrSteps : getWidget(samplerNode, "steps").value);
				setWidgetValue(hrSamplerNode, "cfg", getWidget(samplerNode, "cfg").value);
				setWidgetValue(hrSamplerNode, "scheduler", getWidget(samplerNode, "scheduler").value);
				setWidgetValue(hrSamplerNode, "sampler_name", getWidget(samplerNode, "sampler_name").value);
				setWidgetValue(hrSamplerNode, "denoise", +(popOpt("denoising strength") || "1"));
			}

			let n = createLoraNodes(positiveNode, positive, { node: clipSkipNode, index: 0 }, { node: ckptNode, index: 0 });
			positive = n.text;
			n = createLoraNodes(negativeNode, negative, n.prevClip, n.prevModel);
			negative = n.text;

			setWidgetValue(positiveNode, "text", replaceEmbeddings(positive));
			setWidgetValue(negativeNode, "text", replaceEmbeddings(negative));

			graph.arrange();

			for (const opt of ["model hash", "ensd", "version", "vae hash", "ti hashes", "lora hashes", "hashes"]) {
				delete opts[opt];
			}

			console.warn("Unhandled parameters:", opts);
		}
	}
}
