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
				if (type === "tEXt") {
					// Get the keyword
					let keyword_end = offset + 8;
					while (pngData[keyword_end] !== 0) {
						keyword_end++;
					}
					const keyword = String.fromCharCode(...pngData.slice(offset + 8, keyword_end));
					// Get the text
					const text = String.fromCharCode(...pngData.slice(keyword_end + 1, offset + 8 + length));
					txt_chunks[keyword] = text;
				}

				offset += 12 + length;
			}

			r(txt_chunks);
		};

		reader.readAsArrayBuffer(file);
	});
}
