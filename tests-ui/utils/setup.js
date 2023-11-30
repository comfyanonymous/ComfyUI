require("../../web/scripts/api");

const fs = require("fs");
const path = require("path");
function* walkSync(dir) {
	const files = fs.readdirSync(dir, { withFileTypes: true });
	for (const file of files) {
		if (file.isDirectory()) {
			yield* walkSync(path.join(dir, file.name));
		} else {
			yield path.join(dir, file.name);
		}
	}
}

/**
 * @typedef { import("../../web/types/comfy").ComfyObjectInfo } ComfyObjectInfo
 */

/**
 * @param { { mockExtensions?: string[], mockNodeDefs?: Record<string, ComfyObjectInfo> } } config
 */
export function mockApi({ mockExtensions, mockNodeDefs } = {}) {
	if (!mockExtensions) {
		mockExtensions = Array.from(walkSync(path.resolve("../web/extensions/core")))
			.filter((x) => x.endsWith(".js"))
			.map((x) => path.relative(path.resolve("../web"), x));
	}
	if (!mockNodeDefs) {
		mockNodeDefs = JSON.parse(fs.readFileSync(path.resolve("./data/object_info.json")));
	}

	const events = new EventTarget();
	const mockApi = {
		addEventListener: events.addEventListener.bind(events),
		removeEventListener: events.removeEventListener.bind(events),
		dispatchEvent: events.dispatchEvent.bind(events),
		getSystemStats: jest.fn(),
		getExtensions: jest.fn(() => mockExtensions),
		getNodeDefs: jest.fn(() => mockNodeDefs),
		init: jest.fn(),
		apiURL: jest.fn((x) => "../../web/" + x),
	};
	jest.mock("../../web/scripts/api", () => ({
		get api() {
			return mockApi;
		},
	}));
}
