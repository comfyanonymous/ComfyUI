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
 * @param { 
 * { 
 * 	mockExtensions?: string[], 
 * 	mockNodeDefs?: Record<string, ComfyObjectInfo>,
 * 	users?: boolean | Record<string, string>
* 	settings?: Record<string, string>
* 	userData?: Record<string, any>
 * } } config
 */
export function mockApi({ mockExtensions, mockNodeDefs, users, settings, userData } = {}) {
	if (!mockExtensions) {
		mockExtensions = Array.from(walkSync(path.resolve("../web/extensions/core")))
			.filter((x) => x.endsWith(".js"))
			.map((x) => path.relative(path.resolve("../web"), x));
	}
	if (!mockNodeDefs) {
		mockNodeDefs = JSON.parse(fs.readFileSync(path.resolve("./data/object_info.json")));
	}
	if(!users) {
		users = true;
	}
	if(!settings) {
		settings = {};
	}
	if(!userData) {
		userData = {};
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
		getUsers: jest.fn(() => users),
		getSettings: jest.fn(() => settings ?? {}),
		getUserData: jest.fn(f => {
			if(f in userData) {
				return { status: 200, json: () => userData[f] };
			} else {
				return { status: 404 }
			}
		})
	};
	jest.mock("../../web/scripts/api", () => ({
		get api() {
			return mockApi;
		},
	}));
}
