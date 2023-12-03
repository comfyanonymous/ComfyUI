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
 * @param {{ 
 * 	mockExtensions?: string[], 
 * 	mockNodeDefs?: Record<string, ComfyObjectInfo>,
 * 	users?: boolean | Record<string, string>
* 	settings?: Record<string, string>
* 	userData?: Record<string, any>
 * }} config
 */
export function mockApi(config = {}) {
	let { mockExtensions, mockNodeDefs, users, settings, userData } = {
		users: true,
		settings: {},
		userData: {},
		...config,
	};
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
		createUser: jest.fn((username) => {
			if(username in users) {
				return { status: 400, json: () => "Duplicate" }
			}
			users[username + "!"] = username;
			return { status: 200, json: () => username + "!" }
		}),
		getUsers: jest.fn(() => users),
		getSettings: jest.fn(() => settings),
		storeSettings: jest.fn((v) => Object.assign(settings, v)),
		getUserData: jest.fn((f) => {
			if (f in userData) {
				return { status: 200, json: () => userData[f] };
			} else {
				return { status: 404 };
			}
		}),
		storeUserData: jest.fn((file, data) => {
			userData[file] = data;
		}),
	};
	jest.mock("../../web/scripts/api", () => ({
		get api() {
			return mockApi;
		},
	}));
}
