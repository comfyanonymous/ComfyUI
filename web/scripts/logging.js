import { $el, ComfyDialog } from "./ui.js";
import { api } from "./api.js";

$el("style", {
	textContent: `
        .comfy-logging-logs {
            display: grid;
            color: var(--fg-color);
            white-space: pre-wrap;
        }
        .comfy-logging-log {
            display: contents;
        }
        .comfy-logging-title {
            background: var(--tr-even-bg-color);
            font-weight: bold;
            margin-bottom: 5px;
            text-align: center;
        }
        .comfy-logging-log div {
            background: var(--row-bg);
            padding: 5px;
        }
    `,
	parent: document.body,
});

// Stringify function supporting max depth and removal of circular references
// https://stackoverflow.com/a/57193345
function stringify(val, depth, replacer, space, onGetObjID) {
	depth = isNaN(+depth) ? 1 : depth;
	var recursMap = new WeakMap();
	function _build(val, depth, o, a, r) {
		// (JSON.stringify() has it's own rules, which we respect here by using it for property iteration)
		return !val || typeof val != "object"
			? val
			: ((r = recursMap.has(val)),
			  recursMap.set(val, true),
			  (a = Array.isArray(val)),
			  r
					? (o = (onGetObjID && onGetObjID(val)) || null)
					: JSON.stringify(val, function (k, v) {
							if (a || depth > 0) {
								if (replacer) v = replacer(k, v);
								if (!k) return (a = Array.isArray(v)), (val = v);
								!o && (o = a ? [] : {});
								o[k] = _build(v, a ? depth : depth - 1);
							}
					  }),
			  o === void 0 ? (a ? [] : {}) : o);
	}
	return JSON.stringify(_build(val, depth), null, space);
}

const jsonReplacer = (k, v, ui) => {
	if (v instanceof Array && v.length === 1) {
		v = v[0];
	}
	if (v instanceof Date) {
		v = v.toISOString();
		if (ui) {
			v = v.split("T")[1];
		}
	}
	if (v instanceof Error) {
		let err = "";
		if (v.name) err += v.name + "\n";
		if (v.message) err += v.message + "\n";
		if (v.stack) err += v.stack + "\n";
		if (!err) {
			err = v.toString();
		}
		v = err;
	}
	return v;
};

const fileInput = $el("input", {
	type: "file",
	accept: ".json",
	style: { display: "none" },
	parent: document.body,
});

class ComfyLoggingDialog extends ComfyDialog {
	constructor(logging) {
		super();
		this.logging = logging;
	}

	clear() {
		this.logging.clear();
		this.show();
	}

	export() {
		const blob = new Blob([stringify([...this.logging.entries], 20, jsonReplacer, "\t")], {
			type: "application/json",
		});
		const url = URL.createObjectURL(blob);
		const a = $el("a", {
			href: url,
			download: `comfyui-logs-${Date.now()}.json`,
			style: { display: "none" },
			parent: document.body,
		});
		a.click();
		setTimeout(function () {
			a.remove();
			window.URL.revokeObjectURL(url);
		}, 0);
	}

	import() {
		fileInput.onchange = () => {
			const reader = new FileReader();
			reader.onload = () => {
				fileInput.remove();
				try {
					const obj = JSON.parse(reader.result);
					if (obj instanceof Array) {
						this.show(obj);
					} else {
						throw new Error("Invalid file selected.");
					}
				} catch (error) {
					alert("Unable to load logs: " + error.message);
				}
			};
			reader.readAsText(fileInput.files[0]);
		};
		fileInput.click();
	}

	createButtons() {
		return [
			$el("button", {
				type: "button",
				textContent: "Clear",
				onclick: () => this.clear(),
			}),
			$el("button", {
				type: "button",
				textContent: "Export logs...",
				onclick: () => this.export(),
			}),
			$el("button", {
				type: "button",
				textContent: "View exported logs...",
				onclick: () => this.import(),
			}),
			...super.createButtons(),
		];
	}

	getTypeColor(type) {
		switch (type) {
			case "error":
				return "red";
			case "warn":
				return "orange";
			case "debug":
				return "dodgerblue";
		}
	}

	show(entries) {
		if (!entries) entries = this.logging.entries;
		this.element.style.width = "100%";
		const cols = {
			source: "Source",
			type: "Type",
			timestamp: "Timestamp",
			message: "Message",
		};
		const keys = Object.keys(cols);
		const headers = Object.values(cols).map((title) =>
			$el("div.comfy-logging-title", {
				textContent: title,
			})
		);
		const rows = entries.map((entry, i) => {
			return $el(
				"div.comfy-logging-log",
				{
					$: (el) => el.style.setProperty("--row-bg", `var(--tr-${i % 2 ? "even" : "odd"}-bg-color)`),
				},
				keys.map((key) => {
					let v = entry[key];
					let color;
					if (key === "type") {
						color = this.getTypeColor(v);
					} else {
						v = jsonReplacer(key, v, true);

						if (typeof v === "object") {
							v = stringify(v, 5, jsonReplacer, "  ");
						}
					}

					return $el("div", {
						style: {
							color,
						},
						textContent: v,
					});
				})
			);
		});

		const grid = $el(
			"div.comfy-logging-logs",
			{
				style: {
					gridTemplateColumns: `repeat(${headers.length}, 1fr)`,
				},
			},
			[...headers, ...rows]
		);
		const els = [grid];
		if (!this.logging.enabled) {
			els.unshift(
				$el("h3", {
					style: { textAlign: "center" },
					textContent: "Logging is disabled",
				})
			);
		}
		super.show($el("div", els));
	}
}

export class ComfyLogging {
	/**
	 * @type Array<{ source: string, type: string, timestamp: Date, message: any }>
	 */
	entries = [];

	#enabled;
	#console = {};

	get enabled() {
		return this.#enabled;
	}

	set enabled(value) {
		if (value === this.#enabled) return;
		if (value) {
			this.patchConsole();
		} else {
			this.unpatchConsole();
		}
		this.#enabled = value;
	}

	constructor(app) {
		this.app = app;

		this.dialog = new ComfyLoggingDialog(this);
		this.addSetting();
		this.catchUnhandled();
		this.addInitData();
	}

	addSetting() {
		const settingId = "Comfy.Logging.Enabled";
		const htmlSettingId = settingId.replaceAll(".", "-");
		const setting = this.app.ui.settings.addSetting({
			id: settingId,
			name: settingId,
			defaultValue: true,
			onChange: (value) => {
				this.enabled = value;
			},
			type: (name, setter, value) => {
				return $el("tr", [
					$el("td", [
						$el("label", {
							textContent: "Logging",
							for: htmlSettingId,
						}),
					]),
					$el("td", [
						$el("input", {
							id: htmlSettingId,
							type: "checkbox",
							checked: value,
							onchange: (event) => {
								setter(event.target.checked);
							},
						}),
						$el("button", {
							textContent: "View Logs",
							onclick: () => {
								this.app.ui.settings.element.close();
								this.dialog.show();
							},
							style: {
								fontSize: "14px",
								display: "block",
								marginTop: "5px",
							},
						}),
					]),
				]);
			},
		});
		this.enabled = setting.value;
	}

	patchConsole() {
		// Capture common console outputs
		const self = this;
		for (const type of ["log", "warn", "error", "debug"]) {
			const orig = console[type];
			this.#console[type] = orig;
			console[type] = function () {
				orig.apply(console, arguments);
				self.addEntry("console", type, ...arguments);
			};
		}
	}

	unpatchConsole() {
		// Restore original console functions
		for (const type of Object.keys(this.#console)) {
			console[type] = this.#console[type];
		}
		this.#console = {};
	}

	catchUnhandled() {
		// Capture uncaught errors
		window.addEventListener("error", (e) => {
			this.addEntry("window", "error", e.error ?? "Unknown error");
			return false;
		});

		window.addEventListener("unhandledrejection", (e) => {
			this.addEntry("unhandledrejection", "error", e.reason ?? "Unknown error");
		});
	}

	clear() {
		this.entries = [];
	}

	addEntry(source, type, ...args) {
		if (this.enabled) {
			this.entries.push({
				source,
				type,
				timestamp: new Date(),
				message: args,
			});
		}
	}

	log(source, ...args) {
		this.addEntry(source, "log", ...args);
	}

	async addInitData() {
		if (!this.enabled) return;
		const source = "ComfyUI.Logging";
		this.addEntry(source, "debug", { UserAgent: navigator.userAgent });
		const systemStats = await api.getSystemStats();
		this.addEntry(source, "debug", systemStats);
	}
}
