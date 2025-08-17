import { app } from "../../../scripts/app.js";
app.registerExtension({
	name: "pysssss.ContextMenuHook",
	init() {
		const getOrSet = (target, name, create) => {
			if (name in target) return target[name];
			return (target[name] = create());
		};
		const symbol = getOrSet(window, "__pysssss__", () => Symbol("__pysssss__"));
		const store = getOrSet(window, symbol, () => ({}));
		const contextMenuHook = getOrSet(store, "contextMenuHook", () => ({}));
		for (const e of ["ctor", "preAddItem", "addItem"]) {
			if (!contextMenuHook[e]) {
				contextMenuHook[e] = [];
			}
		}

		// Big ol' hack to get allow customizing the context menu
		// Replace the addItem function with our own that wraps the context of "this" with a proxy
		// That proxy then replaces the constructor with another proxy
		// That proxy then calls the custom ContextMenu that supports filters
		const ctorProxy = new Proxy(LiteGraph.ContextMenu, {
			construct(target, args) {
				return new LiteGraph.ContextMenu(...args);
			},
		});

		function triggerCallbacks(name, getArgs, handler) {
			const callbacks = contextMenuHook[name];
			if (callbacks && callbacks instanceof Array) {
				for (const cb of callbacks) {
					const r = cb(...getArgs());
					handler?.call(this, r);
				}
			} else {
				console.warn("[pysssss 🐍]", `invalid ${name} callbacks`, callbacks, name in contextMenuHook);
			}
		}

		const addItem = LiteGraph.ContextMenu.prototype.addItem;
		LiteGraph.ContextMenu.prototype.addItem = function () {
			const proxy = new Proxy(this, {
				get(target, prop) {
					if (prop === "constructor") {
						return ctorProxy;
					}
					return target[prop];
				},
			});
			proxy.__target__ = this;

			let el;
			let args = arguments;
			triggerCallbacks(
				"preAddItem",
				() => [el, this, args],
				(r) => {
					if (r !== undefined) el = r;
				}
			);

			if (el === undefined) {
				el = addItem.apply(proxy, arguments);
			}

			triggerCallbacks(
				"addItem",
				() => [el, this, args],
				(r) => {
					if (r !== undefined) el = r;
				}
			);
			return el;
		};

		// We also need to patch the ContextMenu constructor to unwrap the parent else it fails a LiteGraph type check
		const ctxMenu = LiteGraph.ContextMenu;
		LiteGraph.ContextMenu = function (values, options) {
			if (options?.parentMenu) {
				if (options.parentMenu.__target__) {
					options.parentMenu = options.parentMenu.__target__;
				}
			}

			triggerCallbacks("ctor", () => [values, options]);
			return ctxMenu.call(this, values, options);
		};
		LiteGraph.ContextMenu.prototype = ctxMenu.prototype;
	},
});
