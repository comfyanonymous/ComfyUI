import { app } from "../../scripts/app.js";
import { $el } from "../../scripts/ui.js";
import { 
	manager_instance, rebootAPI, install_via_git_url, 
	fetchData, md5, icons 
} from  "./common.js";

// https://cenfun.github.io/turbogrid/api.html
import TG from "./turbogrid.esm.js";

const pageCss = `
.cn-manager {
	--grid-font: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans", Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji";
	z-index: 10001;
	width: 80%;
	height: 80%;
	display: flex;
	flex-direction: column;
	gap: 10px;
	color: var(--fg-color);
	font-family: arial, sans-serif;
}

.cn-manager .cn-flex-auto {
	flex: auto;
}

.cn-manager button {
	font-size: 16px;
	color: var(--input-text);
    background-color: var(--comfy-input-bg);
    border-radius: 8px;
    border-color: var(--border-color);
    border-style: solid;
    margin: 0;
	padding: 4px 8px;
	min-width: 100px;
}

.cn-manager button:disabled,
.cn-manager input:disabled,
.cn-manager select:disabled {
	color: gray;
}

.cn-manager button:disabled {
	background-color: var(--comfy-input-bg);
}

.cn-manager .cn-manager-restart {
	display: none;
	background-color: #500000;
	color: white;
}

.cn-manager-header {
	display: flex;
	flex-wrap: wrap;
	gap: 5px;
	align-items: center;
	padding: 0 5px;
}

.cn-manager-header label {
	display: flex;
	gap: 5px;
	align-items: center;
}

.cn-manager-filter {
	height: 28px;
	line-height: 28px;
}

.cn-manager-keywords {
	height: 28px;
	line-height: 28px;
	padding: 0 5px 0 26px;
	background-size: 16px;
	background-position: 5px center;
	background-repeat: no-repeat;
	background-image: url("data:image/svg+xml;charset=utf8,${encodeURIComponent(icons.search.replace("currentColor", "#888"))}");
}

.cn-manager-status {
	padding-left: 10px;
}

.cn-manager-grid {
	flex: auto;
	border: 1px solid var(--border-color);
	overflow: hidden;
}

.cn-manager-selection {
	display: flex;
	flex-wrap: wrap;
	gap: 10px;
	align-items: center;
}

.cn-manager-message {
	
}

.cn-manager-footer {
	display: flex;
	flex-wrap: wrap;
	gap: 10px;
	align-items: center;
}

.cn-manager-grid .tg-turbogrid {
	font-family: var(--grid-font);
	font-size: 15px;
	background: var(--bg-color);
}

.cn-manager-grid .cn-node-name a {
	color: skyblue;
	text-decoration: none;
	word-break: break-word;
}

.cn-manager-grid .cn-node-desc a {
	color: #5555FF;
    font-weight: bold;
	text-decoration: none;
}

.cn-manager-grid .tg-cell a:hover {
	text-decoration: underline;
}

.cn-manager-grid .cn-extensions-button,
.cn-manager-grid .cn-conflicts-button {
	display: inline-block;
	width: 20px;
	height: 20px;
	color: green;
	border: none;
	padding: 0;
	margin: 0;
	background: none;
	min-width: 20px;
}

.cn-manager-grid .cn-conflicts-button {
	color: orange;
}

.cn-manager-grid .cn-extensions-list,
.cn-manager-grid .cn-conflicts-list {
	line-height: normal;
	text-align: left;
	max-height: 80%;
	min-height: 200px;
	min-width: 300px;
	overflow-y: auto;
	font-size: 12px;
	border-radius: 5px;
	padding: 10px;
	filter: drop-shadow(2px 5px 5px rgb(0 0 0 / 30%));
	white-space: normal;
}

.cn-manager-grid .cn-extensions-list {
	border-color: var(--bg-color);
}

.cn-manager-grid .cn-conflicts-list {
	background-color: #CCCC55;
	color: #AA3333;
}

.cn-manager-grid .cn-extensions-list h3,
.cn-manager-grid .cn-conflicts-list h3 {
	margin: 0;
	padding: 5px 0;
	color: #000;
}

.cn-tag-list {
	display: flex;
	flex-wrap: wrap;
	gap: 5px;
	align-items: center;
	margin-bottom: 5px;
}

.cn-tag-list > div {
	background-color: var(--border-color);
	border-radius: 5px;
    padding: 0 5px;
}

.cn-install-buttons {
	display: flex;
	flex-direction: column;
	gap: 3px;
	padding: 3px;
	align-items: center;
    justify-content: center;
    height: 100%;
}

.cn-selected-buttons {
	display: flex;
	gap: 5px;
	align-items: center;
	padding-right: 20px;
}

.cn-manager .cn-btn-enable {
	background-color: blue;
	color: white;
}

.cn-manager .cn-btn-disable {
	background-color: MediumSlateBlue;
	color: white;
}

.cn-manager .cn-btn-update {
	background-color: blue;
	color: white;
}

.cn-manager .cn-btn-try-update {
	background-color: Gray;
	color: white;
}

.cn-manager .cn-btn-try-fix {
	background-color: #6495ED;
	color: white;
}

.cn-manager .cn-btn-install {
	background-color: black;
	color: white;
}

.cn-manager .cn-btn-try-install {
	background-color: Gray;
	color: white;
}

.cn-manager .cn-btn-uninstall {
	background-color: red;
	color: white;
}

@keyframes cn-btn-loading-bg {
    0% {
        left: 0;
    }
    100% {
        left: -105px;
    }
}

.cn-manager button.cn-btn-loading {
    position: relative;
    overflow: hidden;
    border-color: rgb(0 119 207 / 80%);
	background-color: var(--comfy-input-bg);
}

.cn-manager button.cn-btn-loading::after {
    position: absolute;
    top: 0;
    left: 0;
    content: "";
    width: 500px;
    height: 100%;
    background-image: repeating-linear-gradient(
        -45deg,
        rgb(0 119 207 / 30%),
        rgb(0 119 207 / 30%) 10px,
        transparent 10px,
        transparent 15px
    );
    animation: cn-btn-loading-bg 2s linear infinite;
}

.cn-manager-light .cn-node-name a {
	color: blue;
}

.cn-manager-light .cm-warn-note {
	background-color: #ccc !important;
}

.cn-manager-light .cn-btn-install {
	background-color: #333;
}

`;

const pageHtml = `
<div class="cn-manager-header">
	<label>Filter
		<select class="cn-manager-filter"></select>
	</label>
	<input class="cn-manager-keywords" type="search" placeholder="Search" />
	<div class="cn-manager-status"></div>
	<div class="cn-flex-auto"></div>
	<div class="cn-manager-channel"></div>
</div>
<div class="cn-manager-grid"></div>
<div class="cn-manager-selection"></div>
<div class="cn-manager-message"></div>
<div class="cn-manager-footer">
	<button class="cn-manager-close">Close</button>
	<button class="cn-manager-restart">Restart</button>
	<div class="cn-flex-auto"></div>
	<button class="cn-manager-check-update">Check Update</button>
	<button class="cn-manager-check-missing">Check Missing</button>
	<button class="cn-manager-install-url">Install via Git URL</button>
</div>
`;

const ShowMode = {
	NORMAL: "Normal",
	UPDATE: "Update",
	MISSING: "Missing",
	ALTERNATIVES: "Alternatives"
};

export class CustomNodesManager {
	static instance = null;
	static ShowMode = ShowMode;

	constructor(app, manager_dialog) {
		this.app = app;
		this.manager_dialog = manager_dialog;
		this.id = "cn-manager";

		app.registerExtension({
			name: "Comfy.CustomNodesManager",
			afterConfigureGraph: (missingNodeTypes) => {
				const item = this.getFilterItem(ShowMode.MISSING);
				if (item) {
					item.hasData = false;
					item.hashMap = null;
				}
			}
		});

		this.filter = '';
		this.keywords = '';
		this.restartMap = {};

		this.init();
	}

	init() {

		if (!document.querySelector(`style[context="${this.id}"]`)) {
			const $style = document.createElement("style");
			$style.setAttribute("context", this.id);
			$style.innerHTML = pageCss;
			document.head.appendChild($style);
		}

		this.element = $el("div", {
			parent: document.body,
			className: "comfy-modal cn-manager"
		});
		this.element.innerHTML = pageHtml;
		this.initFilter();
		this.bindEvents();
		this.initGrid();
	}

	initFilter() {
		const $filter  = this.element.querySelector(".cn-manager-filter");
		const filterList = [{
			label: "All",
			value: "",
			hasData: true
		}, {
			label: "Installed",
			value: "True",
			hasData: true
		}, {
			label: "Disabled",
			value: "Disabled",
			hasData: true
		}, {
			label: "Import Failed",
			value: "Fail",
			hasData: true
		}, {
			label: "Not Installed",
			value: "False",
			hasData: true
		}, {
			label: "Unknown",
			value: "None",
			hasData: true
		}, {
			label: "Update",
			value: ShowMode.UPDATE,
			hasData: false
		}, {
			label: "Missing",
			value: ShowMode.MISSING,
			hasData: false
		}, {
			label: "Alternatives of A1111",
			value: ShowMode.ALTERNATIVES,
			hasData: false
		}];
		this.filterList = filterList;
		$filter.innerHTML = filterList.map(item => {
			return `<option value="${item.value}">${item.label}</option>`
		}).join("");
	}

	getFilterItem(filter) {
		return this.filterList.find(it => it.value === filter)
	}

	getInstallButtons(installed, title) {

		const buttons = {
			"enable": {
				label: "Enable",
				mode: "toggle_active"
			},
			"disable": {
				label: "Disable",
				mode: "toggle_active"
			},

			"update": {
				label: "Update",
				mode: "update"
			},
			"try-update": {
				label: "Try update",
				mode: "update"
			},

			"try-fix": {
				label: "Try fix",
				mode: "fix"
			},

			"install": {
				label: "Install",
				mode: "install"
			},
			"try-install": {
				label: "Try install",
				mode: "install"
			},
			"uninstall": {
				label: "Uninstall",
				mode: "uninstall"
			}
		}

		const installGroups = {
			"Disabled": ["enable", "uninstall"],
			"Update": ["update", "disable", "uninstall"],
			"Fail": ["try-fix", "uninstall"],
			"True": ["try-update", "disable", "uninstall"],
			"False": ["install"],
			'None': ["try-install"]
		}

		if (!manager_instance.update_check_checkbox.checked) {
			installGroups.True = installGroups.True.filter(it => it !== "try-update");
		}

		if (title === "ComfyUI-Manager") {
			installGroups.True = installGroups.True.filter(it => it !== "disable");
		}

		const list = installGroups[installed];
		if (!list) {
			return "";
		}

		return list.map(id => {
			const bt = buttons[id];
			return `<button class="cn-btn-${id}" group="${installed}" mode="${bt.mode}">${bt.label}</button>`;
		}).join("");
	}

	getButton(target) {
		if(!target) {
			return;
		}
		const mode = target.getAttribute("mode");
		if (!mode) {
			return;
		}
		const group = target.getAttribute("group");
		if (!group) {
			return;
		}
		return {
			group,
			mode,
			target,
			label: target.innerText
		}
	}

	bindEvents() {
		const eventsMap = {
			".cn-manager-filter": {
				change: (e) => {

					if (this.grid) {
						this.grid.selectAll(false);
					}

					const value = e.target.value
					this.filter = value;
					const item = this.getFilterItem(value);
					if (item && !item.hasData) {
						this.loadData(value);
						return;
					}
					this.updateGrid();
				}
			},

			".cn-manager-keywords": {
				input: (e) => {
					const keywords = `${e.target.value}`.trim();
					if (keywords !== this.keywords) {
						this.keywords = keywords;
						this.updateGrid();
					}
				},
				focus: (e) => e.target.select()
			},

			".cn-manager-selection": {
				click: (e) => {
					const btn = this.getButton(e.target);
					if (btn) {
						const nodes = this.selectedMap[btn.group];
						if (nodes) {
							this.installNodes(nodes, btn);
						}
					}
				}
			},

			".cn-manager-close": {
				click: (e) => this.close()
			},

			".cn-manager-restart": {
				click: () => {
					if(rebootAPI()) {
						this.close();
						this.manager_dialog.close();
					}
				}
			},

			".cn-manager-check-update": {
				click: (e) => {
					e.target.classList.add("cn-btn-loading");
					this.setFilter(ShowMode.UPDATE);
					this.loadData(ShowMode.UPDATE);
				}
			},

			".cn-manager-check-missing": {
				click: (e) => {
					e.target.classList.add("cn-btn-loading");
					this.setFilter(ShowMode.MISSING);
					this.loadData(ShowMode.MISSING);
				}
			},

			".cn-manager-install-url": {
				click: (e) => {
					const url = prompt("Please enter the URL of the Git repository to install", "");
					if (url !== null) {
						install_via_git_url(url, this.manager_dialog);
					}
				}
			}
		};
		Object.keys(eventsMap).forEach(selector => {
			const target = this.element.querySelector(selector);
			if (target) {
				const events = eventsMap[selector];
				if (events) {
					Object.keys(events).forEach(type => {
						target.addEventListener(type, events[type]);
					});
				}
			}
		});
	}

	// ===========================================================================================

	initGrid() {
		const container = this.element.querySelector(".cn-manager-grid");
		const grid = new TG.Grid(container);
		this.grid = grid;
		
		let prevViewRowsLength = -1;
		grid.bind('onUpdated', (e, d) => {

			const viewRows = grid.viewRows;
			if (viewRows.length !== prevViewRowsLength) {
				prevViewRowsLength = viewRows.length;
				this.showStatus(`${prevViewRowsLength.toLocaleString()} custom nodes`);
			}

        });

        grid.bind('onSelectChanged', (e, changes) => {
            this.renderSelected();
        });

		grid.bind('onClick', (e, d) => {
			const btn = this.getButton(d.e.target);
			if (btn) {
				this.installNodes([d.rowItem.hash], btn, d.rowItem.title);
			}
        });

		grid.setOption({
			theme: 'dark',
			selectVisible: true,
			selectMultiple: true,
			selectAllVisible: true,

			textSelectable: true,
			scrollbarRound: true,

			frozenColumn: 1,
			rowNotFound: "No Results",

			rowHeight: 40,
			bindWindowResize: true,
			bindContainerResize: true,

			cellResizeObserver: (rowItem, columnItem) => {
				const autoHeightColumns = ['title', 'installed', 'description', "alternatives"];
				return autoHeightColumns.includes(columnItem.id)
			},

			// updateGrid handler for filter and keywords
			rowFilter: (rowItem) => {

				const searchableColumns = ["title", "author", "description"];
				if (this.hasAlternatives()) {
					searchableColumns.push("alternatives");
				}

				let shouldShown = grid.highlightKeywordsFilter(rowItem, searchableColumns, this.keywords);

				if (shouldShown) {
					if(this.filter && rowItem.filterTypes) {
						shouldShown = rowItem.filterTypes.includes(this.filter);
					}
				}

				return shouldShown;
			}
		});

	}

	hasAlternatives() {
		return this.filter === ShowMode.ALTERNATIVES
	}

	renderGrid() {

		// update theme
		const colorPalette = this.app.ui.settings.settingsValues['Comfy.ColorPalette'];
		Array.from(this.element.classList).forEach(cn => {
			if (cn.startsWith("cn-manager-")) {
				this.element.classList.remove(cn);
			}
		});
		this.element.classList.add(`cn-manager-${colorPalette}`);

		const options = {
			theme: colorPalette === "light" ? "" : "dark"
		};

		const rows = this.custom_nodes || [];
		rows.forEach((item, i) => {
			item.id = i + 1;
			const nodeKey = item.files[0];
			const extensionInfo = this.extension_mappings[nodeKey];
			if(extensionInfo) {
				const { extensions, conflicts } = extensionInfo;
				if (extensions.length) {
					item.extensions = extensions.length;
					item.extensionsList = extensions;
				}
				if (conflicts) {
					item.conflicts = conflicts.length;
					item.conflictsList = conflicts;
				}
			}
		});

		const columns = [{
			id: 'id',
			name: 'ID',
			width: 50,
			align: 'center'
		}, {
			id: 'title',
			name: 'Title',
			width: 200,
			minWidth: 100,
			maxWidth: 500,
			classMap: 'cn-node-name',
			formatter: (title, rowItem, columnItem) => {
				return `${rowItem.installed === 'Fail' ? '<font color="red"><B>(IMPORT FAILED)</B></font>' : ''}
					<a href=${rowItem.reference} target="_blank"><b>${title}</b></a>`;
			}
		}, {
			id: 'installed',
			name: 'Install',
			width: 130,
			minWidth: 110,
			maxWidth: 200,
			sortable: false,
			align: 'center',
			formatter: (installed, rowItem, columnItem) => {
				if (rowItem.restart) {
					return `<font color="red">Restart Required</span>`;
				}
				const buttons = this.getInstallButtons(installed, rowItem.title);
				return `<div class="cn-install-buttons">${buttons}</div>`;
			}
		}, {
			id: "alternatives",
			name: "Alternatives",
			width: 400,
			maxWidth: 5000,
			invisible: !this.hasAlternatives(),
			classMap: 'cn-node-desc'
		}, {
			id: 'description',
			name: 'Description',
			width: 400,
			maxWidth: 5000,
			classMap: 'cn-node-desc'
		}, {
			id: "extensions",
			name: "Extensions",
			width: 80,
			align: 'center',
			formatter: (extensions, rowItem, columnItem) => {
				const extensionsList = rowItem.extensionsList;
				if (!extensionsList) {
					return;
				}
				const list = [];
				const eId = `popover_extensions_${columnItem.id}_${rowItem.tg_index}`; 
				list.push(`<button popovertarget="${eId}" title="${extensionsList.length} Extension Nodes" class="cn-extensions-button">${icons.extensions}</button>`)
				list.push(`<div popover id="${eId}" class="cn-extensions-list">`)
				list.push(`<h3>【${rowItem.title}】Extension Nodes (${extensionsList.length})</h3>`);
				extensionsList.forEach(en => {
					list.push(`<li>${en}</li>`);
				})
				list.push("</div>");
				return list.join("");
			}
		}, {
			id: "conflicts",
			name: "Conflicts",
			width: 80,
			align: 'center',
			formatter: (conflicts, rowItem, columnItem) => {
				const conflictsList = rowItem.conflictsList;
				if (!conflictsList) {
					return;
				}
				const list = [];
				const cId = `popover_conflicts_${columnItem.id}_${rowItem.tg_index}`; 
				list.push(`<button popovertarget="${cId}" title="${conflictsList.length} Conflicted Nodes" class="cn-conflicts-button">${icons.conflicts}</button>`)
				list.push(`<div popover id="${cId}" class="cn-conflicts-list">`)
				list.push(`<h3>【${rowItem.title}】Conflicted Nodes (${conflictsList.length})</h3>`);
				conflictsList.forEach(en => {
					let [node_name, extension_name] = en;
					extension_name = extension_name.split('/').filter(it => it).pop();
					if(extension_name.endsWith('.git')) {
						extension_name = extension_name.slice(0, -4);
					}
					list.push(`<li><B>${node_name}</B> [${extension_name}]</li>`);
				})
				list.push("</div>");
				return list.join("");
			}
		}, {
			id: 'author',
			name: 'Author',
			width: 120,
			classMap: "cn-node-author", 
			formatter: (author, rowItem, columnItem) => {
				if (rowItem.trust) {
					return `<span title="This author has been active for more than six months in GitHub">✅ ${author}</span>`;
				}
				return author;
			}
		}, {
			id: 'stars',
			name: '★',
			align: 'center',
			classMap: "cn-node-stars",
			formatter: (stars) => {
				if (stars < 0) {
					return 'N/A';
				}
				if (typeof stars === 'number') {
					return stars.toLocaleString();
				}
				return stars;
			}
		}, {
			id: 'last_update',
			name: 'Last Update',
			align: 'center',
			type: 'date',
			width: 100,
			classMap: "cn-node-last-update",
			formatter: (last_update) => {
				if (last_update < 0) {
					return 'N/A';
				}
				return `${last_update}`.split(' ')[0];
			}
		}];

		this.grid.setData({
			options,
			rows,
			columns
		});

		this.grid.render();
		
	}

	updateGrid() {
		if (this.grid) {
			this.grid.update();
			if (this.hasAlternatives()) {
				this.grid.showColumn("alternatives");
			} else {
				this.grid.hideColumn("alternatives");
			}
		}
	}

	// ===========================================================================================

	renderSelected() {
		const selectedList = this.grid.getSelectedRows();
		if (!selectedList.length) {
			this.showSelection("");
			return;
		}

		const selectedMap = {};
		selectedList.forEach(item => {
			let type = item.installed;
			if (item.restart) {
				type = "Restart Required";
			}
			if (selectedMap[type]) {
				selectedMap[type].push(item.hash);
			} else {
				selectedMap[type] = [item.hash];
			}
		});

		this.selectedMap = selectedMap;

		const list = [];
		Object.keys(selectedMap).forEach(v => {
			const filterItem = this.getFilterItem(v);
			list.push(`<div class="cn-selected-buttons">
				<span>Selected <b>${selectedMap[v].length}</b> ${filterItem ? filterItem.label : v}</span>
				${this.grid.hasMask ? "" : this.getInstallButtons(v)}
			</div>`);
		});

		this.showSelection(list.join(""));
	}

	focusInstall(item, mode) {
		const cellNode = this.grid.getCellNode(item, "installed");
		if (cellNode) {
			const cellBtn = cellNode.querySelector(`button[mode="${mode}"]`);
			if (cellBtn) {
				cellBtn.classList.add("cn-btn-loading");
				return true
			}
		}
	}

	async installNodes(list, btn, title) {
		
		const { target, label, mode} = btn;

		if(mode === "uninstall") {
			title = title || `${list.length} custom nodes`;
			if (!confirm(`Are you sure uninstall ${title}?`)) {
				return;
			}
		}

		target.classList.add("cn-btn-loading");
		this.showLoading();
		this.showError("");

		let needRestart = false;
		let errorMsg = "";
		for (const hash of list) {

			const item = this.grid.getRowItemBy("hash", hash);
			if (!item) {
				errorMsg = `Not found custom node: ${hash}`;
				break;
			}

			this.grid.scrollRowIntoView(item);

			if (!this.focusInstall(item, mode)) {
				this.grid.onNextUpdated(() => {
					this.focusInstall(item, mode);
				});
			}

			this.showStatus(`${label} ${item.title} ...`);

			const data = item.originalData;
			const res = await fetchData(`/customnode/${mode}`, {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify(data)
			});

			if (res.error) {

				errorMsg = `${item.title} ${mode} failed: `;
				if(res.status == 403) {
					errorMsg += `This action is not allowed with this security level configuration.`;
				} else if(res.status == 404) {
					errorMsg += `With the current security level configuration, only custom nodes from the <B>"default channel"</B> can be installed.`;
				} else {
					errorMsg += res.error.message;
				}

				break;
			}

			needRestart = true;

			this.grid.setRowSelected(item, false);
			item.restart = true;
			this.restartMap[item.hash] = true;
			this.grid.updateCell(item, "installed");

			//console.log(res.data);

		}

		this.hideLoading();
		target.classList.remove("cn-btn-loading");

		if (errorMsg) {
			this.showError(errorMsg);
		} else {
			this.showStatus(`${label} ${list.length} custom node(s) successfully`);
		}

		if (needRestart) {
			this.showRestart();
			this.showMessage(`To apply the installed/updated/disabled/enabled custom node, please restart ComfyUI. And refresh browser.`, "red")
		}

	}

	// ===========================================================================================

	async getExtensionMappings() {
		const mode = manager_instance.datasrc_combo.value;
		this.showStatus(`Loading extension mappings (${mode}) ...`);
		const res = await fetchData(`/customnode/getmappings?mode=${mode}`);
		if (res.error) {
			console.log(res.error);
			return {}
		}
	
		const data = res.data;

		const extension_mappings = {};
		const conflicts_map = {};
		Object.keys(data).forEach(k => {
			const [extensions, metadata] = data[k];
			extension_mappings[k] = {
				extensions,
				metadata
			}
			extensions.forEach(node => {
				let l = conflicts_map[node];
				if(!l) {
					l = [];
					conflicts_map[node] = l;
				}
				l.push(k);
			})
		})

		Object.keys(conflicts_map).forEach(node => {
			const list = conflicts_map[node];
			if(list.length > 1) {
				list.forEach(k => {
					const item = extension_mappings[k];
					if(!item) {
						console.log(`not found ${k}`)
						return;
					}

					if (!item.conflicts) {
						item.conflicts = [];
					}
					list.forEach(key => {
						if(k !== key) {
							item.conflicts.push([node, key])
						}
					})
				})
			}
		})
	
		return extension_mappings;
	}

	async getMissingNodes() {
		const mode = manager_instance.datasrc_combo.value;
		this.showStatus(`Loading missing nodes (${mode}) ...`);
		const res = await fetchData(`/customnode/getmappings?mode=${mode}`);
		if (res.error) {
			this.showError(`Failed to get custom node mappings: ${res.error}`);
			return;
		}

		const mappings = res.data;

		// build regex->url map
		const regex_to_url = [];
		this.custom_nodes.forEach(node => {
			if(node.nodename_pattern) {
				regex_to_url.push({
					regex: new RegExp(node.nodename_pattern), 
					url: node.files[0]
				});
			}
		});

		// build name->url map
		const name_to_urls = {};
		for (const url in mappings) {
			const names = mappings[url];

			for(const name in names[0]) {
				let v = name_to_urls[names[0][name]];
				if(v == undefined) {
					v = [];
					name_to_urls[names[0][name]] = v;
				}
				v.push(url);
			}
		}

		const registered_nodes = new Set();
		for (let i in LiteGraph.registered_node_types) {
			registered_nodes.add(LiteGraph.registered_node_types[i].type);
		}

		const missing_nodes = new Set();
		const workflow = app.graph.serialize();
		const group_nodes = workflow.extra && workflow.extra.groupNodes ? workflow.extra.groupNodes : [];
		let nodes = workflow.nodes;

		for (let i in group_nodes) {
			let group_node = group_nodes[i];
			nodes = nodes.concat(group_node.nodes);
		}

		for (let i in nodes) {
			const node_type = nodes[i].type;
			if(node_type.startsWith('workflow/'))
				continue;

			if (!registered_nodes.has(node_type)) {
				const urls = name_to_urls[node_type.trim()];
				if(urls)
					urls.forEach(url => {
						missing_nodes.add(url);
					});
				else {
					for(let j in regex_to_url) {
						if(regex_to_url[j].regex.test(node_type)) {
							missing_nodes.add(regex_to_url[j].url);
						}
					}
				}
			}
		}

		const resUnresolved = await fetchData(`/component/get_unresolved`);
		const unresolved = resUnresolved.data;
		if (unresolved && unresolved.nodes) {
			unresolved.nodes.forEach(node_type => {
				const url = name_to_urls[node_type];
				if(url) {
					missing_nodes.add(url);
				}
			});
		}

		const hashMap = {};
		this.custom_nodes.forEach(item => {
			if (item.files.some(file => missing_nodes.has(file))) {
				hashMap[item.hash] = true;
			}
		});
		return hashMap;
	}

	async getAlternatives() {

		const mode = manager_instance.datasrc_combo.value;
		this.showStatus(`Loading alternatives (${mode}) ...`);
		const res = await fetchData(`/customnode/alternatives?mode=${mode}`);
		if (res.error) {
			this.showError(`Failed to get alternatives: ${res.error}`);
			return [];
		}

		const hashMap = {};
		const { items } = res.data;

		items.forEach(item => {

			const custom_node = this.custom_nodes.find(node => node.files.find(file => file === item.id));
			if (!custom_node) {
				console.log(`Not found custom node: ${item.id}`);
				return;
			}

			const tags = `${item.tags}`.split(",").map(tag => {
				return `<div>${tag.trim()}</div>`;
			}).join("")

			hashMap[custom_node.hash] = {
				alternatives: `<div class="cn-tag-list">${tags}</div> ${item.description}`
			}

		});
	
		return hashMap
	}

	async loadData(show_mode = ShowMode.NORMAL) {
		this.show_mode = show_mode;
		console.log("Show mode:", show_mode);

		this.showLoading();

		this.extension_mappings = await this.getExtensionMappings();

		const mode = manager_instance.datasrc_combo.value;
		this.showStatus(`Loading custom nodes (${mode}) ...`);

		const skip_update = this.show_mode === ShowMode.UPDATE ? "" : "&skip_update=true";
		const res = await fetchData(`/customnode/getlist?mode=${mode}${skip_update}`);
		if (res.error) {
			this.showError("Failed to get custom node list.");
			this.hideLoading();
			return
		}
		
		const { channel, custom_nodes} = res.data;
		this.channel = channel;
		this.custom_nodes = custom_nodes;

		if(this.channel !== 'default') {
			this.element.querySelector(".cn-manager-channel").innerHTML = `Channel: ${this.channel} (Incomplete list)`;
		}

		for (const item of custom_nodes) {
			item.originalData = JSON.parse(JSON.stringify(item));
			const message = item.title + item.files[0];
			item.hash = md5(message);
		}

		const filterItem = this.getFilterItem(this.show_mode);
		if(filterItem) {
			let hashMap;
			if(this.show_mode == ShowMode.UPDATE) {
				hashMap = {};
				custom_nodes.forEach(it => {
					if (it.installed === "Update") {
						hashMap[it.hash] = true;
					}
				});
			} else if(this.show_mode == ShowMode.MISSING) {
				hashMap = await this.getMissingNodes();
			} else if(this.show_mode == ShowMode.ALTERNATIVES) {
				hashMap = await this.getAlternatives();
			}
			filterItem.hashMap = hashMap;
			filterItem.hasData = true;
		}

		custom_nodes.forEach(nodeItem => {
			if (this.restartMap[nodeItem.hash]) {
				nodeItem.restart = true;
			}
			const filterTypes = new Set();
			this.filterList.forEach(filterItem => {
				const { value, hashMap } = filterItem;
				if (hashMap) {
					const hashData = hashMap[nodeItem.hash]
					if (hashData) {
						filterTypes.add(value);
						if (value === ShowMode.UPDATE) {
							nodeItem.installed = "Update";
						}
						if (typeof hashData === "object") {
							Object.assign(nodeItem, hashData);
						}
					}
				} else {
					if (nodeItem.installed === value) {
						filterTypes.add(value);
					}
					const map = {
						"Update": "True",
						"Disabled": "True",
						"Fail": "True",
						"None": "False"
					}
					if (map[nodeItem.installed]) {
						filterTypes.add(map[nodeItem.installed]);
					}
				}
			});
			nodeItem.filterTypes = Array.from(filterTypes);
		});

		this.renderGrid();

		this.hideLoading();
		
	}

	// ===========================================================================================

	showSelection(msg) {
		this.element.querySelector(".cn-manager-selection").innerHTML = msg;
	}

	showError(err) {
		this.showMessage(err, "red");
	}

	showMessage(msg, color) {
		if (color) {
			msg = `<font color="${color}">${msg}</font>`;
		}
		this.element.querySelector(".cn-manager-message").innerHTML = msg;
	}

	showStatus(msg, color) {
		if (color) {
			msg = `<font color="${color}">${msg}</font>`;
		}
		this.element.querySelector(".cn-manager-status").innerHTML = msg;
	}

	showLoading() {
		this.setDisabled(true);
		if (this.grid) {
			this.grid.showLoading();
			this.grid.showMask({
				opacity: 0.05
			});
		}
	}

	hideLoading() {
		this.setDisabled(false);
		if (this.grid) {
			this.grid.hideLoading();
			this.grid.hideMask();
		}
	}

	setDisabled(disabled) {

		const $close = this.element.querySelector(".cn-manager-close");
		const $restart = this.element.querySelector(".cn-manager-restart");

		const list = [
			".cn-manager-header input",
			".cn-manager-header select",
			".cn-manager-footer button",
			".cn-manager-selection button"
		].map(s => {
			return Array.from(this.element.querySelectorAll(s));
		})
		.flat()
		.filter(it => {
			return it !== $close && it !== $restart;
		});
		
		list.forEach($elem => {
			if (disabled) {
				$elem.setAttribute("disabled", "disabled");
			} else {
				$elem.removeAttribute("disabled");
			}
		});

		Array.from(this.element.querySelectorAll(".cn-btn-loading")).forEach($elem => {
			$elem.classList.remove("cn-btn-loading");
		});

	}

	showRestart() {
		this.element.querySelector(".cn-manager-restart").style.display = "block";
	}

	setFilter(filterValue) {
		let filter = "";
		const filterItem = this.getFilterItem(filterValue);
		if(filterItem) {
			filter = filterItem.value;
		}
		this.filter = filter;
		this.element.querySelector(".cn-manager-filter").value = filter;
	}

	setKeywords(keywords = "") {
		this.keywords = keywords;
		this.element.querySelector(".cn-manager-keywords").value = keywords;
	}

	show(show_mode) {
		this.element.style.display = "flex";
		this.setFilter(show_mode);
		this.setKeywords("");
		this.showSelection("");
		this.showMessage("");
		this.loadData(show_mode);
	}

	close() {
		this.element.style.display = "none";
	}
}