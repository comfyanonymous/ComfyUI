import { api } from "../../scripts/api.js";
import { app } from "../../scripts/app.js";
import { $el, ComfyDialog } from "../../scripts/ui.js";
import {
	SUPPORTED_OUTPUT_NODE_TYPES,
	ShareDialog,
	ShareDialogChooser,
	getPotentialOutputsAndOutputNodes,
	showOpenArtShareDialog,
	showShareDialog,
	showYouMLShareDialog
} from "./comfyui-share-common.js";
import { OpenArtShareDialog } from "./comfyui-share-openart.js";
import { free_models, install_pip, install_via_git_url, manager_instance, rebootAPI, setManagerInstance, show_message } from "./common.js";
import { ComponentBuilderDialog, getPureName, load_components, set_component_policy } from "./components-manager.js";
import { CustomNodesManager } from "./custom-nodes-manager.js";
import { ModelManager } from "./model-manager.js";
import { set_double_click_policy } from "./node_fixer.js";
import { SnapshotManager } from "./snapshot.js";

var docStyle = document.createElement('style');
docStyle.innerHTML = `
.comfy-toast {
	position: fixed;
	bottom: 20px;
	left: 50%;
	transform: translateX(-50%);
	background-color: rgba(0, 0, 0, 0.7);
	color: white;
	padding: 10px 20px;
	border-radius: 5px;
	z-index: 1000;
	transition: opacity 0.5s;
}

.comfy-toast-fadeout {
	opacity: 0;
}

#cm-manager-dialog {
	width: 1000px;
	height: 520px;
	box-sizing: content-box;
	z-index: 10000;
	overflow-y: auto;
}

.cb-widget {
	width: 400px;
	height: 25px;
	box-sizing: border-box;
	z-index: 10000;
	margin-top: 10px;
	margin-bottom: 5px;
}

.cb-widget-input {
	width: 305px;
	height: 25px;
	box-sizing: border-box;
}
.cb-widget-input:disabled {
	background-color: #444444;
	color: white;
}

.cb-widget-input-label {
	width: 90px;
	height: 25px;
	box-sizing: border-box;
	color: white;
	text-align: right;
	display: inline-block;
	margin-right: 5px;
}

.cm-menu-container {
	column-gap: 20px;
	display: flex;
	flex-wrap: wrap;
	justify-content: center;
	box-sizing: content-box;
}

.cm-menu-column {
	display: flex;
	flex-direction: column;
	flex: 1 1 auto;
	width: 300px;
	box-sizing: content-box;
}

.cm-title {
	background-color: black;
	text-align: center;
	height: 40px;
	width: calc(100% - 10px);
	font-weight: bold;
	justify-content: center;
	align-content: center;
	vertical-align: middle;
}

#cm-channel-badge {
	color: white;
	background-color: #AA0000;
	width: 220px;
	height: 23px;
	font-size: 13px;
	border-radius: 5px;
	left: 5px;
	top: 5px;
	align-content: center;
	justify-content: center;
	text-align: center;
	font-weight: bold;
	float: left;
	vertical-align: middle;
	position: relative;
}

#custom-nodes-grid a {
	color: #5555FF;
	font-weight: bold;
	text-decoration: none;
}

#custom-nodes-grid a:hover {
	color: #7777FF;
	text-decoration: underline;
}

#external-models-grid a {
	color: #5555FF;
	font-weight: bold;
	text-decoration: none;
}

#external-models-grid a:hover {
	color: #7777FF;
	text-decoration: underline;
}

#alternatives-grid a {
	color: #5555FF;
	font-weight: bold;
	text-decoration: none;
}

#alternatives-grid a:hover {
	color: #7777FF;
	text-decoration: underline;
}

.cm-notice-board {
	width: 290px;
	height: 270px;
	overflow: auto;
	color: var(--input-text);
	border: 1px solid var(--descrip-text);
	padding: 5px 10px;
	overflow-x: hidden;
	box-sizing: content-box;
}

.cm-notice-board > ul {
	display: block;
	list-style-type: disc;
	margin-block-start: 1em;
	margin-block-end: 1em;
	margin-inline-start: 0px;
	margin-inline-end: 0px;
	padding-inline-start: 40px;
}

.cm-conflicted-nodes-text {
	background-color: #CCCC55 !important;
	color: #AA3333 !important;
	font-size: 10px;
	border-radius: 5px;
	padding: 10px;
}

.cm-warn-note {
	background-color: #101010 !important;
	color: #FF3800 !important;
	font-size: 13px;
	border-radius: 5px;
	padding: 10px;
	overflow-x: hidden;
	overflow: auto;
}

.cm-info-note {
	background-color: #101010 !important;
	color: #FF3800 !important;
	font-size: 13px;
	border-radius: 5px;
	padding: 10px;
	overflow-x: hidden;
	overflow: auto;
}
`;

document.head.appendChild(docStyle);

var update_comfyui_button = null;
var fetch_updates_button = null;
var update_all_button = null;
var badge_mode = "none";
let share_option = 'all';

// copied style from https://github.com/pythongosssss/ComfyUI-Custom-Scripts
const style = `
#workflowgallery-button {
	width: 310px;
	height: 27px;
	padding: 0px !important;
	position: relative;
	overflow: hidden;
	font-size: 17px !important;
}
#cm-nodeinfo-button {
	width: 310px;
	height: 27px;
	padding: 0px !important;
	position: relative;
	overflow: hidden;
	font-size: 17px !important;
}
#cm-manual-button {
	width: 310px;
	height: 27px;
	position: relative;
	overflow: hidden;
}

.cm-button {
	width: 310px;
	height: 30px;
	position: relative;
	overflow: hidden;
	font-size: 17px !important;
}

.cm-button-red {
	width: 310px;
	height: 30px;
	position: relative;
	overflow: hidden;
	font-size: 17px !important;
	background-color: #500000 !important;
	color: white !important;
}

.cm-experimental-button {
	width: 290px;
	height: 30px;
	position: relative;
	overflow: hidden;
	font-size: 17px !important;
}

.cm-experimental {
	width: 310px;
	border: 1px solid #555;
	border-radius: 5px;
	padding: 10px;
	align-items: center;
	text-align: center;
	justify-content: center;
	box-sizing: border-box;
}

.cm-experimental-legend {
	margin-top: -20px;
	margin-left: 50%;
	width:auto;
	height:20px;
	font-size: 13px;
	font-weight: bold;
	background-color: #990000;
	color: #CCFFFF;
	border-radius: 5px;
	text-align: center;
	transform: translateX(-50%);
	display: block;
}

.cm-menu-combo {
	cursor: pointer;
	width: 310px;
	box-sizing: border-box;
}

.cm-small-button {
	width: 120px;
	height: 30px;
	position: relative;
	overflow: hidden;
	box-sizing: border-box;
	font-size: 17px !important;
}

#cm-install-customnodes-button {
	width: 200px;
	height: 30px;
	position: relative;
	overflow: hidden;
	box-sizing: border-box;
	font-size: 17px !important;
}

.cm-search-filter {
	width: 200px;
	height: 30px !important;
	position: relative;
	overflow: hidden;
	box-sizing: border-box;
}

.cb-node-label {
	width: 400px;
	height:28px;
	color: black;
	background-color: #777777;
	font-size: 18px;
	text-align: center;
	font-weight: bold;
}

#cm-close-button {
	width: calc(100% - 65px);
	bottom: 10px;
	position: absolute;
	overflow: hidden;
}

#cm-save-button {
	width: calc(100% - 65px);
	bottom:40px;
	position: absolute;
	overflow: hidden;
}
#cm-save-button:disabled {
	background-color: #444444;
}

.pysssss-workflow-arrow-2 {
	position: absolute;
	top: 0;
	bottom: 0;
	right: 0;
	font-size: 12px;
	display: flex;
	align-items: center;
	width: 24px;
	justify-content: center;
	background: rgba(255,255,255,0.1);
	content: "▼";
}
.pysssss-workflow-arrow-2:after {
	content: "▼";
 }
 .pysssss-workflow-arrow-2:hover {
	filter: brightness(1.6);
	background-color: var(--comfy-menu-bg);
 }
.pysssss-workflow-popup-2 ~ .litecontextmenu {
	transform: scale(1.3);
}
#workflowgallery-button-menu {
	z-index: 10000000000 !important;
}
#cm-manual-button-menu {
	z-index: 10000000000 !important;
}
`;



async function init_badge_mode() {
	api.fetchApi('/manager/badge_mode')
		.then(response => response.text())
		.then(data => { badge_mode = data; })
}

async function init_share_option() {
	api.fetchApi('/manager/share_option')
		.then(response => response.text())
		.then(data => {
			share_option = data || 'all';
		});
}

async function init_notice(notice) {
	api.fetchApi('/manager/notice')
		.then(response => response.text())
		.then(data => {
			notice.innerHTML = data;
		})
}

await init_badge_mode();
await init_share_option();

async function fetchNicknames() {
	const response1 = await api.fetchApi(`/customnode/getmappings?mode=nickname`);
	const mappings = await response1.json();

	let result = {};
	let nickname_patterns = [];

	for (let i in mappings) {
		let item = mappings[i];
		var nickname;
		if (item[1].nickname) {
			nickname = item[1].nickname;
		}
		else if (item[1].title) {
			nickname = item[1].title;
		}
		else {
			nickname = item[1].title_aux;
		}

		for (let j in item[0]) {
			result[item[0][j]] = nickname;
		}

		if(item[1].nodename_pattern) {
			nickname_patterns.push([item[1].nodename_pattern, nickname]);
		}
	}

	return [result, nickname_patterns];
}

const [nicknames, nickname_patterns] = await fetchNicknames();

function getNickname(node, nodename) {
	if(node.nickname) {
		return node.nickname;
	}
	else {
		if (nicknames[nodename]) {
			node.nickname = nicknames[nodename];
		}
		else if(node.getInnerNodes) {
			let pure_name = getPureName(node);
			let groupNode = app.graph.extra?.groupNodes?.[pure_name];
			if(groupNode) {
				let packname = groupNode.packname;
				node.nickname = packname;
			}
			return node.nickname;
		}
		else {
			for(let i in nickname_patterns) {
				let item = nickname_patterns[i];
				if(nodename.match(item[0])) {
					node.nickname = item[1];
				}
			}
		}

		return node.nickname;
	}
}

function drawBadge(node, orig, restArgs) {
	let ctx = restArgs[0];
	const r = orig?.apply?.(node, restArgs);

	if (!node.flags.collapsed && badge_mode != 'none' && node.constructor.title_mode != LiteGraph.NO_TITLE) {
		let text = "";
		if (badge_mode.startsWith('id_nick'))
			text = `#${node.id} `;

		let nick = node.getNickname();
		if (nick) {
			if (nick == 'ComfyUI') {
				if(badge_mode.endsWith('hide')) {
					nick = "";
				}
				else {
					nick = "🦊"
				}
			}

			if (nick.length > 25) {
				text += nick.substring(0, 23) + "..";
			}
			else {
				text += nick;
			}
		}

		if (text != "") {
			let fgColor = "white";
			let bgColor = "#0F1F0F";
			let visible = true;

			ctx.save();
			ctx.font = "12px sans-serif";
			const sz = ctx.measureText(text);
			ctx.fillStyle = bgColor;
			ctx.beginPath();
			ctx.roundRect(node.size[0] - sz.width - 12, -LiteGraph.NODE_TITLE_HEIGHT - 20, sz.width + 12, 20, 5);
			ctx.fill();

			ctx.fillStyle = fgColor;
			ctx.fillText(text, node.size[0] - sz.width - 6, -LiteGraph.NODE_TITLE_HEIGHT - 6);
			ctx.restore();

			if (node.has_errors) {
				ctx.save();
				ctx.font = "bold 14px sans-serif";
				const sz2 = ctx.measureText(node.type);
				ctx.fillStyle = 'white';
				ctx.fillText(node.type, node.size[0] / 2 - sz2.width / 2, node.size[1] / 2);
				ctx.restore();
			}
		}
	}
	return r;
}


async function updateComfyUI() {
	let prev_text = update_comfyui_button.innerText;
	update_comfyui_button.innerText = "Updating ComfyUI...";
	update_comfyui_button.disabled = true;
	update_comfyui_button.style.backgroundColor = "gray";

	try {
		const response = await api.fetchApi('/comfyui_manager/update_comfyui');

		if (response.status == 400) {
			show_message('Failed to update ComfyUI.');
			return false;
		}

		if (response.status == 201) {
			show_message('ComfyUI has been successfully updated.');
		}
		else {
			show_message('ComfyUI is already up to date with the latest version.');
		}

		return true;
	}
	catch (exception) {
		show_message(`Failed to update ComfyUI / ${exception}`);
		return false;
	}
	finally {
		update_comfyui_button.disabled = false;
		update_comfyui_button.innerText = prev_text;
		update_comfyui_button.style.backgroundColor = "";
	}
}

async function fetchUpdates(update_check_checkbox) {
	let prev_text = fetch_updates_button.innerText;
	fetch_updates_button.innerText = "Fetching updates...";
	fetch_updates_button.disabled = true;
	fetch_updates_button.style.backgroundColor = "gray";

	try {
		var mode = manager_instance.datasrc_combo.value;

		const response = await api.fetchApi(`/customnode/fetch_updates?mode=${mode}`);

		if (response.status != 200 && response.status != 201) {
			show_message('Failed to fetch updates.');
			return false;
		}

		if (response.status == 201) {
			show_message("There is an updated extension available.<BR><BR><P><B>NOTE:<BR>Fetch Updates is not an update.<BR>Please update from <button id='cm-install-customnodes-button'>Install Custom Nodes</button> </B></P>");

			const button = document.getElementById('cm-install-customnodes-button');
			button.addEventListener("click",
				async function() {
					app.ui.dialog.close();

					if(!CustomNodesManager.instance) {
						CustomNodesManager.instance = new CustomNodesManager(app, self);
					}
					await CustomNodesManager.instance.show(CustomNodesManager.ShowMode.UPDATE);
				}
			);

			update_check_checkbox.checked = false;
		}
		else {
			show_message('All extensions are already up-to-date with the latest versions.');
		}

		return true;
	}
	catch (exception) {
		show_message(`Failed to update custom nodes / ${exception}`);
		return false;
	}
	finally {
		fetch_updates_button.disabled = false;
		fetch_updates_button.innerText = prev_text;
		fetch_updates_button.style.backgroundColor = "";
	}
}

async function updateAll(update_check_checkbox, manager_dialog) {
	let prev_text = update_all_button.innerText;
	update_all_button.innerText = "Updating all...(ComfyUI)";
	update_all_button.disabled = true;
	update_all_button.style.backgroundColor = "gray";

	try {
		var mode = manager_instance.datasrc_combo.value;

		update_all_button.innerText = "Updating all...";
		const response1 = await api.fetchApi('/comfyui_manager/update_comfyui');
		const response2 = await api.fetchApi(`/customnode/update_all?mode=${mode}`);

		if (response2.status == 403) {
			show_message('This action is not allowed with this security level configuration.');
			return false;
		}

		if (response1.status == 400 || response2.status == 400) {
			show_message('Failed to update ComfyUI or several extensions.<BR><BR>See terminal log.<BR>');
			return false;
		}

		if(response1.status == 201 || response2.status == 201) {
			const update_info = await response2.json();

			let failed_list = "";
			if(update_info.failed.length > 0) {
				failed_list = "<BR>FAILED: "+update_info.failed.join(", ");
			}

			let updated_list = "";
			if(update_info.updated.length > 0) {
				updated_list = "<BR>UPDATED: "+update_info.updated.join(", ");
			}

			show_message(
				"ComfyUI and all extensions have been updated to the latest version.<BR>To apply the updated custom node, please <button class='cm-small-button' id='cm-reboot-button5'>RESTART</button> ComfyUI. And refresh browser.<BR>"
				+failed_list
				+updated_list
				);

			const rebootButton = document.getElementById('cm-reboot-button5');
			rebootButton.addEventListener("click",
				function() {
					if(rebootAPI()) {
						manager_dialog.close();
					}
				});
		}
		else {
			show_message('ComfyUI and all extensions are already up-to-date with the latest versions.');
		}

		return true;
	}
	catch (exception) {
		show_message(`Failed to update ComfyUI or several extensions / ${exception}`);
		return false;
	}
	finally {
		update_all_button.disabled = false;
		update_all_button.innerText = prev_text;
		update_all_button.style.backgroundColor = "";
	}
}

function newDOMTokenList(initialTokens) {
	const tmp = document.createElement(`div`);

	const classList = tmp.classList;
	if (initialTokens) {
		initialTokens.forEach(token => {
		classList.add(token);
		});
	}

	return classList;
	}

/**
 * Check whether the node is a potential output node (img, gif or video output)
 */
const isOutputNode = (node) => {
	return SUPPORTED_OUTPUT_NODE_TYPES.includes(node.type);
}

// -----------
class ManagerMenuDialog extends ComfyDialog {
	createControlsMid() {
		let self = this;

		update_comfyui_button =
			$el("button.cm-button", {
				type: "button",
				textContent: "Update ComfyUI",
				onclick:
					() => updateComfyUI()
			});

		fetch_updates_button =
			$el("button.cm-button", {
				type: "button",
				textContent: "Fetch Updates",
				onclick:
					() => fetchUpdates(this.update_check_checkbox)
			});

		update_all_button =
			$el("button.cm-button", {
				type: "button",
				textContent: "Update All",
				onclick:
					() => updateAll(this.update_check_checkbox, self)
			});

		const res =
			[
				$el("button.cm-button", {
					type: "button",
					textContent: "Custom Nodes Manager",
					onclick:
						() => {
							if(!CustomNodesManager.instance) {
								CustomNodesManager.instance = new CustomNodesManager(app, self);
							}
							CustomNodesManager.instance.show(CustomNodesManager.ShowMode.NORMAL);
						}
				}),

				$el("button.cm-button", {
					type: "button",
					textContent: "Install Missing Custom Nodes",
					onclick:
						() => {
							if(!CustomNodesManager.instance) {
								CustomNodesManager.instance = new CustomNodesManager(app, self);
							}
							CustomNodesManager.instance.show(CustomNodesManager.ShowMode.MISSING);
						}
				}),

				
				$el("button.cm-button", {
					type: "button",
					textContent: "Model Manager",
					onclick:
						() => {
							if(!ModelManager.instance) {
								ModelManager.instance = new ModelManager(app, self);
							}
							ModelManager.instance.show();
						}
				}),

				$el("button.cm-button", {
					type: "button",
					textContent: "Install via Git URL",
					onclick: () => {
						var url = prompt("Please enter the URL of the Git repository to install", "");

						if (url !== null) {
							install_via_git_url(url, self);
						}
					}
				}),

				$el("br", {}, []),
				update_all_button,
				update_comfyui_button,
				fetch_updates_button,

				$el("br", {}, []),
				$el("button.cm-button", {
					type: "button",
					textContent: "Alternatives of A1111",
					onclick:
						() => {
							if(!CustomNodesManager.instance) {
								CustomNodesManager.instance = new CustomNodesManager(app, self);
							}
							CustomNodesManager.instance.show(CustomNodesManager.ShowMode.ALTERNATIVES);
						}
				}),

				$el("br", {}, []),
				$el("button.cm-button-red", {
					type: "button",
					textContent: "Restart",
					onclick: () => rebootAPI()
				}),
			];

		return res;
	}

	createControlsLeft() {
		let self = this;

		this.update_check_checkbox = $el("input",{type:'checkbox', id:"skip_update_check"},[])
		const uc_checkbox_text = $el("label",{for:"skip_update_check"},[" Skip update check"])
		uc_checkbox_text.style.color = "var(--fg-color)";
		uc_checkbox_text.style.cursor = "pointer";
		this.update_check_checkbox.checked = true;

		// db mode
		this.datasrc_combo = document.createElement("select");
		this.datasrc_combo.setAttribute("title", "Configure where to retrieve node/model information. If set to 'local,' the channel is ignored, and if set to 'channel (remote),' it fetches the latest information each time the list is opened.");
		this.datasrc_combo.className = "cm-menu-combo";
		this.datasrc_combo.appendChild($el('option', { value: 'cache', text: 'DB: Channel (1day cache)' }, []));
		this.datasrc_combo.appendChild($el('option', { value: 'local', text: 'DB: Local' }, []));
		this.datasrc_combo.appendChild($el('option', { value: 'url', text: 'DB: Channel (remote)' }, []));

		// preview method
		let preview_combo = document.createElement("select");
		preview_combo.setAttribute("title", "Configure how latent variables will be decoded during preview in the sampling process.");
		preview_combo.className = "cm-menu-combo";
		preview_combo.appendChild($el('option', { value: 'auto', text: 'Preview method: Auto' }, []));
		preview_combo.appendChild($el('option', { value: 'taesd', text: 'Preview method: TAESD (slow)' }, []));
		preview_combo.appendChild($el('option', { value: 'latent2rgb', text: 'Preview method: Latent2RGB (fast)' }, []));
		preview_combo.appendChild($el('option', { value: 'none', text: 'Preview method: None (very fast)' }, []));

		api.fetchApi('/manager/preview_method')
			.then(response => response.text())
			.then(data => { preview_combo.value = data; });

		preview_combo.addEventListener('change', function (event) {
			api.fetchApi(`/manager/preview_method?value=${event.target.value}`);
		});

		// nickname
		let badge_combo = document.createElement("select");
		badge_combo.setAttribute("title", "Configure the content to be displayed on the badge at the top right corner of the node. The ID is the identifier of the node. If 'hide built-in' is selected, both unknown nodes and built-in nodes will be omitted, making them indistinguishable");
		badge_combo.className = "cm-menu-combo";
		badge_combo.appendChild($el('option', { value: 'none', text: 'Badge: None' }, []));
		badge_combo.appendChild($el('option', { value: 'nick', text: 'Badge: Nickname' }, []));
		badge_combo.appendChild($el('option', { value: 'nick_hide', text: 'Badge: Nickname (hide built-in)' }, []));
		badge_combo.appendChild($el('option', { value: 'id_nick', text: 'Badge: #ID Nickname' }, []));
		badge_combo.appendChild($el('option', { value: 'id_nick_hide', text: 'Badge: #ID Nickname (hide built-in)' }, []));

		api.fetchApi('/manager/badge_mode')
			.then(response => response.text())
			.then(data => { badge_combo.value = data; badge_mode = data; });

		badge_combo.addEventListener('change', function (event) {
			api.fetchApi(`/manager/badge_mode?value=${event.target.value}`);
			badge_mode = event.target.value;
			app.graph.setDirtyCanvas(true);
		});

		// channel
		let channel_combo = document.createElement("select");
		channel_combo.setAttribute("title", "Configure the channel for retrieving data from the Custom Node list (including missing nodes) or the Model list. Note that the badge utilizes local information.");
		channel_combo.className = "cm-menu-combo";
		api.fetchApi('/manager/channel_url_list')
			.then(response => response.json())
			.then(async data => {
				try {
					let urls = data.list;
					for (let i in urls) {
						if (urls[i] != '') {
							let name_url = urls[i].split('::');
							channel_combo.appendChild($el('option', { value: name_url[0], text: `Channel: ${name_url[0]}` }, []));
						}
					}

					channel_combo.addEventListener('change', function (event) {
						api.fetchApi(`/manager/channel_url_list?value=${event.target.value}`);
					});

					channel_combo.value = data.selected;
				}
				catch (exception) {

				}
			});

		// default ui state
		let default_ui_combo = document.createElement("select");
		default_ui_combo.setAttribute("title", "Set the default state to be displayed in the main menu when the browser starts.");
		default_ui_combo.className = "cm-menu-combo";
		default_ui_combo.appendChild($el('option', { value: 'none', text: 'Default UI: None' }, []));
		default_ui_combo.appendChild($el('option', { value: 'history', text: 'Default UI: History' }, []));
		default_ui_combo.appendChild($el('option', { value: 'queue', text: 'Default UI: Queue' }, []));
		api.fetchApi('/manager/default_ui')
			.then(response => response.text())
			.then(data => { default_ui_combo.value = data; });

		default_ui_combo.addEventListener('change', function (event) {
			api.fetchApi(`/manager/default_ui?value=${event.target.value}`);
		});


		// share
		let share_combo = document.createElement("select");
		share_combo.setAttribute("title", "Hide the share button in the main menu or set the default action upon clicking it. Additionally, configure the default share site when sharing via the context menu's share button.");
		share_combo.className = "cm-menu-combo";
		const share_options = [
			['none', 'None'],
			['openart', 'OpenArt AI'],
			['youml', 'YouML'],
			['matrix', 'Matrix Server'],
			['comfyworkflows', 'ComfyWorkflows'],
			['copus', 'Copus'],
			['all', 'All'],
		];
		for (const option of share_options) {
			share_combo.appendChild($el('option', { value: option[0], text: `Share: ${option[1]}` }, []));
		}

		// default ui state
		let component_policy_combo = document.createElement("select");
		component_policy_combo.setAttribute("title", "When loading the workflow, configure which version of the component to use.");
		component_policy_combo.className = "cm-menu-combo";
		component_policy_combo.appendChild($el('option', { value: 'workflow', text: 'Component: Use workflow version' }, []));
		component_policy_combo.appendChild($el('option', { value: 'higher', text: 'Component: Use higher version' }, []));
		component_policy_combo.appendChild($el('option', { value: 'mine', text: 'Component: Use my version' }, []));
		api.fetchApi('/manager/component/policy')
			.then(response => response.text())
			.then(data => {
				component_policy_combo.value = data;
				set_component_policy(data);
			});

		component_policy_combo.addEventListener('change', function (event) {
			api.fetchApi(`/manager/component/policy?value=${event.target.value}`);
			set_component_policy(event.target.value);
		});

		let dbl_click_policy_combo = document.createElement("select");
		dbl_click_policy_combo.setAttribute("title", "Sets the behavior when you double-click the title area of a node.");
		dbl_click_policy_combo.className = "cm-menu-combo";
		dbl_click_policy_combo.appendChild($el('option', { value: 'none', text: 'Double-Click: None' }, []));
		dbl_click_policy_combo.appendChild($el('option', { value: 'copy-all', text: 'Double-Click: Copy All Connections' }, []));
		dbl_click_policy_combo.appendChild($el('option', { value: 'copy-input', text: 'Double-Click: Copy Input Connections' }, []));
		dbl_click_policy_combo.appendChild($el('option', { value: 'possible-input', text: 'Double-Click: Possible Input Connections' }, []));
		dbl_click_policy_combo.appendChild($el('option', { value: 'dual', text: 'Double-Click: Possible(left) + Copy(right)' }, []));

		api.fetchApi('/manager/dbl_click/policy')
			.then(response => response.text())
			.then(data => {
				dbl_click_policy_combo.value = data;
				set_double_click_policy(data);
			});

		dbl_click_policy_combo.addEventListener('change', function (event) {
			api.fetchApi(`/manager/dbl_click/policy?value=${event.target.value}`);
			set_double_click_policy(event.target.value);
		});

		api.fetchApi('/manager/share_option')
			.then(response => response.text())
			.then(data => {
				share_combo.value = data || 'all';
				share_option = data || 'all';
			});

		share_combo.addEventListener('change', function (event) {
			const value = event.target.value;
			share_option = value;
			api.fetchApi(`/manager/share_option?value=${value}`);
			const shareButton = document.getElementById("shareButton");
			if (value === 'none') {
				shareButton.style.display = "none";
			} else {
				shareButton.style.display = "inline-block";
			}
		});

		return [
			$el("div", {}, [this.update_check_checkbox, uc_checkbox_text]),
			$el("br", {}, []),
			this.datasrc_combo,
			channel_combo,
			preview_combo,
			badge_combo,
			default_ui_combo,
			share_combo,
			component_policy_combo,
			dbl_click_policy_combo,
			$el("br", {}, []),

			$el("br", {}, []),
			$el("filedset.cm-experimental", {}, [
					$el("legend.cm-experimental-legend", {}, ["EXPERIMENTAL"]),
					$el("button.cm-experimental-button", {
						type: "button",
						textContent: "Snapshot Manager",
						onclick:
							() => {
								if(!SnapshotManager.instance)
								SnapshotManager.instance = new SnapshotManager(app, self);
								SnapshotManager.instance.show();
							}
					}),
					$el("button.cm-experimental-button", {
						type: "button",
						textContent: "Install PIP packages",
						onclick:
							() => {
								var url = prompt("Please enumerate the pip packages to be installed.\n\nExample: insightface opencv-python-headless>=4.1.1\n", "");

								if (url !== null) {
									install_pip(url, self);
								}
							}
					}),
					$el("button.cm-experimental-button", {
						type: "button",
						textContent: "Unload models",
						onclick: () => { free_models(); }
					})
				]),
		];
	}

	createControlsRight() {
		const elts = [
				$el("button.cm-button", {
					id: 'cm-manual-button',
					type: "button",
					textContent: "Community Manual",
					onclick: () => { window.open("https://blenderneko.github.io/ComfyUI-docs/", "comfyui-community-manual"); }
				}, [
					$el("div.pysssss-workflow-arrow-2", {
						id: `cm-manual-button-arrow`,
						onclick: (e) => {
							e.preventDefault();
							e.stopPropagation();

							LiteGraph.closeAllContextMenus();
							const menu = new LiteGraph.ContextMenu(
								[
									{
										title: "Comfy Custom Node How To",
										callback: () => { window.open("https://github.com/chrisgoringe/Comfy-Custom-Node-How-To/wiki/aaa_index", "comfyui-community-manual1"); },
									},
									{
										title: "ComfyUI Guide To Making Custom Nodes",
										callback: () => { window.open("https://github.com/Suzie1/ComfyUI_Guide_To_Making_Custom_Nodes/wiki", "comfyui-community-manual2"); },
									},
									{
										title: "ComfyUI Examples",
										callback: () => { window.open("https://comfyanonymous.github.io/ComfyUI_examples", "comfyui-community-manual3"); },
									},
									{
										title: "Close",
										callback: () => {
											LiteGraph.closeAllContextMenus();
										},
									}
								],
								{
									event: e,
									scale: 1.3,
								},
								window
							);
							// set the id so that we can override the context menu's z-index to be above the comfyui manager menu
							menu.root.id = "cm-manual-button-menu";
							menu.root.classList.add("pysssss-workflow-popup-2");
						},
					})
				]),

				$el("button", {
					id: 'workflowgallery-button',
					type: "button",
					style: {
						...(localStorage.getItem("wg_last_visited") ? {height: '50px'} : {})
					},
					onclick: (e) => {
						const last_visited_site = localStorage.getItem("wg_last_visited")
						if (!!last_visited_site) {
							window.open(last_visited_site, last_visited_site);
						} else {
							this.handleWorkflowGalleryButtonClick(e)
						}
					},
				}, [
					$el("p", {
						textContent: 'Workflow Gallery',
						style: {
							'text-align': 'center',
							'color': 'white',
							'font-size': '18px',
							'margin': 0,
							'padding': 0,
						}
					}, [
						$el("p", {
							id: 'workflowgallery-button-last-visited-label',
							textContent: `(${localStorage.getItem("wg_last_visited") ? localStorage.getItem("wg_last_visited").split('/')[2] : ''})`,
							style: {
								'text-align': 'center',
								'color': 'white',
								'font-size': '12px',
								'margin': 0,
								'padding': 0,
							}
						})
					]),
					$el("div.pysssss-workflow-arrow-2", {
						id: `comfyworkflows-button-arrow`,
						onclick: this.handleWorkflowGalleryButtonClick
					})
				]),

				$el("button.cm-button", {
					id: 'cm-nodeinfo-button',
					type: "button",
					textContent: "Nodes Info",
					onclick: () => { window.open("https://ltdrdata.github.io/", "comfyui-node-info"); }
				}),
				$el("br", {}, []),
		];

		var textarea = document.createElement("div");
		textarea.className = "cm-notice-board";
		elts.push(textarea);

		init_notice(textarea);

		return elts;
	}

	constructor() {
		super();

		const close_button = $el("button", { id: "cm-close-button", type: "button", textContent: "Close", onclick: () => this.close() });

		const content =
				$el("div.comfy-modal-content",
					[
						$el("tr.cm-title", {}, [
								$el("font", {size:6, color:"white"}, [`ComfyUI Manager Menu`])]
							),
						$el("br", {}, []),
						$el("div.cm-menu-container",
							[
								$el("div.cm-menu-column", [...this.createControlsLeft()]),
								$el("div.cm-menu-column", [...this.createControlsMid()]),
								$el("div.cm-menu-column", [...this.createControlsRight()])
							]),

						$el("br", {}, []),
						close_button,
					]
				);

		content.style.width = '100%';
		content.style.height = '100%';

		this.element = $el("div.comfy-modal", { id:'cm-manager-dialog', parent: document.body }, [ content ]);
	}

	show() {
		this.element.style.display = "block";
	}

	handleWorkflowGalleryButtonClick(e) {
		e.preventDefault();
		e.stopPropagation();
		LiteGraph.closeAllContextMenus();

		// Modify the style of the button so that the UI can indicate the last
		// visited site right away.
		const modifyButtonStyle = (url) => {
			const workflowGalleryButton = document.getElementById('workflowgallery-button');
			workflowGalleryButton.style.height = '50px';
			const lastVisitedLabel = document.getElementById('workflowgallery-button-last-visited-label');
			lastVisitedLabel.textContent = `(${url.split('/')[2]})`;
		}

		const menu = new LiteGraph.ContextMenu(
			[
				{
					title: "Share your art",
					callback: () => {
						if (share_option === 'openart') {
							showOpenArtShareDialog();
							return;
						} else if (share_option === 'matrix' || share_option === 'comfyworkflows') {
							showShareDialog(share_option);
							return;
						} else if (share_option === 'youml') {
							showYouMLShareDialog();
							return;
						}

						if (!ShareDialogChooser.instance) {
							ShareDialogChooser.instance = new ShareDialogChooser();
						}
						ShareDialogChooser.instance.show();
					},
				},
				{
					title: "Open 'openart.ai'",
					callback: () => {
						const url = "https://openart.ai/workflows/dev";
						localStorage.setItem("wg_last_visited", url);
						window.open(url, url);
						modifyButtonStyle(url);
					},
				},
				{
					title: "Open 'youml.com'",
					callback: () => {
						const url = "https://youml.com/?from=comfyui-share";
						localStorage.setItem("wg_last_visited", url);
						window.open(url, url);
						modifyButtonStyle(url);
					},
				},
				{
					title: "Open 'comfyworkflows.com'",
					callback: () => {
						const url = "https://comfyworkflows.com/";
						localStorage.setItem("wg_last_visited", url);
						window.open(url, url);
						modifyButtonStyle(url);
					},
				},
				{
					title: "Open 'flowt.ai'",
					callback: () => {
						const url = "https://flowt.ai/";
						localStorage.setItem("wg_last_visited", url);
						window.open(url, url);
						modifyButtonStyle(url);
					},
				},
				{
					title: "Open 'esheep'",
					callback: () => {
						const url = "https://www.esheep.com";
						localStorage.setItem("wg_last_visited", url);
						window.open(url, url);
						modifyButtonStyle(url);
					},
				},
				{
					title: "Open 'Copus.io'",
					callback: () => {
						const url = "https://www.copus.io";
						localStorage.setItem("wg_last_visited", url);
						window.open(url, url);
						modifyButtonStyle(url);
					},
				},
				{
					title: "Close",
					callback: () => {
						LiteGraph.closeAllContextMenus();
					},
				}
			],
			{
				event: e,
				scale: 1.3,
			},
			window
		);
		// set the id so that we can override the context menu's z-index to be above the comfyui manager menu
		menu.root.id = "workflowgallery-button-menu";
		menu.root.classList.add("pysssss-workflow-popup-2");
	}
}


app.registerExtension({
	name: "Comfy.ManagerMenu",
	init() {
		$el("style", {
			textContent: style,
			parent: document.head,
		});
	},
	async setup() {
		let orig_clear = app.graph.clear;
		app.graph.clear = function () {
			orig_clear.call(app.graph);
			load_components();
		};

		load_components();

		const menu = document.querySelector(".comfy-menu");
		const separator = document.createElement("hr");

		separator.style.margin = "20px 0";
		separator.style.width = "100%";
		menu.append(separator);

		try {
			// new style Manager buttons

			// unload models button into new style Manager button
			let cmGroup = new (await import("../../scripts/ui/components/buttonGroup.js")).ComfyButtonGroup(
				new(await import("../../scripts/ui/components/button.js")).ComfyButton({
					icon: "puzzle",
					action: () => {
						if(!manager_instance)
							setManagerInstance(new ManagerMenuDialog());
						manager_instance.show();
					},
					tooltip: "ComfyUI Manager",
					content: "Manager",
					classList: "comfyui-button comfyui-menu-mobile-collapse primary"
				}).element,
				new(await import("../../scripts/ui/components/button.js")).ComfyButton({
					icon: "vacuum-outline",
					action: () => {
						free_models();
					},
					tooltip: "Unload Models"
				}).element,
				new(await import("../../scripts/ui/components/button.js")).ComfyButton({
					icon: "vacuum",
					action: () => {
						free_models(true);
					},
					tooltip: "Free model and node cache"
				}).element,
				new(await import("../../scripts/ui/components/button.js")).ComfyButton({
					icon: "share",
					action: () => {
						if (share_option === 'openart') {
							showOpenArtShareDialog();
							return;
						} else if (share_option === 'matrix' || share_option === 'comfyworkflows') {
							showShareDialog(share_option);
							return;
						} else if (share_option === 'youml') {
							showYouMLShareDialog();
							return;
						}

						if(!ShareDialogChooser.instance) {
							ShareDialogChooser.instance = new ShareDialogChooser();
						}
						ShareDialogChooser.instance.show();
					},
					tooltip: "Share"
				}).element
			);

			app.menu?.settingsGroup.element.before(cmGroup.element);
		}
		catch(exception) {
			console.log('ComfyUI is outdated. New style menu based features are disabled.');
		}

		// old style Manager button
		const managerButton = document.createElement("button");
		managerButton.textContent = "Manager";
		managerButton.onclick = () => {
				if(!manager_instance)
					setManagerInstance(new ManagerMenuDialog());
				manager_instance.show();
			}
		menu.append(managerButton);

		const shareButton = document.createElement("button");
		shareButton.id = "shareButton";
		shareButton.textContent = "Share";
		shareButton.onclick = () => {
			if (share_option === 'openart') {
				showOpenArtShareDialog();
				return;
			} else if (share_option === 'matrix' || share_option === 'comfyworkflows') {
				showShareDialog(share_option);
				return;
			} else if (share_option === 'youml') {
				showYouMLShareDialog();
				return;
			}

			if(!ShareDialogChooser.instance) {
				ShareDialogChooser.instance = new ShareDialogChooser();
			}
			ShareDialogChooser.instance.show();
		}
		// make the background color a gradient of blue to green
		shareButton.style.background = "linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%)";
		shareButton.style.color = "black";

		// Load share option from local storage to determine whether to show
		// the share button.
		const shouldShowShareButton = share_option !== 'none';
		shareButton.style.display = shouldShowShareButton ? "inline-block" : "none";

		menu.append(shareButton);
	},

	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		this._addExtraNodeContextMenu(nodeType, app);
	},

	async nodeCreated(node, app) {
		if(!node.badge_enabled) {
			node.getNickname = function () { return getNickname(node, node.comfyClass.trim()) };
			let orig = node.onDrawForeground;
			if(!orig)
				orig = node.__proto__.onDrawForeground;

			node.onDrawForeground = function (ctx) {
				drawBadge(node, orig, arguments)
			};
			node.badge_enabled = true;
		}
	},

	async loadedGraphNode(node, app) {
		if(!node.badge_enabled) {
			const orig = node.onDrawForeground;
			node.getNickname = function () { return getNickname(node, node.type.trim()) };
			node.onDrawForeground = function (ctx) { drawBadge(node, orig, arguments) };
		}
	},

	_addExtraNodeContextMenu(node, app) {
		const origGetExtraMenuOptions = node.prototype.getExtraMenuOptions;
		node.prototype.cm_menu_added = true;
		node.prototype.getExtraMenuOptions = function (_, options) {
			origGetExtraMenuOptions?.apply?.(this, arguments);

			if (node.category.startsWith('group nodes/')) {
				options.push({
					content: "Save As Component",
					callback: (obj) => {
						if (!ComponentBuilderDialog.instance) {
							ComponentBuilderDialog.instance = new ComponentBuilderDialog();
						}
						ComponentBuilderDialog.instance.target_node = node;
						ComponentBuilderDialog.instance.show();
					}
				}, null);
			}

			if (isOutputNode(node)) {
				const { potential_outputs } = getPotentialOutputsAndOutputNodes([this]);
				const hasOutput = potential_outputs.length > 0;

				// Check if the previous menu option is `null`. If it's not,
				// then we need to add a `null` as a separator.
				if (options[options.length - 1] !== null) {
					options.push(null);
				}

				options.push({
					content: "🏞️ Share Output",
					disabled: !hasOutput,
					callback: (obj) => {
						if (!ShareDialog.instance) {
							ShareDialog.instance = new ShareDialog();
						}
						const shareButton = document.getElementById("shareButton");
						if (shareButton) {
							const currentNode = this;
							if (!OpenArtShareDialog.instance) {
								OpenArtShareDialog.instance = new OpenArtShareDialog();
							}
							OpenArtShareDialog.instance.selectedNodeId = currentNode.id;
							if (!ShareDialog.instance) {
								ShareDialog.instance = new ShareDialog(share_option);
							}
							ShareDialog.instance.selectedNodeId = currentNode.id;
							shareButton.click();
						}
					}
				}, null);
			}
		}
	},
});


async function set_default_ui()
{
	let res = await api.fetchApi('/manager/default_ui');
	if(res.status == 200) {
		let mode = await res.text();
		switch(mode) {
		case 'history':
			app.ui.queue.hide();
			app.ui.history.show();
			break;
		case 'queue':
			app.ui.queue.show();
			app.ui.history.hide();
			break;
		default:
			// do nothing
			break;
		}
	}
}

set_default_ui();