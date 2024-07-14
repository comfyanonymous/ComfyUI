import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js"
import { sleep, show_message } from "./common.js";
import { GroupNodeConfig, GroupNodeHandler } from "../../extensions/core/groupNode.js";
import { ComfyDialog, $el } from "../../scripts/ui.js";

let pack_map = {};
let rpack_map = {};

export function getPureName(node) {
	// group nodes/
	let category = null;
	if(node.category) {
		category = node.category.substring(12);
	}
	else {
		category = node.constructor.category?.substring(12);
	}
	if(category) {
		let purename = node.comfyClass.substring(category.length+1);
		return purename;
	}
	else if(node.comfyClass.startsWith('workflow/')) {
		return node.comfyClass.substring(9);
	}
	else {
		return node.comfyClass;
	}
}

function isValidVersionString(version) {
	const versionPattern = /^(\d+)\.(\d+)(\.(\d+))?$/;
	
	const match = version.match(versionPattern);

	return match !== null &&
			parseInt(match[1], 10) >= 0 &&
			parseInt(match[2], 10) >= 0 &&
			(!match[3] || parseInt(match[4], 10) >= 0);
}

function register_pack_map(name, data) {
	if(data.packname) {
		pack_map[data.packname] = name;
		rpack_map[name] = data;
	}
	else {
		rpack_map[name] = data;
	}
}

function storeGroupNode(name, data, register=true) {
	let extra = app.graph.extra;
	if (!extra) app.graph.extra = extra = {};
	let groupNodes = extra.groupNodes;
	if (!groupNodes) extra.groupNodes = groupNodes = {};
	groupNodes[name] = data;

	if(register) {
		register_pack_map(name, data);
	}
}

export async function load_components() {
	let data = await api.fetchApi('/manager/component/loads', {method: "POST"});
	let components = await data.json();

	let start_time = Date.now();
	let failed = [];
	let failed2 = [];

	for(let name in components) {
		if(app.graph.extra?.groupNodes?.[name]) {
			if(data) {
				let data = components[name];

				let category = data.packname;
				if(data.category) {
					category += "/" + data.category;
				}
				if(category == '') {
					category = 'components';
				}

				const config = new GroupNodeConfig(name, data);
				await config.registerType(category);

				register_pack_map(name, data);
				continue;
			}
		}

		let nodeData = components[name];

		storeGroupNode(name, nodeData);

		const config = new GroupNodeConfig(name, nodeData);

		while(true) {
			try {
				let category = nodeData.packname;
				if(nodeData.category) {
					category += "/" + nodeData.category;
				}
				if(category == '') {
					category = 'components';
				}

				await config.registerType(category);
				register_pack_map(name, nodeData);
				break;
			}
			catch {
				let elapsed_time = Date.now() - start_time;
				if (elapsed_time > 5000) {
					failed.push(name);
					break;
				} else {
					await sleep(100);
				}
			}
		}
	}

	// fallback1
	for(let i in failed) {
		let name = failed[i];

		if(app.graph.extra?.groupNodes?.[name]) {
			continue;
		}

		let nodeData = components[name];

		storeGroupNode(name, nodeData);

		const config = new GroupNodeConfig(name, nodeData);
		while(true) {
			try {
				let category = nodeData.packname;
				if(nodeData.workflow.category) {
					category += "/" + nodeData.category;
				}
				if(category == '') {
					category = 'components';
				}

				await config.registerType(category);
				register_pack_map(name, nodeData);
				break;
			}
			catch {
				let elapsed_time = Date.now() - start_time;
				if (elapsed_time > 10000) {
					failed2.push(name);
					break;
				} else {
					await sleep(100);
				}
			}
		}
	}

	// fallback2
	for(let name in failed2) {
		let name = failed2[i];

		let nodeData = components[name];

		storeGroupNode(name, nodeData);

		const config = new GroupNodeConfig(name, nodeData);
		while(true) {
			try {
				let category = nodeData.workflow.packname;
				if(nodeData.workflow.category) {
					category += "/" + nodeData.category;
				}
				if(category == '') {
					category = 'components';
				}

				await config.registerType(category);
				register_pack_map(name, nodeData);
				break;
			}
			catch {
				let elapsed_time = Date.now() - start_time;
				if (elapsed_time > 30000) {
					failed.push(name);
					break;
				} else {
					await sleep(100);
				}
			}
		}
	}
}

async function save_as_component(node, version, author, prefix, nodename, packname, category) {
	let component_name = `${prefix}::${nodename}`;

	let subgraph = app.graph.extra?.groupNodes?.[component_name];
	if(!subgraph) {
		subgraph = app.graph.extra?.groupNodes?.[getPureName(node)];
	}

	subgraph.version = version;
	subgraph.author = author;
	subgraph.datetime = Date.now();
	subgraph.packname = packname;
	subgraph.category = category;

	let body =
		{
			name: component_name,
			workflow: subgraph
		};

	pack_map[packname] = component_name;
	rpack_map[component_name] = subgraph;

	const res = await api.fetchApi('/manager/component/save', {
			method: "POST",
			headers: {
				"Content-Type": "application/json",
			},
			body: JSON.stringify(body),
		});

	if(res.status == 200) {
		storeGroupNode(component_name, subgraph);
		const config = new GroupNodeConfig(component_name, subgraph);

		let category = body.workflow.packname;
		if(body.workflow.category) {
			category += "/" + body.workflow.category;
		}
		if(category == '') {
			category = 'components';
		}

		await config.registerType(category);

		let path = await res.text();
		show_message(`Component '${component_name}' is saved into:\n${path}`);
	}
	else
		show_message(`Failed to save component.`);
}

async function import_component(component_name, component, mode) {
	if(mode) {
		let body =
			{
				name: component_name,
				workflow: component
			};

		const res = await api.fetchApi('/manager/component/save', {
						method: "POST",
						headers: { "Content-Type": "application/json", },
						body: JSON.stringify(body)
					});
	}

	let category = component.packname;
	if(component.category) {
		category += "/" + component.category;
	}
	if(category == '') {
		category = 'components';
	}

	storeGroupNode(component_name, component);
	const config = new GroupNodeConfig(component_name, component);
	await config.registerType(category);
}

function restore_to_loaded_component(component_name) {
	if(rpack_map[component_name]) {
		let component = rpack_map[component_name];
		storeGroupNode(component_name, component, false);
		const config = new GroupNodeConfig(component_name, component);
		config.registerType(component.category);
	}
}

// Using a timestamp prevents duplicate pastes and ensures the prevention of re-deletion of litegrapheditor_clipboard.
let last_paste_timestamp = null;

function versionCompare(v1, v2) {
	let ver1;
	let ver2;
	if(v1 && v1 != '') {
		ver1 = v1.split('.');
		ver1[0] = parseInt(ver1[0]);
		ver1[1] = parseInt(ver1[1]);
		if(ver1.length == 2)
			ver1.push(0);
		else
			ver1[2] = parseInt(ver2[2]);
	}
	else {
		ver1 = [0,0,0];
	}

	if(v2 && v2 != '') {
		ver2 = v2.split('.');
		ver2[0] = parseInt(ver2[0]);
		ver2[1] = parseInt(ver2[1]);
		if(ver2.length == 2)
			ver2.push(0);
		else
			ver2[2] = parseInt(ver2[2]);
	}
	else {
		ver2 = [0,0,0];
	}

	if(ver1[0] > ver2[0])
		return -1;
	else if(ver1[0] < ver2[0])
		return 1;

	if(ver1[1] > ver2[1])
		return -1;
	else if(ver1[1] < ver2[1])
		return 1;

	if(ver1[2] > ver2[2])
		return -1;
	else if(ver1[2] < ver2[2])
		return 1;

	return 0;
}

function checkVersion(name, component) {
	let msg = '';
	if(rpack_map[name]) {
		let old_version = rpack_map[name].version;
		if(!old_version || old_version == '') {
			msg = `  '${name}' Upgrade (V0.0 -> V${component.version})`;
		}
		else {
			let c = versionCompare(old_version, component.version);
			if(c < 0) {
				msg = `  '${name}' Downgrade (V${old_version} -> V${component.version})`;
			}
			else if(c > 0) {
				msg = `  '${name}' Upgrade (V${old_version} -> V${component.version})`;
			}
			else {
				msg = `  '${name}' Same version (V${component.version})`;
			}
		}
	}
	else {
		msg = `'${name}' NEW (V${component.version})`;
	}

	return msg;
}

function handle_import_components(components) {
	let msg = 'Components:\n';
	let cnt = 0;
	for(let name in components) {
		let component = components[name];
		let v = checkVersion(name, component);

		if(cnt < 10) {
			msg += v + '\n';
		}
		else if (cnt == 10) {
			msg += '...\n';
		}
		else {
			// do nothing
		}

		cnt++;
	}

	let last_name = null;
	msg += '\nWill you load components?\n';
	if(confirm(msg)) {
		let mode = confirm('\nWill you save components?\n(cancel=load without save)');

		for(let name in components) {
			let component = components[name];
			import_component(name, component, mode);
			last_name = name;
		}

		if(mode) {
			show_message('Components are saved.');
		}
		else {
			show_message('Components are loaded.');
		}
	}

	if(cnt == 1 && last_name) {
		const node = LiteGraph.createNode(`workflow/${last_name}`);
		node.pos = [app.canvas.graph_mouse[0], app.canvas.graph_mouse[1]];
		app.canvas.graph.add(node, false);
	}
}

function handlePaste(e) {
	let data = (e.clipboardData || window.clipboardData);
	const items = data.items;
	for(const item of items) {
		if(item.kind == 'string' && item.type == 'text/plain') {
			data = data.getData("text/plain");
			try {
				let json_data = JSON.parse(data);
				if(json_data.kind == 'ComfyUI Components' && last_paste_timestamp != json_data.timestamp) {
					last_paste_timestamp = json_data.timestamp;
					handle_import_components(json_data.components);

					// disable paste node
					localStorage.removeItem("litegrapheditor_clipboard", null);
				}
				else {
					console.log('This components are already pasted: ignored');
				}
			}
			catch {
				// nothing to do
			}
		}
	}
}

document.addEventListener("paste", handlePaste);


export class ComponentBuilderDialog extends ComfyDialog {
	constructor() {
		super();
	}

	clear() {
		while (this.element.children.length) {
			this.element.removeChild(this.element.children[0]);
		}
	}

	show() {
		this.invalidateControl();

		this.element.style.display = "block";
		this.element.style.zIndex = 10001;
		this.element.style.width = "500px";
		this.element.style.height = "480px";
	}

	invalidateControl() {
		this.clear();

		let self = this;

		const close_button = $el("button", { id: "cm-close-button", type: "button", textContent: "Close", onclick: () => self.close() });
		this.save_button = $el("button",
					{ id: "cm-save-button", type: "button", textContent: "Save", onclick: () =>
							{
								save_as_component(self.target_node, self.version_string.value.trim(), self.author.value.trim(), self.node_prefix.value.trim(),
												  self.getNodeName(), self.getPackName(), self.category.value.trim());
							}
					});

		let default_nodename = getPureName(this.target_node).trim();

		let groupNode = app.graph.extra.groupNodes[default_nodename];
		let default_packname = groupNode.packname;
		if(!default_packname) {
			default_packname = '';
		}

		let default_category = groupNode.category;
		if(!default_category) {
			default_category = '';
		}

		this.default_ver = groupNode.version;
		if(!this.default_ver) {
			this.default_ver = '0.0';
		}

		let default_author = groupNode.author;
		if(!default_author) {
			default_author = '';
		}

		let delimiterIndex = default_nodename.indexOf('::');
		let default_prefix = "";
		if(delimiterIndex != -1) {
			default_prefix = default_nodename.substring(0, delimiterIndex);
			default_nodename = default_nodename.substring(delimiterIndex + 2);
		}

		if(!default_prefix) {
			this.save_button.disabled = true;
		}

		this.pack_list = this.createPackListCombo();

		let version_string = this.createLabeledInput('input version (e.g. 1.0)', '*Version : ',  this.default_ver);
		this.version_string = version_string[1];
		this.version_string.disabled = true;

		let author = this.createLabeledInput('input author (e.g. Dr.Lt.Data)', 'Author : ',  default_author);
		this.author = author[1];

		let node_prefix = this.createLabeledInput('input node prefix (e.g. mypack)', '*Prefix : ',  default_prefix);
		this.node_prefix = node_prefix[1];

		let manual_nodename = this.createLabeledInput('input node name (e.g. MAKE_BASIC_PIPE)', 'Nodename : ', default_nodename);
		this.manual_nodename = manual_nodename[1];

		let manual_packname = this.createLabeledInput('input pack name (e.g. mypack)', 'Packname : ',  default_packname);
		this.manual_packname = manual_packname[1];

		let category = this.createLabeledInput('input category (e.g. util/pipe)', 'Category : ',  default_category);
		this.category = category[1];

		this.node_label = this.createNodeLabel();

		let author_mode = this.createAuthorModeCheck();
		this.author_mode = author_mode[0];

		const content =
				$el("div.comfy-modal-content",
					[
						$el("tr.cm-title", {}, [
								$el("font", {size:6, color:"white"}, [`ComfyUI-Manager: Component Builder`])]
							),
						$el("br", {}, []),
						$el("div.cm-menu-container",
							[
								author_mode[0],
								author_mode[1],
								category[0],
								author[0],
								node_prefix[0],
								manual_nodename[0],
								manual_packname[0],
								version_string[0],
								this.pack_list,
								$el("br", {}, []),
								this.node_label
							]),

						$el("br", {}, []),
						this.save_button,
						close_button,
					]
				);

		content.style.width = '100%';
		content.style.height = '100%';

		this.element = $el("div.comfy-modal", { id:'cm-manager-dialog', parent: document.body }, [ content ]);
	}

	validateInput() {
		let msg = "";

		if(!isValidVersionString(this.version_string.value)) {
			msg += 'Invalid version string: '+event.value+"\n";
		}

		if(this.node_prefix.value.trim() == '') {
			msg += 'Node prefix cannot be empty\n';
		}

		if(this.manual_nodename.value.trim() == '') {
			msg += 'Node name cannot be empty\n';
		}

		if(msg != '') {
//			alert(msg);
		}

		this.save_button.disabled = msg != "";
	}

	getPackName() {
		if(this.pack_list.selectedIndex == 0) {
			return this.manual_packname.value.trim();
		}

		return this.pack_list.value.trim();
	}

	getNodeName() {
		if(this.manual_nodename.value.trim() != '') {
			return this.manual_nodename.value.trim();
		}

		return getPureName(this.target_node);
	}

	createAuthorModeCheck() {
		let check = $el("input",{type:'checkbox', id:"author-mode"},[])
		const check_label = $el("label",{for:"author-mode"},["Enable author mode"]);
		check_label.style.color = "var(--fg-color)";
		check_label.style.cursor = "pointer";
		check.checked = false;

		let self = this;
		check.onchange = () => {
			self.version_string.disabled = !check.checked;

			if(!check.checked) {
				self.version_string.value = self.default_ver;
			}
			else {
				alert('If you are not the author, it is not recommended to change the version, as it may cause component update issues.');
			}
		};

		return [check, check_label];
	}

	createNodeLabel() {
		let label = $el('p');
		label.className = 'cb-node-label';
		if(this.target_node.comfyClass.includes('::'))
			label.textContent = getPureName(this.target_node);
		else
			label.textContent = " _::" + getPureName(this.target_node);
		return label;
	}

	createLabeledInput(placeholder, label, value) {
		let textbox = $el('input.cb-widget-input', {type:'text', placeholder:placeholder, value:value}, []);

		let self = this;
		textbox.onchange = () => {
			this.validateInput.call(self);
			this.node_label.textContent = this.node_prefix.value + "::" + this.manual_nodename.value;
		}
		let row = $el('span.cb-widget', {}, [ $el('span.cb-widget-input-label', label), textbox]);

		return [row, textbox];
	}

	createPackListCombo() {
		let combo = document.createElement("select");
		combo.className = "cb-widget";
		let default_packname_option = { value: '##manual', text: 'Packname: Manual' };

		combo.appendChild($el('option', default_packname_option, []));
		for(let name in pack_map) {
			combo.appendChild($el('option', { value: name, text: 'Packname: '+ name }, []));
		}

		let self = this;
		combo.onchange = function () {
			if(combo.selectedIndex == 0) {
				self.manual_packname.disabled = false;
			}
			else {
				self.manual_packname.disabled = true;
			}
		};

		return combo;
	}
}

let orig_handleFile = app.handleFile;

function handleFile(file) {
	if (file.name?.endsWith(".json") || file.name?.endsWith(".pack")) {
		const reader = new FileReader();
		reader.onload = async () => {
			let is_component = false;
			const jsonContent = JSON.parse(reader.result);
			for(let name in jsonContent) {
				let cand = jsonContent[name];
				is_component = cand.datetime && cand.version;
				break;
			}

			if(is_component) {
				handle_import_components(jsonContent);
			}
			else {
				orig_handleFile.call(app, file);
			}
		};
		reader.readAsText(file);

		return;
	}

	orig_handleFile.call(app, file);
}

app.handleFile = handleFile;

let current_component_policy = 'workflow';
try {
	api.fetchApi('/manager/component/policy')
		.then(response => response.text())
		.then(data => { current_component_policy = data; });
}
catch {}

function getChangedVersion(groupNodes) {
	if(!Object.keys(pack_map).length || !groupNodes)
		return null;

	let res = {};
	for(let component_name in groupNodes) {
		let data = groupNodes[component_name];

		if(rpack_map[component_name]) {
			let v = versionCompare(data.version, rpack_map[component_name].version);
			res[component_name] = v;
		}
	}

	return res;
}

const loadGraphData = app.loadGraphData;
app.loadGraphData = async function () {
	if(arguments.length == 0)
		return await loadGraphData.apply(this, arguments);

	let graphData = arguments[0];
	let groupNodes = graphData.extra?.groupNodes;
	let res = getChangedVersion(groupNodes);

	if(res) {
		let target_components = null;
		switch(current_component_policy) {
		case 'higher':
			target_components = Object.keys(res).filter(key => res[key] == 1);
			break;

		case 'mine':
			target_components = Object.keys(res);
			break;

		default:
			// do nothing
		}

		if(target_components) {
			for(let i in target_components) {
				let component_name = target_components[i];
				let component = rpack_map[component_name];
				if(component && graphData.extra?.groupNodes) {
					graphData.extra.groupNodes[component_name] = component;
				}
			}
		}
	}
	else {
		console.log('Empty components: policy ignored');
	}

	arguments[0] = graphData;
	return await loadGraphData.apply(this, arguments);
};

export function set_component_policy(v) {
	current_component_policy = v;
}

let graphToPrompt = app.graphToPrompt;
app.graphToPrompt = async function () {
	let p = await graphToPrompt.call(app);
	try {
		let groupNodes = p.workflow.extra?.groupNodes;
		if(groupNodes) {
			p.workflow.extra = { ... p.workflow.extra};

			// get used group nodes
			let used_group_nodes = new Set();
			for(let node of p.workflow.nodes) {
				if(node.type.startsWith('workflow/')) {
					used_group_nodes.add(node.type.substring(9));
				}
			}

			// remove unused group nodes
			let new_groupNodes = {};
			for (let key in p.workflow.extra.groupNodes) {
				if (used_group_nodes.has(key)) {
					new_groupNodes[key] = p.workflow.extra.groupNodes[key];
				}
			}
			p.workflow.extra.groupNodes = new_groupNodes;
		}
	}
	catch(e) {
		console.log(`Failed to filtering group nodes: ${e}`);
	}

	return p;
}
