import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js"
import { ComfyDialog, $el } from "../../scripts/ui.js";
import { manager_instance, rebootAPI, show_message } from  "./common.js";


async function restore_snapshot(target) {
	if(SnapshotManager.instance) {
		try {
			const response = await api.fetchApi(`/snapshot/restore?target=${target}`, { cache: "no-store" });

			if(response.status == 403) {
				show_message('This action is not allowed with this security level configuration.');
				return false;
			}

			if(response.status == 400) {
				show_message(`Restore snapshot failed: ${target.title} / ${exception}`);
			}

			app.ui.dialog.close();
			return true;
		}
		catch(exception) {
			show_message(`Restore snapshot failed: ${target.title} / ${exception}`);
			return false;
		}
		finally {
			await SnapshotManager.instance.invalidateControl();
			SnapshotManager.instance.updateMessage("<BR>To apply the snapshot, please <button id='cm-reboot-button2' class='cm-small-button'>RESTART</button> ComfyUI. And refresh browser.", 'cm-reboot-button2');
		}
	}
}

async function remove_snapshot(target) {
	if(SnapshotManager.instance) {
		try {
			const response = await api.fetchApi(`/snapshot/remove?target=${target}`, { cache: "no-store" });

			if(response.status == 403) {
				show_message('This action is not allowed with this security level configuration.');
				return false;
			}

			if(response.status == 400) {
				show_message(`Remove snapshot failed: ${target.title} / ${exception}`);
			}

			app.ui.dialog.close();
			return true;
		}
		catch(exception) {
			show_message(`Restore snapshot failed: ${target.title} / ${exception}`);
			return false;
		}
		finally {
			await SnapshotManager.instance.invalidateControl();
		}
	}
}

async function save_current_snapshot() {
	try {
		const response = await api.fetchApi('/snapshot/save', { cache: "no-store" });
		app.ui.dialog.close();
		return true;
	}
	catch(exception) {
		show_message(`Backup snapshot failed: ${exception}`);
		return false;
	}
	finally {
		await SnapshotManager.instance.invalidateControl();
		SnapshotManager.instance.updateMessage("<BR>Current snapshot saved.");
	}
}

async function getSnapshotList() {
	const response = await api.fetchApi(`/snapshot/getlist`);
	const data = await response.json();
	return data;
}

export class SnapshotManager extends ComfyDialog {
	static instance = null;

	restore_buttons = [];
	message_box = null;
	data = null;

	clear() {
		this.restore_buttons = [];
		this.message_box = null;
		this.data = null;
	}

	constructor(app, manager_dialog) {
		super();
		this.manager_dialog = manager_dialog;
		this.search_keyword = '';
		this.element = $el("div.comfy-modal", { parent: document.body }, []);
	}

	async remove_item() {
		caller.disableButtons();

		await caller.invalidateControl();
	}

	createControls() {
		return [
			$el("button.cm-small-button", {
				type: "button",
				textContent: "Close",
				onclick: () => { this.close(); }
				})
		];
	}

	startRestore(target) {
		const self = SnapshotManager.instance;

		self.updateMessage(`<BR><font color="green">Restore snapshot '${target.name}'</font>`);

		for(let i in self.restore_buttons) {
			self.restore_buttons[i].disabled = true;
			self.restore_buttons[i].style.backgroundColor = 'gray';
		}
	}

	async invalidateControl() {
		this.clear();
		this.data = (await getSnapshotList()).items;

		while (this.element.children.length) {
			this.element.removeChild(this.element.children[0]);
		}

		await this.createGrid();
		await this.createBottomControls();
	}

	updateMessage(msg, btn_id) {
		this.message_box.innerHTML = msg;
		if(btn_id) {
			const rebootButton = document.getElementById(btn_id);
			const self = this;
			rebootButton.onclick = function() {
				if(rebootAPI()) {
					self.close();
					self.manager_dialog.close();
				}
			};
		}
	}

	async createGrid(models_json) {
		var grid = document.createElement('table');
		grid.setAttribute('id', 'snapshot-list-grid');

		var thead = document.createElement('thead');
		var tbody = document.createElement('tbody');

		var headerRow = document.createElement('tr');
		thead.style.position = "sticky";
		thead.style.top = "0px";
		thead.style.borderCollapse = "collapse";
		thead.style.tableLayout = "fixed";

		var header1 = document.createElement('th');
		header1.innerHTML = '&nbsp;&nbsp;ID&nbsp;&nbsp;';
		header1.style.width = "20px";
		var header2 = document.createElement('th');
		header2.innerHTML = 'Datetime';
		header2.style.width = "100%";
		var header_button = document.createElement('th');
		header_button.innerHTML = 'Action';
		header_button.style.width = "100px";

		thead.appendChild(headerRow);
		headerRow.appendChild(header1);
		headerRow.appendChild(header2);
		headerRow.appendChild(header_button);

		headerRow.style.backgroundColor = "Black";
		headerRow.style.color = "White";
		headerRow.style.textAlign = "center";
		headerRow.style.width = "100%";
		headerRow.style.padding = "0";

		grid.appendChild(thead);
		grid.appendChild(tbody);

		this.grid_rows = {};

		if(this.data)
			for (var i = 0; i < this.data.length; i++) {
				const data = this.data[i];
				var dataRow = document.createElement('tr');
				var data1 = document.createElement('td');
				data1.style.textAlign = "center";
				data1.innerHTML = i+1;
				var data2 = document.createElement('td');
				data2.innerHTML = `&nbsp;${data}`;
				var data_button = document.createElement('td');
				data_button.style.textAlign = "center";

				var restoreBtn = document.createElement('button');
				restoreBtn.innerHTML = 'Restore';
				restoreBtn.style.width = "100px";
				restoreBtn.style.backgroundColor = 'blue';

				restoreBtn.addEventListener('click', function() {
					restore_snapshot(data);
				});

				var removeBtn = document.createElement('button');
				removeBtn.innerHTML = 'Remove';
				removeBtn.style.width = "100px";
				removeBtn.style.backgroundColor = 'red';

				removeBtn.addEventListener('click', function() {
					remove_snapshot(data);
				});

				data_button.appendChild(restoreBtn);
				data_button.appendChild(removeBtn);

				dataRow.style.backgroundColor = "var(--bg-color)";
				dataRow.style.color = "var(--fg-color)";
				dataRow.style.textAlign = "left";

				dataRow.appendChild(data1);
				dataRow.appendChild(data2);
				dataRow.appendChild(data_button);
				tbody.appendChild(dataRow);

				this.grid_rows[i] = {data:data, control:dataRow};
			}

		let self = this;
		const panel = document.createElement('div');
		panel.style.width = "100%";
		panel.appendChild(grid);

		function handleResize() {
		  const parentHeight = self.element.clientHeight;
		  const gridHeight = parentHeight - 200;

		  grid.style.height = gridHeight + "px";
		}
		window.addEventListener("resize", handleResize);

		grid.style.position = "relative";
		grid.style.display = "inline-block";
		grid.style.width = "100%";
		grid.style.height = "100%";
		grid.style.overflowY = "scroll";
		this.element.style.height = "85%";
		this.element.style.width = "80%";
		this.element.appendChild(panel);

		handleResize();
	}

	async createBottomControls() {
		var close_button = document.createElement("button");
		close_button.className = "cm-small-button";
		close_button.innerHTML = "Close";
		close_button.onclick = () => { this.close(); }
		close_button.style.display = "inline-block";

		var save_button = document.createElement("button");
		save_button.className = "cm-small-button";
		save_button.innerHTML = "Save snapshot";
		save_button.onclick = () => { save_current_snapshot(); }
		save_button.style.display = "inline-block";
		save_button.style.horizontalAlign = "right";
		save_button.style.width = "170px";

		this.message_box = $el('div', {id:'custom-download-message'}, [$el('br'), '']);
		this.message_box.style.height = '60px';
		this.message_box.style.verticalAlign = 'middle';

		this.element.appendChild(this.message_box);
		this.element.appendChild(close_button);
		this.element.appendChild(save_button);
	}

	async show() {
		try {
			this.invalidateControl();
			this.element.style.display = "block";
			this.element.style.zIndex = 10001;
		}
		catch(exception) {
			app.ui.dialog.show(`Failed to get external model list. / ${exception}`);
		}
	}
}
