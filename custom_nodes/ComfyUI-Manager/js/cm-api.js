import { api } from "../../scripts/api.js";
import { app } from "../../scripts/app.js";
import { sleep } from "./common.js";

async function tryInstallCustomNode(event) {
	let msg = '-= [ComfyUI Manager] extension installation request =-\n\n';
	msg += `The '${event.detail.sender}' extension requires the installation of the '${event.detail.title}' extension. `;

	if(event.detail.target.installed == 'Disabled') {
		msg += 'However, the extension is currently disabled. Would you like to enable it and reboot?'
	}
	else if(event.detail.target.installed == 'True') {
		msg += 'However, it seems that the extension is in an import-fail state or is not compatible with the current version. Please address this issue.';
	}
	else {
		msg += `Would you like to install it and reboot?`;
	}

	msg += `\n\nRequest message:\n${event.detail.msg}`;

	if(event.detail.target.installed == 'True') {
		alert(msg);
		return;
	}

	let res = confirm(msg);
	if(res) {
		if(event.detail.target.installed == 'Disabled') {
			const response = await api.fetchApi(`/customnode/toggle_active`, {
										method: 'POST',
										headers: { 'Content-Type': 'application/json' },
										body: JSON.stringify(event.detail.target)
									});
		}
		else {
			await sleep(300);
			app.ui.dialog.show(`Installing... '${event.detail.target.title}'`);

			const response = await api.fetchApi(`/customnode/install`, {
										method: 'POST',
										headers: { 'Content-Type': 'application/json' },
										body: JSON.stringify(event.detail.target)
									});

			if(response.status == 403) {
				show_message('This action is not allowed with this security level configuration.');
				return false;
			}
		}

		let response = await api.fetchApi("/manager/reboot");
		if(response.status == 403) {
			show_message('This action is not allowed with this security level configuration.');
			return false;
		}

		await sleep(300);

		app.ui.dialog.show(`Rebooting...`);
	}
}

api.addEventListener("cm-api-try-install-customnode", tryInstallCustomNode);
