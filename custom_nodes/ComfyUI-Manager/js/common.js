import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { $el, ComfyDialog } from "../../scripts/ui.js";

export function show_message(msg) {
	app.ui.dialog.show(msg);
	app.ui.dialog.element.style.zIndex = 10010;
}

export async function sleep(ms) {
	return new Promise(resolve => setTimeout(resolve, ms));
}

export function rebootAPI() {
	if (confirm("Are you sure you'd like to reboot the server?")) {
		try {
			api.fetchApi("/manager/reboot");
		}
		catch(exception) {

		}
		return true;
	}

	return false;
}

export var manager_instance = null;

export function setManagerInstance(obj) {
	manager_instance = obj;
}

export function showToast(message, duration = 3000) {
	const toast = $el("div.comfy-toast", {textContent: message});
	document.body.appendChild(toast);
	setTimeout(() => {
		toast.classList.add("comfy-toast-fadeout");
		setTimeout(() => toast.remove(), 500);
	}, duration);
}

function isValidURL(url) {
	if(url.includes('&'))
		return false;

	const http_pattern = /^(https?|ftp):\/\/[^\s$?#]+$/;
	const ssh_pattern = /^(.+@|ssh:\/\/).+:.+$/;
	return http_pattern.test(url) || ssh_pattern.test(url);
}

export async function install_pip(packages) {
	if(packages.includes('&'))
		app.ui.dialog.show(`Invalid PIP package enumeration: '${packages}'`);

	const res = await api.fetchApi("/customnode/install/pip", {
		method: "POST",
		body: packages,
	});

	if(res.status == 403) {
		show_message('This action is not allowed with this security level configuration.');
		return;
	}

	if(res.status == 200) {
		show_message(`PIP package installation is processed.<br>To apply the pip packages, please click the <button id='cm-reboot-button3'><font size='3px'>RESTART</font></button> button in ComfyUI.`);

		const rebootButton = document.getElementById('cm-reboot-button3');
		const self = this;

		rebootButton.addEventListener("click", rebootAPI);
	}
	else {
		show_message(`Failed to install '${packages}'<BR>See terminal log.`);
	}
}

export async function install_via_git_url(url, manager_dialog) {
	if(!url) {
		return;
	}

	if(!isValidURL(url)) {
		show_message(`Invalid Git url '${url}'`);
		return;
	}

	show_message(`Wait...<BR><BR>Installing '${url}'`);

	const res = await api.fetchApi("/customnode/install/git_url", {
		method: "POST",
		body: url,
	});

	if(res.status == 403) {
		show_message('This action is not allowed with this security level configuration.');
		return;
	}

	if(res.status == 200) {
		show_message(`'${url}' is installed<BR>To apply the installed custom node, please <button id='cm-reboot-button4'><font size='3px'>RESTART</font></button> ComfyUI.`);

		const rebootButton = document.getElementById('cm-reboot-button4');
		const self = this;

		rebootButton.addEventListener("click",
			function() {
				if(rebootAPI()) {
					manager_dialog.close();
				}
			});
	}
	else {
		show_message(`Failed to install '${url}'<BR>See terminal log.`);
	}
}

export async function free_models(free_execution_cache) {
	try {
		let mode = "";
		if(free_execution_cache) {
			mode = '{"unload_models": true, "free_memory": true}';
		}
		else {
			mode = '{"unload_models": true}';
		}

		let res = await api.fetchApi(`/free`, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: mode
		});

		if (res.status == 200) {
			if(free_execution_cache) {
				showToast("'Models' and 'Execution Cache' have been cleared.", 3000);
			}
			else {
				showToast("Models' have been unloaded.", 3000);
			}
		} else {
			showToast('Unloading of models failed. Installed ComfyUI may be an outdated version.', 5000);
		}
	} catch (error) {
		showToast('An error occurred while trying to unload models.', 5000);
	}
}

export function md5(inputString) {
	const hc = '0123456789abcdef';
	const rh = n => {let j,s='';for(j=0;j<=3;j++) s+=hc.charAt((n>>(j*8+4))&0x0F)+hc.charAt((n>>(j*8))&0x0F);return s;}
	const ad = (x,y) => {let l=(x&0xFFFF)+(y&0xFFFF);let m=(x>>16)+(y>>16)+(l>>16);return (m<<16)|(l&0xFFFF);}
	const rl = (n,c) => (n<<c)|(n>>>(32-c));
	const cm = (q,a,b,x,s,t) => ad(rl(ad(ad(a,q),ad(x,t)),s),b);
	const ff = (a,b,c,d,x,s,t) => cm((b&c)|((~b)&d),a,b,x,s,t);
	const gg = (a,b,c,d,x,s,t) => cm((b&d)|(c&(~d)),a,b,x,s,t);
	const hh = (a,b,c,d,x,s,t) => cm(b^c^d,a,b,x,s,t);
	const ii = (a,b,c,d,x,s,t) => cm(c^(b|(~d)),a,b,x,s,t);
	const sb = x => {
	let i;const nblk=((x.length+8)>>6)+1;const blks=[];for(i=0;i<nblk*16;i++) { blks[i]=0 };
	for(i=0;i<x.length;i++) {blks[i>>2]|=x.charCodeAt(i)<<((i%4)*8);}
		blks[i>>2]|=0x80<<((i%4)*8);blks[nblk*16-2]=x.length*8;return blks;
	}
	let i,x=sb(inputString),a=1732584193,b=-271733879,c=-1732584194,d=271733878,olda,oldb,oldc,oldd;
	for(i=0;i<x.length;i+=16) {olda=a;oldb=b;oldc=c;oldd=d;
		a=ff(a,b,c,d,x[i+ 0], 7, -680876936);d=ff(d,a,b,c,x[i+ 1],12, -389564586);c=ff(c,d,a,b,x[i+ 2],17,  606105819);
		b=ff(b,c,d,a,x[i+ 3],22,-1044525330);a=ff(a,b,c,d,x[i+ 4], 7, -176418897);d=ff(d,a,b,c,x[i+ 5],12, 1200080426);
		c=ff(c,d,a,b,x[i+ 6],17,-1473231341);b=ff(b,c,d,a,x[i+ 7],22,  -45705983);a=ff(a,b,c,d,x[i+ 8], 7, 1770035416);
		d=ff(d,a,b,c,x[i+ 9],12,-1958414417);c=ff(c,d,a,b,x[i+10],17,     -42063);b=ff(b,c,d,a,x[i+11],22,-1990404162);
		a=ff(a,b,c,d,x[i+12], 7, 1804603682);d=ff(d,a,b,c,x[i+13],12,  -40341101);c=ff(c,d,a,b,x[i+14],17,-1502002290);
		b=ff(b,c,d,a,x[i+15],22, 1236535329);a=gg(a,b,c,d,x[i+ 1], 5, -165796510);d=gg(d,a,b,c,x[i+ 6], 9,-1069501632);
		c=gg(c,d,a,b,x[i+11],14,  643717713);b=gg(b,c,d,a,x[i+ 0],20, -373897302);a=gg(a,b,c,d,x[i+ 5], 5, -701558691);
		d=gg(d,a,b,c,x[i+10], 9,   38016083);c=gg(c,d,a,b,x[i+15],14, -660478335);b=gg(b,c,d,a,x[i+ 4],20, -405537848);
		a=gg(a,b,c,d,x[i+ 9], 5,  568446438);d=gg(d,a,b,c,x[i+14], 9,-1019803690);c=gg(c,d,a,b,x[i+ 3],14, -187363961);
		b=gg(b,c,d,a,x[i+ 8],20, 1163531501);a=gg(a,b,c,d,x[i+13], 5,-1444681467);d=gg(d,a,b,c,x[i+ 2], 9,  -51403784);
		c=gg(c,d,a,b,x[i+ 7],14, 1735328473);b=gg(b,c,d,a,x[i+12],20,-1926607734);a=hh(a,b,c,d,x[i+ 5], 4,    -378558);
		d=hh(d,a,b,c,x[i+ 8],11,-2022574463);c=hh(c,d,a,b,x[i+11],16, 1839030562);b=hh(b,c,d,a,x[i+14],23,  -35309556);
		a=hh(a,b,c,d,x[i+ 1], 4,-1530992060);d=hh(d,a,b,c,x[i+ 4],11, 1272893353);c=hh(c,d,a,b,x[i+ 7],16, -155497632);
		b=hh(b,c,d,a,x[i+10],23,-1094730640);a=hh(a,b,c,d,x[i+13], 4,  681279174);d=hh(d,a,b,c,x[i+ 0],11, -358537222);
		c=hh(c,d,a,b,x[i+ 3],16, -722521979);b=hh(b,c,d,a,x[i+ 6],23,   76029189);a=hh(a,b,c,d,x[i+ 9], 4, -640364487);
		d=hh(d,a,b,c,x[i+12],11, -421815835);c=hh(c,d,a,b,x[i+15],16,  530742520);b=hh(b,c,d,a,x[i+ 2],23, -995338651);
		a=ii(a,b,c,d,x[i+ 0], 6, -198630844);d=ii(d,a,b,c,x[i+ 7],10, 1126891415);c=ii(c,d,a,b,x[i+14],15,-1416354905);
		b=ii(b,c,d,a,x[i+ 5],21,  -57434055);a=ii(a,b,c,d,x[i+12], 6, 1700485571);d=ii(d,a,b,c,x[i+ 3],10,-1894986606);
		c=ii(c,d,a,b,x[i+10],15,   -1051523);b=ii(b,c,d,a,x[i+ 1],21,-2054922799);a=ii(a,b,c,d,x[i+ 8], 6, 1873313359);
		d=ii(d,a,b,c,x[i+15],10,  -30611744);c=ii(c,d,a,b,x[i+ 6],15,-1560198380);b=ii(b,c,d,a,x[i+13],21, 1309151649);
		a=ii(a,b,c,d,x[i+ 4], 6, -145523070);d=ii(d,a,b,c,x[i+11],10,-1120210379);c=ii(c,d,a,b,x[i+ 2],15,  718787259);
		b=ii(b,c,d,a,x[i+ 9],21, -343485551);a=ad(a,olda);b=ad(b,oldb);c=ad(c,oldc);d=ad(d,oldd);
	}
	return rh(a)+rh(b)+rh(c)+rh(d);
}

export async function fetchData(route, options) {
	let err;
	const res = await api.fetchApi(route, options).catch(e => {
		err = e;
	});

	if (!res) {
		return {
			status: 400,
			error: new Error("Unknown Error")
		}
	}

	const { status, statusText } = res;
	if (err) {
		return {
			status,
			error: err
		}
	}

	if (status !== 200) {
		return {
			status,
			error: new Error(statusText || "Unknown Error")
		}
	}

	const data = await res.json();
	if (!data) {
		return {
			status,
			error: new Error(`Failed to load data: ${route}`)
		}
	}
	return {
		status,
		data
	}
}

export const icons = {
	search: '<svg viewBox="0 0 24 24" width="100%" height="100%" pointer-events="none" xmlns="http://www.w3.org/2000/svg"><path fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="m21 21-4.486-4.494M19 10.5a8.5 8.5 0 1 1-17 0 8.5 8.5 0 0 1 17 0"/></svg>',
	extensions: '<svg viewBox="64 64 896 896" width="100%" height="100%" pointer-events="none" xmlns="http://www.w3.org/2000/svg"><path fill="currentColor" d="M843.5 737.4c-12.4-75.2-79.2-129.1-155.3-125.4S550.9 676 546 752c-153.5-4.8-208-40.7-199.1-113.7 3.3-27.3 19.8-41.9 50.1-49 18.4-4.3 38.8-4.9 57.3-3.2 1.7.2 3.5.3 5.2.5 11.3 2.7 22.8 5 34.3 6.8 34.1 5.6 68.8 8.4 101.8 6.6 92.8-5 156-45.9 159.2-132.7 3.1-84.1-54.7-143.7-147.9-183.6-29.9-12.8-61.6-22.7-93.3-30.2-14.3-3.4-26.3-5.7-35.2-7.2-7.9-75.9-71.5-133.8-147.8-134.4S189.7 168 180.5 243.8s40 146.3 114.2 163.9 149.9-23.3 175.7-95.1c9.4 1.7 18.7 3.6 28 5.8 28.2 6.6 56.4 15.4 82.4 26.6 70.7 30.2 109.3 70.1 107.5 119.9-1.6 44.6-33.6 65.2-96.2 68.6-27.5 1.5-57.6-.9-87.3-5.8-8.3-1.4-15.9-2.8-22.6-4.3-3.9-.8-6.6-1.5-7.8-1.8l-3.1-.6c-2.2-.3-5.9-.8-10.7-1.3-25-2.3-52.1-1.5-78.5 4.6-55.2 12.9-93.9 47.2-101.1 105.8-15.7 126.2 78.6 184.7 276 188.9 29.1 70.4 106.4 107.9 179.6 87 73.3-20.9 119.3-93.4 106.9-168.6M329.1 345.2a83.3 83.3 0 1 1 .01-166.61 83.3 83.3 0 0 1-.01 166.61M695.6 845a83.3 83.3 0 1 1 .01-166.61A83.3 83.3 0 0 1 695.6 845"/></svg>',
	conflicts: '<svg viewBox="0 0 400 400" width="100%" height="100%" pointer-events="none" xmlns="http://www.w3.org/2000/svg"><path fill="currentColor" d="m397.2 350.4.2-.2-180-320-.2.2C213.8 24.2 207.4 20 200 20s-13.8 4.2-17.2 10.4l-.2-.2-180 320 .2.2c-1.6 2.8-2.8 6-2.8 9.6 0 11 9 20 20 20h360c11 0 20-9 20-20 0-3.6-1.2-6.8-2.8-9.6M220 340h-40v-40h40zm0-60h-40V120h40z"/></svg>',
	passed: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 426.667 426.667"><path fill="#6AC259" d="M213.333,0C95.518,0,0,95.514,0,213.333s95.518,213.333,213.333,213.333c117.828,0,213.333-95.514,213.333-213.333S331.157,0,213.333,0z M174.199,322.918l-93.935-93.931l31.309-31.309l62.626,62.622l140.894-140.898l31.309,31.309L174.199,322.918z"/></svg>',
	download: '<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" width="100%" height="100%" viewBox="0 0 32 32"><path fill="currentColor" d="M26 24v4H6v-4H4v4a2 2 0 0 0 2 2h20a2 2 0 0 0 2-2v-4zm0-10l-1.41-1.41L17 20.17V2h-2v18.17l-7.59-7.58L6 14l10 10l10-10z"></path></svg>'
}