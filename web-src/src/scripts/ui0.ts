// import { api } from './api.js';
// import { ComfySettingsDialog } from './ui/settings0';
// import { toggleSwitch } from './ui/toggleSwitch.js';
// import { ComfyPromptStatus } from '../types/comfy.js';
// import { ComfyItems } from '../types/api.js';
// import { $el } from './utils.js';
// import { app } from './app.ts';
// import { clipspace } from './clipspace.ts';
//
// function dragElement(dragEl: HTMLElement, settings: ComfySettingsDialog) {
//     var posDiffX = 0,
//         posDiffY = 0,
//         posStartX = 0,
//         posStartY = 0,
//         newPosX = 0,
//         newPosY = 0;
//
//     if (dragEl.getElementsByClassName('drag-handle')[0]) {
//         // if present, the handle is where you move the DIV from:
//         (dragEl.getElementsByClassName('drag-handle')[0] as HTMLElement).onmousedown = dragMouseDown;
//     } else {
//         // otherwise, move the DIV from anywhere inside the DIV:
//         dragEl.onmousedown = dragMouseDown;
//     }
//
//     // When the element resizes (e.g. view queue) ensure it is still in the windows bounds
//     const resizeObserver = new ResizeObserver(() => {
//         ensureInBounds();
//     }).observe(dragEl);
//
//     function ensureInBounds() {
//         if (dragEl.classList.contains('comfy-menu-manual-pos')) {
//             newPosX = Math.min(document.body.clientWidth - dragEl.clientWidth, Math.max(0, dragEl.offsetLeft));
//             newPosY = Math.min(document.body.clientHeight - dragEl.clientHeight, Math.max(0, dragEl.offsetTop));
//
//             positionElement();
//         }
//     }
//
//     function positionElement() {
//         const halfWidth = document.body.clientWidth / 2;
//         const anchorRight = newPosX + dragEl.clientWidth / 2 > halfWidth;
//
//         // set the element's new position:
//         if (anchorRight) {
//             dragEl.style.left = 'unset';
//             dragEl.style.right = document.body.clientWidth - newPosX - dragEl.clientWidth + 'px';
//         } else {
//             dragEl.style.left = newPosX + 'px';
//             dragEl.style.right = 'unset';
//         }
//
//         dragEl.style.top = newPosY + 'px';
//         dragEl.style.bottom = 'unset';
//
//         if (savePos) {
//             localStorage.setItem(
//                 'Comfy.MenuPosition',
//                 JSON.stringify({
//                     x: dragEl.offsetLeft,
//                     y: dragEl.offsetTop,
//                 })
//             );
//         }
//     }
//
//     function restorePos() {
//         let pos = localStorage.getItem('Comfy.MenuPosition');
//         if (pos) {
//             const newPos = JSON.parse(pos);
//             newPosX = newPos.x;
//             newPosY = newPos.y;
//             positionElement();
//             ensureInBounds();
//         }
//     }
//
//     let savePos: undefined | any = undefined;
//     settings.addSetting({
//         id: 'Comfy.MenuPosition',
//         name: 'Save menu position',
//         type: 'boolean',
//         defaultValue: savePos,
//         onChange(value: any) {
//             if (savePos === undefined && value) {
//                 restorePos();
//             }
//             savePos = value;
//         },
//     });
//
//     function dragMouseDown(e: MouseEvent) {
//         e = e || window.event;
//         e.preventDefault();
//         // get the mouse cursor position at startup:
//         posStartX = e.clientX;
//         posStartY = e.clientY;
//         document.onmouseup = closeDragElement;
//         // call a function whenever the cursor moves:
//         document.onmousemove = elementDrag;
//     }
//
//     function elementDrag(e: MouseEvent) {
//         e = e || window.event;
//         e.preventDefault();
//
//         dragEl.classList.add('comfy-menu-manual-pos');
//
//         // calculate the new cursor position:
//         posDiffX = e.clientX - posStartX;
//         posDiffY = e.clientY - posStartY;
//         posStartX = e.clientX;
//         posStartY = e.clientY;
//
//         newPosX = Math.min(document.body.clientWidth - dragEl.clientWidth, Math.max(0, dragEl.offsetLeft + posDiffX));
//         newPosY = Math.min(document.body.clientHeight - dragEl.clientHeight, Math.max(0, dragEl.offsetTop + posDiffY));
//
//         positionElement();
//     }
//
//     window.addEventListener('resize', () => {
//         ensureInBounds();
//     });
//
//     function closeDragElement() {
//         // stop moving when mouse button is released:
//         document.onmouseup = null;
//         document.onmousemove = null;
//     }
// }
//
// class ComfyList {
//     #type;
//     #text;
//     #reverse;
//     element: HTMLElement;
//     button: HTMLButtonElement | null;
//
//     constructor(text: string, type: string, reverse: boolean) {
//         this.#text = text;
//         this.#type = type || text.toLowerCase();
//         this.#reverse = reverse || false;
//         this.element = $el('div.comfy-list') as HTMLElement;
//         this.element.style.display = 'none';
//         this.button = null;
//     }
//
//     get visible() {
//         return this.element.style.display !== 'none';
//     }
//
//     async load() {
//         const items = await api.getItems(this.#type);
//         this.element.replaceChildren(
//             ...Object.keys(items).flatMap(section => [
//                 $el('h4', {
//                     textContent: section,
//                 }),
//                 $el('div.comfy-list-items', [
//                     ...(this.#reverse
//                         ? (<ComfyItems[]>items[<keyof typeof items>section]).reverse()
//                         : items[<keyof typeof items>section]
//                     ).map((item: any) => {
//                         // Allow items to specify a custom remove action (e.g. for interrupt current prompt)
//                         const removeAction = item.remove || {
//                             name: 'Delete',
//                             cb: () => api.deleteItem(this.#type, item.prompt[1]),
//                         };
//                         return $el('div', { textContent: item.prompt[0] + ': ' }, [
//                             $el('button', {
//                                 textContent: 'Load',
//                                 onclick: async () => {
//                                     await app.loadGraphData(item.prompt[3].extra_pnginfo.workflow);
//                                     if (item.outputs) {
//                                         app.nodeOutputs = item.outputs;
//                                     }
//                                 },
//                             }),
//                             $el('button', {
//                                 textContent: removeAction.name,
//                                 onclick: async () => {
//                                     await removeAction.cb();
//                                     await this.update();
//                                 },
//                             }),
//                         ]);
//                     }),
//                 ]),
//             ]),
//             $el('div.comfy-list-actions', [
//                 $el('button', {
//                     textContent: 'Clear ' + this.#text,
//                     onclick: async () => {
//                         await api.clearItems(this.#type);
//                         await this.load();
//                     },
//                 }),
//                 $el('button', { textContent: 'Refresh', onclick: () => this.load() }),
//             ])
//         );
//     }
//
//     async update() {
//         if (this.visible) {
//             await this.load();
//         }
//     }
//
//     async show() {
//         this.element.style.display = 'block';
//         if (this.button) {
//             this.button.textContent = 'Close';
//         }
//
//         await this.load();
//     }
//
//     hide() {
//         this.element.style.display = 'none';
//         if (this.button) {
//             this.button.textContent = 'View ' + this.#text;
//         }
//     }
//
//     toggle() {
//         if (this.visible) {
//             this.hide();
//             return false;
//         } else {
//             this.show();
//             return true;
//         }
//     }
// }
//
// export class ComfyUI {
//     dialog: ComfyDialog;
//     settings: ComfySettingsDialog;
//     batchCount: number;
//     lastQueueSize: number;
//     queue: ComfyList;
//     history: ComfyList;
//     menuContainer: HTMLElement;
//     queueSize: HTMLElement | null;
//     autoQueueMode?: { text: string; value?: string; tooltip?: string } | string | null;
//     graphHasChanged: boolean = false;
//     autoQueueEnabled: boolean = false;
//
//     constructor() {
//         this.dialog = new ComfyDialog0();
//         this.settings = new ComfySettingsDialog();
//
//         this.batchCount = 1;
//         this.lastQueueSize = 0;
//         this.queue = new ComfyList('Queue', 'queue', true);
//         this.history = new ComfyList('History', 'history', true);
//         this.autoQueueMode = null;
//         this.queueSize = null;
//
//         api.addEventListener('status', () => {
//             this.queue.update();
//             this.history.update();
//         });
//
//         const confirmClear = this.settings.addSetting({
//             id: 'Comfy.ConfirmClear',
//             name: 'Require confirmation when clearing workflow',
//             type: 'boolean',
//             defaultValue: true,
//             onChange: () => undefined,
//         });
//
//         const promptFilename = this.settings.addSetting({
//             id: 'Comfy.PromptFilename',
//             name: 'Prompt for filename when saving workflow',
//             type: 'boolean',
//             defaultValue: true,
//             onChange: () => undefined,
//         });
//
//         /**
//          * file format for preview
//          *
//          * format;quality
//          *
//          * ex)
//          * webp;50 -> webp, quality 50
//          * jpeg;80 -> rgb, jpeg, quality 80
//          *
//          * @type {string}
//          */
//         const previewImage = this.settings.addSetting({
//             id: 'Comfy.PreviewFormat',
//             name: 'When displaying a preview in the image widget, convert it to a lightweight image, e.g. webp, jpeg, webp;50, etc.',
//             type: 'text',
//             defaultValue: '',
//             onChange: () => undefined,
//         });
//
//         this.settings.addSetting({
//             id: 'Comfy.DisableSliders',
//             name: 'Disable sliders.',
//             type: 'boolean',
//             defaultValue: false,
//             onChange: () => undefined,
//         });
//
//         this.settings.addSetting({
//             id: 'Comfy.DisableFloatRounding',
//             name: 'Disable rounding floats (requires page reload).',
//             type: 'boolean',
//             defaultValue: false,
//             onChange: () => undefined,
//         });
//
//         this.settings.addSetting({
//             id: 'Comfy.FloatRoundingPrecision',
//             name: 'Decimal places [0 = auto] (requires page reload).',
//             type: 'slider',
//             attrs: {
//                 min: 0,
//                 max: 6,
//                 step: 1,
//             },
//             defaultValue: 0,
//             onChange: () => undefined,
//         });
//
//         const fileInput = $el('input', {
//             id: 'comfy-file-input',
//             type: 'file',
//             accept: '.json,image/png,.latent,.safetensors,image/webp',
//             style: { display: 'none' },
//             parent: document.body,
//             onchange: () => {
//                 if ('files' in fileInput && Array.isArray(fileInput.files)) {
//                     app.handleFile(fileInput.files[0]);
//                 }
//             },
//         }) as HTMLInputElement;
//
//         const autoQueueModeEl = toggleSwitch(
//             'autoQueueMode',
//             [
//                 { text: 'instant', tooltip: 'A new prompt will be queued as soon as the queue reaches 0' },
//                 {
//                     text: 'change',
//                     tooltip: 'A new prompt will be queued when the queue is at 0 and the graph is/has changed',
//                 },
//             ],
//             {
//                 onChange: value => {
//                     this.autoQueueMode = value.item.value;
//                 },
//             }
//         ) as HTMLElement;
//         autoQueueModeEl.style.display = 'none';
//
//         api.addEventListener('graphChanged', () => {
//             if (this.autoQueueMode === 'change' && this.autoQueueEnabled === true) {
//                 if (this.lastQueueSize === 0) {
//                     this.graphHasChanged = false;
//                     app.queuePrompt(0, this.batchCount);
//                 } else {
//                     this.graphHasChanged = true;
//                 }
//             }
//         });
//
//         this.menuContainer = $el('div.comfy-menu', { parent: document.body }, [
//             $el(
//                 'div.drag-handle',
//                 {
//                     style: {
//                         overflow: 'hidden',
//                         position: 'relative',
//                         width: '100%',
//                         cursor: 'default',
//                     },
//                 },
//                 [
//                     $el('span.drag-handle'),
//                     $el('span', { $: q => (this.queueSize = q as HTMLElement) }),
//                     $el('button.comfy-settings-btn', { textContent: '⚙️', onclick: () => this.settings.show() }),
//                 ]
//             ),
//             $el('button.comfy-queue-btn', {
//                 id: 'queue-button',
//                 textContent: 'Queue Prompt',
//                 onclick: () => app.queuePrompt(0, this.batchCount),
//             }),
//             $el('div', {}, [
//                 $el('label', { innerHTML: 'Extra options' }, [
//                     $el('input', {
//                         type: 'checkbox',
//                         onchange: i => {
//                             let extraOptions = document.getElementById('extraOptions');
//                             if (extraOptions) {
//                                 extraOptions.style.display = i.srcElement.checked ? 'block' : 'none';
//                             }
//
//                             let batchCountInputRange = document.getElementById(
//                                 'batchCountInputRange'
//                             ) as HTMLInputElement;
//                             this.batchCount = i.srcElement.checked ? Number(batchCountInputRange.value) : 1;
//
//                             let autoQueueCheckbox = document.getElementById('autoQueueCheckbox') as HTMLInputElement;
//                             if (autoQueueCheckbox) {
//                                 autoQueueCheckbox.checked = false;
//                             }
//
//                             this.autoQueueEnabled = false;
//                         },
//                     }),
//                 ]),
//             ]),
//             $el('div', { id: 'extraOptions', style: { width: '100%', display: 'none' } }, [
//                 $el('div', [
//                     $el('label', { innerHTML: 'Batch count' }),
//                     $el('input', {
//                         id: 'batchCountInputNumber',
//                         type: 'number',
//                         value: this.batchCount,
//                         min: '1',
//                         style: { width: '35%', 'margin-left': '0.4em' },
//                         oninput: (i: InputEvent & { target: { value: any } }) => {
//                             this.batchCount = i.target?.value;
//                             let batchCountInputRange = <HTMLInputElement | null>(
//                                 document.getElementById('batchCountInputRange')
//                             );
//                             if (batchCountInputRange) {
//                                 batchCountInputRange.value = this.batchCount.toString();
//                             }
//                         },
//                     }),
//                     $el('input', {
//                         id: 'batchCountInputRange',
//                         type: 'range',
//                         min: '1',
//                         max: '100',
//                         value: this.batchCount,
//                         oninput: (i: InputEvent & { srcElement: { value: any } }) => {
//                             this.batchCount = i.srcElement?.value;
//                             let batchCountInputNumber = <HTMLInputElement | null>(
//                                 document.getElementById('batchCountInputNumber')
//                             );
//                             if (batchCountInputNumber) {
//                                 batchCountInputNumber.value = i.srcElement?.value;
//                             }
//                         },
//                     }),
//                 ]),
//
//                 $el('div', [
//                     $el('label', {
//                         for: 'autoQueueCheckbox',
//                         innerHTML: 'Auto Queue',
//                         // textContent: "Auto Queue"
//                     }),
//                     $el('input', {
//                         id: 'autoQueueCheckbox',
//                         type: 'checkbox',
//                         checked: false,
//                         title: 'Automatically queue prompt when the queue size hits 0',
//                         onchange: (e: Event & { target: { checked: boolean } }) => {
//                             this.autoQueueEnabled = e.target?.checked;
//                             autoQueueModeEl.style.display = this.autoQueueEnabled ? '' : 'none';
//                         },
//                     }),
//                     autoQueueModeEl,
//                 ]),
//             ]),
//             $el('div.comfy-menu-btns', [
//                 $el('button', {
//                     id: 'queue-front-button',
//                     textContent: 'Queue Front',
//                     onclick: () => app.queuePrompt(-1, this.batchCount),
//                 }),
//                 $el('button', {
//                     $: b => (this.queue.button = b as HTMLButtonElement),
//                     id: 'comfy-view-queue-button',
//                     textContent: 'View Queue',
//                     onclick: () => {
//                         this.history.hide();
//                         this.queue.toggle();
//                     },
//                 }),
//                 $el('button', {
//                     $: b => (this.history.button = b as HTMLButtonElement),
//                     id: 'comfy-view-history-button',
//                     textContent: 'View History',
//                     onclick: () => {
//                         this.queue.hide();
//                         this.history.toggle();
//                     },
//                 }),
//             ]),
//             this.queue.element,
//             this.history.element,
//             $el('button', {
//                 id: 'comfy-save-button',
//                 textContent: 'Save',
//                 onclick: () => {
//                     let filename: string | null = 'workflow.json';
//                     if (promptFilename.value) {
//                         filename = prompt('Save workflow as:', filename);
//                         if (!filename) return;
//                         if (!filename.toLowerCase().endsWith('.json')) {
//                             filename += '.json';
//                         }
//                     }
//                     app.graphToPrompt().then(p => {
//                         const json = JSON.stringify(p.workflow, null, 2); // convert the data to a JSON string
//                         const blob = new Blob([json], { type: 'application/json' });
//                         const url = URL.createObjectURL(blob);
//                         const a = $el('a', {
//                             href: url,
//                             download: filename,
//                             style: { display: 'none' },
//                             parent: document.body,
//                         }) as HTMLAnchorElement;
//                         a.click();
//                         setTimeout(function () {
//                             a.remove();
//                             window.URL.revokeObjectURL(url);
//                         }, 0);
//                     });
//                 },
//             }),
//             $el('button', {
//                 id: 'comfy-dev-save-api-button',
//                 textContent: 'Save (API Format)',
//                 style: { width: '100%', display: 'none' },
//                 onclick: () => {
//                     let filename: string | null = 'workflow_api.json';
//                     if (promptFilename.value) {
//                         filename = prompt('Save workflow (API) as:', filename);
//                         if (!filename) return;
//                         if (!filename.toLowerCase().endsWith('.json')) {
//                             filename += '.json';
//                         }
//                     }
//                     app.graphToPrompt().then(p => {
//                         const json = JSON.stringify(p.output, null, 2); // convert the data to a JSON string
//                         const blob = new Blob([json], { type: 'application/json' });
//                         const url = URL.createObjectURL(blob);
//                         const a = $el('a', {
//                             href: url,
//                             download: filename,
//                             style: { display: 'none' },
//                             parent: document.body,
//                         }) as HTMLAnchorElement;
//                         a.click();
//                         setTimeout(function () {
//                             a.remove();
//                             window.URL.revokeObjectURL(url);
//                         }, 0);
//                     });
//                 },
//             }),
//             $el('button', { id: 'comfy-load-button', textContent: 'Load', onclick: () => fileInput.click() }),
//             $el('button', {
//                 id: 'comfy-refresh-button',
//                 textContent: 'Refresh',
//                 onclick: () => app.refreshComboInNodes(),
//             }),
//             $el('button', {
//                 id: 'comfy-clipspace-button',
//                 textContent: 'Clipspace',
//                 onclick: () => clipspace.openClipspace?.(),
//             }),
//             $el('button', {
//                 id: 'comfy-clear-button',
//                 textContent: 'Clear',
//                 onclick: () => {
//                     if (!confirmClear.value || confirm('Clear workflow?')) {
//                         app.clean();
//                         app.graph?.clear();
//                     }
//                 },
//             }),
//             $el('button', {
//                 id: 'comfy-load-default-button',
//                 textContent: 'Load Default',
//                 onclick: async () => {
//                     if (!confirmClear.value || confirm('Load default workflow?')) {
//                         await app.loadGraphData();
//                     }
//                 },
//             }),
//         ]) as HTMLElement;
//
//         const devMode = this.settings.addSetting({
//             id: 'Comfy.DevMode',
//             name: 'Enable Dev mode Options',
//             type: 'boolean',
//             defaultValue: false,
//             onChange: function (value: string) {
//                 const devSaveApiButton = document.getElementById('comfy-dev-save-api-button');
//                 if (devSaveApiButton) {
//                     devSaveApiButton.style.display = value ? 'block' : 'none';
//                 }
//             },
//         });
//
//         dragElement(this.menuContainer, this.settings);
//
//         this.setStatus({ exec_info: { queue_remaining: 'X' } });
//     }
//
//     setStatus(status: ComfyPromptStatus) {
//         if (this.queueSize) {
//             this.queueSize.textContent = 'Queue size: ' + (status ? status.exec_info.queue_remaining : 'ERR');
//             if (status) {
//                 if (
//                     this.lastQueueSize != 0 &&
//                     status.exec_info.queue_remaining == 0 &&
//                     this.autoQueueEnabled &&
//                     (this.autoQueueMode === 'instant' || this.graphHasChanged) &&
//                     !app.lastExecutionError
//                 ) {
//                     app.queuePrompt(0, this.batchCount);
//                     status.exec_info.queue_remaining += this.batchCount;
//                     this.graphHasChanged = false;
//                 }
//                 this.lastQueueSize = status.exec_info.queue_remaining;
//             }
//         }
//     }
// }
//
// export { $el };
