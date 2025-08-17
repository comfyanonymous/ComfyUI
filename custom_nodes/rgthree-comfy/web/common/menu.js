import { generateId, wait } from "./shared_utils.js";
import { createElement as $el, getClosestOrSelf, setAttributes } from "./utils_dom.js";
class Menu {
    constructor(options) {
        this.element = $el('menu.rgthree-menu');
        this.callbacks = new Map();
        this.handleWindowPointerDownBound = this.handleWindowPointerDown.bind(this);
        this.setOptions(options);
        this.element.addEventListener('pointerup', async (e) => {
            var _a, _b;
            const target = getClosestOrSelf(e.target, "[data-callback],menu");
            if (e.which !== 1) {
                return;
            }
            const callback = (_a = target === null || target === void 0 ? void 0 : target.dataset) === null || _a === void 0 ? void 0 : _a['callback'];
            if (callback) {
                const halt = await ((_b = this.callbacks.get(callback)) === null || _b === void 0 ? void 0 : _b(e));
                if (halt !== false) {
                    this.close();
                }
            }
            e.preventDefault();
            e.stopPropagation();
            e.stopImmediatePropagation();
        });
    }
    setOptions(options) {
        for (const option of options) {
            if (option.type === 'title') {
                this.element.appendChild($el(`li`, {
                    html: option.label
                }));
            }
            else {
                const id = generateId(8);
                this.callbacks.set(id, async (e) => { var _a; return (_a = option === null || option === void 0 ? void 0 : option.callback) === null || _a === void 0 ? void 0 : _a.call(option, e); });
                this.element.appendChild($el(`li[role="button"][data-callback="${id}"]`, {
                    html: option.label
                }));
            }
        }
    }
    toElement() {
        return this.element;
    }
    async open(e) {
        const parent = e.target.closest('div,dialog,body');
        parent.appendChild(this.element);
        setAttributes(this.element, {
            style: {
                left: `${e.clientX + 16}px`,
                top: `${e.clientY - 16}px`,
            }
        });
        this.element.setAttribute('state', 'measuring-open');
        await wait(16);
        const rect = this.element.getBoundingClientRect();
        if (rect.right > window.innerWidth) {
            this.element.style.left = `${e.clientX - rect.width - 16}px`;
            await wait(16);
        }
        this.element.setAttribute('state', 'open');
        setTimeout(() => {
            window.addEventListener('pointerdown', this.handleWindowPointerDownBound);
        });
    }
    handleWindowPointerDown(e) {
        if (!this.element.contains(e.target)) {
            this.close();
        }
    }
    async close() {
        window.removeEventListener('pointerdown', this.handleWindowPointerDownBound);
        this.element.setAttribute('state', 'measuring-closed');
        await wait(16);
        this.element.setAttribute('state', 'closed');
        this.element.remove();
    }
    isOpen() {
        return (this.element.getAttribute('state') || '').includes('open');
    }
}
export class MenuButton {
    constructor(options) {
        this.element = $el('button.rgthree-button[data-action="open-menu"]');
        this.options = options;
        this.element.innerHTML = options.icon;
        this.menu = new Menu(options.options);
        this.element.addEventListener('pointerdown', (e) => {
            if (!this.menu.isOpen()) {
                this.menu.open(e);
            }
        });
    }
    toElement() {
        return this.element;
    }
}
