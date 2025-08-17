import { generateId, wait } from "./shared_utils.js";
import { createElement as $el, getClosestOrSelf, setAttributes} from "./utils_dom.js";

/**
 * A menu. Mimics the comfy menu.
 */
class Menu {

  private element: HTMLMenuElement = $el('menu.rgthree-menu');
  private callbacks: Map<string, (e: PointerEvent) => Promise<boolean|void>> = new Map();

  private handleWindowPointerDownBound = this.handleWindowPointerDown.bind(this);

  constructor(options: MenuOption[]) {
    this.setOptions(options);
    this.element.addEventListener('pointerup', async (e) => {
      const target = getClosestOrSelf(e.target as HTMLElement, "[data-callback],menu");
      if (e.which !== 1) {
        return;
      }
      const callback = target?.dataset?.['callback'];
      if (callback) {
        const halt = await this.callbacks.get(callback)?.(e);
        if (halt !== false) {
          this.close();
        }
      }
      e.preventDefault();
      e.stopPropagation();
      e.stopImmediatePropagation();
    });
  }

  setOptions(options: MenuOption[]) {
    for (const option of options) {
      if (option.type === 'title') {
        this.element.appendChild($el(`li`, {
          html: option.label
        }));
      } else {
        const id = generateId(8);
        this.callbacks.set(id, async (e: PointerEvent) => { return option?.callback?.(e); });
        this.element.appendChild($el(`li[role="button"][data-callback="${id}"]`, {
          html: option.label
        }));
      }
    }
  }

  toElement() {
    return this.element;
  }

  async open(e: PointerEvent) {
    const parent = (e.target as HTMLElement).closest('div,dialog,body') as HTMLElement
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

  handleWindowPointerDown(e:PointerEvent) {
    if (!this.element.contains(e.target as HTMLElement)) {
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

type MenuOption = {
  label: string;
  type?: 'title'|'item'|'separator';
  callback?: (e: PointerEvent) => void;
}

type MenuButtonOptions = {
  icon: string;
  options: MenuOption[];
}

export class MenuButton {

  private options: MenuButtonOptions;
  private menu: Menu;

  private element: HTMLButtonElement = $el('button.rgthree-button[data-action="open-menu"]')

  constructor(options: MenuButtonOptions) {
    this.options = options;
    this.element.innerHTML = options.icon;
    this.menu = new Menu(options.options);

    this.element.addEventListener('pointerdown', (e) => {
      if (!this.menu.isOpen())  {
        this.menu.open(e);
      }
    });
  }

  toElement() {
    return this.element;
  }

}