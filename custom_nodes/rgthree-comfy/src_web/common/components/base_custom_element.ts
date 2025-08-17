import {$el, getActionEls} from "rgthree/common/utils_dom.js";
import {bind, register} from "../utils_templates";

const CSS_STYLE_SHEETS = new Map<string, string>();
const CSS_STYLE_SHEETS_ADDED = new Map<string, HTMLLinkElement>();
const HTML_TEMPLATE_FILES = new Map<string, string>();

function getCommonPath(name: string, extension: string) {
  return `rgthree/common/components/${name.replace("rgthree-", "").replace(/\-/g, "_")}.${extension}`;
}

/**
 * Fetches the stylesheet for the component, matched by the element name (minus the "rgthree-"
 * prefix).
 */
async function getStyleSheet(name: string, markupOrPath: string) {
  if (markupOrPath.includes("{")) {
    return markupOrPath;
  }
  if (!CSS_STYLE_SHEETS.has(name)) {
    try {
      const path = markupOrPath || getCommonPath(name, "css");
      const text = await (await fetch(path)).text();
      CSS_STYLE_SHEETS.set(name, text);
    } catch (e) {
      // alert("Error loading rgthree custom component css.");
    }
  }
  return CSS_STYLE_SHEETS.get(name)!;
}

/**
 * Adds the stylesheet to the page, once.
 */
async function addStyleSheet(name: string, markupOrPath: string) {
  if (markupOrPath.includes("{")) {
    throw new Error("Page-level stylesheets should be passed a path.");
  }
  if (!CSS_STYLE_SHEETS_ADDED.has(name)) {
    const link = document.createElement("link");
    link.rel = "stylesheet";
    link.href = markupOrPath;
    document.head.appendChild(link);
    CSS_STYLE_SHEETS_ADDED.set(name, link);
  }
  return CSS_STYLE_SHEETS_ADDED.get(name)!;
}

/**
 * Fetches the stylesheet for the component, matched by the element name (minus the "rgthree-"
 * prefix).
 */
async function getTemplateMarkup(name: string, markupOrPath: string) {
  if (markupOrPath.includes("<template")) {
    return markupOrPath;
  }
  if (!HTML_TEMPLATE_FILES.has(name)) {
    try {
      const path = markupOrPath || getCommonPath(name, "html");
      const text = await (await fetch(path)).text();
      HTML_TEMPLATE_FILES.set(name, text);
    } catch (e) {
      // alert("Error loading rgthree custom component markup.");
    }
  }
  return HTML_TEMPLATE_FILES.get(name)!;
}

/**
 * A base custom element.
 */
export abstract class RgthreeCustomElement extends HTMLElement {
  static readonly NAME: `rgthree-${string}` = "rgthree-override";
  static readonly USE_SHADOW: boolean = true;
  static readonly TEMPLATES: string = "";
  static readonly CSS: string = "";

  static create<T extends RgthreeCustomElement>(): T {
    if (this.NAME === "rgthree-override") {
      throw new Error("Must override component NAME");
    }
    if (!window.customElements.get(this.NAME)) {
      window.customElements.define(this.NAME, this as unknown as CustomElementConstructor);
    }
    return document.createElement(this.NAME) as T;
  }

  protected ctor = this.constructor as typeof RgthreeCustomElement;
  protected hasBeenConnected: boolean = false;
  protected connected: boolean = false;
  protected root!: ShadowRoot | HTMLElement;
  protected readonly templates = new Map<string, HTMLTemplateElement>();
  protected firstConnectedPromiseResolver!: Function;
  protected firstConnectedPromise = new Promise(
    (resolve) => (this.firstConnectedPromiseResolver = resolve),
  );

  onFirstConnected(): void {
    // Optionally overridden.
  }
  onReconnected(): void {
    // Optionally overridden.
  }
  onConnected(): void {
    // Optionally overridden.
  }
  onDisconnected(): void {
    // Optionally overridden.
  }
  onAction(action: string, e?: Event): void {
    console.log("onAction", action, e);
    // Optionally overridden.
  }

  getElement<E extends HTMLElement>(query: string) {
    const el = this.querySelector(query);
    if (!el) {
      throw new Error("No element found for query: " + query);
    }
    return el as E;
  }

  private onActionInternal(action: string, e?: Event): void {
    if (typeof (this as any)[action] === "function") {
      (this as any)[action](e);
    } else {
      this.onAction(action, e);
    }
  }

  private onConnectedInternal(): void {
    this.connectActionElements();
    this.onConnected();
  }

  private onDisconnectedInternal(): void {
    this.disconnectActionElements();
    this.onDisconnected();
  }

  async connectedCallback() {
    const elementName = this.ctor.NAME;
    const wasConnected = this.connected;
    if (!wasConnected) {
      this.connected = true;
    }
    if (!this.hasBeenConnected) {
      const [stylesheet, markup] = await Promise.all([
        this.ctor.USE_SHADOW
          ? getStyleSheet(elementName, this.ctor.CSS)
          : addStyleSheet(elementName, this.ctor.CSS),
        getTemplateMarkup(elementName, this.ctor.TEMPLATES),
      ]);

      if (markup) {
        const temp = $el("div");
        const templatesMarkup = markup.match(/<template[^]*?<\/template>/gm) || [];
        for (const markup of templatesMarkup) {
          temp.innerHTML = markup;
          const template = temp.children[0];
          if (!(template instanceof HTMLTemplateElement)) {
            throw new Error("Not a template element.");
          }
          let id = template.getAttribute("id");
          if (!id) {
            id = this.ctor.NAME;
            // throw new Error("Not template id.");
          }
          this.templates.set(id, template);
        }
      }

      // If we're using a shadow, then it's our root as a ShadowRoot. If we're not, then the root is
      // the custom element itself. This allows easy binding on "this.root" but it also means if we
      // want to set an atrtibute or otherwise access the actual custom element, we should use
      // "this" to be compatible with both.
      if (this.ctor.USE_SHADOW) {
        this.root = this.attachShadow({mode: "open"});
        if (typeof stylesheet === "string") {
          const sheet = new CSSStyleSheet();
          sheet.replaceSync(stylesheet);
          this.root.adoptedStyleSheets = [sheet];
        }
      } else {
        this.root = this;
      }

      let template: HTMLTemplateElement | undefined;
      if (this.templates.has(elementName)) {
        template = this.templates.get(elementName);
      } else if (this.templates.has(elementName.replace("rgthree-", ""))) {
        template = this.templates.get(elementName.replace("rgthree-", ""));
      }
      if (template) {
        this.root.appendChild(template.content.cloneNode(true));
        for (const name of template.getAttributeNames()) {
          if (name != "id" && template.getAttribute(name)) {
            this.setAttribute(name, template.getAttribute(name)!);
          }
        }
      }

      this.onFirstConnected();
      this.hasBeenConnected = true;
      this.firstConnectedPromiseResolver();
    } else {
      this.onReconnected();
    }
    this.onConnectedInternal();
  }

  disconnectedCallback() {
    this.connected = false;
    this.onDisconnected();
  }

  private readonly eventElements = new Map<Element, {[event: string]: EventListener}>();

  private connectActionElements() {
    const data = getActionEls(this);
    for (const dataItem of Object.values(data)) {
      const mapItem = this.eventElements.get(dataItem.el) || {};
      for (const [event, action] of Object.entries(dataItem.actions)) {
        if (mapItem[event]) {
          console.warn(`Element already has an event for ${event}`);
          continue;
        }
        mapItem[event] = (e: Event) => {
          this.onActionInternal(action, e);
        };
        dataItem.el.addEventListener(event as keyof ElementEventMap, mapItem[event]);
      }
    }
  }

  private disconnectActionElements() {
    for (const [el, eventData] of this.eventElements.entries()) {
      for (const [event, fn] of Object.entries(eventData)) {
        el.removeEventListener(event, fn);
      }
    }
  }

  async bindWhenConnected(data: any, el?: HTMLElement | ShadowRoot) {
    await this.firstConnectedPromise;
    this.bind(data, el);
  }

  bind(data: any, el?: HTMLElement | ShadowRoot) {
    bind(el || this.root, data);
  }
}
