import { $el, getActionEls } from "../../common/utils_dom.js";
import { bind } from "../utils_templates.js";
const CSS_STYLE_SHEETS = new Map();
const CSS_STYLE_SHEETS_ADDED = new Map();
const HTML_TEMPLATE_FILES = new Map();
function getCommonPath(name, extension) {
    return `rgthree/common/components/${name.replace("rgthree-", "").replace(/\-/g, "_")}.${extension}`;
}
async function getStyleSheet(name, markupOrPath) {
    if (markupOrPath.includes("{")) {
        return markupOrPath;
    }
    if (!CSS_STYLE_SHEETS.has(name)) {
        try {
            const path = markupOrPath || getCommonPath(name, "css");
            const text = await (await fetch(path)).text();
            CSS_STYLE_SHEETS.set(name, text);
        }
        catch (e) {
        }
    }
    return CSS_STYLE_SHEETS.get(name);
}
async function addStyleSheet(name, markupOrPath) {
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
    return CSS_STYLE_SHEETS_ADDED.get(name);
}
async function getTemplateMarkup(name, markupOrPath) {
    if (markupOrPath.includes("<template")) {
        return markupOrPath;
    }
    if (!HTML_TEMPLATE_FILES.has(name)) {
        try {
            const path = markupOrPath || getCommonPath(name, "html");
            const text = await (await fetch(path)).text();
            HTML_TEMPLATE_FILES.set(name, text);
        }
        catch (e) {
        }
    }
    return HTML_TEMPLATE_FILES.get(name);
}
export class RgthreeCustomElement extends HTMLElement {
    constructor() {
        super(...arguments);
        this.ctor = this.constructor;
        this.hasBeenConnected = false;
        this.connected = false;
        this.templates = new Map();
        this.firstConnectedPromise = new Promise((resolve) => (this.firstConnectedPromiseResolver = resolve));
        this.eventElements = new Map();
    }
    static create() {
        if (this.NAME === "rgthree-override") {
            throw new Error("Must override component NAME");
        }
        if (!window.customElements.get(this.NAME)) {
            window.customElements.define(this.NAME, this);
        }
        return document.createElement(this.NAME);
    }
    onFirstConnected() {
    }
    onReconnected() {
    }
    onConnected() {
    }
    onDisconnected() {
    }
    onAction(action, e) {
        console.log("onAction", action, e);
    }
    getElement(query) {
        const el = this.querySelector(query);
        if (!el) {
            throw new Error("No element found for query: " + query);
        }
        return el;
    }
    onActionInternal(action, e) {
        if (typeof this[action] === "function") {
            this[action](e);
        }
        else {
            this.onAction(action, e);
        }
    }
    onConnectedInternal() {
        this.connectActionElements();
        this.onConnected();
    }
    onDisconnectedInternal() {
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
                    }
                    this.templates.set(id, template);
                }
            }
            if (this.ctor.USE_SHADOW) {
                this.root = this.attachShadow({ mode: "open" });
                if (typeof stylesheet === "string") {
                    const sheet = new CSSStyleSheet();
                    sheet.replaceSync(stylesheet);
                    this.root.adoptedStyleSheets = [sheet];
                }
            }
            else {
                this.root = this;
            }
            let template;
            if (this.templates.has(elementName)) {
                template = this.templates.get(elementName);
            }
            else if (this.templates.has(elementName.replace("rgthree-", ""))) {
                template = this.templates.get(elementName.replace("rgthree-", ""));
            }
            if (template) {
                this.root.appendChild(template.content.cloneNode(true));
                for (const name of template.getAttributeNames()) {
                    if (name != "id" && template.getAttribute(name)) {
                        this.setAttribute(name, template.getAttribute(name));
                    }
                }
            }
            this.onFirstConnected();
            this.hasBeenConnected = true;
            this.firstConnectedPromiseResolver();
        }
        else {
            this.onReconnected();
        }
        this.onConnectedInternal();
    }
    disconnectedCallback() {
        this.connected = false;
        this.onDisconnected();
    }
    connectActionElements() {
        const data = getActionEls(this);
        for (const dataItem of Object.values(data)) {
            const mapItem = this.eventElements.get(dataItem.el) || {};
            for (const [event, action] of Object.entries(dataItem.actions)) {
                if (mapItem[event]) {
                    console.warn(`Element already has an event for ${event}`);
                    continue;
                }
                mapItem[event] = (e) => {
                    this.onActionInternal(action, e);
                };
                dataItem.el.addEventListener(event, mapItem[event]);
            }
        }
    }
    disconnectActionElements() {
        for (const [el, eventData] of this.eventElements.entries()) {
            for (const [event, fn] of Object.entries(eventData)) {
                el.removeEventListener(event, fn);
            }
        }
    }
    async bindWhenConnected(data, el) {
        await this.firstConnectedPromise;
        this.bind(data, el);
    }
    bind(data, el) {
        bind(el || this.root, data);
    }
}
RgthreeCustomElement.NAME = "rgthree-override";
RgthreeCustomElement.USE_SHADOW = true;
RgthreeCustomElement.TEMPLATES = "";
RgthreeCustomElement.CSS = "";
