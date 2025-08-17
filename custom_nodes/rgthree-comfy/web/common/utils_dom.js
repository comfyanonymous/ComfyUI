const DIRECT_ATTRIBUTE_MAP = {
    cellpadding: "cellPadding",
    cellspacing: "cellSpacing",
    colspan: "colSpan",
    frameborder: "frameBorder",
    height: "height",
    maxlength: "maxLength",
    nonce: "nonce",
    role: "role",
    rowspan: "rowSpan",
    type: "type",
    usemap: "useMap",
    valign: "vAlign",
    width: "width",
};
const RGX_NUMERIC_STYLE_UNIT = "px";
const RGX_NUMERIC_STYLE = /^((max|min)?(width|height)|margin|padding|(margin|padding)?(left|top|bottom|right)|fontsize|borderwidth)$/i;
const RGX_DEFAULT_VALUE_PROP = /input|textarea|select/i;
function localAssertNotFalsy(input, errorMsg = `Input is not of type.`) {
    if (input == null) {
        throw new Error(errorMsg);
    }
    return input;
}
const RGX_STRING_VALID = "[a-z0-9_-]";
const RGX_TAG = new RegExp(`^([a-z]${RGX_STRING_VALID}*)(\\.|\\[|\\#|$)`, "i");
const RGX_ATTR_ID = new RegExp(`#(${RGX_STRING_VALID}+)`, "gi");
const RGX_ATTR_CLASS = new RegExp(`(^|\\S)\\.([a-z0-9_\\-\\.]+)`, "gi");
const RGX_STRING_CONTENT_TO_SQUARES = "(.*?)(\\[|\\])";
const RGX_ATTRS_MAYBE_OPEN = new RegExp(`\\[${RGX_STRING_CONTENT_TO_SQUARES}`, "gi");
const RGX_ATTRS_FOLLOW_OPEN = new RegExp(`^${RGX_STRING_CONTENT_TO_SQUARES}`, "gi");
export function queryAll(selectors, parent = document) {
    return Array.from(parent.querySelectorAll(selectors)).filter((n) => !!n);
}
export function query(selectors, parent = document) {
    var _a;
    return (_a = parent.querySelector(selectors)) !== null && _a !== void 0 ? _a : null;
}
export function createText(text) {
    return document.createTextNode(text);
}
export function getClosestOrSelf(element, query) {
    const el = element;
    return ((el === null || el === void 0 ? void 0 : el.closest) && ((el.matches(query) && el) || el.closest(query))) || null;
}
export function containsOrSelf(parent, contained) {
    var _a;
    return (parent === contained || ((_a = parent === null || parent === void 0 ? void 0 : parent.contains) === null || _a === void 0 ? void 0 : _a.call(parent, contained)) || false);
}
export function createElement(selectorOrMarkup, attrs) {
    const frag = getHtmlFragment(selectorOrMarkup);
    let element = frag === null || frag === void 0 ? void 0 : frag.firstElementChild;
    let selector = "";
    if (!element) {
        selector = selectorOrMarkup.replace(/[\r\n]\s*/g, "");
        const tag = getSelectorTag(selector) || "div";
        element = document.createElement(tag);
        selector = selector.replace(RGX_TAG, "$2");
        const brackets = selector.match(/(\[[^\]]+\])/g) || [];
        for (const bracket of brackets) {
            selector = selector.replace(bracket, "");
        }
        selector = selector.replace(RGX_ATTR_ID, '[id="$1"]');
        selector = selector.replace(RGX_ATTR_CLASS, (match, p1, p2) => `${p1}[class="${p2.replace(/\./g, " ")}"]`);
        selector += brackets.join("");
    }
    const selectorAttrs = getSelectorAttributes(selector);
    if (selectorAttrs) {
        for (const attr of selectorAttrs) {
            let matches = attr.substring(1, attr.length - 1).split("=");
            let key = localAssertNotFalsy(matches.shift());
            let value = matches.join("=");
            if (value === undefined) {
                setAttribute(element, key, true);
            }
            else {
                value = value.replace(/^['"](.*)['"]$/, "$1");
                setAttribute(element, key, value);
            }
        }
    }
    if (attrs) {
        setAttributes(element, attrs);
    }
    return element;
}
export const $el = createElement;
function getSelectorTag(str) {
    return tryMatch(str, RGX_TAG);
}
function getSelectorAttributes(selector) {
    RGX_ATTRS_MAYBE_OPEN.lastIndex = 0;
    let attrs = [];
    let result;
    while ((result = RGX_ATTRS_MAYBE_OPEN.exec(selector))) {
        let attr = result[0];
        if (attr.endsWith("]")) {
            attrs.push(attr);
        }
        else {
            attr =
                result[0] + getOpenAttributesRecursive(selector.substr(RGX_ATTRS_MAYBE_OPEN.lastIndex), 2);
            RGX_ATTRS_MAYBE_OPEN.lastIndex += attr.length - result[0].length;
            attrs.push(attr);
        }
    }
    return attrs;
}
function getOpenAttributesRecursive(selectorSubstring, openCount) {
    let matches = selectorSubstring.match(RGX_ATTRS_FOLLOW_OPEN);
    let result = "";
    if (matches && matches.length) {
        result = matches[0];
        openCount += result.endsWith("]") ? -1 : 1;
        if (openCount > 0) {
            result += getOpenAttributesRecursive(selectorSubstring.substr(result.length), openCount);
        }
    }
    return result;
}
function tryMatch(str, rgx, index = 1) {
    var _a;
    let found = "";
    try {
        found = ((_a = str.match(rgx)) === null || _a === void 0 ? void 0 : _a[index]) || "";
    }
    catch (e) {
        found = "";
    }
    return found;
}
export function setAttributes(element, data) {
    let attr;
    for (attr in data) {
        if (data.hasOwnProperty(attr)) {
            setAttribute(element, attr, data[attr]);
        }
    }
}
function getHtmlFragment(value) {
    if (value.match(/^\s*<.*?>[\s\S]*<\/[a-z0-9]+>\s*$/)) {
        return document.createRange().createContextualFragment(value.trim());
    }
    return null;
}
function getChild(value) {
    if (value instanceof Node) {
        return value;
    }
    if (typeof value === "string") {
        let child = getHtmlFragment(value);
        if (child) {
            return child;
        }
        if (getSelectorTag(value)) {
            return createElement(value);
        }
        return createText(value);
    }
    if (value && typeof value.toElement === "function") {
        return value.toElement();
    }
    return null;
}
export function setAttribute(element, attribute, value) {
    let isRemoving = value == null;
    if (attribute === "default") {
        attribute = RGX_DEFAULT_VALUE_PROP.test(element.nodeName) ? "value" : "text";
    }
    if (attribute === "text") {
        empty(element).appendChild(createText(value != null ? String(value) : ""));
    }
    else if (attribute === "html") {
        empty(element).innerHTML += value != null ? String(value) : "";
    }
    else if (attribute == "style") {
        if (typeof value === "string") {
            element.style.cssText = isRemoving ? "" : value != null ? String(value) : "";
        }
        else {
            for (const [styleKey, styleValue] of Object.entries(value)) {
                element.style[styleKey] = styleValue;
            }
        }
    }
    else if (attribute == "events") {
        for (const [key, fn] of Object.entries(value)) {
            addEvent(element, key, fn);
        }
    }
    else if (attribute === "parent") {
        value.appendChild(element);
    }
    else if (attribute === "child" || attribute === "children") {
        if (typeof value === "string" && /^\[[^\[\]]+\]$/.test(value)) {
            const parseable = value.replace(/^\[([^\[\]]+)\]$/, '["$1"]').replace(/,/g, '","');
            try {
                const parsed = JSON.parse(parseable);
                value = parsed;
            }
            catch (e) {
                console.error(e);
            }
        }
        if (attribute === "children") {
            empty(element);
        }
        let children = value instanceof Array ? value : [value];
        for (let child of children) {
            child = getChild(child);
            if (child instanceof Node) {
                if (element instanceof HTMLTemplateElement) {
                    element.content.appendChild(child);
                }
                else {
                    element.appendChild(child);
                }
            }
        }
    }
    else if (attribute == "for") {
        element.htmlFor = value != null ? String(value) : "";
        if (isRemoving) {
            element.removeAttribute("for");
        }
    }
    else if (attribute === "class" || attribute === "className" || attribute === "classes") {
        element.className = isRemoving ? "" : Array.isArray(value) ? value.join(" ") : String(value);
    }
    else if (attribute === "dataset") {
        if (typeof value !== "object") {
            console.error("Expecting an object for dataset");
            return;
        }
        for (const [key, val] of Object.entries(value)) {
            element.dataset[key] = String(val);
        }
    }
    else if (attribute.startsWith("on") && typeof value === "function") {
        element.addEventListener(attribute.substring(2), value);
    }
    else if (["checked", "disabled", "readonly", "required", "selected"].includes(attribute)) {
        element[attribute] = !!value;
        if (!value) {
            element.removeAttribute(attribute);
        }
        else {
            element.setAttribute(attribute, attribute);
        }
    }
    else if (DIRECT_ATTRIBUTE_MAP.hasOwnProperty(attribute)) {
        if (isRemoving) {
            element.removeAttribute(DIRECT_ATTRIBUTE_MAP[attribute]);
        }
        else {
            element.setAttribute(DIRECT_ATTRIBUTE_MAP[attribute], String(value));
        }
    }
    else if (isRemoving) {
        element.removeAttribute(attribute);
    }
    else {
        let oldVal = element.getAttribute(attribute);
        if (oldVal !== value) {
            element.setAttribute(attribute, String(value));
        }
    }
}
function addEvent(element, key, fn) {
    element.addEventListener(key, fn);
}
function setStyles(element, styles = null) {
    if (styles) {
        for (let name in styles) {
            setStyle(element, name, styles[name]);
        }
    }
    return element;
}
export function setStyle(element, name, value) {
    name = name.indexOf("float") > -1 ? "cssFloat" : name;
    if (name.indexOf("-") != -1) {
        name = name.replace(/-\D/g, (match) => {
            return match.charAt(1).toUpperCase();
        });
    }
    if (value == String(Number(value)) && RGX_NUMERIC_STYLE.test(name)) {
        value = value + RGX_NUMERIC_STYLE_UNIT;
    }
    if (name === "display" && typeof value !== "string") {
        value = !!value ? null : "none";
    }
    element.style[name] = value === null ? null : String(value);
    return element;
}
export function empty(element) {
    while (element.firstChild) {
        element.removeChild(element.firstChild);
    }
    return element;
}
export function appendChildren(el, children) {
    children = !Array.isArray(children) ? [children] : children;
    for (let child of children) {
        child = getChild(child);
        if (child instanceof Node) {
            if (el instanceof HTMLTemplateElement) {
                el.content.appendChild(child);
            }
            else {
                el.appendChild(child);
            }
        }
    }
}
export function getActionEls(parent = document) {
    const els = Array.from(parent.querySelectorAll("[data-action],[on-action],[on]"));
    if (parent instanceof Element) {
        els.unshift(parent);
    }
    return els
        .map((actionEl) => {
        const actions = {};
        const actionSegments = (actionEl.getAttribute("data-action") ||
            actionEl.getAttribute("on-action") ||
            actionEl.getAttribute("on") ||
            "").split(";");
        for (let segment of actionSegments) {
            let actionsData = segment
                .trim()
                .split(/\s*:\s*/g)
                .filter((i) => !!i.trim());
            if (!actionsData.length)
                continue;
            if (actionsData.length === 1) {
                if (actionEl instanceof HTMLInputElement) {
                    actionsData.unshift("input");
                }
                else {
                    actionsData.unshift("click");
                }
            }
            if (actionsData[0] && actionsData[1]) {
                actions[actionsData[0]] = actionsData[1];
            }
        }
        return {
            el: actionEl,
            actions,
        };
    })
        .filter((el) => !!el);
}
