import * as dom from "./utils_dom.js";
import { getObjectValue } from "./shared_utils.js";
const CONFIG_DEFAULT = {
    attrBind: "data-bind",
    attrIf: "data-if",
    attrIfIs: "data-if-is",
};
const CONFIG = Object.assign({}, CONFIG_DEFAULT, {
    attrBind: "bind",
    attrIf: "if",
    attrIfIs: "if-is",
});
const RGX_COMPARISON = (() => {
    let value = "((?:\\!*)[_a-z0-9\\.\\-\\[\\]'\"]+)";
    let comparison = "((?:<|>|==|\\!=)=?)";
    return new RegExp(`^(?:\\!*)\\(?${value}\\s*${comparison}\\s*${value}\\)?$`, "i");
})();
const RGXPART_BIND_FN_TEMPLATE_STRING = "template|tpl";
const RGXPART_BIND_FN_ELEMENT_STRING = "element|el";
const RGX_BIND_FN_TEMPLATE = new RegExp(`^(?:${RGXPART_BIND_FN_TEMPLATE_STRING})\\(([^\\)]+)\\)`, "i");
const RGX_BIND_FN_ELEMENT = new RegExp(`^(?:${RGXPART_BIND_FN_ELEMENT_STRING})\\(([^\\)]+)\\)`, "i");
const RGX_BIND_FN_TEMPLATE_OR_ELEMENT = new RegExp(`^(?:${RGXPART_BIND_FN_TEMPLATE_STRING}|${RGXPART_BIND_FN_ELEMENT_STRING})\\(([^\\)]+)\\)`, "i");
const RGX_BIND_FN_LENGTH = /^(?:length|len|size)\(([^\)]+)\)/i;
const RGX_BIND_FN_FORMAT = /^(?:format|fmt)\(([^\,]+),([^\)]+)\)/i;
const RGX_BIND_FN_CALL = /^([^\(]+)\(([^\)]*)\)/i;
const EMPTY_PREPROCESS_FN = (data) => data;
const RGX_BIND_DECLARATIONS = /\s*(\!*(?:[\$_a-z0-9-\.\'\"]|\?\?|\|\||\&\&|(?:(?:<|>|==|\!=)=?))+(?:\`[^\`]+\`)?(?:\([^\)]*\))?)(?::(.*?))?(\s|$)/gi;
function localAssertNotFalsy(input, errorMsg = `Input is not of type.`) {
    if (input == null) {
        throw new Error(errorMsg);
    }
    return input;
}
function cleanKey(key) {
    return key.toLowerCase().trim().replace(/\s/g, "");
}
function toArray(value) {
    if (Array.isArray(value)) {
        return value;
    }
    if (value instanceof Set) {
        return Array.from(value);
    }
    if (typeof value === "object" && typeof value.length === "number") {
        return [].slice.call(value);
    }
    return [value];
}
function flattenArray(arr) {
    return toArray(arr).reduce((acc, val) => {
        return acc.concat(Array.isArray(val) ? flattenArray(val) : val);
    }, []);
}
function getObjValue(lookup, obj) {
    let booleanMatch = lookup.match(/^(\!+)(.+?)$/i) || [];
    let booleanNots = [];
    if (booleanMatch[1] && booleanMatch[2]) {
        booleanNots = booleanMatch[1].split("");
        lookup = booleanMatch[2];
    }
    let value = getObjectValue(obj, lookup);
    while (booleanNots.length) {
        value = !value;
        booleanNots.shift();
    }
    return value;
}
function getPrimitiveOrObjValue(stringValue, data) {
    let value;
    if (stringValue == null) {
        return stringValue;
    }
    let negate = getNegates(stringValue);
    if (negate != null) {
        stringValue = stringValue.replace(/^\!+/, "");
    }
    try {
        const cleanedStringValue = stringValue.replace(/^'(.*)'$/, '"$1"');
        value = JSON.parse(cleanedStringValue);
    }
    catch (e) {
        value = getObjValue(stringValue, data);
    }
    value = negate !== null ? (negate === 1 ? !value : !!value) : value;
    return value;
}
function getNegates(stringValue) {
    let negate = null;
    let negateMatches = stringValue.match(/^(\!+)(.*)/);
    if (negateMatches && negateMatches.length >= 3) {
        negate = negateMatches[1].length % 2;
    }
    return negate;
}
function getStringComparisonExpression(bindingPropName, data) {
    let comparisonMatches = bindingPropName.match(RGX_COMPARISON);
    if (!(comparisonMatches === null || comparisonMatches === void 0 ? void 0 : comparisonMatches.length)) {
        return null;
    }
    let a = getPrimitiveOrObjValue(comparisonMatches[1], data);
    let b = getPrimitiveOrObjValue(comparisonMatches[3], data);
    let c = comparisonMatches[2];
    let value = (() => {
        switch (c) {
            case "===":
                return a === b;
            case "==":
                return a == b;
            case "<=":
                return a <= b;
            case ">=":
                return a >= b;
            case "!==":
                return a !== b;
            case "!=":
                return a != b;
            case "<":
                return a < b;
            case ">":
                return a > b;
            default:
                return undefined;
        }
    })();
    return value;
}
function replaceTplElementWithChildren(tplEl, fragOrElOrEls) {
    const els = Array.isArray(fragOrElOrEls) ? fragOrElOrEls : [fragOrElOrEls];
    tplEl.replaceWith(...els);
    const numOfChildren = Array.isArray(fragOrElOrEls)
        ? fragOrElOrEls.length
        : fragOrElOrEls.childElementCount;
    if (numOfChildren === 1) {
        const firstChild = Array.isArray(fragOrElOrEls)
            ? fragOrElOrEls[0]
            : fragOrElOrEls.firstElementChild;
        if (firstChild instanceof Element) {
            if (tplEl.className.length) {
                firstChild.className += ` ${tplEl.className}`;
            }
            let attr = tplEl.getAttribute("data");
            if (attr) {
                firstChild.setAttribute("data", attr);
            }
            attr = tplEl.getAttribute(CONFIG.attrBind);
            if (attr) {
                firstChild.setAttribute(CONFIG.attrBind, attr);
            }
        }
    }
}
function getValueForBinding(bindingPropName, context) {
    console.log("getValueForBinding", bindingPropName, context);
    const data = context.data;
    let stringTemplate = null;
    let stringTemplates = /^(.*?)\`([^\`]*)\`$/.exec(bindingPropName.trim());
    if ((stringTemplates === null || stringTemplates === void 0 ? void 0 : stringTemplates.length) === 3) {
        bindingPropName = stringTemplates[1];
        stringTemplate = stringTemplates[2];
    }
    let value = null;
    let hadALogicalOp = false;
    const opsToValidation = new Map([
        [/\s*\?\?\s*/, (v) => v != null],
        [/\s*\|\|\s*/, (v) => !!v],
        [/\s*\&\&\s*/, (v) => !v],
    ]);
    for (const [op, fn] of opsToValidation.entries()) {
        if (bindingPropName.match(op)) {
            hadALogicalOp = true;
            const bindingPropNames = bindingPropName.split(op);
            for (const propName of bindingPropNames) {
                value = getValueForBindingPropName(propName, context);
                if (fn(value)) {
                    break;
                }
            }
            break;
        }
    }
    if (!hadALogicalOp) {
        value = getValueForBindingPropName(bindingPropName, context);
    }
    return stringTemplate && value != null
        ? stringTemplate.replace(/\$\{value\}/g, String(value))
        : value;
}
function getValueForBindingPropName(bindingPropName, context) {
    var _a, _b, _c;
    const data = context.data;
    let negate = getNegates(bindingPropName);
    if (negate != null) {
        bindingPropName = bindingPropName.replace(/^\!+/, "");
    }
    let value;
    RGX_COMPARISON.lastIndex = 0;
    if (RGX_COMPARISON.test(bindingPropName)) {
        value = getStringComparisonExpression(bindingPropName, data);
    }
    else if (RGX_BIND_FN_LENGTH.test(bindingPropName)) {
        bindingPropName = RGX_BIND_FN_LENGTH.exec(bindingPropName)[1];
        value = getPrimitiveOrObjValue(bindingPropName, data);
        value = (value && value.length) || 0;
    }
    else if (RGX_BIND_FN_FORMAT.test(bindingPropName)) {
        let matches = RGX_BIND_FN_FORMAT.exec(bindingPropName);
        bindingPropName = matches[1];
        value = getPrimitiveOrObjValue(bindingPropName, data);
        value = matches[2].replace(/^['"]/, "").replace(/['"]$/, "").replace(/\$1/g, value);
    }
    else if (RGX_BIND_FN_CALL.test(bindingPropName)) {
        console.log("-----");
        console.log(bindingPropName);
        let matches = RGX_BIND_FN_CALL.exec(bindingPropName);
        const functionName = matches[1];
        const maybeDataName = (_a = matches[2]) !== null && _a !== void 0 ? _a : null;
        value = getPrimitiveOrObjValue(maybeDataName, data);
        console.log(functionName, maybeDataName, value);
        if (typeof (value === null || value === void 0 ? void 0 : value[functionName]) === "function") {
            value = value[functionName](value, data, context.currentElement, context.contextElement);
        }
        else if (typeof (data === null || data === void 0 ? void 0 : data[functionName]) === "function") {
            value = data[functionName](value, data, context.currentElement, context.contextElement);
        }
        else if (typeof ((_b = context.currentElement) === null || _b === void 0 ? void 0 : _b[functionName]) === "function") {
            value = context.currentElement[functionName](value, data, context.currentElement, context.contextElement);
        }
        else if (typeof ((_c = context.contextElement) === null || _c === void 0 ? void 0 : _c[functionName]) === "function") {
            value = context.contextElement[functionName](value, data, context.currentElement, context.contextElement);
        }
        else {
            console.error(`No method named ${functionName} on data or element instance. Just calling regular value.`);
            value = getPrimitiveOrObjValue(bindingPropName, data);
        }
    }
    else {
        value = getPrimitiveOrObjValue(bindingPropName, data);
    }
    if (value !== undefined) {
        value = negate !== null ? (negate === 1 ? !value : !!value) : value;
    }
    return value;
}
function removeBindingAttributes(elOrEls, deep = false) {
    flattenArray(elOrEls || []).forEach((el) => {
        el.removeAttribute(CONFIG.attrBind);
        const innerBinds = dom.queryAll(`:scope [${CONFIG.attrBind}]`, el);
        const innerTplBinds = deep ? [] : dom.queryAll(`:scope [data-tpl] [${CONFIG.attrBind}]`);
        innerBinds.forEach((el) => {
            if (deep || !innerTplBinds.includes(el)) {
                el.removeAttribute(CONFIG.attrBind);
            }
        });
    });
}
const templateCache = {};
export function checkKey(key) {
    return !!templateCache[cleanKey(key)];
}
export function register(key, htmlOrElement = null, preProcessScript) {
    key = cleanKey(key);
    if (templateCache[key]) {
        return templateCache[key];
    }
    let fragment = null;
    if (typeof htmlOrElement === "string") {
        const frag = document.createDocumentFragment();
        if (htmlOrElement.includes("<")) {
            const html = htmlOrElement.trim();
            const htmlParentTag = (html.startsWith("<tr") && "tbody") ||
                (/^<t(body|head|foot)/i.test(html) && "table") ||
                (/^<t(d|h)/i.test(html) && "tr") ||
                "div";
            const temp = document.createElement(htmlParentTag);
            temp.innerHTML = html;
            for (const child of temp.children) {
                frag.appendChild(child);
            }
        }
        else {
            frag.appendChild(dom.createElement(htmlOrElement));
        }
        fragment = frag;
    }
    else if (htmlOrElement instanceof Element) {
        const element = htmlOrElement;
        const tag = element.nodeName.toLowerCase();
        if (tag === "template" && element.content) {
            fragment = element.content;
        }
        else {
            throw Error("Non-template element not handled");
        }
    }
    else if (!htmlOrElement) {
        let element = dom.query(`template[id="${key}"],template[data-id="${key}"]`);
        if (element && element.content) {
            fragment = element.content;
        }
        else {
            throw Error("Non-template element not handled");
        }
    }
    if (fragment) {
        templateCache[key] = {
            fragment,
            preProcessScript: preProcessScript || EMPTY_PREPROCESS_FN,
        };
    }
    return templateCache[key] || null;
}
export function getPreProcessScript(keyOrEl) {
    var _a;
    if (typeof keyOrEl === "string") {
        if (!templateCache[keyOrEl]) {
            throw Error(`Template key does not exist ${keyOrEl}`);
        }
        return templateCache[keyOrEl].preProcessScript;
    }
    if (keyOrEl instanceof Element) {
        const tpl = keyOrEl.getAttribute("data-tpl") || "";
        return ((_a = templateCache[tpl]) === null || _a === void 0 ? void 0 : _a.preProcessScript) || EMPTY_PREPROCESS_FN;
    }
    return EMPTY_PREPROCESS_FN;
}
export function getTemplateFragment(key) {
    key = cleanKey(key);
    if (!checkKey(key)) {
        register(key);
    }
    let templateData = templateCache[key];
    if (templateData && templateData.fragment) {
        let imported;
        if (document.importNode) {
            imported = document.importNode(templateData.fragment, true);
        }
        else {
            imported = templateData.fragment.cloneNode(true);
        }
        imported.__templateid__ = key;
        return imported;
    }
    else {
        throw new Error("Ain't no template called " + key + " (" + typeof templateCache[key] + ")");
    }
}
export function inflate(nodeOrKey, templateData = null, inflateOptions = {}) {
    let node = nodeOrKey;
    if (typeof node === "string") {
        node = getTemplateFragment(node);
    }
    if (node) {
        const els = dom.queryAll("[data-template], [data-templateid], [template]", node);
        for (const child of els) {
            let className = child.className || null;
            let childTemplateId = localAssertNotFalsy(child.getAttribute("data-template") ||
                child.getAttribute("data-templateid") ||
                child.getAttribute("template"), "No child template id provided.");
            const dataAttribute = child.getAttribute("data") || "";
            const childData = (dataAttribute && getObjValue(dataAttribute, templateData)) || templateData;
            const tplsInflateOptions = Object.assign({}, inflateOptions);
            if (tplsInflateOptions.skipInit != null) {
                tplsInflateOptions.skipInit = true;
            }
            let tpls = localAssertNotFalsy(inflate(childTemplateId, childData, tplsInflateOptions), `No template inflated from ${childTemplateId}.`);
            tpls = !Array.isArray(tpls) ? [tpls] : tpls;
            if (className) {
                for (const tpl of tpls) {
                    tpl.classList.add(className);
                }
            }
            if (child.nodeName.toUpperCase() === "TPL") {
                replaceTplElementWithChildren(child, tpls);
            }
            else {
                child.append(...tpls);
            }
            child.remove();
        }
        let children = [];
        for (const child of node.children) {
            let tplAttributes = (child.getAttribute("data-tpl") || "").split(" ");
            if (!tplAttributes.includes(node.__templateid__)) {
                tplAttributes.push(node.__templateid__);
            }
            child.setAttribute("data-tpl", tplAttributes.join(" ").trim());
            children.push(child);
        }
        let childOrChildren = children.length === 1 ? children[0] : children;
        if (!inflateOptions.skipInit) {
            init(childOrChildren, templateData, inflateOptions.bindOptions);
        }
        return childOrChildren;
    }
    return null;
}
export function inflateSingle(nodeOrKey, scriptData = null, bindOptions = {}) {
    const inflated = localAssertNotFalsy(inflate(nodeOrKey, scriptData, bindOptions));
    return Array.isArray(inflated) ? inflated[0] : inflated;
}
export function inflateOnce(nodeOrKey, templateData = null, inflateOptions = {}) {
    let children = inflate(nodeOrKey, templateData, inflateOptions);
    children && removeBindingAttributes(children, false);
    return children;
}
export function inflateSingleOnce(nodeOrKey, scriptData = null, bindOptions = {}) {
    const inflated = inflate(nodeOrKey, scriptData, bindOptions) || [];
    removeBindingAttributes(inflated, false);
    return Array.isArray(inflated) ? inflated[0] : inflated;
}
export function init(els, data, bindOptions = {}) {
    (!els ? [] : els instanceof Element ? [els] : els).forEach((el) => {
        const dataTplAttr = el.getAttribute("data-tpl");
        if (dataTplAttr) {
            const tpls = dataTplAttr.split(" ");
            tpls.forEach((tpl) => {
                const dataAttribute = el.getAttribute("data") || "";
                const childData = (dataAttribute && getObjValue(dataAttribute, data)) || data;
                bind(el, childData, bindOptions);
            });
        }
        else {
            bind(el, data, bindOptions);
        }
    });
}
export function bind(elOrEls, data = {}, bindOptions = {}) {
    var _a;
    if (elOrEls instanceof HTMLElement) {
        data = getPreProcessScript(elOrEls)({ ...data });
    }
    if (typeof data !== "object") {
        data = { value: data };
        if (elOrEls instanceof HTMLElement &&
            elOrEls.children.length === 0 &&
            !elOrEls.getAttribute(CONFIG.attrBind)) {
            dom.setAttributes(elOrEls, { [CONFIG.attrBind]: "value" });
        }
    }
    let passedEls = !Array.isArray(elOrEls) ? [elOrEls] : elOrEls;
    for (const el of passedEls) {
        const conditionEls = toArray(dom.queryAll(`[${CONFIG.attrIf}]`, el));
        const contextElement = (_a = bindOptions.contextElement) !== null && _a !== void 0 ? _a : (el instanceof ShadowRoot ? el.host : el);
        for (const conditionEl of conditionEls) {
            getValueForBindingPropName;
            let isTrue = getValueForBinding(conditionEl.getAttribute(CONFIG.attrIf), {
                data,
                contextElement: contextElement,
                currentElement: conditionEl,
            });
            conditionEl.setAttribute(CONFIG.attrIfIs, String(!!isTrue));
        }
        let toBindEls = toArray(dom.queryAll(`:not([${CONFIG.attrIfIs}="false"]) [${CONFIG.attrBind}]:not([data-tpl]):not([${CONFIG.attrIfIs}="false"])`, el));
        if (el instanceof HTMLElement && el.getAttribute(CONFIG.attrBind)) {
            toBindEls.unshift(el);
        }
        if (toBindEls.length) {
            let innerBindsElements = dom.queryAll(`:scope [data-tpl] [${CONFIG.attrBind}], :scope [data-autobind="false"] [${CONFIG.attrBind}]`, el);
            toBindEls = toBindEls.filter((maybeBind) => !innerBindsElements.includes(maybeBind));
            toBindEls.forEach((child) => {
                RGX_BIND_DECLARATIONS.lastIndex = 0;
                let bindings = [];
                let bindingMatch;
                while ((bindingMatch = RGX_BIND_DECLARATIONS.exec(child.getAttribute(CONFIG.attrBind).replace(/\s+/, " ").trim())) !== null) {
                    bindings.push([bindingMatch[1], bindingMatch[2]]);
                }
                bindings.forEach((bindings) => {
                    let bindingDataProperty = localAssertNotFalsy(bindings.shift());
                    let bindingFields = ((bindings.length && bindings[0]) || "default")
                        .trim()
                        .replace(/^\[(.*?)\]$/i, "$1")
                        .split(",");
                    let value = getValueForBinding(bindingDataProperty, {
                        data,
                        contextElement: contextElement,
                        currentElement: child,
                    });
                    if (value === undefined) {
                        if (bindOptions.onlyDefined === true) {
                            return;
                        }
                        else {
                            value = null;
                        }
                    }
                    bindingFields.forEach((field) => {
                        if (field.startsWith("style.")) {
                            let stringVal = String(value);
                            if (value &&
                                !stringVal.includes("url(") &&
                                stringVal !== "none" &&
                                (field.includes("background-image") || stringVal.startsWith("http"))) {
                                value = `url(${value})`;
                            }
                            dom.setStyle(child, field.replace("style.", ""), value);
                        }
                        else if (field.startsWith("el.")) {
                            if (field === "el.remove") {
                                if (value === true) {
                                    child.remove();
                                }
                            }
                            else if (field === "el.toggle") {
                                dom.setStyle(child, "display", value === true ? "" : "none");
                            }
                            else if (field.startsWith("el.classList.toggle")) {
                                const cssClass = field.replace(/el.classList.toggle\(['"]?(.*?)['"]?\)/, "$1");
                                child.classList.toggle(cssClass, !!value);
                            }
                        }
                        else if (RGX_BIND_FN_TEMPLATE_OR_ELEMENT.test(field)) {
                            dom.empty(child);
                            let elementOrTemplateName = RGX_BIND_FN_TEMPLATE_OR_ELEMENT.exec(field)[1];
                            if (Array.isArray(value) || value instanceof Set) {
                                const arrayVals = toArray(value);
                                let isElement = RGX_BIND_FN_ELEMENT.test(field);
                                let frag = document.createDocumentFragment();
                                arrayVals.forEach((item, index) => {
                                    let itemData;
                                    if (typeof item === "object") {
                                        itemData = Object.assign({ $index: index }, item);
                                    }
                                    else {
                                        itemData = { $index: index, value: item };
                                    }
                                    const els = bindToElOrTemplate(elementOrTemplateName, itemData);
                                    frag.append(...els);
                                });
                                if (child.nodeName.toUpperCase() === "TPL") {
                                    replaceTplElementWithChildren(child, frag);
                                }
                                else {
                                    dom.empty(child).appendChild(frag);
                                }
                            }
                            else if (value) {
                                const els = bindToElOrTemplate(elementOrTemplateName, value);
                                if (child.nodeName.toUpperCase() === "TPL") {
                                    replaceTplElementWithChildren(child, els);
                                }
                                else {
                                    child.append(...els);
                                }
                            }
                        }
                        else {
                            dom.setAttributes(child, { [field]: value });
                        }
                    });
                });
            });
        }
        if (bindOptions.singleScoped !== true) {
            let toInitEls = toArray(el.querySelectorAll(":scope *[data-tpl]"));
            if (toInitEls.length) {
                let innerInits = dom.queryAll(':scope *[data-tpl] *[data-tpl], :scope [data-autobind="false"] [data-tpl]', el);
                toInitEls = toInitEls.filter((maybeInitEl) => {
                    if (innerInits.includes(maybeInitEl)) {
                        return false;
                    }
                    let tplKey = maybeInitEl.getAttribute("data-tpl");
                    if (data && (data[tplKey] || data[tplKey.replace("tpl:", "")])) {
                        return true;
                    }
                    return maybeInitEl.getAttribute("data-autobind") !== "false";
                });
                toInitEls.forEach((toInitEl) => {
                    var tplKey = toInitEl.getAttribute("data-tpl");
                    init(toInitEl, (data && (data[tplKey] || data[tplKey.replace("tpl:", "")])) || data);
                });
            }
        }
    }
}
function bindToElOrTemplate(elementOrTemplateName, data) {
    let el = getTemplateFragment(elementOrTemplateName);
    if (!el) {
        el = dom.createElement(elementOrTemplateName, data);
    }
    else {
        el = inflateOnce(el, data, { skipInit: false });
    }
    const els = (Array.isArray(el) ? el : [el]).filter((el) => !!el);
    els.forEach((el) => {
        el.removeAttribute("data-tpl");
        let toBindEls = dom.queryAll("[data-tpl]", el);
        let innerBindsElements = dom.queryAll(`:scope [data-tpl] [${CONFIG.attrBind}], :scope [data-autobind="false"] [${CONFIG.attrBind}]`, el);
        toBindEls = toBindEls.filter((maybeBind) => !innerBindsElements.includes(maybeBind));
        toBindEls.forEach((c) => c.removeAttribute("data-tpl"));
    });
    return els;
}
