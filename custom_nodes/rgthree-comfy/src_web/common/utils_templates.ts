import * as dom from "./utils_dom";
import {getObjectValue} from "./shared_utils";

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

export interface BindOptions {
  /**
   * If true then only those data-bind keys in the data map will be bound,
   * and no `data-bind` fields will be unbound.
   */
  onlyDefined?: boolean;

  /** If true, then binding/init will not be called on nested templates. */
  singleScoped?: boolean;

  /** Context elemnt. */
  contextElement?: HTMLElement;
}

export interface InflateOptions {
  skipInit?: boolean;
  bindOptions?: BindOptions;
}

export interface TemplateData {
  fragment: DocumentFragment;
  preProcessScript: (data: any) => void;
}

export interface BindingContext {
  data: any;
  contextElement: HTMLElement;
  currentElement: HTMLElement;
}

// const RGX_COMPARISON = /^\(?([a-z0-9\.\-\[\]'"]+)((?:<|>|==)=?)([a-z0-9\.\-\[\]'"]+)\)?$/i;
const RGX_COMPARISON = (() => {
  // /^\(?([a-z0-9\.\-\[\]'"]+)((?:<|>|==)=?)([a-z0-9\.\-\[\]'"]+)\)?$/i;
  let value = "((?:\\!*)[_a-z0-9\\.\\-\\[\\]'\"]+)";
  let comparison = "((?:<|>|==|\\!=)=?)";
  return new RegExp(`^(?:\\!*)\\(?${value}\\s*${comparison}\\s*${value}\\)?$`, "i");
})();

const RGXPART_BIND_FN_TEMPLATE_STRING = "template|tpl";
const RGXPART_BIND_FN_ELEMENT_STRING = "element|el";
const RGX_BIND_FN_TEMPLATE = new RegExp(
  `^(?:${RGXPART_BIND_FN_TEMPLATE_STRING})\\(([^\\)]+)\\)`,
  "i",
);
const RGX_BIND_FN_ELEMENT = new RegExp(
  `^(?:${RGXPART_BIND_FN_ELEMENT_STRING})\\(([^\\)]+)\\)`,
  "i",
);
const RGX_BIND_FN_TEMPLATE_OR_ELEMENT = new RegExp(
  `^(?:${RGXPART_BIND_FN_TEMPLATE_STRING}|${RGXPART_BIND_FN_ELEMENT_STRING})\\(([^\\)]+)\\)`,
  "i",
);
const RGX_BIND_FN_LENGTH = /^(?:length|len|size)\(([^\)]+)\)/i;
const RGX_BIND_FN_FORMAT = /^(?:format|fmt)\(([^\,]+),([^\)]+)\)/i;
const RGX_BIND_FN_CALL = /^([^\(]+)\(([^\)]*)\)/i;

const EMPTY_PREPROCESS_FN = (data: any) => data;

// This is used within exec, so we don't need to check the first part since it's
// always the lastIndex start position
// const RGX_BIND_DECLARATIONS = /\s*((?:[\$_a-z0-9-\.]|\?\?|\|\|)+(?:\([^\)]+\))?)(?::(.*?))?(\s|$)/ig;
const RGX_BIND_DECLARATIONS =
  /\s*(\!*(?:[\$_a-z0-9-\.\'\"]|\?\?|\|\||\&\&|(?:(?:<|>|==|\!=)=?))+(?:\`[^\`]+\`)?(?:\([^\)]*\))?)(?::(.*?))?(\s|$)/gi;

/**
 * Asserts that something is not null of undefined.
 */
function localAssertNotFalsy<T>(input?: T | null, errorMsg = `Input is not of type.`): T {
  if (input == null) {
    throw new Error(errorMsg);
  }
  return input;
}

/**
 * Cleans a key.
 */
function cleanKey(key: string) {
  return key.toLowerCase().trim().replace(/\s/g, "");
}

/**
 * Ensures the value is an array, converting array-like items to an array.
 */
function toArray(value: any | any[]): any[] {
  if (Array.isArray(value)) {
    return value;
  }
  if (value instanceof Set) {
    return Array.from(value);
  }
  // Array-like.
  if (typeof value === "object" && typeof value.length === "number") {
    return [].slice.call(value);
  }
  return [value];
}

/**
 * Flattens an array.
 */
function flattenArray(arr: any | any[]): any[] {
  return toArray(arr).reduce((acc, val) => {
    return acc.concat(Array.isArray(val) ? flattenArray(val) : val);
  }, []);
}

/**
 * Gets an object value by a string lookup.
 */
function getObjValue(lookup: string, obj: any): any {
  // If we want to cast as a boolean via a /!+/ prefix.
  let booleanMatch: string[] = lookup.match(/^(\!+)(.+?)$/i) || [];
  let booleanNots: string[] = [];
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

/**
 * Gets a primotove or object value.
 */
function getPrimitiveOrObjValue(stringValue: string | null | undefined, data: any) {
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
  } catch (e) {
    value = getObjValue(stringValue, data);
  }
  value = negate !== null ? (negate === 1 ? !value : !!value) : value;
  return value;
}

/**
 * Get the negates for a string. A `null` value means there are no negates,
 * otherwise a `1` means it should be negated, and a `0` means double-negate
 * (e.g. cast as boolean).
 *
 *   'boolVar' => null
 *   '!boolVar' => 1
 *   '!!boolVar' => 0
 *   '!!!boolVar' => 1
 *   '!!!!boolVar' => 0
 */
function getNegates(stringValue: string): number | null {
  let negate = null;
  let negateMatches = stringValue.match(/^(\!+)(.*)/);
  if (negateMatches && negateMatches.length >= 3) {
    negate = negateMatches[1]!.length % 2;
  }
  return negate;
}

/**
 * Returns the boolean for a comparison of object values. A `null` value would
 * be if the experssion does not match a comparison, and undefined if there was
 * but it's not a known comparison (===, ==, <=, >=, !==, !=, <, >).
 */
function getStringComparisonExpression(
  bindingPropName: string,
  data: any,
): boolean | null | undefined {
  let comparisonMatches = bindingPropName.match(RGX_COMPARISON);
  if (!comparisonMatches?.length) {
    return null;
  }
  let a = getPrimitiveOrObjValue(comparisonMatches[1]!, data);
  let b = getPrimitiveOrObjValue(comparisonMatches[3]!, data);
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

/**
 * Replaces a <tpl> element with children with special attribute copies for
 * single children.
 */
function replaceTplElementWithChildren(
  tplEl: HTMLElement,
  fragOrElOrEls: DocumentFragment | HTMLElement | Array<DocumentFragment | HTMLElement>,
) {
  const els = Array.isArray(fragOrElOrEls) ? fragOrElOrEls : [fragOrElOrEls];
  tplEl.replaceWith(...els);
  // dom.replaceChild(tplEl, fragOrElOrEls);
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

/**
 * Entry point to get data from a binding name. Checks for null coelescing and
 * logical-or operators.
 *
 * Handles templates as well:
 *
 *   my.value`some string ${value} is cool.`
 */
function getValueForBinding(bindingPropName: string, context: BindingContext) {
  console.log("getValueForBinding", bindingPropName, context);
  const data = context.data;
  let stringTemplate = null;
  let stringTemplates = /^(.*?)\`([^\`]*)\`$/.exec(bindingPropName.trim());
  if (stringTemplates?.length === 3) {
    bindingPropName = stringTemplates[1]!;
    stringTemplate = stringTemplates[2];
  }
  let value = null;

  let hadALogicalOp = false;
  const opsToValidation = new Map([
    [/\s*\?\?\s*/, (v: any) => v != null],
    [/\s*\|\|\s*/, (v: any) => !!v],
    [/\s*\&\&\s*/, (v: any) => !v],
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

/**
 * Gets the value for a binding prop name.
 */
function getValueForBindingPropName(bindingPropName: string, context: BindingContext) {
  const data = context.data;
  let negate = getNegates(bindingPropName);
  if (negate != null) {
    bindingPropName = bindingPropName.replace(/^\!+/, "");
  }
  let value;
  RGX_COMPARISON.lastIndex = 0;
  if (RGX_COMPARISON.test(bindingPropName)) {
    value = getStringComparisonExpression(bindingPropName, data);
  } else if (RGX_BIND_FN_LENGTH.test(bindingPropName)) {
    bindingPropName = RGX_BIND_FN_LENGTH.exec(bindingPropName)![1]!;
    value = getPrimitiveOrObjValue(bindingPropName, data);
    value = (value && value.length) || 0;
  } else if (RGX_BIND_FN_FORMAT.test(bindingPropName)) {
    let matches = RGX_BIND_FN_FORMAT.exec(bindingPropName);
    bindingPropName = matches![1]!;
    value = getPrimitiveOrObjValue(bindingPropName, data);
    value = matches![2]!.replace(/^['"]/, "").replace(/['"]$/, "").replace(/\$1/g, value);
  } else if (RGX_BIND_FN_CALL.test(bindingPropName)) {
    console.log("-----");
    console.log(bindingPropName);
    let matches = RGX_BIND_FN_CALL.exec(bindingPropName);
    const functionName = matches![1]!;
    const maybeDataName = matches![2] ?? null;
    value = getPrimitiveOrObjValue(maybeDataName, data);
    console.log(functionName, maybeDataName, value);
    // First, see if the instance has this call
    if (typeof value?.[functionName] === "function") {
      value = value[functionName](value, data, context.currentElement, context.contextElement);
    } else if (typeof data?.[functionName] === "function") {
      value = data[functionName](value, data, context.currentElement, context.contextElement);
    } else if (typeof (context.currentElement as any)?.[functionName] === "function") {
      value = (context.currentElement as any)[functionName](
        value,
        data,
        context.currentElement,
        context.contextElement,
      );
    } else if (typeof (context.contextElement as any)?.[functionName] === "function") {
      value = (context.contextElement as any)[functionName](
        value,
        data,
        context.currentElement,
        context.contextElement,
      );
    } else {
      console.error(
        `No method named ${functionName} on data or element instance. Just calling regular value.`,
      );
      value = getPrimitiveOrObjValue(bindingPropName, data);
    }
  } else {
    value = getPrimitiveOrObjValue(bindingPropName, data);
  }

  if (value !== undefined) {
    value = negate !== null ? (negate === 1 ? !value : !!value) : value;
  }
  return value;
}

/**
 * Removes data-bind attributes, ostensibly "freezing" the current element.
 *
 * @param deep Will remove all data-bind attributes when true. default behavior
 *    is only up to the next data-tpl.
 */
function removeBindingAttributes(
  elOrEls: DocumentFragment | HTMLElement | HTMLElement[],
  deep = false,
) {
  flattenArray(elOrEls || []).forEach((el) => {
    el.removeAttribute(CONFIG.attrBind);
    const innerBinds = dom.queryAll(`:scope [${CONFIG.attrBind}]`, el);
    // If we're deep, then pretend there are no data-tpl.
    const innerTplBinds = deep ? [] : dom.queryAll(`:scope [data-tpl] [${CONFIG.attrBind}]`);

    innerBinds.forEach((el) => {
      if (deep || !innerTplBinds.includes(el)) {
        el.removeAttribute(CONFIG.attrBind);
      }
    });
  });
}

const templateCache: {[key: string]: TemplateData} = {};

/**
 * Checks if a template exists.
 */
export function checkKey(key: string) {
  return !!templateCache[cleanKey(key)];
}

/**
 * Register a template to it's key and a DocumentFragment to store the markup.
 * Uses `<template>` shadow DOM if possible. Overloaded to accept a register
 * a script as second param.
 */
export function register(
  key: string,
  htmlOrElement: string | Element | null = null,
  preProcessScript?: (data: any) => void,
) {
  key = cleanKey(key);
  if (templateCache[key]) {
    return templateCache[key];
  }

  let fragment: DocumentFragment | null = null;
  if (typeof htmlOrElement === "string") {
    const frag = document.createDocumentFragment();
    if (htmlOrElement.includes("<")) {
      const html = htmlOrElement.trim();
      const htmlParentTag =
        (html.startsWith("<tr") && "tbody") ||
        (/^<t(body|head|foot)/i.test(html) && "table") ||
        (/^<t(d|h)/i.test(html) && "tr") ||
        "div";
      const temp = document.createElement(htmlParentTag);
      temp.innerHTML = html;
      for (const child of temp.children) {
        frag.appendChild(child);
      }
    } else {
      frag.appendChild(dom.createElement(htmlOrElement));
    }
    fragment = frag;
  } else if (htmlOrElement instanceof Element) {
    const element = htmlOrElement as HTMLElement;
    const tag = element.nodeName.toLowerCase();
    if (tag === "template" && (element as HTMLTemplateElement).content) {
      fragment = (element as HTMLTemplateElement).content;
    } else {
      throw Error("Non-template element not handled");
    }
  } else if (!htmlOrElement) {
    let element = dom.query(`template[id="${key}"],template[data-id="${key}"]`);
    if (element && (element as HTMLTemplateElement).content) {
      fragment = (element as HTMLTemplateElement).content;
    } else {
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

export function getPreProcessScript(keyOrEl: string | Element) {
  if (typeof keyOrEl === "string") {
    if (!templateCache[keyOrEl]) {
      throw Error(`Template key does not exist ${keyOrEl}`);
    }
    return templateCache[keyOrEl].preProcessScript;
  }
  if (keyOrEl instanceof Element) {
    const tpl = keyOrEl.getAttribute("data-tpl") || "";
    return templateCache[tpl]?.preProcessScript || EMPTY_PREPROCESS_FN;
  }
  return EMPTY_PREPROCESS_FN;
}

/** Gets a template Node. */
export function getTemplateFragment(key: string): DocumentFragment {
  key = cleanKey(key);
  if (!checkKey(key)) {
    register(key);
  }
  let templateData = templateCache[key];
  if (templateData && templateData.fragment) {
    let imported: DocumentFragment;
    if (document.importNode) {
      imported = document.importNode(templateData.fragment, true);
    } else {
      imported = templateData.fragment.cloneNode(true) as DocumentFragment;
    }
    (imported as any).__templateid__ = key;
    return imported;
  } else {
    throw new Error("Ain't no template called " + key + " (" + typeof templateCache[key] + ")");
  }
}

// ### templates::inflate
//
// Inflate a template.
//
export function inflate(
  nodeOrKey: string | HTMLElement | DocumentFragment,
  templateData: any = null,
  inflateOptions: InflateOptions = {},
) {
  let node = nodeOrKey as HTMLElement | DocumentFragment;
  if (typeof node === "string") {
    node = getTemplateFragment(node);
  }
  if (node) {
    // Check for nested templates by way of a [data-template] attribute
    // Commented out line below, as :scope doesn't seem to work when node is a document fragment..
    // dom.queryAll([':scope [data-template]', ':scope [data-templateid]',':scope [template]'], node).forEach((child: HTMLElement) => {
    // const els = dom.queryAll(['[data-template]', '[data-templateid]','[template]'], node);
    const els = dom.queryAll("[data-template], [data-templateid], [template]", node);
    for (const child of els) {
      // If there's a class name specified on the template call, then we want to add it
      let className = child.className || null;
      let childTemplateId = localAssertNotFalsy(
        child.getAttribute("data-template") ||
          child.getAttribute("data-templateid") ||
          child.getAttribute("template"),
        "No child template id provided.",
      );

      const dataAttribute = child.getAttribute("data") || "";

      const childData = (dataAttribute && getObjValue(dataAttribute, templateData)) || templateData;
      const tplsInflateOptions = Object.assign({}, inflateOptions);
      // If we passed in skipInit we'll use it, otherwise set to true assuming
      // this pass is initializing final markup
      if (tplsInflateOptions.skipInit != null) {
        tplsInflateOptions.skipInit = true;
      }
      let tpls = localAssertNotFalsy(
        inflate(childTemplateId, childData, tplsInflateOptions),
        `No template inflated from ${childTemplateId}.`,
      );
      tpls = !Array.isArray(tpls) ? [tpls] : tpls;
      if (className) {
        for (const tpl of tpls) {
          tpl.classList.add(className);
        }
      }
      if (child.nodeName.toUpperCase() === "TPL") {
        replaceTplElementWithChildren(child, tpls);
      } else {
        child.append(...tpls);
      }
      // Old.
      // tpls.reverse().forEach((tplChild) => {
      //   dom.insertAfter(tplChild, child);
      //   if (className) {
      //     tplChild.classList.add(className);
      //   }
      // });
      child.remove();
    }

    let children: HTMLElement[] = [];
    for (const child of node.children) {
      let tplAttributes = (child.getAttribute("data-tpl") || "").split(" ");
      if (!tplAttributes.includes((node as any).__templateid__)) {
        tplAttributes.push((node as any).__templateid__);
      }
      child.setAttribute("data-tpl", tplAttributes.join(" ").trim());
      children.push(child as HTMLElement);
    }
    let childOrChildren = children.length === 1 ? children[0]! : children;
    if (!inflateOptions.skipInit) {
      init(childOrChildren, templateData, inflateOptions.bindOptions);
    }
    return childOrChildren;
  }
  return null;
}

export function inflateSingle(
  nodeOrKey: string | HTMLElement,
  scriptData: any = null,
  bindOptions: InflateOptions = {},
): HTMLElement {
  const inflated = localAssertNotFalsy(inflate(nodeOrKey, scriptData, bindOptions));
  return Array.isArray(inflated) ? inflated[0]! : inflated;
}

/**
 * Same as inflate, but removes bindings after inflating.
 * Useful when an element only needs to be inflated once without a desire to rebind
 * (or accidentally unbind elements)
 */
export function inflateOnce(
  nodeOrKey: string | HTMLElement | DocumentFragment,
  templateData: any = null,
  inflateOptions: InflateOptions = {},
) {
  let children = inflate(nodeOrKey, templateData, inflateOptions);
  children && removeBindingAttributes(children, false);
  return children;
}

export function inflateSingleOnce(
  nodeOrKey: string | HTMLElement,
  scriptData: any = null,
  bindOptions: InflateOptions = {},
): HTMLElement {
  const inflated = inflate(nodeOrKey, scriptData, bindOptions) || [];
  removeBindingAttributes(inflated, false);
  return Array.isArray(inflated) ? inflated[0]! : inflated;
}

/**
 * Initialize a template and bind to it's data.
 * Different than bind in that it will check for a registered script
 * and call that (bind simply binds the data to data-bind fields)
 */
export function init(els: HTMLElement | HTMLElement[], data: any, bindOptions: BindOptions = {}) {
  (!els ? [] : els instanceof Element ? [els] : els).forEach((el) => {
    const dataTplAttr = el.getAttribute("data-tpl");
    if (dataTplAttr) {
      const tpls = dataTplAttr.split(" ");
      tpls.forEach((tpl) => {
        // if (templateCache[tpl].script)
        //   templateCache[tpl].script(el, (data && (data[tpl] || data[tpl.replace('tpl:','')])) || data, options);
        // else
        const dataAttribute = el.getAttribute("data") || "";
        const childData = (dataAttribute && getObjValue(dataAttribute, data)) || data;
        bind(el, childData, bindOptions);
      });
    } else {
      bind(el, data, bindOptions);
    }
  });
}

// ### templates::bind
//
// Binds all elements under a template to the passed `data` JSON object. This *does not* call a registered script.
// It will stop binding elements once it reaches an element in the DOM with a `data-autobind` attribute set to false
// (which **MooVeeStar.View**s do automatically -- so each view is in control of it's binding)
//
//  - If a single empty element is passed, and data is a string value, then it will be used
//    as the value for that element
//  - If `options.onlyDefined === true` then no `data-bind` fields will be unbound, only those
//    `data-bind` keys in the data map will be bound
//
export function bind(
  elOrEls: HTMLElement | ShadowRoot | HTMLElement[],
  data: any = {},
  bindOptions: BindOptions = {},
) {
  if (elOrEls instanceof HTMLElement) {
    data = getPreProcessScript(elOrEls)({...data});
  }

  // If `els` is a single empty element w/ no `[data-bind]` set _and_
  // `data` is a string, set it to be the value of the el
  if (typeof data !== "object") {
    data = {value: data};
    if (
      elOrEls instanceof HTMLElement &&
      elOrEls.children.length === 0 &&
      !elOrEls.getAttribute(CONFIG.attrBind)
    ) {
      dom.setAttributes(elOrEls, {[CONFIG.attrBind]: "value"});
    }
  }

  // Get all children to be bind that are not inner binds
  let passedEls = !Array.isArray(elOrEls) ? [elOrEls] : elOrEls;
  for (const el of passedEls) {
    // First, get any condition els, evaluate them, then we'll skip them and children from binding
    // if they are false.
    const conditionEls = toArray(dom.queryAll(`[${CONFIG.attrIf}]`, el));
    const contextElement =
      bindOptions.contextElement ?? (el instanceof ShadowRoot ? (el.host as HTMLElement) : el);

    for (const conditionEl of conditionEls) {
      getValueForBindingPropName;
      // const isTrue = getStringComparisonExpression(conditionEl.getAttribute(CONFIG.attrIf), data);
      let isTrue = getValueForBinding(conditionEl.getAttribute(CONFIG.attrIf), {
        data,
        contextElement: contextElement,
        currentElement: conditionEl,
      });
      conditionEl.setAttribute(CONFIG.attrIfIs, String(!!isTrue));
    }

    let toBindEls = toArray(
      dom.queryAll(
        `:not([${CONFIG.attrIfIs}="false"]) [${CONFIG.attrBind}]:not([data-tpl]):not([${CONFIG.attrIfIs}="false"])`,
        el,
      ),
    );
    if (el instanceof HTMLElement && el.getAttribute(CONFIG.attrBind)) {
      toBindEls.unshift(el);
    }

    if (toBindEls.length) {
      // Exclude any els that are in their own data-tpl (which will follow)
      // let innerBindsElements = dom.queryAll([':scope [data-tpl] [data-bind]', ':scope [data-autobind="false"] [data-bind]'], el);
      let innerBindsElements = dom.queryAll(
        `:scope [data-tpl] [${CONFIG.attrBind}], :scope [data-autobind="false"] [${CONFIG.attrBind}]`,
        el,
      );
      toBindEls = toBindEls.filter((maybeBind) => !innerBindsElements.includes(maybeBind));
      toBindEls.forEach((child) => {
        // Get the bindings this elements wants
        // let bindings = child.getAttribute('data-bind').replace(/\s+/,' ').trim().split(' ') || [];
        RGX_BIND_DECLARATIONS.lastIndex = 0;
        let bindings = [];
        let bindingMatch;
        while (
          (bindingMatch = RGX_BIND_DECLARATIONS.exec(
            child.getAttribute(CONFIG.attrBind).replace(/\s+/, " ").trim(),
          )) !== null
        ) {
          bindings.push([bindingMatch[1], bindingMatch[2]]);
        }

        // let bindingStrings: string[] = child.getAttribute(CONFIG.attrBind).split(' ') || [];
        // bindingStrings.forEach((bindingString) => {
        bindings.forEach((bindings) => {
          // let bindingStringsSplit = bindings.split(':');
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
            } else {
              value = null;
            }
          }
          bindingFields.forEach((field) => {
            if (field.startsWith("style.")) {
              let stringVal = String(value);
              if (
                value &&
                !stringVal.includes("url(") &&
                stringVal !== "none" &&
                (field.includes("background-image") || stringVal.startsWith("http"))
              ) {
                value = `url(${value})`;
              }
              dom.setStyle(child, field.replace("style.", ""), value);

              // special element methods.
            } else if (field.startsWith("el.")) {
              if (field === "el.remove") {
                if (value === true) {
                  child.remove();
                }
              } else if (field === "el.toggle") {
                dom.setStyle(child, "display", value === true ? "" : "none");
              } else if (field.startsWith("el.classList.toggle")) {
                const cssClass = field.replace(/el.classList.toggle\(['"]?(.*?)['"]?\)/, "$1");
                child.classList.toggle(cssClass, !!value);
              }

              // [array]:tpl(<templatename>) will inflate the specified template for each item
            } else if (RGX_BIND_FN_TEMPLATE_OR_ELEMENT.test(field)) {
              dom.empty(child);
              let elementOrTemplateName = RGX_BIND_FN_TEMPLATE_OR_ELEMENT.exec(field)![1]!;
              if (Array.isArray(value) || value instanceof Set) {
                const arrayVals = toArray(value);
                let isElement = RGX_BIND_FN_ELEMENT.test(field);
                let frag = document.createDocumentFragment();
                arrayVals.forEach((item, index) => {
                  let itemData: {};
                  if (typeof item === "object") {
                    itemData = Object.assign({$index: index}, item);
                  } else {
                    itemData = {$index: index, value: item};
                  }

                  const els = bindToElOrTemplate(elementOrTemplateName, itemData);
                  frag.append(...els);
                });
                // If we're a <tpl>
                if (child.nodeName.toUpperCase() === "TPL") {
                  replaceTplElementWithChildren(child, frag);
                } else {
                  dom.empty(child).appendChild(frag);
                }
              } else if (value) {
                const els = bindToElOrTemplate(elementOrTemplateName, value);
                // If we're a <tpl>
                if (child.nodeName.toUpperCase() === "TPL") {
                  replaceTplElementWithChildren(child, els);
                } else {
                  child.append(...els);
                }
              }
            } else {
              dom.setAttributes(child, {[field]: value});
            }
          });
        });
      });
    }

    // Now loop over children w/ "data-tpl" and init them, unless they have an "data-autobind" set to "false"
    // (as in, they have a separate View Controller rendering their data)
    if (bindOptions.singleScoped !== true) {
      let toInitEls = toArray(el.querySelectorAll(":scope *[data-tpl]"));
      if (toInitEls.length) {
        // let innerInits = dom.queryAll([':scope *[data-tpl] *[data-tpl]', ':scope [data-autobind="false"] [data-tpl]'], el);
        let innerInits = dom.queryAll(
          ':scope *[data-tpl] *[data-tpl], :scope [data-autobind="false"] [data-tpl]',
          el,
        );
        toInitEls = toInitEls.filter((maybeInitEl) => {
          // If the el is inside another [data-tpl] don't init now (it will recursively next time)
          if (innerInits.includes(maybeInitEl)) {
            return false;
          }

          // If we passed in a specific map in data for this, then init
          let tplKey = maybeInitEl.getAttribute("data-tpl");
          if (data && (data[tplKey] || data[tplKey.replace("tpl:", "")])) {
            return true;
          }

          // Only init cascadingly if autobind is not "false"
          // (as in, a separate controller handles it's own rendering)
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

function bindToElOrTemplate(elementOrTemplateName: string, data: any) {
  let el: DocumentFragment | HTMLElement | HTMLElement[] | null =
    getTemplateFragment(elementOrTemplateName);
  if (!el) {
    el = dom.createElement(elementOrTemplateName, data);
  } else {
    // Inflate each template passing in item and have them init (force false skipInit)
    el = inflateOnce(el, data, {skipInit: false});
  }
  // Then, remove data-tpl b/c we just inflated it (and, presumably, it's data is
  // already set so we don't want to set it again below).
  // const els = (Array.isArray(el) ? el : [el]).filter(el => !!el) as HTMLElement[];
  // els.forEach(el => {
  //   el.removeAttribute('data-tpl');
  //   dom.queryAll('[data-tpl]', el).forEach(c => c.removeAttribute('data-tpl'));
  // });
  const els = (Array.isArray(el) ? el : [el]).filter((el) => !!el) as HTMLElement[];
  els.forEach((el) => {
    el.removeAttribute("data-tpl");
    let toBindEls = dom.queryAll("[data-tpl]", el);
    // let innerBindsElements = dom.queryAll([':scope [data-tpl] [data-bind]', ':scope [data-autobind="false"] [data-bind]'], el);
    let innerBindsElements = dom.queryAll(
      `:scope [data-tpl] [${CONFIG.attrBind}], :scope [data-autobind="false"] [${CONFIG.attrBind}]`,
      el,
    );
    toBindEls = toBindEls.filter((maybeBind) => !innerBindsElements.includes(maybeBind));
    toBindEls.forEach((c) => c.removeAttribute("data-tpl"));
  });
  return els;
}
