var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
var __classPrivateFieldSet = (this && this.__classPrivateFieldSet) || function (receiver, state, value, kind, f) {
    if (kind === "m") throw new TypeError("Private method is not writable");
    if (kind === "a" && !f) throw new TypeError("Private accessor was defined without a setter");
    if (typeof state === "function" ? receiver !== state || !f : !state.has(receiver)) throw new TypeError("Cannot write private member to an object whose class did not declare it");
    return (kind === "a" ? f.call(receiver, value) : f ? f.value = value : state.set(receiver, value)), value;
};
var __classPrivateFieldGet = (this && this.__classPrivateFieldGet) || function (receiver, state, kind, f) {
    if (kind === "a" && !f) throw new TypeError("Private accessor was defined without a getter");
    if (typeof state === "function" ? receiver !== state || !f : !state.has(receiver)) throw new TypeError("Cannot read private member from an object whose class did not declare it");
    return kind === "m" ? f : kind === "a" ? f.call(receiver) : f ? f.value : state.get(receiver);
};
var _PyDict_dict;
import { check, deepFreeze } from "./shared_utils.js";
const MEMOIZED = { parser: null };
class ExecuteContext {
    constructor(existing = {}) {
        Object.assign(this, !!window.structuredClone ? structuredClone(existing) : { ...existing });
    }
}
class InitialExecuteContext extends ExecuteContext {
}
const TYPE_TO_HANDLER = new Map([
    ["module", handleChildren],
    ["expression_statement", handleChildren],
    ["interpolation", handleInterpolation],
    ["block", handleChildren],
    ["comment", handleSwallow],
    ["return_statement", handleReturn],
    ["assignment", handleAssignment],
    ["named_expression", handleNamedExpression],
    ["identifier", handleIdentifier],
    ["attribute", handleAttribute],
    ["subscript", handleSubscript],
    ["call", handleCall],
    ["argument_list", handleArgumentsList],
    ["for_statement", handleForStatement],
    ["list_comprehension", handleListComprehension],
    ["comparison_operator", handleComparisonOperator],
    ["boolean_operator", handleBooleanOperator],
    ["binary_operator", handleBinaryOperator],
    ["not_operator", handleNotOperator],
    ["unary_operator", handleUnaryOperator],
    ["integer", handleNumber],
    ["float", handleNumber],
    ["string", handleString],
    ["tuple", handleList],
    ["list", handleList],
    ["dictionary", handleDictionary],
    ["pair", handleDictionaryPair],
    ["true", async (...args) => true],
    ["false", async (...args) => false],
]);
const DEFAULT_BUILT_INS = {
    round: { fn: (n) => Math.round(Number(n)) },
    ceil: { fn: (n) => Math.ceil(Number(n)) },
    floor: { fn: (n) => Math.floor(Number(n)) },
    len: { fn: (n) => { var _a, _b; return (_b = (_a = n === null || n === void 0 ? void 0 : n.__len__) === null || _a === void 0 ? void 0 : _a.call(n)) !== null && _b !== void 0 ? _b : n === null || n === void 0 ? void 0 : n.length; } },
    int: { fn: (n) => Math.floor(Number(n)) },
    float: { fn: (n) => Number(n) },
    str: { fn: (n) => String(n) },
    bool: { fn: (n) => !!n },
    list: { fn: (tupl = []) => new PyList(tupl) },
    tuple: { fn: (list = []) => new PyTuple(list) },
    dict: { fn: (dict = {}) => new PyDict(dict) },
    dir: { fn: (...args) => console.dir(...__unwrap__(...args)) },
    print: { fn: (...args) => console.log(...__unwrap__(...args)) },
    log: { fn: (...args) => console.log(...__unwrap__(...args)) },
};
export async function execute(code, ctx, additionalBuiltins) {
    var _a, _b;
    const builtIns = deepFreeze({ ...DEFAULT_BUILT_INS, ...(additionalBuiltins !== null && additionalBuiltins !== void 0 ? additionalBuiltins : {}) });
    ctx = new InitialExecuteContext(ctx);
    const root = (await parse(code)).rootNode;
    const value = await handleNode(new Node(root), ctx, builtIns);
    console.log("=====");
    console.log(`value`, (_b = (_a = value === null || value === void 0 ? void 0 : value.__unwrap__) === null || _a === void 0 ? void 0 : _a.call(value)) !== null && _b !== void 0 ? _b : value);
    console.log("context", ctx);
    return value;
}
async function parse(code) {
    if (!MEMOIZED.parser) {
        const TreeSitter = (await import("../lib/tree-sitter.js"));
        await TreeSitter.Parser.init();
        const lang = await TreeSitter.Language.load("rgthree/lib/tree-sitter-python.wasm");
        MEMOIZED.parser = new TreeSitter.Parser();
        MEMOIZED.parser.setLanguage(lang);
    }
    return MEMOIZED.parser.parse(code);
}
async function handleNode(node, ctx, builtIns) {
    const type = node.type;
    if (ctx.hasOwnProperty("__returned__"))
        return ctx["__returned__"];
    const handler = TYPE_TO_HANDLER.get(type);
    check(handler, "Unhandled type: " + type, node);
    return handler(node, ctx, builtIns);
}
async function handleChildren(node, ctx, builtIns) {
    let lastValue = null;
    for (const child of node.children) {
        if (!child)
            continue;
        lastValue = await handleNode(child, ctx, builtIns);
    }
    return lastValue;
}
async function handleSwallow(node, ctx, builtIns) {
}
async function handleReturn(node, ctx, builtIns) {
    const value = node.children.length > 1 ? handleNode(node.child(1), ctx, builtIns) : undefined;
    ctx["__returned__"] = value;
    return value;
}
async function handleIdentifier(node, ctx, builtIns) {
    var _a, _b;
    let value = ctx[node.text];
    if (value === undefined) {
        value = (_b = (_a = builtIns[node.text]) === null || _a === void 0 ? void 0 : _a.fn) !== null && _b !== void 0 ? _b : undefined;
    }
    return maybeWrapValue(value);
}
async function handleAttribute(node, ctx, builtIns) {
    const children = node.children;
    check(children.length === 3, "Expected 3 children for attribute.");
    check(children[1].type === ".", "Expected middle child to be '.' for attribute.");
    const inst = await handleNode(children[0], ctx, builtIns);
    const attr = children[2].text;
    checkAttributeAccessibility(inst, attr);
    let attribute = maybeWrapValue(inst[attr]);
    return typeof attribute === "function" ? attribute.bind(inst) : attribute;
}
async function handleSubscript(node, ctx, builtIns) {
    const children = node.children;
    check(children.length === 4, "Expected 4 children for subscript.");
    check(children[1].type === "[", "Expected 2nd child to be '[' for subscript.");
    check(children[3].type === "]", "Expected 4thd child to be ']' for subscript.");
    const inst = await handleNode(children[0], ctx, builtIns);
    const attr = await handleNode(children[2], ctx, builtIns);
    if (inst instanceof PyTuple && isInt(attr)) {
        return maybeWrapValue(inst.__at__(attr));
    }
    if (inst instanceof PyDict && typeof attr === "string") {
        return maybeWrapValue(inst.get(attr));
    }
    checkAttributeAccessibility(inst, attr);
    let attribute = maybeWrapValue(inst[attr]);
    return typeof attribute === "function" ? attribute.bind(inst) : attribute;
}
async function handleAssignment(node, ctx, builtIns) {
    check(node.children.length === 3, "Expected 3 children for assignment: identifier/attr, =, and value.");
    check(node.children[1].type === "=", "Expected middle child to be an '='.");
    let right = await handleNode(node.children[2], ctx, builtIns);
    const leftNode = node.children[0];
    let leftObj = ctx;
    let leftProp = "";
    if (leftNode.type === "identifier") {
        leftProp = leftNode.text;
    }
    else if (leftNode.type === "attribute") {
        leftObj = await handleNode(leftNode.children[0], ctx, builtIns);
        check(leftNode.children[2].type === "identifier", "Expected left hand assignment attribute to be an identifier.", leftNode);
        leftProp = leftNode.children[2].text;
    }
    else if (leftNode.type === "subscript") {
        leftObj = await handleNode(leftNode.children[0], ctx, builtIns);
        check(leftNode.children[1].type === "[");
        check(leftNode.children[3].type === "]");
        leftProp = await handleNode(leftNode.children[2], ctx, builtIns);
    }
    else {
        throw new Error(`Unhandled left-hand assignement type: ${leftNode.type}`);
    }
    if (leftProp == null) {
        throw new Error(`No property to assign value`);
    }
    if (leftObj instanceof PyTuple) {
        check(isInt(leftProp), "Expected an int for list assignment");
        leftObj.__put__(leftProp, right);
    }
    else if (leftObj instanceof PyDict) {
        check(typeof leftProp === "string", "Expected a string for dict assignment");
        leftObj.__put__(leftProp, right);
    }
    else {
        check(typeof leftProp === "string", "Expected a string for object assignment");
        if (!(leftObj instanceof InitialExecuteContext)) {
            checkAttributeAccessibility(leftObj, leftProp);
        }
        leftObj[leftProp] = right;
    }
    return right;
}
async function handleNamedExpression(node, ctx, builtIns) {
    check(node.children.length === 3, "Expected three children for named expression.");
    check(node.child(0).type === "identifier", "Expected identifier first in named expression.");
    const varName = node.child(0).text;
    ctx[varName] = await handleNode(node.child(2), ctx, builtIns);
    return maybeWrapValue(ctx[varName]);
}
async function handleCall(node, ctx, builtIns) {
    check(node.children.length === 2, "Expected 2 children for call, identifier and arguments.");
    const fn = await handleNode(node.children[0], ctx, builtIns);
    const args = await handleNode(node.children[1], ctx, builtIns);
    console.log("handleCall", fn, args);
    return fn(...args);
}
async function handleArgumentsList(node, ctx, builtIns) {
    const args = (await handleList(node, ctx, builtIns)).__unwrap__(false);
    return [...args];
}
async function handleForStatement(node, ctx, builtIns) {
    const childs = node.children;
    check(childs.length === 6);
    check(childs[4].type === ":");
    check(childs[5].type === "block");
    await helperGetLoopForIn(node, ctx, builtIns, async (forCtx) => {
        await handleNode(childs[5], forCtx, builtIns);
    });
}
async function handleListComprehension(node, ctx, builtIns) {
    const finalList = new PyList();
    const newCtx = { ...ctx };
    let finalEntryNode;
    const loopNodes = [];
    for (const child of node.children) {
        if (!child || ["[", "]"].includes(child.type))
            continue;
        if (child.type === "identifier" || child.type === "attribute") {
            if (finalEntryNode) {
                throw Error("Already have a list comprehension finalEntryNode.");
            }
            finalEntryNode = child;
        }
        else if (child.type === "for_in_clause") {
            loopNodes.push({ forIn: child });
        }
        else if (child.type === "if_clause") {
            loopNodes[loopNodes.length - 1]["if"] = child;
        }
    }
    if (!finalEntryNode) {
        throw Error("No list comprehension finalEntryNode.");
    }
    console.log(`handleListComprehension.loopNodes`, loopNodes);
    const handleLoop = async (loopNodes) => {
        const loopNode = loopNodes.shift();
        await helperGetLoopForIn(loopNode.forIn, newCtx, builtIns, async (forCtx) => {
            if (loopNode.if) {
                const ifNode = loopNode.if;
                check(ifNode.children.length === 2, "Expected 2 children for if_clause.");
                check(ifNode.child(0).text === "if", "Expected first child to be 'if'.");
                const good = await handleNode(ifNode.child(1), forCtx, builtIns);
                if (!good)
                    return;
            }
            Object.assign(newCtx, forCtx);
            if (loopNodes.length) {
                await handleLoop(loopNodes);
            }
            else {
                finalList.append(await handleNode(finalEntryNode, newCtx, builtIns));
            }
        }, () => ({ ...newCtx }));
        loopNodes.unshift(loopNode);
    };
    await handleLoop(loopNodes);
    return finalList;
}
async function helperGetLoopForIn(node, ctx, builtIns, eachFn, provideForCtx) {
    var _a;
    const childs = node.children;
    check(childs.length >= 3);
    check(childs[0].type === "for");
    check(["identifier", "pattern_list"].includes(childs[1].type), "Expected identifier for for loop.");
    check(childs[2].type === "in");
    let identifiers;
    if (childs[1].type === "identifier") {
        identifiers = [childs[1].text];
    }
    else {
        identifiers = childs[1].children
            .map((n) => {
            if (n.type === ",")
                return null;
            check(n.type === "identifier");
            return node.text;
        })
            .filter((n) => n != null);
    }
    const iterable = await handleNode(childs[3], ctx, builtIns);
    check(iterable instanceof PyTuple, "Expected for loop instance to be a list/tuple.");
    for (const item of iterable.__unwrap__(false)) {
        const forCtx = (_a = provideForCtx === null || provideForCtx === void 0 ? void 0 : provideForCtx()) !== null && _a !== void 0 ? _a : ctx;
        if (identifiers.length === 1) {
            forCtx[identifiers[0]] = item;
        }
        else {
            check(Array.isArray(item) && identifiers.length === item.length, "Expected iterable to be a list, like using dict.items()");
            for (let i = 0; i < identifiers.length; i++) {
                forCtx[identifiers[i]] = item[i];
            }
        }
        await eachFn(forCtx);
    }
}
async function handleNumber(node, ctx, builtIns) {
    return Number(node.text);
}
async function handleString(node, ctx, builtIns) {
    let str = "";
    for (const child of node.children) {
        if (!child || ["string_start", "string_end"].includes(child.type))
            continue;
        if (child.type === "string_content") {
            str += child.text;
        }
        else if (child.type === "interpolation") {
            check(child.children.length === 3, "Expected interpolation");
            str += await handleNode(child, ctx, builtIns);
        }
    }
    return str;
}
async function handleInterpolation(node, ...args) {
    check(node.children.length === 3, "Expected interpolation to be three nodes length.");
    check(node.children[0].type === "{" && node.children[2].type === "}", 'Expected interpolation to be wrapped in "{" and "}".');
    return await handleNode(node.children[1], ...args);
}
async function handleList(node, ctx, builtIns) {
    const list = [];
    for (const child of node.children) {
        if (!child || ["(", "[", ",", "]", ")"].includes(child.type))
            continue;
        list.push(await handleNode(child, ctx, builtIns));
    }
    if (node.type === "tuple") {
        return new PyTuple(list);
    }
    return new PyList(list);
}
async function handleComparisonOperator(node, ctx, builtIns) {
    const op = node.child(1).text;
    const left = await handleNode(node.child(0), ctx, builtIns);
    const right = await handleNode(node.child(2), ctx, builtIns);
    if (op === "==")
        return left === right;
    if (op === "!=")
        return left !== right;
    if (op === ">")
        return left > right;
    if (op === ">=")
        return left >= right;
    if (op === "<")
        return left < right;
    if (op === "<=")
        return left <= right;
    if (op === "in")
        return (right.__unwrap__ ? right.__unwrap__(false) : right).includes(left);
    throw new Error(`Comparison not handled: "${op}"`);
}
async function handleBooleanOperator(node, ctx, builtIns) {
    const op = node.child(1).text;
    const left = await handleNode(node.child(0), ctx, builtIns);
    if (!left && op === "and")
        return left;
    const right = await handleNode(node.child(2), ctx, builtIns);
    if (op === "and")
        return left && right;
    if (op === "or")
        return left || right;
}
async function handleBinaryOperator(node, ctx, builtIns) {
    const op = node.child(1).text;
    const left = await handleNode(node.child(0), ctx, builtIns);
    const right = await handleNode(node.child(2), ctx, builtIns);
    if (left.constructor !== right.constructor) {
        throw new Error(`Can only run ${op} operator on same type.`);
    }
    if (op === "+")
        return left.__add__ ? left.__add__(right) : left + right;
    if (op === "-")
        return left - right;
    if (op === "/")
        return left / right;
    if (op === "//")
        return Math.floor(left / right);
    if (op === "*")
        return left * right;
    if (op === "%")
        return left % right;
    if (op === "&")
        return left & right;
    if (op === "|")
        return left | right;
    if (op === "^")
        return left ^ right;
    if (op === "<<")
        return left << right;
    if (op === ">>")
        return left >> right;
    throw new Error(`Comparison not handled: "${op}"`);
}
async function handleNotOperator(node, ctx, builtIns) {
    check(node.children.length === 2, "Expected 2 children for not operator.");
    check(node.child(0).text === "not", "Expected first child to be 'not'.");
    const value = await handleNode(node.child(1), ctx, builtIns);
    return !value;
}
async function handleUnaryOperator(node, ctx, builtIns) {
    check(node.children.length === 2, "Expected 2 children for not operator.");
    const value = await handleNode(node.child(1), ctx, builtIns);
    const op = node.child(0).text;
    if (op === "-")
        return value * -1;
    console.warn(`Unhandled unary operator: ${op}`);
    return value;
}
async function handleDictionary(node, ctx, builtIns) {
    const dict = new PyDict();
    for (const child of node.children) {
        if (!child || ["{", ",", "}"].includes(child.type))
            continue;
        check(child.type === "pair", "Expected a pair type for dict.");
        const pair = await handleNode(child, ctx, builtIns);
        dict.__put__(pair[0], pair[1]);
    }
    return dict;
}
async function handleDictionaryPair(node, ctx, builtIns) {
    check(node.children.length === 3, "Expected 3 children for dict pair.");
    let varName = await handleNode(node.child(0), ctx, builtIns);
    let varValue = await handleNode(node.child(2), ctx, builtIns);
    check(typeof varName === "string", "Expected varname to be string.");
    return [varName, varValue];
}
class Node {
    constructor(node) {
        this.type = node.type;
        this.text = node.text;
        if (this.type === "ERROR") {
            throw new Error(`Error found in parsing near "${this.text}"`);
        }
        this.children = [];
        for (const child of node.children) {
            this.children.push(new Node(child));
        }
        this.node = node;
    }
    child(index) {
        const child = this.children[index];
        if (!child)
            throw Error(`No child at index ${index}.`);
        return child;
    }
    log(tab = "", showNode = false) {
        console.log(`${tab}--- Node`);
        console.log(`${tab} type: ${this.type}`);
        console.log(`${tab} text: ${this.text}`);
        console.log(`${tab} children:`, this.children);
        if (showNode) {
            console.log(`${tab} node:`, this.node);
        }
    }
}
export class PyTuple {
    constructor(...args) {
        if (args.length === 1 && args[0] instanceof PyTuple) {
            args = args[0].__unwrap__(false);
        }
        if (args.length === 1 && Array.isArray(args[0])) {
            args = [...args[0]];
        }
        this.list = [...args];
    }
    count(v) {
    }
    index() {
    }
    __at__(index) {
        index = this.__get_relative_index__(index);
        return this.list[index];
    }
    __len__() {
        return this.list.length;
    }
    __add__(v) {
        if (!(v instanceof PyTuple)) {
            throw new Error("Can only concatenate tuple to tuple.");
        }
        return new PyTuple(this.__unwrap__(false).concat(v.__unwrap__(false)));
    }
    __put__(index, v) {
        throw new Error("Tuple does not support item assignment");
    }
    __get_relative_index__(index) {
        if (index >= 0) {
            check(this.list.length > index, `Index ${index} out of range.`);
            return index;
        }
        const relIndex = this.list.length + index;
        check(relIndex >= 0, `Index ${index} out of range.`);
        return relIndex;
    }
    __unwrap__(deep = true) {
        var _a;
        const l = [...this.list];
        if (deep) {
            for (let i = 0; i < l.length; i++) {
                l[i] = ((_a = l[i]) === null || _a === void 0 ? void 0 : _a.__unwrap__) ? l[i].__unwrap__(deep) : l[i];
            }
        }
        return l;
    }
}
__decorate([
    Exposed
], PyTuple.prototype, "count", null);
__decorate([
    Exposed
], PyTuple.prototype, "index", null);
export class PyList extends PyTuple {
    append(...args) {
        this.list.push(...args);
    }
    clear() {
        this.list.length = 0;
    }
    copy() {
    }
    count() {
    }
    extend() {
    }
    index() {
    }
    insert() {
    }
    pop() {
    }
    remove() {
    }
    reverse() {
    }
    sort() {
    }
    __add__(v) {
        if (!(v instanceof PyList)) {
            throw new Error("Can only concatenate list to list.");
        }
        return new PyList(this.__unwrap__(false).concat(v.__unwrap__(false)));
    }
    __put__(index, v) {
        index = this.__get_relative_index__(index);
        this.list[index] = v;
    }
}
__decorate([
    Exposed
], PyList.prototype, "append", null);
__decorate([
    Exposed
], PyList.prototype, "clear", null);
__decorate([
    Exposed
], PyList.prototype, "copy", null);
__decorate([
    Exposed
], PyList.prototype, "count", null);
__decorate([
    Exposed
], PyList.prototype, "extend", null);
__decorate([
    Exposed
], PyList.prototype, "index", null);
__decorate([
    Exposed
], PyList.prototype, "insert", null);
__decorate([
    Exposed
], PyList.prototype, "pop", null);
__decorate([
    Exposed
], PyList.prototype, "remove", null);
__decorate([
    Exposed
], PyList.prototype, "reverse", null);
__decorate([
    Exposed
], PyList.prototype, "sort", null);
class PyInt {
}
class PyDict {
    constructor(dict) {
        _PyDict_dict.set(this, void 0);
        __classPrivateFieldSet(this, _PyDict_dict, { ...(dict !== null && dict !== void 0 ? dict : {}) }, "f");
    }
    clear() { }
    copy() { }
    fromkeys() { }
    get(key) {
        return __classPrivateFieldGet(this, _PyDict_dict, "f")[key];
    }
    items() {
        return new PyTuple(Object.entries(__classPrivateFieldGet(this, _PyDict_dict, "f")).map((e) => new PyTuple(e)));
    }
    keys() { }
    pop() { }
    popitem() { }
    setdefault() { }
    update() { }
    values() { }
    __put__(key, v) {
        __classPrivateFieldGet(this, _PyDict_dict, "f")[key] = v;
    }
    __len__() {
        return Object.keys(__classPrivateFieldGet(this, _PyDict_dict, "f")).length;
    }
    __unwrap__(deep = true) {
        var _a;
        const d = { ...__classPrivateFieldGet(this, _PyDict_dict, "f") };
        if (deep) {
            for (let k of Object.keys(d)) {
                d[k] = ((_a = d[k]) === null || _a === void 0 ? void 0 : _a.__unwrap__) ? d[k].__unwrap__(deep) : d[k];
            }
        }
        return d;
    }
}
_PyDict_dict = new WeakMap();
__decorate([
    Exposed
], PyDict.prototype, "clear", null);
__decorate([
    Exposed
], PyDict.prototype, "copy", null);
__decorate([
    Exposed
], PyDict.prototype, "fromkeys", null);
__decorate([
    Exposed
], PyDict.prototype, "get", null);
__decorate([
    Exposed
], PyDict.prototype, "items", null);
__decorate([
    Exposed
], PyDict.prototype, "keys", null);
__decorate([
    Exposed
], PyDict.prototype, "pop", null);
__decorate([
    Exposed
], PyDict.prototype, "popitem", null);
__decorate([
    Exposed
], PyDict.prototype, "setdefault", null);
__decorate([
    Exposed
], PyDict.prototype, "update", null);
__decorate([
    Exposed
], PyDict.prototype, "values", null);
function __unwrap__(...args) {
    var _a;
    for (let i = 0; i < args.length; i++) {
        args[i] = ((_a = args[i]) === null || _a === void 0 ? void 0 : _a.__unwrap__) ? args[i].__unwrap__(true) : args[i];
    }
    return args;
}
function checkAttributeAccessibility(inst, attr) {
    var _a, _b, _c, _d, _e, _f;
    const instType = typeof inst;
    check(instType === "object" || instType === "function", `Instance of type ${instType} does not have attributes.`);
    check(!attr.startsWith("__") && !attr.endsWith("__"), `"${attr}" is not accessible.`);
    const attrType = typeof inst[attr];
    if (attrType === "function") {
        const allowedMethods = (_c = (_b = (_a = inst.constructor) === null || _a === void 0 ? void 0 : _a.__ALLOWED_METHODS__) !== null && _b !== void 0 ? _b : inst.__ALLOWED_METHODS__) !== null && _c !== void 0 ? _c : [];
        check(allowedMethods.includes(attr), `Method ${attr} is not accessible.`);
    }
    else {
        const allowedProps = (_f = (_e = (_d = inst.constructor) === null || _d === void 0 ? void 0 : _d.__ALLOWED_PROPERTIES__) !== null && _e !== void 0 ? _e : inst.__ALLOWED_PROPERTIES__) !== null && _f !== void 0 ? _f : [];
        check(allowedProps.includes(attr), `Property ${attr} is not accessible.`);
    }
}
function maybeWrapValue(value) {
    if (Array.isArray(value)) {
        return new PyList(value);
    }
    return value;
}
function isInt(value) {
    return typeof value === "number" && Math.round(value) === value;
}
function isIntLike(value) {
    let is = isInt(value);
    if (!is) {
        is = typeof value === "string" && !!/^\d+$/.exec(value);
    }
    return is;
}
export function Exposed(target, key) {
    const descriptor = Object.getOwnPropertyDescriptor(target, key);
    if (typeof (descriptor === null || descriptor === void 0 ? void 0 : descriptor.value) === "function") {
        target.constructor.__ALLOWED_METHODS__ = target.constructor.__ALLOWED_METHODS__ || [];
        target.constructor.__ALLOWED_METHODS__.push(key);
    }
    else {
        target.constructor.__ALLOWED_PROPERTIES__ = target.constructor.__ALLOWED_PROPERTIES__ || [];
        target.constructor.__ALLOWED_PROPERTIES__.push(key);
    }
}
