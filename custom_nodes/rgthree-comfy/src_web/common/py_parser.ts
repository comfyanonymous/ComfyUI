/**
 * @fileoverview An AST executor using the TreeSitter parser to parse python-like code and execute
 * in JS. This parser is self-contained and isolated from other parts of the app (like Comfy-UI
 * specific types, etc). Instead, additional handlers, builtins, and types can be passed into the
 * pure functions below.
 */
import type {Parser, Node as TreeSitterNode, Tree} from "web-tree-sitter";

import {check, deepFreeze} from "./shared_utils.js";

// Hacky memoization because I don't feel like writing a decorator.
const MEMOIZED = {parser: null as unknown as Parser};

interface Dict extends Object {
  [k: string]: unknown;
}
interface ExecutionContextData extends Object {
  [k: string]: unknown;
}
class ExecuteContext implements ExecutionContextData {
  [k: string]: unknown;

  constructor(existing: Object = {}) {
    Object.assign(this, !!window.structuredClone ? structuredClone(existing) : {...existing});
  }
}
class InitialExecuteContext extends ExecuteContext {}

type NodeHandlerArgs = [ExecutionContextData, BuiltInFns];
type NodeHandler = (node: Node, ...args: NodeHandlerArgs) => Promise<any>;

const TYPE_TO_HANDLER = new Map<string, NodeHandler>([
  ["module", handleChildren],
  ["expression_statement", handleChildren],
  ["interpolation", handleInterpolation],
  ["block", handleChildren], // Block of code, like in a for loop

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

  // Types
  ["integer", handleNumber],
  ["float", handleNumber],
  ["string", handleString],
  ["tuple", handleList],
  ["list", handleList],
  ["dictionary", handleDictionary],
  ["pair", handleDictionaryPair],
  ["true", async (...args: any[]) => true],
  ["false", async (...args: any[]) => false],
]);

type BuiltInFn = {fn: Function};
type BuiltInFns = {[key: string]: BuiltInFn};

const DEFAULT_BUILT_INS: BuiltInFns = {
  round: {fn: (n: any) => Math.round(Number(n))},
  ceil: {fn: (n: any) => Math.ceil(Number(n))},
  floor: {fn: (n: any) => Math.floor(Number(n))},
  // Function(name="sqrt", call=math.sqrt, args=(1, 1)),
  // Function(name="min", call=min, args=(2, None)),
  // Function(name="max", call=max, args=(2, None)),
  // Function(name=".random_int", call=random.randint, args=(2, 2)),
  // Function(name=".random_choice", call=random.choice, args=(1, 1)),
  // Function(name=".random_seed", call=random.seed, args=(1, 1)),
  // Function(name="re", call=re.compile, args=(1, 1)),
  len: {fn: (n: any) => n?.__len__?.() ?? n?.length},
  // Function(name="enumerate", call=enumerate, args=(1, 1)),
  // Function(name="range", call=range, args=(1, 3)),

  // Types
  int: {fn: (n: any) => Math.floor(Number(n))},
  float: {fn: (n: any) => Number(n)},
  str: {fn: (n: any) => String(n)},
  bool: {fn: (n: any) => !!n},
  list: {fn: (tupl: any[] = []) => new PyList(tupl)},
  tuple: {fn: (list: any[] = []) => new PyTuple(list)},
  dict: {fn: (dict: Dict = {}) => new PyDict(dict)},

  // Special
  dir: {fn: (...args: any[]) => console.dir(...__unwrap__(...args))},
  print: {fn: (...args: any[]) => console.log(...__unwrap__(...args))},
  log: {fn: (...args: any[]) => console.log(...__unwrap__(...args))},
};

/**
 * The main entry point to parse code.
 */
export async function execute(
  code: string,
  ctx: ExecutionContextData,
  additionalBuiltins?: BuiltInFns,
) {
  const builtIns = deepFreeze({...DEFAULT_BUILT_INS, ...(additionalBuiltins ?? {})});
  // When we start the execution, we create an InitialExecuteContext as an instance so we can check
  // if we're the initial, global context during execution (as we may pass in a new context in the
  // like if evaluating a list comprehension, or setting on an object).
  ctx = new InitialExecuteContext(ctx);

  const root = (await parse(code)).rootNode;
  const value = await handleNode(new Node(root), ctx, builtIns);

  console.log("=====");
  console.log(`value`, value?.__unwrap__?.() ?? value);
  console.log("context", ctx);

  return value;
}

/**
 * Parses a code string to a `Tree`.
 */
async function parse(code: string): Promise<Tree> {
  if (!MEMOIZED.parser) {
    // @ts-ignore - Path is rewritten.
    const TreeSitter = (await import("rgthree/lib/tree-sitter.js")) as TreeSitter;
    await TreeSitter.Parser.init();
    const lang = await TreeSitter.Language.load("rgthree/lib/tree-sitter-python.wasm");
    MEMOIZED.parser = new TreeSitter.Parser() as Parser;
    MEMOIZED.parser.setLanguage(lang);
  }
  return MEMOIZED.parser.parse(code)!;
}

/**
 * The generic node handler, calls out to specific handlers based on the node type. This is
 * recursively called from other handlers.
 */
async function handleNode(
  node: Node,
  ctx: ExecutionContextData,
  builtIns: BuiltInFns,
): Promise<any> {
  const type = node.type as string;

  // If we have a returned value, then just return it, which should recursively settle.
  if (ctx.hasOwnProperty("__returned__")) return ctx["__returned__"];

  // console.log(`-----`);
  // console.log(`eval_node`);
  // console.log(`type: ${type}`);
  // console.log(`text: ${node.text}`);
  // console.log(`children: ${node.children?.length ?? 0}`);
  // console.log(ctx);
  // console.log(node);

  const handler = TYPE_TO_HANDLER.get(type);
  check(handler, "Unhandled type: " + type, node);
  return handler(node, ctx, builtIns);
}

/**
 * Generic handler to loop over children of a node, and evaluate each.
 */
async function handleChildren(node: Node, ctx: ExecutionContextData, builtIns: BuiltInFns) {
  let lastValue = null;
  for (const child of node.children) {
    if (!child) continue;
    lastValue = await handleNode(child, ctx, builtIns);
  }
  return lastValue;
}

/**
 * Swallows the execution. Likely just to allow development.
 */
async function handleSwallow(node: Node, ctx: ExecutionContextData, builtIns: BuiltInFns) {
  // No op
}

/**
 * Handles a return statement.
 */
async function handleReturn(node: Node, ctx: ExecutionContextData, builtIns: BuiltInFns) {
  const value = node.children.length > 1 ? handleNode(node.child(1), ctx, builtIns) : undefined;
  // Mark that we have a return value, as we may be deeper in evaluation, like going through an
  // if condition's body.
  ctx["__returned__"] = value;
  return value;
}

/**
 * Handles the retrieval of a variable identifier, already be set in the context.
 */
async function handleIdentifier(node: Node, ctx: ExecutionContextData, builtIns: BuiltInFns) {
  let value = ctx[node.text];
  if (value === undefined) {
    value = builtIns[node.text]?.fn ?? undefined;
  }
  return maybeWrapValue(value);
}

async function handleAttribute(node: Node, ctx: ExecutionContextData, builtIns: BuiltInFns) {
  const children = node.children;
  check(children.length === 3, "Expected 3 children for attribute.");
  check(children[1]!.type === ".", "Expected middle child to be '.' for attribute.");
  const inst = await handleNode(children[0]!, ctx, builtIns);
  // const attr = await handleNode(node.child(2), inst);
  // console.log('handleAttribute', inst, attr);
  const attr = children[2]!.text;
  checkAttributeAccessibility(inst, attr);
  let attribute = maybeWrapValue(inst[attr]);
  // check(attribute !== undefined, `"${attr}" not found on instance of type ${typeof inst}.`);
  // If the attribute is a function, then bind it to the instance.
  return typeof attribute === "function" ? attribute.bind(inst) : attribute;
}

async function handleSubscript(node: Node, ctx: ExecutionContextData, builtIns: BuiltInFns) {
  const children = node.children;
  check(children.length === 4, "Expected 4 children for subscript.");
  check(children[1]!.type === "[", "Expected 2nd child to be '[' for subscript.");
  check(children[3]!.type === "]", "Expected 4thd child to be ']' for subscript.");
  const inst = await handleNode(children[0]!, ctx, builtIns);
  const attr = await handleNode(children[2]!, ctx, builtIns);
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

/**
 * Handles the assignment.
 */
async function handleAssignment(node: Node, ctx: ExecutionContextData, builtIns: BuiltInFns) {
  check(
    node.children.length === 3,
    "Expected 3 children for assignment: identifier/attr, =, and value.",
  );
  check(node.children[1]!.type === "=", "Expected middle child to be an '='.");

  let right = await handleNode(node.children[2]!, ctx, builtIns);
  const leftNode = node.children[0]!;
  let leftObj: any = ctx;
  let leftProp: string | number = "";
  if (leftNode.type === "identifier") {
    leftProp = leftNode.text;
  } else if (leftNode.type === "attribute") {
    leftObj = await handleNode(leftNode.children[0]!, ctx, builtIns);
    check(
      leftNode.children[2]!.type === "identifier",
      "Expected left hand assignment attribute to be an identifier.",
      leftNode,
    );
    leftProp = leftNode.children[2]!.text;
  } else if (leftNode.type === "subscript") {
    leftObj = await handleNode(leftNode.children[0]!, ctx, builtIns);
    check(leftNode.children[1]!.type === "[");
    check(leftNode.children[3]!.type === "]");
    leftProp = await handleNode(leftNode.children[2]!, ctx, builtIns);
  } else {
    throw new Error(`Unhandled left-hand assignement type: ${leftNode.type}`);
  }

  if (leftProp == null) {
    throw new Error(`No property to assign value`);
  }
  // If we're a PyTuple or extended from, then try add like a list (PyTuple will fail, PyList will
  // allow).
  if (leftObj instanceof PyTuple) {
    check(isInt(leftProp), "Expected an int for list assignment");
    leftObj.__put__(leftProp, right);
  } else if (leftObj instanceof PyDict) {
    check(typeof leftProp === "string", "Expected a string for dict assignment");
    leftObj.__put__(leftProp, right);
  } else {
    check(typeof leftProp === "string", "Expected a string for object assignment");
    // InitialExecutionContext can have anything added, otherwise we're a specific context and
    // should check for attribute accessibility.
    if (!(leftObj instanceof InitialExecuteContext)) {
      checkAttributeAccessibility(leftObj, leftProp);
    }
    leftObj[leftProp] = right;
  }
  return right;
}

/**
 * Handles a named expression, like assigning a var in a list comprehension with:
 * `[name for node in node_list if (name := node.name)]`
 */
async function handleNamedExpression(node: Node, ctx: ExecutionContextData, builtIns: BuiltInFns) {
  check(node.children.length === 3, "Expected three children for named expression.");
  check(node.child(0).type === "identifier", "Expected identifier first in named expression.");
  const varName = node.child(0).text;
  ctx[varName] = await handleNode(node.child(2), ctx, builtIns);
  return maybeWrapValue(ctx[varName]);
}

/**
 * Handles a function call.
 */
async function handleCall(node: Node, ctx: ExecutionContextData, builtIns: BuiltInFns) {
  check(node.children.length === 2, "Expected 2 children for call, identifier and arguments.");
  const fn = await handleNode(node.children[0]!, ctx, builtIns);
  const args = await handleNode(node.children[1]!, ctx, builtIns);
  console.log("handleCall", fn, args);
  return fn(...args);
}

async function handleArgumentsList(node: Node, ctx: ExecutionContextData, builtIns: BuiltInFns) {
  const args = (await handleList(node, ctx, builtIns)).__unwrap__(false);
  return [...args];
}

/**
 * Handles a simple for...in loop.
 */
async function handleForStatement(node: Node, ctx: ExecutionContextData, builtIns: BuiltInFns) {
  const childs = node.children;
  check(childs.length === 6);
  check(childs[4]!.type === ":");
  check(childs[5]!.type === "block");
  await helperGetLoopForIn(node, ctx, builtIns, async (forCtx) => {
    await handleNode(childs[5]!, forCtx, builtIns);
  });
}

async function handleListComprehension(
  node: Node,
  ctx: ExecutionContextData,
  builtIns: BuiltInFns,
) {
  // Create a new context that we don't want to pollute our outer one.
  const finalList = new PyList();
  const newCtx = {...ctx};

  let finalEntryNode;
  const loopNodes: {forIn: Node; if?: Node}[] = [];

  for (const child of node.children) {
    if (!child || ["[", "]"].includes(child.type)) continue;
    if (child.type === "identifier" || child.type === "attribute") {
      if (finalEntryNode) {
        throw Error("Already have a list comprehension finalEntryNode.");
      }
      finalEntryNode = child;
    } else if (child.type === "for_in_clause") {
      loopNodes.push({forIn: child});
    } else if (child.type === "if_clause") {
      loopNodes[loopNodes.length - 1]!["if"] = child;
    }
  }
  if (!finalEntryNode) {
    throw Error("No list comprehension finalEntryNode.");
  }

  console.log(`handleListComprehension.loopNodes`, loopNodes);

  const handleLoop = async (loopNodes: {forIn: Node; if?: Node}[]) => {
    const loopNode = loopNodes.shift()!;
    await helperGetLoopForIn(
      loopNode.forIn,
      newCtx,
      builtIns,
      async (forCtx) => {
        if (loopNode.if) {
          const ifNode = loopNode.if;
          check(ifNode.children.length === 2, "Expected 2 children for if_clause.");
          check(ifNode.child(0).text === "if", "Expected first child to be 'if'.");
          const good = await handleNode(ifNode.child(1), forCtx, builtIns);
          if (!good) return;
        }
        Object.assign(newCtx, forCtx);
        if (loopNodes.length) {
          await handleLoop(loopNodes);
        } else {
          finalList.append(await handleNode(finalEntryNode, newCtx, builtIns));
        }
      },
      () => ({...newCtx}),
    );
    loopNodes.unshift(loopNode);
  };

  await handleLoop(loopNodes);
  return finalList;
}

/**
 * Handles the identifiers, iterable, and initial looping with context setting. Handles both simple
 * identifiers (like `for item in items`) or a pattern list (like `for key, val in mydict.items()`).
 *
 * @param eachFn The function to call for each iteration. Will be passed the current context with
 *     the identifiers assigned.
 * @param provideForCtx An optional function that can provide an `ctx`. If not supplied the passed
 *     `ctx` param will be used. This is useful for providing a new ctx to use for cases like an
 *     if condition in a list comprhension where we don't want to add to the current context unless
 *     the condition is met.
 */
async function helperGetLoopForIn(
  node: Node,
  ctx: ExecutionContextData,
  builtIns: BuiltInFns,
  eachFn: (forCtx: ExecutionContextData) => Promise<void>,
  provideForCtx?: () => ExecutionContextData,
) {
  const childs = node.children;
  check(childs.length >= 3);
  check(childs[0]!.type === "for");
  check(
    ["identifier", "pattern_list"].includes(childs[1]!.type),
    "Expected identifier for for loop.",
  );
  check(childs[2]!.type === "in");

  let identifiers: string[];
  if (childs[1]!.type === "identifier") {
    // identifier: for k in my_list
    identifiers = [childs[1]!.text];
  } else {
    // pattern_list: for k,v in my_dict.items()
    identifiers = childs[1]!.children
      .map((n) => {
        if (n.type === ",") return null;
        check(n.type === "identifier");
        return node.text;
      })
      .filter((n) => n != null);
  }
  const iterable = await handleNode(childs[3]!, ctx, builtIns);
  check(iterable instanceof PyTuple, "Expected for loop instance to be a list/tuple.");

  for (const item of iterable.__unwrap__(false)) {
    const forCtx = provideForCtx?.() ?? ctx;
    if (identifiers.length === 1) {
      forCtx[identifiers[0]!] = item;
    } else {
      check(
        Array.isArray(item) && identifiers.length === item.length,
        "Expected iterable to be a list, like using dict.items()",
      );
      for (let i = 0; i < identifiers.length; i++) {
        forCtx[identifiers[i]!] = item[i];
      }
    }
    await eachFn(forCtx);
  }
}

async function handleNumber(node: Node, ctx: ExecutionContextData, builtIns: BuiltInFns) {
  return Number(node.text);
}

async function handleString(node: Node, ctx: ExecutionContextData, builtIns: BuiltInFns) {
  // check(node.children.length === 3, "Expected 3 children for str (quotes and value).");
  let str = "";
  for (const child of node.children) {
    if (!child || ["string_start", "string_end"].includes(child.type)) continue;
    if (child.type === "string_content") {
      str += child.text;
    } else if (child.type === "interpolation") {
      check(child.children.length === 3, "Expected interpolation");
      str += await handleNode(child, ctx, builtIns);
    }
  }
  return str;
}

async function handleInterpolation(node: Node, ...args: NodeHandlerArgs) {
  check(node.children.length === 3, "Expected interpolation to be three nodes length.");
  check(
    node.children[0]!.type === "{" && node.children[2]!.type === "}",
    'Expected interpolation to be wrapped in "{" and "}".',
  );
  return await handleNode(node.children[1]!, ...args);
}

async function handleList(node: Node, ctx: ExecutionContextData, builtIns: BuiltInFns) {
  const list = [];
  for (const child of node.children) {
    if (!child || ["(", "[", ",", "]", ")"].includes(child.type)) continue;
    list.push(await handleNode(child, ctx, builtIns));
  }
  if (node.type === "tuple") {
    return new PyTuple(list);
  }
  return new PyList(list);
}

async function handleComparisonOperator(
  node: Node,
  ctx: ExecutionContextData,
  builtIns: BuiltInFns,
) {
  const op = node.child(1).text;
  const left = await handleNode(node.child(0), ctx, builtIns);
  const right = await handleNode(node.child(2), ctx, builtIns);
  if (op === "==") return left === right; // Python '==' is equiv to '===' in JS.
  if (op === "!=") return left !== right;
  if (op === ">") return left > right;
  if (op === ">=") return left >= right;
  if (op === "<") return left < right;
  if (op === "<=") return left <= right;
  if (op === "in") return (right.__unwrap__ ? right.__unwrap__(false) : right).includes(left);
  throw new Error(`Comparison not handled: "${op}"`);
}
async function handleBooleanOperator(node: Node, ctx: ExecutionContextData, builtIns: BuiltInFns) {
  const op = node.child(1).text;
  const left = await handleNode(node.child(0), ctx, builtIns);
  // If we're an AND and already false, then don't even evaluate the right.
  if (!left && op === "and") return left;
  const right = await handleNode(node.child(2), ctx, builtIns);
  if (op === "and") return left && right;
  if (op === "or") return left || right;
}

async function handleBinaryOperator(node: Node, ctx: ExecutionContextData, builtIns: BuiltInFns) {
  const op = node.child(1).text;
  const left = await handleNode(node.child(0), ctx, builtIns);
  const right = await handleNode(node.child(2), ctx, builtIns);
  if (left.constructor !== right.constructor) {
    throw new Error(`Can only run ${op} operator on same type.`);
  }
  if (op === "+") return left.__add__ ? left.__add__(right) : left + right;
  if (op === "-") return left - right;
  if (op === "/") return left / right;
  if (op === "//") return Math.floor(left / right);
  if (op === "*") return left * right;
  if (op === "%") return left % right;
  if (op === "&") return left & right;
  if (op === "|") return left | right;
  if (op === "^") return left ^ right;
  if (op === "<<") return left << right;
  if (op === ">>") return left >> right;
  throw new Error(`Comparison not handled: "${op}"`);
}

async function handleNotOperator(node: Node, ctx: ExecutionContextData, builtIns: BuiltInFns) {
  check(node.children.length === 2, "Expected 2 children for not operator.");
  check(node.child(0).text === "not", "Expected first child to be 'not'.");
  const value = await handleNode(node.child(1), ctx, builtIns);
  return !value;
}

async function handleUnaryOperator(node: Node, ctx: ExecutionContextData, builtIns: BuiltInFns) {
  check(node.children.length === 2, "Expected 2 children for not operator.");
  const value = await handleNode(node.child(1), ctx, builtIns);
  const op = node.child(0).text;
  if (op === "-") return value * -1;
  console.warn(`Unhandled unary operator: ${op}`);
  return value;
}

async function handleDictionary(node: Node, ctx: ExecutionContextData, builtIns: BuiltInFns) {
  const dict = new PyDict();
  for (const child of node.children) {
    if (!child || ["{", ",", "}"].includes(child.type)) continue;
    check(child.type === "pair", "Expected a pair type for dict.");
    const pair = await handleNode(child, ctx, builtIns);
    dict.__put__(pair[0], pair[1]);
  }
  return dict;
}

async function handleDictionaryPair(node: Node, ctx: ExecutionContextData, builtIns: BuiltInFns) {
  check(node.children.length === 3, "Expected 3 children for dict pair.");
  let varName = await handleNode(node.child(0)!, ctx, builtIns);
  let varValue = await handleNode(node.child(2)!, ctx, builtIns);
  check(typeof varName === "string", "Expected varname to be string.");
  return [varName, varValue];
}

/**
 * Wraps some common functionality of a TreeSitterNode.
 */
class Node {
  type: string;
  text: string;
  children: Node[];
  private node: TreeSitterNode;

  constructor(node: TreeSitterNode) {
    this.type = node.type;
    this.text = node.text;
    if (this.type === "ERROR") {
      throw new Error(`Error found in parsing near "${this.text}"`);
    }
    this.children = [];
    for (const child of node.children) {
      this.children.push(new Node(child!));
    }
    this.node = node;
  }

  child(index: number): Node {
    const child = this.children[index];
    if (!child) throw Error(`No child at index ${index}.`);
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

/**
 * A type that mimics a Python Tuple.
 */
export class PyTuple {
  protected list: any[];
  constructor(...args: any[]) {
    if (args.length === 1 && args[0] instanceof PyTuple) {
      args = args[0].__unwrap__(false);
    }
    if (args.length === 1 && Array.isArray(args[0])) {
      args = [...args[0]];
    }
    this.list = [...args];
  }

  @Exposed count(v: any) {
    // TODO
  }

  @Exposed index() {
    // TODO
  }

  __at__(index: number) {
    index = this.__get_relative_index__(index);
    return this.list[index];
  }

  __len__() {
    return this.list.length;
  }

  __add__(v: any) {
    if (!(v instanceof PyTuple)) {
      throw new Error("Can only concatenate tuple to tuple.");
    }
    return new PyTuple(this.__unwrap__(false).concat(v.__unwrap__(false)));
  }

  /** Puts the value to the current, existing index. Not available for Tuple. */
  __put__(index: number, v: any) {
    throw new Error("Tuple does not support item assignment");
  }

  /** Gets the index for the current list, with negative index support. Throws if out of range. */
  protected __get_relative_index__(index: number) {
    if (index >= 0) {
      check(this.list.length > index, `Index ${index} out of range.`);
      return index;
    }
    const relIndex = this.list.length + index;
    check(relIndex >= 0, `Index ${index} out of range.`);
    return relIndex;
  }

  /**
   * Recursively unwraps the PyTuple returning an Array.
   */
  __unwrap__(deep = true) {
    const l = [...this.list];
    if (deep) {
      for (let i = 0; i < l.length; i++) {
        l[i] = l[i]?.__unwrap__ ? l[i].__unwrap__(deep) : l[i];
      }
    }
    return l;
  }

  // a = [
  //   "__add__",
  //   "__class__",
  //   "__class_getitem__",
  //   "__contains__",
  //   "__delattr__",
  //   "__dir__",
  //   "__doc__",
  //   "__eq__",
  //   "__format__",
  //   "__ge__",
  //   "__getattribute__",
  //   "__getitem__",
  //   "__getnewargs__",
  //   "__gt__",
  //   "__hash__",
  //   "__init__",
  //   "__init_subclass__",
  //   "__iter__",
  //   "__le__",
  //   "__len__",
  //   "__lt__",
  //   "__mul__",
  //   "__ne__",
  //   "__new__",
  //   "__reduce__",
  //   "__reduce_ex__",
  //   "__repr__",
  //   "__rmul__",
  //   "__setattr__",
  //   "__sizeof__",
  //   "__str__",
  //   "__subclasshook__",
  //   "count",
  //   "index",
  // ];
}

/**
 * A type that mimics a Python List.
 */
export class PyList extends PyTuple {
  @Exposed append(...args: any[]) {
    this.list.push(...args);
  }

  @Exposed clear() {
    this.list.length = 0;
  }

  @Exposed copy() {
    // TODO
  }

  @Exposed override count() {
    // TODO
  }
  @Exposed extend() {
    // TODO
  }
  @Exposed override index() {
    // TODO
  }
  @Exposed insert() {
    // TODO
  }
  @Exposed pop() {
    // TODO
  }
  @Exposed remove() {
    // TODO
  }
  @Exposed reverse() {
    // TODO
  }
  @Exposed sort() {
    // TODO
  }

  override __add__(v: any) {
    if (!(v instanceof PyList)) {
      throw new Error("Can only concatenate list to list.");
    }
    return new PyList(this.__unwrap__(false).concat(v.__unwrap__(false)));
  }

  /** Assigns an element to the current, existing index. Overriden for support on lists. */
  override __put__(index: number, v: any) {
    index = this.__get_relative_index__(index);
    this.list[index] = v;
  }

  // aa = [
  //   "__add__",
  //   "__class__",
  //   "__class_getitem__",
  //   "__contains__",
  //   "__delattr__",
  //   "__delitem__",
  //   "__dir__",
  //   "__doc__",
  //   "__eq__",
  //   "__format__",
  //   "__ge__",
  //   "__getattribute__",
  //   "__getitem__",
  //   "__gt__",
  //   "__hash__",
  //   "__iadd__",
  //   "__imul__",
  //   "__init__",
  //   "__init_subclass__",
  //   "__iter__",
  //   "__le__",
  //   "__len__",
  //   "__lt__",
  //   "__mul__",
  //   "__ne__",
  //   "__new__",
  //   "__reduce__",
  //   "__reduce_ex__",
  //   "__repr__",
  //   "__reversed__",
  //   "__rmul__",
  //   "__setattr__",
  //   "__setitem__",
  //   "__sizeof__",
  //   "__str__",
  //   "__subclasshook__",
  // ];
}

class PyInt {}

class PyDict {
  #dict: {[key: string]: any};
  constructor(dict?: {[key: string]: any}) {
    this.#dict = {...(dict ?? {})};
  }

  @Exposed clear() {} // Removes all the elements from the dictionary
  @Exposed copy() {} // Returns a copy of the dictionary
  @Exposed fromkeys() {} // Returns a dictionary with the specified keys and value
  /** Returns the value of the specified key. */
  @Exposed get(key: string) {
    return this.#dict[key];
  }
  /** Returns a list containing a tuple for each key value pair. */
  @Exposed items() {
    return new PyTuple(Object.entries(this.#dict).map((e) => new PyTuple(e)));
  }
  @Exposed keys() {} // Returns a list containing the dictionary's keys
  @Exposed pop() {} // Removes the element with the specified key
  @Exposed popitem() {} // Removes the last inserted key-value pair
  @Exposed setdefault() {} // Returns the value of the specified key. If the key does not exist: insert the key, with the specified value
  @Exposed update() {} // Updates the dictionary with the specified key-value pairs
  @Exposed values() {} // Returns a list of all the values in the dictionary

  __put__(key: string, v: any) {
    this.#dict[key] = v;
  }

  __len__() {
    return Object.keys(this.#dict).length;
  }

  // a = [
  //   "__class__",
  //   "__class_getitem__",
  //   "__contains__",
  //   "__delattr__",
  //   "__delitem__",
  //   "__dir__",
  //   "__doc__",
  //   "__eq__",
  //   "__format__",
  //   "__ge__",
  //   "__getattribute__",
  //   "__getitem__",
  //   "__gt__",
  //   "__hash__",
  //   "__init__",
  //   "__init_subclass__",
  //   "__ior__",
  //   "__iter__",
  //   "__le__",
  //   "__lt__",
  //   "__ne__",
  //   "__new__",
  //   "__or__",
  //   "__reduce__",
  //   "__reduce_ex__",
  //   "__repr__",
  //   "__reversed__",
  //   "__ror__",
  //   "__setattr__",
  //   "__setitem__",
  //   "__sizeof__",
  //   "__str__",
  //   "__subclasshook__",
  // ];

  /**
   * Recursively unwraps the PyDict returning an Object.
   */
  __unwrap__(deep = true) {
    const d = {...this.#dict};
    if (deep) {
      for (let k of Object.keys(d)) {
        d[k] = d[k]?.__unwrap__ ? d[k].__unwrap__(deep) : d[k];
      }
    }
    return d;
  }
}

/**
 * Deeply unwraps a list of values.
 */
function __unwrap__(...args: any[]) {
  for (let i = 0; i < args.length; i++) {
    args[i] = args[i]?.__unwrap__ ? args[i].__unwrap__(true) : args[i];
  }
  return args;
}

/**
 * Checks if access to the attribute/method is allowed.
 */
function checkAttributeAccessibility(inst: any, attr: string) {
  const instType = typeof inst;
  check(
    instType === "object" || instType === "function",
    `Instance of type ${instType} does not have attributes.`,
  );

  // If the attr starts and ends with a "__" then consider it unaccessible.
  check(!attr.startsWith("__") && !attr.endsWith("__"), `"${attr}" is not accessible.`);

  const attrType = typeof inst[attr];
  if (attrType === "function") {
    const allowedMethods = inst.constructor?.__ALLOWED_METHODS__ ?? inst.__ALLOWED_METHODS__ ?? [];
    check(allowedMethods.includes(attr), `Method ${attr} is not accessible.`);
  } else {
    const allowedProps =
      inst.constructor?.__ALLOWED_PROPERTIES__ ?? inst.__ALLOWED_PROPERTIES__ ?? [];
    check(allowedProps.includes(attr), `Property ${attr} is not accessible.`);
  }
}

function maybeWrapValue(value: any) {
  if (Array.isArray(value)) {
    return new PyList(value);
  }
  return value;
}

function isInt(value: any): value is number {
  return typeof value === "number" && Math.round(value) === value;
}

function isIntLike(value: any): boolean {
  let is = isInt(value);
  if (!is) {
    is = typeof value === "string" && !!/^\d+$/.exec(value);
  }
  return is;
}

/**
 * An experimental decorator to add allowed properties and methods to an instance. Decorated
 * properties and methods on a class, and they'll be added to a static __ALLOWED_PROPERTIES__ and
 * __ALLOWED_METHODS__ lists, which can then be checked while parsing to ensure entered code
 * cannot end up calling something more.
 *
 * Note: The decorator does no work on static members; only on instance properties, methods, and
 *   getters (or setters). If you wish to allow access to only a getter and not setter, then you'll
 *   need not define the setter (or vice-versa), as adding `@Exposed` to a getter/setter decorates
 *   the property entirely, not just that individual getter/setter.
 */
export function Exposed(target: any, key: string) {
  const descriptor = Object.getOwnPropertyDescriptor(target, key);
  if (typeof descriptor?.value === "function") {
    target.constructor.__ALLOWED_METHODS__ = target.constructor.__ALLOWED_METHODS__ || [];
    target.constructor.__ALLOWED_METHODS__.push(key);
  } else {
    target.constructor.__ALLOWED_PROPERTIES__ = target.constructor.__ALLOWED_PROPERTIES__ || [];
    target.constructor.__ALLOWED_PROPERTIES__.push(key);
  }
}
