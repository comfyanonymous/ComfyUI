"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.arrowFunctionToExpression = arrowFunctionToExpression;
exports.ensureBlock = ensureBlock;
exports.toComputedKey = toComputedKey;
exports.unwrapFunctionEnvironment = unwrapFunctionEnvironment;
var _t = require("@babel/types");
var _helperEnvironmentVisitor = require("@babel/helper-environment-visitor");
var _helperFunctionName = require("@babel/helper-function-name");
var _visitors = require("../visitors.js");
const {
  arrowFunctionExpression,
  assignmentExpression,
  binaryExpression,
  blockStatement,
  callExpression,
  conditionalExpression,
  expressionStatement,
  identifier,
  isIdentifier,
  jsxIdentifier,
  logicalExpression,
  LOGICAL_OPERATORS,
  memberExpression,
  metaProperty,
  numericLiteral,
  objectExpression,
  restElement,
  returnStatement,
  sequenceExpression,
  spreadElement,
  stringLiteral,
  super: _super,
  thisExpression,
  toExpression,
  unaryExpression
} = _t;
function toComputedKey() {
  let key;
  if (this.isMemberExpression()) {
    key = this.node.property;
  } else if (this.isProperty() || this.isMethod()) {
    key = this.node.key;
  } else {
    throw new ReferenceError("todo");
  }
  if (!this.node.computed) {
    if (isIdentifier(key)) key = stringLiteral(key.name);
  }
  return key;
}
function ensureBlock() {
  const body = this.get("body");
  const bodyNode = body.node;
  if (Array.isArray(body)) {
    throw new Error("Can't convert array path to a block statement");
  }
  if (!bodyNode) {
    throw new Error("Can't convert node without a body");
  }
  if (body.isBlockStatement()) {
    return bodyNode;
  }
  const statements = [];
  let stringPath = "body";
  let key;
  let listKey;
  if (body.isStatement()) {
    listKey = "body";
    key = 0;
    statements.push(body.node);
  } else {
    stringPath += ".body.0";
    if (this.isFunction()) {
      key = "argument";
      statements.push(returnStatement(body.node));
    } else {
      key = "expression";
      statements.push(expressionStatement(body.node));
    }
  }
  this.node.body = blockStatement(statements);
  const parentPath = this.get(stringPath);
  body.setup(parentPath, listKey ? parentPath.node[listKey] : parentPath.node, listKey, key);
  return this.node;
}
{
  exports.arrowFunctionToShadowed = function () {
    if (!this.isArrowFunctionExpression()) return;
    this.arrowFunctionToExpression();
  };
}
function unwrapFunctionEnvironment() {
  if (!this.isArrowFunctionExpression() && !this.isFunctionExpression() && !this.isFunctionDeclaration()) {
    throw this.buildCodeFrameError("Can only unwrap the environment of a function.");
  }
  hoistFunctionEnvironment(this);
}
function setType(path, type) {
  path.node.type = type;
}
function arrowFunctionToExpression({
  allowInsertArrow = true,
  allowInsertArrowWithRest = allowInsertArrow,
  noNewArrows = !(_arguments$ => (_arguments$ = arguments[0]) == null ? void 0 : _arguments$.specCompliant)()
} = {}) {
  if (!this.isArrowFunctionExpression()) {
    throw this.buildCodeFrameError("Cannot convert non-arrow function to a function expression.");
  }
  const {
    thisBinding,
    fnPath: fn
  } = hoistFunctionEnvironment(this, noNewArrows, allowInsertArrow, allowInsertArrowWithRest);
  fn.ensureBlock();
  setType(fn, "FunctionExpression");
  if (!noNewArrows) {
    const checkBinding = thisBinding ? null : fn.scope.generateUidIdentifier("arrowCheckId");
    if (checkBinding) {
      fn.parentPath.scope.push({
        id: checkBinding,
        init: objectExpression([])
      });
    }
    fn.get("body").unshiftContainer("body", expressionStatement(callExpression(this.hub.addHelper("newArrowCheck"), [thisExpression(), checkBinding ? identifier(checkBinding.name) : identifier(thisBinding)])));
    fn.replaceWith(callExpression(memberExpression((0, _helperFunctionName.default)(this, true) || fn.node, identifier("bind")), [checkBinding ? identifier(checkBinding.name) : thisExpression()]));
    return fn.get("callee.object");
  }
  return fn;
}
const getSuperCallsVisitor = (0, _visitors.merge)([{
  CallExpression(child, {
    allSuperCalls
  }) {
    if (!child.get("callee").isSuper()) return;
    allSuperCalls.push(child);
  }
}, _helperEnvironmentVisitor.default]);
function hoistFunctionEnvironment(fnPath, noNewArrows = true, allowInsertArrow = true, allowInsertArrowWithRest = true) {
  let arrowParent;
  let thisEnvFn = fnPath.findParent(p => {
    if (p.isArrowFunctionExpression()) {
      var _arrowParent;
      (_arrowParent = arrowParent) != null ? _arrowParent : arrowParent = p;
      return false;
    }
    return p.isFunction() || p.isProgram() || p.isClassProperty({
      static: false
    }) || p.isClassPrivateProperty({
      static: false
    });
  });
  const inConstructor = thisEnvFn.isClassMethod({
    kind: "constructor"
  });
  if (thisEnvFn.isClassProperty() || thisEnvFn.isClassPrivateProperty()) {
    if (arrowParent) {
      thisEnvFn = arrowParent;
    } else if (allowInsertArrow) {
      fnPath.replaceWith(callExpression(arrowFunctionExpression([], toExpression(fnPath.node)), []));
      thisEnvFn = fnPath.get("callee");
      fnPath = thisEnvFn.get("body");
    } else {
      throw fnPath.buildCodeFrameError("Unable to transform arrow inside class property");
    }
  }
  const {
    thisPaths,
    argumentsPaths,
    newTargetPaths,
    superProps,
    superCalls
  } = getScopeInformation(fnPath);
  if (inConstructor && superCalls.length > 0) {
    if (!allowInsertArrow) {
      throw superCalls[0].buildCodeFrameError("When using '@babel/plugin-transform-arrow-functions', " + "it's not possible to compile `super()` in an arrow function without compiling classes.\n" + "Please add '@babel/plugin-transform-classes' to your Babel configuration.");
    }
    if (!allowInsertArrowWithRest) {
      throw superCalls[0].buildCodeFrameError("When using '@babel/plugin-transform-parameters', " + "it's not possible to compile `super()` in an arrow function with default or rest parameters without compiling classes.\n" + "Please add '@babel/plugin-transform-classes' to your Babel configuration.");
    }
    const allSuperCalls = [];
    thisEnvFn.traverse(getSuperCallsVisitor, {
      allSuperCalls
    });
    const superBinding = getSuperBinding(thisEnvFn);
    allSuperCalls.forEach(superCall => {
      const callee = identifier(superBinding);
      callee.loc = superCall.node.callee.loc;
      superCall.get("callee").replaceWith(callee);
    });
  }
  if (argumentsPaths.length > 0) {
    const argumentsBinding = getBinding(thisEnvFn, "arguments", () => {
      const args = () => identifier("arguments");
      if (thisEnvFn.scope.path.isProgram()) {
        return conditionalExpression(binaryExpression("===", unaryExpression("typeof", args()), stringLiteral("undefined")), thisEnvFn.scope.buildUndefinedNode(), args());
      } else {
        return args();
      }
    });
    argumentsPaths.forEach(argumentsChild => {
      const argsRef = identifier(argumentsBinding);
      argsRef.loc = argumentsChild.node.loc;
      argumentsChild.replaceWith(argsRef);
    });
  }
  if (newTargetPaths.length > 0) {
    const newTargetBinding = getBinding(thisEnvFn, "newtarget", () => metaProperty(identifier("new"), identifier("target")));
    newTargetPaths.forEach(targetChild => {
      const targetRef = identifier(newTargetBinding);
      targetRef.loc = targetChild.node.loc;
      targetChild.replaceWith(targetRef);
    });
  }
  if (superProps.length > 0) {
    if (!allowInsertArrow) {
      throw superProps[0].buildCodeFrameError("When using '@babel/plugin-transform-arrow-functions', " + "it's not possible to compile `super.prop` in an arrow function without compiling classes.\n" + "Please add '@babel/plugin-transform-classes' to your Babel configuration.");
    }
    const flatSuperProps = superProps.reduce((acc, superProp) => acc.concat(standardizeSuperProperty(superProp)), []);
    flatSuperProps.forEach(superProp => {
      const key = superProp.node.computed ? "" : superProp.get("property").node.name;
      const superParentPath = superProp.parentPath;
      const isAssignment = superParentPath.isAssignmentExpression({
        left: superProp.node
      });
      const isCall = superParentPath.isCallExpression({
        callee: superProp.node
      });
      const isTaggedTemplate = superParentPath.isTaggedTemplateExpression({
        tag: superProp.node
      });
      const superBinding = getSuperPropBinding(thisEnvFn, isAssignment, key);
      const args = [];
      if (superProp.node.computed) {
        args.push(superProp.get("property").node);
      }
      if (isAssignment) {
        const value = superParentPath.node.right;
        args.push(value);
      }
      const call = callExpression(identifier(superBinding), args);
      if (isCall) {
        superParentPath.unshiftContainer("arguments", thisExpression());
        superProp.replaceWith(memberExpression(call, identifier("call")));
        thisPaths.push(superParentPath.get("arguments.0"));
      } else if (isAssignment) {
        superParentPath.replaceWith(call);
      } else if (isTaggedTemplate) {
        superProp.replaceWith(callExpression(memberExpression(call, identifier("bind"), false), [thisExpression()]));
        thisPaths.push(superProp.get("arguments.0"));
      } else {
        superProp.replaceWith(call);
      }
    });
  }
  let thisBinding;
  if (thisPaths.length > 0 || !noNewArrows) {
    thisBinding = getThisBinding(thisEnvFn, inConstructor);
    if (noNewArrows || inConstructor && hasSuperClass(thisEnvFn)) {
      thisPaths.forEach(thisChild => {
        const thisRef = thisChild.isJSX() ? jsxIdentifier(thisBinding) : identifier(thisBinding);
        thisRef.loc = thisChild.node.loc;
        thisChild.replaceWith(thisRef);
      });
      if (!noNewArrows) thisBinding = null;
    }
  }
  return {
    thisBinding,
    fnPath
  };
}
function isLogicalOp(op) {
  return LOGICAL_OPERATORS.includes(op);
}
function standardizeSuperProperty(superProp) {
  if (superProp.parentPath.isAssignmentExpression() && superProp.parentPath.node.operator !== "=") {
    const assignmentPath = superProp.parentPath;
    const op = assignmentPath.node.operator.slice(0, -1);
    const value = assignmentPath.node.right;
    const isLogicalAssignment = isLogicalOp(op);
    if (superProp.node.computed) {
      const tmp = superProp.scope.generateDeclaredUidIdentifier("tmp");
      const object = superProp.node.object;
      const property = superProp.node.property;
      assignmentPath.get("left").replaceWith(memberExpression(object, assignmentExpression("=", tmp, property), true));
      assignmentPath.get("right").replaceWith(rightExpression(isLogicalAssignment ? "=" : op, memberExpression(object, identifier(tmp.name), true), value));
    } else {
      const object = superProp.node.object;
      const property = superProp.node.property;
      assignmentPath.get("left").replaceWith(memberExpression(object, property));
      assignmentPath.get("right").replaceWith(rightExpression(isLogicalAssignment ? "=" : op, memberExpression(object, identifier(property.name)), value));
    }
    if (isLogicalAssignment) {
      assignmentPath.replaceWith(logicalExpression(op, assignmentPath.node.left, assignmentPath.node.right));
    } else {
      assignmentPath.node.operator = "=";
    }
    return [assignmentPath.get("left"), assignmentPath.get("right").get("left")];
  } else if (superProp.parentPath.isUpdateExpression()) {
    const updateExpr = superProp.parentPath;
    const tmp = superProp.scope.generateDeclaredUidIdentifier("tmp");
    const computedKey = superProp.node.computed ? superProp.scope.generateDeclaredUidIdentifier("prop") : null;
    const parts = [assignmentExpression("=", tmp, memberExpression(superProp.node.object, computedKey ? assignmentExpression("=", computedKey, superProp.node.property) : superProp.node.property, superProp.node.computed)), assignmentExpression("=", memberExpression(superProp.node.object, computedKey ? identifier(computedKey.name) : superProp.node.property, superProp.node.computed), binaryExpression(superProp.parentPath.node.operator[0], identifier(tmp.name), numericLiteral(1)))];
    if (!superProp.parentPath.node.prefix) {
      parts.push(identifier(tmp.name));
    }
    updateExpr.replaceWith(sequenceExpression(parts));
    const left = updateExpr.get("expressions.0.right");
    const right = updateExpr.get("expressions.1.left");
    return [left, right];
  }
  return [superProp];
  function rightExpression(op, left, right) {
    if (op === "=") {
      return assignmentExpression("=", left, right);
    } else {
      return binaryExpression(op, left, right);
    }
  }
}
function hasSuperClass(thisEnvFn) {
  return thisEnvFn.isClassMethod() && !!thisEnvFn.parentPath.parentPath.node.superClass;
}
const assignSuperThisVisitor = (0, _visitors.merge)([{
  CallExpression(child, {
    supers,
    thisBinding
  }) {
    if (!child.get("callee").isSuper()) return;
    if (supers.has(child.node)) return;
    supers.add(child.node);
    child.replaceWithMultiple([child.node, assignmentExpression("=", identifier(thisBinding), identifier("this"))]);
  }
}, _helperEnvironmentVisitor.default]);
function getThisBinding(thisEnvFn, inConstructor) {
  return getBinding(thisEnvFn, "this", thisBinding => {
    if (!inConstructor || !hasSuperClass(thisEnvFn)) return thisExpression();
    thisEnvFn.traverse(assignSuperThisVisitor, {
      supers: new WeakSet(),
      thisBinding
    });
  });
}
function getSuperBinding(thisEnvFn) {
  return getBinding(thisEnvFn, "supercall", () => {
    const argsBinding = thisEnvFn.scope.generateUidIdentifier("args");
    return arrowFunctionExpression([restElement(argsBinding)], callExpression(_super(), [spreadElement(identifier(argsBinding.name))]));
  });
}
function getSuperPropBinding(thisEnvFn, isAssignment, propName) {
  const op = isAssignment ? "set" : "get";
  return getBinding(thisEnvFn, `superprop_${op}:${propName || ""}`, () => {
    const argsList = [];
    let fnBody;
    if (propName) {
      fnBody = memberExpression(_super(), identifier(propName));
    } else {
      const method = thisEnvFn.scope.generateUidIdentifier("prop");
      argsList.unshift(method);
      fnBody = memberExpression(_super(), identifier(method.name), true);
    }
    if (isAssignment) {
      const valueIdent = thisEnvFn.scope.generateUidIdentifier("value");
      argsList.push(valueIdent);
      fnBody = assignmentExpression("=", fnBody, identifier(valueIdent.name));
    }
    return arrowFunctionExpression(argsList, fnBody);
  });
}
function getBinding(thisEnvFn, key, init) {
  const cacheKey = "binding:" + key;
  let data = thisEnvFn.getData(cacheKey);
  if (!data) {
    const id = thisEnvFn.scope.generateUidIdentifier(key);
    data = id.name;
    thisEnvFn.setData(cacheKey, data);
    thisEnvFn.scope.push({
      id: id,
      init: init(data)
    });
  }
  return data;
}
const getScopeInformationVisitor = (0, _visitors.merge)([{
  ThisExpression(child, {
    thisPaths
  }) {
    thisPaths.push(child);
  },
  JSXIdentifier(child, {
    thisPaths
  }) {
    if (child.node.name !== "this") return;
    if (!child.parentPath.isJSXMemberExpression({
      object: child.node
    }) && !child.parentPath.isJSXOpeningElement({
      name: child.node
    })) {
      return;
    }
    thisPaths.push(child);
  },
  CallExpression(child, {
    superCalls
  }) {
    if (child.get("callee").isSuper()) superCalls.push(child);
  },
  MemberExpression(child, {
    superProps
  }) {
    if (child.get("object").isSuper()) superProps.push(child);
  },
  Identifier(child, {
    argumentsPaths
  }) {
    if (!child.isReferencedIdentifier({
      name: "arguments"
    })) return;
    let curr = child.scope;
    do {
      if (curr.hasOwnBinding("arguments")) {
        curr.rename("arguments");
        return;
      }
      if (curr.path.isFunction() && !curr.path.isArrowFunctionExpression()) {
        break;
      }
    } while (curr = curr.parent);
    argumentsPaths.push(child);
  },
  MetaProperty(child, {
    newTargetPaths
  }) {
    if (!child.get("meta").isIdentifier({
      name: "new"
    })) return;
    if (!child.get("property").isIdentifier({
      name: "target"
    })) return;
    newTargetPaths.push(child);
  }
}, _helperEnvironmentVisitor.default]);
function getScopeInformation(fnPath) {
  const thisPaths = [];
  const argumentsPaths = [];
  const newTargetPaths = [];
  const superProps = [];
  const superCalls = [];
  fnPath.traverse(getScopeInformationVisitor, {
    thisPaths,
    argumentsPaths,
    newTargetPaths,
    superProps,
    superCalls
  });
  return {
    thisPaths,
    argumentsPaths,
    newTargetPaths,
    superProps,
    superCalls
  };
}

//# sourceMappingURL=conversion.js.map
