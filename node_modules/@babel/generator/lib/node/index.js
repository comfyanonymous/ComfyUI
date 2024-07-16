"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.needsParens = needsParens;
exports.needsWhitespace = needsWhitespace;
exports.needsWhitespaceAfter = needsWhitespaceAfter;
exports.needsWhitespaceBefore = needsWhitespaceBefore;
var whitespace = require("./whitespace.js");
var parens = require("./parentheses.js");
var _t = require("@babel/types");
const {
  FLIPPED_ALIAS_KEYS,
  isCallExpression,
  isExpressionStatement,
  isMemberExpression,
  isNewExpression
} = _t;
function expandAliases(obj) {
  const map = new Map();
  function add(type, func) {
    const fn = map.get(type);
    map.set(type, fn ? function (node, parent, stack) {
      var _fn;
      return (_fn = fn(node, parent, stack)) != null ? _fn : func(node, parent, stack);
    } : func);
  }
  for (const type of Object.keys(obj)) {
    const aliases = FLIPPED_ALIAS_KEYS[type];
    if (aliases) {
      for (const alias of aliases) {
        add(alias, obj[type]);
      }
    } else {
      add(type, obj[type]);
    }
  }
  return map;
}
const expandedParens = expandAliases(parens);
const expandedWhitespaceNodes = expandAliases(whitespace.nodes);
function isOrHasCallExpression(node) {
  if (isCallExpression(node)) {
    return true;
  }
  return isMemberExpression(node) && isOrHasCallExpression(node.object);
}
function needsWhitespace(node, parent, type) {
  var _expandedWhitespaceNo;
  if (!node) return false;
  if (isExpressionStatement(node)) {
    node = node.expression;
  }
  const flag = (_expandedWhitespaceNo = expandedWhitespaceNodes.get(node.type)) == null ? void 0 : _expandedWhitespaceNo(node, parent);
  if (typeof flag === "number") {
    return (flag & type) !== 0;
  }
  return false;
}
function needsWhitespaceBefore(node, parent) {
  return needsWhitespace(node, parent, 1);
}
function needsWhitespaceAfter(node, parent) {
  return needsWhitespace(node, parent, 2);
}
function needsParens(node, parent, printStack) {
  var _expandedParens$get;
  if (!parent) return false;
  if (isNewExpression(parent) && parent.callee === node) {
    if (isOrHasCallExpression(node)) return true;
  }
  return (_expandedParens$get = expandedParens.get(node.type)) == null ? void 0 : _expandedParens$get(node, parent, printStack);
}

//# sourceMappingURL=index.js.map
