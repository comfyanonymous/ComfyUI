"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = matchesPattern;
var _index = require("./generated/index.js");
function matchesPattern(member, match, allowPartial) {
  if (!(0, _index.isMemberExpression)(member)) return false;
  const parts = Array.isArray(match) ? match : match.split(".");
  const nodes = [];
  let node;
  for (node = member; (0, _index.isMemberExpression)(node); node = node.object) {
    nodes.push(node.property);
  }
  nodes.push(node);
  if (nodes.length < parts.length) return false;
  if (!allowPartial && nodes.length > parts.length) return false;
  for (let i = 0, j = nodes.length - 1; i < parts.length; i++, j--) {
    const node = nodes[j];
    let value;
    if ((0, _index.isIdentifier)(node)) {
      value = node.name;
    } else if ((0, _index.isStringLiteral)(node)) {
      value = node.value;
    } else if ((0, _index.isThisExpression)(node)) {
      value = "this";
    } else {
      return false;
    }
    if (parts[i] !== value) return false;
  }
  return true;
}

//# sourceMappingURL=matchesPattern.js.map
