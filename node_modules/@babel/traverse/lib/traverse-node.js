"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.traverseNode = traverseNode;
var _context = require("./context.js");
var _t = require("@babel/types");
const {
  VISITOR_KEYS
} = _t;
function traverseNode(node, opts, scope, state, path, skipKeys, visitSelf) {
  const keys = VISITOR_KEYS[node.type];
  if (!keys) return false;
  const context = new _context.default(scope, opts, state, path);
  if (visitSelf) {
    if (skipKeys != null && skipKeys[path.parentKey]) return false;
    return context.visitQueue([path]);
  }
  for (const key of keys) {
    if (skipKeys != null && skipKeys[key]) continue;
    if (context.visit(node, key)) {
      return true;
    }
  }
  return false;
}

//# sourceMappingURL=traverse-node.js.map
