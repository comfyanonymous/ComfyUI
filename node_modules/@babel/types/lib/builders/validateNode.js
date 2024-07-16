"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = validateNode;
var _validate = require("../validators/validate.js");
var _index = require("../index.js");
function validateNode(node) {
  const keys = _index.BUILDER_KEYS[node.type];
  for (const key of keys) {
    (0, _validate.default)(node, key, node[key]);
  }
  return node;
}

//# sourceMappingURL=validateNode.js.map
