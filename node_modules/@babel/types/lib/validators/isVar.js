"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = isVar;
var _index = require("./generated/index.js");
var _index2 = require("../constants/index.js");
function isVar(node) {
  return (0, _index.isVariableDeclaration)(node, {
    kind: "var"
  }) && !node[_index2.BLOCK_SCOPED_SYMBOL];
}

//# sourceMappingURL=isVar.js.map
