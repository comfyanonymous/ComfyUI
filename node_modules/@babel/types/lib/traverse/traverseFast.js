"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = traverseFast;
var _index = require("../definitions/index.js");
function traverseFast(node, enter, opts) {
  if (!node) return;
  const keys = _index.VISITOR_KEYS[node.type];
  if (!keys) return;
  opts = opts || {};
  enter(node, opts);
  for (const key of keys) {
    const subNode = node[key];
    if (Array.isArray(subNode)) {
      for (const node of subNode) {
        traverseFast(node, enter, opts);
      }
    } else {
      traverseFast(subNode, enter, opts);
    }
  }
}

//# sourceMappingURL=traverseFast.js.map
