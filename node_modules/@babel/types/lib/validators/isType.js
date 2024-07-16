"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = isType;
var _index = require("../definitions/index.js");
function isType(nodeType, targetType) {
  if (nodeType === targetType) return true;
  if (nodeType == null) return false;
  if (_index.ALIAS_KEYS[targetType]) return false;
  const aliases = _index.FLIPPED_ALIAS_KEYS[targetType];
  if (aliases) {
    if (aliases[0] === nodeType) return true;
    for (const alias of aliases) {
      if (nodeType === alias) return true;
    }
  }
  return false;
}

//# sourceMappingURL=isType.js.map
