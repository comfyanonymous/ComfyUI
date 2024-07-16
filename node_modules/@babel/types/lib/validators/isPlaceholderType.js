"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = isPlaceholderType;
var _index = require("../definitions/index.js");
function isPlaceholderType(placeholderType, targetType) {
  if (placeholderType === targetType) return true;
  const aliases = _index.PLACEHOLDERS_ALIAS[placeholderType];
  if (aliases) {
    for (const alias of aliases) {
      if (targetType === alias) return true;
    }
  }
  return false;
}

//# sourceMappingURL=isPlaceholderType.js.map
