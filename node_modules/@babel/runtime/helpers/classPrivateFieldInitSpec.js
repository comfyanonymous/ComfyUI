var checkPrivateRedeclaration = require("./checkPrivateRedeclaration.js");
function _classPrivateFieldInitSpec(obj, privateMap, value) {
  checkPrivateRedeclaration(obj, privateMap);
  privateMap.set(obj, value);
}
module.exports = _classPrivateFieldInitSpec, module.exports.__esModule = true, module.exports["default"] = module.exports;