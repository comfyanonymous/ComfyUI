var classApplyDescriptorGet = require("./classApplyDescriptorGet.js");
var classPrivateFieldGet2 = require("./classPrivateFieldGet2.js");
function _classPrivateFieldGet(receiver, privateMap) {
  var descriptor = classPrivateFieldGet2(privateMap, receiver);
  return classApplyDescriptorGet(receiver, descriptor);
}
module.exports = _classPrivateFieldGet, module.exports.__esModule = true, module.exports["default"] = module.exports;