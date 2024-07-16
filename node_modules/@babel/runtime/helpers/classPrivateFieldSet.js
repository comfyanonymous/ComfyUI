var classApplyDescriptorSet = require("./classApplyDescriptorSet.js");
var classPrivateFieldGet2 = require("./classPrivateFieldGet2.js");
function _classPrivateFieldSet(receiver, privateMap, value) {
  var descriptor = classPrivateFieldGet2(privateMap, receiver);
  classApplyDescriptorSet(receiver, descriptor, value);
  return value;
}
module.exports = _classPrivateFieldSet, module.exports.__esModule = true, module.exports["default"] = module.exports;