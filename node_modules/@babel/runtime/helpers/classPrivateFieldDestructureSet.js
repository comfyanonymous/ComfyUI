var classApplyDescriptorDestructureSet = require("./classApplyDescriptorDestructureSet.js");
var classPrivateFieldGet2 = require("./classPrivateFieldGet2.js");
function _classPrivateFieldDestructureSet(receiver, privateMap) {
  var descriptor = classPrivateFieldGet2(privateMap, receiver);
  return classApplyDescriptorDestructureSet(receiver, descriptor);
}
module.exports = _classPrivateFieldDestructureSet, module.exports.__esModule = true, module.exports["default"] = module.exports;