var classApplyDescriptorDestructureSet = require("./classApplyDescriptorDestructureSet.js");
var assertClassBrand = require("./assertClassBrand.js");
var classCheckPrivateStaticFieldDescriptor = require("./classCheckPrivateStaticFieldDescriptor.js");
function _classStaticPrivateFieldDestructureSet(receiver, classConstructor, descriptor) {
  assertClassBrand(classConstructor, receiver);
  classCheckPrivateStaticFieldDescriptor(descriptor, "set");
  return classApplyDescriptorDestructureSet(receiver, descriptor);
}
module.exports = _classStaticPrivateFieldDestructureSet, module.exports.__esModule = true, module.exports["default"] = module.exports;