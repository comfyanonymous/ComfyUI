var classApplyDescriptorSet = require("./classApplyDescriptorSet.js");
var assertClassBrand = require("./assertClassBrand.js");
var classCheckPrivateStaticFieldDescriptor = require("./classCheckPrivateStaticFieldDescriptor.js");
function _classStaticPrivateFieldSpecSet(receiver, classConstructor, descriptor, value) {
  assertClassBrand(classConstructor, receiver);
  classCheckPrivateStaticFieldDescriptor(descriptor, "set");
  classApplyDescriptorSet(receiver, descriptor, value);
  return value;
}
module.exports = _classStaticPrivateFieldSpecSet, module.exports.__esModule = true, module.exports["default"] = module.exports;