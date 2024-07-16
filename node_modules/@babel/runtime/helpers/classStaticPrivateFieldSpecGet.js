var classApplyDescriptorGet = require("./classApplyDescriptorGet.js");
var assertClassBrand = require("./assertClassBrand.js");
var classCheckPrivateStaticFieldDescriptor = require("./classCheckPrivateStaticFieldDescriptor.js");
function _classStaticPrivateFieldSpecGet(receiver, classConstructor, descriptor) {
  assertClassBrand(classConstructor, receiver);
  classCheckPrivateStaticFieldDescriptor(descriptor, "get");
  return classApplyDescriptorGet(receiver, descriptor);
}
module.exports = _classStaticPrivateFieldSpecGet, module.exports.__esModule = true, module.exports["default"] = module.exports;