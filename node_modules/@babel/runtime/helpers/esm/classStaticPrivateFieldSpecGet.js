import classApplyDescriptorGet from "./classApplyDescriptorGet.js";
import assertClassBrand from "./assertClassBrand.js";
import classCheckPrivateStaticFieldDescriptor from "./classCheckPrivateStaticFieldDescriptor.js";
export default function _classStaticPrivateFieldSpecGet(receiver, classConstructor, descriptor) {
  assertClassBrand(classConstructor, receiver);
  classCheckPrivateStaticFieldDescriptor(descriptor, "get");
  return classApplyDescriptorGet(receiver, descriptor);
}