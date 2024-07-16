import classApplyDescriptorSet from "./classApplyDescriptorSet.js";
import assertClassBrand from "./assertClassBrand.js";
import classCheckPrivateStaticFieldDescriptor from "./classCheckPrivateStaticFieldDescriptor.js";
export default function _classStaticPrivateFieldSpecSet(receiver, classConstructor, descriptor, value) {
  assertClassBrand(classConstructor, receiver);
  classCheckPrivateStaticFieldDescriptor(descriptor, "set");
  classApplyDescriptorSet(receiver, descriptor, value);
  return value;
}