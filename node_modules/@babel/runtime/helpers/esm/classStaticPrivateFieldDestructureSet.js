import classApplyDescriptorDestructureSet from "./classApplyDescriptorDestructureSet.js";
import assertClassBrand from "./assertClassBrand.js";
import classCheckPrivateStaticFieldDescriptor from "./classCheckPrivateStaticFieldDescriptor.js";
export default function _classStaticPrivateFieldDestructureSet(receiver, classConstructor, descriptor) {
  assertClassBrand(classConstructor, receiver);
  classCheckPrivateStaticFieldDescriptor(descriptor, "set");
  return classApplyDescriptorDestructureSet(receiver, descriptor);
}