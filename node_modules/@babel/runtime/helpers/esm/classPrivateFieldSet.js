import classApplyDescriptorSet from "./classApplyDescriptorSet.js";
import classPrivateFieldGet2 from "./classPrivateFieldGet2.js";
export default function _classPrivateFieldSet(receiver, privateMap, value) {
  var descriptor = classPrivateFieldGet2(privateMap, receiver);
  classApplyDescriptorSet(receiver, descriptor, value);
  return value;
}