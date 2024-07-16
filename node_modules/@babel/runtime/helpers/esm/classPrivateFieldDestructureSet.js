import classApplyDescriptorDestructureSet from "./classApplyDescriptorDestructureSet.js";
import classPrivateFieldGet2 from "./classPrivateFieldGet2.js";
export default function _classPrivateFieldDestructureSet(receiver, privateMap) {
  var descriptor = classPrivateFieldGet2(privateMap, receiver);
  return classApplyDescriptorDestructureSet(receiver, descriptor);
}