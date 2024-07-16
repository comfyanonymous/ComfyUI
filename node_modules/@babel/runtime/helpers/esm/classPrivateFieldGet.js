import classApplyDescriptorGet from "./classApplyDescriptorGet.js";
import classPrivateFieldGet2 from "./classPrivateFieldGet2.js";
export default function _classPrivateFieldGet(receiver, privateMap) {
  var descriptor = classPrivateFieldGet2(privateMap, receiver);
  return classApplyDescriptorGet(receiver, descriptor);
}