import checkPrivateRedeclaration from "./checkPrivateRedeclaration.js";
export default function _classPrivateFieldInitSpec(obj, privateMap, value) {
  checkPrivateRedeclaration(obj, privateMap);
  privateMap.set(obj, value);
}