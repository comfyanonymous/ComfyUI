import checkPrivateRedeclaration from "./checkPrivateRedeclaration.js";
export default function _classPrivateMethodInitSpec(obj, privateSet) {
  checkPrivateRedeclaration(obj, privateSet);
  privateSet.add(obj);
}