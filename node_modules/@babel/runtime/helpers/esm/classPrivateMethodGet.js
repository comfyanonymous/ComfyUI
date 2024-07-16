import assertClassBrand from "./assertClassBrand.js";
export default function _classPrivateMethodGet(receiver, privateSet, fn) {
  assertClassBrand(privateSet, receiver);
  return fn;
}