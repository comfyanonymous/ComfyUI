import assertClassBrand from "./assertClassBrand.js";
export default function _classPrivateSetter(s, r, a, t) {
  return r(assertClassBrand(s, a), t), t;
}