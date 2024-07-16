import assertClassBrand from "./assertClassBrand.js";
export default function _classCheckPrivateStaticAccess(receiver, classConstructor, returnValue) {
  return assertClassBrand(classConstructor, receiver, returnValue);
}