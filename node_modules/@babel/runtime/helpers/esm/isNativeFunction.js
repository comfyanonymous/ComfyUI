export default function _isNativeFunction(fn) {
  try {
    return Function.toString.call(fn).indexOf("[native code]") !== -1;
  } catch (e) {
    return typeof fn === "function";
  }
}