import _typeof from "./typeof.js";
import toPrimitive from "./toPrimitive.js";
export default function toPropertyKey(t) {
  var i = toPrimitive(t, "string");
  return "symbol" == _typeof(i) ? i : i + "";
}