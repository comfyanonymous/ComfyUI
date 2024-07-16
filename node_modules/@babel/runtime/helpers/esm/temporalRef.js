import undef from "./temporalUndefined.js";
import err from "./tdz.js";
export default function _temporalRef(val, name) {
  return val === undef ? err(name) : val;
}