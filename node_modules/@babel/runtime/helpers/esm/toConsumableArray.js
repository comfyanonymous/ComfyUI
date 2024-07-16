import arrayWithoutHoles from "./arrayWithoutHoles.js";
import iterableToArray from "./iterableToArray.js";
import unsupportedIterableToArray from "./unsupportedIterableToArray.js";
import nonIterableSpread from "./nonIterableSpread.js";
export default function _toConsumableArray(arr) {
  return arrayWithoutHoles(arr) || iterableToArray(arr) || unsupportedIterableToArray(arr) || nonIterableSpread();
}