import arrayWithHoles from "./arrayWithHoles.js";
import iterableToArray from "./iterableToArray.js";
import unsupportedIterableToArray from "./unsupportedIterableToArray.js";
import nonIterableRest from "./nonIterableRest.js";
export default function _toArray(arr) {
  return arrayWithHoles(arr) || iterableToArray(arr) || unsupportedIterableToArray(arr) || nonIterableRest();
}