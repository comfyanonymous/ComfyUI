import arrayWithHoles from "./arrayWithHoles.js";
import iterableToArrayLimit from "./iterableToArrayLimit.js";
import unsupportedIterableToArray from "./unsupportedIterableToArray.js";
import nonIterableRest from "./nonIterableRest.js";
export default function _slicedToArray(arr, i) {
  return arrayWithHoles(arr) || iterableToArrayLimit(arr, i) || unsupportedIterableToArray(arr, i) || nonIterableRest();
}