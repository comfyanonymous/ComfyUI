import arrayWithHoles from "./arrayWithHoles.js";
import iterableToArrayLimitLoose from "./iterableToArrayLimitLoose.js";
import unsupportedIterableToArray from "./unsupportedIterableToArray.js";
import nonIterableRest from "./nonIterableRest.js";
export default function _slicedToArrayLoose(arr, i) {
  return arrayWithHoles(arr) || iterableToArrayLimitLoose(arr, i) || unsupportedIterableToArray(arr, i) || nonIterableRest();
}