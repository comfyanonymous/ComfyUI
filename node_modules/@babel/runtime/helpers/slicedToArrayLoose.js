var arrayWithHoles = require("./arrayWithHoles.js");
var iterableToArrayLimitLoose = require("./iterableToArrayLimitLoose.js");
var unsupportedIterableToArray = require("./unsupportedIterableToArray.js");
var nonIterableRest = require("./nonIterableRest.js");
function _slicedToArrayLoose(arr, i) {
  return arrayWithHoles(arr) || iterableToArrayLimitLoose(arr, i) || unsupportedIterableToArray(arr, i) || nonIterableRest();
}
module.exports = _slicedToArrayLoose, module.exports.__esModule = true, module.exports["default"] = module.exports;