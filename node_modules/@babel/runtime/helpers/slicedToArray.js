var arrayWithHoles = require("./arrayWithHoles.js");
var iterableToArrayLimit = require("./iterableToArrayLimit.js");
var unsupportedIterableToArray = require("./unsupportedIterableToArray.js");
var nonIterableRest = require("./nonIterableRest.js");
function _slicedToArray(arr, i) {
  return arrayWithHoles(arr) || iterableToArrayLimit(arr, i) || unsupportedIterableToArray(arr, i) || nonIterableRest();
}
module.exports = _slicedToArray, module.exports.__esModule = true, module.exports["default"] = module.exports;