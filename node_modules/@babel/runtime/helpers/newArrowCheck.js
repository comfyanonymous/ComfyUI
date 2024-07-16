function _newArrowCheck(innerThis, boundThis) {
  if (innerThis !== boundThis) {
    throw new TypeError("Cannot instantiate an arrow function");
  }
}
module.exports = _newArrowCheck, module.exports.__esModule = true, module.exports["default"] = module.exports;