var defineProperty = require("./defineProperty.js");
function _objectSpread(target) {
  for (var i = 1; i < arguments.length; i++) {
    var source = arguments[i] != null ? Object(arguments[i]) : {};
    var ownKeys = Object.keys(source);
    if (typeof Object.getOwnPropertySymbols === 'function') {
      ownKeys.push.apply(ownKeys, Object.getOwnPropertySymbols(source).filter(function (sym) {
        return Object.getOwnPropertyDescriptor(source, sym).enumerable;
      }));
    }
    ownKeys.forEach(function (key) {
      defineProperty(target, key, source[key]);
    });
  }
  return target;
}
module.exports = _objectSpread, module.exports.__esModule = true, module.exports["default"] = module.exports;