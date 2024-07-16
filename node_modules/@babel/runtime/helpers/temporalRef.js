var temporalUndefined = require("./temporalUndefined.js");
var tdz = require("./tdz.js");
function _temporalRef(val, name) {
  return val === temporalUndefined ? tdz(name) : val;
}
module.exports = _temporalRef, module.exports.__esModule = true, module.exports["default"] = module.exports;