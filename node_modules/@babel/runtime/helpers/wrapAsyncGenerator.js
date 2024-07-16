var AsyncGenerator = require("./AsyncGenerator.js");
function _wrapAsyncGenerator(fn) {
  return function () {
    return new AsyncGenerator(fn.apply(this, arguments));
  };
}
module.exports = _wrapAsyncGenerator, module.exports.__esModule = true, module.exports["default"] = module.exports;