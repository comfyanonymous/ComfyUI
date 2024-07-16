function _writeOnlyError(name) {
  throw new TypeError("\"" + name + "\" is write-only");
}
module.exports = _writeOnlyError, module.exports.__esModule = true, module.exports["default"] = module.exports;