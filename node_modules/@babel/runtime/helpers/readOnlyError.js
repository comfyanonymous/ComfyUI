function _readOnlyError(name) {
  throw new TypeError("\"" + name + "\" is read-only");
}
module.exports = _readOnlyError, module.exports.__esModule = true, module.exports["default"] = module.exports;