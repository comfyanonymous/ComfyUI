function _classPrivateFieldBase(receiver, privateKey) {
  if (!Object.prototype.hasOwnProperty.call(receiver, privateKey)) {
    throw new TypeError("attempted to use private field on non-instance");
  }
  return receiver;
}
module.exports = _classPrivateFieldBase, module.exports.__esModule = true, module.exports["default"] = module.exports;