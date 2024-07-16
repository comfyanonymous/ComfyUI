function _taggedTemplateLiteralLoose(strings, raw) {
  if (!raw) {
    raw = strings.slice(0);
  }
  strings.raw = raw;
  return strings;
}
module.exports = _taggedTemplateLiteralLoose, module.exports.__esModule = true, module.exports["default"] = module.exports;