"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = validate;
exports.validateChild = validateChild;
exports.validateField = validateField;
var _index = require("../definitions/index.js");
function validate(node, key, val) {
  if (!node) return;
  const fields = _index.NODE_FIELDS[node.type];
  if (!fields) return;
  const field = fields[key];
  validateField(node, key, val, field);
  validateChild(node, key, val);
}
function validateField(node, key, val, field) {
  if (!(field != null && field.validate)) return;
  if (field.optional && val == null) return;
  field.validate(node, key, val);
}
function validateChild(node, key, val) {
  if (val == null) return;
  const validate = _index.NODE_PARENT_VALIDATIONS[val.type];
  if (!validate) return;
  validate(node, key, val);
}

//# sourceMappingURL=validate.js.map
