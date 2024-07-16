"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = void 0;
class Binding {
  constructor({
    identifier,
    scope,
    path,
    kind
  }) {
    this.identifier = void 0;
    this.scope = void 0;
    this.path = void 0;
    this.kind = void 0;
    this.constantViolations = [];
    this.constant = true;
    this.referencePaths = [];
    this.referenced = false;
    this.references = 0;
    this.identifier = identifier;
    this.scope = scope;
    this.path = path;
    this.kind = kind;
    if ((kind === "var" || kind === "hoisted") && isDeclaredInLoop(path)) {
      this.reassign(path);
    }
    this.clearValue();
  }
  deoptValue() {
    this.clearValue();
    this.hasDeoptedValue = true;
  }
  setValue(value) {
    if (this.hasDeoptedValue) return;
    this.hasValue = true;
    this.value = value;
  }
  clearValue() {
    this.hasDeoptedValue = false;
    this.hasValue = false;
    this.value = null;
  }
  reassign(path) {
    this.constant = false;
    if (this.constantViolations.indexOf(path) !== -1) {
      return;
    }
    this.constantViolations.push(path);
  }
  reference(path) {
    if (this.referencePaths.indexOf(path) !== -1) {
      return;
    }
    this.referenced = true;
    this.references++;
    this.referencePaths.push(path);
  }
  dereference() {
    this.references--;
    this.referenced = !!this.references;
  }
}
exports.default = Binding;
function isDeclaredInLoop(path) {
  for (let {
    parentPath,
    key
  } = path; parentPath; ({
    parentPath,
    key
  } = parentPath)) {
    if (parentPath.isFunctionParent()) return false;
    if (parentPath.isWhile() || parentPath.isForXStatement() || parentPath.isForStatement() && key === "body") {
      return true;
    }
  }
  return false;
}

//# sourceMappingURL=binding.js.map
