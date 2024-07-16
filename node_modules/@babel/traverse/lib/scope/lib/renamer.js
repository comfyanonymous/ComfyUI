"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = void 0;
var _helperSplitExportDeclaration = require("@babel/helper-split-export-declaration");
var t = require("@babel/types");
var _helperEnvironmentVisitor = require("@babel/helper-environment-visitor");
var _traverseNode = require("../../traverse-node.js");
var _visitors = require("../../visitors.js");
const renameVisitor = {
  ReferencedIdentifier({
    node
  }, state) {
    if (node.name === state.oldName) {
      node.name = state.newName;
    }
  },
  Scope(path, state) {
    if (!path.scope.bindingIdentifierEquals(state.oldName, state.binding.identifier)) {
      path.skip();
      if (path.isMethod()) {
        (0, _helperEnvironmentVisitor.requeueComputedKeyAndDecorators)(path);
      }
    }
  },
  ObjectProperty({
    node,
    scope
  }, state) {
    const {
      name
    } = node.key;
    if (node.shorthand && (name === state.oldName || name === state.newName) && scope.getBindingIdentifier(name) === state.binding.identifier) {
      var _node$extra;
      node.shorthand = false;
      if ((_node$extra = node.extra) != null && _node$extra.shorthand) node.extra.shorthand = false;
    }
  },
  "AssignmentExpression|Declaration|VariableDeclarator"(path, state) {
    if (path.isVariableDeclaration()) return;
    const ids = path.getOuterBindingIdentifiers();
    for (const name in ids) {
      if (name === state.oldName) ids[name].name = state.newName;
    }
  }
};
class Renamer {
  constructor(binding, oldName, newName) {
    this.newName = newName;
    this.oldName = oldName;
    this.binding = binding;
  }
  maybeConvertFromExportDeclaration(parentDeclar) {
    const maybeExportDeclar = parentDeclar.parentPath;
    if (!maybeExportDeclar.isExportDeclaration()) {
      return;
    }
    if (maybeExportDeclar.isExportDefaultDeclaration()) {
      const {
        declaration
      } = maybeExportDeclar.node;
      if (t.isDeclaration(declaration) && !declaration.id) {
        return;
      }
    }
    if (maybeExportDeclar.isExportAllDeclaration()) {
      return;
    }
    (0, _helperSplitExportDeclaration.default)(maybeExportDeclar);
  }
  maybeConvertFromClassFunctionDeclaration(path) {
    return path;
  }
  maybeConvertFromClassFunctionExpression(path) {
    return path;
  }
  rename() {
    const {
      binding,
      oldName,
      newName
    } = this;
    const {
      scope,
      path
    } = binding;
    const parentDeclar = path.find(path => path.isDeclaration() || path.isFunctionExpression() || path.isClassExpression());
    if (parentDeclar) {
      const bindingIds = parentDeclar.getOuterBindingIdentifiers();
      if (bindingIds[oldName] === binding.identifier) {
        this.maybeConvertFromExportDeclaration(parentDeclar);
      }
    }
    const blockToTraverse = arguments[0] || scope.block;
    (0, _traverseNode.traverseNode)(blockToTraverse, (0, _visitors.explode)(renameVisitor), scope, this, scope.path, {
      discriminant: true
    });
    if (!arguments[0]) {
      scope.removeOwnBinding(oldName);
      scope.bindings[newName] = binding;
      this.binding.identifier.name = newName;
    }
    if (parentDeclar) {
      this.maybeConvertFromClassFunctionDeclaration(path);
      this.maybeConvertFromClassFunctionExpression(path);
    }
  }
}
exports.default = Renamer;

//# sourceMappingURL=renamer.js.map
