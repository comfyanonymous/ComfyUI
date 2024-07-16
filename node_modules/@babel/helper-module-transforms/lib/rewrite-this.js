"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = rewriteThis;
var _helperEnvironmentVisitor = require("@babel/helper-environment-visitor");
var _core = require("@babel/core");
const {
  numericLiteral,
  unaryExpression
} = _core.types;
const rewriteThisVisitor = _core.traverse.visitors.merge([_helperEnvironmentVisitor.default, {
  ThisExpression(path) {
    path.replaceWith(unaryExpression("void", numericLiteral(0), true));
  }
}]);
function rewriteThis(programPath) {
  (0, _core.traverse)(programPath.node, Object.assign({}, rewriteThisVisitor, {
    noScope: true
  }));
}

//# sourceMappingURL=rewrite-this.js.map
