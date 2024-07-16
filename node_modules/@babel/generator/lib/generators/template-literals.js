"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.TaggedTemplateExpression = TaggedTemplateExpression;
exports.TemplateElement = TemplateElement;
exports.TemplateLiteral = TemplateLiteral;
function TaggedTemplateExpression(node) {
  this.print(node.tag, node);
  this.print(node.typeParameters, node);
  this.print(node.quasi, node);
}
function TemplateElement(node, parent) {
  const isFirst = parent.quasis[0] === node;
  const isLast = parent.quasis[parent.quasis.length - 1] === node;
  const value = (isFirst ? "`" : "}") + node.value.raw + (isLast ? "`" : "${");
  this.token(value, true);
}
function TemplateLiteral(node) {
  const quasis = node.quasis;
  for (let i = 0; i < quasis.length; i++) {
    this.print(quasis[i], node);
    if (i + 1 < quasis.length) {
      this.print(node.expressions[i], node);
    }
  }
}

//# sourceMappingURL=template-literals.js.map
