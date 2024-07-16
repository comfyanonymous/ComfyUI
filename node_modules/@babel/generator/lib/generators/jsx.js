"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.JSXAttribute = JSXAttribute;
exports.JSXClosingElement = JSXClosingElement;
exports.JSXClosingFragment = JSXClosingFragment;
exports.JSXElement = JSXElement;
exports.JSXEmptyExpression = JSXEmptyExpression;
exports.JSXExpressionContainer = JSXExpressionContainer;
exports.JSXFragment = JSXFragment;
exports.JSXIdentifier = JSXIdentifier;
exports.JSXMemberExpression = JSXMemberExpression;
exports.JSXNamespacedName = JSXNamespacedName;
exports.JSXOpeningElement = JSXOpeningElement;
exports.JSXOpeningFragment = JSXOpeningFragment;
exports.JSXSpreadAttribute = JSXSpreadAttribute;
exports.JSXSpreadChild = JSXSpreadChild;
exports.JSXText = JSXText;
function JSXAttribute(node) {
  this.print(node.name, node);
  if (node.value) {
    this.tokenChar(61);
    this.print(node.value, node);
  }
}
function JSXIdentifier(node) {
  this.word(node.name);
}
function JSXNamespacedName(node) {
  this.print(node.namespace, node);
  this.tokenChar(58);
  this.print(node.name, node);
}
function JSXMemberExpression(node) {
  this.print(node.object, node);
  this.tokenChar(46);
  this.print(node.property, node);
}
function JSXSpreadAttribute(node) {
  this.tokenChar(123);
  this.token("...");
  this.print(node.argument, node);
  this.tokenChar(125);
}
function JSXExpressionContainer(node) {
  this.tokenChar(123);
  this.print(node.expression, node);
  this.tokenChar(125);
}
function JSXSpreadChild(node) {
  this.tokenChar(123);
  this.token("...");
  this.print(node.expression, node);
  this.tokenChar(125);
}
function JSXText(node) {
  const raw = this.getPossibleRaw(node);
  if (raw !== undefined) {
    this.token(raw, true);
  } else {
    this.token(node.value, true);
  }
}
function JSXElement(node) {
  const open = node.openingElement;
  this.print(open, node);
  if (open.selfClosing) return;
  this.indent();
  for (const child of node.children) {
    this.print(child, node);
  }
  this.dedent();
  this.print(node.closingElement, node);
}
function spaceSeparator() {
  this.space();
}
function JSXOpeningElement(node) {
  this.tokenChar(60);
  this.print(node.name, node);
  this.print(node.typeParameters, node);
  if (node.attributes.length > 0) {
    this.space();
    this.printJoin(node.attributes, node, {
      separator: spaceSeparator
    });
  }
  if (node.selfClosing) {
    this.space();
    this.token("/>");
  } else {
    this.tokenChar(62);
  }
}
function JSXClosingElement(node) {
  this.token("</");
  this.print(node.name, node);
  this.tokenChar(62);
}
function JSXEmptyExpression() {
  this.printInnerComments();
}
function JSXFragment(node) {
  this.print(node.openingFragment, node);
  this.indent();
  for (const child of node.children) {
    this.print(child, node);
  }
  this.dedent();
  this.print(node.closingFragment, node);
}
function JSXOpeningFragment() {
  this.tokenChar(60);
  this.tokenChar(62);
}
function JSXClosingFragment() {
  this.token("</");
  this.tokenChar(62);
}

//# sourceMappingURL=jsx.js.map
