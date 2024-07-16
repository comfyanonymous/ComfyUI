"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.ArrowFunctionExpression = ArrowFunctionExpression;
exports.AssignmentExpression = AssignmentExpression;
exports.Binary = Binary;
exports.BinaryExpression = BinaryExpression;
exports.ClassExpression = ClassExpression;
exports.ConditionalExpression = ConditionalExpression;
exports.DoExpression = DoExpression;
exports.FunctionExpression = FunctionExpression;
exports.FunctionTypeAnnotation = FunctionTypeAnnotation;
exports.Identifier = Identifier;
exports.LogicalExpression = LogicalExpression;
exports.NullableTypeAnnotation = NullableTypeAnnotation;
exports.ObjectExpression = ObjectExpression;
exports.OptionalIndexedAccessType = OptionalIndexedAccessType;
exports.OptionalCallExpression = exports.OptionalMemberExpression = OptionalMemberExpression;
exports.SequenceExpression = SequenceExpression;
exports.TSTypeAssertion = exports.TSSatisfiesExpression = exports.TSAsExpression = TSAsExpression;
exports.TSInferType = TSInferType;
exports.TSInstantiationExpression = TSInstantiationExpression;
exports.TSIntersectionType = exports.TSUnionType = TSUnionType;
exports.UnaryLike = UnaryLike;
exports.IntersectionTypeAnnotation = exports.UnionTypeAnnotation = UnionTypeAnnotation;
exports.UpdateExpression = UpdateExpression;
exports.AwaitExpression = exports.YieldExpression = YieldExpression;
var _t = require("@babel/types");
const {
  isArrayTypeAnnotation,
  isArrowFunctionExpression,
  isBinaryExpression,
  isCallExpression,
  isExportDeclaration,
  isForOfStatement,
  isIndexedAccessType,
  isMemberExpression,
  isObjectPattern,
  isOptionalMemberExpression,
  isYieldExpression
} = _t;
const PRECEDENCE = new Map([["||", 0], ["??", 0], ["|>", 0], ["&&", 1], ["|", 2], ["^", 3], ["&", 4], ["==", 5], ["===", 5], ["!=", 5], ["!==", 5], ["<", 6], [">", 6], ["<=", 6], [">=", 6], ["in", 6], ["instanceof", 6], [">>", 7], ["<<", 7], [">>>", 7], ["+", 8], ["-", 8], ["*", 9], ["/", 9], ["%", 9], ["**", 10]]);
function isTSTypeExpression(nodeType) {
  return nodeType === "TSAsExpression" || nodeType === "TSSatisfiesExpression" || nodeType === "TSTypeAssertion";
}
const isClassExtendsClause = (node, parent) => {
  const parentType = parent.type;
  return (parentType === "ClassDeclaration" || parentType === "ClassExpression") && parent.superClass === node;
};
const hasPostfixPart = (node, parent) => {
  const parentType = parent.type;
  return (parentType === "MemberExpression" || parentType === "OptionalMemberExpression") && parent.object === node || (parentType === "CallExpression" || parentType === "OptionalCallExpression" || parentType === "NewExpression") && parent.callee === node || parentType === "TaggedTemplateExpression" && parent.tag === node || parentType === "TSNonNullExpression";
};
function NullableTypeAnnotation(node, parent) {
  return isArrayTypeAnnotation(parent);
}
function FunctionTypeAnnotation(node, parent, printStack) {
  if (printStack.length < 3) return;
  const parentType = parent.type;
  return parentType === "UnionTypeAnnotation" || parentType === "IntersectionTypeAnnotation" || parentType === "ArrayTypeAnnotation" || parentType === "TypeAnnotation" && isArrowFunctionExpression(printStack[printStack.length - 3]);
}
function UpdateExpression(node, parent) {
  return hasPostfixPart(node, parent) || isClassExtendsClause(node, parent);
}
function ObjectExpression(node, parent, printStack) {
  return isFirstInContext(printStack, 1 | 2);
}
function DoExpression(node, parent, printStack) {
  return !node.async && isFirstInContext(printStack, 1);
}
function Binary(node, parent) {
  const parentType = parent.type;
  if (node.operator === "**" && parentType === "BinaryExpression" && parent.operator === "**") {
    return parent.left === node;
  }
  if (isClassExtendsClause(node, parent)) {
    return true;
  }
  if (hasPostfixPart(node, parent) || parentType === "UnaryExpression" || parentType === "SpreadElement" || parentType === "AwaitExpression") {
    return true;
  }
  if (parentType === "BinaryExpression" || parentType === "LogicalExpression") {
    const parentPos = PRECEDENCE.get(parent.operator);
    const nodePos = PRECEDENCE.get(node.operator);
    if (parentPos === nodePos && parent.right === node && parentType !== "LogicalExpression" || parentPos > nodePos) {
      return true;
    }
  }
  return undefined;
}
function UnionTypeAnnotation(node, parent) {
  const parentType = parent.type;
  return parentType === "ArrayTypeAnnotation" || parentType === "NullableTypeAnnotation" || parentType === "IntersectionTypeAnnotation" || parentType === "UnionTypeAnnotation";
}
function OptionalIndexedAccessType(node, parent) {
  return isIndexedAccessType(parent) && parent.objectType === node;
}
function TSAsExpression() {
  return true;
}
function TSUnionType(node, parent) {
  const parentType = parent.type;
  return parentType === "TSArrayType" || parentType === "TSOptionalType" || parentType === "TSIntersectionType" || parentType === "TSUnionType" || parentType === "TSRestType";
}
function TSInferType(node, parent) {
  const parentType = parent.type;
  return parentType === "TSArrayType" || parentType === "TSOptionalType";
}
function TSInstantiationExpression(node, parent) {
  const parentType = parent.type;
  return (parentType === "CallExpression" || parentType === "OptionalCallExpression" || parentType === "NewExpression" || parentType === "TSInstantiationExpression") && !!parent.typeParameters;
}
function BinaryExpression(node, parent) {
  if (node.operator === "in") {
    const parentType = parent.type;
    return parentType === "VariableDeclarator" || parentType === "ForStatement" || parentType === "ForInStatement" || parentType === "ForOfStatement";
  }
  return false;
}
function SequenceExpression(node, parent) {
  const parentType = parent.type;
  if (parentType === "ForStatement" || parentType === "ThrowStatement" || parentType === "ReturnStatement" || parentType === "IfStatement" && parent.test === node || parentType === "WhileStatement" && parent.test === node || parentType === "ForInStatement" && parent.right === node || parentType === "SwitchStatement" && parent.discriminant === node || parentType === "ExpressionStatement" && parent.expression === node) {
    return false;
  }
  return true;
}
function YieldExpression(node, parent) {
  const parentType = parent.type;
  return parentType === "BinaryExpression" || parentType === "LogicalExpression" || parentType === "UnaryExpression" || parentType === "SpreadElement" || hasPostfixPart(node, parent) || parentType === "AwaitExpression" && isYieldExpression(node) || parentType === "ConditionalExpression" && node === parent.test || isClassExtendsClause(node, parent);
}
function ClassExpression(node, parent, printStack) {
  return isFirstInContext(printStack, 1 | 4);
}
function UnaryLike(node, parent) {
  return hasPostfixPart(node, parent) || isBinaryExpression(parent) && parent.operator === "**" && parent.left === node || isClassExtendsClause(node, parent);
}
function FunctionExpression(node, parent, printStack) {
  return isFirstInContext(printStack, 1 | 4);
}
function ArrowFunctionExpression(node, parent) {
  return isExportDeclaration(parent) || ConditionalExpression(node, parent);
}
function ConditionalExpression(node, parent) {
  const parentType = parent.type;
  if (parentType === "UnaryExpression" || parentType === "SpreadElement" || parentType === "BinaryExpression" || parentType === "LogicalExpression" || parentType === "ConditionalExpression" && parent.test === node || parentType === "AwaitExpression" || isTSTypeExpression(parentType)) {
    return true;
  }
  return UnaryLike(node, parent);
}
function OptionalMemberExpression(node, parent) {
  return isCallExpression(parent) && parent.callee === node || isMemberExpression(parent) && parent.object === node;
}
function AssignmentExpression(node, parent) {
  if (isObjectPattern(node.left)) {
    return true;
  } else {
    return ConditionalExpression(node, parent);
  }
}
function LogicalExpression(node, parent) {
  const parentType = parent.type;
  if (isTSTypeExpression(parentType)) return true;
  if (parentType !== "LogicalExpression") return false;
  switch (node.operator) {
    case "||":
      return parent.operator === "??" || parent.operator === "&&";
    case "&&":
      return parent.operator === "??";
    case "??":
      return parent.operator !== "??";
  }
}
function Identifier(node, parent, printStack) {
  var _node$extra;
  const parentType = parent.type;
  if ((_node$extra = node.extra) != null && _node$extra.parenthesized && parentType === "AssignmentExpression" && parent.left === node) {
    const rightType = parent.right.type;
    if ((rightType === "FunctionExpression" || rightType === "ClassExpression") && parent.right.id == null) {
      return true;
    }
  }
  if (node.name === "let") {
    const isFollowedByBracket = isMemberExpression(parent, {
      object: node,
      computed: true
    }) || isOptionalMemberExpression(parent, {
      object: node,
      computed: true,
      optional: false
    });
    return isFirstInContext(printStack, isFollowedByBracket ? 1 | 8 | 16 | 32 : 32);
  }
  return node.name === "async" && isForOfStatement(parent) && node === parent.left;
}
function isFirstInContext(printStack, checkParam) {
  const expressionStatement = checkParam & 1;
  const arrowBody = checkParam & 2;
  const exportDefault = checkParam & 4;
  const forHead = checkParam & 8;
  const forInHead = checkParam & 16;
  const forOfHead = checkParam & 32;
  let i = printStack.length - 1;
  if (i <= 0) return;
  let node = printStack[i];
  i--;
  let parent = printStack[i];
  while (i >= 0) {
    const parentType = parent.type;
    if (expressionStatement && parentType === "ExpressionStatement" && parent.expression === node || exportDefault && parentType === "ExportDefaultDeclaration" && node === parent.declaration || arrowBody && parentType === "ArrowFunctionExpression" && parent.body === node || forHead && parentType === "ForStatement" && parent.init === node || forInHead && parentType === "ForInStatement" && parent.left === node || forOfHead && parentType === "ForOfStatement" && parent.left === node) {
      return true;
    }
    if (i > 0 && (hasPostfixPart(node, parent) && parentType !== "NewExpression" || parentType === "SequenceExpression" && parent.expressions[0] === node || parentType === "UpdateExpression" && !parent.prefix || parentType === "ConditionalExpression" && parent.test === node || (parentType === "BinaryExpression" || parentType === "LogicalExpression") && parent.left === node || parentType === "AssignmentExpression" && parent.left === node)) {
      node = parent;
      i--;
      parent = printStack[i];
    } else {
      return false;
    }
  }
  return false;
}

//# sourceMappingURL=parentheses.js.map
