"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.anyTypeAnnotation = anyTypeAnnotation;
exports.argumentPlaceholder = argumentPlaceholder;
exports.arrayExpression = arrayExpression;
exports.arrayPattern = arrayPattern;
exports.arrayTypeAnnotation = arrayTypeAnnotation;
exports.arrowFunctionExpression = arrowFunctionExpression;
exports.assignmentExpression = assignmentExpression;
exports.assignmentPattern = assignmentPattern;
exports.awaitExpression = awaitExpression;
exports.bigIntLiteral = bigIntLiteral;
exports.binaryExpression = binaryExpression;
exports.bindExpression = bindExpression;
exports.blockStatement = blockStatement;
exports.booleanLiteral = booleanLiteral;
exports.booleanLiteralTypeAnnotation = booleanLiteralTypeAnnotation;
exports.booleanTypeAnnotation = booleanTypeAnnotation;
exports.breakStatement = breakStatement;
exports.callExpression = callExpression;
exports.catchClause = catchClause;
exports.classAccessorProperty = classAccessorProperty;
exports.classBody = classBody;
exports.classDeclaration = classDeclaration;
exports.classExpression = classExpression;
exports.classImplements = classImplements;
exports.classMethod = classMethod;
exports.classPrivateMethod = classPrivateMethod;
exports.classPrivateProperty = classPrivateProperty;
exports.classProperty = classProperty;
exports.conditionalExpression = conditionalExpression;
exports.continueStatement = continueStatement;
exports.debuggerStatement = debuggerStatement;
exports.decimalLiteral = decimalLiteral;
exports.declareClass = declareClass;
exports.declareExportAllDeclaration = declareExportAllDeclaration;
exports.declareExportDeclaration = declareExportDeclaration;
exports.declareFunction = declareFunction;
exports.declareInterface = declareInterface;
exports.declareModule = declareModule;
exports.declareModuleExports = declareModuleExports;
exports.declareOpaqueType = declareOpaqueType;
exports.declareTypeAlias = declareTypeAlias;
exports.declareVariable = declareVariable;
exports.declaredPredicate = declaredPredicate;
exports.decorator = decorator;
exports.directive = directive;
exports.directiveLiteral = directiveLiteral;
exports.doExpression = doExpression;
exports.doWhileStatement = doWhileStatement;
exports.emptyStatement = emptyStatement;
exports.emptyTypeAnnotation = emptyTypeAnnotation;
exports.enumBooleanBody = enumBooleanBody;
exports.enumBooleanMember = enumBooleanMember;
exports.enumDeclaration = enumDeclaration;
exports.enumDefaultedMember = enumDefaultedMember;
exports.enumNumberBody = enumNumberBody;
exports.enumNumberMember = enumNumberMember;
exports.enumStringBody = enumStringBody;
exports.enumStringMember = enumStringMember;
exports.enumSymbolBody = enumSymbolBody;
exports.existsTypeAnnotation = existsTypeAnnotation;
exports.exportAllDeclaration = exportAllDeclaration;
exports.exportDefaultDeclaration = exportDefaultDeclaration;
exports.exportDefaultSpecifier = exportDefaultSpecifier;
exports.exportNamedDeclaration = exportNamedDeclaration;
exports.exportNamespaceSpecifier = exportNamespaceSpecifier;
exports.exportSpecifier = exportSpecifier;
exports.expressionStatement = expressionStatement;
exports.file = file;
exports.forInStatement = forInStatement;
exports.forOfStatement = forOfStatement;
exports.forStatement = forStatement;
exports.functionDeclaration = functionDeclaration;
exports.functionExpression = functionExpression;
exports.functionTypeAnnotation = functionTypeAnnotation;
exports.functionTypeParam = functionTypeParam;
exports.genericTypeAnnotation = genericTypeAnnotation;
exports.identifier = identifier;
exports.ifStatement = ifStatement;
exports.import = _import;
exports.importAttribute = importAttribute;
exports.importDeclaration = importDeclaration;
exports.importDefaultSpecifier = importDefaultSpecifier;
exports.importExpression = importExpression;
exports.importNamespaceSpecifier = importNamespaceSpecifier;
exports.importSpecifier = importSpecifier;
exports.indexedAccessType = indexedAccessType;
exports.inferredPredicate = inferredPredicate;
exports.interfaceDeclaration = interfaceDeclaration;
exports.interfaceExtends = interfaceExtends;
exports.interfaceTypeAnnotation = interfaceTypeAnnotation;
exports.interpreterDirective = interpreterDirective;
exports.intersectionTypeAnnotation = intersectionTypeAnnotation;
exports.jSXAttribute = exports.jsxAttribute = jsxAttribute;
exports.jSXClosingElement = exports.jsxClosingElement = jsxClosingElement;
exports.jSXClosingFragment = exports.jsxClosingFragment = jsxClosingFragment;
exports.jSXElement = exports.jsxElement = jsxElement;
exports.jSXEmptyExpression = exports.jsxEmptyExpression = jsxEmptyExpression;
exports.jSXExpressionContainer = exports.jsxExpressionContainer = jsxExpressionContainer;
exports.jSXFragment = exports.jsxFragment = jsxFragment;
exports.jSXIdentifier = exports.jsxIdentifier = jsxIdentifier;
exports.jSXMemberExpression = exports.jsxMemberExpression = jsxMemberExpression;
exports.jSXNamespacedName = exports.jsxNamespacedName = jsxNamespacedName;
exports.jSXOpeningElement = exports.jsxOpeningElement = jsxOpeningElement;
exports.jSXOpeningFragment = exports.jsxOpeningFragment = jsxOpeningFragment;
exports.jSXSpreadAttribute = exports.jsxSpreadAttribute = jsxSpreadAttribute;
exports.jSXSpreadChild = exports.jsxSpreadChild = jsxSpreadChild;
exports.jSXText = exports.jsxText = jsxText;
exports.labeledStatement = labeledStatement;
exports.logicalExpression = logicalExpression;
exports.memberExpression = memberExpression;
exports.metaProperty = metaProperty;
exports.mixedTypeAnnotation = mixedTypeAnnotation;
exports.moduleExpression = moduleExpression;
exports.newExpression = newExpression;
exports.noop = noop;
exports.nullLiteral = nullLiteral;
exports.nullLiteralTypeAnnotation = nullLiteralTypeAnnotation;
exports.nullableTypeAnnotation = nullableTypeAnnotation;
exports.numberLiteral = NumberLiteral;
exports.numberLiteralTypeAnnotation = numberLiteralTypeAnnotation;
exports.numberTypeAnnotation = numberTypeAnnotation;
exports.numericLiteral = numericLiteral;
exports.objectExpression = objectExpression;
exports.objectMethod = objectMethod;
exports.objectPattern = objectPattern;
exports.objectProperty = objectProperty;
exports.objectTypeAnnotation = objectTypeAnnotation;
exports.objectTypeCallProperty = objectTypeCallProperty;
exports.objectTypeIndexer = objectTypeIndexer;
exports.objectTypeInternalSlot = objectTypeInternalSlot;
exports.objectTypeProperty = objectTypeProperty;
exports.objectTypeSpreadProperty = objectTypeSpreadProperty;
exports.opaqueType = opaqueType;
exports.optionalCallExpression = optionalCallExpression;
exports.optionalIndexedAccessType = optionalIndexedAccessType;
exports.optionalMemberExpression = optionalMemberExpression;
exports.parenthesizedExpression = parenthesizedExpression;
exports.pipelineBareFunction = pipelineBareFunction;
exports.pipelinePrimaryTopicReference = pipelinePrimaryTopicReference;
exports.pipelineTopicExpression = pipelineTopicExpression;
exports.placeholder = placeholder;
exports.privateName = privateName;
exports.program = program;
exports.qualifiedTypeIdentifier = qualifiedTypeIdentifier;
exports.recordExpression = recordExpression;
exports.regExpLiteral = regExpLiteral;
exports.regexLiteral = RegexLiteral;
exports.restElement = restElement;
exports.restProperty = RestProperty;
exports.returnStatement = returnStatement;
exports.sequenceExpression = sequenceExpression;
exports.spreadElement = spreadElement;
exports.spreadProperty = SpreadProperty;
exports.staticBlock = staticBlock;
exports.stringLiteral = stringLiteral;
exports.stringLiteralTypeAnnotation = stringLiteralTypeAnnotation;
exports.stringTypeAnnotation = stringTypeAnnotation;
exports.super = _super;
exports.switchCase = switchCase;
exports.switchStatement = switchStatement;
exports.symbolTypeAnnotation = symbolTypeAnnotation;
exports.taggedTemplateExpression = taggedTemplateExpression;
exports.templateElement = templateElement;
exports.templateLiteral = templateLiteral;
exports.thisExpression = thisExpression;
exports.thisTypeAnnotation = thisTypeAnnotation;
exports.throwStatement = throwStatement;
exports.topicReference = topicReference;
exports.tryStatement = tryStatement;
exports.tSAnyKeyword = exports.tsAnyKeyword = tsAnyKeyword;
exports.tSArrayType = exports.tsArrayType = tsArrayType;
exports.tSAsExpression = exports.tsAsExpression = tsAsExpression;
exports.tSBigIntKeyword = exports.tsBigIntKeyword = tsBigIntKeyword;
exports.tSBooleanKeyword = exports.tsBooleanKeyword = tsBooleanKeyword;
exports.tSCallSignatureDeclaration = exports.tsCallSignatureDeclaration = tsCallSignatureDeclaration;
exports.tSConditionalType = exports.tsConditionalType = tsConditionalType;
exports.tSConstructSignatureDeclaration = exports.tsConstructSignatureDeclaration = tsConstructSignatureDeclaration;
exports.tSConstructorType = exports.tsConstructorType = tsConstructorType;
exports.tSDeclareFunction = exports.tsDeclareFunction = tsDeclareFunction;
exports.tSDeclareMethod = exports.tsDeclareMethod = tsDeclareMethod;
exports.tSEnumDeclaration = exports.tsEnumDeclaration = tsEnumDeclaration;
exports.tSEnumMember = exports.tsEnumMember = tsEnumMember;
exports.tSExportAssignment = exports.tsExportAssignment = tsExportAssignment;
exports.tSExpressionWithTypeArguments = exports.tsExpressionWithTypeArguments = tsExpressionWithTypeArguments;
exports.tSExternalModuleReference = exports.tsExternalModuleReference = tsExternalModuleReference;
exports.tSFunctionType = exports.tsFunctionType = tsFunctionType;
exports.tSImportEqualsDeclaration = exports.tsImportEqualsDeclaration = tsImportEqualsDeclaration;
exports.tSImportType = exports.tsImportType = tsImportType;
exports.tSIndexSignature = exports.tsIndexSignature = tsIndexSignature;
exports.tSIndexedAccessType = exports.tsIndexedAccessType = tsIndexedAccessType;
exports.tSInferType = exports.tsInferType = tsInferType;
exports.tSInstantiationExpression = exports.tsInstantiationExpression = tsInstantiationExpression;
exports.tSInterfaceBody = exports.tsInterfaceBody = tsInterfaceBody;
exports.tSInterfaceDeclaration = exports.tsInterfaceDeclaration = tsInterfaceDeclaration;
exports.tSIntersectionType = exports.tsIntersectionType = tsIntersectionType;
exports.tSIntrinsicKeyword = exports.tsIntrinsicKeyword = tsIntrinsicKeyword;
exports.tSLiteralType = exports.tsLiteralType = tsLiteralType;
exports.tSMappedType = exports.tsMappedType = tsMappedType;
exports.tSMethodSignature = exports.tsMethodSignature = tsMethodSignature;
exports.tSModuleBlock = exports.tsModuleBlock = tsModuleBlock;
exports.tSModuleDeclaration = exports.tsModuleDeclaration = tsModuleDeclaration;
exports.tSNamedTupleMember = exports.tsNamedTupleMember = tsNamedTupleMember;
exports.tSNamespaceExportDeclaration = exports.tsNamespaceExportDeclaration = tsNamespaceExportDeclaration;
exports.tSNeverKeyword = exports.tsNeverKeyword = tsNeverKeyword;
exports.tSNonNullExpression = exports.tsNonNullExpression = tsNonNullExpression;
exports.tSNullKeyword = exports.tsNullKeyword = tsNullKeyword;
exports.tSNumberKeyword = exports.tsNumberKeyword = tsNumberKeyword;
exports.tSObjectKeyword = exports.tsObjectKeyword = tsObjectKeyword;
exports.tSOptionalType = exports.tsOptionalType = tsOptionalType;
exports.tSParameterProperty = exports.tsParameterProperty = tsParameterProperty;
exports.tSParenthesizedType = exports.tsParenthesizedType = tsParenthesizedType;
exports.tSPropertySignature = exports.tsPropertySignature = tsPropertySignature;
exports.tSQualifiedName = exports.tsQualifiedName = tsQualifiedName;
exports.tSRestType = exports.tsRestType = tsRestType;
exports.tSSatisfiesExpression = exports.tsSatisfiesExpression = tsSatisfiesExpression;
exports.tSStringKeyword = exports.tsStringKeyword = tsStringKeyword;
exports.tSSymbolKeyword = exports.tsSymbolKeyword = tsSymbolKeyword;
exports.tSThisType = exports.tsThisType = tsThisType;
exports.tSTupleType = exports.tsTupleType = tsTupleType;
exports.tSTypeAliasDeclaration = exports.tsTypeAliasDeclaration = tsTypeAliasDeclaration;
exports.tSTypeAnnotation = exports.tsTypeAnnotation = tsTypeAnnotation;
exports.tSTypeAssertion = exports.tsTypeAssertion = tsTypeAssertion;
exports.tSTypeLiteral = exports.tsTypeLiteral = tsTypeLiteral;
exports.tSTypeOperator = exports.tsTypeOperator = tsTypeOperator;
exports.tSTypeParameter = exports.tsTypeParameter = tsTypeParameter;
exports.tSTypeParameterDeclaration = exports.tsTypeParameterDeclaration = tsTypeParameterDeclaration;
exports.tSTypeParameterInstantiation = exports.tsTypeParameterInstantiation = tsTypeParameterInstantiation;
exports.tSTypePredicate = exports.tsTypePredicate = tsTypePredicate;
exports.tSTypeQuery = exports.tsTypeQuery = tsTypeQuery;
exports.tSTypeReference = exports.tsTypeReference = tsTypeReference;
exports.tSUndefinedKeyword = exports.tsUndefinedKeyword = tsUndefinedKeyword;
exports.tSUnionType = exports.tsUnionType = tsUnionType;
exports.tSUnknownKeyword = exports.tsUnknownKeyword = tsUnknownKeyword;
exports.tSVoidKeyword = exports.tsVoidKeyword = tsVoidKeyword;
exports.tupleExpression = tupleExpression;
exports.tupleTypeAnnotation = tupleTypeAnnotation;
exports.typeAlias = typeAlias;
exports.typeAnnotation = typeAnnotation;
exports.typeCastExpression = typeCastExpression;
exports.typeParameter = typeParameter;
exports.typeParameterDeclaration = typeParameterDeclaration;
exports.typeParameterInstantiation = typeParameterInstantiation;
exports.typeofTypeAnnotation = typeofTypeAnnotation;
exports.unaryExpression = unaryExpression;
exports.unionTypeAnnotation = unionTypeAnnotation;
exports.updateExpression = updateExpression;
exports.v8IntrinsicIdentifier = v8IntrinsicIdentifier;
exports.variableDeclaration = variableDeclaration;
exports.variableDeclarator = variableDeclarator;
exports.variance = variance;
exports.voidTypeAnnotation = voidTypeAnnotation;
exports.whileStatement = whileStatement;
exports.withStatement = withStatement;
exports.yieldExpression = yieldExpression;
var _validateNode = require("../validateNode.js");
var _deprecationWarning = require("../../utils/deprecationWarning.js");
function arrayExpression(elements = []) {
  return (0, _validateNode.default)({
    type: "ArrayExpression",
    elements
  });
}
function assignmentExpression(operator, left, right) {
  return (0, _validateNode.default)({
    type: "AssignmentExpression",
    operator,
    left,
    right
  });
}
function binaryExpression(operator, left, right) {
  return (0, _validateNode.default)({
    type: "BinaryExpression",
    operator,
    left,
    right
  });
}
function interpreterDirective(value) {
  return (0, _validateNode.default)({
    type: "InterpreterDirective",
    value
  });
}
function directive(value) {
  return (0, _validateNode.default)({
    type: "Directive",
    value
  });
}
function directiveLiteral(value) {
  return (0, _validateNode.default)({
    type: "DirectiveLiteral",
    value
  });
}
function blockStatement(body, directives = []) {
  return (0, _validateNode.default)({
    type: "BlockStatement",
    body,
    directives
  });
}
function breakStatement(label = null) {
  return (0, _validateNode.default)({
    type: "BreakStatement",
    label
  });
}
function callExpression(callee, _arguments) {
  return (0, _validateNode.default)({
    type: "CallExpression",
    callee,
    arguments: _arguments
  });
}
function catchClause(param = null, body) {
  return (0, _validateNode.default)({
    type: "CatchClause",
    param,
    body
  });
}
function conditionalExpression(test, consequent, alternate) {
  return (0, _validateNode.default)({
    type: "ConditionalExpression",
    test,
    consequent,
    alternate
  });
}
function continueStatement(label = null) {
  return (0, _validateNode.default)({
    type: "ContinueStatement",
    label
  });
}
function debuggerStatement() {
  return {
    type: "DebuggerStatement"
  };
}
function doWhileStatement(test, body) {
  return (0, _validateNode.default)({
    type: "DoWhileStatement",
    test,
    body
  });
}
function emptyStatement() {
  return {
    type: "EmptyStatement"
  };
}
function expressionStatement(expression) {
  return (0, _validateNode.default)({
    type: "ExpressionStatement",
    expression
  });
}
function file(program, comments = null, tokens = null) {
  return (0, _validateNode.default)({
    type: "File",
    program,
    comments,
    tokens
  });
}
function forInStatement(left, right, body) {
  return (0, _validateNode.default)({
    type: "ForInStatement",
    left,
    right,
    body
  });
}
function forStatement(init = null, test = null, update = null, body) {
  return (0, _validateNode.default)({
    type: "ForStatement",
    init,
    test,
    update,
    body
  });
}
function functionDeclaration(id = null, params, body, generator = false, async = false) {
  return (0, _validateNode.default)({
    type: "FunctionDeclaration",
    id,
    params,
    body,
    generator,
    async
  });
}
function functionExpression(id = null, params, body, generator = false, async = false) {
  return (0, _validateNode.default)({
    type: "FunctionExpression",
    id,
    params,
    body,
    generator,
    async
  });
}
function identifier(name) {
  return (0, _validateNode.default)({
    type: "Identifier",
    name
  });
}
function ifStatement(test, consequent, alternate = null) {
  return (0, _validateNode.default)({
    type: "IfStatement",
    test,
    consequent,
    alternate
  });
}
function labeledStatement(label, body) {
  return (0, _validateNode.default)({
    type: "LabeledStatement",
    label,
    body
  });
}
function stringLiteral(value) {
  return (0, _validateNode.default)({
    type: "StringLiteral",
    value
  });
}
function numericLiteral(value) {
  return (0, _validateNode.default)({
    type: "NumericLiteral",
    value
  });
}
function nullLiteral() {
  return {
    type: "NullLiteral"
  };
}
function booleanLiteral(value) {
  return (0, _validateNode.default)({
    type: "BooleanLiteral",
    value
  });
}
function regExpLiteral(pattern, flags = "") {
  return (0, _validateNode.default)({
    type: "RegExpLiteral",
    pattern,
    flags
  });
}
function logicalExpression(operator, left, right) {
  return (0, _validateNode.default)({
    type: "LogicalExpression",
    operator,
    left,
    right
  });
}
function memberExpression(object, property, computed = false, optional = null) {
  return (0, _validateNode.default)({
    type: "MemberExpression",
    object,
    property,
    computed,
    optional
  });
}
function newExpression(callee, _arguments) {
  return (0, _validateNode.default)({
    type: "NewExpression",
    callee,
    arguments: _arguments
  });
}
function program(body, directives = [], sourceType = "script", interpreter = null) {
  return (0, _validateNode.default)({
    type: "Program",
    body,
    directives,
    sourceType,
    interpreter
  });
}
function objectExpression(properties) {
  return (0, _validateNode.default)({
    type: "ObjectExpression",
    properties
  });
}
function objectMethod(kind = "method", key, params, body, computed = false, generator = false, async = false) {
  return (0, _validateNode.default)({
    type: "ObjectMethod",
    kind,
    key,
    params,
    body,
    computed,
    generator,
    async
  });
}
function objectProperty(key, value, computed = false, shorthand = false, decorators = null) {
  return (0, _validateNode.default)({
    type: "ObjectProperty",
    key,
    value,
    computed,
    shorthand,
    decorators
  });
}
function restElement(argument) {
  return (0, _validateNode.default)({
    type: "RestElement",
    argument
  });
}
function returnStatement(argument = null) {
  return (0, _validateNode.default)({
    type: "ReturnStatement",
    argument
  });
}
function sequenceExpression(expressions) {
  return (0, _validateNode.default)({
    type: "SequenceExpression",
    expressions
  });
}
function parenthesizedExpression(expression) {
  return (0, _validateNode.default)({
    type: "ParenthesizedExpression",
    expression
  });
}
function switchCase(test = null, consequent) {
  return (0, _validateNode.default)({
    type: "SwitchCase",
    test,
    consequent
  });
}
function switchStatement(discriminant, cases) {
  return (0, _validateNode.default)({
    type: "SwitchStatement",
    discriminant,
    cases
  });
}
function thisExpression() {
  return {
    type: "ThisExpression"
  };
}
function throwStatement(argument) {
  return (0, _validateNode.default)({
    type: "ThrowStatement",
    argument
  });
}
function tryStatement(block, handler = null, finalizer = null) {
  return (0, _validateNode.default)({
    type: "TryStatement",
    block,
    handler,
    finalizer
  });
}
function unaryExpression(operator, argument, prefix = true) {
  return (0, _validateNode.default)({
    type: "UnaryExpression",
    operator,
    argument,
    prefix
  });
}
function updateExpression(operator, argument, prefix = false) {
  return (0, _validateNode.default)({
    type: "UpdateExpression",
    operator,
    argument,
    prefix
  });
}
function variableDeclaration(kind, declarations) {
  return (0, _validateNode.default)({
    type: "VariableDeclaration",
    kind,
    declarations
  });
}
function variableDeclarator(id, init = null) {
  return (0, _validateNode.default)({
    type: "VariableDeclarator",
    id,
    init
  });
}
function whileStatement(test, body) {
  return (0, _validateNode.default)({
    type: "WhileStatement",
    test,
    body
  });
}
function withStatement(object, body) {
  return (0, _validateNode.default)({
    type: "WithStatement",
    object,
    body
  });
}
function assignmentPattern(left, right) {
  return (0, _validateNode.default)({
    type: "AssignmentPattern",
    left,
    right
  });
}
function arrayPattern(elements) {
  return (0, _validateNode.default)({
    type: "ArrayPattern",
    elements
  });
}
function arrowFunctionExpression(params, body, async = false) {
  return (0, _validateNode.default)({
    type: "ArrowFunctionExpression",
    params,
    body,
    async,
    expression: null
  });
}
function classBody(body) {
  return (0, _validateNode.default)({
    type: "ClassBody",
    body
  });
}
function classExpression(id = null, superClass = null, body, decorators = null) {
  return (0, _validateNode.default)({
    type: "ClassExpression",
    id,
    superClass,
    body,
    decorators
  });
}
function classDeclaration(id = null, superClass = null, body, decorators = null) {
  return (0, _validateNode.default)({
    type: "ClassDeclaration",
    id,
    superClass,
    body,
    decorators
  });
}
function exportAllDeclaration(source) {
  return (0, _validateNode.default)({
    type: "ExportAllDeclaration",
    source
  });
}
function exportDefaultDeclaration(declaration) {
  return (0, _validateNode.default)({
    type: "ExportDefaultDeclaration",
    declaration
  });
}
function exportNamedDeclaration(declaration = null, specifiers = [], source = null) {
  return (0, _validateNode.default)({
    type: "ExportNamedDeclaration",
    declaration,
    specifiers,
    source
  });
}
function exportSpecifier(local, exported) {
  return (0, _validateNode.default)({
    type: "ExportSpecifier",
    local,
    exported
  });
}
function forOfStatement(left, right, body, _await = false) {
  return (0, _validateNode.default)({
    type: "ForOfStatement",
    left,
    right,
    body,
    await: _await
  });
}
function importDeclaration(specifiers, source) {
  return (0, _validateNode.default)({
    type: "ImportDeclaration",
    specifiers,
    source
  });
}
function importDefaultSpecifier(local) {
  return (0, _validateNode.default)({
    type: "ImportDefaultSpecifier",
    local
  });
}
function importNamespaceSpecifier(local) {
  return (0, _validateNode.default)({
    type: "ImportNamespaceSpecifier",
    local
  });
}
function importSpecifier(local, imported) {
  return (0, _validateNode.default)({
    type: "ImportSpecifier",
    local,
    imported
  });
}
function importExpression(source, options = null) {
  return (0, _validateNode.default)({
    type: "ImportExpression",
    source,
    options
  });
}
function metaProperty(meta, property) {
  return (0, _validateNode.default)({
    type: "MetaProperty",
    meta,
    property
  });
}
function classMethod(kind = "method", key, params, body, computed = false, _static = false, generator = false, async = false) {
  return (0, _validateNode.default)({
    type: "ClassMethod",
    kind,
    key,
    params,
    body,
    computed,
    static: _static,
    generator,
    async
  });
}
function objectPattern(properties) {
  return (0, _validateNode.default)({
    type: "ObjectPattern",
    properties
  });
}
function spreadElement(argument) {
  return (0, _validateNode.default)({
    type: "SpreadElement",
    argument
  });
}
function _super() {
  return {
    type: "Super"
  };
}
function taggedTemplateExpression(tag, quasi) {
  return (0, _validateNode.default)({
    type: "TaggedTemplateExpression",
    tag,
    quasi
  });
}
function templateElement(value, tail = false) {
  return (0, _validateNode.default)({
    type: "TemplateElement",
    value,
    tail
  });
}
function templateLiteral(quasis, expressions) {
  return (0, _validateNode.default)({
    type: "TemplateLiteral",
    quasis,
    expressions
  });
}
function yieldExpression(argument = null, delegate = false) {
  return (0, _validateNode.default)({
    type: "YieldExpression",
    argument,
    delegate
  });
}
function awaitExpression(argument) {
  return (0, _validateNode.default)({
    type: "AwaitExpression",
    argument
  });
}
function _import() {
  return {
    type: "Import"
  };
}
function bigIntLiteral(value) {
  return (0, _validateNode.default)({
    type: "BigIntLiteral",
    value
  });
}
function exportNamespaceSpecifier(exported) {
  return (0, _validateNode.default)({
    type: "ExportNamespaceSpecifier",
    exported
  });
}
function optionalMemberExpression(object, property, computed = false, optional) {
  return (0, _validateNode.default)({
    type: "OptionalMemberExpression",
    object,
    property,
    computed,
    optional
  });
}
function optionalCallExpression(callee, _arguments, optional) {
  return (0, _validateNode.default)({
    type: "OptionalCallExpression",
    callee,
    arguments: _arguments,
    optional
  });
}
function classProperty(key, value = null, typeAnnotation = null, decorators = null, computed = false, _static = false) {
  return (0, _validateNode.default)({
    type: "ClassProperty",
    key,
    value,
    typeAnnotation,
    decorators,
    computed,
    static: _static
  });
}
function classAccessorProperty(key, value = null, typeAnnotation = null, decorators = null, computed = false, _static = false) {
  return (0, _validateNode.default)({
    type: "ClassAccessorProperty",
    key,
    value,
    typeAnnotation,
    decorators,
    computed,
    static: _static
  });
}
function classPrivateProperty(key, value = null, decorators = null, _static = false) {
  return (0, _validateNode.default)({
    type: "ClassPrivateProperty",
    key,
    value,
    decorators,
    static: _static
  });
}
function classPrivateMethod(kind = "method", key, params, body, _static = false) {
  return (0, _validateNode.default)({
    type: "ClassPrivateMethod",
    kind,
    key,
    params,
    body,
    static: _static
  });
}
function privateName(id) {
  return (0, _validateNode.default)({
    type: "PrivateName",
    id
  });
}
function staticBlock(body) {
  return (0, _validateNode.default)({
    type: "StaticBlock",
    body
  });
}
function anyTypeAnnotation() {
  return {
    type: "AnyTypeAnnotation"
  };
}
function arrayTypeAnnotation(elementType) {
  return (0, _validateNode.default)({
    type: "ArrayTypeAnnotation",
    elementType
  });
}
function booleanTypeAnnotation() {
  return {
    type: "BooleanTypeAnnotation"
  };
}
function booleanLiteralTypeAnnotation(value) {
  return (0, _validateNode.default)({
    type: "BooleanLiteralTypeAnnotation",
    value
  });
}
function nullLiteralTypeAnnotation() {
  return {
    type: "NullLiteralTypeAnnotation"
  };
}
function classImplements(id, typeParameters = null) {
  return (0, _validateNode.default)({
    type: "ClassImplements",
    id,
    typeParameters
  });
}
function declareClass(id, typeParameters = null, _extends = null, body) {
  return (0, _validateNode.default)({
    type: "DeclareClass",
    id,
    typeParameters,
    extends: _extends,
    body
  });
}
function declareFunction(id) {
  return (0, _validateNode.default)({
    type: "DeclareFunction",
    id
  });
}
function declareInterface(id, typeParameters = null, _extends = null, body) {
  return (0, _validateNode.default)({
    type: "DeclareInterface",
    id,
    typeParameters,
    extends: _extends,
    body
  });
}
function declareModule(id, body, kind = null) {
  return (0, _validateNode.default)({
    type: "DeclareModule",
    id,
    body,
    kind
  });
}
function declareModuleExports(typeAnnotation) {
  return (0, _validateNode.default)({
    type: "DeclareModuleExports",
    typeAnnotation
  });
}
function declareTypeAlias(id, typeParameters = null, right) {
  return (0, _validateNode.default)({
    type: "DeclareTypeAlias",
    id,
    typeParameters,
    right
  });
}
function declareOpaqueType(id, typeParameters = null, supertype = null) {
  return (0, _validateNode.default)({
    type: "DeclareOpaqueType",
    id,
    typeParameters,
    supertype
  });
}
function declareVariable(id) {
  return (0, _validateNode.default)({
    type: "DeclareVariable",
    id
  });
}
function declareExportDeclaration(declaration = null, specifiers = null, source = null) {
  return (0, _validateNode.default)({
    type: "DeclareExportDeclaration",
    declaration,
    specifiers,
    source
  });
}
function declareExportAllDeclaration(source) {
  return (0, _validateNode.default)({
    type: "DeclareExportAllDeclaration",
    source
  });
}
function declaredPredicate(value) {
  return (0, _validateNode.default)({
    type: "DeclaredPredicate",
    value
  });
}
function existsTypeAnnotation() {
  return {
    type: "ExistsTypeAnnotation"
  };
}
function functionTypeAnnotation(typeParameters = null, params, rest = null, returnType) {
  return (0, _validateNode.default)({
    type: "FunctionTypeAnnotation",
    typeParameters,
    params,
    rest,
    returnType
  });
}
function functionTypeParam(name = null, typeAnnotation) {
  return (0, _validateNode.default)({
    type: "FunctionTypeParam",
    name,
    typeAnnotation
  });
}
function genericTypeAnnotation(id, typeParameters = null) {
  return (0, _validateNode.default)({
    type: "GenericTypeAnnotation",
    id,
    typeParameters
  });
}
function inferredPredicate() {
  return {
    type: "InferredPredicate"
  };
}
function interfaceExtends(id, typeParameters = null) {
  return (0, _validateNode.default)({
    type: "InterfaceExtends",
    id,
    typeParameters
  });
}
function interfaceDeclaration(id, typeParameters = null, _extends = null, body) {
  return (0, _validateNode.default)({
    type: "InterfaceDeclaration",
    id,
    typeParameters,
    extends: _extends,
    body
  });
}
function interfaceTypeAnnotation(_extends = null, body) {
  return (0, _validateNode.default)({
    type: "InterfaceTypeAnnotation",
    extends: _extends,
    body
  });
}
function intersectionTypeAnnotation(types) {
  return (0, _validateNode.default)({
    type: "IntersectionTypeAnnotation",
    types
  });
}
function mixedTypeAnnotation() {
  return {
    type: "MixedTypeAnnotation"
  };
}
function emptyTypeAnnotation() {
  return {
    type: "EmptyTypeAnnotation"
  };
}
function nullableTypeAnnotation(typeAnnotation) {
  return (0, _validateNode.default)({
    type: "NullableTypeAnnotation",
    typeAnnotation
  });
}
function numberLiteralTypeAnnotation(value) {
  return (0, _validateNode.default)({
    type: "NumberLiteralTypeAnnotation",
    value
  });
}
function numberTypeAnnotation() {
  return {
    type: "NumberTypeAnnotation"
  };
}
function objectTypeAnnotation(properties, indexers = [], callProperties = [], internalSlots = [], exact = false) {
  return (0, _validateNode.default)({
    type: "ObjectTypeAnnotation",
    properties,
    indexers,
    callProperties,
    internalSlots,
    exact
  });
}
function objectTypeInternalSlot(id, value, optional, _static, method) {
  return (0, _validateNode.default)({
    type: "ObjectTypeInternalSlot",
    id,
    value,
    optional,
    static: _static,
    method
  });
}
function objectTypeCallProperty(value) {
  return (0, _validateNode.default)({
    type: "ObjectTypeCallProperty",
    value,
    static: null
  });
}
function objectTypeIndexer(id = null, key, value, variance = null) {
  return (0, _validateNode.default)({
    type: "ObjectTypeIndexer",
    id,
    key,
    value,
    variance,
    static: null
  });
}
function objectTypeProperty(key, value, variance = null) {
  return (0, _validateNode.default)({
    type: "ObjectTypeProperty",
    key,
    value,
    variance,
    kind: null,
    method: null,
    optional: null,
    proto: null,
    static: null
  });
}
function objectTypeSpreadProperty(argument) {
  return (0, _validateNode.default)({
    type: "ObjectTypeSpreadProperty",
    argument
  });
}
function opaqueType(id, typeParameters = null, supertype = null, impltype) {
  return (0, _validateNode.default)({
    type: "OpaqueType",
    id,
    typeParameters,
    supertype,
    impltype
  });
}
function qualifiedTypeIdentifier(id, qualification) {
  return (0, _validateNode.default)({
    type: "QualifiedTypeIdentifier",
    id,
    qualification
  });
}
function stringLiteralTypeAnnotation(value) {
  return (0, _validateNode.default)({
    type: "StringLiteralTypeAnnotation",
    value
  });
}
function stringTypeAnnotation() {
  return {
    type: "StringTypeAnnotation"
  };
}
function symbolTypeAnnotation() {
  return {
    type: "SymbolTypeAnnotation"
  };
}
function thisTypeAnnotation() {
  return {
    type: "ThisTypeAnnotation"
  };
}
function tupleTypeAnnotation(types) {
  return (0, _validateNode.default)({
    type: "TupleTypeAnnotation",
    types
  });
}
function typeofTypeAnnotation(argument) {
  return (0, _validateNode.default)({
    type: "TypeofTypeAnnotation",
    argument
  });
}
function typeAlias(id, typeParameters = null, right) {
  return (0, _validateNode.default)({
    type: "TypeAlias",
    id,
    typeParameters,
    right
  });
}
function typeAnnotation(typeAnnotation) {
  return (0, _validateNode.default)({
    type: "TypeAnnotation",
    typeAnnotation
  });
}
function typeCastExpression(expression, typeAnnotation) {
  return (0, _validateNode.default)({
    type: "TypeCastExpression",
    expression,
    typeAnnotation
  });
}
function typeParameter(bound = null, _default = null, variance = null) {
  return (0, _validateNode.default)({
    type: "TypeParameter",
    bound,
    default: _default,
    variance,
    name: null
  });
}
function typeParameterDeclaration(params) {
  return (0, _validateNode.default)({
    type: "TypeParameterDeclaration",
    params
  });
}
function typeParameterInstantiation(params) {
  return (0, _validateNode.default)({
    type: "TypeParameterInstantiation",
    params
  });
}
function unionTypeAnnotation(types) {
  return (0, _validateNode.default)({
    type: "UnionTypeAnnotation",
    types
  });
}
function variance(kind) {
  return (0, _validateNode.default)({
    type: "Variance",
    kind
  });
}
function voidTypeAnnotation() {
  return {
    type: "VoidTypeAnnotation"
  };
}
function enumDeclaration(id, body) {
  return (0, _validateNode.default)({
    type: "EnumDeclaration",
    id,
    body
  });
}
function enumBooleanBody(members) {
  return (0, _validateNode.default)({
    type: "EnumBooleanBody",
    members,
    explicitType: null,
    hasUnknownMembers: null
  });
}
function enumNumberBody(members) {
  return (0, _validateNode.default)({
    type: "EnumNumberBody",
    members,
    explicitType: null,
    hasUnknownMembers: null
  });
}
function enumStringBody(members) {
  return (0, _validateNode.default)({
    type: "EnumStringBody",
    members,
    explicitType: null,
    hasUnknownMembers: null
  });
}
function enumSymbolBody(members) {
  return (0, _validateNode.default)({
    type: "EnumSymbolBody",
    members,
    hasUnknownMembers: null
  });
}
function enumBooleanMember(id) {
  return (0, _validateNode.default)({
    type: "EnumBooleanMember",
    id,
    init: null
  });
}
function enumNumberMember(id, init) {
  return (0, _validateNode.default)({
    type: "EnumNumberMember",
    id,
    init
  });
}
function enumStringMember(id, init) {
  return (0, _validateNode.default)({
    type: "EnumStringMember",
    id,
    init
  });
}
function enumDefaultedMember(id) {
  return (0, _validateNode.default)({
    type: "EnumDefaultedMember",
    id
  });
}
function indexedAccessType(objectType, indexType) {
  return (0, _validateNode.default)({
    type: "IndexedAccessType",
    objectType,
    indexType
  });
}
function optionalIndexedAccessType(objectType, indexType) {
  return (0, _validateNode.default)({
    type: "OptionalIndexedAccessType",
    objectType,
    indexType,
    optional: null
  });
}
function jsxAttribute(name, value = null) {
  return (0, _validateNode.default)({
    type: "JSXAttribute",
    name,
    value
  });
}
function jsxClosingElement(name) {
  return (0, _validateNode.default)({
    type: "JSXClosingElement",
    name
  });
}
function jsxElement(openingElement, closingElement = null, children, selfClosing = null) {
  return (0, _validateNode.default)({
    type: "JSXElement",
    openingElement,
    closingElement,
    children,
    selfClosing
  });
}
function jsxEmptyExpression() {
  return {
    type: "JSXEmptyExpression"
  };
}
function jsxExpressionContainer(expression) {
  return (0, _validateNode.default)({
    type: "JSXExpressionContainer",
    expression
  });
}
function jsxSpreadChild(expression) {
  return (0, _validateNode.default)({
    type: "JSXSpreadChild",
    expression
  });
}
function jsxIdentifier(name) {
  return (0, _validateNode.default)({
    type: "JSXIdentifier",
    name
  });
}
function jsxMemberExpression(object, property) {
  return (0, _validateNode.default)({
    type: "JSXMemberExpression",
    object,
    property
  });
}
function jsxNamespacedName(namespace, name) {
  return (0, _validateNode.default)({
    type: "JSXNamespacedName",
    namespace,
    name
  });
}
function jsxOpeningElement(name, attributes, selfClosing = false) {
  return (0, _validateNode.default)({
    type: "JSXOpeningElement",
    name,
    attributes,
    selfClosing
  });
}
function jsxSpreadAttribute(argument) {
  return (0, _validateNode.default)({
    type: "JSXSpreadAttribute",
    argument
  });
}
function jsxText(value) {
  return (0, _validateNode.default)({
    type: "JSXText",
    value
  });
}
function jsxFragment(openingFragment, closingFragment, children) {
  return (0, _validateNode.default)({
    type: "JSXFragment",
    openingFragment,
    closingFragment,
    children
  });
}
function jsxOpeningFragment() {
  return {
    type: "JSXOpeningFragment"
  };
}
function jsxClosingFragment() {
  return {
    type: "JSXClosingFragment"
  };
}
function noop() {
  return {
    type: "Noop"
  };
}
function placeholder(expectedNode, name) {
  return (0, _validateNode.default)({
    type: "Placeholder",
    expectedNode,
    name
  });
}
function v8IntrinsicIdentifier(name) {
  return (0, _validateNode.default)({
    type: "V8IntrinsicIdentifier",
    name
  });
}
function argumentPlaceholder() {
  return {
    type: "ArgumentPlaceholder"
  };
}
function bindExpression(object, callee) {
  return (0, _validateNode.default)({
    type: "BindExpression",
    object,
    callee
  });
}
function importAttribute(key, value) {
  return (0, _validateNode.default)({
    type: "ImportAttribute",
    key,
    value
  });
}
function decorator(expression) {
  return (0, _validateNode.default)({
    type: "Decorator",
    expression
  });
}
function doExpression(body, async = false) {
  return (0, _validateNode.default)({
    type: "DoExpression",
    body,
    async
  });
}
function exportDefaultSpecifier(exported) {
  return (0, _validateNode.default)({
    type: "ExportDefaultSpecifier",
    exported
  });
}
function recordExpression(properties) {
  return (0, _validateNode.default)({
    type: "RecordExpression",
    properties
  });
}
function tupleExpression(elements = []) {
  return (0, _validateNode.default)({
    type: "TupleExpression",
    elements
  });
}
function decimalLiteral(value) {
  return (0, _validateNode.default)({
    type: "DecimalLiteral",
    value
  });
}
function moduleExpression(body) {
  return (0, _validateNode.default)({
    type: "ModuleExpression",
    body
  });
}
function topicReference() {
  return {
    type: "TopicReference"
  };
}
function pipelineTopicExpression(expression) {
  return (0, _validateNode.default)({
    type: "PipelineTopicExpression",
    expression
  });
}
function pipelineBareFunction(callee) {
  return (0, _validateNode.default)({
    type: "PipelineBareFunction",
    callee
  });
}
function pipelinePrimaryTopicReference() {
  return {
    type: "PipelinePrimaryTopicReference"
  };
}
function tsParameterProperty(parameter) {
  return (0, _validateNode.default)({
    type: "TSParameterProperty",
    parameter
  });
}
function tsDeclareFunction(id = null, typeParameters = null, params, returnType = null) {
  return (0, _validateNode.default)({
    type: "TSDeclareFunction",
    id,
    typeParameters,
    params,
    returnType
  });
}
function tsDeclareMethod(decorators = null, key, typeParameters = null, params, returnType = null) {
  return (0, _validateNode.default)({
    type: "TSDeclareMethod",
    decorators,
    key,
    typeParameters,
    params,
    returnType
  });
}
function tsQualifiedName(left, right) {
  return (0, _validateNode.default)({
    type: "TSQualifiedName",
    left,
    right
  });
}
function tsCallSignatureDeclaration(typeParameters = null, parameters, typeAnnotation = null) {
  return (0, _validateNode.default)({
    type: "TSCallSignatureDeclaration",
    typeParameters,
    parameters,
    typeAnnotation
  });
}
function tsConstructSignatureDeclaration(typeParameters = null, parameters, typeAnnotation = null) {
  return (0, _validateNode.default)({
    type: "TSConstructSignatureDeclaration",
    typeParameters,
    parameters,
    typeAnnotation
  });
}
function tsPropertySignature(key, typeAnnotation = null) {
  return (0, _validateNode.default)({
    type: "TSPropertySignature",
    key,
    typeAnnotation,
    kind: null
  });
}
function tsMethodSignature(key, typeParameters = null, parameters, typeAnnotation = null) {
  return (0, _validateNode.default)({
    type: "TSMethodSignature",
    key,
    typeParameters,
    parameters,
    typeAnnotation,
    kind: null
  });
}
function tsIndexSignature(parameters, typeAnnotation = null) {
  return (0, _validateNode.default)({
    type: "TSIndexSignature",
    parameters,
    typeAnnotation
  });
}
function tsAnyKeyword() {
  return {
    type: "TSAnyKeyword"
  };
}
function tsBooleanKeyword() {
  return {
    type: "TSBooleanKeyword"
  };
}
function tsBigIntKeyword() {
  return {
    type: "TSBigIntKeyword"
  };
}
function tsIntrinsicKeyword() {
  return {
    type: "TSIntrinsicKeyword"
  };
}
function tsNeverKeyword() {
  return {
    type: "TSNeverKeyword"
  };
}
function tsNullKeyword() {
  return {
    type: "TSNullKeyword"
  };
}
function tsNumberKeyword() {
  return {
    type: "TSNumberKeyword"
  };
}
function tsObjectKeyword() {
  return {
    type: "TSObjectKeyword"
  };
}
function tsStringKeyword() {
  return {
    type: "TSStringKeyword"
  };
}
function tsSymbolKeyword() {
  return {
    type: "TSSymbolKeyword"
  };
}
function tsUndefinedKeyword() {
  return {
    type: "TSUndefinedKeyword"
  };
}
function tsUnknownKeyword() {
  return {
    type: "TSUnknownKeyword"
  };
}
function tsVoidKeyword() {
  return {
    type: "TSVoidKeyword"
  };
}
function tsThisType() {
  return {
    type: "TSThisType"
  };
}
function tsFunctionType(typeParameters = null, parameters, typeAnnotation = null) {
  return (0, _validateNode.default)({
    type: "TSFunctionType",
    typeParameters,
    parameters,
    typeAnnotation
  });
}
function tsConstructorType(typeParameters = null, parameters, typeAnnotation = null) {
  return (0, _validateNode.default)({
    type: "TSConstructorType",
    typeParameters,
    parameters,
    typeAnnotation
  });
}
function tsTypeReference(typeName, typeParameters = null) {
  return (0, _validateNode.default)({
    type: "TSTypeReference",
    typeName,
    typeParameters
  });
}
function tsTypePredicate(parameterName, typeAnnotation = null, asserts = null) {
  return (0, _validateNode.default)({
    type: "TSTypePredicate",
    parameterName,
    typeAnnotation,
    asserts
  });
}
function tsTypeQuery(exprName, typeParameters = null) {
  return (0, _validateNode.default)({
    type: "TSTypeQuery",
    exprName,
    typeParameters
  });
}
function tsTypeLiteral(members) {
  return (0, _validateNode.default)({
    type: "TSTypeLiteral",
    members
  });
}
function tsArrayType(elementType) {
  return (0, _validateNode.default)({
    type: "TSArrayType",
    elementType
  });
}
function tsTupleType(elementTypes) {
  return (0, _validateNode.default)({
    type: "TSTupleType",
    elementTypes
  });
}
function tsOptionalType(typeAnnotation) {
  return (0, _validateNode.default)({
    type: "TSOptionalType",
    typeAnnotation
  });
}
function tsRestType(typeAnnotation) {
  return (0, _validateNode.default)({
    type: "TSRestType",
    typeAnnotation
  });
}
function tsNamedTupleMember(label, elementType, optional = false) {
  return (0, _validateNode.default)({
    type: "TSNamedTupleMember",
    label,
    elementType,
    optional
  });
}
function tsUnionType(types) {
  return (0, _validateNode.default)({
    type: "TSUnionType",
    types
  });
}
function tsIntersectionType(types) {
  return (0, _validateNode.default)({
    type: "TSIntersectionType",
    types
  });
}
function tsConditionalType(checkType, extendsType, trueType, falseType) {
  return (0, _validateNode.default)({
    type: "TSConditionalType",
    checkType,
    extendsType,
    trueType,
    falseType
  });
}
function tsInferType(typeParameter) {
  return (0, _validateNode.default)({
    type: "TSInferType",
    typeParameter
  });
}
function tsParenthesizedType(typeAnnotation) {
  return (0, _validateNode.default)({
    type: "TSParenthesizedType",
    typeAnnotation
  });
}
function tsTypeOperator(typeAnnotation) {
  return (0, _validateNode.default)({
    type: "TSTypeOperator",
    typeAnnotation,
    operator: null
  });
}
function tsIndexedAccessType(objectType, indexType) {
  return (0, _validateNode.default)({
    type: "TSIndexedAccessType",
    objectType,
    indexType
  });
}
function tsMappedType(typeParameter, typeAnnotation = null, nameType = null) {
  return (0, _validateNode.default)({
    type: "TSMappedType",
    typeParameter,
    typeAnnotation,
    nameType
  });
}
function tsLiteralType(literal) {
  return (0, _validateNode.default)({
    type: "TSLiteralType",
    literal
  });
}
function tsExpressionWithTypeArguments(expression, typeParameters = null) {
  return (0, _validateNode.default)({
    type: "TSExpressionWithTypeArguments",
    expression,
    typeParameters
  });
}
function tsInterfaceDeclaration(id, typeParameters = null, _extends = null, body) {
  return (0, _validateNode.default)({
    type: "TSInterfaceDeclaration",
    id,
    typeParameters,
    extends: _extends,
    body
  });
}
function tsInterfaceBody(body) {
  return (0, _validateNode.default)({
    type: "TSInterfaceBody",
    body
  });
}
function tsTypeAliasDeclaration(id, typeParameters = null, typeAnnotation) {
  return (0, _validateNode.default)({
    type: "TSTypeAliasDeclaration",
    id,
    typeParameters,
    typeAnnotation
  });
}
function tsInstantiationExpression(expression, typeParameters = null) {
  return (0, _validateNode.default)({
    type: "TSInstantiationExpression",
    expression,
    typeParameters
  });
}
function tsAsExpression(expression, typeAnnotation) {
  return (0, _validateNode.default)({
    type: "TSAsExpression",
    expression,
    typeAnnotation
  });
}
function tsSatisfiesExpression(expression, typeAnnotation) {
  return (0, _validateNode.default)({
    type: "TSSatisfiesExpression",
    expression,
    typeAnnotation
  });
}
function tsTypeAssertion(typeAnnotation, expression) {
  return (0, _validateNode.default)({
    type: "TSTypeAssertion",
    typeAnnotation,
    expression
  });
}
function tsEnumDeclaration(id, members) {
  return (0, _validateNode.default)({
    type: "TSEnumDeclaration",
    id,
    members
  });
}
function tsEnumMember(id, initializer = null) {
  return (0, _validateNode.default)({
    type: "TSEnumMember",
    id,
    initializer
  });
}
function tsModuleDeclaration(id, body) {
  return (0, _validateNode.default)({
    type: "TSModuleDeclaration",
    id,
    body
  });
}
function tsModuleBlock(body) {
  return (0, _validateNode.default)({
    type: "TSModuleBlock",
    body
  });
}
function tsImportType(argument, qualifier = null, typeParameters = null) {
  return (0, _validateNode.default)({
    type: "TSImportType",
    argument,
    qualifier,
    typeParameters
  });
}
function tsImportEqualsDeclaration(id, moduleReference) {
  return (0, _validateNode.default)({
    type: "TSImportEqualsDeclaration",
    id,
    moduleReference,
    isExport: null
  });
}
function tsExternalModuleReference(expression) {
  return (0, _validateNode.default)({
    type: "TSExternalModuleReference",
    expression
  });
}
function tsNonNullExpression(expression) {
  return (0, _validateNode.default)({
    type: "TSNonNullExpression",
    expression
  });
}
function tsExportAssignment(expression) {
  return (0, _validateNode.default)({
    type: "TSExportAssignment",
    expression
  });
}
function tsNamespaceExportDeclaration(id) {
  return (0, _validateNode.default)({
    type: "TSNamespaceExportDeclaration",
    id
  });
}
function tsTypeAnnotation(typeAnnotation) {
  return (0, _validateNode.default)({
    type: "TSTypeAnnotation",
    typeAnnotation
  });
}
function tsTypeParameterInstantiation(params) {
  return (0, _validateNode.default)({
    type: "TSTypeParameterInstantiation",
    params
  });
}
function tsTypeParameterDeclaration(params) {
  return (0, _validateNode.default)({
    type: "TSTypeParameterDeclaration",
    params
  });
}
function tsTypeParameter(constraint = null, _default = null, name) {
  return (0, _validateNode.default)({
    type: "TSTypeParameter",
    constraint,
    default: _default,
    name
  });
}
function NumberLiteral(value) {
  (0, _deprecationWarning.default)("NumberLiteral", "NumericLiteral", "The node type ");
  return numericLiteral(value);
}
function RegexLiteral(pattern, flags = "") {
  (0, _deprecationWarning.default)("RegexLiteral", "RegExpLiteral", "The node type ");
  return regExpLiteral(pattern, flags);
}
function RestProperty(argument) {
  (0, _deprecationWarning.default)("RestProperty", "RestElement", "The node type ");
  return restElement(argument);
}
function SpreadProperty(argument) {
  (0, _deprecationWarning.default)("SpreadProperty", "SpreadElement", "The node type ");
  return spreadElement(argument);
}

//# sourceMappingURL=index.js.map
