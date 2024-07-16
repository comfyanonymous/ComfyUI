"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.TSAnyKeyword = TSAnyKeyword;
exports.TSArrayType = TSArrayType;
exports.TSSatisfiesExpression = exports.TSAsExpression = TSTypeExpression;
exports.TSBigIntKeyword = TSBigIntKeyword;
exports.TSBooleanKeyword = TSBooleanKeyword;
exports.TSCallSignatureDeclaration = TSCallSignatureDeclaration;
exports.TSConditionalType = TSConditionalType;
exports.TSConstructSignatureDeclaration = TSConstructSignatureDeclaration;
exports.TSConstructorType = TSConstructorType;
exports.TSDeclareFunction = TSDeclareFunction;
exports.TSDeclareMethod = TSDeclareMethod;
exports.TSEnumDeclaration = TSEnumDeclaration;
exports.TSEnumMember = TSEnumMember;
exports.TSExportAssignment = TSExportAssignment;
exports.TSExpressionWithTypeArguments = TSExpressionWithTypeArguments;
exports.TSExternalModuleReference = TSExternalModuleReference;
exports.TSFunctionType = TSFunctionType;
exports.TSImportEqualsDeclaration = TSImportEqualsDeclaration;
exports.TSImportType = TSImportType;
exports.TSIndexSignature = TSIndexSignature;
exports.TSIndexedAccessType = TSIndexedAccessType;
exports.TSInferType = TSInferType;
exports.TSInstantiationExpression = TSInstantiationExpression;
exports.TSInterfaceBody = TSInterfaceBody;
exports.TSInterfaceDeclaration = TSInterfaceDeclaration;
exports.TSIntersectionType = TSIntersectionType;
exports.TSIntrinsicKeyword = TSIntrinsicKeyword;
exports.TSLiteralType = TSLiteralType;
exports.TSMappedType = TSMappedType;
exports.TSMethodSignature = TSMethodSignature;
exports.TSModuleBlock = TSModuleBlock;
exports.TSModuleDeclaration = TSModuleDeclaration;
exports.TSNamedTupleMember = TSNamedTupleMember;
exports.TSNamespaceExportDeclaration = TSNamespaceExportDeclaration;
exports.TSNeverKeyword = TSNeverKeyword;
exports.TSNonNullExpression = TSNonNullExpression;
exports.TSNullKeyword = TSNullKeyword;
exports.TSNumberKeyword = TSNumberKeyword;
exports.TSObjectKeyword = TSObjectKeyword;
exports.TSOptionalType = TSOptionalType;
exports.TSParameterProperty = TSParameterProperty;
exports.TSParenthesizedType = TSParenthesizedType;
exports.TSPropertySignature = TSPropertySignature;
exports.TSQualifiedName = TSQualifiedName;
exports.TSRestType = TSRestType;
exports.TSStringKeyword = TSStringKeyword;
exports.TSSymbolKeyword = TSSymbolKeyword;
exports.TSThisType = TSThisType;
exports.TSTupleType = TSTupleType;
exports.TSTypeAliasDeclaration = TSTypeAliasDeclaration;
exports.TSTypeAnnotation = TSTypeAnnotation;
exports.TSTypeAssertion = TSTypeAssertion;
exports.TSTypeLiteral = TSTypeLiteral;
exports.TSTypeOperator = TSTypeOperator;
exports.TSTypeParameter = TSTypeParameter;
exports.TSTypeParameterDeclaration = exports.TSTypeParameterInstantiation = TSTypeParameterInstantiation;
exports.TSTypePredicate = TSTypePredicate;
exports.TSTypeQuery = TSTypeQuery;
exports.TSTypeReference = TSTypeReference;
exports.TSUndefinedKeyword = TSUndefinedKeyword;
exports.TSUnionType = TSUnionType;
exports.TSUnknownKeyword = TSUnknownKeyword;
exports.TSVoidKeyword = TSVoidKeyword;
exports.tsPrintClassMemberModifiers = tsPrintClassMemberModifiers;
exports.tsPrintFunctionOrConstructorType = tsPrintFunctionOrConstructorType;
exports.tsPrintPropertyOrMethodName = tsPrintPropertyOrMethodName;
exports.tsPrintSignatureDeclarationBase = tsPrintSignatureDeclarationBase;
exports.tsPrintTypeLiteralOrInterfaceBody = tsPrintTypeLiteralOrInterfaceBody;
function TSTypeAnnotation(node) {
  this.tokenChar(58);
  this.space();
  if (node.optional) this.tokenChar(63);
  this.print(node.typeAnnotation, node);
}
function TSTypeParameterInstantiation(node, parent) {
  this.tokenChar(60);
  this.printList(node.params, node, {});
  if (parent.type === "ArrowFunctionExpression" && node.params.length === 1) {
    this.tokenChar(44);
  }
  this.tokenChar(62);
}
function TSTypeParameter(node) {
  if (node.in) {
    this.word("in");
    this.space();
  }
  if (node.out) {
    this.word("out");
    this.space();
  }
  this.word(node.name);
  if (node.constraint) {
    this.space();
    this.word("extends");
    this.space();
    this.print(node.constraint, node);
  }
  if (node.default) {
    this.space();
    this.tokenChar(61);
    this.space();
    this.print(node.default, node);
  }
}
function TSParameterProperty(node) {
  if (node.accessibility) {
    this.word(node.accessibility);
    this.space();
  }
  if (node.readonly) {
    this.word("readonly");
    this.space();
  }
  this._param(node.parameter);
}
function TSDeclareFunction(node, parent) {
  if (node.declare) {
    this.word("declare");
    this.space();
  }
  this._functionHead(node, parent);
  this.tokenChar(59);
}
function TSDeclareMethod(node) {
  this._classMethodHead(node);
  this.tokenChar(59);
}
function TSQualifiedName(node) {
  this.print(node.left, node);
  this.tokenChar(46);
  this.print(node.right, node);
}
function TSCallSignatureDeclaration(node) {
  this.tsPrintSignatureDeclarationBase(node);
  this.tokenChar(59);
}
function TSConstructSignatureDeclaration(node) {
  this.word("new");
  this.space();
  this.tsPrintSignatureDeclarationBase(node);
  this.tokenChar(59);
}
function TSPropertySignature(node) {
  const {
    readonly
  } = node;
  if (readonly) {
    this.word("readonly");
    this.space();
  }
  this.tsPrintPropertyOrMethodName(node);
  this.print(node.typeAnnotation, node);
  this.tokenChar(59);
}
function tsPrintPropertyOrMethodName(node) {
  if (node.computed) {
    this.tokenChar(91);
  }
  this.print(node.key, node);
  if (node.computed) {
    this.tokenChar(93);
  }
  if (node.optional) {
    this.tokenChar(63);
  }
}
function TSMethodSignature(node) {
  const {
    kind
  } = node;
  if (kind === "set" || kind === "get") {
    this.word(kind);
    this.space();
  }
  this.tsPrintPropertyOrMethodName(node);
  this.tsPrintSignatureDeclarationBase(node);
  this.tokenChar(59);
}
function TSIndexSignature(node) {
  const {
    readonly,
    static: isStatic
  } = node;
  if (isStatic) {
    this.word("static");
    this.space();
  }
  if (readonly) {
    this.word("readonly");
    this.space();
  }
  this.tokenChar(91);
  this._parameters(node.parameters, node);
  this.tokenChar(93);
  this.print(node.typeAnnotation, node);
  this.tokenChar(59);
}
function TSAnyKeyword() {
  this.word("any");
}
function TSBigIntKeyword() {
  this.word("bigint");
}
function TSUnknownKeyword() {
  this.word("unknown");
}
function TSNumberKeyword() {
  this.word("number");
}
function TSObjectKeyword() {
  this.word("object");
}
function TSBooleanKeyword() {
  this.word("boolean");
}
function TSStringKeyword() {
  this.word("string");
}
function TSSymbolKeyword() {
  this.word("symbol");
}
function TSVoidKeyword() {
  this.word("void");
}
function TSUndefinedKeyword() {
  this.word("undefined");
}
function TSNullKeyword() {
  this.word("null");
}
function TSNeverKeyword() {
  this.word("never");
}
function TSIntrinsicKeyword() {
  this.word("intrinsic");
}
function TSThisType() {
  this.word("this");
}
function TSFunctionType(node) {
  this.tsPrintFunctionOrConstructorType(node);
}
function TSConstructorType(node) {
  if (node.abstract) {
    this.word("abstract");
    this.space();
  }
  this.word("new");
  this.space();
  this.tsPrintFunctionOrConstructorType(node);
}
function tsPrintFunctionOrConstructorType(node) {
  const {
    typeParameters
  } = node;
  const parameters = node.parameters;
  this.print(typeParameters, node);
  this.tokenChar(40);
  this._parameters(parameters, node);
  this.tokenChar(41);
  this.space();
  this.token("=>");
  this.space();
  const returnType = node.typeAnnotation;
  this.print(returnType.typeAnnotation, node);
}
function TSTypeReference(node) {
  this.print(node.typeName, node, true);
  this.print(node.typeParameters, node, true);
}
function TSTypePredicate(node) {
  if (node.asserts) {
    this.word("asserts");
    this.space();
  }
  this.print(node.parameterName);
  if (node.typeAnnotation) {
    this.space();
    this.word("is");
    this.space();
    this.print(node.typeAnnotation.typeAnnotation);
  }
}
function TSTypeQuery(node) {
  this.word("typeof");
  this.space();
  this.print(node.exprName);
  if (node.typeParameters) {
    this.print(node.typeParameters, node);
  }
}
function TSTypeLiteral(node) {
  this.tsPrintTypeLiteralOrInterfaceBody(node.members, node);
}
function tsPrintTypeLiteralOrInterfaceBody(members, node) {
  tsPrintBraced(this, members, node);
}
function tsPrintBraced(printer, members, node) {
  printer.token("{");
  if (members.length) {
    printer.indent();
    printer.newline();
    for (const member of members) {
      printer.print(member, node);
      printer.newline();
    }
    printer.dedent();
  }
  printer.rightBrace(node);
}
function TSArrayType(node) {
  this.print(node.elementType, node, true);
  this.token("[]");
}
function TSTupleType(node) {
  this.tokenChar(91);
  this.printList(node.elementTypes, node);
  this.tokenChar(93);
}
function TSOptionalType(node) {
  this.print(node.typeAnnotation, node);
  this.tokenChar(63);
}
function TSRestType(node) {
  this.token("...");
  this.print(node.typeAnnotation, node);
}
function TSNamedTupleMember(node) {
  this.print(node.label, node);
  if (node.optional) this.tokenChar(63);
  this.tokenChar(58);
  this.space();
  this.print(node.elementType, node);
}
function TSUnionType(node) {
  tsPrintUnionOrIntersectionType(this, node, "|");
}
function TSIntersectionType(node) {
  tsPrintUnionOrIntersectionType(this, node, "&");
}
function tsPrintUnionOrIntersectionType(printer, node, sep) {
  printer.printJoin(node.types, node, {
    separator() {
      this.space();
      this.token(sep);
      this.space();
    }
  });
}
function TSConditionalType(node) {
  this.print(node.checkType);
  this.space();
  this.word("extends");
  this.space();
  this.print(node.extendsType);
  this.space();
  this.tokenChar(63);
  this.space();
  this.print(node.trueType);
  this.space();
  this.tokenChar(58);
  this.space();
  this.print(node.falseType);
}
function TSInferType(node) {
  this.token("infer");
  this.space();
  this.print(node.typeParameter);
}
function TSParenthesizedType(node) {
  this.tokenChar(40);
  this.print(node.typeAnnotation, node);
  this.tokenChar(41);
}
function TSTypeOperator(node) {
  this.word(node.operator);
  this.space();
  this.print(node.typeAnnotation, node);
}
function TSIndexedAccessType(node) {
  this.print(node.objectType, node, true);
  this.tokenChar(91);
  this.print(node.indexType, node);
  this.tokenChar(93);
}
function TSMappedType(node) {
  const {
    nameType,
    optional,
    readonly,
    typeParameter,
    typeAnnotation
  } = node;
  this.tokenChar(123);
  this.space();
  if (readonly) {
    tokenIfPlusMinus(this, readonly);
    this.word("readonly");
    this.space();
  }
  this.tokenChar(91);
  this.word(typeParameter.name);
  this.space();
  this.word("in");
  this.space();
  this.print(typeParameter.constraint, typeParameter);
  if (nameType) {
    this.space();
    this.word("as");
    this.space();
    this.print(nameType, node);
  }
  this.tokenChar(93);
  if (optional) {
    tokenIfPlusMinus(this, optional);
    this.tokenChar(63);
  }
  if (typeAnnotation) {
    this.tokenChar(58);
    this.space();
    this.print(typeAnnotation, node);
  }
  this.space();
  this.tokenChar(125);
}
function tokenIfPlusMinus(self, tok) {
  if (tok !== true) {
    self.token(tok);
  }
}
function TSLiteralType(node) {
  this.print(node.literal, node);
}
function TSExpressionWithTypeArguments(node) {
  this.print(node.expression, node);
  this.print(node.typeParameters, node);
}
function TSInterfaceDeclaration(node) {
  const {
    declare,
    id,
    typeParameters,
    extends: extendz,
    body
  } = node;
  if (declare) {
    this.word("declare");
    this.space();
  }
  this.word("interface");
  this.space();
  this.print(id, node);
  this.print(typeParameters, node);
  if (extendz != null && extendz.length) {
    this.space();
    this.word("extends");
    this.space();
    this.printList(extendz, node);
  }
  this.space();
  this.print(body, node);
}
function TSInterfaceBody(node) {
  this.tsPrintTypeLiteralOrInterfaceBody(node.body, node);
}
function TSTypeAliasDeclaration(node) {
  const {
    declare,
    id,
    typeParameters,
    typeAnnotation
  } = node;
  if (declare) {
    this.word("declare");
    this.space();
  }
  this.word("type");
  this.space();
  this.print(id, node);
  this.print(typeParameters, node);
  this.space();
  this.tokenChar(61);
  this.space();
  this.print(typeAnnotation, node);
  this.tokenChar(59);
}
function TSTypeExpression(node) {
  var _expression$trailingC;
  const {
    type,
    expression,
    typeAnnotation
  } = node;
  const forceParens = !!((_expression$trailingC = expression.trailingComments) != null && _expression$trailingC.length);
  this.print(expression, node, true, undefined, forceParens);
  this.space();
  this.word(type === "TSAsExpression" ? "as" : "satisfies");
  this.space();
  this.print(typeAnnotation, node);
}
function TSTypeAssertion(node) {
  const {
    typeAnnotation,
    expression
  } = node;
  this.tokenChar(60);
  this.print(typeAnnotation, node);
  this.tokenChar(62);
  this.space();
  this.print(expression, node);
}
function TSInstantiationExpression(node) {
  this.print(node.expression, node);
  this.print(node.typeParameters, node);
}
function TSEnumDeclaration(node) {
  const {
    declare,
    const: isConst,
    id,
    members
  } = node;
  if (declare) {
    this.word("declare");
    this.space();
  }
  if (isConst) {
    this.word("const");
    this.space();
  }
  this.word("enum");
  this.space();
  this.print(id, node);
  this.space();
  tsPrintBraced(this, members, node);
}
function TSEnumMember(node) {
  const {
    id,
    initializer
  } = node;
  this.print(id, node);
  if (initializer) {
    this.space();
    this.tokenChar(61);
    this.space();
    this.print(initializer, node);
  }
  this.tokenChar(44);
}
function TSModuleDeclaration(node) {
  const {
    declare,
    id
  } = node;
  if (declare) {
    this.word("declare");
    this.space();
  }
  if (!node.global) {
    this.word(id.type === "Identifier" ? "namespace" : "module");
    this.space();
  }
  this.print(id, node);
  if (!node.body) {
    this.tokenChar(59);
    return;
  }
  let body = node.body;
  while (body.type === "TSModuleDeclaration") {
    this.tokenChar(46);
    this.print(body.id, body);
    body = body.body;
  }
  this.space();
  this.print(body, node);
}
function TSModuleBlock(node) {
  tsPrintBraced(this, node.body, node);
}
function TSImportType(node) {
  const {
    argument,
    qualifier,
    typeParameters
  } = node;
  this.word("import");
  this.tokenChar(40);
  this.print(argument, node);
  this.tokenChar(41);
  if (qualifier) {
    this.tokenChar(46);
    this.print(qualifier, node);
  }
  if (typeParameters) {
    this.print(typeParameters, node);
  }
}
function TSImportEqualsDeclaration(node) {
  const {
    isExport,
    id,
    moduleReference
  } = node;
  if (isExport) {
    this.word("export");
    this.space();
  }
  this.word("import");
  this.space();
  this.print(id, node);
  this.space();
  this.tokenChar(61);
  this.space();
  this.print(moduleReference, node);
  this.tokenChar(59);
}
function TSExternalModuleReference(node) {
  this.token("require(");
  this.print(node.expression, node);
  this.tokenChar(41);
}
function TSNonNullExpression(node) {
  this.print(node.expression, node);
  this.tokenChar(33);
}
function TSExportAssignment(node) {
  this.word("export");
  this.space();
  this.tokenChar(61);
  this.space();
  this.print(node.expression, node);
  this.tokenChar(59);
}
function TSNamespaceExportDeclaration(node) {
  this.word("export");
  this.space();
  this.word("as");
  this.space();
  this.word("namespace");
  this.space();
  this.print(node.id, node);
}
function tsPrintSignatureDeclarationBase(node) {
  const {
    typeParameters
  } = node;
  const parameters = node.parameters;
  this.print(typeParameters, node);
  this.tokenChar(40);
  this._parameters(parameters, node);
  this.tokenChar(41);
  const returnType = node.typeAnnotation;
  this.print(returnType, node);
}
function tsPrintClassMemberModifiers(node) {
  const isField = node.type === "ClassAccessorProperty" || node.type === "ClassProperty";
  if (isField && node.declare) {
    this.word("declare");
    this.space();
  }
  if (node.accessibility) {
    this.word(node.accessibility);
    this.space();
  }
  if (node.static) {
    this.word("static");
    this.space();
  }
  if (node.override) {
    this.word("override");
    this.space();
  }
  if (node.abstract) {
    this.word("abstract");
    this.space();
  }
  if (isField && node.readonly) {
    this.word("readonly");
    this.space();
  }
}

//# sourceMappingURL=typescript.js.map
