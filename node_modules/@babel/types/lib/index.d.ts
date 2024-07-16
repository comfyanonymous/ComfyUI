declare function isCompatTag(tagName?: string): boolean;

type ReturnedChild = JSXSpreadChild | JSXElement | JSXFragment | Expression;
declare function buildChildren(node: JSXElement | JSXFragment): ReturnedChild[];

declare function assertNode(node?: any): asserts node is Node;

declare function assertArrayExpression(node: object | null | undefined, opts?: object | null): asserts node is ArrayExpression;
declare function assertAssignmentExpression(node: object | null | undefined, opts?: object | null): asserts node is AssignmentExpression;
declare function assertBinaryExpression(node: object | null | undefined, opts?: object | null): asserts node is BinaryExpression;
declare function assertInterpreterDirective(node: object | null | undefined, opts?: object | null): asserts node is InterpreterDirective;
declare function assertDirective(node: object | null | undefined, opts?: object | null): asserts node is Directive;
declare function assertDirectiveLiteral(node: object | null | undefined, opts?: object | null): asserts node is DirectiveLiteral;
declare function assertBlockStatement(node: object | null | undefined, opts?: object | null): asserts node is BlockStatement;
declare function assertBreakStatement(node: object | null | undefined, opts?: object | null): asserts node is BreakStatement;
declare function assertCallExpression(node: object | null | undefined, opts?: object | null): asserts node is CallExpression;
declare function assertCatchClause(node: object | null | undefined, opts?: object | null): asserts node is CatchClause;
declare function assertConditionalExpression(node: object | null | undefined, opts?: object | null): asserts node is ConditionalExpression;
declare function assertContinueStatement(node: object | null | undefined, opts?: object | null): asserts node is ContinueStatement;
declare function assertDebuggerStatement(node: object | null | undefined, opts?: object | null): asserts node is DebuggerStatement;
declare function assertDoWhileStatement(node: object | null | undefined, opts?: object | null): asserts node is DoWhileStatement;
declare function assertEmptyStatement(node: object | null | undefined, opts?: object | null): asserts node is EmptyStatement;
declare function assertExpressionStatement(node: object | null | undefined, opts?: object | null): asserts node is ExpressionStatement;
declare function assertFile(node: object | null | undefined, opts?: object | null): asserts node is File;
declare function assertForInStatement(node: object | null | undefined, opts?: object | null): asserts node is ForInStatement;
declare function assertForStatement(node: object | null | undefined, opts?: object | null): asserts node is ForStatement;
declare function assertFunctionDeclaration(node: object | null | undefined, opts?: object | null): asserts node is FunctionDeclaration;
declare function assertFunctionExpression(node: object | null | undefined, opts?: object | null): asserts node is FunctionExpression;
declare function assertIdentifier(node: object | null | undefined, opts?: object | null): asserts node is Identifier;
declare function assertIfStatement(node: object | null | undefined, opts?: object | null): asserts node is IfStatement;
declare function assertLabeledStatement(node: object | null | undefined, opts?: object | null): asserts node is LabeledStatement;
declare function assertStringLiteral(node: object | null | undefined, opts?: object | null): asserts node is StringLiteral;
declare function assertNumericLiteral(node: object | null | undefined, opts?: object | null): asserts node is NumericLiteral;
declare function assertNullLiteral(node: object | null | undefined, opts?: object | null): asserts node is NullLiteral;
declare function assertBooleanLiteral(node: object | null | undefined, opts?: object | null): asserts node is BooleanLiteral;
declare function assertRegExpLiteral(node: object | null | undefined, opts?: object | null): asserts node is RegExpLiteral;
declare function assertLogicalExpression(node: object | null | undefined, opts?: object | null): asserts node is LogicalExpression;
declare function assertMemberExpression(node: object | null | undefined, opts?: object | null): asserts node is MemberExpression;
declare function assertNewExpression(node: object | null | undefined, opts?: object | null): asserts node is NewExpression;
declare function assertProgram(node: object | null | undefined, opts?: object | null): asserts node is Program;
declare function assertObjectExpression(node: object | null | undefined, opts?: object | null): asserts node is ObjectExpression;
declare function assertObjectMethod(node: object | null | undefined, opts?: object | null): asserts node is ObjectMethod;
declare function assertObjectProperty(node: object | null | undefined, opts?: object | null): asserts node is ObjectProperty;
declare function assertRestElement(node: object | null | undefined, opts?: object | null): asserts node is RestElement;
declare function assertReturnStatement(node: object | null | undefined, opts?: object | null): asserts node is ReturnStatement;
declare function assertSequenceExpression(node: object | null | undefined, opts?: object | null): asserts node is SequenceExpression;
declare function assertParenthesizedExpression(node: object | null | undefined, opts?: object | null): asserts node is ParenthesizedExpression;
declare function assertSwitchCase(node: object | null | undefined, opts?: object | null): asserts node is SwitchCase;
declare function assertSwitchStatement(node: object | null | undefined, opts?: object | null): asserts node is SwitchStatement;
declare function assertThisExpression(node: object | null | undefined, opts?: object | null): asserts node is ThisExpression;
declare function assertThrowStatement(node: object | null | undefined, opts?: object | null): asserts node is ThrowStatement;
declare function assertTryStatement(node: object | null | undefined, opts?: object | null): asserts node is TryStatement;
declare function assertUnaryExpression(node: object | null | undefined, opts?: object | null): asserts node is UnaryExpression;
declare function assertUpdateExpression(node: object | null | undefined, opts?: object | null): asserts node is UpdateExpression;
declare function assertVariableDeclaration(node: object | null | undefined, opts?: object | null): asserts node is VariableDeclaration;
declare function assertVariableDeclarator(node: object | null | undefined, opts?: object | null): asserts node is VariableDeclarator;
declare function assertWhileStatement(node: object | null | undefined, opts?: object | null): asserts node is WhileStatement;
declare function assertWithStatement(node: object | null | undefined, opts?: object | null): asserts node is WithStatement;
declare function assertAssignmentPattern(node: object | null | undefined, opts?: object | null): asserts node is AssignmentPattern;
declare function assertArrayPattern(node: object | null | undefined, opts?: object | null): asserts node is ArrayPattern;
declare function assertArrowFunctionExpression(node: object | null | undefined, opts?: object | null): asserts node is ArrowFunctionExpression;
declare function assertClassBody(node: object | null | undefined, opts?: object | null): asserts node is ClassBody;
declare function assertClassExpression(node: object | null | undefined, opts?: object | null): asserts node is ClassExpression;
declare function assertClassDeclaration(node: object | null | undefined, opts?: object | null): asserts node is ClassDeclaration;
declare function assertExportAllDeclaration(node: object | null | undefined, opts?: object | null): asserts node is ExportAllDeclaration;
declare function assertExportDefaultDeclaration(node: object | null | undefined, opts?: object | null): asserts node is ExportDefaultDeclaration;
declare function assertExportNamedDeclaration(node: object | null | undefined, opts?: object | null): asserts node is ExportNamedDeclaration;
declare function assertExportSpecifier(node: object | null | undefined, opts?: object | null): asserts node is ExportSpecifier;
declare function assertForOfStatement(node: object | null | undefined, opts?: object | null): asserts node is ForOfStatement;
declare function assertImportDeclaration(node: object | null | undefined, opts?: object | null): asserts node is ImportDeclaration;
declare function assertImportDefaultSpecifier(node: object | null | undefined, opts?: object | null): asserts node is ImportDefaultSpecifier;
declare function assertImportNamespaceSpecifier(node: object | null | undefined, opts?: object | null): asserts node is ImportNamespaceSpecifier;
declare function assertImportSpecifier(node: object | null | undefined, opts?: object | null): asserts node is ImportSpecifier;
declare function assertImportExpression(node: object | null | undefined, opts?: object | null): asserts node is ImportExpression;
declare function assertMetaProperty(node: object | null | undefined, opts?: object | null): asserts node is MetaProperty;
declare function assertClassMethod(node: object | null | undefined, opts?: object | null): asserts node is ClassMethod;
declare function assertObjectPattern(node: object | null | undefined, opts?: object | null): asserts node is ObjectPattern;
declare function assertSpreadElement(node: object | null | undefined, opts?: object | null): asserts node is SpreadElement;
declare function assertSuper(node: object | null | undefined, opts?: object | null): asserts node is Super;
declare function assertTaggedTemplateExpression(node: object | null | undefined, opts?: object | null): asserts node is TaggedTemplateExpression;
declare function assertTemplateElement(node: object | null | undefined, opts?: object | null): asserts node is TemplateElement;
declare function assertTemplateLiteral(node: object | null | undefined, opts?: object | null): asserts node is TemplateLiteral;
declare function assertYieldExpression(node: object | null | undefined, opts?: object | null): asserts node is YieldExpression;
declare function assertAwaitExpression(node: object | null | undefined, opts?: object | null): asserts node is AwaitExpression;
declare function assertImport(node: object | null | undefined, opts?: object | null): asserts node is Import;
declare function assertBigIntLiteral(node: object | null | undefined, opts?: object | null): asserts node is BigIntLiteral;
declare function assertExportNamespaceSpecifier(node: object | null | undefined, opts?: object | null): asserts node is ExportNamespaceSpecifier;
declare function assertOptionalMemberExpression(node: object | null | undefined, opts?: object | null): asserts node is OptionalMemberExpression;
declare function assertOptionalCallExpression(node: object | null | undefined, opts?: object | null): asserts node is OptionalCallExpression;
declare function assertClassProperty(node: object | null | undefined, opts?: object | null): asserts node is ClassProperty;
declare function assertClassAccessorProperty(node: object | null | undefined, opts?: object | null): asserts node is ClassAccessorProperty;
declare function assertClassPrivateProperty(node: object | null | undefined, opts?: object | null): asserts node is ClassPrivateProperty;
declare function assertClassPrivateMethod(node: object | null | undefined, opts?: object | null): asserts node is ClassPrivateMethod;
declare function assertPrivateName(node: object | null | undefined, opts?: object | null): asserts node is PrivateName;
declare function assertStaticBlock(node: object | null | undefined, opts?: object | null): asserts node is StaticBlock;
declare function assertAnyTypeAnnotation(node: object | null | undefined, opts?: object | null): asserts node is AnyTypeAnnotation;
declare function assertArrayTypeAnnotation(node: object | null | undefined, opts?: object | null): asserts node is ArrayTypeAnnotation;
declare function assertBooleanTypeAnnotation(node: object | null | undefined, opts?: object | null): asserts node is BooleanTypeAnnotation;
declare function assertBooleanLiteralTypeAnnotation(node: object | null | undefined, opts?: object | null): asserts node is BooleanLiteralTypeAnnotation;
declare function assertNullLiteralTypeAnnotation(node: object | null | undefined, opts?: object | null): asserts node is NullLiteralTypeAnnotation;
declare function assertClassImplements(node: object | null | undefined, opts?: object | null): asserts node is ClassImplements;
declare function assertDeclareClass(node: object | null | undefined, opts?: object | null): asserts node is DeclareClass;
declare function assertDeclareFunction(node: object | null | undefined, opts?: object | null): asserts node is DeclareFunction;
declare function assertDeclareInterface(node: object | null | undefined, opts?: object | null): asserts node is DeclareInterface;
declare function assertDeclareModule(node: object | null | undefined, opts?: object | null): asserts node is DeclareModule;
declare function assertDeclareModuleExports(node: object | null | undefined, opts?: object | null): asserts node is DeclareModuleExports;
declare function assertDeclareTypeAlias(node: object | null | undefined, opts?: object | null): asserts node is DeclareTypeAlias;
declare function assertDeclareOpaqueType(node: object | null | undefined, opts?: object | null): asserts node is DeclareOpaqueType;
declare function assertDeclareVariable(node: object | null | undefined, opts?: object | null): asserts node is DeclareVariable;
declare function assertDeclareExportDeclaration(node: object | null | undefined, opts?: object | null): asserts node is DeclareExportDeclaration;
declare function assertDeclareExportAllDeclaration(node: object | null | undefined, opts?: object | null): asserts node is DeclareExportAllDeclaration;
declare function assertDeclaredPredicate(node: object | null | undefined, opts?: object | null): asserts node is DeclaredPredicate;
declare function assertExistsTypeAnnotation(node: object | null | undefined, opts?: object | null): asserts node is ExistsTypeAnnotation;
declare function assertFunctionTypeAnnotation(node: object | null | undefined, opts?: object | null): asserts node is FunctionTypeAnnotation;
declare function assertFunctionTypeParam(node: object | null | undefined, opts?: object | null): asserts node is FunctionTypeParam;
declare function assertGenericTypeAnnotation(node: object | null | undefined, opts?: object | null): asserts node is GenericTypeAnnotation;
declare function assertInferredPredicate(node: object | null | undefined, opts?: object | null): asserts node is InferredPredicate;
declare function assertInterfaceExtends(node: object | null | undefined, opts?: object | null): asserts node is InterfaceExtends;
declare function assertInterfaceDeclaration(node: object | null | undefined, opts?: object | null): asserts node is InterfaceDeclaration;
declare function assertInterfaceTypeAnnotation(node: object | null | undefined, opts?: object | null): asserts node is InterfaceTypeAnnotation;
declare function assertIntersectionTypeAnnotation(node: object | null | undefined, opts?: object | null): asserts node is IntersectionTypeAnnotation;
declare function assertMixedTypeAnnotation(node: object | null | undefined, opts?: object | null): asserts node is MixedTypeAnnotation;
declare function assertEmptyTypeAnnotation(node: object | null | undefined, opts?: object | null): asserts node is EmptyTypeAnnotation;
declare function assertNullableTypeAnnotation(node: object | null | undefined, opts?: object | null): asserts node is NullableTypeAnnotation;
declare function assertNumberLiteralTypeAnnotation(node: object | null | undefined, opts?: object | null): asserts node is NumberLiteralTypeAnnotation;
declare function assertNumberTypeAnnotation(node: object | null | undefined, opts?: object | null): asserts node is NumberTypeAnnotation;
declare function assertObjectTypeAnnotation(node: object | null | undefined, opts?: object | null): asserts node is ObjectTypeAnnotation;
declare function assertObjectTypeInternalSlot(node: object | null | undefined, opts?: object | null): asserts node is ObjectTypeInternalSlot;
declare function assertObjectTypeCallProperty(node: object | null | undefined, opts?: object | null): asserts node is ObjectTypeCallProperty;
declare function assertObjectTypeIndexer(node: object | null | undefined, opts?: object | null): asserts node is ObjectTypeIndexer;
declare function assertObjectTypeProperty(node: object | null | undefined, opts?: object | null): asserts node is ObjectTypeProperty;
declare function assertObjectTypeSpreadProperty(node: object | null | undefined, opts?: object | null): asserts node is ObjectTypeSpreadProperty;
declare function assertOpaqueType(node: object | null | undefined, opts?: object | null): asserts node is OpaqueType;
declare function assertQualifiedTypeIdentifier(node: object | null | undefined, opts?: object | null): asserts node is QualifiedTypeIdentifier;
declare function assertStringLiteralTypeAnnotation(node: object | null | undefined, opts?: object | null): asserts node is StringLiteralTypeAnnotation;
declare function assertStringTypeAnnotation(node: object | null | undefined, opts?: object | null): asserts node is StringTypeAnnotation;
declare function assertSymbolTypeAnnotation(node: object | null | undefined, opts?: object | null): asserts node is SymbolTypeAnnotation;
declare function assertThisTypeAnnotation(node: object | null | undefined, opts?: object | null): asserts node is ThisTypeAnnotation;
declare function assertTupleTypeAnnotation(node: object | null | undefined, opts?: object | null): asserts node is TupleTypeAnnotation;
declare function assertTypeofTypeAnnotation(node: object | null | undefined, opts?: object | null): asserts node is TypeofTypeAnnotation;
declare function assertTypeAlias(node: object | null | undefined, opts?: object | null): asserts node is TypeAlias;
declare function assertTypeAnnotation(node: object | null | undefined, opts?: object | null): asserts node is TypeAnnotation;
declare function assertTypeCastExpression(node: object | null | undefined, opts?: object | null): asserts node is TypeCastExpression;
declare function assertTypeParameter(node: object | null | undefined, opts?: object | null): asserts node is TypeParameter;
declare function assertTypeParameterDeclaration(node: object | null | undefined, opts?: object | null): asserts node is TypeParameterDeclaration;
declare function assertTypeParameterInstantiation(node: object | null | undefined, opts?: object | null): asserts node is TypeParameterInstantiation;
declare function assertUnionTypeAnnotation(node: object | null | undefined, opts?: object | null): asserts node is UnionTypeAnnotation;
declare function assertVariance(node: object | null | undefined, opts?: object | null): asserts node is Variance;
declare function assertVoidTypeAnnotation(node: object | null | undefined, opts?: object | null): asserts node is VoidTypeAnnotation;
declare function assertEnumDeclaration(node: object | null | undefined, opts?: object | null): asserts node is EnumDeclaration;
declare function assertEnumBooleanBody(node: object | null | undefined, opts?: object | null): asserts node is EnumBooleanBody;
declare function assertEnumNumberBody(node: object | null | undefined, opts?: object | null): asserts node is EnumNumberBody;
declare function assertEnumStringBody(node: object | null | undefined, opts?: object | null): asserts node is EnumStringBody;
declare function assertEnumSymbolBody(node: object | null | undefined, opts?: object | null): asserts node is EnumSymbolBody;
declare function assertEnumBooleanMember(node: object | null | undefined, opts?: object | null): asserts node is EnumBooleanMember;
declare function assertEnumNumberMember(node: object | null | undefined, opts?: object | null): asserts node is EnumNumberMember;
declare function assertEnumStringMember(node: object | null | undefined, opts?: object | null): asserts node is EnumStringMember;
declare function assertEnumDefaultedMember(node: object | null | undefined, opts?: object | null): asserts node is EnumDefaultedMember;
declare function assertIndexedAccessType(node: object | null | undefined, opts?: object | null): asserts node is IndexedAccessType;
declare function assertOptionalIndexedAccessType(node: object | null | undefined, opts?: object | null): asserts node is OptionalIndexedAccessType;
declare function assertJSXAttribute(node: object | null | undefined, opts?: object | null): asserts node is JSXAttribute;
declare function assertJSXClosingElement(node: object | null | undefined, opts?: object | null): asserts node is JSXClosingElement;
declare function assertJSXElement(node: object | null | undefined, opts?: object | null): asserts node is JSXElement;
declare function assertJSXEmptyExpression(node: object | null | undefined, opts?: object | null): asserts node is JSXEmptyExpression;
declare function assertJSXExpressionContainer(node: object | null | undefined, opts?: object | null): asserts node is JSXExpressionContainer;
declare function assertJSXSpreadChild(node: object | null | undefined, opts?: object | null): asserts node is JSXSpreadChild;
declare function assertJSXIdentifier(node: object | null | undefined, opts?: object | null): asserts node is JSXIdentifier;
declare function assertJSXMemberExpression(node: object | null | undefined, opts?: object | null): asserts node is JSXMemberExpression;
declare function assertJSXNamespacedName(node: object | null | undefined, opts?: object | null): asserts node is JSXNamespacedName;
declare function assertJSXOpeningElement(node: object | null | undefined, opts?: object | null): asserts node is JSXOpeningElement;
declare function assertJSXSpreadAttribute(node: object | null | undefined, opts?: object | null): asserts node is JSXSpreadAttribute;
declare function assertJSXText(node: object | null | undefined, opts?: object | null): asserts node is JSXText;
declare function assertJSXFragment(node: object | null | undefined, opts?: object | null): asserts node is JSXFragment;
declare function assertJSXOpeningFragment(node: object | null | undefined, opts?: object | null): asserts node is JSXOpeningFragment;
declare function assertJSXClosingFragment(node: object | null | undefined, opts?: object | null): asserts node is JSXClosingFragment;
declare function assertNoop(node: object | null | undefined, opts?: object | null): asserts node is Noop;
declare function assertPlaceholder(node: object | null | undefined, opts?: object | null): asserts node is Placeholder;
declare function assertV8IntrinsicIdentifier(node: object | null | undefined, opts?: object | null): asserts node is V8IntrinsicIdentifier;
declare function assertArgumentPlaceholder(node: object | null | undefined, opts?: object | null): asserts node is ArgumentPlaceholder;
declare function assertBindExpression(node: object | null | undefined, opts?: object | null): asserts node is BindExpression;
declare function assertImportAttribute(node: object | null | undefined, opts?: object | null): asserts node is ImportAttribute;
declare function assertDecorator(node: object | null | undefined, opts?: object | null): asserts node is Decorator;
declare function assertDoExpression(node: object | null | undefined, opts?: object | null): asserts node is DoExpression;
declare function assertExportDefaultSpecifier(node: object | null | undefined, opts?: object | null): asserts node is ExportDefaultSpecifier;
declare function assertRecordExpression(node: object | null | undefined, opts?: object | null): asserts node is RecordExpression;
declare function assertTupleExpression(node: object | null | undefined, opts?: object | null): asserts node is TupleExpression;
declare function assertDecimalLiteral(node: object | null | undefined, opts?: object | null): asserts node is DecimalLiteral;
declare function assertModuleExpression(node: object | null | undefined, opts?: object | null): asserts node is ModuleExpression;
declare function assertTopicReference(node: object | null | undefined, opts?: object | null): asserts node is TopicReference;
declare function assertPipelineTopicExpression(node: object | null | undefined, opts?: object | null): asserts node is PipelineTopicExpression;
declare function assertPipelineBareFunction(node: object | null | undefined, opts?: object | null): asserts node is PipelineBareFunction;
declare function assertPipelinePrimaryTopicReference(node: object | null | undefined, opts?: object | null): asserts node is PipelinePrimaryTopicReference;
declare function assertTSParameterProperty(node: object | null | undefined, opts?: object | null): asserts node is TSParameterProperty;
declare function assertTSDeclareFunction(node: object | null | undefined, opts?: object | null): asserts node is TSDeclareFunction;
declare function assertTSDeclareMethod(node: object | null | undefined, opts?: object | null): asserts node is TSDeclareMethod;
declare function assertTSQualifiedName(node: object | null | undefined, opts?: object | null): asserts node is TSQualifiedName;
declare function assertTSCallSignatureDeclaration(node: object | null | undefined, opts?: object | null): asserts node is TSCallSignatureDeclaration;
declare function assertTSConstructSignatureDeclaration(node: object | null | undefined, opts?: object | null): asserts node is TSConstructSignatureDeclaration;
declare function assertTSPropertySignature(node: object | null | undefined, opts?: object | null): asserts node is TSPropertySignature;
declare function assertTSMethodSignature(node: object | null | undefined, opts?: object | null): asserts node is TSMethodSignature;
declare function assertTSIndexSignature(node: object | null | undefined, opts?: object | null): asserts node is TSIndexSignature;
declare function assertTSAnyKeyword(node: object | null | undefined, opts?: object | null): asserts node is TSAnyKeyword;
declare function assertTSBooleanKeyword(node: object | null | undefined, opts?: object | null): asserts node is TSBooleanKeyword;
declare function assertTSBigIntKeyword(node: object | null | undefined, opts?: object | null): asserts node is TSBigIntKeyword;
declare function assertTSIntrinsicKeyword(node: object | null | undefined, opts?: object | null): asserts node is TSIntrinsicKeyword;
declare function assertTSNeverKeyword(node: object | null | undefined, opts?: object | null): asserts node is TSNeverKeyword;
declare function assertTSNullKeyword(node: object | null | undefined, opts?: object | null): asserts node is TSNullKeyword;
declare function assertTSNumberKeyword(node: object | null | undefined, opts?: object | null): asserts node is TSNumberKeyword;
declare function assertTSObjectKeyword(node: object | null | undefined, opts?: object | null): asserts node is TSObjectKeyword;
declare function assertTSStringKeyword(node: object | null | undefined, opts?: object | null): asserts node is TSStringKeyword;
declare function assertTSSymbolKeyword(node: object | null | undefined, opts?: object | null): asserts node is TSSymbolKeyword;
declare function assertTSUndefinedKeyword(node: object | null | undefined, opts?: object | null): asserts node is TSUndefinedKeyword;
declare function assertTSUnknownKeyword(node: object | null | undefined, opts?: object | null): asserts node is TSUnknownKeyword;
declare function assertTSVoidKeyword(node: object | null | undefined, opts?: object | null): asserts node is TSVoidKeyword;
declare function assertTSThisType(node: object | null | undefined, opts?: object | null): asserts node is TSThisType;
declare function assertTSFunctionType(node: object | null | undefined, opts?: object | null): asserts node is TSFunctionType;
declare function assertTSConstructorType(node: object | null | undefined, opts?: object | null): asserts node is TSConstructorType;
declare function assertTSTypeReference(node: object | null | undefined, opts?: object | null): asserts node is TSTypeReference;
declare function assertTSTypePredicate(node: object | null | undefined, opts?: object | null): asserts node is TSTypePredicate;
declare function assertTSTypeQuery(node: object | null | undefined, opts?: object | null): asserts node is TSTypeQuery;
declare function assertTSTypeLiteral(node: object | null | undefined, opts?: object | null): asserts node is TSTypeLiteral;
declare function assertTSArrayType(node: object | null | undefined, opts?: object | null): asserts node is TSArrayType;
declare function assertTSTupleType(node: object | null | undefined, opts?: object | null): asserts node is TSTupleType;
declare function assertTSOptionalType(node: object | null | undefined, opts?: object | null): asserts node is TSOptionalType;
declare function assertTSRestType(node: object | null | undefined, opts?: object | null): asserts node is TSRestType;
declare function assertTSNamedTupleMember(node: object | null | undefined, opts?: object | null): asserts node is TSNamedTupleMember;
declare function assertTSUnionType(node: object | null | undefined, opts?: object | null): asserts node is TSUnionType;
declare function assertTSIntersectionType(node: object | null | undefined, opts?: object | null): asserts node is TSIntersectionType;
declare function assertTSConditionalType(node: object | null | undefined, opts?: object | null): asserts node is TSConditionalType;
declare function assertTSInferType(node: object | null | undefined, opts?: object | null): asserts node is TSInferType;
declare function assertTSParenthesizedType(node: object | null | undefined, opts?: object | null): asserts node is TSParenthesizedType;
declare function assertTSTypeOperator(node: object | null | undefined, opts?: object | null): asserts node is TSTypeOperator;
declare function assertTSIndexedAccessType(node: object | null | undefined, opts?: object | null): asserts node is TSIndexedAccessType;
declare function assertTSMappedType(node: object | null | undefined, opts?: object | null): asserts node is TSMappedType;
declare function assertTSLiteralType(node: object | null | undefined, opts?: object | null): asserts node is TSLiteralType;
declare function assertTSExpressionWithTypeArguments(node: object | null | undefined, opts?: object | null): asserts node is TSExpressionWithTypeArguments;
declare function assertTSInterfaceDeclaration(node: object | null | undefined, opts?: object | null): asserts node is TSInterfaceDeclaration;
declare function assertTSInterfaceBody(node: object | null | undefined, opts?: object | null): asserts node is TSInterfaceBody;
declare function assertTSTypeAliasDeclaration(node: object | null | undefined, opts?: object | null): asserts node is TSTypeAliasDeclaration;
declare function assertTSInstantiationExpression(node: object | null | undefined, opts?: object | null): asserts node is TSInstantiationExpression;
declare function assertTSAsExpression(node: object | null | undefined, opts?: object | null): asserts node is TSAsExpression;
declare function assertTSSatisfiesExpression(node: object | null | undefined, opts?: object | null): asserts node is TSSatisfiesExpression;
declare function assertTSTypeAssertion(node: object | null | undefined, opts?: object | null): asserts node is TSTypeAssertion;
declare function assertTSEnumDeclaration(node: object | null | undefined, opts?: object | null): asserts node is TSEnumDeclaration;
declare function assertTSEnumMember(node: object | null | undefined, opts?: object | null): asserts node is TSEnumMember;
declare function assertTSModuleDeclaration(node: object | null | undefined, opts?: object | null): asserts node is TSModuleDeclaration;
declare function assertTSModuleBlock(node: object | null | undefined, opts?: object | null): asserts node is TSModuleBlock;
declare function assertTSImportType(node: object | null | undefined, opts?: object | null): asserts node is TSImportType;
declare function assertTSImportEqualsDeclaration(node: object | null | undefined, opts?: object | null): asserts node is TSImportEqualsDeclaration;
declare function assertTSExternalModuleReference(node: object | null | undefined, opts?: object | null): asserts node is TSExternalModuleReference;
declare function assertTSNonNullExpression(node: object | null | undefined, opts?: object | null): asserts node is TSNonNullExpression;
declare function assertTSExportAssignment(node: object | null | undefined, opts?: object | null): asserts node is TSExportAssignment;
declare function assertTSNamespaceExportDeclaration(node: object | null | undefined, opts?: object | null): asserts node is TSNamespaceExportDeclaration;
declare function assertTSTypeAnnotation(node: object | null | undefined, opts?: object | null): asserts node is TSTypeAnnotation;
declare function assertTSTypeParameterInstantiation(node: object | null | undefined, opts?: object | null): asserts node is TSTypeParameterInstantiation;
declare function assertTSTypeParameterDeclaration(node: object | null | undefined, opts?: object | null): asserts node is TSTypeParameterDeclaration;
declare function assertTSTypeParameter(node: object | null | undefined, opts?: object | null): asserts node is TSTypeParameter;
declare function assertStandardized(node: object | null | undefined, opts?: object | null): asserts node is Standardized;
declare function assertExpression(node: object | null | undefined, opts?: object | null): asserts node is Expression;
declare function assertBinary(node: object | null | undefined, opts?: object | null): asserts node is Binary;
declare function assertScopable(node: object | null | undefined, opts?: object | null): asserts node is Scopable;
declare function assertBlockParent(node: object | null | undefined, opts?: object | null): asserts node is BlockParent;
declare function assertBlock(node: object | null | undefined, opts?: object | null): asserts node is Block;
declare function assertStatement(node: object | null | undefined, opts?: object | null): asserts node is Statement;
declare function assertTerminatorless(node: object | null | undefined, opts?: object | null): asserts node is Terminatorless;
declare function assertCompletionStatement(node: object | null | undefined, opts?: object | null): asserts node is CompletionStatement;
declare function assertConditional(node: object | null | undefined, opts?: object | null): asserts node is Conditional;
declare function assertLoop(node: object | null | undefined, opts?: object | null): asserts node is Loop;
declare function assertWhile(node: object | null | undefined, opts?: object | null): asserts node is While;
declare function assertExpressionWrapper(node: object | null | undefined, opts?: object | null): asserts node is ExpressionWrapper;
declare function assertFor(node: object | null | undefined, opts?: object | null): asserts node is For;
declare function assertForXStatement(node: object | null | undefined, opts?: object | null): asserts node is ForXStatement;
declare function assertFunction(node: object | null | undefined, opts?: object | null): asserts node is Function;
declare function assertFunctionParent(node: object | null | undefined, opts?: object | null): asserts node is FunctionParent;
declare function assertPureish(node: object | null | undefined, opts?: object | null): asserts node is Pureish;
declare function assertDeclaration(node: object | null | undefined, opts?: object | null): asserts node is Declaration;
declare function assertPatternLike(node: object | null | undefined, opts?: object | null): asserts node is PatternLike;
declare function assertLVal(node: object | null | undefined, opts?: object | null): asserts node is LVal;
declare function assertTSEntityName(node: object | null | undefined, opts?: object | null): asserts node is TSEntityName;
declare function assertLiteral(node: object | null | undefined, opts?: object | null): asserts node is Literal;
declare function assertImmutable(node: object | null | undefined, opts?: object | null): asserts node is Immutable;
declare function assertUserWhitespacable(node: object | null | undefined, opts?: object | null): asserts node is UserWhitespacable;
declare function assertMethod(node: object | null | undefined, opts?: object | null): asserts node is Method;
declare function assertObjectMember(node: object | null | undefined, opts?: object | null): asserts node is ObjectMember;
declare function assertProperty(node: object | null | undefined, opts?: object | null): asserts node is Property;
declare function assertUnaryLike(node: object | null | undefined, opts?: object | null): asserts node is UnaryLike;
declare function assertPattern(node: object | null | undefined, opts?: object | null): asserts node is Pattern;
declare function assertClass(node: object | null | undefined, opts?: object | null): asserts node is Class;
declare function assertImportOrExportDeclaration(node: object | null | undefined, opts?: object | null): asserts node is ImportOrExportDeclaration;
declare function assertExportDeclaration(node: object | null | undefined, opts?: object | null): asserts node is ExportDeclaration;
declare function assertModuleSpecifier(node: object | null | undefined, opts?: object | null): asserts node is ModuleSpecifier;
declare function assertAccessor(node: object | null | undefined, opts?: object | null): asserts node is Accessor;
declare function assertPrivate(node: object | null | undefined, opts?: object | null): asserts node is Private;
declare function assertFlow(node: object | null | undefined, opts?: object | null): asserts node is Flow;
declare function assertFlowType(node: object | null | undefined, opts?: object | null): asserts node is FlowType;
declare function assertFlowBaseAnnotation(node: object | null | undefined, opts?: object | null): asserts node is FlowBaseAnnotation;
declare function assertFlowDeclaration(node: object | null | undefined, opts?: object | null): asserts node is FlowDeclaration;
declare function assertFlowPredicate(node: object | null | undefined, opts?: object | null): asserts node is FlowPredicate;
declare function assertEnumBody(node: object | null | undefined, opts?: object | null): asserts node is EnumBody;
declare function assertEnumMember(node: object | null | undefined, opts?: object | null): asserts node is EnumMember;
declare function assertJSX(node: object | null | undefined, opts?: object | null): asserts node is JSX;
declare function assertMiscellaneous(node: object | null | undefined, opts?: object | null): asserts node is Miscellaneous;
declare function assertTypeScript(node: object | null | undefined, opts?: object | null): asserts node is TypeScript;
declare function assertTSTypeElement(node: object | null | undefined, opts?: object | null): asserts node is TSTypeElement;
declare function assertTSType(node: object | null | undefined, opts?: object | null): asserts node is TSType;
declare function assertTSBaseType(node: object | null | undefined, opts?: object | null): asserts node is TSBaseType;
declare function assertNumberLiteral(node: any, opts: any): void;
declare function assertRegexLiteral(node: any, opts: any): void;
declare function assertRestProperty(node: any, opts: any): void;
declare function assertSpreadProperty(node: any, opts: any): void;
declare function assertModuleDeclaration(node: any, opts: any): void;

declare const _default$4: {
    (type: "string"): StringTypeAnnotation;
    (type: "number"): NumberTypeAnnotation;
    (type: "undefined"): VoidTypeAnnotation;
    (type: "boolean"): BooleanTypeAnnotation;
    (type: "function"): GenericTypeAnnotation;
    (type: "object"): GenericTypeAnnotation;
    (type: "symbol"): GenericTypeAnnotation;
    (type: "bigint"): AnyTypeAnnotation;
};
//# sourceMappingURL=createTypeAnnotationBasedOnTypeof.d.ts.map

/**
 * Takes an array of `types` and flattens them, removing duplicates and
 * returns a `UnionTypeAnnotation` node containing them.
 */
declare function createFlowUnionType<T extends FlowType>(types: [T] | Array<T>): T | UnionTypeAnnotation;

/**
 * Takes an array of `types` and flattens them, removing duplicates and
 * returns a `UnionTypeAnnotation` node containing them.
 */
declare function createTSUnionType(typeAnnotations: Array<TSTypeAnnotation | TSType>): TSType;

declare function arrayExpression(elements?: Array<null | Expression | SpreadElement>): ArrayExpression;
declare function assignmentExpression(operator: string, left: LVal | OptionalMemberExpression, right: Expression): AssignmentExpression;
declare function binaryExpression(operator: "+" | "-" | "/" | "%" | "*" | "**" | "&" | "|" | ">>" | ">>>" | "<<" | "^" | "==" | "===" | "!=" | "!==" | "in" | "instanceof" | ">" | "<" | ">=" | "<=" | "|>", left: Expression | PrivateName, right: Expression): BinaryExpression;
declare function interpreterDirective(value: string): InterpreterDirective;
declare function directive(value: DirectiveLiteral): Directive;
declare function directiveLiteral(value: string): DirectiveLiteral;
declare function blockStatement(body: Array<Statement>, directives?: Array<Directive>): BlockStatement;
declare function breakStatement(label?: Identifier | null): BreakStatement;
declare function callExpression(callee: Expression | Super | V8IntrinsicIdentifier, _arguments: Array<Expression | SpreadElement | ArgumentPlaceholder>): CallExpression;
declare function catchClause(param: Identifier | ArrayPattern | ObjectPattern | null | undefined, body: BlockStatement): CatchClause;
declare function conditionalExpression(test: Expression, consequent: Expression, alternate: Expression): ConditionalExpression;
declare function continueStatement(label?: Identifier | null): ContinueStatement;
declare function debuggerStatement(): DebuggerStatement;
declare function doWhileStatement(test: Expression, body: Statement): DoWhileStatement;
declare function emptyStatement(): EmptyStatement;
declare function expressionStatement(expression: Expression): ExpressionStatement;
declare function file(program: Program, comments?: Array<CommentBlock | CommentLine> | null, tokens?: Array<any> | null): File;
declare function forInStatement(left: VariableDeclaration | LVal, right: Expression, body: Statement): ForInStatement;
declare function forStatement(init: VariableDeclaration | Expression | null | undefined, test: Expression | null | undefined, update: Expression | null | undefined, body: Statement): ForStatement;
declare function functionDeclaration(id: Identifier | null | undefined, params: Array<Identifier | Pattern | RestElement>, body: BlockStatement, generator?: boolean, async?: boolean): FunctionDeclaration;
declare function functionExpression(id: Identifier | null | undefined, params: Array<Identifier | Pattern | RestElement>, body: BlockStatement, generator?: boolean, async?: boolean): FunctionExpression;
declare function identifier(name: string): Identifier;
declare function ifStatement(test: Expression, consequent: Statement, alternate?: Statement | null): IfStatement;
declare function labeledStatement(label: Identifier, body: Statement): LabeledStatement;
declare function stringLiteral(value: string): StringLiteral;
declare function numericLiteral(value: number): NumericLiteral;
declare function nullLiteral(): NullLiteral;
declare function booleanLiteral(value: boolean): BooleanLiteral;
declare function regExpLiteral(pattern: string, flags?: string): RegExpLiteral;
declare function logicalExpression(operator: "||" | "&&" | "??", left: Expression, right: Expression): LogicalExpression;
declare function memberExpression(object: Expression | Super, property: Expression | Identifier | PrivateName, computed?: boolean, optional?: true | false | null): MemberExpression;
declare function newExpression(callee: Expression | Super | V8IntrinsicIdentifier, _arguments: Array<Expression | SpreadElement | ArgumentPlaceholder>): NewExpression;
declare function program(body: Array<Statement>, directives?: Array<Directive>, sourceType?: "script" | "module", interpreter?: InterpreterDirective | null): Program;
declare function objectExpression(properties: Array<ObjectMethod | ObjectProperty | SpreadElement>): ObjectExpression;
declare function objectMethod(kind: "method" | "get" | "set" | undefined, key: Expression | Identifier | StringLiteral | NumericLiteral | BigIntLiteral, params: Array<Identifier | Pattern | RestElement>, body: BlockStatement, computed?: boolean, generator?: boolean, async?: boolean): ObjectMethod;
declare function objectProperty(key: Expression | Identifier | StringLiteral | NumericLiteral | BigIntLiteral | DecimalLiteral | PrivateName, value: Expression | PatternLike, computed?: boolean, shorthand?: boolean, decorators?: Array<Decorator> | null): ObjectProperty;
declare function restElement(argument: LVal): RestElement;
declare function returnStatement(argument?: Expression | null): ReturnStatement;
declare function sequenceExpression(expressions: Array<Expression>): SequenceExpression;
declare function parenthesizedExpression(expression: Expression): ParenthesizedExpression;
declare function switchCase(test: Expression | null | undefined, consequent: Array<Statement>): SwitchCase;
declare function switchStatement(discriminant: Expression, cases: Array<SwitchCase>): SwitchStatement;
declare function thisExpression(): ThisExpression;
declare function throwStatement(argument: Expression): ThrowStatement;
declare function tryStatement(block: BlockStatement, handler?: CatchClause | null, finalizer?: BlockStatement | null): TryStatement;
declare function unaryExpression(operator: "void" | "throw" | "delete" | "!" | "+" | "-" | "~" | "typeof", argument: Expression, prefix?: boolean): UnaryExpression;
declare function updateExpression(operator: "++" | "--", argument: Expression, prefix?: boolean): UpdateExpression;
declare function variableDeclaration(kind: "var" | "let" | "const" | "using" | "await using", declarations: Array<VariableDeclarator>): VariableDeclaration;
declare function variableDeclarator(id: LVal, init?: Expression | null): VariableDeclarator;
declare function whileStatement(test: Expression, body: Statement): WhileStatement;
declare function withStatement(object: Expression, body: Statement): WithStatement;
declare function assignmentPattern(left: Identifier | ObjectPattern | ArrayPattern | MemberExpression | TSAsExpression | TSSatisfiesExpression | TSTypeAssertion | TSNonNullExpression, right: Expression): AssignmentPattern;
declare function arrayPattern(elements: Array<null | PatternLike | LVal>): ArrayPattern;
declare function arrowFunctionExpression(params: Array<Identifier | Pattern | RestElement>, body: BlockStatement | Expression, async?: boolean): ArrowFunctionExpression;
declare function classBody(body: Array<ClassMethod | ClassPrivateMethod | ClassProperty | ClassPrivateProperty | ClassAccessorProperty | TSDeclareMethod | TSIndexSignature | StaticBlock>): ClassBody;
declare function classExpression(id: Identifier | null | undefined, superClass: Expression | null | undefined, body: ClassBody, decorators?: Array<Decorator> | null): ClassExpression;
declare function classDeclaration(id: Identifier | null | undefined, superClass: Expression | null | undefined, body: ClassBody, decorators?: Array<Decorator> | null): ClassDeclaration;
declare function exportAllDeclaration(source: StringLiteral): ExportAllDeclaration;
declare function exportDefaultDeclaration(declaration: TSDeclareFunction | FunctionDeclaration | ClassDeclaration | Expression): ExportDefaultDeclaration;
declare function exportNamedDeclaration(declaration?: Declaration | null, specifiers?: Array<ExportSpecifier | ExportDefaultSpecifier | ExportNamespaceSpecifier>, source?: StringLiteral | null): ExportNamedDeclaration;
declare function exportSpecifier(local: Identifier, exported: Identifier | StringLiteral): ExportSpecifier;
declare function forOfStatement(left: VariableDeclaration | LVal, right: Expression, body: Statement, _await?: boolean): ForOfStatement;
declare function importDeclaration(specifiers: Array<ImportSpecifier | ImportDefaultSpecifier | ImportNamespaceSpecifier>, source: StringLiteral): ImportDeclaration;
declare function importDefaultSpecifier(local: Identifier): ImportDefaultSpecifier;
declare function importNamespaceSpecifier(local: Identifier): ImportNamespaceSpecifier;
declare function importSpecifier(local: Identifier, imported: Identifier | StringLiteral): ImportSpecifier;
declare function importExpression(source: Expression, options?: Expression | null): ImportExpression;
declare function metaProperty(meta: Identifier, property: Identifier): MetaProperty;
declare function classMethod(kind: "get" | "set" | "method" | "constructor" | undefined, key: Identifier | StringLiteral | NumericLiteral | BigIntLiteral | Expression, params: Array<Identifier | Pattern | RestElement | TSParameterProperty>, body: BlockStatement, computed?: boolean, _static?: boolean, generator?: boolean, async?: boolean): ClassMethod;
declare function objectPattern(properties: Array<RestElement | ObjectProperty>): ObjectPattern;
declare function spreadElement(argument: Expression): SpreadElement;
declare function _super(): Super;

declare function taggedTemplateExpression(tag: Expression, quasi: TemplateLiteral): TaggedTemplateExpression;
declare function templateElement(value: {
    raw: string;
    cooked?: string;
}, tail?: boolean): TemplateElement;
declare function templateLiteral(quasis: Array<TemplateElement>, expressions: Array<Expression | TSType>): TemplateLiteral;
declare function yieldExpression(argument?: Expression | null, delegate?: boolean): YieldExpression;
declare function awaitExpression(argument: Expression): AwaitExpression;
declare function _import(): Import;

declare function bigIntLiteral(value: string): BigIntLiteral;
declare function exportNamespaceSpecifier(exported: Identifier): ExportNamespaceSpecifier;
declare function optionalMemberExpression(object: Expression, property: Expression | Identifier, computed: boolean | undefined, optional: boolean): OptionalMemberExpression;
declare function optionalCallExpression(callee: Expression, _arguments: Array<Expression | SpreadElement | ArgumentPlaceholder>, optional: boolean): OptionalCallExpression;
declare function classProperty(key: Identifier | StringLiteral | NumericLiteral | BigIntLiteral | Expression, value?: Expression | null, typeAnnotation?: TypeAnnotation | TSTypeAnnotation | Noop | null, decorators?: Array<Decorator> | null, computed?: boolean, _static?: boolean): ClassProperty;
declare function classAccessorProperty(key: Identifier | StringLiteral | NumericLiteral | BigIntLiteral | Expression | PrivateName, value?: Expression | null, typeAnnotation?: TypeAnnotation | TSTypeAnnotation | Noop | null, decorators?: Array<Decorator> | null, computed?: boolean, _static?: boolean): ClassAccessorProperty;
declare function classPrivateProperty(key: PrivateName, value?: Expression | null, decorators?: Array<Decorator> | null, _static?: boolean): ClassPrivateProperty;
declare function classPrivateMethod(kind: "get" | "set" | "method" | undefined, key: PrivateName, params: Array<Identifier | Pattern | RestElement | TSParameterProperty>, body: BlockStatement, _static?: boolean): ClassPrivateMethod;
declare function privateName(id: Identifier): PrivateName;
declare function staticBlock(body: Array<Statement>): StaticBlock;
declare function anyTypeAnnotation(): AnyTypeAnnotation;
declare function arrayTypeAnnotation(elementType: FlowType): ArrayTypeAnnotation;
declare function booleanTypeAnnotation(): BooleanTypeAnnotation;
declare function booleanLiteralTypeAnnotation(value: boolean): BooleanLiteralTypeAnnotation;
declare function nullLiteralTypeAnnotation(): NullLiteralTypeAnnotation;
declare function classImplements(id: Identifier, typeParameters?: TypeParameterInstantiation | null): ClassImplements;
declare function declareClass(id: Identifier, typeParameters: TypeParameterDeclaration | null | undefined, _extends: Array<InterfaceExtends> | null | undefined, body: ObjectTypeAnnotation): DeclareClass;
declare function declareFunction(id: Identifier): DeclareFunction;
declare function declareInterface(id: Identifier, typeParameters: TypeParameterDeclaration | null | undefined, _extends: Array<InterfaceExtends> | null | undefined, body: ObjectTypeAnnotation): DeclareInterface;
declare function declareModule(id: Identifier | StringLiteral, body: BlockStatement, kind?: "CommonJS" | "ES" | null): DeclareModule;
declare function declareModuleExports(typeAnnotation: TypeAnnotation): DeclareModuleExports;
declare function declareTypeAlias(id: Identifier, typeParameters: TypeParameterDeclaration | null | undefined, right: FlowType): DeclareTypeAlias;
declare function declareOpaqueType(id: Identifier, typeParameters?: TypeParameterDeclaration | null, supertype?: FlowType | null): DeclareOpaqueType;
declare function declareVariable(id: Identifier): DeclareVariable;
declare function declareExportDeclaration(declaration?: Flow | null, specifiers?: Array<ExportSpecifier | ExportNamespaceSpecifier> | null, source?: StringLiteral | null): DeclareExportDeclaration;
declare function declareExportAllDeclaration(source: StringLiteral): DeclareExportAllDeclaration;
declare function declaredPredicate(value: Flow): DeclaredPredicate;
declare function existsTypeAnnotation(): ExistsTypeAnnotation;
declare function functionTypeAnnotation(typeParameters: TypeParameterDeclaration | null | undefined, params: Array<FunctionTypeParam>, rest: FunctionTypeParam | null | undefined, returnType: FlowType): FunctionTypeAnnotation;
declare function functionTypeParam(name: Identifier | null | undefined, typeAnnotation: FlowType): FunctionTypeParam;
declare function genericTypeAnnotation(id: Identifier | QualifiedTypeIdentifier, typeParameters?: TypeParameterInstantiation | null): GenericTypeAnnotation;
declare function inferredPredicate(): InferredPredicate;
declare function interfaceExtends(id: Identifier | QualifiedTypeIdentifier, typeParameters?: TypeParameterInstantiation | null): InterfaceExtends;
declare function interfaceDeclaration(id: Identifier, typeParameters: TypeParameterDeclaration | null | undefined, _extends: Array<InterfaceExtends> | null | undefined, body: ObjectTypeAnnotation): InterfaceDeclaration;
declare function interfaceTypeAnnotation(_extends: Array<InterfaceExtends> | null | undefined, body: ObjectTypeAnnotation): InterfaceTypeAnnotation;
declare function intersectionTypeAnnotation(types: Array<FlowType>): IntersectionTypeAnnotation;
declare function mixedTypeAnnotation(): MixedTypeAnnotation;
declare function emptyTypeAnnotation(): EmptyTypeAnnotation;
declare function nullableTypeAnnotation(typeAnnotation: FlowType): NullableTypeAnnotation;
declare function numberLiteralTypeAnnotation(value: number): NumberLiteralTypeAnnotation;
declare function numberTypeAnnotation(): NumberTypeAnnotation;
declare function objectTypeAnnotation(properties: Array<ObjectTypeProperty | ObjectTypeSpreadProperty>, indexers?: Array<ObjectTypeIndexer>, callProperties?: Array<ObjectTypeCallProperty>, internalSlots?: Array<ObjectTypeInternalSlot>, exact?: boolean): ObjectTypeAnnotation;
declare function objectTypeInternalSlot(id: Identifier, value: FlowType, optional: boolean, _static: boolean, method: boolean): ObjectTypeInternalSlot;
declare function objectTypeCallProperty(value: FlowType): ObjectTypeCallProperty;
declare function objectTypeIndexer(id: Identifier | null | undefined, key: FlowType, value: FlowType, variance?: Variance | null): ObjectTypeIndexer;
declare function objectTypeProperty(key: Identifier | StringLiteral, value: FlowType, variance?: Variance | null): ObjectTypeProperty;
declare function objectTypeSpreadProperty(argument: FlowType): ObjectTypeSpreadProperty;
declare function opaqueType(id: Identifier, typeParameters: TypeParameterDeclaration | null | undefined, supertype: FlowType | null | undefined, impltype: FlowType): OpaqueType;
declare function qualifiedTypeIdentifier(id: Identifier, qualification: Identifier | QualifiedTypeIdentifier): QualifiedTypeIdentifier;
declare function stringLiteralTypeAnnotation(value: string): StringLiteralTypeAnnotation;
declare function stringTypeAnnotation(): StringTypeAnnotation;
declare function symbolTypeAnnotation(): SymbolTypeAnnotation;
declare function thisTypeAnnotation(): ThisTypeAnnotation;
declare function tupleTypeAnnotation(types: Array<FlowType>): TupleTypeAnnotation;
declare function typeofTypeAnnotation(argument: FlowType): TypeofTypeAnnotation;
declare function typeAlias(id: Identifier, typeParameters: TypeParameterDeclaration | null | undefined, right: FlowType): TypeAlias;
declare function typeAnnotation(typeAnnotation: FlowType): TypeAnnotation;
declare function typeCastExpression(expression: Expression, typeAnnotation: TypeAnnotation): TypeCastExpression;
declare function typeParameter(bound?: TypeAnnotation | null, _default?: FlowType | null, variance?: Variance | null): TypeParameter;
declare function typeParameterDeclaration(params: Array<TypeParameter>): TypeParameterDeclaration;
declare function typeParameterInstantiation(params: Array<FlowType>): TypeParameterInstantiation;
declare function unionTypeAnnotation(types: Array<FlowType>): UnionTypeAnnotation;
declare function variance(kind: "minus" | "plus"): Variance;
declare function voidTypeAnnotation(): VoidTypeAnnotation;
declare function enumDeclaration(id: Identifier, body: EnumBooleanBody | EnumNumberBody | EnumStringBody | EnumSymbolBody): EnumDeclaration;
declare function enumBooleanBody(members: Array<EnumBooleanMember>): EnumBooleanBody;
declare function enumNumberBody(members: Array<EnumNumberMember>): EnumNumberBody;
declare function enumStringBody(members: Array<EnumStringMember | EnumDefaultedMember>): EnumStringBody;
declare function enumSymbolBody(members: Array<EnumDefaultedMember>): EnumSymbolBody;
declare function enumBooleanMember(id: Identifier): EnumBooleanMember;
declare function enumNumberMember(id: Identifier, init: NumericLiteral): EnumNumberMember;
declare function enumStringMember(id: Identifier, init: StringLiteral): EnumStringMember;
declare function enumDefaultedMember(id: Identifier): EnumDefaultedMember;
declare function indexedAccessType(objectType: FlowType, indexType: FlowType): IndexedAccessType;
declare function optionalIndexedAccessType(objectType: FlowType, indexType: FlowType): OptionalIndexedAccessType;
declare function jsxAttribute(name: JSXIdentifier | JSXNamespacedName, value?: JSXElement | JSXFragment | StringLiteral | JSXExpressionContainer | null): JSXAttribute;

declare function jsxClosingElement(name: JSXIdentifier | JSXMemberExpression | JSXNamespacedName): JSXClosingElement;

declare function jsxElement(openingElement: JSXOpeningElement, closingElement: JSXClosingElement | null | undefined, children: Array<JSXText | JSXExpressionContainer | JSXSpreadChild | JSXElement | JSXFragment>, selfClosing?: boolean | null): JSXElement;

declare function jsxEmptyExpression(): JSXEmptyExpression;

declare function jsxExpressionContainer(expression: Expression | JSXEmptyExpression): JSXExpressionContainer;

declare function jsxSpreadChild(expression: Expression): JSXSpreadChild;

declare function jsxIdentifier(name: string): JSXIdentifier;

declare function jsxMemberExpression(object: JSXMemberExpression | JSXIdentifier, property: JSXIdentifier): JSXMemberExpression;

declare function jsxNamespacedName(namespace: JSXIdentifier, name: JSXIdentifier): JSXNamespacedName;

declare function jsxOpeningElement(name: JSXIdentifier | JSXMemberExpression | JSXNamespacedName, attributes: Array<JSXAttribute | JSXSpreadAttribute>, selfClosing?: boolean): JSXOpeningElement;

declare function jsxSpreadAttribute(argument: Expression): JSXSpreadAttribute;

declare function jsxText(value: string): JSXText;

declare function jsxFragment(openingFragment: JSXOpeningFragment, closingFragment: JSXClosingFragment, children: Array<JSXText | JSXExpressionContainer | JSXSpreadChild | JSXElement | JSXFragment>): JSXFragment;

declare function jsxOpeningFragment(): JSXOpeningFragment;

declare function jsxClosingFragment(): JSXClosingFragment;

declare function noop(): Noop;
declare function placeholder(expectedNode: "Identifier" | "StringLiteral" | "Expression" | "Statement" | "Declaration" | "BlockStatement" | "ClassBody" | "Pattern", name: Identifier): Placeholder;
declare function v8IntrinsicIdentifier(name: string): V8IntrinsicIdentifier;
declare function argumentPlaceholder(): ArgumentPlaceholder;
declare function bindExpression(object: Expression, callee: Expression): BindExpression;
declare function importAttribute(key: Identifier | StringLiteral, value: StringLiteral): ImportAttribute;
declare function decorator(expression: Expression): Decorator;
declare function doExpression(body: BlockStatement, async?: boolean): DoExpression;
declare function exportDefaultSpecifier(exported: Identifier): ExportDefaultSpecifier;
declare function recordExpression(properties: Array<ObjectProperty | SpreadElement>): RecordExpression;
declare function tupleExpression(elements?: Array<Expression | SpreadElement>): TupleExpression;
declare function decimalLiteral(value: string): DecimalLiteral;
declare function moduleExpression(body: Program): ModuleExpression;
declare function topicReference(): TopicReference;
declare function pipelineTopicExpression(expression: Expression): PipelineTopicExpression;
declare function pipelineBareFunction(callee: Expression): PipelineBareFunction;
declare function pipelinePrimaryTopicReference(): PipelinePrimaryTopicReference;
declare function tsParameterProperty(parameter: Identifier | AssignmentPattern): TSParameterProperty;

declare function tsDeclareFunction(id: Identifier | null | undefined, typeParameters: TSTypeParameterDeclaration | Noop | null | undefined, params: Array<Identifier | Pattern | RestElement>, returnType?: TSTypeAnnotation | Noop | null): TSDeclareFunction;

declare function tsDeclareMethod(decorators: Array<Decorator> | null | undefined, key: Identifier | StringLiteral | NumericLiteral | BigIntLiteral | Expression, typeParameters: TSTypeParameterDeclaration | Noop | null | undefined, params: Array<Identifier | Pattern | RestElement | TSParameterProperty>, returnType?: TSTypeAnnotation | Noop | null): TSDeclareMethod;

declare function tsQualifiedName(left: TSEntityName, right: Identifier): TSQualifiedName;

declare function tsCallSignatureDeclaration(typeParameters: TSTypeParameterDeclaration | null | undefined, parameters: Array<ArrayPattern | Identifier | ObjectPattern | RestElement>, typeAnnotation?: TSTypeAnnotation | null): TSCallSignatureDeclaration;

declare function tsConstructSignatureDeclaration(typeParameters: TSTypeParameterDeclaration | null | undefined, parameters: Array<ArrayPattern | Identifier | ObjectPattern | RestElement>, typeAnnotation?: TSTypeAnnotation | null): TSConstructSignatureDeclaration;

declare function tsPropertySignature(key: Expression, typeAnnotation?: TSTypeAnnotation | null): TSPropertySignature;

declare function tsMethodSignature(key: Expression, typeParameters: TSTypeParameterDeclaration | null | undefined, parameters: Array<ArrayPattern | Identifier | ObjectPattern | RestElement>, typeAnnotation?: TSTypeAnnotation | null): TSMethodSignature;

declare function tsIndexSignature(parameters: Array<Identifier>, typeAnnotation?: TSTypeAnnotation | null): TSIndexSignature;

declare function tsAnyKeyword(): TSAnyKeyword;

declare function tsBooleanKeyword(): TSBooleanKeyword;

declare function tsBigIntKeyword(): TSBigIntKeyword;

declare function tsIntrinsicKeyword(): TSIntrinsicKeyword;

declare function tsNeverKeyword(): TSNeverKeyword;

declare function tsNullKeyword(): TSNullKeyword;

declare function tsNumberKeyword(): TSNumberKeyword;

declare function tsObjectKeyword(): TSObjectKeyword;

declare function tsStringKeyword(): TSStringKeyword;

declare function tsSymbolKeyword(): TSSymbolKeyword;

declare function tsUndefinedKeyword(): TSUndefinedKeyword;

declare function tsUnknownKeyword(): TSUnknownKeyword;

declare function tsVoidKeyword(): TSVoidKeyword;

declare function tsThisType(): TSThisType;

declare function tsFunctionType(typeParameters: TSTypeParameterDeclaration | null | undefined, parameters: Array<ArrayPattern | Identifier | ObjectPattern | RestElement>, typeAnnotation?: TSTypeAnnotation | null): TSFunctionType;

declare function tsConstructorType(typeParameters: TSTypeParameterDeclaration | null | undefined, parameters: Array<ArrayPattern | Identifier | ObjectPattern | RestElement>, typeAnnotation?: TSTypeAnnotation | null): TSConstructorType;

declare function tsTypeReference(typeName: TSEntityName, typeParameters?: TSTypeParameterInstantiation | null): TSTypeReference;

declare function tsTypePredicate(parameterName: Identifier | TSThisType, typeAnnotation?: TSTypeAnnotation | null, asserts?: boolean | null): TSTypePredicate;

declare function tsTypeQuery(exprName: TSEntityName | TSImportType, typeParameters?: TSTypeParameterInstantiation | null): TSTypeQuery;

declare function tsTypeLiteral(members: Array<TSTypeElement>): TSTypeLiteral;

declare function tsArrayType(elementType: TSType): TSArrayType;

declare function tsTupleType(elementTypes: Array<TSType | TSNamedTupleMember>): TSTupleType;

declare function tsOptionalType(typeAnnotation: TSType): TSOptionalType;

declare function tsRestType(typeAnnotation: TSType): TSRestType;

declare function tsNamedTupleMember(label: Identifier, elementType: TSType, optional?: boolean): TSNamedTupleMember;

declare function tsUnionType(types: Array<TSType>): TSUnionType;

declare function tsIntersectionType(types: Array<TSType>): TSIntersectionType;

declare function tsConditionalType(checkType: TSType, extendsType: TSType, trueType: TSType, falseType: TSType): TSConditionalType;

declare function tsInferType(typeParameter: TSTypeParameter): TSInferType;

declare function tsParenthesizedType(typeAnnotation: TSType): TSParenthesizedType;

declare function tsTypeOperator(typeAnnotation: TSType): TSTypeOperator;

declare function tsIndexedAccessType(objectType: TSType, indexType: TSType): TSIndexedAccessType;

declare function tsMappedType(typeParameter: TSTypeParameter, typeAnnotation?: TSType | null, nameType?: TSType | null): TSMappedType;

declare function tsLiteralType(literal: NumericLiteral | StringLiteral | BooleanLiteral | BigIntLiteral | TemplateLiteral | UnaryExpression): TSLiteralType;

declare function tsExpressionWithTypeArguments(expression: TSEntityName, typeParameters?: TSTypeParameterInstantiation | null): TSExpressionWithTypeArguments;

declare function tsInterfaceDeclaration(id: Identifier, typeParameters: TSTypeParameterDeclaration | null | undefined, _extends: Array<TSExpressionWithTypeArguments> | null | undefined, body: TSInterfaceBody): TSInterfaceDeclaration;

declare function tsInterfaceBody(body: Array<TSTypeElement>): TSInterfaceBody;

declare function tsTypeAliasDeclaration(id: Identifier, typeParameters: TSTypeParameterDeclaration | null | undefined, typeAnnotation: TSType): TSTypeAliasDeclaration;

declare function tsInstantiationExpression(expression: Expression, typeParameters?: TSTypeParameterInstantiation | null): TSInstantiationExpression;

declare function tsAsExpression(expression: Expression, typeAnnotation: TSType): TSAsExpression;

declare function tsSatisfiesExpression(expression: Expression, typeAnnotation: TSType): TSSatisfiesExpression;

declare function tsTypeAssertion(typeAnnotation: TSType, expression: Expression): TSTypeAssertion;

declare function tsEnumDeclaration(id: Identifier, members: Array<TSEnumMember>): TSEnumDeclaration;

declare function tsEnumMember(id: Identifier | StringLiteral, initializer?: Expression | null): TSEnumMember;

declare function tsModuleDeclaration(id: Identifier | StringLiteral, body: TSModuleBlock | TSModuleDeclaration): TSModuleDeclaration;

declare function tsModuleBlock(body: Array<Statement>): TSModuleBlock;

declare function tsImportType(argument: StringLiteral, qualifier?: TSEntityName | null, typeParameters?: TSTypeParameterInstantiation | null): TSImportType;

declare function tsImportEqualsDeclaration(id: Identifier, moduleReference: TSEntityName | TSExternalModuleReference): TSImportEqualsDeclaration;

declare function tsExternalModuleReference(expression: StringLiteral): TSExternalModuleReference;

declare function tsNonNullExpression(expression: Expression): TSNonNullExpression;

declare function tsExportAssignment(expression: Expression): TSExportAssignment;

declare function tsNamespaceExportDeclaration(id: Identifier): TSNamespaceExportDeclaration;

declare function tsTypeAnnotation(typeAnnotation: TSType): TSTypeAnnotation;

declare function tsTypeParameterInstantiation(params: Array<TSType>): TSTypeParameterInstantiation;

declare function tsTypeParameterDeclaration(params: Array<TSTypeParameter>): TSTypeParameterDeclaration;

declare function tsTypeParameter(constraint: TSType | null | undefined, _default: TSType | null | undefined, name: string): TSTypeParameter;

/** @deprecated */
declare function NumberLiteral$1(value: number): NumericLiteral;

/** @deprecated */
declare function RegexLiteral$1(pattern: string, flags?: string): RegExpLiteral;

/** @deprecated */
declare function RestProperty$1(argument: LVal): RestElement;

/** @deprecated */
declare function SpreadProperty$1(argument: Expression): SpreadElement;

declare function buildUndefinedNode(): UnaryExpression;

/**
 * Create a clone of a `node` including only properties belonging to the node.
 * If the second parameter is `false`, cloneNode performs a shallow clone.
 * If the third parameter is true, the cloned nodes exclude location properties.
 */
declare function cloneNode<T extends Node>(node: T, deep?: boolean, withoutLoc?: boolean): T;

/**
 * Create a shallow clone of a `node`, including only
 * properties belonging to the node.
 * @deprecated Use t.cloneNode instead.
 */
declare function clone<T extends Node>(node: T): T;

/**
 * Create a deep clone of a `node` and all of it's child nodes
 * including only properties belonging to the node.
 * @deprecated Use t.cloneNode instead.
 */
declare function cloneDeep<T extends Node>(node: T): T;

/**
 * Create a deep clone of a `node` and all of it's child nodes
 * including only properties belonging to the node.
 * excluding `_private` and location properties.
 */
declare function cloneDeepWithoutLoc<T extends Node>(node: T): T;

/**
 * Create a shallow clone of a `node` excluding `_private` and location properties.
 */
declare function cloneWithoutLoc<T extends Node>(node: T): T;

/**
 * Add comment of certain type to a node.
 */
declare function addComment<T extends Node>(node: T, type: CommentTypeShorthand, content: string, line?: boolean): T;

/**
 * Add comments of certain type to a node.
 */
declare function addComments<T extends Node>(node: T, type: CommentTypeShorthand, comments: Array<Comment>): T;

declare function inheritInnerComments(child: Node, parent: Node): void;

declare function inheritLeadingComments(child: Node, parent: Node): void;

/**
 * Inherit all unique comments from `parent` node to `child` node.
 */
declare function inheritsComments<T extends Node>(child: T, parent: Node): T;

declare function inheritTrailingComments(child: Node, parent: Node): void;

/**
 * Remove comment properties from a node.
 */
declare function removeComments<T extends Node>(node: T): T;

declare const STANDARDIZED_TYPES: ("StringLiteral" | "NumericLiteral" | "JSXText" | "JSXIdentifier" | "AnyTypeAnnotation" | "ArgumentPlaceholder" | "ArrayExpression" | "ArrayPattern" | "ArrayTypeAnnotation" | "ArrowFunctionExpression" | "AssignmentExpression" | "AssignmentPattern" | "AwaitExpression" | "BigIntLiteral" | "BinaryExpression" | "BindExpression" | "BlockStatement" | "BooleanLiteral" | "BooleanLiteralTypeAnnotation" | "BooleanTypeAnnotation" | "BreakStatement" | "CallExpression" | "CatchClause" | "ClassAccessorProperty" | "ClassBody" | "ClassDeclaration" | "ClassExpression" | "ClassImplements" | "ClassMethod" | "ClassPrivateMethod" | "ClassPrivateProperty" | "ClassProperty" | "ConditionalExpression" | "ContinueStatement" | "DebuggerStatement" | "DecimalLiteral" | "DeclareClass" | "DeclareExportAllDeclaration" | "DeclareExportDeclaration" | "DeclareFunction" | "DeclareInterface" | "DeclareModule" | "DeclareModuleExports" | "DeclareOpaqueType" | "DeclareTypeAlias" | "DeclareVariable" | "DeclaredPredicate" | "Decorator" | "Directive" | "DirectiveLiteral" | "DoExpression" | "DoWhileStatement" | "EmptyStatement" | "EmptyTypeAnnotation" | "EnumBooleanBody" | "EnumBooleanMember" | "EnumDeclaration" | "EnumDefaultedMember" | "EnumNumberBody" | "EnumNumberMember" | "EnumStringBody" | "EnumStringMember" | "EnumSymbolBody" | "ExistsTypeAnnotation" | "ExportAllDeclaration" | "ExportDefaultDeclaration" | "ExportDefaultSpecifier" | "ExportNamedDeclaration" | "ExportNamespaceSpecifier" | "ExportSpecifier" | "ExpressionStatement" | "File" | "ForInStatement" | "ForOfStatement" | "ForStatement" | "FunctionDeclaration" | "FunctionExpression" | "FunctionTypeAnnotation" | "FunctionTypeParam" | "GenericTypeAnnotation" | "Identifier" | "IfStatement" | "Import" | "ImportAttribute" | "ImportDeclaration" | "ImportDefaultSpecifier" | "ImportExpression" | "ImportNamespaceSpecifier" | "ImportSpecifier" | "IndexedAccessType" | "InferredPredicate" | "InterfaceDeclaration" | "InterfaceExtends" | "InterfaceTypeAnnotation" | "InterpreterDirective" | "IntersectionTypeAnnotation" | "JSXAttribute" | "JSXClosingElement" | "JSXClosingFragment" | "JSXElement" | "JSXEmptyExpression" | "JSXExpressionContainer" | "JSXFragment" | "JSXMemberExpression" | "JSXNamespacedName" | "JSXOpeningElement" | "JSXOpeningFragment" | "JSXSpreadAttribute" | "JSXSpreadChild" | "LabeledStatement" | "LogicalExpression" | "MemberExpression" | "MetaProperty" | "MixedTypeAnnotation" | "ModuleExpression" | "NewExpression" | "Noop" | "NullLiteral" | "NullLiteralTypeAnnotation" | "NullableTypeAnnotation" | "NumberLiteral" | "NumberLiteralTypeAnnotation" | "NumberTypeAnnotation" | "ObjectExpression" | "ObjectMethod" | "ObjectPattern" | "ObjectProperty" | "ObjectTypeAnnotation" | "ObjectTypeCallProperty" | "ObjectTypeIndexer" | "ObjectTypeInternalSlot" | "ObjectTypeProperty" | "ObjectTypeSpreadProperty" | "OpaqueType" | "OptionalCallExpression" | "OptionalIndexedAccessType" | "OptionalMemberExpression" | "ParenthesizedExpression" | "PipelineBareFunction" | "PipelinePrimaryTopicReference" | "PipelineTopicExpression" | "Placeholder" | "PrivateName" | "Program" | "QualifiedTypeIdentifier" | "RecordExpression" | "RegExpLiteral" | "RegexLiteral" | "RestElement" | "RestProperty" | "ReturnStatement" | "SequenceExpression" | "SpreadElement" | "SpreadProperty" | "StaticBlock" | "StringLiteralTypeAnnotation" | "StringTypeAnnotation" | "Super" | "SwitchCase" | "SwitchStatement" | "SymbolTypeAnnotation" | "TSAnyKeyword" | "TSArrayType" | "TSAsExpression" | "TSBigIntKeyword" | "TSBooleanKeyword" | "TSCallSignatureDeclaration" | "TSConditionalType" | "TSConstructSignatureDeclaration" | "TSConstructorType" | "TSDeclareFunction" | "TSDeclareMethod" | "TSEnumDeclaration" | "TSEnumMember" | "TSExportAssignment" | "TSExpressionWithTypeArguments" | "TSExternalModuleReference" | "TSFunctionType" | "TSImportEqualsDeclaration" | "TSImportType" | "TSIndexSignature" | "TSIndexedAccessType" | "TSInferType" | "TSInstantiationExpression" | "TSInterfaceBody" | "TSInterfaceDeclaration" | "TSIntersectionType" | "TSIntrinsicKeyword" | "TSLiteralType" | "TSMappedType" | "TSMethodSignature" | "TSModuleBlock" | "TSModuleDeclaration" | "TSNamedTupleMember" | "TSNamespaceExportDeclaration" | "TSNeverKeyword" | "TSNonNullExpression" | "TSNullKeyword" | "TSNumberKeyword" | "TSObjectKeyword" | "TSOptionalType" | "TSParameterProperty" | "TSParenthesizedType" | "TSPropertySignature" | "TSQualifiedName" | "TSRestType" | "TSSatisfiesExpression" | "TSStringKeyword" | "TSSymbolKeyword" | "TSThisType" | "TSTupleType" | "TSTypeAliasDeclaration" | "TSTypeAnnotation" | "TSTypeAssertion" | "TSTypeLiteral" | "TSTypeOperator" | "TSTypeParameter" | "TSTypeParameterDeclaration" | "TSTypeParameterInstantiation" | "TSTypePredicate" | "TSTypeQuery" | "TSTypeReference" | "TSUndefinedKeyword" | "TSUnionType" | "TSUnknownKeyword" | "TSVoidKeyword" | "TaggedTemplateExpression" | "TemplateElement" | "TemplateLiteral" | "ThisExpression" | "ThisTypeAnnotation" | "ThrowStatement" | "TopicReference" | "TryStatement" | "TupleExpression" | "TupleTypeAnnotation" | "TypeAlias" | "TypeAnnotation" | "TypeCastExpression" | "TypeParameter" | "TypeParameterDeclaration" | "TypeParameterInstantiation" | "TypeofTypeAnnotation" | "UnaryExpression" | "UnionTypeAnnotation" | "UpdateExpression" | "V8IntrinsicIdentifier" | "VariableDeclaration" | "VariableDeclarator" | "Variance" | "VoidTypeAnnotation" | "WhileStatement" | "WithStatement" | "YieldExpression" | keyof Aliases)[];
declare const EXPRESSION_TYPES: ("StringLiteral" | "NumericLiteral" | "JSXText" | "JSXIdentifier" | "AnyTypeAnnotation" | "ArgumentPlaceholder" | "ArrayExpression" | "ArrayPattern" | "ArrayTypeAnnotation" | "ArrowFunctionExpression" | "AssignmentExpression" | "AssignmentPattern" | "AwaitExpression" | "BigIntLiteral" | "BinaryExpression" | "BindExpression" | "BlockStatement" | "BooleanLiteral" | "BooleanLiteralTypeAnnotation" | "BooleanTypeAnnotation" | "BreakStatement" | "CallExpression" | "CatchClause" | "ClassAccessorProperty" | "ClassBody" | "ClassDeclaration" | "ClassExpression" | "ClassImplements" | "ClassMethod" | "ClassPrivateMethod" | "ClassPrivateProperty" | "ClassProperty" | "ConditionalExpression" | "ContinueStatement" | "DebuggerStatement" | "DecimalLiteral" | "DeclareClass" | "DeclareExportAllDeclaration" | "DeclareExportDeclaration" | "DeclareFunction" | "DeclareInterface" | "DeclareModule" | "DeclareModuleExports" | "DeclareOpaqueType" | "DeclareTypeAlias" | "DeclareVariable" | "DeclaredPredicate" | "Decorator" | "Directive" | "DirectiveLiteral" | "DoExpression" | "DoWhileStatement" | "EmptyStatement" | "EmptyTypeAnnotation" | "EnumBooleanBody" | "EnumBooleanMember" | "EnumDeclaration" | "EnumDefaultedMember" | "EnumNumberBody" | "EnumNumberMember" | "EnumStringBody" | "EnumStringMember" | "EnumSymbolBody" | "ExistsTypeAnnotation" | "ExportAllDeclaration" | "ExportDefaultDeclaration" | "ExportDefaultSpecifier" | "ExportNamedDeclaration" | "ExportNamespaceSpecifier" | "ExportSpecifier" | "ExpressionStatement" | "File" | "ForInStatement" | "ForOfStatement" | "ForStatement" | "FunctionDeclaration" | "FunctionExpression" | "FunctionTypeAnnotation" | "FunctionTypeParam" | "GenericTypeAnnotation" | "Identifier" | "IfStatement" | "Import" | "ImportAttribute" | "ImportDeclaration" | "ImportDefaultSpecifier" | "ImportExpression" | "ImportNamespaceSpecifier" | "ImportSpecifier" | "IndexedAccessType" | "InferredPredicate" | "InterfaceDeclaration" | "InterfaceExtends" | "InterfaceTypeAnnotation" | "InterpreterDirective" | "IntersectionTypeAnnotation" | "JSXAttribute" | "JSXClosingElement" | "JSXClosingFragment" | "JSXElement" | "JSXEmptyExpression" | "JSXExpressionContainer" | "JSXFragment" | "JSXMemberExpression" | "JSXNamespacedName" | "JSXOpeningElement" | "JSXOpeningFragment" | "JSXSpreadAttribute" | "JSXSpreadChild" | "LabeledStatement" | "LogicalExpression" | "MemberExpression" | "MetaProperty" | "MixedTypeAnnotation" | "ModuleExpression" | "NewExpression" | "Noop" | "NullLiteral" | "NullLiteralTypeAnnotation" | "NullableTypeAnnotation" | "NumberLiteral" | "NumberLiteralTypeAnnotation" | "NumberTypeAnnotation" | "ObjectExpression" | "ObjectMethod" | "ObjectPattern" | "ObjectProperty" | "ObjectTypeAnnotation" | "ObjectTypeCallProperty" | "ObjectTypeIndexer" | "ObjectTypeInternalSlot" | "ObjectTypeProperty" | "ObjectTypeSpreadProperty" | "OpaqueType" | "OptionalCallExpression" | "OptionalIndexedAccessType" | "OptionalMemberExpression" | "ParenthesizedExpression" | "PipelineBareFunction" | "PipelinePrimaryTopicReference" | "PipelineTopicExpression" | "Placeholder" | "PrivateName" | "Program" | "QualifiedTypeIdentifier" | "RecordExpression" | "RegExpLiteral" | "RegexLiteral" | "RestElement" | "RestProperty" | "ReturnStatement" | "SequenceExpression" | "SpreadElement" | "SpreadProperty" | "StaticBlock" | "StringLiteralTypeAnnotation" | "StringTypeAnnotation" | "Super" | "SwitchCase" | "SwitchStatement" | "SymbolTypeAnnotation" | "TSAnyKeyword" | "TSArrayType" | "TSAsExpression" | "TSBigIntKeyword" | "TSBooleanKeyword" | "TSCallSignatureDeclaration" | "TSConditionalType" | "TSConstructSignatureDeclaration" | "TSConstructorType" | "TSDeclareFunction" | "TSDeclareMethod" | "TSEnumDeclaration" | "TSEnumMember" | "TSExportAssignment" | "TSExpressionWithTypeArguments" | "TSExternalModuleReference" | "TSFunctionType" | "TSImportEqualsDeclaration" | "TSImportType" | "TSIndexSignature" | "TSIndexedAccessType" | "TSInferType" | "TSInstantiationExpression" | "TSInterfaceBody" | "TSInterfaceDeclaration" | "TSIntersectionType" | "TSIntrinsicKeyword" | "TSLiteralType" | "TSMappedType" | "TSMethodSignature" | "TSModuleBlock" | "TSModuleDeclaration" | "TSNamedTupleMember" | "TSNamespaceExportDeclaration" | "TSNeverKeyword" | "TSNonNullExpression" | "TSNullKeyword" | "TSNumberKeyword" | "TSObjectKeyword" | "TSOptionalType" | "TSParameterProperty" | "TSParenthesizedType" | "TSPropertySignature" | "TSQualifiedName" | "TSRestType" | "TSSatisfiesExpression" | "TSStringKeyword" | "TSSymbolKeyword" | "TSThisType" | "TSTupleType" | "TSTypeAliasDeclaration" | "TSTypeAnnotation" | "TSTypeAssertion" | "TSTypeLiteral" | "TSTypeOperator" | "TSTypeParameter" | "TSTypeParameterDeclaration" | "TSTypeParameterInstantiation" | "TSTypePredicate" | "TSTypeQuery" | "TSTypeReference" | "TSUndefinedKeyword" | "TSUnionType" | "TSUnknownKeyword" | "TSVoidKeyword" | "TaggedTemplateExpression" | "TemplateElement" | "TemplateLiteral" | "ThisExpression" | "ThisTypeAnnotation" | "ThrowStatement" | "TopicReference" | "TryStatement" | "TupleExpression" | "TupleTypeAnnotation" | "TypeAlias" | "TypeAnnotation" | "TypeCastExpression" | "TypeParameter" | "TypeParameterDeclaration" | "TypeParameterInstantiation" | "TypeofTypeAnnotation" | "UnaryExpression" | "UnionTypeAnnotation" | "UpdateExpression" | "V8IntrinsicIdentifier" | "VariableDeclaration" | "VariableDeclarator" | "Variance" | "VoidTypeAnnotation" | "WhileStatement" | "WithStatement" | "YieldExpression" | keyof Aliases)[];
declare const BINARY_TYPES: ("StringLiteral" | "NumericLiteral" | "JSXText" | "JSXIdentifier" | "AnyTypeAnnotation" | "ArgumentPlaceholder" | "ArrayExpression" | "ArrayPattern" | "ArrayTypeAnnotation" | "ArrowFunctionExpression" | "AssignmentExpression" | "AssignmentPattern" | "AwaitExpression" | "BigIntLiteral" | "BinaryExpression" | "BindExpression" | "BlockStatement" | "BooleanLiteral" | "BooleanLiteralTypeAnnotation" | "BooleanTypeAnnotation" | "BreakStatement" | "CallExpression" | "CatchClause" | "ClassAccessorProperty" | "ClassBody" | "ClassDeclaration" | "ClassExpression" | "ClassImplements" | "ClassMethod" | "ClassPrivateMethod" | "ClassPrivateProperty" | "ClassProperty" | "ConditionalExpression" | "ContinueStatement" | "DebuggerStatement" | "DecimalLiteral" | "DeclareClass" | "DeclareExportAllDeclaration" | "DeclareExportDeclaration" | "DeclareFunction" | "DeclareInterface" | "DeclareModule" | "DeclareModuleExports" | "DeclareOpaqueType" | "DeclareTypeAlias" | "DeclareVariable" | "DeclaredPredicate" | "Decorator" | "Directive" | "DirectiveLiteral" | "DoExpression" | "DoWhileStatement" | "EmptyStatement" | "EmptyTypeAnnotation" | "EnumBooleanBody" | "EnumBooleanMember" | "EnumDeclaration" | "EnumDefaultedMember" | "EnumNumberBody" | "EnumNumberMember" | "EnumStringBody" | "EnumStringMember" | "EnumSymbolBody" | "ExistsTypeAnnotation" | "ExportAllDeclaration" | "ExportDefaultDeclaration" | "ExportDefaultSpecifier" | "ExportNamedDeclaration" | "ExportNamespaceSpecifier" | "ExportSpecifier" | "ExpressionStatement" | "File" | "ForInStatement" | "ForOfStatement" | "ForStatement" | "FunctionDeclaration" | "FunctionExpression" | "FunctionTypeAnnotation" | "FunctionTypeParam" | "GenericTypeAnnotation" | "Identifier" | "IfStatement" | "Import" | "ImportAttribute" | "ImportDeclaration" | "ImportDefaultSpecifier" | "ImportExpression" | "ImportNamespaceSpecifier" | "ImportSpecifier" | "IndexedAccessType" | "InferredPredicate" | "InterfaceDeclaration" | "InterfaceExtends" | "InterfaceTypeAnnotation" | "InterpreterDirective" | "IntersectionTypeAnnotation" | "JSXAttribute" | "JSXClosingElement" | "JSXClosingFragment" | "JSXElement" | "JSXEmptyExpression" | "JSXExpressionContainer" | "JSXFragment" | "JSXMemberExpression" | "JSXNamespacedName" | "JSXOpeningElement" | "JSXOpeningFragment" | "JSXSpreadAttribute" | "JSXSpreadChild" | "LabeledStatement" | "LogicalExpression" | "MemberExpression" | "MetaProperty" | "MixedTypeAnnotation" | "ModuleExpression" | "NewExpression" | "Noop" | "NullLiteral" | "NullLiteralTypeAnnotation" | "NullableTypeAnnotation" | "NumberLiteral" | "NumberLiteralTypeAnnotation" | "NumberTypeAnnotation" | "ObjectExpression" | "ObjectMethod" | "ObjectPattern" | "ObjectProperty" | "ObjectTypeAnnotation" | "ObjectTypeCallProperty" | "ObjectTypeIndexer" | "ObjectTypeInternalSlot" | "ObjectTypeProperty" | "ObjectTypeSpreadProperty" | "OpaqueType" | "OptionalCallExpression" | "OptionalIndexedAccessType" | "OptionalMemberExpression" | "ParenthesizedExpression" | "PipelineBareFunction" | "PipelinePrimaryTopicReference" | "PipelineTopicExpression" | "Placeholder" | "PrivateName" | "Program" | "QualifiedTypeIdentifier" | "RecordExpression" | "RegExpLiteral" | "RegexLiteral" | "RestElement" | "RestProperty" | "ReturnStatement" | "SequenceExpression" | "SpreadElement" | "SpreadProperty" | "StaticBlock" | "StringLiteralTypeAnnotation" | "StringTypeAnnotation" | "Super" | "SwitchCase" | "SwitchStatement" | "SymbolTypeAnnotation" | "TSAnyKeyword" | "TSArrayType" | "TSAsExpression" | "TSBigIntKeyword" | "TSBooleanKeyword" | "TSCallSignatureDeclaration" | "TSConditionalType" | "TSConstructSignatureDeclaration" | "TSConstructorType" | "TSDeclareFunction" | "TSDeclareMethod" | "TSEnumDeclaration" | "TSEnumMember" | "TSExportAssignment" | "TSExpressionWithTypeArguments" | "TSExternalModuleReference" | "TSFunctionType" | "TSImportEqualsDeclaration" | "TSImportType" | "TSIndexSignature" | "TSIndexedAccessType" | "TSInferType" | "TSInstantiationExpression" | "TSInterfaceBody" | "TSInterfaceDeclaration" | "TSIntersectionType" | "TSIntrinsicKeyword" | "TSLiteralType" | "TSMappedType" | "TSMethodSignature" | "TSModuleBlock" | "TSModuleDeclaration" | "TSNamedTupleMember" | "TSNamespaceExportDeclaration" | "TSNeverKeyword" | "TSNonNullExpression" | "TSNullKeyword" | "TSNumberKeyword" | "TSObjectKeyword" | "TSOptionalType" | "TSParameterProperty" | "TSParenthesizedType" | "TSPropertySignature" | "TSQualifiedName" | "TSRestType" | "TSSatisfiesExpression" | "TSStringKeyword" | "TSSymbolKeyword" | "TSThisType" | "TSTupleType" | "TSTypeAliasDeclaration" | "TSTypeAnnotation" | "TSTypeAssertion" | "TSTypeLiteral" | "TSTypeOperator" | "TSTypeParameter" | "TSTypeParameterDeclaration" | "TSTypeParameterInstantiation" | "TSTypePredicate" | "TSTypeQuery" | "TSTypeReference" | "TSUndefinedKeyword" | "TSUnionType" | "TSUnknownKeyword" | "TSVoidKeyword" | "TaggedTemplateExpression" | "TemplateElement" | "TemplateLiteral" | "ThisExpression" | "ThisTypeAnnotation" | "ThrowStatement" | "TopicReference" | "TryStatement" | "TupleExpression" | "TupleTypeAnnotation" | "TypeAlias" | "TypeAnnotation" | "TypeCastExpression" | "TypeParameter" | "TypeParameterDeclaration" | "TypeParameterInstantiation" | "TypeofTypeAnnotation" | "UnaryExpression" | "UnionTypeAnnotation" | "UpdateExpression" | "V8IntrinsicIdentifier" | "VariableDeclaration" | "VariableDeclarator" | "Variance" | "VoidTypeAnnotation" | "WhileStatement" | "WithStatement" | "YieldExpression" | keyof Aliases)[];
declare const SCOPABLE_TYPES: ("StringLiteral" | "NumericLiteral" | "JSXText" | "JSXIdentifier" | "AnyTypeAnnotation" | "ArgumentPlaceholder" | "ArrayExpression" | "ArrayPattern" | "ArrayTypeAnnotation" | "ArrowFunctionExpression" | "AssignmentExpression" | "AssignmentPattern" | "AwaitExpression" | "BigIntLiteral" | "BinaryExpression" | "BindExpression" | "BlockStatement" | "BooleanLiteral" | "BooleanLiteralTypeAnnotation" | "BooleanTypeAnnotation" | "BreakStatement" | "CallExpression" | "CatchClause" | "ClassAccessorProperty" | "ClassBody" | "ClassDeclaration" | "ClassExpression" | "ClassImplements" | "ClassMethod" | "ClassPrivateMethod" | "ClassPrivateProperty" | "ClassProperty" | "ConditionalExpression" | "ContinueStatement" | "DebuggerStatement" | "DecimalLiteral" | "DeclareClass" | "DeclareExportAllDeclaration" | "DeclareExportDeclaration" | "DeclareFunction" | "DeclareInterface" | "DeclareModule" | "DeclareModuleExports" | "DeclareOpaqueType" | "DeclareTypeAlias" | "DeclareVariable" | "DeclaredPredicate" | "Decorator" | "Directive" | "DirectiveLiteral" | "DoExpression" | "DoWhileStatement" | "EmptyStatement" | "EmptyTypeAnnotation" | "EnumBooleanBody" | "EnumBooleanMember" | "EnumDeclaration" | "EnumDefaultedMember" | "EnumNumberBody" | "EnumNumberMember" | "EnumStringBody" | "EnumStringMember" | "EnumSymbolBody" | "ExistsTypeAnnotation" | "ExportAllDeclaration" | "ExportDefaultDeclaration" | "ExportDefaultSpecifier" | "ExportNamedDeclaration" | "ExportNamespaceSpecifier" | "ExportSpecifier" | "ExpressionStatement" | "File" | "ForInStatement" | "ForOfStatement" | "ForStatement" | "FunctionDeclaration" | "FunctionExpression" | "FunctionTypeAnnotation" | "FunctionTypeParam" | "GenericTypeAnnotation" | "Identifier" | "IfStatement" | "Import" | "ImportAttribute" | "ImportDeclaration" | "ImportDefaultSpecifier" | "ImportExpression" | "ImportNamespaceSpecifier" | "ImportSpecifier" | "IndexedAccessType" | "InferredPredicate" | "InterfaceDeclaration" | "InterfaceExtends" | "InterfaceTypeAnnotation" | "InterpreterDirective" | "IntersectionTypeAnnotation" | "JSXAttribute" | "JSXClosingElement" | "JSXClosingFragment" | "JSXElement" | "JSXEmptyExpression" | "JSXExpressionContainer" | "JSXFragment" | "JSXMemberExpression" | "JSXNamespacedName" | "JSXOpeningElement" | "JSXOpeningFragment" | "JSXSpreadAttribute" | "JSXSpreadChild" | "LabeledStatement" | "LogicalExpression" | "MemberExpression" | "MetaProperty" | "MixedTypeAnnotation" | "ModuleExpression" | "NewExpression" | "Noop" | "NullLiteral" | "NullLiteralTypeAnnotation" | "NullableTypeAnnotation" | "NumberLiteral" | "NumberLiteralTypeAnnotation" | "NumberTypeAnnotation" | "ObjectExpression" | "ObjectMethod" | "ObjectPattern" | "ObjectProperty" | "ObjectTypeAnnotation" | "ObjectTypeCallProperty" | "ObjectTypeIndexer" | "ObjectTypeInternalSlot" | "ObjectTypeProperty" | "ObjectTypeSpreadProperty" | "OpaqueType" | "OptionalCallExpression" | "OptionalIndexedAccessType" | "OptionalMemberExpression" | "ParenthesizedExpression" | "PipelineBareFunction" | "PipelinePrimaryTopicReference" | "PipelineTopicExpression" | "Placeholder" | "PrivateName" | "Program" | "QualifiedTypeIdentifier" | "RecordExpression" | "RegExpLiteral" | "RegexLiteral" | "RestElement" | "RestProperty" | "ReturnStatement" | "SequenceExpression" | "SpreadElement" | "SpreadProperty" | "StaticBlock" | "StringLiteralTypeAnnotation" | "StringTypeAnnotation" | "Super" | "SwitchCase" | "SwitchStatement" | "SymbolTypeAnnotation" | "TSAnyKeyword" | "TSArrayType" | "TSAsExpression" | "TSBigIntKeyword" | "TSBooleanKeyword" | "TSCallSignatureDeclaration" | "TSConditionalType" | "TSConstructSignatureDeclaration" | "TSConstructorType" | "TSDeclareFunction" | "TSDeclareMethod" | "TSEnumDeclaration" | "TSEnumMember" | "TSExportAssignment" | "TSExpressionWithTypeArguments" | "TSExternalModuleReference" | "TSFunctionType" | "TSImportEqualsDeclaration" | "TSImportType" | "TSIndexSignature" | "TSIndexedAccessType" | "TSInferType" | "TSInstantiationExpression" | "TSInterfaceBody" | "TSInterfaceDeclaration" | "TSIntersectionType" | "TSIntrinsicKeyword" | "TSLiteralType" | "TSMappedType" | "TSMethodSignature" | "TSModuleBlock" | "TSModuleDeclaration" | "TSNamedTupleMember" | "TSNamespaceExportDeclaration" | "TSNeverKeyword" | "TSNonNullExpression" | "TSNullKeyword" | "TSNumberKeyword" | "TSObjectKeyword" | "TSOptionalType" | "TSParameterProperty" | "TSParenthesizedType" | "TSPropertySignature" | "TSQualifiedName" | "TSRestType" | "TSSatisfiesExpression" | "TSStringKeyword" | "TSSymbolKeyword" | "TSThisType" | "TSTupleType" | "TSTypeAliasDeclaration" | "TSTypeAnnotation" | "TSTypeAssertion" | "TSTypeLiteral" | "TSTypeOperator" | "TSTypeParameter" | "TSTypeParameterDeclaration" | "TSTypeParameterInstantiation" | "TSTypePredicate" | "TSTypeQuery" | "TSTypeReference" | "TSUndefinedKeyword" | "TSUnionType" | "TSUnknownKeyword" | "TSVoidKeyword" | "TaggedTemplateExpression" | "TemplateElement" | "TemplateLiteral" | "ThisExpression" | "ThisTypeAnnotation" | "ThrowStatement" | "TopicReference" | "TryStatement" | "TupleExpression" | "TupleTypeAnnotation" | "TypeAlias" | "TypeAnnotation" | "TypeCastExpression" | "TypeParameter" | "TypeParameterDeclaration" | "TypeParameterInstantiation" | "TypeofTypeAnnotation" | "UnaryExpression" | "UnionTypeAnnotation" | "UpdateExpression" | "V8IntrinsicIdentifier" | "VariableDeclaration" | "VariableDeclarator" | "Variance" | "VoidTypeAnnotation" | "WhileStatement" | "WithStatement" | "YieldExpression" | keyof Aliases)[];
declare const BLOCKPARENT_TYPES: ("StringLiteral" | "NumericLiteral" | "JSXText" | "JSXIdentifier" | "AnyTypeAnnotation" | "ArgumentPlaceholder" | "ArrayExpression" | "ArrayPattern" | "ArrayTypeAnnotation" | "ArrowFunctionExpression" | "AssignmentExpression" | "AssignmentPattern" | "AwaitExpression" | "BigIntLiteral" | "BinaryExpression" | "BindExpression" | "BlockStatement" | "BooleanLiteral" | "BooleanLiteralTypeAnnotation" | "BooleanTypeAnnotation" | "BreakStatement" | "CallExpression" | "CatchClause" | "ClassAccessorProperty" | "ClassBody" | "ClassDeclaration" | "ClassExpression" | "ClassImplements" | "ClassMethod" | "ClassPrivateMethod" | "ClassPrivateProperty" | "ClassProperty" | "ConditionalExpression" | "ContinueStatement" | "DebuggerStatement" | "DecimalLiteral" | "DeclareClass" | "DeclareExportAllDeclaration" | "DeclareExportDeclaration" | "DeclareFunction" | "DeclareInterface" | "DeclareModule" | "DeclareModuleExports" | "DeclareOpaqueType" | "DeclareTypeAlias" | "DeclareVariable" | "DeclaredPredicate" | "Decorator" | "Directive" | "DirectiveLiteral" | "DoExpression" | "DoWhileStatement" | "EmptyStatement" | "EmptyTypeAnnotation" | "EnumBooleanBody" | "EnumBooleanMember" | "EnumDeclaration" | "EnumDefaultedMember" | "EnumNumberBody" | "EnumNumberMember" | "EnumStringBody" | "EnumStringMember" | "EnumSymbolBody" | "ExistsTypeAnnotation" | "ExportAllDeclaration" | "ExportDefaultDeclaration" | "ExportDefaultSpecifier" | "ExportNamedDeclaration" | "ExportNamespaceSpecifier" | "ExportSpecifier" | "ExpressionStatement" | "File" | "ForInStatement" | "ForOfStatement" | "ForStatement" | "FunctionDeclaration" | "FunctionExpression" | "FunctionTypeAnnotation" | "FunctionTypeParam" | "GenericTypeAnnotation" | "Identifier" | "IfStatement" | "Import" | "ImportAttribute" | "ImportDeclaration" | "ImportDefaultSpecifier" | "ImportExpression" | "ImportNamespaceSpecifier" | "ImportSpecifier" | "IndexedAccessType" | "InferredPredicate" | "InterfaceDeclaration" | "InterfaceExtends" | "InterfaceTypeAnnotation" | "InterpreterDirective" | "IntersectionTypeAnnotation" | "JSXAttribute" | "JSXClosingElement" | "JSXClosingFragment" | "JSXElement" | "JSXEmptyExpression" | "JSXExpressionContainer" | "JSXFragment" | "JSXMemberExpression" | "JSXNamespacedName" | "JSXOpeningElement" | "JSXOpeningFragment" | "JSXSpreadAttribute" | "JSXSpreadChild" | "LabeledStatement" | "LogicalExpression" | "MemberExpression" | "MetaProperty" | "MixedTypeAnnotation" | "ModuleExpression" | "NewExpression" | "Noop" | "NullLiteral" | "NullLiteralTypeAnnotation" | "NullableTypeAnnotation" | "NumberLiteral" | "NumberLiteralTypeAnnotation" | "NumberTypeAnnotation" | "ObjectExpression" | "ObjectMethod" | "ObjectPattern" | "ObjectProperty" | "ObjectTypeAnnotation" | "ObjectTypeCallProperty" | "ObjectTypeIndexer" | "ObjectTypeInternalSlot" | "ObjectTypeProperty" | "ObjectTypeSpreadProperty" | "OpaqueType" | "OptionalCallExpression" | "OptionalIndexedAccessType" | "OptionalMemberExpression" | "ParenthesizedExpression" | "PipelineBareFunction" | "PipelinePrimaryTopicReference" | "PipelineTopicExpression" | "Placeholder" | "PrivateName" | "Program" | "QualifiedTypeIdentifier" | "RecordExpression" | "RegExpLiteral" | "RegexLiteral" | "RestElement" | "RestProperty" | "ReturnStatement" | "SequenceExpression" | "SpreadElement" | "SpreadProperty" | "StaticBlock" | "StringLiteralTypeAnnotation" | "StringTypeAnnotation" | "Super" | "SwitchCase" | "SwitchStatement" | "SymbolTypeAnnotation" | "TSAnyKeyword" | "TSArrayType" | "TSAsExpression" | "TSBigIntKeyword" | "TSBooleanKeyword" | "TSCallSignatureDeclaration" | "TSConditionalType" | "TSConstructSignatureDeclaration" | "TSConstructorType" | "TSDeclareFunction" | "TSDeclareMethod" | "TSEnumDeclaration" | "TSEnumMember" | "TSExportAssignment" | "TSExpressionWithTypeArguments" | "TSExternalModuleReference" | "TSFunctionType" | "TSImportEqualsDeclaration" | "TSImportType" | "TSIndexSignature" | "TSIndexedAccessType" | "TSInferType" | "TSInstantiationExpression" | "TSInterfaceBody" | "TSInterfaceDeclaration" | "TSIntersectionType" | "TSIntrinsicKeyword" | "TSLiteralType" | "TSMappedType" | "TSMethodSignature" | "TSModuleBlock" | "TSModuleDeclaration" | "TSNamedTupleMember" | "TSNamespaceExportDeclaration" | "TSNeverKeyword" | "TSNonNullExpression" | "TSNullKeyword" | "TSNumberKeyword" | "TSObjectKeyword" | "TSOptionalType" | "TSParameterProperty" | "TSParenthesizedType" | "TSPropertySignature" | "TSQualifiedName" | "TSRestType" | "TSSatisfiesExpression" | "TSStringKeyword" | "TSSymbolKeyword" | "TSThisType" | "TSTupleType" | "TSTypeAliasDeclaration" | "TSTypeAnnotation" | "TSTypeAssertion" | "TSTypeLiteral" | "TSTypeOperator" | "TSTypeParameter" | "TSTypeParameterDeclaration" | "TSTypeParameterInstantiation" | "TSTypePredicate" | "TSTypeQuery" | "TSTypeReference" | "TSUndefinedKeyword" | "TSUnionType" | "TSUnknownKeyword" | "TSVoidKeyword" | "TaggedTemplateExpression" | "TemplateElement" | "TemplateLiteral" | "ThisExpression" | "ThisTypeAnnotation" | "ThrowStatement" | "TopicReference" | "TryStatement" | "TupleExpression" | "TupleTypeAnnotation" | "TypeAlias" | "TypeAnnotation" | "TypeCastExpression" | "TypeParameter" | "TypeParameterDeclaration" | "TypeParameterInstantiation" | "TypeofTypeAnnotation" | "UnaryExpression" | "UnionTypeAnnotation" | "UpdateExpression" | "V8IntrinsicIdentifier" | "VariableDeclaration" | "VariableDeclarator" | "Variance" | "VoidTypeAnnotation" | "WhileStatement" | "WithStatement" | "YieldExpression" | keyof Aliases)[];
declare const BLOCK_TYPES: ("StringLiteral" | "NumericLiteral" | "JSXText" | "JSXIdentifier" | "AnyTypeAnnotation" | "ArgumentPlaceholder" | "ArrayExpression" | "ArrayPattern" | "ArrayTypeAnnotation" | "ArrowFunctionExpression" | "AssignmentExpression" | "AssignmentPattern" | "AwaitExpression" | "BigIntLiteral" | "BinaryExpression" | "BindExpression" | "BlockStatement" | "BooleanLiteral" | "BooleanLiteralTypeAnnotation" | "BooleanTypeAnnotation" | "BreakStatement" | "CallExpression" | "CatchClause" | "ClassAccessorProperty" | "ClassBody" | "ClassDeclaration" | "ClassExpression" | "ClassImplements" | "ClassMethod" | "ClassPrivateMethod" | "ClassPrivateProperty" | "ClassProperty" | "ConditionalExpression" | "ContinueStatement" | "DebuggerStatement" | "DecimalLiteral" | "DeclareClass" | "DeclareExportAllDeclaration" | "DeclareExportDeclaration" | "DeclareFunction" | "DeclareInterface" | "DeclareModule" | "DeclareModuleExports" | "DeclareOpaqueType" | "DeclareTypeAlias" | "DeclareVariable" | "DeclaredPredicate" | "Decorator" | "Directive" | "DirectiveLiteral" | "DoExpression" | "DoWhileStatement" | "EmptyStatement" | "EmptyTypeAnnotation" | "EnumBooleanBody" | "EnumBooleanMember" | "EnumDeclaration" | "EnumDefaultedMember" | "EnumNumberBody" | "EnumNumberMember" | "EnumStringBody" | "EnumStringMember" | "EnumSymbolBody" | "ExistsTypeAnnotation" | "ExportAllDeclaration" | "ExportDefaultDeclaration" | "ExportDefaultSpecifier" | "ExportNamedDeclaration" | "ExportNamespaceSpecifier" | "ExportSpecifier" | "ExpressionStatement" | "File" | "ForInStatement" | "ForOfStatement" | "ForStatement" | "FunctionDeclaration" | "FunctionExpression" | "FunctionTypeAnnotation" | "FunctionTypeParam" | "GenericTypeAnnotation" | "Identifier" | "IfStatement" | "Import" | "ImportAttribute" | "ImportDeclaration" | "ImportDefaultSpecifier" | "ImportExpression" | "ImportNamespaceSpecifier" | "ImportSpecifier" | "IndexedAccessType" | "InferredPredicate" | "InterfaceDeclaration" | "InterfaceExtends" | "InterfaceTypeAnnotation" | "InterpreterDirective" | "IntersectionTypeAnnotation" | "JSXAttribute" | "JSXClosingElement" | "JSXClosingFragment" | "JSXElement" | "JSXEmptyExpression" | "JSXExpressionContainer" | "JSXFragment" | "JSXMemberExpression" | "JSXNamespacedName" | "JSXOpeningElement" | "JSXOpeningFragment" | "JSXSpreadAttribute" | "JSXSpreadChild" | "LabeledStatement" | "LogicalExpression" | "MemberExpression" | "MetaProperty" | "MixedTypeAnnotation" | "ModuleExpression" | "NewExpression" | "Noop" | "NullLiteral" | "NullLiteralTypeAnnotation" | "NullableTypeAnnotation" | "NumberLiteral" | "NumberLiteralTypeAnnotation" | "NumberTypeAnnotation" | "ObjectExpression" | "ObjectMethod" | "ObjectPattern" | "ObjectProperty" | "ObjectTypeAnnotation" | "ObjectTypeCallProperty" | "ObjectTypeIndexer" | "ObjectTypeInternalSlot" | "ObjectTypeProperty" | "ObjectTypeSpreadProperty" | "OpaqueType" | "OptionalCallExpression" | "OptionalIndexedAccessType" | "OptionalMemberExpression" | "ParenthesizedExpression" | "PipelineBareFunction" | "PipelinePrimaryTopicReference" | "PipelineTopicExpression" | "Placeholder" | "PrivateName" | "Program" | "QualifiedTypeIdentifier" | "RecordExpression" | "RegExpLiteral" | "RegexLiteral" | "RestElement" | "RestProperty" | "ReturnStatement" | "SequenceExpression" | "SpreadElement" | "SpreadProperty" | "StaticBlock" | "StringLiteralTypeAnnotation" | "StringTypeAnnotation" | "Super" | "SwitchCase" | "SwitchStatement" | "SymbolTypeAnnotation" | "TSAnyKeyword" | "TSArrayType" | "TSAsExpression" | "TSBigIntKeyword" | "TSBooleanKeyword" | "TSCallSignatureDeclaration" | "TSConditionalType" | "TSConstructSignatureDeclaration" | "TSConstructorType" | "TSDeclareFunction" | "TSDeclareMethod" | "TSEnumDeclaration" | "TSEnumMember" | "TSExportAssignment" | "TSExpressionWithTypeArguments" | "TSExternalModuleReference" | "TSFunctionType" | "TSImportEqualsDeclaration" | "TSImportType" | "TSIndexSignature" | "TSIndexedAccessType" | "TSInferType" | "TSInstantiationExpression" | "TSInterfaceBody" | "TSInterfaceDeclaration" | "TSIntersectionType" | "TSIntrinsicKeyword" | "TSLiteralType" | "TSMappedType" | "TSMethodSignature" | "TSModuleBlock" | "TSModuleDeclaration" | "TSNamedTupleMember" | "TSNamespaceExportDeclaration" | "TSNeverKeyword" | "TSNonNullExpression" | "TSNullKeyword" | "TSNumberKeyword" | "TSObjectKeyword" | "TSOptionalType" | "TSParameterProperty" | "TSParenthesizedType" | "TSPropertySignature" | "TSQualifiedName" | "TSRestType" | "TSSatisfiesExpression" | "TSStringKeyword" | "TSSymbolKeyword" | "TSThisType" | "TSTupleType" | "TSTypeAliasDeclaration" | "TSTypeAnnotation" | "TSTypeAssertion" | "TSTypeLiteral" | "TSTypeOperator" | "TSTypeParameter" | "TSTypeParameterDeclaration" | "TSTypeParameterInstantiation" | "TSTypePredicate" | "TSTypeQuery" | "TSTypeReference" | "TSUndefinedKeyword" | "TSUnionType" | "TSUnknownKeyword" | "TSVoidKeyword" | "TaggedTemplateExpression" | "TemplateElement" | "TemplateLiteral" | "ThisExpression" | "ThisTypeAnnotation" | "ThrowStatement" | "TopicReference" | "TryStatement" | "TupleExpression" | "TupleTypeAnnotation" | "TypeAlias" | "TypeAnnotation" | "TypeCastExpression" | "TypeParameter" | "TypeParameterDeclaration" | "TypeParameterInstantiation" | "TypeofTypeAnnotation" | "UnaryExpression" | "UnionTypeAnnotation" | "UpdateExpression" | "V8IntrinsicIdentifier" | "VariableDeclaration" | "VariableDeclarator" | "Variance" | "VoidTypeAnnotation" | "WhileStatement" | "WithStatement" | "YieldExpression" | keyof Aliases)[];
declare const STATEMENT_TYPES: ("StringLiteral" | "NumericLiteral" | "JSXText" | "JSXIdentifier" | "AnyTypeAnnotation" | "ArgumentPlaceholder" | "ArrayExpression" | "ArrayPattern" | "ArrayTypeAnnotation" | "ArrowFunctionExpression" | "AssignmentExpression" | "AssignmentPattern" | "AwaitExpression" | "BigIntLiteral" | "BinaryExpression" | "BindExpression" | "BlockStatement" | "BooleanLiteral" | "BooleanLiteralTypeAnnotation" | "BooleanTypeAnnotation" | "BreakStatement" | "CallExpression" | "CatchClause" | "ClassAccessorProperty" | "ClassBody" | "ClassDeclaration" | "ClassExpression" | "ClassImplements" | "ClassMethod" | "ClassPrivateMethod" | "ClassPrivateProperty" | "ClassProperty" | "ConditionalExpression" | "ContinueStatement" | "DebuggerStatement" | "DecimalLiteral" | "DeclareClass" | "DeclareExportAllDeclaration" | "DeclareExportDeclaration" | "DeclareFunction" | "DeclareInterface" | "DeclareModule" | "DeclareModuleExports" | "DeclareOpaqueType" | "DeclareTypeAlias" | "DeclareVariable" | "DeclaredPredicate" | "Decorator" | "Directive" | "DirectiveLiteral" | "DoExpression" | "DoWhileStatement" | "EmptyStatement" | "EmptyTypeAnnotation" | "EnumBooleanBody" | "EnumBooleanMember" | "EnumDeclaration" | "EnumDefaultedMember" | "EnumNumberBody" | "EnumNumberMember" | "EnumStringBody" | "EnumStringMember" | "EnumSymbolBody" | "ExistsTypeAnnotation" | "ExportAllDeclaration" | "ExportDefaultDeclaration" | "ExportDefaultSpecifier" | "ExportNamedDeclaration" | "ExportNamespaceSpecifier" | "ExportSpecifier" | "ExpressionStatement" | "File" | "ForInStatement" | "ForOfStatement" | "ForStatement" | "FunctionDeclaration" | "FunctionExpression" | "FunctionTypeAnnotation" | "FunctionTypeParam" | "GenericTypeAnnotation" | "Identifier" | "IfStatement" | "Import" | "ImportAttribute" | "ImportDeclaration" | "ImportDefaultSpecifier" | "ImportExpression" | "ImportNamespaceSpecifier" | "ImportSpecifier" | "IndexedAccessType" | "InferredPredicate" | "InterfaceDeclaration" | "InterfaceExtends" | "InterfaceTypeAnnotation" | "InterpreterDirective" | "IntersectionTypeAnnotation" | "JSXAttribute" | "JSXClosingElement" | "JSXClosingFragment" | "JSXElement" | "JSXEmptyExpression" | "JSXExpressionContainer" | "JSXFragment" | "JSXMemberExpression" | "JSXNamespacedName" | "JSXOpeningElement" | "JSXOpeningFragment" | "JSXSpreadAttribute" | "JSXSpreadChild" | "LabeledStatement" | "LogicalExpression" | "MemberExpression" | "MetaProperty" | "MixedTypeAnnotation" | "ModuleExpression" | "NewExpression" | "Noop" | "NullLiteral" | "NullLiteralTypeAnnotation" | "NullableTypeAnnotation" | "NumberLiteral" | "NumberLiteralTypeAnnotation" | "NumberTypeAnnotation" | "ObjectExpression" | "ObjectMethod" | "ObjectPattern" | "ObjectProperty" | "ObjectTypeAnnotation" | "ObjectTypeCallProperty" | "ObjectTypeIndexer" | "ObjectTypeInternalSlot" | "ObjectTypeProperty" | "ObjectTypeSpreadProperty" | "OpaqueType" | "OptionalCallExpression" | "OptionalIndexedAccessType" | "OptionalMemberExpression" | "ParenthesizedExpression" | "PipelineBareFunction" | "PipelinePrimaryTopicReference" | "PipelineTopicExpression" | "Placeholder" | "PrivateName" | "Program" | "QualifiedTypeIdentifier" | "RecordExpression" | "RegExpLiteral" | "RegexLiteral" | "RestElement" | "RestProperty" | "ReturnStatement" | "SequenceExpression" | "SpreadElement" | "SpreadProperty" | "StaticBlock" | "StringLiteralTypeAnnotation" | "StringTypeAnnotation" | "Super" | "SwitchCase" | "SwitchStatement" | "SymbolTypeAnnotation" | "TSAnyKeyword" | "TSArrayType" | "TSAsExpression" | "TSBigIntKeyword" | "TSBooleanKeyword" | "TSCallSignatureDeclaration" | "TSConditionalType" | "TSConstructSignatureDeclaration" | "TSConstructorType" | "TSDeclareFunction" | "TSDeclareMethod" | "TSEnumDeclaration" | "TSEnumMember" | "TSExportAssignment" | "TSExpressionWithTypeArguments" | "TSExternalModuleReference" | "TSFunctionType" | "TSImportEqualsDeclaration" | "TSImportType" | "TSIndexSignature" | "TSIndexedAccessType" | "TSInferType" | "TSInstantiationExpression" | "TSInterfaceBody" | "TSInterfaceDeclaration" | "TSIntersectionType" | "TSIntrinsicKeyword" | "TSLiteralType" | "TSMappedType" | "TSMethodSignature" | "TSModuleBlock" | "TSModuleDeclaration" | "TSNamedTupleMember" | "TSNamespaceExportDeclaration" | "TSNeverKeyword" | "TSNonNullExpression" | "TSNullKeyword" | "TSNumberKeyword" | "TSObjectKeyword" | "TSOptionalType" | "TSParameterProperty" | "TSParenthesizedType" | "TSPropertySignature" | "TSQualifiedName" | "TSRestType" | "TSSatisfiesExpression" | "TSStringKeyword" | "TSSymbolKeyword" | "TSThisType" | "TSTupleType" | "TSTypeAliasDeclaration" | "TSTypeAnnotation" | "TSTypeAssertion" | "TSTypeLiteral" | "TSTypeOperator" | "TSTypeParameter" | "TSTypeParameterDeclaration" | "TSTypeParameterInstantiation" | "TSTypePredicate" | "TSTypeQuery" | "TSTypeReference" | "TSUndefinedKeyword" | "TSUnionType" | "TSUnknownKeyword" | "TSVoidKeyword" | "TaggedTemplateExpression" | "TemplateElement" | "TemplateLiteral" | "ThisExpression" | "ThisTypeAnnotation" | "ThrowStatement" | "TopicReference" | "TryStatement" | "TupleExpression" | "TupleTypeAnnotation" | "TypeAlias" | "TypeAnnotation" | "TypeCastExpression" | "TypeParameter" | "TypeParameterDeclaration" | "TypeParameterInstantiation" | "TypeofTypeAnnotation" | "UnaryExpression" | "UnionTypeAnnotation" | "UpdateExpression" | "V8IntrinsicIdentifier" | "VariableDeclaration" | "VariableDeclarator" | "Variance" | "VoidTypeAnnotation" | "WhileStatement" | "WithStatement" | "YieldExpression" | keyof Aliases)[];
declare const TERMINATORLESS_TYPES: ("StringLiteral" | "NumericLiteral" | "JSXText" | "JSXIdentifier" | "AnyTypeAnnotation" | "ArgumentPlaceholder" | "ArrayExpression" | "ArrayPattern" | "ArrayTypeAnnotation" | "ArrowFunctionExpression" | "AssignmentExpression" | "AssignmentPattern" | "AwaitExpression" | "BigIntLiteral" | "BinaryExpression" | "BindExpression" | "BlockStatement" | "BooleanLiteral" | "BooleanLiteralTypeAnnotation" | "BooleanTypeAnnotation" | "BreakStatement" | "CallExpression" | "CatchClause" | "ClassAccessorProperty" | "ClassBody" | "ClassDeclaration" | "ClassExpression" | "ClassImplements" | "ClassMethod" | "ClassPrivateMethod" | "ClassPrivateProperty" | "ClassProperty" | "ConditionalExpression" | "ContinueStatement" | "DebuggerStatement" | "DecimalLiteral" | "DeclareClass" | "DeclareExportAllDeclaration" | "DeclareExportDeclaration" | "DeclareFunction" | "DeclareInterface" | "DeclareModule" | "DeclareModuleExports" | "DeclareOpaqueType" | "DeclareTypeAlias" | "DeclareVariable" | "DeclaredPredicate" | "Decorator" | "Directive" | "DirectiveLiteral" | "DoExpression" | "DoWhileStatement" | "EmptyStatement" | "EmptyTypeAnnotation" | "EnumBooleanBody" | "EnumBooleanMember" | "EnumDeclaration" | "EnumDefaultedMember" | "EnumNumberBody" | "EnumNumberMember" | "EnumStringBody" | "EnumStringMember" | "EnumSymbolBody" | "ExistsTypeAnnotation" | "ExportAllDeclaration" | "ExportDefaultDeclaration" | "ExportDefaultSpecifier" | "ExportNamedDeclaration" | "ExportNamespaceSpecifier" | "ExportSpecifier" | "ExpressionStatement" | "File" | "ForInStatement" | "ForOfStatement" | "ForStatement" | "FunctionDeclaration" | "FunctionExpression" | "FunctionTypeAnnotation" | "FunctionTypeParam" | "GenericTypeAnnotation" | "Identifier" | "IfStatement" | "Import" | "ImportAttribute" | "ImportDeclaration" | "ImportDefaultSpecifier" | "ImportExpression" | "ImportNamespaceSpecifier" | "ImportSpecifier" | "IndexedAccessType" | "InferredPredicate" | "InterfaceDeclaration" | "InterfaceExtends" | "InterfaceTypeAnnotation" | "InterpreterDirective" | "IntersectionTypeAnnotation" | "JSXAttribute" | "JSXClosingElement" | "JSXClosingFragment" | "JSXElement" | "JSXEmptyExpression" | "JSXExpressionContainer" | "JSXFragment" | "JSXMemberExpression" | "JSXNamespacedName" | "JSXOpeningElement" | "JSXOpeningFragment" | "JSXSpreadAttribute" | "JSXSpreadChild" | "LabeledStatement" | "LogicalExpression" | "MemberExpression" | "MetaProperty" | "MixedTypeAnnotation" | "ModuleExpression" | "NewExpression" | "Noop" | "NullLiteral" | "NullLiteralTypeAnnotation" | "NullableTypeAnnotation" | "NumberLiteral" | "NumberLiteralTypeAnnotation" | "NumberTypeAnnotation" | "ObjectExpression" | "ObjectMethod" | "ObjectPattern" | "ObjectProperty" | "ObjectTypeAnnotation" | "ObjectTypeCallProperty" | "ObjectTypeIndexer" | "ObjectTypeInternalSlot" | "ObjectTypeProperty" | "ObjectTypeSpreadProperty" | "OpaqueType" | "OptionalCallExpression" | "OptionalIndexedAccessType" | "OptionalMemberExpression" | "ParenthesizedExpression" | "PipelineBareFunction" | "PipelinePrimaryTopicReference" | "PipelineTopicExpression" | "Placeholder" | "PrivateName" | "Program" | "QualifiedTypeIdentifier" | "RecordExpression" | "RegExpLiteral" | "RegexLiteral" | "RestElement" | "RestProperty" | "ReturnStatement" | "SequenceExpression" | "SpreadElement" | "SpreadProperty" | "StaticBlock" | "StringLiteralTypeAnnotation" | "StringTypeAnnotation" | "Super" | "SwitchCase" | "SwitchStatement" | "SymbolTypeAnnotation" | "TSAnyKeyword" | "TSArrayType" | "TSAsExpression" | "TSBigIntKeyword" | "TSBooleanKeyword" | "TSCallSignatureDeclaration" | "TSConditionalType" | "TSConstructSignatureDeclaration" | "TSConstructorType" | "TSDeclareFunction" | "TSDeclareMethod" | "TSEnumDeclaration" | "TSEnumMember" | "TSExportAssignment" | "TSExpressionWithTypeArguments" | "TSExternalModuleReference" | "TSFunctionType" | "TSImportEqualsDeclaration" | "TSImportType" | "TSIndexSignature" | "TSIndexedAccessType" | "TSInferType" | "TSInstantiationExpression" | "TSInterfaceBody" | "TSInterfaceDeclaration" | "TSIntersectionType" | "TSIntrinsicKeyword" | "TSLiteralType" | "TSMappedType" | "TSMethodSignature" | "TSModuleBlock" | "TSModuleDeclaration" | "TSNamedTupleMember" | "TSNamespaceExportDeclaration" | "TSNeverKeyword" | "TSNonNullExpression" | "TSNullKeyword" | "TSNumberKeyword" | "TSObjectKeyword" | "TSOptionalType" | "TSParameterProperty" | "TSParenthesizedType" | "TSPropertySignature" | "TSQualifiedName" | "TSRestType" | "TSSatisfiesExpression" | "TSStringKeyword" | "TSSymbolKeyword" | "TSThisType" | "TSTupleType" | "TSTypeAliasDeclaration" | "TSTypeAnnotation" | "TSTypeAssertion" | "TSTypeLiteral" | "TSTypeOperator" | "TSTypeParameter" | "TSTypeParameterDeclaration" | "TSTypeParameterInstantiation" | "TSTypePredicate" | "TSTypeQuery" | "TSTypeReference" | "TSUndefinedKeyword" | "TSUnionType" | "TSUnknownKeyword" | "TSVoidKeyword" | "TaggedTemplateExpression" | "TemplateElement" | "TemplateLiteral" | "ThisExpression" | "ThisTypeAnnotation" | "ThrowStatement" | "TopicReference" | "TryStatement" | "TupleExpression" | "TupleTypeAnnotation" | "TypeAlias" | "TypeAnnotation" | "TypeCastExpression" | "TypeParameter" | "TypeParameterDeclaration" | "TypeParameterInstantiation" | "TypeofTypeAnnotation" | "UnaryExpression" | "UnionTypeAnnotation" | "UpdateExpression" | "V8IntrinsicIdentifier" | "VariableDeclaration" | "VariableDeclarator" | "Variance" | "VoidTypeAnnotation" | "WhileStatement" | "WithStatement" | "YieldExpression" | keyof Aliases)[];
declare const COMPLETIONSTATEMENT_TYPES: ("StringLiteral" | "NumericLiteral" | "JSXText" | "JSXIdentifier" | "AnyTypeAnnotation" | "ArgumentPlaceholder" | "ArrayExpression" | "ArrayPattern" | "ArrayTypeAnnotation" | "ArrowFunctionExpression" | "AssignmentExpression" | "AssignmentPattern" | "AwaitExpression" | "BigIntLiteral" | "BinaryExpression" | "BindExpression" | "BlockStatement" | "BooleanLiteral" | "BooleanLiteralTypeAnnotation" | "BooleanTypeAnnotation" | "BreakStatement" | "CallExpression" | "CatchClause" | "ClassAccessorProperty" | "ClassBody" | "ClassDeclaration" | "ClassExpression" | "ClassImplements" | "ClassMethod" | "ClassPrivateMethod" | "ClassPrivateProperty" | "ClassProperty" | "ConditionalExpression" | "ContinueStatement" | "DebuggerStatement" | "DecimalLiteral" | "DeclareClass" | "DeclareExportAllDeclaration" | "DeclareExportDeclaration" | "DeclareFunction" | "DeclareInterface" | "DeclareModule" | "DeclareModuleExports" | "DeclareOpaqueType" | "DeclareTypeAlias" | "DeclareVariable" | "DeclaredPredicate" | "Decorator" | "Directive" | "DirectiveLiteral" | "DoExpression" | "DoWhileStatement" | "EmptyStatement" | "EmptyTypeAnnotation" | "EnumBooleanBody" | "EnumBooleanMember" | "EnumDeclaration" | "EnumDefaultedMember" | "EnumNumberBody" | "EnumNumberMember" | "EnumStringBody" | "EnumStringMember" | "EnumSymbolBody" | "ExistsTypeAnnotation" | "ExportAllDeclaration" | "ExportDefaultDeclaration" | "ExportDefaultSpecifier" | "ExportNamedDeclaration" | "ExportNamespaceSpecifier" | "ExportSpecifier" | "ExpressionStatement" | "File" | "ForInStatement" | "ForOfStatement" | "ForStatement" | "FunctionDeclaration" | "FunctionExpression" | "FunctionTypeAnnotation" | "FunctionTypeParam" | "GenericTypeAnnotation" | "Identifier" | "IfStatement" | "Import" | "ImportAttribute" | "ImportDeclaration" | "ImportDefaultSpecifier" | "ImportExpression" | "ImportNamespaceSpecifier" | "ImportSpecifier" | "IndexedAccessType" | "InferredPredicate" | "InterfaceDeclaration" | "InterfaceExtends" | "InterfaceTypeAnnotation" | "InterpreterDirective" | "IntersectionTypeAnnotation" | "JSXAttribute" | "JSXClosingElement" | "JSXClosingFragment" | "JSXElement" | "JSXEmptyExpression" | "JSXExpressionContainer" | "JSXFragment" | "JSXMemberExpression" | "JSXNamespacedName" | "JSXOpeningElement" | "JSXOpeningFragment" | "JSXSpreadAttribute" | "JSXSpreadChild" | "LabeledStatement" | "LogicalExpression" | "MemberExpression" | "MetaProperty" | "MixedTypeAnnotation" | "ModuleExpression" | "NewExpression" | "Noop" | "NullLiteral" | "NullLiteralTypeAnnotation" | "NullableTypeAnnotation" | "NumberLiteral" | "NumberLiteralTypeAnnotation" | "NumberTypeAnnotation" | "ObjectExpression" | "ObjectMethod" | "ObjectPattern" | "ObjectProperty" | "ObjectTypeAnnotation" | "ObjectTypeCallProperty" | "ObjectTypeIndexer" | "ObjectTypeInternalSlot" | "ObjectTypeProperty" | "ObjectTypeSpreadProperty" | "OpaqueType" | "OptionalCallExpression" | "OptionalIndexedAccessType" | "OptionalMemberExpression" | "ParenthesizedExpression" | "PipelineBareFunction" | "PipelinePrimaryTopicReference" | "PipelineTopicExpression" | "Placeholder" | "PrivateName" | "Program" | "QualifiedTypeIdentifier" | "RecordExpression" | "RegExpLiteral" | "RegexLiteral" | "RestElement" | "RestProperty" | "ReturnStatement" | "SequenceExpression" | "SpreadElement" | "SpreadProperty" | "StaticBlock" | "StringLiteralTypeAnnotation" | "StringTypeAnnotation" | "Super" | "SwitchCase" | "SwitchStatement" | "SymbolTypeAnnotation" | "TSAnyKeyword" | "TSArrayType" | "TSAsExpression" | "TSBigIntKeyword" | "TSBooleanKeyword" | "TSCallSignatureDeclaration" | "TSConditionalType" | "TSConstructSignatureDeclaration" | "TSConstructorType" | "TSDeclareFunction" | "TSDeclareMethod" | "TSEnumDeclaration" | "TSEnumMember" | "TSExportAssignment" | "TSExpressionWithTypeArguments" | "TSExternalModuleReference" | "TSFunctionType" | "TSImportEqualsDeclaration" | "TSImportType" | "TSIndexSignature" | "TSIndexedAccessType" | "TSInferType" | "TSInstantiationExpression" | "TSInterfaceBody" | "TSInterfaceDeclaration" | "TSIntersectionType" | "TSIntrinsicKeyword" | "TSLiteralType" | "TSMappedType" | "TSMethodSignature" | "TSModuleBlock" | "TSModuleDeclaration" | "TSNamedTupleMember" | "TSNamespaceExportDeclaration" | "TSNeverKeyword" | "TSNonNullExpression" | "TSNullKeyword" | "TSNumberKeyword" | "TSObjectKeyword" | "TSOptionalType" | "TSParameterProperty" | "TSParenthesizedType" | "TSPropertySignature" | "TSQualifiedName" | "TSRestType" | "TSSatisfiesExpression" | "TSStringKeyword" | "TSSymbolKeyword" | "TSThisType" | "TSTupleType" | "TSTypeAliasDeclaration" | "TSTypeAnnotation" | "TSTypeAssertion" | "TSTypeLiteral" | "TSTypeOperator" | "TSTypeParameter" | "TSTypeParameterDeclaration" | "TSTypeParameterInstantiation" | "TSTypePredicate" | "TSTypeQuery" | "TSTypeReference" | "TSUndefinedKeyword" | "TSUnionType" | "TSUnknownKeyword" | "TSVoidKeyword" | "TaggedTemplateExpression" | "TemplateElement" | "TemplateLiteral" | "ThisExpression" | "ThisTypeAnnotation" | "ThrowStatement" | "TopicReference" | "TryStatement" | "TupleExpression" | "TupleTypeAnnotation" | "TypeAlias" | "TypeAnnotation" | "TypeCastExpression" | "TypeParameter" | "TypeParameterDeclaration" | "TypeParameterInstantiation" | "TypeofTypeAnnotation" | "UnaryExpression" | "UnionTypeAnnotation" | "UpdateExpression" | "V8IntrinsicIdentifier" | "VariableDeclaration" | "VariableDeclarator" | "Variance" | "VoidTypeAnnotation" | "WhileStatement" | "WithStatement" | "YieldExpression" | keyof Aliases)[];
declare const CONDITIONAL_TYPES: ("StringLiteral" | "NumericLiteral" | "JSXText" | "JSXIdentifier" | "AnyTypeAnnotation" | "ArgumentPlaceholder" | "ArrayExpression" | "ArrayPattern" | "ArrayTypeAnnotation" | "ArrowFunctionExpression" | "AssignmentExpression" | "AssignmentPattern" | "AwaitExpression" | "BigIntLiteral" | "BinaryExpression" | "BindExpression" | "BlockStatement" | "BooleanLiteral" | "BooleanLiteralTypeAnnotation" | "BooleanTypeAnnotation" | "BreakStatement" | "CallExpression" | "CatchClause" | "ClassAccessorProperty" | "ClassBody" | "ClassDeclaration" | "ClassExpression" | "ClassImplements" | "ClassMethod" | "ClassPrivateMethod" | "ClassPrivateProperty" | "ClassProperty" | "ConditionalExpression" | "ContinueStatement" | "DebuggerStatement" | "DecimalLiteral" | "DeclareClass" | "DeclareExportAllDeclaration" | "DeclareExportDeclaration" | "DeclareFunction" | "DeclareInterface" | "DeclareModule" | "DeclareModuleExports" | "DeclareOpaqueType" | "DeclareTypeAlias" | "DeclareVariable" | "DeclaredPredicate" | "Decorator" | "Directive" | "DirectiveLiteral" | "DoExpression" | "DoWhileStatement" | "EmptyStatement" | "EmptyTypeAnnotation" | "EnumBooleanBody" | "EnumBooleanMember" | "EnumDeclaration" | "EnumDefaultedMember" | "EnumNumberBody" | "EnumNumberMember" | "EnumStringBody" | "EnumStringMember" | "EnumSymbolBody" | "ExistsTypeAnnotation" | "ExportAllDeclaration" | "ExportDefaultDeclaration" | "ExportDefaultSpecifier" | "ExportNamedDeclaration" | "ExportNamespaceSpecifier" | "ExportSpecifier" | "ExpressionStatement" | "File" | "ForInStatement" | "ForOfStatement" | "ForStatement" | "FunctionDeclaration" | "FunctionExpression" | "FunctionTypeAnnotation" | "FunctionTypeParam" | "GenericTypeAnnotation" | "Identifier" | "IfStatement" | "Import" | "ImportAttribute" | "ImportDeclaration" | "ImportDefaultSpecifier" | "ImportExpression" | "ImportNamespaceSpecifier" | "ImportSpecifier" | "IndexedAccessType" | "InferredPredicate" | "InterfaceDeclaration" | "InterfaceExtends" | "InterfaceTypeAnnotation" | "InterpreterDirective" | "IntersectionTypeAnnotation" | "JSXAttribute" | "JSXClosingElement" | "JSXClosingFragment" | "JSXElement" | "JSXEmptyExpression" | "JSXExpressionContainer" | "JSXFragment" | "JSXMemberExpression" | "JSXNamespacedName" | "JSXOpeningElement" | "JSXOpeningFragment" | "JSXSpreadAttribute" | "JSXSpreadChild" | "LabeledStatement" | "LogicalExpression" | "MemberExpression" | "MetaProperty" | "MixedTypeAnnotation" | "ModuleExpression" | "NewExpression" | "Noop" | "NullLiteral" | "NullLiteralTypeAnnotation" | "NullableTypeAnnotation" | "NumberLiteral" | "NumberLiteralTypeAnnotation" | "NumberTypeAnnotation" | "ObjectExpression" | "ObjectMethod" | "ObjectPattern" | "ObjectProperty" | "ObjectTypeAnnotation" | "ObjectTypeCallProperty" | "ObjectTypeIndexer" | "ObjectTypeInternalSlot" | "ObjectTypeProperty" | "ObjectTypeSpreadProperty" | "OpaqueType" | "OptionalCallExpression" | "OptionalIndexedAccessType" | "OptionalMemberExpression" | "ParenthesizedExpression" | "PipelineBareFunction" | "PipelinePrimaryTopicReference" | "PipelineTopicExpression" | "Placeholder" | "PrivateName" | "Program" | "QualifiedTypeIdentifier" | "RecordExpression" | "RegExpLiteral" | "RegexLiteral" | "RestElement" | "RestProperty" | "ReturnStatement" | "SequenceExpression" | "SpreadElement" | "SpreadProperty" | "StaticBlock" | "StringLiteralTypeAnnotation" | "StringTypeAnnotation" | "Super" | "SwitchCase" | "SwitchStatement" | "SymbolTypeAnnotation" | "TSAnyKeyword" | "TSArrayType" | "TSAsExpression" | "TSBigIntKeyword" | "TSBooleanKeyword" | "TSCallSignatureDeclaration" | "TSConditionalType" | "TSConstructSignatureDeclaration" | "TSConstructorType" | "TSDeclareFunction" | "TSDeclareMethod" | "TSEnumDeclaration" | "TSEnumMember" | "TSExportAssignment" | "TSExpressionWithTypeArguments" | "TSExternalModuleReference" | "TSFunctionType" | "TSImportEqualsDeclaration" | "TSImportType" | "TSIndexSignature" | "TSIndexedAccessType" | "TSInferType" | "TSInstantiationExpression" | "TSInterfaceBody" | "TSInterfaceDeclaration" | "TSIntersectionType" | "TSIntrinsicKeyword" | "TSLiteralType" | "TSMappedType" | "TSMethodSignature" | "TSModuleBlock" | "TSModuleDeclaration" | "TSNamedTupleMember" | "TSNamespaceExportDeclaration" | "TSNeverKeyword" | "TSNonNullExpression" | "TSNullKeyword" | "TSNumberKeyword" | "TSObjectKeyword" | "TSOptionalType" | "TSParameterProperty" | "TSParenthesizedType" | "TSPropertySignature" | "TSQualifiedName" | "TSRestType" | "TSSatisfiesExpression" | "TSStringKeyword" | "TSSymbolKeyword" | "TSThisType" | "TSTupleType" | "TSTypeAliasDeclaration" | "TSTypeAnnotation" | "TSTypeAssertion" | "TSTypeLiteral" | "TSTypeOperator" | "TSTypeParameter" | "TSTypeParameterDeclaration" | "TSTypeParameterInstantiation" | "TSTypePredicate" | "TSTypeQuery" | "TSTypeReference" | "TSUndefinedKeyword" | "TSUnionType" | "TSUnknownKeyword" | "TSVoidKeyword" | "TaggedTemplateExpression" | "TemplateElement" | "TemplateLiteral" | "ThisExpression" | "ThisTypeAnnotation" | "ThrowStatement" | "TopicReference" | "TryStatement" | "TupleExpression" | "TupleTypeAnnotation" | "TypeAlias" | "TypeAnnotation" | "TypeCastExpression" | "TypeParameter" | "TypeParameterDeclaration" | "TypeParameterInstantiation" | "TypeofTypeAnnotation" | "UnaryExpression" | "UnionTypeAnnotation" | "UpdateExpression" | "V8IntrinsicIdentifier" | "VariableDeclaration" | "VariableDeclarator" | "Variance" | "VoidTypeAnnotation" | "WhileStatement" | "WithStatement" | "YieldExpression" | keyof Aliases)[];
declare const LOOP_TYPES: ("StringLiteral" | "NumericLiteral" | "JSXText" | "JSXIdentifier" | "AnyTypeAnnotation" | "ArgumentPlaceholder" | "ArrayExpression" | "ArrayPattern" | "ArrayTypeAnnotation" | "ArrowFunctionExpression" | "AssignmentExpression" | "AssignmentPattern" | "AwaitExpression" | "BigIntLiteral" | "BinaryExpression" | "BindExpression" | "BlockStatement" | "BooleanLiteral" | "BooleanLiteralTypeAnnotation" | "BooleanTypeAnnotation" | "BreakStatement" | "CallExpression" | "CatchClause" | "ClassAccessorProperty" | "ClassBody" | "ClassDeclaration" | "ClassExpression" | "ClassImplements" | "ClassMethod" | "ClassPrivateMethod" | "ClassPrivateProperty" | "ClassProperty" | "ConditionalExpression" | "ContinueStatement" | "DebuggerStatement" | "DecimalLiteral" | "DeclareClass" | "DeclareExportAllDeclaration" | "DeclareExportDeclaration" | "DeclareFunction" | "DeclareInterface" | "DeclareModule" | "DeclareModuleExports" | "DeclareOpaqueType" | "DeclareTypeAlias" | "DeclareVariable" | "DeclaredPredicate" | "Decorator" | "Directive" | "DirectiveLiteral" | "DoExpression" | "DoWhileStatement" | "EmptyStatement" | "EmptyTypeAnnotation" | "EnumBooleanBody" | "EnumBooleanMember" | "EnumDeclaration" | "EnumDefaultedMember" | "EnumNumberBody" | "EnumNumberMember" | "EnumStringBody" | "EnumStringMember" | "EnumSymbolBody" | "ExistsTypeAnnotation" | "ExportAllDeclaration" | "ExportDefaultDeclaration" | "ExportDefaultSpecifier" | "ExportNamedDeclaration" | "ExportNamespaceSpecifier" | "ExportSpecifier" | "ExpressionStatement" | "File" | "ForInStatement" | "ForOfStatement" | "ForStatement" | "FunctionDeclaration" | "FunctionExpression" | "FunctionTypeAnnotation" | "FunctionTypeParam" | "GenericTypeAnnotation" | "Identifier" | "IfStatement" | "Import" | "ImportAttribute" | "ImportDeclaration" | "ImportDefaultSpecifier" | "ImportExpression" | "ImportNamespaceSpecifier" | "ImportSpecifier" | "IndexedAccessType" | "InferredPredicate" | "InterfaceDeclaration" | "InterfaceExtends" | "InterfaceTypeAnnotation" | "InterpreterDirective" | "IntersectionTypeAnnotation" | "JSXAttribute" | "JSXClosingElement" | "JSXClosingFragment" | "JSXElement" | "JSXEmptyExpression" | "JSXExpressionContainer" | "JSXFragment" | "JSXMemberExpression" | "JSXNamespacedName" | "JSXOpeningElement" | "JSXOpeningFragment" | "JSXSpreadAttribute" | "JSXSpreadChild" | "LabeledStatement" | "LogicalExpression" | "MemberExpression" | "MetaProperty" | "MixedTypeAnnotation" | "ModuleExpression" | "NewExpression" | "Noop" | "NullLiteral" | "NullLiteralTypeAnnotation" | "NullableTypeAnnotation" | "NumberLiteral" | "NumberLiteralTypeAnnotation" | "NumberTypeAnnotation" | "ObjectExpression" | "ObjectMethod" | "ObjectPattern" | "ObjectProperty" | "ObjectTypeAnnotation" | "ObjectTypeCallProperty" | "ObjectTypeIndexer" | "ObjectTypeInternalSlot" | "ObjectTypeProperty" | "ObjectTypeSpreadProperty" | "OpaqueType" | "OptionalCallExpression" | "OptionalIndexedAccessType" | "OptionalMemberExpression" | "ParenthesizedExpression" | "PipelineBareFunction" | "PipelinePrimaryTopicReference" | "PipelineTopicExpression" | "Placeholder" | "PrivateName" | "Program" | "QualifiedTypeIdentifier" | "RecordExpression" | "RegExpLiteral" | "RegexLiteral" | "RestElement" | "RestProperty" | "ReturnStatement" | "SequenceExpression" | "SpreadElement" | "SpreadProperty" | "StaticBlock" | "StringLiteralTypeAnnotation" | "StringTypeAnnotation" | "Super" | "SwitchCase" | "SwitchStatement" | "SymbolTypeAnnotation" | "TSAnyKeyword" | "TSArrayType" | "TSAsExpression" | "TSBigIntKeyword" | "TSBooleanKeyword" | "TSCallSignatureDeclaration" | "TSConditionalType" | "TSConstructSignatureDeclaration" | "TSConstructorType" | "TSDeclareFunction" | "TSDeclareMethod" | "TSEnumDeclaration" | "TSEnumMember" | "TSExportAssignment" | "TSExpressionWithTypeArguments" | "TSExternalModuleReference" | "TSFunctionType" | "TSImportEqualsDeclaration" | "TSImportType" | "TSIndexSignature" | "TSIndexedAccessType" | "TSInferType" | "TSInstantiationExpression" | "TSInterfaceBody" | "TSInterfaceDeclaration" | "TSIntersectionType" | "TSIntrinsicKeyword" | "TSLiteralType" | "TSMappedType" | "TSMethodSignature" | "TSModuleBlock" | "TSModuleDeclaration" | "TSNamedTupleMember" | "TSNamespaceExportDeclaration" | "TSNeverKeyword" | "TSNonNullExpression" | "TSNullKeyword" | "TSNumberKeyword" | "TSObjectKeyword" | "TSOptionalType" | "TSParameterProperty" | "TSParenthesizedType" | "TSPropertySignature" | "TSQualifiedName" | "TSRestType" | "TSSatisfiesExpression" | "TSStringKeyword" | "TSSymbolKeyword" | "TSThisType" | "TSTupleType" | "TSTypeAliasDeclaration" | "TSTypeAnnotation" | "TSTypeAssertion" | "TSTypeLiteral" | "TSTypeOperator" | "TSTypeParameter" | "TSTypeParameterDeclaration" | "TSTypeParameterInstantiation" | "TSTypePredicate" | "TSTypeQuery" | "TSTypeReference" | "TSUndefinedKeyword" | "TSUnionType" | "TSUnknownKeyword" | "TSVoidKeyword" | "TaggedTemplateExpression" | "TemplateElement" | "TemplateLiteral" | "ThisExpression" | "ThisTypeAnnotation" | "ThrowStatement" | "TopicReference" | "TryStatement" | "TupleExpression" | "TupleTypeAnnotation" | "TypeAlias" | "TypeAnnotation" | "TypeCastExpression" | "TypeParameter" | "TypeParameterDeclaration" | "TypeParameterInstantiation" | "TypeofTypeAnnotation" | "UnaryExpression" | "UnionTypeAnnotation" | "UpdateExpression" | "V8IntrinsicIdentifier" | "VariableDeclaration" | "VariableDeclarator" | "Variance" | "VoidTypeAnnotation" | "WhileStatement" | "WithStatement" | "YieldExpression" | keyof Aliases)[];
declare const WHILE_TYPES: ("StringLiteral" | "NumericLiteral" | "JSXText" | "JSXIdentifier" | "AnyTypeAnnotation" | "ArgumentPlaceholder" | "ArrayExpression" | "ArrayPattern" | "ArrayTypeAnnotation" | "ArrowFunctionExpression" | "AssignmentExpression" | "AssignmentPattern" | "AwaitExpression" | "BigIntLiteral" | "BinaryExpression" | "BindExpression" | "BlockStatement" | "BooleanLiteral" | "BooleanLiteralTypeAnnotation" | "BooleanTypeAnnotation" | "BreakStatement" | "CallExpression" | "CatchClause" | "ClassAccessorProperty" | "ClassBody" | "ClassDeclaration" | "ClassExpression" | "ClassImplements" | "ClassMethod" | "ClassPrivateMethod" | "ClassPrivateProperty" | "ClassProperty" | "ConditionalExpression" | "ContinueStatement" | "DebuggerStatement" | "DecimalLiteral" | "DeclareClass" | "DeclareExportAllDeclaration" | "DeclareExportDeclaration" | "DeclareFunction" | "DeclareInterface" | "DeclareModule" | "DeclareModuleExports" | "DeclareOpaqueType" | "DeclareTypeAlias" | "DeclareVariable" | "DeclaredPredicate" | "Decorator" | "Directive" | "DirectiveLiteral" | "DoExpression" | "DoWhileStatement" | "EmptyStatement" | "EmptyTypeAnnotation" | "EnumBooleanBody" | "EnumBooleanMember" | "EnumDeclaration" | "EnumDefaultedMember" | "EnumNumberBody" | "EnumNumberMember" | "EnumStringBody" | "EnumStringMember" | "EnumSymbolBody" | "ExistsTypeAnnotation" | "ExportAllDeclaration" | "ExportDefaultDeclaration" | "ExportDefaultSpecifier" | "ExportNamedDeclaration" | "ExportNamespaceSpecifier" | "ExportSpecifier" | "ExpressionStatement" | "File" | "ForInStatement" | "ForOfStatement" | "ForStatement" | "FunctionDeclaration" | "FunctionExpression" | "FunctionTypeAnnotation" | "FunctionTypeParam" | "GenericTypeAnnotation" | "Identifier" | "IfStatement" | "Import" | "ImportAttribute" | "ImportDeclaration" | "ImportDefaultSpecifier" | "ImportExpression" | "ImportNamespaceSpecifier" | "ImportSpecifier" | "IndexedAccessType" | "InferredPredicate" | "InterfaceDeclaration" | "InterfaceExtends" | "InterfaceTypeAnnotation" | "InterpreterDirective" | "IntersectionTypeAnnotation" | "JSXAttribute" | "JSXClosingElement" | "JSXClosingFragment" | "JSXElement" | "JSXEmptyExpression" | "JSXExpressionContainer" | "JSXFragment" | "JSXMemberExpression" | "JSXNamespacedName" | "JSXOpeningElement" | "JSXOpeningFragment" | "JSXSpreadAttribute" | "JSXSpreadChild" | "LabeledStatement" | "LogicalExpression" | "MemberExpression" | "MetaProperty" | "MixedTypeAnnotation" | "ModuleExpression" | "NewExpression" | "Noop" | "NullLiteral" | "NullLiteralTypeAnnotation" | "NullableTypeAnnotation" | "NumberLiteral" | "NumberLiteralTypeAnnotation" | "NumberTypeAnnotation" | "ObjectExpression" | "ObjectMethod" | "ObjectPattern" | "ObjectProperty" | "ObjectTypeAnnotation" | "ObjectTypeCallProperty" | "ObjectTypeIndexer" | "ObjectTypeInternalSlot" | "ObjectTypeProperty" | "ObjectTypeSpreadProperty" | "OpaqueType" | "OptionalCallExpression" | "OptionalIndexedAccessType" | "OptionalMemberExpression" | "ParenthesizedExpression" | "PipelineBareFunction" | "PipelinePrimaryTopicReference" | "PipelineTopicExpression" | "Placeholder" | "PrivateName" | "Program" | "QualifiedTypeIdentifier" | "RecordExpression" | "RegExpLiteral" | "RegexLiteral" | "RestElement" | "RestProperty" | "ReturnStatement" | "SequenceExpression" | "SpreadElement" | "SpreadProperty" | "StaticBlock" | "StringLiteralTypeAnnotation" | "StringTypeAnnotation" | "Super" | "SwitchCase" | "SwitchStatement" | "SymbolTypeAnnotation" | "TSAnyKeyword" | "TSArrayType" | "TSAsExpression" | "TSBigIntKeyword" | "TSBooleanKeyword" | "TSCallSignatureDeclaration" | "TSConditionalType" | "TSConstructSignatureDeclaration" | "TSConstructorType" | "TSDeclareFunction" | "TSDeclareMethod" | "TSEnumDeclaration" | "TSEnumMember" | "TSExportAssignment" | "TSExpressionWithTypeArguments" | "TSExternalModuleReference" | "TSFunctionType" | "TSImportEqualsDeclaration" | "TSImportType" | "TSIndexSignature" | "TSIndexedAccessType" | "TSInferType" | "TSInstantiationExpression" | "TSInterfaceBody" | "TSInterfaceDeclaration" | "TSIntersectionType" | "TSIntrinsicKeyword" | "TSLiteralType" | "TSMappedType" | "TSMethodSignature" | "TSModuleBlock" | "TSModuleDeclaration" | "TSNamedTupleMember" | "TSNamespaceExportDeclaration" | "TSNeverKeyword" | "TSNonNullExpression" | "TSNullKeyword" | "TSNumberKeyword" | "TSObjectKeyword" | "TSOptionalType" | "TSParameterProperty" | "TSParenthesizedType" | "TSPropertySignature" | "TSQualifiedName" | "TSRestType" | "TSSatisfiesExpression" | "TSStringKeyword" | "TSSymbolKeyword" | "TSThisType" | "TSTupleType" | "TSTypeAliasDeclaration" | "TSTypeAnnotation" | "TSTypeAssertion" | "TSTypeLiteral" | "TSTypeOperator" | "TSTypeParameter" | "TSTypeParameterDeclaration" | "TSTypeParameterInstantiation" | "TSTypePredicate" | "TSTypeQuery" | "TSTypeReference" | "TSUndefinedKeyword" | "TSUnionType" | "TSUnknownKeyword" | "TSVoidKeyword" | "TaggedTemplateExpression" | "TemplateElement" | "TemplateLiteral" | "ThisExpression" | "ThisTypeAnnotation" | "ThrowStatement" | "TopicReference" | "TryStatement" | "TupleExpression" | "TupleTypeAnnotation" | "TypeAlias" | "TypeAnnotation" | "TypeCastExpression" | "TypeParameter" | "TypeParameterDeclaration" | "TypeParameterInstantiation" | "TypeofTypeAnnotation" | "UnaryExpression" | "UnionTypeAnnotation" | "UpdateExpression" | "V8IntrinsicIdentifier" | "VariableDeclaration" | "VariableDeclarator" | "Variance" | "VoidTypeAnnotation" | "WhileStatement" | "WithStatement" | "YieldExpression" | keyof Aliases)[];
declare const EXPRESSIONWRAPPER_TYPES: ("StringLiteral" | "NumericLiteral" | "JSXText" | "JSXIdentifier" | "AnyTypeAnnotation" | "ArgumentPlaceholder" | "ArrayExpression" | "ArrayPattern" | "ArrayTypeAnnotation" | "ArrowFunctionExpression" | "AssignmentExpression" | "AssignmentPattern" | "AwaitExpression" | "BigIntLiteral" | "BinaryExpression" | "BindExpression" | "BlockStatement" | "BooleanLiteral" | "BooleanLiteralTypeAnnotation" | "BooleanTypeAnnotation" | "BreakStatement" | "CallExpression" | "CatchClause" | "ClassAccessorProperty" | "ClassBody" | "ClassDeclaration" | "ClassExpression" | "ClassImplements" | "ClassMethod" | "ClassPrivateMethod" | "ClassPrivateProperty" | "ClassProperty" | "ConditionalExpression" | "ContinueStatement" | "DebuggerStatement" | "DecimalLiteral" | "DeclareClass" | "DeclareExportAllDeclaration" | "DeclareExportDeclaration" | "DeclareFunction" | "DeclareInterface" | "DeclareModule" | "DeclareModuleExports" | "DeclareOpaqueType" | "DeclareTypeAlias" | "DeclareVariable" | "DeclaredPredicate" | "Decorator" | "Directive" | "DirectiveLiteral" | "DoExpression" | "DoWhileStatement" | "EmptyStatement" | "EmptyTypeAnnotation" | "EnumBooleanBody" | "EnumBooleanMember" | "EnumDeclaration" | "EnumDefaultedMember" | "EnumNumberBody" | "EnumNumberMember" | "EnumStringBody" | "EnumStringMember" | "EnumSymbolBody" | "ExistsTypeAnnotation" | "ExportAllDeclaration" | "ExportDefaultDeclaration" | "ExportDefaultSpecifier" | "ExportNamedDeclaration" | "ExportNamespaceSpecifier" | "ExportSpecifier" | "ExpressionStatement" | "File" | "ForInStatement" | "ForOfStatement" | "ForStatement" | "FunctionDeclaration" | "FunctionExpression" | "FunctionTypeAnnotation" | "FunctionTypeParam" | "GenericTypeAnnotation" | "Identifier" | "IfStatement" | "Import" | "ImportAttribute" | "ImportDeclaration" | "ImportDefaultSpecifier" | "ImportExpression" | "ImportNamespaceSpecifier" | "ImportSpecifier" | "IndexedAccessType" | "InferredPredicate" | "InterfaceDeclaration" | "InterfaceExtends" | "InterfaceTypeAnnotation" | "InterpreterDirective" | "IntersectionTypeAnnotation" | "JSXAttribute" | "JSXClosingElement" | "JSXClosingFragment" | "JSXElement" | "JSXEmptyExpression" | "JSXExpressionContainer" | "JSXFragment" | "JSXMemberExpression" | "JSXNamespacedName" | "JSXOpeningElement" | "JSXOpeningFragment" | "JSXSpreadAttribute" | "JSXSpreadChild" | "LabeledStatement" | "LogicalExpression" | "MemberExpression" | "MetaProperty" | "MixedTypeAnnotation" | "ModuleExpression" | "NewExpression" | "Noop" | "NullLiteral" | "NullLiteralTypeAnnotation" | "NullableTypeAnnotation" | "NumberLiteral" | "NumberLiteralTypeAnnotation" | "NumberTypeAnnotation" | "ObjectExpression" | "ObjectMethod" | "ObjectPattern" | "ObjectProperty" | "ObjectTypeAnnotation" | "ObjectTypeCallProperty" | "ObjectTypeIndexer" | "ObjectTypeInternalSlot" | "ObjectTypeProperty" | "ObjectTypeSpreadProperty" | "OpaqueType" | "OptionalCallExpression" | "OptionalIndexedAccessType" | "OptionalMemberExpression" | "ParenthesizedExpression" | "PipelineBareFunction" | "PipelinePrimaryTopicReference" | "PipelineTopicExpression" | "Placeholder" | "PrivateName" | "Program" | "QualifiedTypeIdentifier" | "RecordExpression" | "RegExpLiteral" | "RegexLiteral" | "RestElement" | "RestProperty" | "ReturnStatement" | "SequenceExpression" | "SpreadElement" | "SpreadProperty" | "StaticBlock" | "StringLiteralTypeAnnotation" | "StringTypeAnnotation" | "Super" | "SwitchCase" | "SwitchStatement" | "SymbolTypeAnnotation" | "TSAnyKeyword" | "TSArrayType" | "TSAsExpression" | "TSBigIntKeyword" | "TSBooleanKeyword" | "TSCallSignatureDeclaration" | "TSConditionalType" | "TSConstructSignatureDeclaration" | "TSConstructorType" | "TSDeclareFunction" | "TSDeclareMethod" | "TSEnumDeclaration" | "TSEnumMember" | "TSExportAssignment" | "TSExpressionWithTypeArguments" | "TSExternalModuleReference" | "TSFunctionType" | "TSImportEqualsDeclaration" | "TSImportType" | "TSIndexSignature" | "TSIndexedAccessType" | "TSInferType" | "TSInstantiationExpression" | "TSInterfaceBody" | "TSInterfaceDeclaration" | "TSIntersectionType" | "TSIntrinsicKeyword" | "TSLiteralType" | "TSMappedType" | "TSMethodSignature" | "TSModuleBlock" | "TSModuleDeclaration" | "TSNamedTupleMember" | "TSNamespaceExportDeclaration" | "TSNeverKeyword" | "TSNonNullExpression" | "TSNullKeyword" | "TSNumberKeyword" | "TSObjectKeyword" | "TSOptionalType" | "TSParameterProperty" | "TSParenthesizedType" | "TSPropertySignature" | "TSQualifiedName" | "TSRestType" | "TSSatisfiesExpression" | "TSStringKeyword" | "TSSymbolKeyword" | "TSThisType" | "TSTupleType" | "TSTypeAliasDeclaration" | "TSTypeAnnotation" | "TSTypeAssertion" | "TSTypeLiteral" | "TSTypeOperator" | "TSTypeParameter" | "TSTypeParameterDeclaration" | "TSTypeParameterInstantiation" | "TSTypePredicate" | "TSTypeQuery" | "TSTypeReference" | "TSUndefinedKeyword" | "TSUnionType" | "TSUnknownKeyword" | "TSVoidKeyword" | "TaggedTemplateExpression" | "TemplateElement" | "TemplateLiteral" | "ThisExpression" | "ThisTypeAnnotation" | "ThrowStatement" | "TopicReference" | "TryStatement" | "TupleExpression" | "TupleTypeAnnotation" | "TypeAlias" | "TypeAnnotation" | "TypeCastExpression" | "TypeParameter" | "TypeParameterDeclaration" | "TypeParameterInstantiation" | "TypeofTypeAnnotation" | "UnaryExpression" | "UnionTypeAnnotation" | "UpdateExpression" | "V8IntrinsicIdentifier" | "VariableDeclaration" | "VariableDeclarator" | "Variance" | "VoidTypeAnnotation" | "WhileStatement" | "WithStatement" | "YieldExpression" | keyof Aliases)[];
declare const FOR_TYPES: ("StringLiteral" | "NumericLiteral" | "JSXText" | "JSXIdentifier" | "AnyTypeAnnotation" | "ArgumentPlaceholder" | "ArrayExpression" | "ArrayPattern" | "ArrayTypeAnnotation" | "ArrowFunctionExpression" | "AssignmentExpression" | "AssignmentPattern" | "AwaitExpression" | "BigIntLiteral" | "BinaryExpression" | "BindExpression" | "BlockStatement" | "BooleanLiteral" | "BooleanLiteralTypeAnnotation" | "BooleanTypeAnnotation" | "BreakStatement" | "CallExpression" | "CatchClause" | "ClassAccessorProperty" | "ClassBody" | "ClassDeclaration" | "ClassExpression" | "ClassImplements" | "ClassMethod" | "ClassPrivateMethod" | "ClassPrivateProperty" | "ClassProperty" | "ConditionalExpression" | "ContinueStatement" | "DebuggerStatement" | "DecimalLiteral" | "DeclareClass" | "DeclareExportAllDeclaration" | "DeclareExportDeclaration" | "DeclareFunction" | "DeclareInterface" | "DeclareModule" | "DeclareModuleExports" | "DeclareOpaqueType" | "DeclareTypeAlias" | "DeclareVariable" | "DeclaredPredicate" | "Decorator" | "Directive" | "DirectiveLiteral" | "DoExpression" | "DoWhileStatement" | "EmptyStatement" | "EmptyTypeAnnotation" | "EnumBooleanBody" | "EnumBooleanMember" | "EnumDeclaration" | "EnumDefaultedMember" | "EnumNumberBody" | "EnumNumberMember" | "EnumStringBody" | "EnumStringMember" | "EnumSymbolBody" | "ExistsTypeAnnotation" | "ExportAllDeclaration" | "ExportDefaultDeclaration" | "ExportDefaultSpecifier" | "ExportNamedDeclaration" | "ExportNamespaceSpecifier" | "ExportSpecifier" | "ExpressionStatement" | "File" | "ForInStatement" | "ForOfStatement" | "ForStatement" | "FunctionDeclaration" | "FunctionExpression" | "FunctionTypeAnnotation" | "FunctionTypeParam" | "GenericTypeAnnotation" | "Identifier" | "IfStatement" | "Import" | "ImportAttribute" | "ImportDeclaration" | "ImportDefaultSpecifier" | "ImportExpression" | "ImportNamespaceSpecifier" | "ImportSpecifier" | "IndexedAccessType" | "InferredPredicate" | "InterfaceDeclaration" | "InterfaceExtends" | "InterfaceTypeAnnotation" | "InterpreterDirective" | "IntersectionTypeAnnotation" | "JSXAttribute" | "JSXClosingElement" | "JSXClosingFragment" | "JSXElement" | "JSXEmptyExpression" | "JSXExpressionContainer" | "JSXFragment" | "JSXMemberExpression" | "JSXNamespacedName" | "JSXOpeningElement" | "JSXOpeningFragment" | "JSXSpreadAttribute" | "JSXSpreadChild" | "LabeledStatement" | "LogicalExpression" | "MemberExpression" | "MetaProperty" | "MixedTypeAnnotation" | "ModuleExpression" | "NewExpression" | "Noop" | "NullLiteral" | "NullLiteralTypeAnnotation" | "NullableTypeAnnotation" | "NumberLiteral" | "NumberLiteralTypeAnnotation" | "NumberTypeAnnotation" | "ObjectExpression" | "ObjectMethod" | "ObjectPattern" | "ObjectProperty" | "ObjectTypeAnnotation" | "ObjectTypeCallProperty" | "ObjectTypeIndexer" | "ObjectTypeInternalSlot" | "ObjectTypeProperty" | "ObjectTypeSpreadProperty" | "OpaqueType" | "OptionalCallExpression" | "OptionalIndexedAccessType" | "OptionalMemberExpression" | "ParenthesizedExpression" | "PipelineBareFunction" | "PipelinePrimaryTopicReference" | "PipelineTopicExpression" | "Placeholder" | "PrivateName" | "Program" | "QualifiedTypeIdentifier" | "RecordExpression" | "RegExpLiteral" | "RegexLiteral" | "RestElement" | "RestProperty" | "ReturnStatement" | "SequenceExpression" | "SpreadElement" | "SpreadProperty" | "StaticBlock" | "StringLiteralTypeAnnotation" | "StringTypeAnnotation" | "Super" | "SwitchCase" | "SwitchStatement" | "SymbolTypeAnnotation" | "TSAnyKeyword" | "TSArrayType" | "TSAsExpression" | "TSBigIntKeyword" | "TSBooleanKeyword" | "TSCallSignatureDeclaration" | "TSConditionalType" | "TSConstructSignatureDeclaration" | "TSConstructorType" | "TSDeclareFunction" | "TSDeclareMethod" | "TSEnumDeclaration" | "TSEnumMember" | "TSExportAssignment" | "TSExpressionWithTypeArguments" | "TSExternalModuleReference" | "TSFunctionType" | "TSImportEqualsDeclaration" | "TSImportType" | "TSIndexSignature" | "TSIndexedAccessType" | "TSInferType" | "TSInstantiationExpression" | "TSInterfaceBody" | "TSInterfaceDeclaration" | "TSIntersectionType" | "TSIntrinsicKeyword" | "TSLiteralType" | "TSMappedType" | "TSMethodSignature" | "TSModuleBlock" | "TSModuleDeclaration" | "TSNamedTupleMember" | "TSNamespaceExportDeclaration" | "TSNeverKeyword" | "TSNonNullExpression" | "TSNullKeyword" | "TSNumberKeyword" | "TSObjectKeyword" | "TSOptionalType" | "TSParameterProperty" | "TSParenthesizedType" | "TSPropertySignature" | "TSQualifiedName" | "TSRestType" | "TSSatisfiesExpression" | "TSStringKeyword" | "TSSymbolKeyword" | "TSThisType" | "TSTupleType" | "TSTypeAliasDeclaration" | "TSTypeAnnotation" | "TSTypeAssertion" | "TSTypeLiteral" | "TSTypeOperator" | "TSTypeParameter" | "TSTypeParameterDeclaration" | "TSTypeParameterInstantiation" | "TSTypePredicate" | "TSTypeQuery" | "TSTypeReference" | "TSUndefinedKeyword" | "TSUnionType" | "TSUnknownKeyword" | "TSVoidKeyword" | "TaggedTemplateExpression" | "TemplateElement" | "TemplateLiteral" | "ThisExpression" | "ThisTypeAnnotation" | "ThrowStatement" | "TopicReference" | "TryStatement" | "TupleExpression" | "TupleTypeAnnotation" | "TypeAlias" | "TypeAnnotation" | "TypeCastExpression" | "TypeParameter" | "TypeParameterDeclaration" | "TypeParameterInstantiation" | "TypeofTypeAnnotation" | "UnaryExpression" | "UnionTypeAnnotation" | "UpdateExpression" | "V8IntrinsicIdentifier" | "VariableDeclaration" | "VariableDeclarator" | "Variance" | "VoidTypeAnnotation" | "WhileStatement" | "WithStatement" | "YieldExpression" | keyof Aliases)[];
declare const FORXSTATEMENT_TYPES: ("StringLiteral" | "NumericLiteral" | "JSXText" | "JSXIdentifier" | "AnyTypeAnnotation" | "ArgumentPlaceholder" | "ArrayExpression" | "ArrayPattern" | "ArrayTypeAnnotation" | "ArrowFunctionExpression" | "AssignmentExpression" | "AssignmentPattern" | "AwaitExpression" | "BigIntLiteral" | "BinaryExpression" | "BindExpression" | "BlockStatement" | "BooleanLiteral" | "BooleanLiteralTypeAnnotation" | "BooleanTypeAnnotation" | "BreakStatement" | "CallExpression" | "CatchClause" | "ClassAccessorProperty" | "ClassBody" | "ClassDeclaration" | "ClassExpression" | "ClassImplements" | "ClassMethod" | "ClassPrivateMethod" | "ClassPrivateProperty" | "ClassProperty" | "ConditionalExpression" | "ContinueStatement" | "DebuggerStatement" | "DecimalLiteral" | "DeclareClass" | "DeclareExportAllDeclaration" | "DeclareExportDeclaration" | "DeclareFunction" | "DeclareInterface" | "DeclareModule" | "DeclareModuleExports" | "DeclareOpaqueType" | "DeclareTypeAlias" | "DeclareVariable" | "DeclaredPredicate" | "Decorator" | "Directive" | "DirectiveLiteral" | "DoExpression" | "DoWhileStatement" | "EmptyStatement" | "EmptyTypeAnnotation" | "EnumBooleanBody" | "EnumBooleanMember" | "EnumDeclaration" | "EnumDefaultedMember" | "EnumNumberBody" | "EnumNumberMember" | "EnumStringBody" | "EnumStringMember" | "EnumSymbolBody" | "ExistsTypeAnnotation" | "ExportAllDeclaration" | "ExportDefaultDeclaration" | "ExportDefaultSpecifier" | "ExportNamedDeclaration" | "ExportNamespaceSpecifier" | "ExportSpecifier" | "ExpressionStatement" | "File" | "ForInStatement" | "ForOfStatement" | "ForStatement" | "FunctionDeclaration" | "FunctionExpression" | "FunctionTypeAnnotation" | "FunctionTypeParam" | "GenericTypeAnnotation" | "Identifier" | "IfStatement" | "Import" | "ImportAttribute" | "ImportDeclaration" | "ImportDefaultSpecifier" | "ImportExpression" | "ImportNamespaceSpecifier" | "ImportSpecifier" | "IndexedAccessType" | "InferredPredicate" | "InterfaceDeclaration" | "InterfaceExtends" | "InterfaceTypeAnnotation" | "InterpreterDirective" | "IntersectionTypeAnnotation" | "JSXAttribute" | "JSXClosingElement" | "JSXClosingFragment" | "JSXElement" | "JSXEmptyExpression" | "JSXExpressionContainer" | "JSXFragment" | "JSXMemberExpression" | "JSXNamespacedName" | "JSXOpeningElement" | "JSXOpeningFragment" | "JSXSpreadAttribute" | "JSXSpreadChild" | "LabeledStatement" | "LogicalExpression" | "MemberExpression" | "MetaProperty" | "MixedTypeAnnotation" | "ModuleExpression" | "NewExpression" | "Noop" | "NullLiteral" | "NullLiteralTypeAnnotation" | "NullableTypeAnnotation" | "NumberLiteral" | "NumberLiteralTypeAnnotation" | "NumberTypeAnnotation" | "ObjectExpression" | "ObjectMethod" | "ObjectPattern" | "ObjectProperty" | "ObjectTypeAnnotation" | "ObjectTypeCallProperty" | "ObjectTypeIndexer" | "ObjectTypeInternalSlot" | "ObjectTypeProperty" | "ObjectTypeSpreadProperty" | "OpaqueType" | "OptionalCallExpression" | "OptionalIndexedAccessType" | "OptionalMemberExpression" | "ParenthesizedExpression" | "PipelineBareFunction" | "PipelinePrimaryTopicReference" | "PipelineTopicExpression" | "Placeholder" | "PrivateName" | "Program" | "QualifiedTypeIdentifier" | "RecordExpression" | "RegExpLiteral" | "RegexLiteral" | "RestElement" | "RestProperty" | "ReturnStatement" | "SequenceExpression" | "SpreadElement" | "SpreadProperty" | "StaticBlock" | "StringLiteralTypeAnnotation" | "StringTypeAnnotation" | "Super" | "SwitchCase" | "SwitchStatement" | "SymbolTypeAnnotation" | "TSAnyKeyword" | "TSArrayType" | "TSAsExpression" | "TSBigIntKeyword" | "TSBooleanKeyword" | "TSCallSignatureDeclaration" | "TSConditionalType" | "TSConstructSignatureDeclaration" | "TSConstructorType" | "TSDeclareFunction" | "TSDeclareMethod" | "TSEnumDeclaration" | "TSEnumMember" | "TSExportAssignment" | "TSExpressionWithTypeArguments" | "TSExternalModuleReference" | "TSFunctionType" | "TSImportEqualsDeclaration" | "TSImportType" | "TSIndexSignature" | "TSIndexedAccessType" | "TSInferType" | "TSInstantiationExpression" | "TSInterfaceBody" | "TSInterfaceDeclaration" | "TSIntersectionType" | "TSIntrinsicKeyword" | "TSLiteralType" | "TSMappedType" | "TSMethodSignature" | "TSModuleBlock" | "TSModuleDeclaration" | "TSNamedTupleMember" | "TSNamespaceExportDeclaration" | "TSNeverKeyword" | "TSNonNullExpression" | "TSNullKeyword" | "TSNumberKeyword" | "TSObjectKeyword" | "TSOptionalType" | "TSParameterProperty" | "TSParenthesizedType" | "TSPropertySignature" | "TSQualifiedName" | "TSRestType" | "TSSatisfiesExpression" | "TSStringKeyword" | "TSSymbolKeyword" | "TSThisType" | "TSTupleType" | "TSTypeAliasDeclaration" | "TSTypeAnnotation" | "TSTypeAssertion" | "TSTypeLiteral" | "TSTypeOperator" | "TSTypeParameter" | "TSTypeParameterDeclaration" | "TSTypeParameterInstantiation" | "TSTypePredicate" | "TSTypeQuery" | "TSTypeReference" | "TSUndefinedKeyword" | "TSUnionType" | "TSUnknownKeyword" | "TSVoidKeyword" | "TaggedTemplateExpression" | "TemplateElement" | "TemplateLiteral" | "ThisExpression" | "ThisTypeAnnotation" | "ThrowStatement" | "TopicReference" | "TryStatement" | "TupleExpression" | "TupleTypeAnnotation" | "TypeAlias" | "TypeAnnotation" | "TypeCastExpression" | "TypeParameter" | "TypeParameterDeclaration" | "TypeParameterInstantiation" | "TypeofTypeAnnotation" | "UnaryExpression" | "UnionTypeAnnotation" | "UpdateExpression" | "V8IntrinsicIdentifier" | "VariableDeclaration" | "VariableDeclarator" | "Variance" | "VoidTypeAnnotation" | "WhileStatement" | "WithStatement" | "YieldExpression" | keyof Aliases)[];
declare const FUNCTION_TYPES: ("StringLiteral" | "NumericLiteral" | "JSXText" | "JSXIdentifier" | "AnyTypeAnnotation" | "ArgumentPlaceholder" | "ArrayExpression" | "ArrayPattern" | "ArrayTypeAnnotation" | "ArrowFunctionExpression" | "AssignmentExpression" | "AssignmentPattern" | "AwaitExpression" | "BigIntLiteral" | "BinaryExpression" | "BindExpression" | "BlockStatement" | "BooleanLiteral" | "BooleanLiteralTypeAnnotation" | "BooleanTypeAnnotation" | "BreakStatement" | "CallExpression" | "CatchClause" | "ClassAccessorProperty" | "ClassBody" | "ClassDeclaration" | "ClassExpression" | "ClassImplements" | "ClassMethod" | "ClassPrivateMethod" | "ClassPrivateProperty" | "ClassProperty" | "ConditionalExpression" | "ContinueStatement" | "DebuggerStatement" | "DecimalLiteral" | "DeclareClass" | "DeclareExportAllDeclaration" | "DeclareExportDeclaration" | "DeclareFunction" | "DeclareInterface" | "DeclareModule" | "DeclareModuleExports" | "DeclareOpaqueType" | "DeclareTypeAlias" | "DeclareVariable" | "DeclaredPredicate" | "Decorator" | "Directive" | "DirectiveLiteral" | "DoExpression" | "DoWhileStatement" | "EmptyStatement" | "EmptyTypeAnnotation" | "EnumBooleanBody" | "EnumBooleanMember" | "EnumDeclaration" | "EnumDefaultedMember" | "EnumNumberBody" | "EnumNumberMember" | "EnumStringBody" | "EnumStringMember" | "EnumSymbolBody" | "ExistsTypeAnnotation" | "ExportAllDeclaration" | "ExportDefaultDeclaration" | "ExportDefaultSpecifier" | "ExportNamedDeclaration" | "ExportNamespaceSpecifier" | "ExportSpecifier" | "ExpressionStatement" | "File" | "ForInStatement" | "ForOfStatement" | "ForStatement" | "FunctionDeclaration" | "FunctionExpression" | "FunctionTypeAnnotation" | "FunctionTypeParam" | "GenericTypeAnnotation" | "Identifier" | "IfStatement" | "Import" | "ImportAttribute" | "ImportDeclaration" | "ImportDefaultSpecifier" | "ImportExpression" | "ImportNamespaceSpecifier" | "ImportSpecifier" | "IndexedAccessType" | "InferredPredicate" | "InterfaceDeclaration" | "InterfaceExtends" | "InterfaceTypeAnnotation" | "InterpreterDirective" | "IntersectionTypeAnnotation" | "JSXAttribute" | "JSXClosingElement" | "JSXClosingFragment" | "JSXElement" | "JSXEmptyExpression" | "JSXExpressionContainer" | "JSXFragment" | "JSXMemberExpression" | "JSXNamespacedName" | "JSXOpeningElement" | "JSXOpeningFragment" | "JSXSpreadAttribute" | "JSXSpreadChild" | "LabeledStatement" | "LogicalExpression" | "MemberExpression" | "MetaProperty" | "MixedTypeAnnotation" | "ModuleExpression" | "NewExpression" | "Noop" | "NullLiteral" | "NullLiteralTypeAnnotation" | "NullableTypeAnnotation" | "NumberLiteral" | "NumberLiteralTypeAnnotation" | "NumberTypeAnnotation" | "ObjectExpression" | "ObjectMethod" | "ObjectPattern" | "ObjectProperty" | "ObjectTypeAnnotation" | "ObjectTypeCallProperty" | "ObjectTypeIndexer" | "ObjectTypeInternalSlot" | "ObjectTypeProperty" | "ObjectTypeSpreadProperty" | "OpaqueType" | "OptionalCallExpression" | "OptionalIndexedAccessType" | "OptionalMemberExpression" | "ParenthesizedExpression" | "PipelineBareFunction" | "PipelinePrimaryTopicReference" | "PipelineTopicExpression" | "Placeholder" | "PrivateName" | "Program" | "QualifiedTypeIdentifier" | "RecordExpression" | "RegExpLiteral" | "RegexLiteral" | "RestElement" | "RestProperty" | "ReturnStatement" | "SequenceExpression" | "SpreadElement" | "SpreadProperty" | "StaticBlock" | "StringLiteralTypeAnnotation" | "StringTypeAnnotation" | "Super" | "SwitchCase" | "SwitchStatement" | "SymbolTypeAnnotation" | "TSAnyKeyword" | "TSArrayType" | "TSAsExpression" | "TSBigIntKeyword" | "TSBooleanKeyword" | "TSCallSignatureDeclaration" | "TSConditionalType" | "TSConstructSignatureDeclaration" | "TSConstructorType" | "TSDeclareFunction" | "TSDeclareMethod" | "TSEnumDeclaration" | "TSEnumMember" | "TSExportAssignment" | "TSExpressionWithTypeArguments" | "TSExternalModuleReference" | "TSFunctionType" | "TSImportEqualsDeclaration" | "TSImportType" | "TSIndexSignature" | "TSIndexedAccessType" | "TSInferType" | "TSInstantiationExpression" | "TSInterfaceBody" | "TSInterfaceDeclaration" | "TSIntersectionType" | "TSIntrinsicKeyword" | "TSLiteralType" | "TSMappedType" | "TSMethodSignature" | "TSModuleBlock" | "TSModuleDeclaration" | "TSNamedTupleMember" | "TSNamespaceExportDeclaration" | "TSNeverKeyword" | "TSNonNullExpression" | "TSNullKeyword" | "TSNumberKeyword" | "TSObjectKeyword" | "TSOptionalType" | "TSParameterProperty" | "TSParenthesizedType" | "TSPropertySignature" | "TSQualifiedName" | "TSRestType" | "TSSatisfiesExpression" | "TSStringKeyword" | "TSSymbolKeyword" | "TSThisType" | "TSTupleType" | "TSTypeAliasDeclaration" | "TSTypeAnnotation" | "TSTypeAssertion" | "TSTypeLiteral" | "TSTypeOperator" | "TSTypeParameter" | "TSTypeParameterDeclaration" | "TSTypeParameterInstantiation" | "TSTypePredicate" | "TSTypeQuery" | "TSTypeReference" | "TSUndefinedKeyword" | "TSUnionType" | "TSUnknownKeyword" | "TSVoidKeyword" | "TaggedTemplateExpression" | "TemplateElement" | "TemplateLiteral" | "ThisExpression" | "ThisTypeAnnotation" | "ThrowStatement" | "TopicReference" | "TryStatement" | "TupleExpression" | "TupleTypeAnnotation" | "TypeAlias" | "TypeAnnotation" | "TypeCastExpression" | "TypeParameter" | "TypeParameterDeclaration" | "TypeParameterInstantiation" | "TypeofTypeAnnotation" | "UnaryExpression" | "UnionTypeAnnotation" | "UpdateExpression" | "V8IntrinsicIdentifier" | "VariableDeclaration" | "VariableDeclarator" | "Variance" | "VoidTypeAnnotation" | "WhileStatement" | "WithStatement" | "YieldExpression" | keyof Aliases)[];
declare const FUNCTIONPARENT_TYPES: ("StringLiteral" | "NumericLiteral" | "JSXText" | "JSXIdentifier" | "AnyTypeAnnotation" | "ArgumentPlaceholder" | "ArrayExpression" | "ArrayPattern" | "ArrayTypeAnnotation" | "ArrowFunctionExpression" | "AssignmentExpression" | "AssignmentPattern" | "AwaitExpression" | "BigIntLiteral" | "BinaryExpression" | "BindExpression" | "BlockStatement" | "BooleanLiteral" | "BooleanLiteralTypeAnnotation" | "BooleanTypeAnnotation" | "BreakStatement" | "CallExpression" | "CatchClause" | "ClassAccessorProperty" | "ClassBody" | "ClassDeclaration" | "ClassExpression" | "ClassImplements" | "ClassMethod" | "ClassPrivateMethod" | "ClassPrivateProperty" | "ClassProperty" | "ConditionalExpression" | "ContinueStatement" | "DebuggerStatement" | "DecimalLiteral" | "DeclareClass" | "DeclareExportAllDeclaration" | "DeclareExportDeclaration" | "DeclareFunction" | "DeclareInterface" | "DeclareModule" | "DeclareModuleExports" | "DeclareOpaqueType" | "DeclareTypeAlias" | "DeclareVariable" | "DeclaredPredicate" | "Decorator" | "Directive" | "DirectiveLiteral" | "DoExpression" | "DoWhileStatement" | "EmptyStatement" | "EmptyTypeAnnotation" | "EnumBooleanBody" | "EnumBooleanMember" | "EnumDeclaration" | "EnumDefaultedMember" | "EnumNumberBody" | "EnumNumberMember" | "EnumStringBody" | "EnumStringMember" | "EnumSymbolBody" | "ExistsTypeAnnotation" | "ExportAllDeclaration" | "ExportDefaultDeclaration" | "ExportDefaultSpecifier" | "ExportNamedDeclaration" | "ExportNamespaceSpecifier" | "ExportSpecifier" | "ExpressionStatement" | "File" | "ForInStatement" | "ForOfStatement" | "ForStatement" | "FunctionDeclaration" | "FunctionExpression" | "FunctionTypeAnnotation" | "FunctionTypeParam" | "GenericTypeAnnotation" | "Identifier" | "IfStatement" | "Import" | "ImportAttribute" | "ImportDeclaration" | "ImportDefaultSpecifier" | "ImportExpression" | "ImportNamespaceSpecifier" | "ImportSpecifier" | "IndexedAccessType" | "InferredPredicate" | "InterfaceDeclaration" | "InterfaceExtends" | "InterfaceTypeAnnotation" | "InterpreterDirective" | "IntersectionTypeAnnotation" | "JSXAttribute" | "JSXClosingElement" | "JSXClosingFragment" | "JSXElement" | "JSXEmptyExpression" | "JSXExpressionContainer" | "JSXFragment" | "JSXMemberExpression" | "JSXNamespacedName" | "JSXOpeningElement" | "JSXOpeningFragment" | "JSXSpreadAttribute" | "JSXSpreadChild" | "LabeledStatement" | "LogicalExpression" | "MemberExpression" | "MetaProperty" | "MixedTypeAnnotation" | "ModuleExpression" | "NewExpression" | "Noop" | "NullLiteral" | "NullLiteralTypeAnnotation" | "NullableTypeAnnotation" | "NumberLiteral" | "NumberLiteralTypeAnnotation" | "NumberTypeAnnotation" | "ObjectExpression" | "ObjectMethod" | "ObjectPattern" | "ObjectProperty" | "ObjectTypeAnnotation" | "ObjectTypeCallProperty" | "ObjectTypeIndexer" | "ObjectTypeInternalSlot" | "ObjectTypeProperty" | "ObjectTypeSpreadProperty" | "OpaqueType" | "OptionalCallExpression" | "OptionalIndexedAccessType" | "OptionalMemberExpression" | "ParenthesizedExpression" | "PipelineBareFunction" | "PipelinePrimaryTopicReference" | "PipelineTopicExpression" | "Placeholder" | "PrivateName" | "Program" | "QualifiedTypeIdentifier" | "RecordExpression" | "RegExpLiteral" | "RegexLiteral" | "RestElement" | "RestProperty" | "ReturnStatement" | "SequenceExpression" | "SpreadElement" | "SpreadProperty" | "StaticBlock" | "StringLiteralTypeAnnotation" | "StringTypeAnnotation" | "Super" | "SwitchCase" | "SwitchStatement" | "SymbolTypeAnnotation" | "TSAnyKeyword" | "TSArrayType" | "TSAsExpression" | "TSBigIntKeyword" | "TSBooleanKeyword" | "TSCallSignatureDeclaration" | "TSConditionalType" | "TSConstructSignatureDeclaration" | "TSConstructorType" | "TSDeclareFunction" | "TSDeclareMethod" | "TSEnumDeclaration" | "TSEnumMember" | "TSExportAssignment" | "TSExpressionWithTypeArguments" | "TSExternalModuleReference" | "TSFunctionType" | "TSImportEqualsDeclaration" | "TSImportType" | "TSIndexSignature" | "TSIndexedAccessType" | "TSInferType" | "TSInstantiationExpression" | "TSInterfaceBody" | "TSInterfaceDeclaration" | "TSIntersectionType" | "TSIntrinsicKeyword" | "TSLiteralType" | "TSMappedType" | "TSMethodSignature" | "TSModuleBlock" | "TSModuleDeclaration" | "TSNamedTupleMember" | "TSNamespaceExportDeclaration" | "TSNeverKeyword" | "TSNonNullExpression" | "TSNullKeyword" | "TSNumberKeyword" | "TSObjectKeyword" | "TSOptionalType" | "TSParameterProperty" | "TSParenthesizedType" | "TSPropertySignature" | "TSQualifiedName" | "TSRestType" | "TSSatisfiesExpression" | "TSStringKeyword" | "TSSymbolKeyword" | "TSThisType" | "TSTupleType" | "TSTypeAliasDeclaration" | "TSTypeAnnotation" | "TSTypeAssertion" | "TSTypeLiteral" | "TSTypeOperator" | "TSTypeParameter" | "TSTypeParameterDeclaration" | "TSTypeParameterInstantiation" | "TSTypePredicate" | "TSTypeQuery" | "TSTypeReference" | "TSUndefinedKeyword" | "TSUnionType" | "TSUnknownKeyword" | "TSVoidKeyword" | "TaggedTemplateExpression" | "TemplateElement" | "TemplateLiteral" | "ThisExpression" | "ThisTypeAnnotation" | "ThrowStatement" | "TopicReference" | "TryStatement" | "TupleExpression" | "TupleTypeAnnotation" | "TypeAlias" | "TypeAnnotation" | "TypeCastExpression" | "TypeParameter" | "TypeParameterDeclaration" | "TypeParameterInstantiation" | "TypeofTypeAnnotation" | "UnaryExpression" | "UnionTypeAnnotation" | "UpdateExpression" | "V8IntrinsicIdentifier" | "VariableDeclaration" | "VariableDeclarator" | "Variance" | "VoidTypeAnnotation" | "WhileStatement" | "WithStatement" | "YieldExpression" | keyof Aliases)[];
declare const PUREISH_TYPES: ("StringLiteral" | "NumericLiteral" | "JSXText" | "JSXIdentifier" | "AnyTypeAnnotation" | "ArgumentPlaceholder" | "ArrayExpression" | "ArrayPattern" | "ArrayTypeAnnotation" | "ArrowFunctionExpression" | "AssignmentExpression" | "AssignmentPattern" | "AwaitExpression" | "BigIntLiteral" | "BinaryExpression" | "BindExpression" | "BlockStatement" | "BooleanLiteral" | "BooleanLiteralTypeAnnotation" | "BooleanTypeAnnotation" | "BreakStatement" | "CallExpression" | "CatchClause" | "ClassAccessorProperty" | "ClassBody" | "ClassDeclaration" | "ClassExpression" | "ClassImplements" | "ClassMethod" | "ClassPrivateMethod" | "ClassPrivateProperty" | "ClassProperty" | "ConditionalExpression" | "ContinueStatement" | "DebuggerStatement" | "DecimalLiteral" | "DeclareClass" | "DeclareExportAllDeclaration" | "DeclareExportDeclaration" | "DeclareFunction" | "DeclareInterface" | "DeclareModule" | "DeclareModuleExports" | "DeclareOpaqueType" | "DeclareTypeAlias" | "DeclareVariable" | "DeclaredPredicate" | "Decorator" | "Directive" | "DirectiveLiteral" | "DoExpression" | "DoWhileStatement" | "EmptyStatement" | "EmptyTypeAnnotation" | "EnumBooleanBody" | "EnumBooleanMember" | "EnumDeclaration" | "EnumDefaultedMember" | "EnumNumberBody" | "EnumNumberMember" | "EnumStringBody" | "EnumStringMember" | "EnumSymbolBody" | "ExistsTypeAnnotation" | "ExportAllDeclaration" | "ExportDefaultDeclaration" | "ExportDefaultSpecifier" | "ExportNamedDeclaration" | "ExportNamespaceSpecifier" | "ExportSpecifier" | "ExpressionStatement" | "File" | "ForInStatement" | "ForOfStatement" | "ForStatement" | "FunctionDeclaration" | "FunctionExpression" | "FunctionTypeAnnotation" | "FunctionTypeParam" | "GenericTypeAnnotation" | "Identifier" | "IfStatement" | "Import" | "ImportAttribute" | "ImportDeclaration" | "ImportDefaultSpecifier" | "ImportExpression" | "ImportNamespaceSpecifier" | "ImportSpecifier" | "IndexedAccessType" | "InferredPredicate" | "InterfaceDeclaration" | "InterfaceExtends" | "InterfaceTypeAnnotation" | "InterpreterDirective" | "IntersectionTypeAnnotation" | "JSXAttribute" | "JSXClosingElement" | "JSXClosingFragment" | "JSXElement" | "JSXEmptyExpression" | "JSXExpressionContainer" | "JSXFragment" | "JSXMemberExpression" | "JSXNamespacedName" | "JSXOpeningElement" | "JSXOpeningFragment" | "JSXSpreadAttribute" | "JSXSpreadChild" | "LabeledStatement" | "LogicalExpression" | "MemberExpression" | "MetaProperty" | "MixedTypeAnnotation" | "ModuleExpression" | "NewExpression" | "Noop" | "NullLiteral" | "NullLiteralTypeAnnotation" | "NullableTypeAnnotation" | "NumberLiteral" | "NumberLiteralTypeAnnotation" | "NumberTypeAnnotation" | "ObjectExpression" | "ObjectMethod" | "ObjectPattern" | "ObjectProperty" | "ObjectTypeAnnotation" | "ObjectTypeCallProperty" | "ObjectTypeIndexer" | "ObjectTypeInternalSlot" | "ObjectTypeProperty" | "ObjectTypeSpreadProperty" | "OpaqueType" | "OptionalCallExpression" | "OptionalIndexedAccessType" | "OptionalMemberExpression" | "ParenthesizedExpression" | "PipelineBareFunction" | "PipelinePrimaryTopicReference" | "PipelineTopicExpression" | "Placeholder" | "PrivateName" | "Program" | "QualifiedTypeIdentifier" | "RecordExpression" | "RegExpLiteral" | "RegexLiteral" | "RestElement" | "RestProperty" | "ReturnStatement" | "SequenceExpression" | "SpreadElement" | "SpreadProperty" | "StaticBlock" | "StringLiteralTypeAnnotation" | "StringTypeAnnotation" | "Super" | "SwitchCase" | "SwitchStatement" | "SymbolTypeAnnotation" | "TSAnyKeyword" | "TSArrayType" | "TSAsExpression" | "TSBigIntKeyword" | "TSBooleanKeyword" | "TSCallSignatureDeclaration" | "TSConditionalType" | "TSConstructSignatureDeclaration" | "TSConstructorType" | "TSDeclareFunction" | "TSDeclareMethod" | "TSEnumDeclaration" | "TSEnumMember" | "TSExportAssignment" | "TSExpressionWithTypeArguments" | "TSExternalModuleReference" | "TSFunctionType" | "TSImportEqualsDeclaration" | "TSImportType" | "TSIndexSignature" | "TSIndexedAccessType" | "TSInferType" | "TSInstantiationExpression" | "TSInterfaceBody" | "TSInterfaceDeclaration" | "TSIntersectionType" | "TSIntrinsicKeyword" | "TSLiteralType" | "TSMappedType" | "TSMethodSignature" | "TSModuleBlock" | "TSModuleDeclaration" | "TSNamedTupleMember" | "TSNamespaceExportDeclaration" | "TSNeverKeyword" | "TSNonNullExpression" | "TSNullKeyword" | "TSNumberKeyword" | "TSObjectKeyword" | "TSOptionalType" | "TSParameterProperty" | "TSParenthesizedType" | "TSPropertySignature" | "TSQualifiedName" | "TSRestType" | "TSSatisfiesExpression" | "TSStringKeyword" | "TSSymbolKeyword" | "TSThisType" | "TSTupleType" | "TSTypeAliasDeclaration" | "TSTypeAnnotation" | "TSTypeAssertion" | "TSTypeLiteral" | "TSTypeOperator" | "TSTypeParameter" | "TSTypeParameterDeclaration" | "TSTypeParameterInstantiation" | "TSTypePredicate" | "TSTypeQuery" | "TSTypeReference" | "TSUndefinedKeyword" | "TSUnionType" | "TSUnknownKeyword" | "TSVoidKeyword" | "TaggedTemplateExpression" | "TemplateElement" | "TemplateLiteral" | "ThisExpression" | "ThisTypeAnnotation" | "ThrowStatement" | "TopicReference" | "TryStatement" | "TupleExpression" | "TupleTypeAnnotation" | "TypeAlias" | "TypeAnnotation" | "TypeCastExpression" | "TypeParameter" | "TypeParameterDeclaration" | "TypeParameterInstantiation" | "TypeofTypeAnnotation" | "UnaryExpression" | "UnionTypeAnnotation" | "UpdateExpression" | "V8IntrinsicIdentifier" | "VariableDeclaration" | "VariableDeclarator" | "Variance" | "VoidTypeAnnotation" | "WhileStatement" | "WithStatement" | "YieldExpression" | keyof Aliases)[];
declare const DECLARATION_TYPES: ("StringLiteral" | "NumericLiteral" | "JSXText" | "JSXIdentifier" | "AnyTypeAnnotation" | "ArgumentPlaceholder" | "ArrayExpression" | "ArrayPattern" | "ArrayTypeAnnotation" | "ArrowFunctionExpression" | "AssignmentExpression" | "AssignmentPattern" | "AwaitExpression" | "BigIntLiteral" | "BinaryExpression" | "BindExpression" | "BlockStatement" | "BooleanLiteral" | "BooleanLiteralTypeAnnotation" | "BooleanTypeAnnotation" | "BreakStatement" | "CallExpression" | "CatchClause" | "ClassAccessorProperty" | "ClassBody" | "ClassDeclaration" | "ClassExpression" | "ClassImplements" | "ClassMethod" | "ClassPrivateMethod" | "ClassPrivateProperty" | "ClassProperty" | "ConditionalExpression" | "ContinueStatement" | "DebuggerStatement" | "DecimalLiteral" | "DeclareClass" | "DeclareExportAllDeclaration" | "DeclareExportDeclaration" | "DeclareFunction" | "DeclareInterface" | "DeclareModule" | "DeclareModuleExports" | "DeclareOpaqueType" | "DeclareTypeAlias" | "DeclareVariable" | "DeclaredPredicate" | "Decorator" | "Directive" | "DirectiveLiteral" | "DoExpression" | "DoWhileStatement" | "EmptyStatement" | "EmptyTypeAnnotation" | "EnumBooleanBody" | "EnumBooleanMember" | "EnumDeclaration" | "EnumDefaultedMember" | "EnumNumberBody" | "EnumNumberMember" | "EnumStringBody" | "EnumStringMember" | "EnumSymbolBody" | "ExistsTypeAnnotation" | "ExportAllDeclaration" | "ExportDefaultDeclaration" | "ExportDefaultSpecifier" | "ExportNamedDeclaration" | "ExportNamespaceSpecifier" | "ExportSpecifier" | "ExpressionStatement" | "File" | "ForInStatement" | "ForOfStatement" | "ForStatement" | "FunctionDeclaration" | "FunctionExpression" | "FunctionTypeAnnotation" | "FunctionTypeParam" | "GenericTypeAnnotation" | "Identifier" | "IfStatement" | "Import" | "ImportAttribute" | "ImportDeclaration" | "ImportDefaultSpecifier" | "ImportExpression" | "ImportNamespaceSpecifier" | "ImportSpecifier" | "IndexedAccessType" | "InferredPredicate" | "InterfaceDeclaration" | "InterfaceExtends" | "InterfaceTypeAnnotation" | "InterpreterDirective" | "IntersectionTypeAnnotation" | "JSXAttribute" | "JSXClosingElement" | "JSXClosingFragment" | "JSXElement" | "JSXEmptyExpression" | "JSXExpressionContainer" | "JSXFragment" | "JSXMemberExpression" | "JSXNamespacedName" | "JSXOpeningElement" | "JSXOpeningFragment" | "JSXSpreadAttribute" | "JSXSpreadChild" | "LabeledStatement" | "LogicalExpression" | "MemberExpression" | "MetaProperty" | "MixedTypeAnnotation" | "ModuleExpression" | "NewExpression" | "Noop" | "NullLiteral" | "NullLiteralTypeAnnotation" | "NullableTypeAnnotation" | "NumberLiteral" | "NumberLiteralTypeAnnotation" | "NumberTypeAnnotation" | "ObjectExpression" | "ObjectMethod" | "ObjectPattern" | "ObjectProperty" | "ObjectTypeAnnotation" | "ObjectTypeCallProperty" | "ObjectTypeIndexer" | "ObjectTypeInternalSlot" | "ObjectTypeProperty" | "ObjectTypeSpreadProperty" | "OpaqueType" | "OptionalCallExpression" | "OptionalIndexedAccessType" | "OptionalMemberExpression" | "ParenthesizedExpression" | "PipelineBareFunction" | "PipelinePrimaryTopicReference" | "PipelineTopicExpression" | "Placeholder" | "PrivateName" | "Program" | "QualifiedTypeIdentifier" | "RecordExpression" | "RegExpLiteral" | "RegexLiteral" | "RestElement" | "RestProperty" | "ReturnStatement" | "SequenceExpression" | "SpreadElement" | "SpreadProperty" | "StaticBlock" | "StringLiteralTypeAnnotation" | "StringTypeAnnotation" | "Super" | "SwitchCase" | "SwitchStatement" | "SymbolTypeAnnotation" | "TSAnyKeyword" | "TSArrayType" | "TSAsExpression" | "TSBigIntKeyword" | "TSBooleanKeyword" | "TSCallSignatureDeclaration" | "TSConditionalType" | "TSConstructSignatureDeclaration" | "TSConstructorType" | "TSDeclareFunction" | "TSDeclareMethod" | "TSEnumDeclaration" | "TSEnumMember" | "TSExportAssignment" | "TSExpressionWithTypeArguments" | "TSExternalModuleReference" | "TSFunctionType" | "TSImportEqualsDeclaration" | "TSImportType" | "TSIndexSignature" | "TSIndexedAccessType" | "TSInferType" | "TSInstantiationExpression" | "TSInterfaceBody" | "TSInterfaceDeclaration" | "TSIntersectionType" | "TSIntrinsicKeyword" | "TSLiteralType" | "TSMappedType" | "TSMethodSignature" | "TSModuleBlock" | "TSModuleDeclaration" | "TSNamedTupleMember" | "TSNamespaceExportDeclaration" | "TSNeverKeyword" | "TSNonNullExpression" | "TSNullKeyword" | "TSNumberKeyword" | "TSObjectKeyword" | "TSOptionalType" | "TSParameterProperty" | "TSParenthesizedType" | "TSPropertySignature" | "TSQualifiedName" | "TSRestType" | "TSSatisfiesExpression" | "TSStringKeyword" | "TSSymbolKeyword" | "TSThisType" | "TSTupleType" | "TSTypeAliasDeclaration" | "TSTypeAnnotation" | "TSTypeAssertion" | "TSTypeLiteral" | "TSTypeOperator" | "TSTypeParameter" | "TSTypeParameterDeclaration" | "TSTypeParameterInstantiation" | "TSTypePredicate" | "TSTypeQuery" | "TSTypeReference" | "TSUndefinedKeyword" | "TSUnionType" | "TSUnknownKeyword" | "TSVoidKeyword" | "TaggedTemplateExpression" | "TemplateElement" | "TemplateLiteral" | "ThisExpression" | "ThisTypeAnnotation" | "ThrowStatement" | "TopicReference" | "TryStatement" | "TupleExpression" | "TupleTypeAnnotation" | "TypeAlias" | "TypeAnnotation" | "TypeCastExpression" | "TypeParameter" | "TypeParameterDeclaration" | "TypeParameterInstantiation" | "TypeofTypeAnnotation" | "UnaryExpression" | "UnionTypeAnnotation" | "UpdateExpression" | "V8IntrinsicIdentifier" | "VariableDeclaration" | "VariableDeclarator" | "Variance" | "VoidTypeAnnotation" | "WhileStatement" | "WithStatement" | "YieldExpression" | keyof Aliases)[];
declare const PATTERNLIKE_TYPES: ("StringLiteral" | "NumericLiteral" | "JSXText" | "JSXIdentifier" | "AnyTypeAnnotation" | "ArgumentPlaceholder" | "ArrayExpression" | "ArrayPattern" | "ArrayTypeAnnotation" | "ArrowFunctionExpression" | "AssignmentExpression" | "AssignmentPattern" | "AwaitExpression" | "BigIntLiteral" | "BinaryExpression" | "BindExpression" | "BlockStatement" | "BooleanLiteral" | "BooleanLiteralTypeAnnotation" | "BooleanTypeAnnotation" | "BreakStatement" | "CallExpression" | "CatchClause" | "ClassAccessorProperty" | "ClassBody" | "ClassDeclaration" | "ClassExpression" | "ClassImplements" | "ClassMethod" | "ClassPrivateMethod" | "ClassPrivateProperty" | "ClassProperty" | "ConditionalExpression" | "ContinueStatement" | "DebuggerStatement" | "DecimalLiteral" | "DeclareClass" | "DeclareExportAllDeclaration" | "DeclareExportDeclaration" | "DeclareFunction" | "DeclareInterface" | "DeclareModule" | "DeclareModuleExports" | "DeclareOpaqueType" | "DeclareTypeAlias" | "DeclareVariable" | "DeclaredPredicate" | "Decorator" | "Directive" | "DirectiveLiteral" | "DoExpression" | "DoWhileStatement" | "EmptyStatement" | "EmptyTypeAnnotation" | "EnumBooleanBody" | "EnumBooleanMember" | "EnumDeclaration" | "EnumDefaultedMember" | "EnumNumberBody" | "EnumNumberMember" | "EnumStringBody" | "EnumStringMember" | "EnumSymbolBody" | "ExistsTypeAnnotation" | "ExportAllDeclaration" | "ExportDefaultDeclaration" | "ExportDefaultSpecifier" | "ExportNamedDeclaration" | "ExportNamespaceSpecifier" | "ExportSpecifier" | "ExpressionStatement" | "File" | "ForInStatement" | "ForOfStatement" | "ForStatement" | "FunctionDeclaration" | "FunctionExpression" | "FunctionTypeAnnotation" | "FunctionTypeParam" | "GenericTypeAnnotation" | "Identifier" | "IfStatement" | "Import" | "ImportAttribute" | "ImportDeclaration" | "ImportDefaultSpecifier" | "ImportExpression" | "ImportNamespaceSpecifier" | "ImportSpecifier" | "IndexedAccessType" | "InferredPredicate" | "InterfaceDeclaration" | "InterfaceExtends" | "InterfaceTypeAnnotation" | "InterpreterDirective" | "IntersectionTypeAnnotation" | "JSXAttribute" | "JSXClosingElement" | "JSXClosingFragment" | "JSXElement" | "JSXEmptyExpression" | "JSXExpressionContainer" | "JSXFragment" | "JSXMemberExpression" | "JSXNamespacedName" | "JSXOpeningElement" | "JSXOpeningFragment" | "JSXSpreadAttribute" | "JSXSpreadChild" | "LabeledStatement" | "LogicalExpression" | "MemberExpression" | "MetaProperty" | "MixedTypeAnnotation" | "ModuleExpression" | "NewExpression" | "Noop" | "NullLiteral" | "NullLiteralTypeAnnotation" | "NullableTypeAnnotation" | "NumberLiteral" | "NumberLiteralTypeAnnotation" | "NumberTypeAnnotation" | "ObjectExpression" | "ObjectMethod" | "ObjectPattern" | "ObjectProperty" | "ObjectTypeAnnotation" | "ObjectTypeCallProperty" | "ObjectTypeIndexer" | "ObjectTypeInternalSlot" | "ObjectTypeProperty" | "ObjectTypeSpreadProperty" | "OpaqueType" | "OptionalCallExpression" | "OptionalIndexedAccessType" | "OptionalMemberExpression" | "ParenthesizedExpression" | "PipelineBareFunction" | "PipelinePrimaryTopicReference" | "PipelineTopicExpression" | "Placeholder" | "PrivateName" | "Program" | "QualifiedTypeIdentifier" | "RecordExpression" | "RegExpLiteral" | "RegexLiteral" | "RestElement" | "RestProperty" | "ReturnStatement" | "SequenceExpression" | "SpreadElement" | "SpreadProperty" | "StaticBlock" | "StringLiteralTypeAnnotation" | "StringTypeAnnotation" | "Super" | "SwitchCase" | "SwitchStatement" | "SymbolTypeAnnotation" | "TSAnyKeyword" | "TSArrayType" | "TSAsExpression" | "TSBigIntKeyword" | "TSBooleanKeyword" | "TSCallSignatureDeclaration" | "TSConditionalType" | "TSConstructSignatureDeclaration" | "TSConstructorType" | "TSDeclareFunction" | "TSDeclareMethod" | "TSEnumDeclaration" | "TSEnumMember" | "TSExportAssignment" | "TSExpressionWithTypeArguments" | "TSExternalModuleReference" | "TSFunctionType" | "TSImportEqualsDeclaration" | "TSImportType" | "TSIndexSignature" | "TSIndexedAccessType" | "TSInferType" | "TSInstantiationExpression" | "TSInterfaceBody" | "TSInterfaceDeclaration" | "TSIntersectionType" | "TSIntrinsicKeyword" | "TSLiteralType" | "TSMappedType" | "TSMethodSignature" | "TSModuleBlock" | "TSModuleDeclaration" | "TSNamedTupleMember" | "TSNamespaceExportDeclaration" | "TSNeverKeyword" | "TSNonNullExpression" | "TSNullKeyword" | "TSNumberKeyword" | "TSObjectKeyword" | "TSOptionalType" | "TSParameterProperty" | "TSParenthesizedType" | "TSPropertySignature" | "TSQualifiedName" | "TSRestType" | "TSSatisfiesExpression" | "TSStringKeyword" | "TSSymbolKeyword" | "TSThisType" | "TSTupleType" | "TSTypeAliasDeclaration" | "TSTypeAnnotation" | "TSTypeAssertion" | "TSTypeLiteral" | "TSTypeOperator" | "TSTypeParameter" | "TSTypeParameterDeclaration" | "TSTypeParameterInstantiation" | "TSTypePredicate" | "TSTypeQuery" | "TSTypeReference" | "TSUndefinedKeyword" | "TSUnionType" | "TSUnknownKeyword" | "TSVoidKeyword" | "TaggedTemplateExpression" | "TemplateElement" | "TemplateLiteral" | "ThisExpression" | "ThisTypeAnnotation" | "ThrowStatement" | "TopicReference" | "TryStatement" | "TupleExpression" | "TupleTypeAnnotation" | "TypeAlias" | "TypeAnnotation" | "TypeCastExpression" | "TypeParameter" | "TypeParameterDeclaration" | "TypeParameterInstantiation" | "TypeofTypeAnnotation" | "UnaryExpression" | "UnionTypeAnnotation" | "UpdateExpression" | "V8IntrinsicIdentifier" | "VariableDeclaration" | "VariableDeclarator" | "Variance" | "VoidTypeAnnotation" | "WhileStatement" | "WithStatement" | "YieldExpression" | keyof Aliases)[];
declare const LVAL_TYPES: ("StringLiteral" | "NumericLiteral" | "JSXText" | "JSXIdentifier" | "AnyTypeAnnotation" | "ArgumentPlaceholder" | "ArrayExpression" | "ArrayPattern" | "ArrayTypeAnnotation" | "ArrowFunctionExpression" | "AssignmentExpression" | "AssignmentPattern" | "AwaitExpression" | "BigIntLiteral" | "BinaryExpression" | "BindExpression" | "BlockStatement" | "BooleanLiteral" | "BooleanLiteralTypeAnnotation" | "BooleanTypeAnnotation" | "BreakStatement" | "CallExpression" | "CatchClause" | "ClassAccessorProperty" | "ClassBody" | "ClassDeclaration" | "ClassExpression" | "ClassImplements" | "ClassMethod" | "ClassPrivateMethod" | "ClassPrivateProperty" | "ClassProperty" | "ConditionalExpression" | "ContinueStatement" | "DebuggerStatement" | "DecimalLiteral" | "DeclareClass" | "DeclareExportAllDeclaration" | "DeclareExportDeclaration" | "DeclareFunction" | "DeclareInterface" | "DeclareModule" | "DeclareModuleExports" | "DeclareOpaqueType" | "DeclareTypeAlias" | "DeclareVariable" | "DeclaredPredicate" | "Decorator" | "Directive" | "DirectiveLiteral" | "DoExpression" | "DoWhileStatement" | "EmptyStatement" | "EmptyTypeAnnotation" | "EnumBooleanBody" | "EnumBooleanMember" | "EnumDeclaration" | "EnumDefaultedMember" | "EnumNumberBody" | "EnumNumberMember" | "EnumStringBody" | "EnumStringMember" | "EnumSymbolBody" | "ExistsTypeAnnotation" | "ExportAllDeclaration" | "ExportDefaultDeclaration" | "ExportDefaultSpecifier" | "ExportNamedDeclaration" | "ExportNamespaceSpecifier" | "ExportSpecifier" | "ExpressionStatement" | "File" | "ForInStatement" | "ForOfStatement" | "ForStatement" | "FunctionDeclaration" | "FunctionExpression" | "FunctionTypeAnnotation" | "FunctionTypeParam" | "GenericTypeAnnotation" | "Identifier" | "IfStatement" | "Import" | "ImportAttribute" | "ImportDeclaration" | "ImportDefaultSpecifier" | "ImportExpression" | "ImportNamespaceSpecifier" | "ImportSpecifier" | "IndexedAccessType" | "InferredPredicate" | "InterfaceDeclaration" | "InterfaceExtends" | "InterfaceTypeAnnotation" | "InterpreterDirective" | "IntersectionTypeAnnotation" | "JSXAttribute" | "JSXClosingElement" | "JSXClosingFragment" | "JSXElement" | "JSXEmptyExpression" | "JSXExpressionContainer" | "JSXFragment" | "JSXMemberExpression" | "JSXNamespacedName" | "JSXOpeningElement" | "JSXOpeningFragment" | "JSXSpreadAttribute" | "JSXSpreadChild" | "LabeledStatement" | "LogicalExpression" | "MemberExpression" | "MetaProperty" | "MixedTypeAnnotation" | "ModuleExpression" | "NewExpression" | "Noop" | "NullLiteral" | "NullLiteralTypeAnnotation" | "NullableTypeAnnotation" | "NumberLiteral" | "NumberLiteralTypeAnnotation" | "NumberTypeAnnotation" | "ObjectExpression" | "ObjectMethod" | "ObjectPattern" | "ObjectProperty" | "ObjectTypeAnnotation" | "ObjectTypeCallProperty" | "ObjectTypeIndexer" | "ObjectTypeInternalSlot" | "ObjectTypeProperty" | "ObjectTypeSpreadProperty" | "OpaqueType" | "OptionalCallExpression" | "OptionalIndexedAccessType" | "OptionalMemberExpression" | "ParenthesizedExpression" | "PipelineBareFunction" | "PipelinePrimaryTopicReference" | "PipelineTopicExpression" | "Placeholder" | "PrivateName" | "Program" | "QualifiedTypeIdentifier" | "RecordExpression" | "RegExpLiteral" | "RegexLiteral" | "RestElement" | "RestProperty" | "ReturnStatement" | "SequenceExpression" | "SpreadElement" | "SpreadProperty" | "StaticBlock" | "StringLiteralTypeAnnotation" | "StringTypeAnnotation" | "Super" | "SwitchCase" | "SwitchStatement" | "SymbolTypeAnnotation" | "TSAnyKeyword" | "TSArrayType" | "TSAsExpression" | "TSBigIntKeyword" | "TSBooleanKeyword" | "TSCallSignatureDeclaration" | "TSConditionalType" | "TSConstructSignatureDeclaration" | "TSConstructorType" | "TSDeclareFunction" | "TSDeclareMethod" | "TSEnumDeclaration" | "TSEnumMember" | "TSExportAssignment" | "TSExpressionWithTypeArguments" | "TSExternalModuleReference" | "TSFunctionType" | "TSImportEqualsDeclaration" | "TSImportType" | "TSIndexSignature" | "TSIndexedAccessType" | "TSInferType" | "TSInstantiationExpression" | "TSInterfaceBody" | "TSInterfaceDeclaration" | "TSIntersectionType" | "TSIntrinsicKeyword" | "TSLiteralType" | "TSMappedType" | "TSMethodSignature" | "TSModuleBlock" | "TSModuleDeclaration" | "TSNamedTupleMember" | "TSNamespaceExportDeclaration" | "TSNeverKeyword" | "TSNonNullExpression" | "TSNullKeyword" | "TSNumberKeyword" | "TSObjectKeyword" | "TSOptionalType" | "TSParameterProperty" | "TSParenthesizedType" | "TSPropertySignature" | "TSQualifiedName" | "TSRestType" | "TSSatisfiesExpression" | "TSStringKeyword" | "TSSymbolKeyword" | "TSThisType" | "TSTupleType" | "TSTypeAliasDeclaration" | "TSTypeAnnotation" | "TSTypeAssertion" | "TSTypeLiteral" | "TSTypeOperator" | "TSTypeParameter" | "TSTypeParameterDeclaration" | "TSTypeParameterInstantiation" | "TSTypePredicate" | "TSTypeQuery" | "TSTypeReference" | "TSUndefinedKeyword" | "TSUnionType" | "TSUnknownKeyword" | "TSVoidKeyword" | "TaggedTemplateExpression" | "TemplateElement" | "TemplateLiteral" | "ThisExpression" | "ThisTypeAnnotation" | "ThrowStatement" | "TopicReference" | "TryStatement" | "TupleExpression" | "TupleTypeAnnotation" | "TypeAlias" | "TypeAnnotation" | "TypeCastExpression" | "TypeParameter" | "TypeParameterDeclaration" | "TypeParameterInstantiation" | "TypeofTypeAnnotation" | "UnaryExpression" | "UnionTypeAnnotation" | "UpdateExpression" | "V8IntrinsicIdentifier" | "VariableDeclaration" | "VariableDeclarator" | "Variance" | "VoidTypeAnnotation" | "WhileStatement" | "WithStatement" | "YieldExpression" | keyof Aliases)[];
declare const TSENTITYNAME_TYPES: ("StringLiteral" | "NumericLiteral" | "JSXText" | "JSXIdentifier" | "AnyTypeAnnotation" | "ArgumentPlaceholder" | "ArrayExpression" | "ArrayPattern" | "ArrayTypeAnnotation" | "ArrowFunctionExpression" | "AssignmentExpression" | "AssignmentPattern" | "AwaitExpression" | "BigIntLiteral" | "BinaryExpression" | "BindExpression" | "BlockStatement" | "BooleanLiteral" | "BooleanLiteralTypeAnnotation" | "BooleanTypeAnnotation" | "BreakStatement" | "CallExpression" | "CatchClause" | "ClassAccessorProperty" | "ClassBody" | "ClassDeclaration" | "ClassExpression" | "ClassImplements" | "ClassMethod" | "ClassPrivateMethod" | "ClassPrivateProperty" | "ClassProperty" | "ConditionalExpression" | "ContinueStatement" | "DebuggerStatement" | "DecimalLiteral" | "DeclareClass" | "DeclareExportAllDeclaration" | "DeclareExportDeclaration" | "DeclareFunction" | "DeclareInterface" | "DeclareModule" | "DeclareModuleExports" | "DeclareOpaqueType" | "DeclareTypeAlias" | "DeclareVariable" | "DeclaredPredicate" | "Decorator" | "Directive" | "DirectiveLiteral" | "DoExpression" | "DoWhileStatement" | "EmptyStatement" | "EmptyTypeAnnotation" | "EnumBooleanBody" | "EnumBooleanMember" | "EnumDeclaration" | "EnumDefaultedMember" | "EnumNumberBody" | "EnumNumberMember" | "EnumStringBody" | "EnumStringMember" | "EnumSymbolBody" | "ExistsTypeAnnotation" | "ExportAllDeclaration" | "ExportDefaultDeclaration" | "ExportDefaultSpecifier" | "ExportNamedDeclaration" | "ExportNamespaceSpecifier" | "ExportSpecifier" | "ExpressionStatement" | "File" | "ForInStatement" | "ForOfStatement" | "ForStatement" | "FunctionDeclaration" | "FunctionExpression" | "FunctionTypeAnnotation" | "FunctionTypeParam" | "GenericTypeAnnotation" | "Identifier" | "IfStatement" | "Import" | "ImportAttribute" | "ImportDeclaration" | "ImportDefaultSpecifier" | "ImportExpression" | "ImportNamespaceSpecifier" | "ImportSpecifier" | "IndexedAccessType" | "InferredPredicate" | "InterfaceDeclaration" | "InterfaceExtends" | "InterfaceTypeAnnotation" | "InterpreterDirective" | "IntersectionTypeAnnotation" | "JSXAttribute" | "JSXClosingElement" | "JSXClosingFragment" | "JSXElement" | "JSXEmptyExpression" | "JSXExpressionContainer" | "JSXFragment" | "JSXMemberExpression" | "JSXNamespacedName" | "JSXOpeningElement" | "JSXOpeningFragment" | "JSXSpreadAttribute" | "JSXSpreadChild" | "LabeledStatement" | "LogicalExpression" | "MemberExpression" | "MetaProperty" | "MixedTypeAnnotation" | "ModuleExpression" | "NewExpression" | "Noop" | "NullLiteral" | "NullLiteralTypeAnnotation" | "NullableTypeAnnotation" | "NumberLiteral" | "NumberLiteralTypeAnnotation" | "NumberTypeAnnotation" | "ObjectExpression" | "ObjectMethod" | "ObjectPattern" | "ObjectProperty" | "ObjectTypeAnnotation" | "ObjectTypeCallProperty" | "ObjectTypeIndexer" | "ObjectTypeInternalSlot" | "ObjectTypeProperty" | "ObjectTypeSpreadProperty" | "OpaqueType" | "OptionalCallExpression" | "OptionalIndexedAccessType" | "OptionalMemberExpression" | "ParenthesizedExpression" | "PipelineBareFunction" | "PipelinePrimaryTopicReference" | "PipelineTopicExpression" | "Placeholder" | "PrivateName" | "Program" | "QualifiedTypeIdentifier" | "RecordExpression" | "RegExpLiteral" | "RegexLiteral" | "RestElement" | "RestProperty" | "ReturnStatement" | "SequenceExpression" | "SpreadElement" | "SpreadProperty" | "StaticBlock" | "StringLiteralTypeAnnotation" | "StringTypeAnnotation" | "Super" | "SwitchCase" | "SwitchStatement" | "SymbolTypeAnnotation" | "TSAnyKeyword" | "TSArrayType" | "TSAsExpression" | "TSBigIntKeyword" | "TSBooleanKeyword" | "TSCallSignatureDeclaration" | "TSConditionalType" | "TSConstructSignatureDeclaration" | "TSConstructorType" | "TSDeclareFunction" | "TSDeclareMethod" | "TSEnumDeclaration" | "TSEnumMember" | "TSExportAssignment" | "TSExpressionWithTypeArguments" | "TSExternalModuleReference" | "TSFunctionType" | "TSImportEqualsDeclaration" | "TSImportType" | "TSIndexSignature" | "TSIndexedAccessType" | "TSInferType" | "TSInstantiationExpression" | "TSInterfaceBody" | "TSInterfaceDeclaration" | "TSIntersectionType" | "TSIntrinsicKeyword" | "TSLiteralType" | "TSMappedType" | "TSMethodSignature" | "TSModuleBlock" | "TSModuleDeclaration" | "TSNamedTupleMember" | "TSNamespaceExportDeclaration" | "TSNeverKeyword" | "TSNonNullExpression" | "TSNullKeyword" | "TSNumberKeyword" | "TSObjectKeyword" | "TSOptionalType" | "TSParameterProperty" | "TSParenthesizedType" | "TSPropertySignature" | "TSQualifiedName" | "TSRestType" | "TSSatisfiesExpression" | "TSStringKeyword" | "TSSymbolKeyword" | "TSThisType" | "TSTupleType" | "TSTypeAliasDeclaration" | "TSTypeAnnotation" | "TSTypeAssertion" | "TSTypeLiteral" | "TSTypeOperator" | "TSTypeParameter" | "TSTypeParameterDeclaration" | "TSTypeParameterInstantiation" | "TSTypePredicate" | "TSTypeQuery" | "TSTypeReference" | "TSUndefinedKeyword" | "TSUnionType" | "TSUnknownKeyword" | "TSVoidKeyword" | "TaggedTemplateExpression" | "TemplateElement" | "TemplateLiteral" | "ThisExpression" | "ThisTypeAnnotation" | "ThrowStatement" | "TopicReference" | "TryStatement" | "TupleExpression" | "TupleTypeAnnotation" | "TypeAlias" | "TypeAnnotation" | "TypeCastExpression" | "TypeParameter" | "TypeParameterDeclaration" | "TypeParameterInstantiation" | "TypeofTypeAnnotation" | "UnaryExpression" | "UnionTypeAnnotation" | "UpdateExpression" | "V8IntrinsicIdentifier" | "VariableDeclaration" | "VariableDeclarator" | "Variance" | "VoidTypeAnnotation" | "WhileStatement" | "WithStatement" | "YieldExpression" | keyof Aliases)[];
declare const LITERAL_TYPES: ("StringLiteral" | "NumericLiteral" | "JSXText" | "JSXIdentifier" | "AnyTypeAnnotation" | "ArgumentPlaceholder" | "ArrayExpression" | "ArrayPattern" | "ArrayTypeAnnotation" | "ArrowFunctionExpression" | "AssignmentExpression" | "AssignmentPattern" | "AwaitExpression" | "BigIntLiteral" | "BinaryExpression" | "BindExpression" | "BlockStatement" | "BooleanLiteral" | "BooleanLiteralTypeAnnotation" | "BooleanTypeAnnotation" | "BreakStatement" | "CallExpression" | "CatchClause" | "ClassAccessorProperty" | "ClassBody" | "ClassDeclaration" | "ClassExpression" | "ClassImplements" | "ClassMethod" | "ClassPrivateMethod" | "ClassPrivateProperty" | "ClassProperty" | "ConditionalExpression" | "ContinueStatement" | "DebuggerStatement" | "DecimalLiteral" | "DeclareClass" | "DeclareExportAllDeclaration" | "DeclareExportDeclaration" | "DeclareFunction" | "DeclareInterface" | "DeclareModule" | "DeclareModuleExports" | "DeclareOpaqueType" | "DeclareTypeAlias" | "DeclareVariable" | "DeclaredPredicate" | "Decorator" | "Directive" | "DirectiveLiteral" | "DoExpression" | "DoWhileStatement" | "EmptyStatement" | "EmptyTypeAnnotation" | "EnumBooleanBody" | "EnumBooleanMember" | "EnumDeclaration" | "EnumDefaultedMember" | "EnumNumberBody" | "EnumNumberMember" | "EnumStringBody" | "EnumStringMember" | "EnumSymbolBody" | "ExistsTypeAnnotation" | "ExportAllDeclaration" | "ExportDefaultDeclaration" | "ExportDefaultSpecifier" | "ExportNamedDeclaration" | "ExportNamespaceSpecifier" | "ExportSpecifier" | "ExpressionStatement" | "File" | "ForInStatement" | "ForOfStatement" | "ForStatement" | "FunctionDeclaration" | "FunctionExpression" | "FunctionTypeAnnotation" | "FunctionTypeParam" | "GenericTypeAnnotation" | "Identifier" | "IfStatement" | "Import" | "ImportAttribute" | "ImportDeclaration" | "ImportDefaultSpecifier" | "ImportExpression" | "ImportNamespaceSpecifier" | "ImportSpecifier" | "IndexedAccessType" | "InferredPredicate" | "InterfaceDeclaration" | "InterfaceExtends" | "InterfaceTypeAnnotation" | "InterpreterDirective" | "IntersectionTypeAnnotation" | "JSXAttribute" | "JSXClosingElement" | "JSXClosingFragment" | "JSXElement" | "JSXEmptyExpression" | "JSXExpressionContainer" | "JSXFragment" | "JSXMemberExpression" | "JSXNamespacedName" | "JSXOpeningElement" | "JSXOpeningFragment" | "JSXSpreadAttribute" | "JSXSpreadChild" | "LabeledStatement" | "LogicalExpression" | "MemberExpression" | "MetaProperty" | "MixedTypeAnnotation" | "ModuleExpression" | "NewExpression" | "Noop" | "NullLiteral" | "NullLiteralTypeAnnotation" | "NullableTypeAnnotation" | "NumberLiteral" | "NumberLiteralTypeAnnotation" | "NumberTypeAnnotation" | "ObjectExpression" | "ObjectMethod" | "ObjectPattern" | "ObjectProperty" | "ObjectTypeAnnotation" | "ObjectTypeCallProperty" | "ObjectTypeIndexer" | "ObjectTypeInternalSlot" | "ObjectTypeProperty" | "ObjectTypeSpreadProperty" | "OpaqueType" | "OptionalCallExpression" | "OptionalIndexedAccessType" | "OptionalMemberExpression" | "ParenthesizedExpression" | "PipelineBareFunction" | "PipelinePrimaryTopicReference" | "PipelineTopicExpression" | "Placeholder" | "PrivateName" | "Program" | "QualifiedTypeIdentifier" | "RecordExpression" | "RegExpLiteral" | "RegexLiteral" | "RestElement" | "RestProperty" | "ReturnStatement" | "SequenceExpression" | "SpreadElement" | "SpreadProperty" | "StaticBlock" | "StringLiteralTypeAnnotation" | "StringTypeAnnotation" | "Super" | "SwitchCase" | "SwitchStatement" | "SymbolTypeAnnotation" | "TSAnyKeyword" | "TSArrayType" | "TSAsExpression" | "TSBigIntKeyword" | "TSBooleanKeyword" | "TSCallSignatureDeclaration" | "TSConditionalType" | "TSConstructSignatureDeclaration" | "TSConstructorType" | "TSDeclareFunction" | "TSDeclareMethod" | "TSEnumDeclaration" | "TSEnumMember" | "TSExportAssignment" | "TSExpressionWithTypeArguments" | "TSExternalModuleReference" | "TSFunctionType" | "TSImportEqualsDeclaration" | "TSImportType" | "TSIndexSignature" | "TSIndexedAccessType" | "TSInferType" | "TSInstantiationExpression" | "TSInterfaceBody" | "TSInterfaceDeclaration" | "TSIntersectionType" | "TSIntrinsicKeyword" | "TSLiteralType" | "TSMappedType" | "TSMethodSignature" | "TSModuleBlock" | "TSModuleDeclaration" | "TSNamedTupleMember" | "TSNamespaceExportDeclaration" | "TSNeverKeyword" | "TSNonNullExpression" | "TSNullKeyword" | "TSNumberKeyword" | "TSObjectKeyword" | "TSOptionalType" | "TSParameterProperty" | "TSParenthesizedType" | "TSPropertySignature" | "TSQualifiedName" | "TSRestType" | "TSSatisfiesExpression" | "TSStringKeyword" | "TSSymbolKeyword" | "TSThisType" | "TSTupleType" | "TSTypeAliasDeclaration" | "TSTypeAnnotation" | "TSTypeAssertion" | "TSTypeLiteral" | "TSTypeOperator" | "TSTypeParameter" | "TSTypeParameterDeclaration" | "TSTypeParameterInstantiation" | "TSTypePredicate" | "TSTypeQuery" | "TSTypeReference" | "TSUndefinedKeyword" | "TSUnionType" | "TSUnknownKeyword" | "TSVoidKeyword" | "TaggedTemplateExpression" | "TemplateElement" | "TemplateLiteral" | "ThisExpression" | "ThisTypeAnnotation" | "ThrowStatement" | "TopicReference" | "TryStatement" | "TupleExpression" | "TupleTypeAnnotation" | "TypeAlias" | "TypeAnnotation" | "TypeCastExpression" | "TypeParameter" | "TypeParameterDeclaration" | "TypeParameterInstantiation" | "TypeofTypeAnnotation" | "UnaryExpression" | "UnionTypeAnnotation" | "UpdateExpression" | "V8IntrinsicIdentifier" | "VariableDeclaration" | "VariableDeclarator" | "Variance" | "VoidTypeAnnotation" | "WhileStatement" | "WithStatement" | "YieldExpression" | keyof Aliases)[];
declare const IMMUTABLE_TYPES: ("StringLiteral" | "NumericLiteral" | "JSXText" | "JSXIdentifier" | "AnyTypeAnnotation" | "ArgumentPlaceholder" | "ArrayExpression" | "ArrayPattern" | "ArrayTypeAnnotation" | "ArrowFunctionExpression" | "AssignmentExpression" | "AssignmentPattern" | "AwaitExpression" | "BigIntLiteral" | "BinaryExpression" | "BindExpression" | "BlockStatement" | "BooleanLiteral" | "BooleanLiteralTypeAnnotation" | "BooleanTypeAnnotation" | "BreakStatement" | "CallExpression" | "CatchClause" | "ClassAccessorProperty" | "ClassBody" | "ClassDeclaration" | "ClassExpression" | "ClassImplements" | "ClassMethod" | "ClassPrivateMethod" | "ClassPrivateProperty" | "ClassProperty" | "ConditionalExpression" | "ContinueStatement" | "DebuggerStatement" | "DecimalLiteral" | "DeclareClass" | "DeclareExportAllDeclaration" | "DeclareExportDeclaration" | "DeclareFunction" | "DeclareInterface" | "DeclareModule" | "DeclareModuleExports" | "DeclareOpaqueType" | "DeclareTypeAlias" | "DeclareVariable" | "DeclaredPredicate" | "Decorator" | "Directive" | "DirectiveLiteral" | "DoExpression" | "DoWhileStatement" | "EmptyStatement" | "EmptyTypeAnnotation" | "EnumBooleanBody" | "EnumBooleanMember" | "EnumDeclaration" | "EnumDefaultedMember" | "EnumNumberBody" | "EnumNumberMember" | "EnumStringBody" | "EnumStringMember" | "EnumSymbolBody" | "ExistsTypeAnnotation" | "ExportAllDeclaration" | "ExportDefaultDeclaration" | "ExportDefaultSpecifier" | "ExportNamedDeclaration" | "ExportNamespaceSpecifier" | "ExportSpecifier" | "ExpressionStatement" | "File" | "ForInStatement" | "ForOfStatement" | "ForStatement" | "FunctionDeclaration" | "FunctionExpression" | "FunctionTypeAnnotation" | "FunctionTypeParam" | "GenericTypeAnnotation" | "Identifier" | "IfStatement" | "Import" | "ImportAttribute" | "ImportDeclaration" | "ImportDefaultSpecifier" | "ImportExpression" | "ImportNamespaceSpecifier" | "ImportSpecifier" | "IndexedAccessType" | "InferredPredicate" | "InterfaceDeclaration" | "InterfaceExtends" | "InterfaceTypeAnnotation" | "InterpreterDirective" | "IntersectionTypeAnnotation" | "JSXAttribute" | "JSXClosingElement" | "JSXClosingFragment" | "JSXElement" | "JSXEmptyExpression" | "JSXExpressionContainer" | "JSXFragment" | "JSXMemberExpression" | "JSXNamespacedName" | "JSXOpeningElement" | "JSXOpeningFragment" | "JSXSpreadAttribute" | "JSXSpreadChild" | "LabeledStatement" | "LogicalExpression" | "MemberExpression" | "MetaProperty" | "MixedTypeAnnotation" | "ModuleExpression" | "NewExpression" | "Noop" | "NullLiteral" | "NullLiteralTypeAnnotation" | "NullableTypeAnnotation" | "NumberLiteral" | "NumberLiteralTypeAnnotation" | "NumberTypeAnnotation" | "ObjectExpression" | "ObjectMethod" | "ObjectPattern" | "ObjectProperty" | "ObjectTypeAnnotation" | "ObjectTypeCallProperty" | "ObjectTypeIndexer" | "ObjectTypeInternalSlot" | "ObjectTypeProperty" | "ObjectTypeSpreadProperty" | "OpaqueType" | "OptionalCallExpression" | "OptionalIndexedAccessType" | "OptionalMemberExpression" | "ParenthesizedExpression" | "PipelineBareFunction" | "PipelinePrimaryTopicReference" | "PipelineTopicExpression" | "Placeholder" | "PrivateName" | "Program" | "QualifiedTypeIdentifier" | "RecordExpression" | "RegExpLiteral" | "RegexLiteral" | "RestElement" | "RestProperty" | "ReturnStatement" | "SequenceExpression" | "SpreadElement" | "SpreadProperty" | "StaticBlock" | "StringLiteralTypeAnnotation" | "StringTypeAnnotation" | "Super" | "SwitchCase" | "SwitchStatement" | "SymbolTypeAnnotation" | "TSAnyKeyword" | "TSArrayType" | "TSAsExpression" | "TSBigIntKeyword" | "TSBooleanKeyword" | "TSCallSignatureDeclaration" | "TSConditionalType" | "TSConstructSignatureDeclaration" | "TSConstructorType" | "TSDeclareFunction" | "TSDeclareMethod" | "TSEnumDeclaration" | "TSEnumMember" | "TSExportAssignment" | "TSExpressionWithTypeArguments" | "TSExternalModuleReference" | "TSFunctionType" | "TSImportEqualsDeclaration" | "TSImportType" | "TSIndexSignature" | "TSIndexedAccessType" | "TSInferType" | "TSInstantiationExpression" | "TSInterfaceBody" | "TSInterfaceDeclaration" | "TSIntersectionType" | "TSIntrinsicKeyword" | "TSLiteralType" | "TSMappedType" | "TSMethodSignature" | "TSModuleBlock" | "TSModuleDeclaration" | "TSNamedTupleMember" | "TSNamespaceExportDeclaration" | "TSNeverKeyword" | "TSNonNullExpression" | "TSNullKeyword" | "TSNumberKeyword" | "TSObjectKeyword" | "TSOptionalType" | "TSParameterProperty" | "TSParenthesizedType" | "TSPropertySignature" | "TSQualifiedName" | "TSRestType" | "TSSatisfiesExpression" | "TSStringKeyword" | "TSSymbolKeyword" | "TSThisType" | "TSTupleType" | "TSTypeAliasDeclaration" | "TSTypeAnnotation" | "TSTypeAssertion" | "TSTypeLiteral" | "TSTypeOperator" | "TSTypeParameter" | "TSTypeParameterDeclaration" | "TSTypeParameterInstantiation" | "TSTypePredicate" | "TSTypeQuery" | "TSTypeReference" | "TSUndefinedKeyword" | "TSUnionType" | "TSUnknownKeyword" | "TSVoidKeyword" | "TaggedTemplateExpression" | "TemplateElement" | "TemplateLiteral" | "ThisExpression" | "ThisTypeAnnotation" | "ThrowStatement" | "TopicReference" | "TryStatement" | "TupleExpression" | "TupleTypeAnnotation" | "TypeAlias" | "TypeAnnotation" | "TypeCastExpression" | "TypeParameter" | "TypeParameterDeclaration" | "TypeParameterInstantiation" | "TypeofTypeAnnotation" | "UnaryExpression" | "UnionTypeAnnotation" | "UpdateExpression" | "V8IntrinsicIdentifier" | "VariableDeclaration" | "VariableDeclarator" | "Variance" | "VoidTypeAnnotation" | "WhileStatement" | "WithStatement" | "YieldExpression" | keyof Aliases)[];
declare const USERWHITESPACABLE_TYPES: ("StringLiteral" | "NumericLiteral" | "JSXText" | "JSXIdentifier" | "AnyTypeAnnotation" | "ArgumentPlaceholder" | "ArrayExpression" | "ArrayPattern" | "ArrayTypeAnnotation" | "ArrowFunctionExpression" | "AssignmentExpression" | "AssignmentPattern" | "AwaitExpression" | "BigIntLiteral" | "BinaryExpression" | "BindExpression" | "BlockStatement" | "BooleanLiteral" | "BooleanLiteralTypeAnnotation" | "BooleanTypeAnnotation" | "BreakStatement" | "CallExpression" | "CatchClause" | "ClassAccessorProperty" | "ClassBody" | "ClassDeclaration" | "ClassExpression" | "ClassImplements" | "ClassMethod" | "ClassPrivateMethod" | "ClassPrivateProperty" | "ClassProperty" | "ConditionalExpression" | "ContinueStatement" | "DebuggerStatement" | "DecimalLiteral" | "DeclareClass" | "DeclareExportAllDeclaration" | "DeclareExportDeclaration" | "DeclareFunction" | "DeclareInterface" | "DeclareModule" | "DeclareModuleExports" | "DeclareOpaqueType" | "DeclareTypeAlias" | "DeclareVariable" | "DeclaredPredicate" | "Decorator" | "Directive" | "DirectiveLiteral" | "DoExpression" | "DoWhileStatement" | "EmptyStatement" | "EmptyTypeAnnotation" | "EnumBooleanBody" | "EnumBooleanMember" | "EnumDeclaration" | "EnumDefaultedMember" | "EnumNumberBody" | "EnumNumberMember" | "EnumStringBody" | "EnumStringMember" | "EnumSymbolBody" | "ExistsTypeAnnotation" | "ExportAllDeclaration" | "ExportDefaultDeclaration" | "ExportDefaultSpecifier" | "ExportNamedDeclaration" | "ExportNamespaceSpecifier" | "ExportSpecifier" | "ExpressionStatement" | "File" | "ForInStatement" | "ForOfStatement" | "ForStatement" | "FunctionDeclaration" | "FunctionExpression" | "FunctionTypeAnnotation" | "FunctionTypeParam" | "GenericTypeAnnotation" | "Identifier" | "IfStatement" | "Import" | "ImportAttribute" | "ImportDeclaration" | "ImportDefaultSpecifier" | "ImportExpression" | "ImportNamespaceSpecifier" | "ImportSpecifier" | "IndexedAccessType" | "InferredPredicate" | "InterfaceDeclaration" | "InterfaceExtends" | "InterfaceTypeAnnotation" | "InterpreterDirective" | "IntersectionTypeAnnotation" | "JSXAttribute" | "JSXClosingElement" | "JSXClosingFragment" | "JSXElement" | "JSXEmptyExpression" | "JSXExpressionContainer" | "JSXFragment" | "JSXMemberExpression" | "JSXNamespacedName" | "JSXOpeningElement" | "JSXOpeningFragment" | "JSXSpreadAttribute" | "JSXSpreadChild" | "LabeledStatement" | "LogicalExpression" | "MemberExpression" | "MetaProperty" | "MixedTypeAnnotation" | "ModuleExpression" | "NewExpression" | "Noop" | "NullLiteral" | "NullLiteralTypeAnnotation" | "NullableTypeAnnotation" | "NumberLiteral" | "NumberLiteralTypeAnnotation" | "NumberTypeAnnotation" | "ObjectExpression" | "ObjectMethod" | "ObjectPattern" | "ObjectProperty" | "ObjectTypeAnnotation" | "ObjectTypeCallProperty" | "ObjectTypeIndexer" | "ObjectTypeInternalSlot" | "ObjectTypeProperty" | "ObjectTypeSpreadProperty" | "OpaqueType" | "OptionalCallExpression" | "OptionalIndexedAccessType" | "OptionalMemberExpression" | "ParenthesizedExpression" | "PipelineBareFunction" | "PipelinePrimaryTopicReference" | "PipelineTopicExpression" | "Placeholder" | "PrivateName" | "Program" | "QualifiedTypeIdentifier" | "RecordExpression" | "RegExpLiteral" | "RegexLiteral" | "RestElement" | "RestProperty" | "ReturnStatement" | "SequenceExpression" | "SpreadElement" | "SpreadProperty" | "StaticBlock" | "StringLiteralTypeAnnotation" | "StringTypeAnnotation" | "Super" | "SwitchCase" | "SwitchStatement" | "SymbolTypeAnnotation" | "TSAnyKeyword" | "TSArrayType" | "TSAsExpression" | "TSBigIntKeyword" | "TSBooleanKeyword" | "TSCallSignatureDeclaration" | "TSConditionalType" | "TSConstructSignatureDeclaration" | "TSConstructorType" | "TSDeclareFunction" | "TSDeclareMethod" | "TSEnumDeclaration" | "TSEnumMember" | "TSExportAssignment" | "TSExpressionWithTypeArguments" | "TSExternalModuleReference" | "TSFunctionType" | "TSImportEqualsDeclaration" | "TSImportType" | "TSIndexSignature" | "TSIndexedAccessType" | "TSInferType" | "TSInstantiationExpression" | "TSInterfaceBody" | "TSInterfaceDeclaration" | "TSIntersectionType" | "TSIntrinsicKeyword" | "TSLiteralType" | "TSMappedType" | "TSMethodSignature" | "TSModuleBlock" | "TSModuleDeclaration" | "TSNamedTupleMember" | "TSNamespaceExportDeclaration" | "TSNeverKeyword" | "TSNonNullExpression" | "TSNullKeyword" | "TSNumberKeyword" | "TSObjectKeyword" | "TSOptionalType" | "TSParameterProperty" | "TSParenthesizedType" | "TSPropertySignature" | "TSQualifiedName" | "TSRestType" | "TSSatisfiesExpression" | "TSStringKeyword" | "TSSymbolKeyword" | "TSThisType" | "TSTupleType" | "TSTypeAliasDeclaration" | "TSTypeAnnotation" | "TSTypeAssertion" | "TSTypeLiteral" | "TSTypeOperator" | "TSTypeParameter" | "TSTypeParameterDeclaration" | "TSTypeParameterInstantiation" | "TSTypePredicate" | "TSTypeQuery" | "TSTypeReference" | "TSUndefinedKeyword" | "TSUnionType" | "TSUnknownKeyword" | "TSVoidKeyword" | "TaggedTemplateExpression" | "TemplateElement" | "TemplateLiteral" | "ThisExpression" | "ThisTypeAnnotation" | "ThrowStatement" | "TopicReference" | "TryStatement" | "TupleExpression" | "TupleTypeAnnotation" | "TypeAlias" | "TypeAnnotation" | "TypeCastExpression" | "TypeParameter" | "TypeParameterDeclaration" | "TypeParameterInstantiation" | "TypeofTypeAnnotation" | "UnaryExpression" | "UnionTypeAnnotation" | "UpdateExpression" | "V8IntrinsicIdentifier" | "VariableDeclaration" | "VariableDeclarator" | "Variance" | "VoidTypeAnnotation" | "WhileStatement" | "WithStatement" | "YieldExpression" | keyof Aliases)[];
declare const METHOD_TYPES: ("StringLiteral" | "NumericLiteral" | "JSXText" | "JSXIdentifier" | "AnyTypeAnnotation" | "ArgumentPlaceholder" | "ArrayExpression" | "ArrayPattern" | "ArrayTypeAnnotation" | "ArrowFunctionExpression" | "AssignmentExpression" | "AssignmentPattern" | "AwaitExpression" | "BigIntLiteral" | "BinaryExpression" | "BindExpression" | "BlockStatement" | "BooleanLiteral" | "BooleanLiteralTypeAnnotation" | "BooleanTypeAnnotation" | "BreakStatement" | "CallExpression" | "CatchClause" | "ClassAccessorProperty" | "ClassBody" | "ClassDeclaration" | "ClassExpression" | "ClassImplements" | "ClassMethod" | "ClassPrivateMethod" | "ClassPrivateProperty" | "ClassProperty" | "ConditionalExpression" | "ContinueStatement" | "DebuggerStatement" | "DecimalLiteral" | "DeclareClass" | "DeclareExportAllDeclaration" | "DeclareExportDeclaration" | "DeclareFunction" | "DeclareInterface" | "DeclareModule" | "DeclareModuleExports" | "DeclareOpaqueType" | "DeclareTypeAlias" | "DeclareVariable" | "DeclaredPredicate" | "Decorator" | "Directive" | "DirectiveLiteral" | "DoExpression" | "DoWhileStatement" | "EmptyStatement" | "EmptyTypeAnnotation" | "EnumBooleanBody" | "EnumBooleanMember" | "EnumDeclaration" | "EnumDefaultedMember" | "EnumNumberBody" | "EnumNumberMember" | "EnumStringBody" | "EnumStringMember" | "EnumSymbolBody" | "ExistsTypeAnnotation" | "ExportAllDeclaration" | "ExportDefaultDeclaration" | "ExportDefaultSpecifier" | "ExportNamedDeclaration" | "ExportNamespaceSpecifier" | "ExportSpecifier" | "ExpressionStatement" | "File" | "ForInStatement" | "ForOfStatement" | "ForStatement" | "FunctionDeclaration" | "FunctionExpression" | "FunctionTypeAnnotation" | "FunctionTypeParam" | "GenericTypeAnnotation" | "Identifier" | "IfStatement" | "Import" | "ImportAttribute" | "ImportDeclaration" | "ImportDefaultSpecifier" | "ImportExpression" | "ImportNamespaceSpecifier" | "ImportSpecifier" | "IndexedAccessType" | "InferredPredicate" | "InterfaceDeclaration" | "InterfaceExtends" | "InterfaceTypeAnnotation" | "InterpreterDirective" | "IntersectionTypeAnnotation" | "JSXAttribute" | "JSXClosingElement" | "JSXClosingFragment" | "JSXElement" | "JSXEmptyExpression" | "JSXExpressionContainer" | "JSXFragment" | "JSXMemberExpression" | "JSXNamespacedName" | "JSXOpeningElement" | "JSXOpeningFragment" | "JSXSpreadAttribute" | "JSXSpreadChild" | "LabeledStatement" | "LogicalExpression" | "MemberExpression" | "MetaProperty" | "MixedTypeAnnotation" | "ModuleExpression" | "NewExpression" | "Noop" | "NullLiteral" | "NullLiteralTypeAnnotation" | "NullableTypeAnnotation" | "NumberLiteral" | "NumberLiteralTypeAnnotation" | "NumberTypeAnnotation" | "ObjectExpression" | "ObjectMethod" | "ObjectPattern" | "ObjectProperty" | "ObjectTypeAnnotation" | "ObjectTypeCallProperty" | "ObjectTypeIndexer" | "ObjectTypeInternalSlot" | "ObjectTypeProperty" | "ObjectTypeSpreadProperty" | "OpaqueType" | "OptionalCallExpression" | "OptionalIndexedAccessType" | "OptionalMemberExpression" | "ParenthesizedExpression" | "PipelineBareFunction" | "PipelinePrimaryTopicReference" | "PipelineTopicExpression" | "Placeholder" | "PrivateName" | "Program" | "QualifiedTypeIdentifier" | "RecordExpression" | "RegExpLiteral" | "RegexLiteral" | "RestElement" | "RestProperty" | "ReturnStatement" | "SequenceExpression" | "SpreadElement" | "SpreadProperty" | "StaticBlock" | "StringLiteralTypeAnnotation" | "StringTypeAnnotation" | "Super" | "SwitchCase" | "SwitchStatement" | "SymbolTypeAnnotation" | "TSAnyKeyword" | "TSArrayType" | "TSAsExpression" | "TSBigIntKeyword" | "TSBooleanKeyword" | "TSCallSignatureDeclaration" | "TSConditionalType" | "TSConstructSignatureDeclaration" | "TSConstructorType" | "TSDeclareFunction" | "TSDeclareMethod" | "TSEnumDeclaration" | "TSEnumMember" | "TSExportAssignment" | "TSExpressionWithTypeArguments" | "TSExternalModuleReference" | "TSFunctionType" | "TSImportEqualsDeclaration" | "TSImportType" | "TSIndexSignature" | "TSIndexedAccessType" | "TSInferType" | "TSInstantiationExpression" | "TSInterfaceBody" | "TSInterfaceDeclaration" | "TSIntersectionType" | "TSIntrinsicKeyword" | "TSLiteralType" | "TSMappedType" | "TSMethodSignature" | "TSModuleBlock" | "TSModuleDeclaration" | "TSNamedTupleMember" | "TSNamespaceExportDeclaration" | "TSNeverKeyword" | "TSNonNullExpression" | "TSNullKeyword" | "TSNumberKeyword" | "TSObjectKeyword" | "TSOptionalType" | "TSParameterProperty" | "TSParenthesizedType" | "TSPropertySignature" | "TSQualifiedName" | "TSRestType" | "TSSatisfiesExpression" | "TSStringKeyword" | "TSSymbolKeyword" | "TSThisType" | "TSTupleType" | "TSTypeAliasDeclaration" | "TSTypeAnnotation" | "TSTypeAssertion" | "TSTypeLiteral" | "TSTypeOperator" | "TSTypeParameter" | "TSTypeParameterDeclaration" | "TSTypeParameterInstantiation" | "TSTypePredicate" | "TSTypeQuery" | "TSTypeReference" | "TSUndefinedKeyword" | "TSUnionType" | "TSUnknownKeyword" | "TSVoidKeyword" | "TaggedTemplateExpression" | "TemplateElement" | "TemplateLiteral" | "ThisExpression" | "ThisTypeAnnotation" | "ThrowStatement" | "TopicReference" | "TryStatement" | "TupleExpression" | "TupleTypeAnnotation" | "TypeAlias" | "TypeAnnotation" | "TypeCastExpression" | "TypeParameter" | "TypeParameterDeclaration" | "TypeParameterInstantiation" | "TypeofTypeAnnotation" | "UnaryExpression" | "UnionTypeAnnotation" | "UpdateExpression" | "V8IntrinsicIdentifier" | "VariableDeclaration" | "VariableDeclarator" | "Variance" | "VoidTypeAnnotation" | "WhileStatement" | "WithStatement" | "YieldExpression" | keyof Aliases)[];
declare const OBJECTMEMBER_TYPES: ("StringLiteral" | "NumericLiteral" | "JSXText" | "JSXIdentifier" | "AnyTypeAnnotation" | "ArgumentPlaceholder" | "ArrayExpression" | "ArrayPattern" | "ArrayTypeAnnotation" | "ArrowFunctionExpression" | "AssignmentExpression" | "AssignmentPattern" | "AwaitExpression" | "BigIntLiteral" | "BinaryExpression" | "BindExpression" | "BlockStatement" | "BooleanLiteral" | "BooleanLiteralTypeAnnotation" | "BooleanTypeAnnotation" | "BreakStatement" | "CallExpression" | "CatchClause" | "ClassAccessorProperty" | "ClassBody" | "ClassDeclaration" | "ClassExpression" | "ClassImplements" | "ClassMethod" | "ClassPrivateMethod" | "ClassPrivateProperty" | "ClassProperty" | "ConditionalExpression" | "ContinueStatement" | "DebuggerStatement" | "DecimalLiteral" | "DeclareClass" | "DeclareExportAllDeclaration" | "DeclareExportDeclaration" | "DeclareFunction" | "DeclareInterface" | "DeclareModule" | "DeclareModuleExports" | "DeclareOpaqueType" | "DeclareTypeAlias" | "DeclareVariable" | "DeclaredPredicate" | "Decorator" | "Directive" | "DirectiveLiteral" | "DoExpression" | "DoWhileStatement" | "EmptyStatement" | "EmptyTypeAnnotation" | "EnumBooleanBody" | "EnumBooleanMember" | "EnumDeclaration" | "EnumDefaultedMember" | "EnumNumberBody" | "EnumNumberMember" | "EnumStringBody" | "EnumStringMember" | "EnumSymbolBody" | "ExistsTypeAnnotation" | "ExportAllDeclaration" | "ExportDefaultDeclaration" | "ExportDefaultSpecifier" | "ExportNamedDeclaration" | "ExportNamespaceSpecifier" | "ExportSpecifier" | "ExpressionStatement" | "File" | "ForInStatement" | "ForOfStatement" | "ForStatement" | "FunctionDeclaration" | "FunctionExpression" | "FunctionTypeAnnotation" | "FunctionTypeParam" | "GenericTypeAnnotation" | "Identifier" | "IfStatement" | "Import" | "ImportAttribute" | "ImportDeclaration" | "ImportDefaultSpecifier" | "ImportExpression" | "ImportNamespaceSpecifier" | "ImportSpecifier" | "IndexedAccessType" | "InferredPredicate" | "InterfaceDeclaration" | "InterfaceExtends" | "InterfaceTypeAnnotation" | "InterpreterDirective" | "IntersectionTypeAnnotation" | "JSXAttribute" | "JSXClosingElement" | "JSXClosingFragment" | "JSXElement" | "JSXEmptyExpression" | "JSXExpressionContainer" | "JSXFragment" | "JSXMemberExpression" | "JSXNamespacedName" | "JSXOpeningElement" | "JSXOpeningFragment" | "JSXSpreadAttribute" | "JSXSpreadChild" | "LabeledStatement" | "LogicalExpression" | "MemberExpression" | "MetaProperty" | "MixedTypeAnnotation" | "ModuleExpression" | "NewExpression" | "Noop" | "NullLiteral" | "NullLiteralTypeAnnotation" | "NullableTypeAnnotation" | "NumberLiteral" | "NumberLiteralTypeAnnotation" | "NumberTypeAnnotation" | "ObjectExpression" | "ObjectMethod" | "ObjectPattern" | "ObjectProperty" | "ObjectTypeAnnotation" | "ObjectTypeCallProperty" | "ObjectTypeIndexer" | "ObjectTypeInternalSlot" | "ObjectTypeProperty" | "ObjectTypeSpreadProperty" | "OpaqueType" | "OptionalCallExpression" | "OptionalIndexedAccessType" | "OptionalMemberExpression" | "ParenthesizedExpression" | "PipelineBareFunction" | "PipelinePrimaryTopicReference" | "PipelineTopicExpression" | "Placeholder" | "PrivateName" | "Program" | "QualifiedTypeIdentifier" | "RecordExpression" | "RegExpLiteral" | "RegexLiteral" | "RestElement" | "RestProperty" | "ReturnStatement" | "SequenceExpression" | "SpreadElement" | "SpreadProperty" | "StaticBlock" | "StringLiteralTypeAnnotation" | "StringTypeAnnotation" | "Super" | "SwitchCase" | "SwitchStatement" | "SymbolTypeAnnotation" | "TSAnyKeyword" | "TSArrayType" | "TSAsExpression" | "TSBigIntKeyword" | "TSBooleanKeyword" | "TSCallSignatureDeclaration" | "TSConditionalType" | "TSConstructSignatureDeclaration" | "TSConstructorType" | "TSDeclareFunction" | "TSDeclareMethod" | "TSEnumDeclaration" | "TSEnumMember" | "TSExportAssignment" | "TSExpressionWithTypeArguments" | "TSExternalModuleReference" | "TSFunctionType" | "TSImportEqualsDeclaration" | "TSImportType" | "TSIndexSignature" | "TSIndexedAccessType" | "TSInferType" | "TSInstantiationExpression" | "TSInterfaceBody" | "TSInterfaceDeclaration" | "TSIntersectionType" | "TSIntrinsicKeyword" | "TSLiteralType" | "TSMappedType" | "TSMethodSignature" | "TSModuleBlock" | "TSModuleDeclaration" | "TSNamedTupleMember" | "TSNamespaceExportDeclaration" | "TSNeverKeyword" | "TSNonNullExpression" | "TSNullKeyword" | "TSNumberKeyword" | "TSObjectKeyword" | "TSOptionalType" | "TSParameterProperty" | "TSParenthesizedType" | "TSPropertySignature" | "TSQualifiedName" | "TSRestType" | "TSSatisfiesExpression" | "TSStringKeyword" | "TSSymbolKeyword" | "TSThisType" | "TSTupleType" | "TSTypeAliasDeclaration" | "TSTypeAnnotation" | "TSTypeAssertion" | "TSTypeLiteral" | "TSTypeOperator" | "TSTypeParameter" | "TSTypeParameterDeclaration" | "TSTypeParameterInstantiation" | "TSTypePredicate" | "TSTypeQuery" | "TSTypeReference" | "TSUndefinedKeyword" | "TSUnionType" | "TSUnknownKeyword" | "TSVoidKeyword" | "TaggedTemplateExpression" | "TemplateElement" | "TemplateLiteral" | "ThisExpression" | "ThisTypeAnnotation" | "ThrowStatement" | "TopicReference" | "TryStatement" | "TupleExpression" | "TupleTypeAnnotation" | "TypeAlias" | "TypeAnnotation" | "TypeCastExpression" | "TypeParameter" | "TypeParameterDeclaration" | "TypeParameterInstantiation" | "TypeofTypeAnnotation" | "UnaryExpression" | "UnionTypeAnnotation" | "UpdateExpression" | "V8IntrinsicIdentifier" | "VariableDeclaration" | "VariableDeclarator" | "Variance" | "VoidTypeAnnotation" | "WhileStatement" | "WithStatement" | "YieldExpression" | keyof Aliases)[];
declare const PROPERTY_TYPES: ("StringLiteral" | "NumericLiteral" | "JSXText" | "JSXIdentifier" | "AnyTypeAnnotation" | "ArgumentPlaceholder" | "ArrayExpression" | "ArrayPattern" | "ArrayTypeAnnotation" | "ArrowFunctionExpression" | "AssignmentExpression" | "AssignmentPattern" | "AwaitExpression" | "BigIntLiteral" | "BinaryExpression" | "BindExpression" | "BlockStatement" | "BooleanLiteral" | "BooleanLiteralTypeAnnotation" | "BooleanTypeAnnotation" | "BreakStatement" | "CallExpression" | "CatchClause" | "ClassAccessorProperty" | "ClassBody" | "ClassDeclaration" | "ClassExpression" | "ClassImplements" | "ClassMethod" | "ClassPrivateMethod" | "ClassPrivateProperty" | "ClassProperty" | "ConditionalExpression" | "ContinueStatement" | "DebuggerStatement" | "DecimalLiteral" | "DeclareClass" | "DeclareExportAllDeclaration" | "DeclareExportDeclaration" | "DeclareFunction" | "DeclareInterface" | "DeclareModule" | "DeclareModuleExports" | "DeclareOpaqueType" | "DeclareTypeAlias" | "DeclareVariable" | "DeclaredPredicate" | "Decorator" | "Directive" | "DirectiveLiteral" | "DoExpression" | "DoWhileStatement" | "EmptyStatement" | "EmptyTypeAnnotation" | "EnumBooleanBody" | "EnumBooleanMember" | "EnumDeclaration" | "EnumDefaultedMember" | "EnumNumberBody" | "EnumNumberMember" | "EnumStringBody" | "EnumStringMember" | "EnumSymbolBody" | "ExistsTypeAnnotation" | "ExportAllDeclaration" | "ExportDefaultDeclaration" | "ExportDefaultSpecifier" | "ExportNamedDeclaration" | "ExportNamespaceSpecifier" | "ExportSpecifier" | "ExpressionStatement" | "File" | "ForInStatement" | "ForOfStatement" | "ForStatement" | "FunctionDeclaration" | "FunctionExpression" | "FunctionTypeAnnotation" | "FunctionTypeParam" | "GenericTypeAnnotation" | "Identifier" | "IfStatement" | "Import" | "ImportAttribute" | "ImportDeclaration" | "ImportDefaultSpecifier" | "ImportExpression" | "ImportNamespaceSpecifier" | "ImportSpecifier" | "IndexedAccessType" | "InferredPredicate" | "InterfaceDeclaration" | "InterfaceExtends" | "InterfaceTypeAnnotation" | "InterpreterDirective" | "IntersectionTypeAnnotation" | "JSXAttribute" | "JSXClosingElement" | "JSXClosingFragment" | "JSXElement" | "JSXEmptyExpression" | "JSXExpressionContainer" | "JSXFragment" | "JSXMemberExpression" | "JSXNamespacedName" | "JSXOpeningElement" | "JSXOpeningFragment" | "JSXSpreadAttribute" | "JSXSpreadChild" | "LabeledStatement" | "LogicalExpression" | "MemberExpression" | "MetaProperty" | "MixedTypeAnnotation" | "ModuleExpression" | "NewExpression" | "Noop" | "NullLiteral" | "NullLiteralTypeAnnotation" | "NullableTypeAnnotation" | "NumberLiteral" | "NumberLiteralTypeAnnotation" | "NumberTypeAnnotation" | "ObjectExpression" | "ObjectMethod" | "ObjectPattern" | "ObjectProperty" | "ObjectTypeAnnotation" | "ObjectTypeCallProperty" | "ObjectTypeIndexer" | "ObjectTypeInternalSlot" | "ObjectTypeProperty" | "ObjectTypeSpreadProperty" | "OpaqueType" | "OptionalCallExpression" | "OptionalIndexedAccessType" | "OptionalMemberExpression" | "ParenthesizedExpression" | "PipelineBareFunction" | "PipelinePrimaryTopicReference" | "PipelineTopicExpression" | "Placeholder" | "PrivateName" | "Program" | "QualifiedTypeIdentifier" | "RecordExpression" | "RegExpLiteral" | "RegexLiteral" | "RestElement" | "RestProperty" | "ReturnStatement" | "SequenceExpression" | "SpreadElement" | "SpreadProperty" | "StaticBlock" | "StringLiteralTypeAnnotation" | "StringTypeAnnotation" | "Super" | "SwitchCase" | "SwitchStatement" | "SymbolTypeAnnotation" | "TSAnyKeyword" | "TSArrayType" | "TSAsExpression" | "TSBigIntKeyword" | "TSBooleanKeyword" | "TSCallSignatureDeclaration" | "TSConditionalType" | "TSConstructSignatureDeclaration" | "TSConstructorType" | "TSDeclareFunction" | "TSDeclareMethod" | "TSEnumDeclaration" | "TSEnumMember" | "TSExportAssignment" | "TSExpressionWithTypeArguments" | "TSExternalModuleReference" | "TSFunctionType" | "TSImportEqualsDeclaration" | "TSImportType" | "TSIndexSignature" | "TSIndexedAccessType" | "TSInferType" | "TSInstantiationExpression" | "TSInterfaceBody" | "TSInterfaceDeclaration" | "TSIntersectionType" | "TSIntrinsicKeyword" | "TSLiteralType" | "TSMappedType" | "TSMethodSignature" | "TSModuleBlock" | "TSModuleDeclaration" | "TSNamedTupleMember" | "TSNamespaceExportDeclaration" | "TSNeverKeyword" | "TSNonNullExpression" | "TSNullKeyword" | "TSNumberKeyword" | "TSObjectKeyword" | "TSOptionalType" | "TSParameterProperty" | "TSParenthesizedType" | "TSPropertySignature" | "TSQualifiedName" | "TSRestType" | "TSSatisfiesExpression" | "TSStringKeyword" | "TSSymbolKeyword" | "TSThisType" | "TSTupleType" | "TSTypeAliasDeclaration" | "TSTypeAnnotation" | "TSTypeAssertion" | "TSTypeLiteral" | "TSTypeOperator" | "TSTypeParameter" | "TSTypeParameterDeclaration" | "TSTypeParameterInstantiation" | "TSTypePredicate" | "TSTypeQuery" | "TSTypeReference" | "TSUndefinedKeyword" | "TSUnionType" | "TSUnknownKeyword" | "TSVoidKeyword" | "TaggedTemplateExpression" | "TemplateElement" | "TemplateLiteral" | "ThisExpression" | "ThisTypeAnnotation" | "ThrowStatement" | "TopicReference" | "TryStatement" | "TupleExpression" | "TupleTypeAnnotation" | "TypeAlias" | "TypeAnnotation" | "TypeCastExpression" | "TypeParameter" | "TypeParameterDeclaration" | "TypeParameterInstantiation" | "TypeofTypeAnnotation" | "UnaryExpression" | "UnionTypeAnnotation" | "UpdateExpression" | "V8IntrinsicIdentifier" | "VariableDeclaration" | "VariableDeclarator" | "Variance" | "VoidTypeAnnotation" | "WhileStatement" | "WithStatement" | "YieldExpression" | keyof Aliases)[];
declare const UNARYLIKE_TYPES: ("StringLiteral" | "NumericLiteral" | "JSXText" | "JSXIdentifier" | "AnyTypeAnnotation" | "ArgumentPlaceholder" | "ArrayExpression" | "ArrayPattern" | "ArrayTypeAnnotation" | "ArrowFunctionExpression" | "AssignmentExpression" | "AssignmentPattern" | "AwaitExpression" | "BigIntLiteral" | "BinaryExpression" | "BindExpression" | "BlockStatement" | "BooleanLiteral" | "BooleanLiteralTypeAnnotation" | "BooleanTypeAnnotation" | "BreakStatement" | "CallExpression" | "CatchClause" | "ClassAccessorProperty" | "ClassBody" | "ClassDeclaration" | "ClassExpression" | "ClassImplements" | "ClassMethod" | "ClassPrivateMethod" | "ClassPrivateProperty" | "ClassProperty" | "ConditionalExpression" | "ContinueStatement" | "DebuggerStatement" | "DecimalLiteral" | "DeclareClass" | "DeclareExportAllDeclaration" | "DeclareExportDeclaration" | "DeclareFunction" | "DeclareInterface" | "DeclareModule" | "DeclareModuleExports" | "DeclareOpaqueType" | "DeclareTypeAlias" | "DeclareVariable" | "DeclaredPredicate" | "Decorator" | "Directive" | "DirectiveLiteral" | "DoExpression" | "DoWhileStatement" | "EmptyStatement" | "EmptyTypeAnnotation" | "EnumBooleanBody" | "EnumBooleanMember" | "EnumDeclaration" | "EnumDefaultedMember" | "EnumNumberBody" | "EnumNumberMember" | "EnumStringBody" | "EnumStringMember" | "EnumSymbolBody" | "ExistsTypeAnnotation" | "ExportAllDeclaration" | "ExportDefaultDeclaration" | "ExportDefaultSpecifier" | "ExportNamedDeclaration" | "ExportNamespaceSpecifier" | "ExportSpecifier" | "ExpressionStatement" | "File" | "ForInStatement" | "ForOfStatement" | "ForStatement" | "FunctionDeclaration" | "FunctionExpression" | "FunctionTypeAnnotation" | "FunctionTypeParam" | "GenericTypeAnnotation" | "Identifier" | "IfStatement" | "Import" | "ImportAttribute" | "ImportDeclaration" | "ImportDefaultSpecifier" | "ImportExpression" | "ImportNamespaceSpecifier" | "ImportSpecifier" | "IndexedAccessType" | "InferredPredicate" | "InterfaceDeclaration" | "InterfaceExtends" | "InterfaceTypeAnnotation" | "InterpreterDirective" | "IntersectionTypeAnnotation" | "JSXAttribute" | "JSXClosingElement" | "JSXClosingFragment" | "JSXElement" | "JSXEmptyExpression" | "JSXExpressionContainer" | "JSXFragment" | "JSXMemberExpression" | "JSXNamespacedName" | "JSXOpeningElement" | "JSXOpeningFragment" | "JSXSpreadAttribute" | "JSXSpreadChild" | "LabeledStatement" | "LogicalExpression" | "MemberExpression" | "MetaProperty" | "MixedTypeAnnotation" | "ModuleExpression" | "NewExpression" | "Noop" | "NullLiteral" | "NullLiteralTypeAnnotation" | "NullableTypeAnnotation" | "NumberLiteral" | "NumberLiteralTypeAnnotation" | "NumberTypeAnnotation" | "ObjectExpression" | "ObjectMethod" | "ObjectPattern" | "ObjectProperty" | "ObjectTypeAnnotation" | "ObjectTypeCallProperty" | "ObjectTypeIndexer" | "ObjectTypeInternalSlot" | "ObjectTypeProperty" | "ObjectTypeSpreadProperty" | "OpaqueType" | "OptionalCallExpression" | "OptionalIndexedAccessType" | "OptionalMemberExpression" | "ParenthesizedExpression" | "PipelineBareFunction" | "PipelinePrimaryTopicReference" | "PipelineTopicExpression" | "Placeholder" | "PrivateName" | "Program" | "QualifiedTypeIdentifier" | "RecordExpression" | "RegExpLiteral" | "RegexLiteral" | "RestElement" | "RestProperty" | "ReturnStatement" | "SequenceExpression" | "SpreadElement" | "SpreadProperty" | "StaticBlock" | "StringLiteralTypeAnnotation" | "StringTypeAnnotation" | "Super" | "SwitchCase" | "SwitchStatement" | "SymbolTypeAnnotation" | "TSAnyKeyword" | "TSArrayType" | "TSAsExpression" | "TSBigIntKeyword" | "TSBooleanKeyword" | "TSCallSignatureDeclaration" | "TSConditionalType" | "TSConstructSignatureDeclaration" | "TSConstructorType" | "TSDeclareFunction" | "TSDeclareMethod" | "TSEnumDeclaration" | "TSEnumMember" | "TSExportAssignment" | "TSExpressionWithTypeArguments" | "TSExternalModuleReference" | "TSFunctionType" | "TSImportEqualsDeclaration" | "TSImportType" | "TSIndexSignature" | "TSIndexedAccessType" | "TSInferType" | "TSInstantiationExpression" | "TSInterfaceBody" | "TSInterfaceDeclaration" | "TSIntersectionType" | "TSIntrinsicKeyword" | "TSLiteralType" | "TSMappedType" | "TSMethodSignature" | "TSModuleBlock" | "TSModuleDeclaration" | "TSNamedTupleMember" | "TSNamespaceExportDeclaration" | "TSNeverKeyword" | "TSNonNullExpression" | "TSNullKeyword" | "TSNumberKeyword" | "TSObjectKeyword" | "TSOptionalType" | "TSParameterProperty" | "TSParenthesizedType" | "TSPropertySignature" | "TSQualifiedName" | "TSRestType" | "TSSatisfiesExpression" | "TSStringKeyword" | "TSSymbolKeyword" | "TSThisType" | "TSTupleType" | "TSTypeAliasDeclaration" | "TSTypeAnnotation" | "TSTypeAssertion" | "TSTypeLiteral" | "TSTypeOperator" | "TSTypeParameter" | "TSTypeParameterDeclaration" | "TSTypeParameterInstantiation" | "TSTypePredicate" | "TSTypeQuery" | "TSTypeReference" | "TSUndefinedKeyword" | "TSUnionType" | "TSUnknownKeyword" | "TSVoidKeyword" | "TaggedTemplateExpression" | "TemplateElement" | "TemplateLiteral" | "ThisExpression" | "ThisTypeAnnotation" | "ThrowStatement" | "TopicReference" | "TryStatement" | "TupleExpression" | "TupleTypeAnnotation" | "TypeAlias" | "TypeAnnotation" | "TypeCastExpression" | "TypeParameter" | "TypeParameterDeclaration" | "TypeParameterInstantiation" | "TypeofTypeAnnotation" | "UnaryExpression" | "UnionTypeAnnotation" | "UpdateExpression" | "V8IntrinsicIdentifier" | "VariableDeclaration" | "VariableDeclarator" | "Variance" | "VoidTypeAnnotation" | "WhileStatement" | "WithStatement" | "YieldExpression" | keyof Aliases)[];
declare const PATTERN_TYPES: ("StringLiteral" | "NumericLiteral" | "JSXText" | "JSXIdentifier" | "AnyTypeAnnotation" | "ArgumentPlaceholder" | "ArrayExpression" | "ArrayPattern" | "ArrayTypeAnnotation" | "ArrowFunctionExpression" | "AssignmentExpression" | "AssignmentPattern" | "AwaitExpression" | "BigIntLiteral" | "BinaryExpression" | "BindExpression" | "BlockStatement" | "BooleanLiteral" | "BooleanLiteralTypeAnnotation" | "BooleanTypeAnnotation" | "BreakStatement" | "CallExpression" | "CatchClause" | "ClassAccessorProperty" | "ClassBody" | "ClassDeclaration" | "ClassExpression" | "ClassImplements" | "ClassMethod" | "ClassPrivateMethod" | "ClassPrivateProperty" | "ClassProperty" | "ConditionalExpression" | "ContinueStatement" | "DebuggerStatement" | "DecimalLiteral" | "DeclareClass" | "DeclareExportAllDeclaration" | "DeclareExportDeclaration" | "DeclareFunction" | "DeclareInterface" | "DeclareModule" | "DeclareModuleExports" | "DeclareOpaqueType" | "DeclareTypeAlias" | "DeclareVariable" | "DeclaredPredicate" | "Decorator" | "Directive" | "DirectiveLiteral" | "DoExpression" | "DoWhileStatement" | "EmptyStatement" | "EmptyTypeAnnotation" | "EnumBooleanBody" | "EnumBooleanMember" | "EnumDeclaration" | "EnumDefaultedMember" | "EnumNumberBody" | "EnumNumberMember" | "EnumStringBody" | "EnumStringMember" | "EnumSymbolBody" | "ExistsTypeAnnotation" | "ExportAllDeclaration" | "ExportDefaultDeclaration" | "ExportDefaultSpecifier" | "ExportNamedDeclaration" | "ExportNamespaceSpecifier" | "ExportSpecifier" | "ExpressionStatement" | "File" | "ForInStatement" | "ForOfStatement" | "ForStatement" | "FunctionDeclaration" | "FunctionExpression" | "FunctionTypeAnnotation" | "FunctionTypeParam" | "GenericTypeAnnotation" | "Identifier" | "IfStatement" | "Import" | "ImportAttribute" | "ImportDeclaration" | "ImportDefaultSpecifier" | "ImportExpression" | "ImportNamespaceSpecifier" | "ImportSpecifier" | "IndexedAccessType" | "InferredPredicate" | "InterfaceDeclaration" | "InterfaceExtends" | "InterfaceTypeAnnotation" | "InterpreterDirective" | "IntersectionTypeAnnotation" | "JSXAttribute" | "JSXClosingElement" | "JSXClosingFragment" | "JSXElement" | "JSXEmptyExpression" | "JSXExpressionContainer" | "JSXFragment" | "JSXMemberExpression" | "JSXNamespacedName" | "JSXOpeningElement" | "JSXOpeningFragment" | "JSXSpreadAttribute" | "JSXSpreadChild" | "LabeledStatement" | "LogicalExpression" | "MemberExpression" | "MetaProperty" | "MixedTypeAnnotation" | "ModuleExpression" | "NewExpression" | "Noop" | "NullLiteral" | "NullLiteralTypeAnnotation" | "NullableTypeAnnotation" | "NumberLiteral" | "NumberLiteralTypeAnnotation" | "NumberTypeAnnotation" | "ObjectExpression" | "ObjectMethod" | "ObjectPattern" | "ObjectProperty" | "ObjectTypeAnnotation" | "ObjectTypeCallProperty" | "ObjectTypeIndexer" | "ObjectTypeInternalSlot" | "ObjectTypeProperty" | "ObjectTypeSpreadProperty" | "OpaqueType" | "OptionalCallExpression" | "OptionalIndexedAccessType" | "OptionalMemberExpression" | "ParenthesizedExpression" | "PipelineBareFunction" | "PipelinePrimaryTopicReference" | "PipelineTopicExpression" | "Placeholder" | "PrivateName" | "Program" | "QualifiedTypeIdentifier" | "RecordExpression" | "RegExpLiteral" | "RegexLiteral" | "RestElement" | "RestProperty" | "ReturnStatement" | "SequenceExpression" | "SpreadElement" | "SpreadProperty" | "StaticBlock" | "StringLiteralTypeAnnotation" | "StringTypeAnnotation" | "Super" | "SwitchCase" | "SwitchStatement" | "SymbolTypeAnnotation" | "TSAnyKeyword" | "TSArrayType" | "TSAsExpression" | "TSBigIntKeyword" | "TSBooleanKeyword" | "TSCallSignatureDeclaration" | "TSConditionalType" | "TSConstructSignatureDeclaration" | "TSConstructorType" | "TSDeclareFunction" | "TSDeclareMethod" | "TSEnumDeclaration" | "TSEnumMember" | "TSExportAssignment" | "TSExpressionWithTypeArguments" | "TSExternalModuleReference" | "TSFunctionType" | "TSImportEqualsDeclaration" | "TSImportType" | "TSIndexSignature" | "TSIndexedAccessType" | "TSInferType" | "TSInstantiationExpression" | "TSInterfaceBody" | "TSInterfaceDeclaration" | "TSIntersectionType" | "TSIntrinsicKeyword" | "TSLiteralType" | "TSMappedType" | "TSMethodSignature" | "TSModuleBlock" | "TSModuleDeclaration" | "TSNamedTupleMember" | "TSNamespaceExportDeclaration" | "TSNeverKeyword" | "TSNonNullExpression" | "TSNullKeyword" | "TSNumberKeyword" | "TSObjectKeyword" | "TSOptionalType" | "TSParameterProperty" | "TSParenthesizedType" | "TSPropertySignature" | "TSQualifiedName" | "TSRestType" | "TSSatisfiesExpression" | "TSStringKeyword" | "TSSymbolKeyword" | "TSThisType" | "TSTupleType" | "TSTypeAliasDeclaration" | "TSTypeAnnotation" | "TSTypeAssertion" | "TSTypeLiteral" | "TSTypeOperator" | "TSTypeParameter" | "TSTypeParameterDeclaration" | "TSTypeParameterInstantiation" | "TSTypePredicate" | "TSTypeQuery" | "TSTypeReference" | "TSUndefinedKeyword" | "TSUnionType" | "TSUnknownKeyword" | "TSVoidKeyword" | "TaggedTemplateExpression" | "TemplateElement" | "TemplateLiteral" | "ThisExpression" | "ThisTypeAnnotation" | "ThrowStatement" | "TopicReference" | "TryStatement" | "TupleExpression" | "TupleTypeAnnotation" | "TypeAlias" | "TypeAnnotation" | "TypeCastExpression" | "TypeParameter" | "TypeParameterDeclaration" | "TypeParameterInstantiation" | "TypeofTypeAnnotation" | "UnaryExpression" | "UnionTypeAnnotation" | "UpdateExpression" | "V8IntrinsicIdentifier" | "VariableDeclaration" | "VariableDeclarator" | "Variance" | "VoidTypeAnnotation" | "WhileStatement" | "WithStatement" | "YieldExpression" | keyof Aliases)[];
declare const CLASS_TYPES: ("StringLiteral" | "NumericLiteral" | "JSXText" | "JSXIdentifier" | "AnyTypeAnnotation" | "ArgumentPlaceholder" | "ArrayExpression" | "ArrayPattern" | "ArrayTypeAnnotation" | "ArrowFunctionExpression" | "AssignmentExpression" | "AssignmentPattern" | "AwaitExpression" | "BigIntLiteral" | "BinaryExpression" | "BindExpression" | "BlockStatement" | "BooleanLiteral" | "BooleanLiteralTypeAnnotation" | "BooleanTypeAnnotation" | "BreakStatement" | "CallExpression" | "CatchClause" | "ClassAccessorProperty" | "ClassBody" | "ClassDeclaration" | "ClassExpression" | "ClassImplements" | "ClassMethod" | "ClassPrivateMethod" | "ClassPrivateProperty" | "ClassProperty" | "ConditionalExpression" | "ContinueStatement" | "DebuggerStatement" | "DecimalLiteral" | "DeclareClass" | "DeclareExportAllDeclaration" | "DeclareExportDeclaration" | "DeclareFunction" | "DeclareInterface" | "DeclareModule" | "DeclareModuleExports" | "DeclareOpaqueType" | "DeclareTypeAlias" | "DeclareVariable" | "DeclaredPredicate" | "Decorator" | "Directive" | "DirectiveLiteral" | "DoExpression" | "DoWhileStatement" | "EmptyStatement" | "EmptyTypeAnnotation" | "EnumBooleanBody" | "EnumBooleanMember" | "EnumDeclaration" | "EnumDefaultedMember" | "EnumNumberBody" | "EnumNumberMember" | "EnumStringBody" | "EnumStringMember" | "EnumSymbolBody" | "ExistsTypeAnnotation" | "ExportAllDeclaration" | "ExportDefaultDeclaration" | "ExportDefaultSpecifier" | "ExportNamedDeclaration" | "ExportNamespaceSpecifier" | "ExportSpecifier" | "ExpressionStatement" | "File" | "ForInStatement" | "ForOfStatement" | "ForStatement" | "FunctionDeclaration" | "FunctionExpression" | "FunctionTypeAnnotation" | "FunctionTypeParam" | "GenericTypeAnnotation" | "Identifier" | "IfStatement" | "Import" | "ImportAttribute" | "ImportDeclaration" | "ImportDefaultSpecifier" | "ImportExpression" | "ImportNamespaceSpecifier" | "ImportSpecifier" | "IndexedAccessType" | "InferredPredicate" | "InterfaceDeclaration" | "InterfaceExtends" | "InterfaceTypeAnnotation" | "InterpreterDirective" | "IntersectionTypeAnnotation" | "JSXAttribute" | "JSXClosingElement" | "JSXClosingFragment" | "JSXElement" | "JSXEmptyExpression" | "JSXExpressionContainer" | "JSXFragment" | "JSXMemberExpression" | "JSXNamespacedName" | "JSXOpeningElement" | "JSXOpeningFragment" | "JSXSpreadAttribute" | "JSXSpreadChild" | "LabeledStatement" | "LogicalExpression" | "MemberExpression" | "MetaProperty" | "MixedTypeAnnotation" | "ModuleExpression" | "NewExpression" | "Noop" | "NullLiteral" | "NullLiteralTypeAnnotation" | "NullableTypeAnnotation" | "NumberLiteral" | "NumberLiteralTypeAnnotation" | "NumberTypeAnnotation" | "ObjectExpression" | "ObjectMethod" | "ObjectPattern" | "ObjectProperty" | "ObjectTypeAnnotation" | "ObjectTypeCallProperty" | "ObjectTypeIndexer" | "ObjectTypeInternalSlot" | "ObjectTypeProperty" | "ObjectTypeSpreadProperty" | "OpaqueType" | "OptionalCallExpression" | "OptionalIndexedAccessType" | "OptionalMemberExpression" | "ParenthesizedExpression" | "PipelineBareFunction" | "PipelinePrimaryTopicReference" | "PipelineTopicExpression" | "Placeholder" | "PrivateName" | "Program" | "QualifiedTypeIdentifier" | "RecordExpression" | "RegExpLiteral" | "RegexLiteral" | "RestElement" | "RestProperty" | "ReturnStatement" | "SequenceExpression" | "SpreadElement" | "SpreadProperty" | "StaticBlock" | "StringLiteralTypeAnnotation" | "StringTypeAnnotation" | "Super" | "SwitchCase" | "SwitchStatement" | "SymbolTypeAnnotation" | "TSAnyKeyword" | "TSArrayType" | "TSAsExpression" | "TSBigIntKeyword" | "TSBooleanKeyword" | "TSCallSignatureDeclaration" | "TSConditionalType" | "TSConstructSignatureDeclaration" | "TSConstructorType" | "TSDeclareFunction" | "TSDeclareMethod" | "TSEnumDeclaration" | "TSEnumMember" | "TSExportAssignment" | "TSExpressionWithTypeArguments" | "TSExternalModuleReference" | "TSFunctionType" | "TSImportEqualsDeclaration" | "TSImportType" | "TSIndexSignature" | "TSIndexedAccessType" | "TSInferType" | "TSInstantiationExpression" | "TSInterfaceBody" | "TSInterfaceDeclaration" | "TSIntersectionType" | "TSIntrinsicKeyword" | "TSLiteralType" | "TSMappedType" | "TSMethodSignature" | "TSModuleBlock" | "TSModuleDeclaration" | "TSNamedTupleMember" | "TSNamespaceExportDeclaration" | "TSNeverKeyword" | "TSNonNullExpression" | "TSNullKeyword" | "TSNumberKeyword" | "TSObjectKeyword" | "TSOptionalType" | "TSParameterProperty" | "TSParenthesizedType" | "TSPropertySignature" | "TSQualifiedName" | "TSRestType" | "TSSatisfiesExpression" | "TSStringKeyword" | "TSSymbolKeyword" | "TSThisType" | "TSTupleType" | "TSTypeAliasDeclaration" | "TSTypeAnnotation" | "TSTypeAssertion" | "TSTypeLiteral" | "TSTypeOperator" | "TSTypeParameter" | "TSTypeParameterDeclaration" | "TSTypeParameterInstantiation" | "TSTypePredicate" | "TSTypeQuery" | "TSTypeReference" | "TSUndefinedKeyword" | "TSUnionType" | "TSUnknownKeyword" | "TSVoidKeyword" | "TaggedTemplateExpression" | "TemplateElement" | "TemplateLiteral" | "ThisExpression" | "ThisTypeAnnotation" | "ThrowStatement" | "TopicReference" | "TryStatement" | "TupleExpression" | "TupleTypeAnnotation" | "TypeAlias" | "TypeAnnotation" | "TypeCastExpression" | "TypeParameter" | "TypeParameterDeclaration" | "TypeParameterInstantiation" | "TypeofTypeAnnotation" | "UnaryExpression" | "UnionTypeAnnotation" | "UpdateExpression" | "V8IntrinsicIdentifier" | "VariableDeclaration" | "VariableDeclarator" | "Variance" | "VoidTypeAnnotation" | "WhileStatement" | "WithStatement" | "YieldExpression" | keyof Aliases)[];
declare const IMPORTOREXPORTDECLARATION_TYPES: ("StringLiteral" | "NumericLiteral" | "JSXText" | "JSXIdentifier" | "AnyTypeAnnotation" | "ArgumentPlaceholder" | "ArrayExpression" | "ArrayPattern" | "ArrayTypeAnnotation" | "ArrowFunctionExpression" | "AssignmentExpression" | "AssignmentPattern" | "AwaitExpression" | "BigIntLiteral" | "BinaryExpression" | "BindExpression" | "BlockStatement" | "BooleanLiteral" | "BooleanLiteralTypeAnnotation" | "BooleanTypeAnnotation" | "BreakStatement" | "CallExpression" | "CatchClause" | "ClassAccessorProperty" | "ClassBody" | "ClassDeclaration" | "ClassExpression" | "ClassImplements" | "ClassMethod" | "ClassPrivateMethod" | "ClassPrivateProperty" | "ClassProperty" | "ConditionalExpression" | "ContinueStatement" | "DebuggerStatement" | "DecimalLiteral" | "DeclareClass" | "DeclareExportAllDeclaration" | "DeclareExportDeclaration" | "DeclareFunction" | "DeclareInterface" | "DeclareModule" | "DeclareModuleExports" | "DeclareOpaqueType" | "DeclareTypeAlias" | "DeclareVariable" | "DeclaredPredicate" | "Decorator" | "Directive" | "DirectiveLiteral" | "DoExpression" | "DoWhileStatement" | "EmptyStatement" | "EmptyTypeAnnotation" | "EnumBooleanBody" | "EnumBooleanMember" | "EnumDeclaration" | "EnumDefaultedMember" | "EnumNumberBody" | "EnumNumberMember" | "EnumStringBody" | "EnumStringMember" | "EnumSymbolBody" | "ExistsTypeAnnotation" | "ExportAllDeclaration" | "ExportDefaultDeclaration" | "ExportDefaultSpecifier" | "ExportNamedDeclaration" | "ExportNamespaceSpecifier" | "ExportSpecifier" | "ExpressionStatement" | "File" | "ForInStatement" | "ForOfStatement" | "ForStatement" | "FunctionDeclaration" | "FunctionExpression" | "FunctionTypeAnnotation" | "FunctionTypeParam" | "GenericTypeAnnotation" | "Identifier" | "IfStatement" | "Import" | "ImportAttribute" | "ImportDeclaration" | "ImportDefaultSpecifier" | "ImportExpression" | "ImportNamespaceSpecifier" | "ImportSpecifier" | "IndexedAccessType" | "InferredPredicate" | "InterfaceDeclaration" | "InterfaceExtends" | "InterfaceTypeAnnotation" | "InterpreterDirective" | "IntersectionTypeAnnotation" | "JSXAttribute" | "JSXClosingElement" | "JSXClosingFragment" | "JSXElement" | "JSXEmptyExpression" | "JSXExpressionContainer" | "JSXFragment" | "JSXMemberExpression" | "JSXNamespacedName" | "JSXOpeningElement" | "JSXOpeningFragment" | "JSXSpreadAttribute" | "JSXSpreadChild" | "LabeledStatement" | "LogicalExpression" | "MemberExpression" | "MetaProperty" | "MixedTypeAnnotation" | "ModuleExpression" | "NewExpression" | "Noop" | "NullLiteral" | "NullLiteralTypeAnnotation" | "NullableTypeAnnotation" | "NumberLiteral" | "NumberLiteralTypeAnnotation" | "NumberTypeAnnotation" | "ObjectExpression" | "ObjectMethod" | "ObjectPattern" | "ObjectProperty" | "ObjectTypeAnnotation" | "ObjectTypeCallProperty" | "ObjectTypeIndexer" | "ObjectTypeInternalSlot" | "ObjectTypeProperty" | "ObjectTypeSpreadProperty" | "OpaqueType" | "OptionalCallExpression" | "OptionalIndexedAccessType" | "OptionalMemberExpression" | "ParenthesizedExpression" | "PipelineBareFunction" | "PipelinePrimaryTopicReference" | "PipelineTopicExpression" | "Placeholder" | "PrivateName" | "Program" | "QualifiedTypeIdentifier" | "RecordExpression" | "RegExpLiteral" | "RegexLiteral" | "RestElement" | "RestProperty" | "ReturnStatement" | "SequenceExpression" | "SpreadElement" | "SpreadProperty" | "StaticBlock" | "StringLiteralTypeAnnotation" | "StringTypeAnnotation" | "Super" | "SwitchCase" | "SwitchStatement" | "SymbolTypeAnnotation" | "TSAnyKeyword" | "TSArrayType" | "TSAsExpression" | "TSBigIntKeyword" | "TSBooleanKeyword" | "TSCallSignatureDeclaration" | "TSConditionalType" | "TSConstructSignatureDeclaration" | "TSConstructorType" | "TSDeclareFunction" | "TSDeclareMethod" | "TSEnumDeclaration" | "TSEnumMember" | "TSExportAssignment" | "TSExpressionWithTypeArguments" | "TSExternalModuleReference" | "TSFunctionType" | "TSImportEqualsDeclaration" | "TSImportType" | "TSIndexSignature" | "TSIndexedAccessType" | "TSInferType" | "TSInstantiationExpression" | "TSInterfaceBody" | "TSInterfaceDeclaration" | "TSIntersectionType" | "TSIntrinsicKeyword" | "TSLiteralType" | "TSMappedType" | "TSMethodSignature" | "TSModuleBlock" | "TSModuleDeclaration" | "TSNamedTupleMember" | "TSNamespaceExportDeclaration" | "TSNeverKeyword" | "TSNonNullExpression" | "TSNullKeyword" | "TSNumberKeyword" | "TSObjectKeyword" | "TSOptionalType" | "TSParameterProperty" | "TSParenthesizedType" | "TSPropertySignature" | "TSQualifiedName" | "TSRestType" | "TSSatisfiesExpression" | "TSStringKeyword" | "TSSymbolKeyword" | "TSThisType" | "TSTupleType" | "TSTypeAliasDeclaration" | "TSTypeAnnotation" | "TSTypeAssertion" | "TSTypeLiteral" | "TSTypeOperator" | "TSTypeParameter" | "TSTypeParameterDeclaration" | "TSTypeParameterInstantiation" | "TSTypePredicate" | "TSTypeQuery" | "TSTypeReference" | "TSUndefinedKeyword" | "TSUnionType" | "TSUnknownKeyword" | "TSVoidKeyword" | "TaggedTemplateExpression" | "TemplateElement" | "TemplateLiteral" | "ThisExpression" | "ThisTypeAnnotation" | "ThrowStatement" | "TopicReference" | "TryStatement" | "TupleExpression" | "TupleTypeAnnotation" | "TypeAlias" | "TypeAnnotation" | "TypeCastExpression" | "TypeParameter" | "TypeParameterDeclaration" | "TypeParameterInstantiation" | "TypeofTypeAnnotation" | "UnaryExpression" | "UnionTypeAnnotation" | "UpdateExpression" | "V8IntrinsicIdentifier" | "VariableDeclaration" | "VariableDeclarator" | "Variance" | "VoidTypeAnnotation" | "WhileStatement" | "WithStatement" | "YieldExpression" | keyof Aliases)[];
declare const EXPORTDECLARATION_TYPES: ("StringLiteral" | "NumericLiteral" | "JSXText" | "JSXIdentifier" | "AnyTypeAnnotation" | "ArgumentPlaceholder" | "ArrayExpression" | "ArrayPattern" | "ArrayTypeAnnotation" | "ArrowFunctionExpression" | "AssignmentExpression" | "AssignmentPattern" | "AwaitExpression" | "BigIntLiteral" | "BinaryExpression" | "BindExpression" | "BlockStatement" | "BooleanLiteral" | "BooleanLiteralTypeAnnotation" | "BooleanTypeAnnotation" | "BreakStatement" | "CallExpression" | "CatchClause" | "ClassAccessorProperty" | "ClassBody" | "ClassDeclaration" | "ClassExpression" | "ClassImplements" | "ClassMethod" | "ClassPrivateMethod" | "ClassPrivateProperty" | "ClassProperty" | "ConditionalExpression" | "ContinueStatement" | "DebuggerStatement" | "DecimalLiteral" | "DeclareClass" | "DeclareExportAllDeclaration" | "DeclareExportDeclaration" | "DeclareFunction" | "DeclareInterface" | "DeclareModule" | "DeclareModuleExports" | "DeclareOpaqueType" | "DeclareTypeAlias" | "DeclareVariable" | "DeclaredPredicate" | "Decorator" | "Directive" | "DirectiveLiteral" | "DoExpression" | "DoWhileStatement" | "EmptyStatement" | "EmptyTypeAnnotation" | "EnumBooleanBody" | "EnumBooleanMember" | "EnumDeclaration" | "EnumDefaultedMember" | "EnumNumberBody" | "EnumNumberMember" | "EnumStringBody" | "EnumStringMember" | "EnumSymbolBody" | "ExistsTypeAnnotation" | "ExportAllDeclaration" | "ExportDefaultDeclaration" | "ExportDefaultSpecifier" | "ExportNamedDeclaration" | "ExportNamespaceSpecifier" | "ExportSpecifier" | "ExpressionStatement" | "File" | "ForInStatement" | "ForOfStatement" | "ForStatement" | "FunctionDeclaration" | "FunctionExpression" | "FunctionTypeAnnotation" | "FunctionTypeParam" | "GenericTypeAnnotation" | "Identifier" | "IfStatement" | "Import" | "ImportAttribute" | "ImportDeclaration" | "ImportDefaultSpecifier" | "ImportExpression" | "ImportNamespaceSpecifier" | "ImportSpecifier" | "IndexedAccessType" | "InferredPredicate" | "InterfaceDeclaration" | "InterfaceExtends" | "InterfaceTypeAnnotation" | "InterpreterDirective" | "IntersectionTypeAnnotation" | "JSXAttribute" | "JSXClosingElement" | "JSXClosingFragment" | "JSXElement" | "JSXEmptyExpression" | "JSXExpressionContainer" | "JSXFragment" | "JSXMemberExpression" | "JSXNamespacedName" | "JSXOpeningElement" | "JSXOpeningFragment" | "JSXSpreadAttribute" | "JSXSpreadChild" | "LabeledStatement" | "LogicalExpression" | "MemberExpression" | "MetaProperty" | "MixedTypeAnnotation" | "ModuleExpression" | "NewExpression" | "Noop" | "NullLiteral" | "NullLiteralTypeAnnotation" | "NullableTypeAnnotation" | "NumberLiteral" | "NumberLiteralTypeAnnotation" | "NumberTypeAnnotation" | "ObjectExpression" | "ObjectMethod" | "ObjectPattern" | "ObjectProperty" | "ObjectTypeAnnotation" | "ObjectTypeCallProperty" | "ObjectTypeIndexer" | "ObjectTypeInternalSlot" | "ObjectTypeProperty" | "ObjectTypeSpreadProperty" | "OpaqueType" | "OptionalCallExpression" | "OptionalIndexedAccessType" | "OptionalMemberExpression" | "ParenthesizedExpression" | "PipelineBareFunction" | "PipelinePrimaryTopicReference" | "PipelineTopicExpression" | "Placeholder" | "PrivateName" | "Program" | "QualifiedTypeIdentifier" | "RecordExpression" | "RegExpLiteral" | "RegexLiteral" | "RestElement" | "RestProperty" | "ReturnStatement" | "SequenceExpression" | "SpreadElement" | "SpreadProperty" | "StaticBlock" | "StringLiteralTypeAnnotation" | "StringTypeAnnotation" | "Super" | "SwitchCase" | "SwitchStatement" | "SymbolTypeAnnotation" | "TSAnyKeyword" | "TSArrayType" | "TSAsExpression" | "TSBigIntKeyword" | "TSBooleanKeyword" | "TSCallSignatureDeclaration" | "TSConditionalType" | "TSConstructSignatureDeclaration" | "TSConstructorType" | "TSDeclareFunction" | "TSDeclareMethod" | "TSEnumDeclaration" | "TSEnumMember" | "TSExportAssignment" | "TSExpressionWithTypeArguments" | "TSExternalModuleReference" | "TSFunctionType" | "TSImportEqualsDeclaration" | "TSImportType" | "TSIndexSignature" | "TSIndexedAccessType" | "TSInferType" | "TSInstantiationExpression" | "TSInterfaceBody" | "TSInterfaceDeclaration" | "TSIntersectionType" | "TSIntrinsicKeyword" | "TSLiteralType" | "TSMappedType" | "TSMethodSignature" | "TSModuleBlock" | "TSModuleDeclaration" | "TSNamedTupleMember" | "TSNamespaceExportDeclaration" | "TSNeverKeyword" | "TSNonNullExpression" | "TSNullKeyword" | "TSNumberKeyword" | "TSObjectKeyword" | "TSOptionalType" | "TSParameterProperty" | "TSParenthesizedType" | "TSPropertySignature" | "TSQualifiedName" | "TSRestType" | "TSSatisfiesExpression" | "TSStringKeyword" | "TSSymbolKeyword" | "TSThisType" | "TSTupleType" | "TSTypeAliasDeclaration" | "TSTypeAnnotation" | "TSTypeAssertion" | "TSTypeLiteral" | "TSTypeOperator" | "TSTypeParameter" | "TSTypeParameterDeclaration" | "TSTypeParameterInstantiation" | "TSTypePredicate" | "TSTypeQuery" | "TSTypeReference" | "TSUndefinedKeyword" | "TSUnionType" | "TSUnknownKeyword" | "TSVoidKeyword" | "TaggedTemplateExpression" | "TemplateElement" | "TemplateLiteral" | "ThisExpression" | "ThisTypeAnnotation" | "ThrowStatement" | "TopicReference" | "TryStatement" | "TupleExpression" | "TupleTypeAnnotation" | "TypeAlias" | "TypeAnnotation" | "TypeCastExpression" | "TypeParameter" | "TypeParameterDeclaration" | "TypeParameterInstantiation" | "TypeofTypeAnnotation" | "UnaryExpression" | "UnionTypeAnnotation" | "UpdateExpression" | "V8IntrinsicIdentifier" | "VariableDeclaration" | "VariableDeclarator" | "Variance" | "VoidTypeAnnotation" | "WhileStatement" | "WithStatement" | "YieldExpression" | keyof Aliases)[];
declare const MODULESPECIFIER_TYPES: ("StringLiteral" | "NumericLiteral" | "JSXText" | "JSXIdentifier" | "AnyTypeAnnotation" | "ArgumentPlaceholder" | "ArrayExpression" | "ArrayPattern" | "ArrayTypeAnnotation" | "ArrowFunctionExpression" | "AssignmentExpression" | "AssignmentPattern" | "AwaitExpression" | "BigIntLiteral" | "BinaryExpression" | "BindExpression" | "BlockStatement" | "BooleanLiteral" | "BooleanLiteralTypeAnnotation" | "BooleanTypeAnnotation" | "BreakStatement" | "CallExpression" | "CatchClause" | "ClassAccessorProperty" | "ClassBody" | "ClassDeclaration" | "ClassExpression" | "ClassImplements" | "ClassMethod" | "ClassPrivateMethod" | "ClassPrivateProperty" | "ClassProperty" | "ConditionalExpression" | "ContinueStatement" | "DebuggerStatement" | "DecimalLiteral" | "DeclareClass" | "DeclareExportAllDeclaration" | "DeclareExportDeclaration" | "DeclareFunction" | "DeclareInterface" | "DeclareModule" | "DeclareModuleExports" | "DeclareOpaqueType" | "DeclareTypeAlias" | "DeclareVariable" | "DeclaredPredicate" | "Decorator" | "Directive" | "DirectiveLiteral" | "DoExpression" | "DoWhileStatement" | "EmptyStatement" | "EmptyTypeAnnotation" | "EnumBooleanBody" | "EnumBooleanMember" | "EnumDeclaration" | "EnumDefaultedMember" | "EnumNumberBody" | "EnumNumberMember" | "EnumStringBody" | "EnumStringMember" | "EnumSymbolBody" | "ExistsTypeAnnotation" | "ExportAllDeclaration" | "ExportDefaultDeclaration" | "ExportDefaultSpecifier" | "ExportNamedDeclaration" | "ExportNamespaceSpecifier" | "ExportSpecifier" | "ExpressionStatement" | "File" | "ForInStatement" | "ForOfStatement" | "ForStatement" | "FunctionDeclaration" | "FunctionExpression" | "FunctionTypeAnnotation" | "FunctionTypeParam" | "GenericTypeAnnotation" | "Identifier" | "IfStatement" | "Import" | "ImportAttribute" | "ImportDeclaration" | "ImportDefaultSpecifier" | "ImportExpression" | "ImportNamespaceSpecifier" | "ImportSpecifier" | "IndexedAccessType" | "InferredPredicate" | "InterfaceDeclaration" | "InterfaceExtends" | "InterfaceTypeAnnotation" | "InterpreterDirective" | "IntersectionTypeAnnotation" | "JSXAttribute" | "JSXClosingElement" | "JSXClosingFragment" | "JSXElement" | "JSXEmptyExpression" | "JSXExpressionContainer" | "JSXFragment" | "JSXMemberExpression" | "JSXNamespacedName" | "JSXOpeningElement" | "JSXOpeningFragment" | "JSXSpreadAttribute" | "JSXSpreadChild" | "LabeledStatement" | "LogicalExpression" | "MemberExpression" | "MetaProperty" | "MixedTypeAnnotation" | "ModuleExpression" | "NewExpression" | "Noop" | "NullLiteral" | "NullLiteralTypeAnnotation" | "NullableTypeAnnotation" | "NumberLiteral" | "NumberLiteralTypeAnnotation" | "NumberTypeAnnotation" | "ObjectExpression" | "ObjectMethod" | "ObjectPattern" | "ObjectProperty" | "ObjectTypeAnnotation" | "ObjectTypeCallProperty" | "ObjectTypeIndexer" | "ObjectTypeInternalSlot" | "ObjectTypeProperty" | "ObjectTypeSpreadProperty" | "OpaqueType" | "OptionalCallExpression" | "OptionalIndexedAccessType" | "OptionalMemberExpression" | "ParenthesizedExpression" | "PipelineBareFunction" | "PipelinePrimaryTopicReference" | "PipelineTopicExpression" | "Placeholder" | "PrivateName" | "Program" | "QualifiedTypeIdentifier" | "RecordExpression" | "RegExpLiteral" | "RegexLiteral" | "RestElement" | "RestProperty" | "ReturnStatement" | "SequenceExpression" | "SpreadElement" | "SpreadProperty" | "StaticBlock" | "StringLiteralTypeAnnotation" | "StringTypeAnnotation" | "Super" | "SwitchCase" | "SwitchStatement" | "SymbolTypeAnnotation" | "TSAnyKeyword" | "TSArrayType" | "TSAsExpression" | "TSBigIntKeyword" | "TSBooleanKeyword" | "TSCallSignatureDeclaration" | "TSConditionalType" | "TSConstructSignatureDeclaration" | "TSConstructorType" | "TSDeclareFunction" | "TSDeclareMethod" | "TSEnumDeclaration" | "TSEnumMember" | "TSExportAssignment" | "TSExpressionWithTypeArguments" | "TSExternalModuleReference" | "TSFunctionType" | "TSImportEqualsDeclaration" | "TSImportType" | "TSIndexSignature" | "TSIndexedAccessType" | "TSInferType" | "TSInstantiationExpression" | "TSInterfaceBody" | "TSInterfaceDeclaration" | "TSIntersectionType" | "TSIntrinsicKeyword" | "TSLiteralType" | "TSMappedType" | "TSMethodSignature" | "TSModuleBlock" | "TSModuleDeclaration" | "TSNamedTupleMember" | "TSNamespaceExportDeclaration" | "TSNeverKeyword" | "TSNonNullExpression" | "TSNullKeyword" | "TSNumberKeyword" | "TSObjectKeyword" | "TSOptionalType" | "TSParameterProperty" | "TSParenthesizedType" | "TSPropertySignature" | "TSQualifiedName" | "TSRestType" | "TSSatisfiesExpression" | "TSStringKeyword" | "TSSymbolKeyword" | "TSThisType" | "TSTupleType" | "TSTypeAliasDeclaration" | "TSTypeAnnotation" | "TSTypeAssertion" | "TSTypeLiteral" | "TSTypeOperator" | "TSTypeParameter" | "TSTypeParameterDeclaration" | "TSTypeParameterInstantiation" | "TSTypePredicate" | "TSTypeQuery" | "TSTypeReference" | "TSUndefinedKeyword" | "TSUnionType" | "TSUnknownKeyword" | "TSVoidKeyword" | "TaggedTemplateExpression" | "TemplateElement" | "TemplateLiteral" | "ThisExpression" | "ThisTypeAnnotation" | "ThrowStatement" | "TopicReference" | "TryStatement" | "TupleExpression" | "TupleTypeAnnotation" | "TypeAlias" | "TypeAnnotation" | "TypeCastExpression" | "TypeParameter" | "TypeParameterDeclaration" | "TypeParameterInstantiation" | "TypeofTypeAnnotation" | "UnaryExpression" | "UnionTypeAnnotation" | "UpdateExpression" | "V8IntrinsicIdentifier" | "VariableDeclaration" | "VariableDeclarator" | "Variance" | "VoidTypeAnnotation" | "WhileStatement" | "WithStatement" | "YieldExpression" | keyof Aliases)[];
declare const ACCESSOR_TYPES: ("StringLiteral" | "NumericLiteral" | "JSXText" | "JSXIdentifier" | "AnyTypeAnnotation" | "ArgumentPlaceholder" | "ArrayExpression" | "ArrayPattern" | "ArrayTypeAnnotation" | "ArrowFunctionExpression" | "AssignmentExpression" | "AssignmentPattern" | "AwaitExpression" | "BigIntLiteral" | "BinaryExpression" | "BindExpression" | "BlockStatement" | "BooleanLiteral" | "BooleanLiteralTypeAnnotation" | "BooleanTypeAnnotation" | "BreakStatement" | "CallExpression" | "CatchClause" | "ClassAccessorProperty" | "ClassBody" | "ClassDeclaration" | "ClassExpression" | "ClassImplements" | "ClassMethod" | "ClassPrivateMethod" | "ClassPrivateProperty" | "ClassProperty" | "ConditionalExpression" | "ContinueStatement" | "DebuggerStatement" | "DecimalLiteral" | "DeclareClass" | "DeclareExportAllDeclaration" | "DeclareExportDeclaration" | "DeclareFunction" | "DeclareInterface" | "DeclareModule" | "DeclareModuleExports" | "DeclareOpaqueType" | "DeclareTypeAlias" | "DeclareVariable" | "DeclaredPredicate" | "Decorator" | "Directive" | "DirectiveLiteral" | "DoExpression" | "DoWhileStatement" | "EmptyStatement" | "EmptyTypeAnnotation" | "EnumBooleanBody" | "EnumBooleanMember" | "EnumDeclaration" | "EnumDefaultedMember" | "EnumNumberBody" | "EnumNumberMember" | "EnumStringBody" | "EnumStringMember" | "EnumSymbolBody" | "ExistsTypeAnnotation" | "ExportAllDeclaration" | "ExportDefaultDeclaration" | "ExportDefaultSpecifier" | "ExportNamedDeclaration" | "ExportNamespaceSpecifier" | "ExportSpecifier" | "ExpressionStatement" | "File" | "ForInStatement" | "ForOfStatement" | "ForStatement" | "FunctionDeclaration" | "FunctionExpression" | "FunctionTypeAnnotation" | "FunctionTypeParam" | "GenericTypeAnnotation" | "Identifier" | "IfStatement" | "Import" | "ImportAttribute" | "ImportDeclaration" | "ImportDefaultSpecifier" | "ImportExpression" | "ImportNamespaceSpecifier" | "ImportSpecifier" | "IndexedAccessType" | "InferredPredicate" | "InterfaceDeclaration" | "InterfaceExtends" | "InterfaceTypeAnnotation" | "InterpreterDirective" | "IntersectionTypeAnnotation" | "JSXAttribute" | "JSXClosingElement" | "JSXClosingFragment" | "JSXElement" | "JSXEmptyExpression" | "JSXExpressionContainer" | "JSXFragment" | "JSXMemberExpression" | "JSXNamespacedName" | "JSXOpeningElement" | "JSXOpeningFragment" | "JSXSpreadAttribute" | "JSXSpreadChild" | "LabeledStatement" | "LogicalExpression" | "MemberExpression" | "MetaProperty" | "MixedTypeAnnotation" | "ModuleExpression" | "NewExpression" | "Noop" | "NullLiteral" | "NullLiteralTypeAnnotation" | "NullableTypeAnnotation" | "NumberLiteral" | "NumberLiteralTypeAnnotation" | "NumberTypeAnnotation" | "ObjectExpression" | "ObjectMethod" | "ObjectPattern" | "ObjectProperty" | "ObjectTypeAnnotation" | "ObjectTypeCallProperty" | "ObjectTypeIndexer" | "ObjectTypeInternalSlot" | "ObjectTypeProperty" | "ObjectTypeSpreadProperty" | "OpaqueType" | "OptionalCallExpression" | "OptionalIndexedAccessType" | "OptionalMemberExpression" | "ParenthesizedExpression" | "PipelineBareFunction" | "PipelinePrimaryTopicReference" | "PipelineTopicExpression" | "Placeholder" | "PrivateName" | "Program" | "QualifiedTypeIdentifier" | "RecordExpression" | "RegExpLiteral" | "RegexLiteral" | "RestElement" | "RestProperty" | "ReturnStatement" | "SequenceExpression" | "SpreadElement" | "SpreadProperty" | "StaticBlock" | "StringLiteralTypeAnnotation" | "StringTypeAnnotation" | "Super" | "SwitchCase" | "SwitchStatement" | "SymbolTypeAnnotation" | "TSAnyKeyword" | "TSArrayType" | "TSAsExpression" | "TSBigIntKeyword" | "TSBooleanKeyword" | "TSCallSignatureDeclaration" | "TSConditionalType" | "TSConstructSignatureDeclaration" | "TSConstructorType" | "TSDeclareFunction" | "TSDeclareMethod" | "TSEnumDeclaration" | "TSEnumMember" | "TSExportAssignment" | "TSExpressionWithTypeArguments" | "TSExternalModuleReference" | "TSFunctionType" | "TSImportEqualsDeclaration" | "TSImportType" | "TSIndexSignature" | "TSIndexedAccessType" | "TSInferType" | "TSInstantiationExpression" | "TSInterfaceBody" | "TSInterfaceDeclaration" | "TSIntersectionType" | "TSIntrinsicKeyword" | "TSLiteralType" | "TSMappedType" | "TSMethodSignature" | "TSModuleBlock" | "TSModuleDeclaration" | "TSNamedTupleMember" | "TSNamespaceExportDeclaration" | "TSNeverKeyword" | "TSNonNullExpression" | "TSNullKeyword" | "TSNumberKeyword" | "TSObjectKeyword" | "TSOptionalType" | "TSParameterProperty" | "TSParenthesizedType" | "TSPropertySignature" | "TSQualifiedName" | "TSRestType" | "TSSatisfiesExpression" | "TSStringKeyword" | "TSSymbolKeyword" | "TSThisType" | "TSTupleType" | "TSTypeAliasDeclaration" | "TSTypeAnnotation" | "TSTypeAssertion" | "TSTypeLiteral" | "TSTypeOperator" | "TSTypeParameter" | "TSTypeParameterDeclaration" | "TSTypeParameterInstantiation" | "TSTypePredicate" | "TSTypeQuery" | "TSTypeReference" | "TSUndefinedKeyword" | "TSUnionType" | "TSUnknownKeyword" | "TSVoidKeyword" | "TaggedTemplateExpression" | "TemplateElement" | "TemplateLiteral" | "ThisExpression" | "ThisTypeAnnotation" | "ThrowStatement" | "TopicReference" | "TryStatement" | "TupleExpression" | "TupleTypeAnnotation" | "TypeAlias" | "TypeAnnotation" | "TypeCastExpression" | "TypeParameter" | "TypeParameterDeclaration" | "TypeParameterInstantiation" | "TypeofTypeAnnotation" | "UnaryExpression" | "UnionTypeAnnotation" | "UpdateExpression" | "V8IntrinsicIdentifier" | "VariableDeclaration" | "VariableDeclarator" | "Variance" | "VoidTypeAnnotation" | "WhileStatement" | "WithStatement" | "YieldExpression" | keyof Aliases)[];
declare const PRIVATE_TYPES: ("StringLiteral" | "NumericLiteral" | "JSXText" | "JSXIdentifier" | "AnyTypeAnnotation" | "ArgumentPlaceholder" | "ArrayExpression" | "ArrayPattern" | "ArrayTypeAnnotation" | "ArrowFunctionExpression" | "AssignmentExpression" | "AssignmentPattern" | "AwaitExpression" | "BigIntLiteral" | "BinaryExpression" | "BindExpression" | "BlockStatement" | "BooleanLiteral" | "BooleanLiteralTypeAnnotation" | "BooleanTypeAnnotation" | "BreakStatement" | "CallExpression" | "CatchClause" | "ClassAccessorProperty" | "ClassBody" | "ClassDeclaration" | "ClassExpression" | "ClassImplements" | "ClassMethod" | "ClassPrivateMethod" | "ClassPrivateProperty" | "ClassProperty" | "ConditionalExpression" | "ContinueStatement" | "DebuggerStatement" | "DecimalLiteral" | "DeclareClass" | "DeclareExportAllDeclaration" | "DeclareExportDeclaration" | "DeclareFunction" | "DeclareInterface" | "DeclareModule" | "DeclareModuleExports" | "DeclareOpaqueType" | "DeclareTypeAlias" | "DeclareVariable" | "DeclaredPredicate" | "Decorator" | "Directive" | "DirectiveLiteral" | "DoExpression" | "DoWhileStatement" | "EmptyStatement" | "EmptyTypeAnnotation" | "EnumBooleanBody" | "EnumBooleanMember" | "EnumDeclaration" | "EnumDefaultedMember" | "EnumNumberBody" | "EnumNumberMember" | "EnumStringBody" | "EnumStringMember" | "EnumSymbolBody" | "ExistsTypeAnnotation" | "ExportAllDeclaration" | "ExportDefaultDeclaration" | "ExportDefaultSpecifier" | "ExportNamedDeclaration" | "ExportNamespaceSpecifier" | "ExportSpecifier" | "ExpressionStatement" | "File" | "ForInStatement" | "ForOfStatement" | "ForStatement" | "FunctionDeclaration" | "FunctionExpression" | "FunctionTypeAnnotation" | "FunctionTypeParam" | "GenericTypeAnnotation" | "Identifier" | "IfStatement" | "Import" | "ImportAttribute" | "ImportDeclaration" | "ImportDefaultSpecifier" | "ImportExpression" | "ImportNamespaceSpecifier" | "ImportSpecifier" | "IndexedAccessType" | "InferredPredicate" | "InterfaceDeclaration" | "InterfaceExtends" | "InterfaceTypeAnnotation" | "InterpreterDirective" | "IntersectionTypeAnnotation" | "JSXAttribute" | "JSXClosingElement" | "JSXClosingFragment" | "JSXElement" | "JSXEmptyExpression" | "JSXExpressionContainer" | "JSXFragment" | "JSXMemberExpression" | "JSXNamespacedName" | "JSXOpeningElement" | "JSXOpeningFragment" | "JSXSpreadAttribute" | "JSXSpreadChild" | "LabeledStatement" | "LogicalExpression" | "MemberExpression" | "MetaProperty" | "MixedTypeAnnotation" | "ModuleExpression" | "NewExpression" | "Noop" | "NullLiteral" | "NullLiteralTypeAnnotation" | "NullableTypeAnnotation" | "NumberLiteral" | "NumberLiteralTypeAnnotation" | "NumberTypeAnnotation" | "ObjectExpression" | "ObjectMethod" | "ObjectPattern" | "ObjectProperty" | "ObjectTypeAnnotation" | "ObjectTypeCallProperty" | "ObjectTypeIndexer" | "ObjectTypeInternalSlot" | "ObjectTypeProperty" | "ObjectTypeSpreadProperty" | "OpaqueType" | "OptionalCallExpression" | "OptionalIndexedAccessType" | "OptionalMemberExpression" | "ParenthesizedExpression" | "PipelineBareFunction" | "PipelinePrimaryTopicReference" | "PipelineTopicExpression" | "Placeholder" | "PrivateName" | "Program" | "QualifiedTypeIdentifier" | "RecordExpression" | "RegExpLiteral" | "RegexLiteral" | "RestElement" | "RestProperty" | "ReturnStatement" | "SequenceExpression" | "SpreadElement" | "SpreadProperty" | "StaticBlock" | "StringLiteralTypeAnnotation" | "StringTypeAnnotation" | "Super" | "SwitchCase" | "SwitchStatement" | "SymbolTypeAnnotation" | "TSAnyKeyword" | "TSArrayType" | "TSAsExpression" | "TSBigIntKeyword" | "TSBooleanKeyword" | "TSCallSignatureDeclaration" | "TSConditionalType" | "TSConstructSignatureDeclaration" | "TSConstructorType" | "TSDeclareFunction" | "TSDeclareMethod" | "TSEnumDeclaration" | "TSEnumMember" | "TSExportAssignment" | "TSExpressionWithTypeArguments" | "TSExternalModuleReference" | "TSFunctionType" | "TSImportEqualsDeclaration" | "TSImportType" | "TSIndexSignature" | "TSIndexedAccessType" | "TSInferType" | "TSInstantiationExpression" | "TSInterfaceBody" | "TSInterfaceDeclaration" | "TSIntersectionType" | "TSIntrinsicKeyword" | "TSLiteralType" | "TSMappedType" | "TSMethodSignature" | "TSModuleBlock" | "TSModuleDeclaration" | "TSNamedTupleMember" | "TSNamespaceExportDeclaration" | "TSNeverKeyword" | "TSNonNullExpression" | "TSNullKeyword" | "TSNumberKeyword" | "TSObjectKeyword" | "TSOptionalType" | "TSParameterProperty" | "TSParenthesizedType" | "TSPropertySignature" | "TSQualifiedName" | "TSRestType" | "TSSatisfiesExpression" | "TSStringKeyword" | "TSSymbolKeyword" | "TSThisType" | "TSTupleType" | "TSTypeAliasDeclaration" | "TSTypeAnnotation" | "TSTypeAssertion" | "TSTypeLiteral" | "TSTypeOperator" | "TSTypeParameter" | "TSTypeParameterDeclaration" | "TSTypeParameterInstantiation" | "TSTypePredicate" | "TSTypeQuery" | "TSTypeReference" | "TSUndefinedKeyword" | "TSUnionType" | "TSUnknownKeyword" | "TSVoidKeyword" | "TaggedTemplateExpression" | "TemplateElement" | "TemplateLiteral" | "ThisExpression" | "ThisTypeAnnotation" | "ThrowStatement" | "TopicReference" | "TryStatement" | "TupleExpression" | "TupleTypeAnnotation" | "TypeAlias" | "TypeAnnotation" | "TypeCastExpression" | "TypeParameter" | "TypeParameterDeclaration" | "TypeParameterInstantiation" | "TypeofTypeAnnotation" | "UnaryExpression" | "UnionTypeAnnotation" | "UpdateExpression" | "V8IntrinsicIdentifier" | "VariableDeclaration" | "VariableDeclarator" | "Variance" | "VoidTypeAnnotation" | "WhileStatement" | "WithStatement" | "YieldExpression" | keyof Aliases)[];
declare const FLOW_TYPES: ("StringLiteral" | "NumericLiteral" | "JSXText" | "JSXIdentifier" | "AnyTypeAnnotation" | "ArgumentPlaceholder" | "ArrayExpression" | "ArrayPattern" | "ArrayTypeAnnotation" | "ArrowFunctionExpression" | "AssignmentExpression" | "AssignmentPattern" | "AwaitExpression" | "BigIntLiteral" | "BinaryExpression" | "BindExpression" | "BlockStatement" | "BooleanLiteral" | "BooleanLiteralTypeAnnotation" | "BooleanTypeAnnotation" | "BreakStatement" | "CallExpression" | "CatchClause" | "ClassAccessorProperty" | "ClassBody" | "ClassDeclaration" | "ClassExpression" | "ClassImplements" | "ClassMethod" | "ClassPrivateMethod" | "ClassPrivateProperty" | "ClassProperty" | "ConditionalExpression" | "ContinueStatement" | "DebuggerStatement" | "DecimalLiteral" | "DeclareClass" | "DeclareExportAllDeclaration" | "DeclareExportDeclaration" | "DeclareFunction" | "DeclareInterface" | "DeclareModule" | "DeclareModuleExports" | "DeclareOpaqueType" | "DeclareTypeAlias" | "DeclareVariable" | "DeclaredPredicate" | "Decorator" | "Directive" | "DirectiveLiteral" | "DoExpression" | "DoWhileStatement" | "EmptyStatement" | "EmptyTypeAnnotation" | "EnumBooleanBody" | "EnumBooleanMember" | "EnumDeclaration" | "EnumDefaultedMember" | "EnumNumberBody" | "EnumNumberMember" | "EnumStringBody" | "EnumStringMember" | "EnumSymbolBody" | "ExistsTypeAnnotation" | "ExportAllDeclaration" | "ExportDefaultDeclaration" | "ExportDefaultSpecifier" | "ExportNamedDeclaration" | "ExportNamespaceSpecifier" | "ExportSpecifier" | "ExpressionStatement" | "File" | "ForInStatement" | "ForOfStatement" | "ForStatement" | "FunctionDeclaration" | "FunctionExpression" | "FunctionTypeAnnotation" | "FunctionTypeParam" | "GenericTypeAnnotation" | "Identifier" | "IfStatement" | "Import" | "ImportAttribute" | "ImportDeclaration" | "ImportDefaultSpecifier" | "ImportExpression" | "ImportNamespaceSpecifier" | "ImportSpecifier" | "IndexedAccessType" | "InferredPredicate" | "InterfaceDeclaration" | "InterfaceExtends" | "InterfaceTypeAnnotation" | "InterpreterDirective" | "IntersectionTypeAnnotation" | "JSXAttribute" | "JSXClosingElement" | "JSXClosingFragment" | "JSXElement" | "JSXEmptyExpression" | "JSXExpressionContainer" | "JSXFragment" | "JSXMemberExpression" | "JSXNamespacedName" | "JSXOpeningElement" | "JSXOpeningFragment" | "JSXSpreadAttribute" | "JSXSpreadChild" | "LabeledStatement" | "LogicalExpression" | "MemberExpression" | "MetaProperty" | "MixedTypeAnnotation" | "ModuleExpression" | "NewExpression" | "Noop" | "NullLiteral" | "NullLiteralTypeAnnotation" | "NullableTypeAnnotation" | "NumberLiteral" | "NumberLiteralTypeAnnotation" | "NumberTypeAnnotation" | "ObjectExpression" | "ObjectMethod" | "ObjectPattern" | "ObjectProperty" | "ObjectTypeAnnotation" | "ObjectTypeCallProperty" | "ObjectTypeIndexer" | "ObjectTypeInternalSlot" | "ObjectTypeProperty" | "ObjectTypeSpreadProperty" | "OpaqueType" | "OptionalCallExpression" | "OptionalIndexedAccessType" | "OptionalMemberExpression" | "ParenthesizedExpression" | "PipelineBareFunction" | "PipelinePrimaryTopicReference" | "PipelineTopicExpression" | "Placeholder" | "PrivateName" | "Program" | "QualifiedTypeIdentifier" | "RecordExpression" | "RegExpLiteral" | "RegexLiteral" | "RestElement" | "RestProperty" | "ReturnStatement" | "SequenceExpression" | "SpreadElement" | "SpreadProperty" | "StaticBlock" | "StringLiteralTypeAnnotation" | "StringTypeAnnotation" | "Super" | "SwitchCase" | "SwitchStatement" | "SymbolTypeAnnotation" | "TSAnyKeyword" | "TSArrayType" | "TSAsExpression" | "TSBigIntKeyword" | "TSBooleanKeyword" | "TSCallSignatureDeclaration" | "TSConditionalType" | "TSConstructSignatureDeclaration" | "TSConstructorType" | "TSDeclareFunction" | "TSDeclareMethod" | "TSEnumDeclaration" | "TSEnumMember" | "TSExportAssignment" | "TSExpressionWithTypeArguments" | "TSExternalModuleReference" | "TSFunctionType" | "TSImportEqualsDeclaration" | "TSImportType" | "TSIndexSignature" | "TSIndexedAccessType" | "TSInferType" | "TSInstantiationExpression" | "TSInterfaceBody" | "TSInterfaceDeclaration" | "TSIntersectionType" | "TSIntrinsicKeyword" | "TSLiteralType" | "TSMappedType" | "TSMethodSignature" | "TSModuleBlock" | "TSModuleDeclaration" | "TSNamedTupleMember" | "TSNamespaceExportDeclaration" | "TSNeverKeyword" | "TSNonNullExpression" | "TSNullKeyword" | "TSNumberKeyword" | "TSObjectKeyword" | "TSOptionalType" | "TSParameterProperty" | "TSParenthesizedType" | "TSPropertySignature" | "TSQualifiedName" | "TSRestType" | "TSSatisfiesExpression" | "TSStringKeyword" | "TSSymbolKeyword" | "TSThisType" | "TSTupleType" | "TSTypeAliasDeclaration" | "TSTypeAnnotation" | "TSTypeAssertion" | "TSTypeLiteral" | "TSTypeOperator" | "TSTypeParameter" | "TSTypeParameterDeclaration" | "TSTypeParameterInstantiation" | "TSTypePredicate" | "TSTypeQuery" | "TSTypeReference" | "TSUndefinedKeyword" | "TSUnionType" | "TSUnknownKeyword" | "TSVoidKeyword" | "TaggedTemplateExpression" | "TemplateElement" | "TemplateLiteral" | "ThisExpression" | "ThisTypeAnnotation" | "ThrowStatement" | "TopicReference" | "TryStatement" | "TupleExpression" | "TupleTypeAnnotation" | "TypeAlias" | "TypeAnnotation" | "TypeCastExpression" | "TypeParameter" | "TypeParameterDeclaration" | "TypeParameterInstantiation" | "TypeofTypeAnnotation" | "UnaryExpression" | "UnionTypeAnnotation" | "UpdateExpression" | "V8IntrinsicIdentifier" | "VariableDeclaration" | "VariableDeclarator" | "Variance" | "VoidTypeAnnotation" | "WhileStatement" | "WithStatement" | "YieldExpression" | keyof Aliases)[];
declare const FLOWTYPE_TYPES: ("StringLiteral" | "NumericLiteral" | "JSXText" | "JSXIdentifier" | "AnyTypeAnnotation" | "ArgumentPlaceholder" | "ArrayExpression" | "ArrayPattern" | "ArrayTypeAnnotation" | "ArrowFunctionExpression" | "AssignmentExpression" | "AssignmentPattern" | "AwaitExpression" | "BigIntLiteral" | "BinaryExpression" | "BindExpression" | "BlockStatement" | "BooleanLiteral" | "BooleanLiteralTypeAnnotation" | "BooleanTypeAnnotation" | "BreakStatement" | "CallExpression" | "CatchClause" | "ClassAccessorProperty" | "ClassBody" | "ClassDeclaration" | "ClassExpression" | "ClassImplements" | "ClassMethod" | "ClassPrivateMethod" | "ClassPrivateProperty" | "ClassProperty" | "ConditionalExpression" | "ContinueStatement" | "DebuggerStatement" | "DecimalLiteral" | "DeclareClass" | "DeclareExportAllDeclaration" | "DeclareExportDeclaration" | "DeclareFunction" | "DeclareInterface" | "DeclareModule" | "DeclareModuleExports" | "DeclareOpaqueType" | "DeclareTypeAlias" | "DeclareVariable" | "DeclaredPredicate" | "Decorator" | "Directive" | "DirectiveLiteral" | "DoExpression" | "DoWhileStatement" | "EmptyStatement" | "EmptyTypeAnnotation" | "EnumBooleanBody" | "EnumBooleanMember" | "EnumDeclaration" | "EnumDefaultedMember" | "EnumNumberBody" | "EnumNumberMember" | "EnumStringBody" | "EnumStringMember" | "EnumSymbolBody" | "ExistsTypeAnnotation" | "ExportAllDeclaration" | "ExportDefaultDeclaration" | "ExportDefaultSpecifier" | "ExportNamedDeclaration" | "ExportNamespaceSpecifier" | "ExportSpecifier" | "ExpressionStatement" | "File" | "ForInStatement" | "ForOfStatement" | "ForStatement" | "FunctionDeclaration" | "FunctionExpression" | "FunctionTypeAnnotation" | "FunctionTypeParam" | "GenericTypeAnnotation" | "Identifier" | "IfStatement" | "Import" | "ImportAttribute" | "ImportDeclaration" | "ImportDefaultSpecifier" | "ImportExpression" | "ImportNamespaceSpecifier" | "ImportSpecifier" | "IndexedAccessType" | "InferredPredicate" | "InterfaceDeclaration" | "InterfaceExtends" | "InterfaceTypeAnnotation" | "InterpreterDirective" | "IntersectionTypeAnnotation" | "JSXAttribute" | "JSXClosingElement" | "JSXClosingFragment" | "JSXElement" | "JSXEmptyExpression" | "JSXExpressionContainer" | "JSXFragment" | "JSXMemberExpression" | "JSXNamespacedName" | "JSXOpeningElement" | "JSXOpeningFragment" | "JSXSpreadAttribute" | "JSXSpreadChild" | "LabeledStatement" | "LogicalExpression" | "MemberExpression" | "MetaProperty" | "MixedTypeAnnotation" | "ModuleExpression" | "NewExpression" | "Noop" | "NullLiteral" | "NullLiteralTypeAnnotation" | "NullableTypeAnnotation" | "NumberLiteral" | "NumberLiteralTypeAnnotation" | "NumberTypeAnnotation" | "ObjectExpression" | "ObjectMethod" | "ObjectPattern" | "ObjectProperty" | "ObjectTypeAnnotation" | "ObjectTypeCallProperty" | "ObjectTypeIndexer" | "ObjectTypeInternalSlot" | "ObjectTypeProperty" | "ObjectTypeSpreadProperty" | "OpaqueType" | "OptionalCallExpression" | "OptionalIndexedAccessType" | "OptionalMemberExpression" | "ParenthesizedExpression" | "PipelineBareFunction" | "PipelinePrimaryTopicReference" | "PipelineTopicExpression" | "Placeholder" | "PrivateName" | "Program" | "QualifiedTypeIdentifier" | "RecordExpression" | "RegExpLiteral" | "RegexLiteral" | "RestElement" | "RestProperty" | "ReturnStatement" | "SequenceExpression" | "SpreadElement" | "SpreadProperty" | "StaticBlock" | "StringLiteralTypeAnnotation" | "StringTypeAnnotation" | "Super" | "SwitchCase" | "SwitchStatement" | "SymbolTypeAnnotation" | "TSAnyKeyword" | "TSArrayType" | "TSAsExpression" | "TSBigIntKeyword" | "TSBooleanKeyword" | "TSCallSignatureDeclaration" | "TSConditionalType" | "TSConstructSignatureDeclaration" | "TSConstructorType" | "TSDeclareFunction" | "TSDeclareMethod" | "TSEnumDeclaration" | "TSEnumMember" | "TSExportAssignment" | "TSExpressionWithTypeArguments" | "TSExternalModuleReference" | "TSFunctionType" | "TSImportEqualsDeclaration" | "TSImportType" | "TSIndexSignature" | "TSIndexedAccessType" | "TSInferType" | "TSInstantiationExpression" | "TSInterfaceBody" | "TSInterfaceDeclaration" | "TSIntersectionType" | "TSIntrinsicKeyword" | "TSLiteralType" | "TSMappedType" | "TSMethodSignature" | "TSModuleBlock" | "TSModuleDeclaration" | "TSNamedTupleMember" | "TSNamespaceExportDeclaration" | "TSNeverKeyword" | "TSNonNullExpression" | "TSNullKeyword" | "TSNumberKeyword" | "TSObjectKeyword" | "TSOptionalType" | "TSParameterProperty" | "TSParenthesizedType" | "TSPropertySignature" | "TSQualifiedName" | "TSRestType" | "TSSatisfiesExpression" | "TSStringKeyword" | "TSSymbolKeyword" | "TSThisType" | "TSTupleType" | "TSTypeAliasDeclaration" | "TSTypeAnnotation" | "TSTypeAssertion" | "TSTypeLiteral" | "TSTypeOperator" | "TSTypeParameter" | "TSTypeParameterDeclaration" | "TSTypeParameterInstantiation" | "TSTypePredicate" | "TSTypeQuery" | "TSTypeReference" | "TSUndefinedKeyword" | "TSUnionType" | "TSUnknownKeyword" | "TSVoidKeyword" | "TaggedTemplateExpression" | "TemplateElement" | "TemplateLiteral" | "ThisExpression" | "ThisTypeAnnotation" | "ThrowStatement" | "TopicReference" | "TryStatement" | "TupleExpression" | "TupleTypeAnnotation" | "TypeAlias" | "TypeAnnotation" | "TypeCastExpression" | "TypeParameter" | "TypeParameterDeclaration" | "TypeParameterInstantiation" | "TypeofTypeAnnotation" | "UnaryExpression" | "UnionTypeAnnotation" | "UpdateExpression" | "V8IntrinsicIdentifier" | "VariableDeclaration" | "VariableDeclarator" | "Variance" | "VoidTypeAnnotation" | "WhileStatement" | "WithStatement" | "YieldExpression" | keyof Aliases)[];
declare const FLOWBASEANNOTATION_TYPES: ("StringLiteral" | "NumericLiteral" | "JSXText" | "JSXIdentifier" | "AnyTypeAnnotation" | "ArgumentPlaceholder" | "ArrayExpression" | "ArrayPattern" | "ArrayTypeAnnotation" | "ArrowFunctionExpression" | "AssignmentExpression" | "AssignmentPattern" | "AwaitExpression" | "BigIntLiteral" | "BinaryExpression" | "BindExpression" | "BlockStatement" | "BooleanLiteral" | "BooleanLiteralTypeAnnotation" | "BooleanTypeAnnotation" | "BreakStatement" | "CallExpression" | "CatchClause" | "ClassAccessorProperty" | "ClassBody" | "ClassDeclaration" | "ClassExpression" | "ClassImplements" | "ClassMethod" | "ClassPrivateMethod" | "ClassPrivateProperty" | "ClassProperty" | "ConditionalExpression" | "ContinueStatement" | "DebuggerStatement" | "DecimalLiteral" | "DeclareClass" | "DeclareExportAllDeclaration" | "DeclareExportDeclaration" | "DeclareFunction" | "DeclareInterface" | "DeclareModule" | "DeclareModuleExports" | "DeclareOpaqueType" | "DeclareTypeAlias" | "DeclareVariable" | "DeclaredPredicate" | "Decorator" | "Directive" | "DirectiveLiteral" | "DoExpression" | "DoWhileStatement" | "EmptyStatement" | "EmptyTypeAnnotation" | "EnumBooleanBody" | "EnumBooleanMember" | "EnumDeclaration" | "EnumDefaultedMember" | "EnumNumberBody" | "EnumNumberMember" | "EnumStringBody" | "EnumStringMember" | "EnumSymbolBody" | "ExistsTypeAnnotation" | "ExportAllDeclaration" | "ExportDefaultDeclaration" | "ExportDefaultSpecifier" | "ExportNamedDeclaration" | "ExportNamespaceSpecifier" | "ExportSpecifier" | "ExpressionStatement" | "File" | "ForInStatement" | "ForOfStatement" | "ForStatement" | "FunctionDeclaration" | "FunctionExpression" | "FunctionTypeAnnotation" | "FunctionTypeParam" | "GenericTypeAnnotation" | "Identifier" | "IfStatement" | "Import" | "ImportAttribute" | "ImportDeclaration" | "ImportDefaultSpecifier" | "ImportExpression" | "ImportNamespaceSpecifier" | "ImportSpecifier" | "IndexedAccessType" | "InferredPredicate" | "InterfaceDeclaration" | "InterfaceExtends" | "InterfaceTypeAnnotation" | "InterpreterDirective" | "IntersectionTypeAnnotation" | "JSXAttribute" | "JSXClosingElement" | "JSXClosingFragment" | "JSXElement" | "JSXEmptyExpression" | "JSXExpressionContainer" | "JSXFragment" | "JSXMemberExpression" | "JSXNamespacedName" | "JSXOpeningElement" | "JSXOpeningFragment" | "JSXSpreadAttribute" | "JSXSpreadChild" | "LabeledStatement" | "LogicalExpression" | "MemberExpression" | "MetaProperty" | "MixedTypeAnnotation" | "ModuleExpression" | "NewExpression" | "Noop" | "NullLiteral" | "NullLiteralTypeAnnotation" | "NullableTypeAnnotation" | "NumberLiteral" | "NumberLiteralTypeAnnotation" | "NumberTypeAnnotation" | "ObjectExpression" | "ObjectMethod" | "ObjectPattern" | "ObjectProperty" | "ObjectTypeAnnotation" | "ObjectTypeCallProperty" | "ObjectTypeIndexer" | "ObjectTypeInternalSlot" | "ObjectTypeProperty" | "ObjectTypeSpreadProperty" | "OpaqueType" | "OptionalCallExpression" | "OptionalIndexedAccessType" | "OptionalMemberExpression" | "ParenthesizedExpression" | "PipelineBareFunction" | "PipelinePrimaryTopicReference" | "PipelineTopicExpression" | "Placeholder" | "PrivateName" | "Program" | "QualifiedTypeIdentifier" | "RecordExpression" | "RegExpLiteral" | "RegexLiteral" | "RestElement" | "RestProperty" | "ReturnStatement" | "SequenceExpression" | "SpreadElement" | "SpreadProperty" | "StaticBlock" | "StringLiteralTypeAnnotation" | "StringTypeAnnotation" | "Super" | "SwitchCase" | "SwitchStatement" | "SymbolTypeAnnotation" | "TSAnyKeyword" | "TSArrayType" | "TSAsExpression" | "TSBigIntKeyword" | "TSBooleanKeyword" | "TSCallSignatureDeclaration" | "TSConditionalType" | "TSConstructSignatureDeclaration" | "TSConstructorType" | "TSDeclareFunction" | "TSDeclareMethod" | "TSEnumDeclaration" | "TSEnumMember" | "TSExportAssignment" | "TSExpressionWithTypeArguments" | "TSExternalModuleReference" | "TSFunctionType" | "TSImportEqualsDeclaration" | "TSImportType" | "TSIndexSignature" | "TSIndexedAccessType" | "TSInferType" | "TSInstantiationExpression" | "TSInterfaceBody" | "TSInterfaceDeclaration" | "TSIntersectionType" | "TSIntrinsicKeyword" | "TSLiteralType" | "TSMappedType" | "TSMethodSignature" | "TSModuleBlock" | "TSModuleDeclaration" | "TSNamedTupleMember" | "TSNamespaceExportDeclaration" | "TSNeverKeyword" | "TSNonNullExpression" | "TSNullKeyword" | "TSNumberKeyword" | "TSObjectKeyword" | "TSOptionalType" | "TSParameterProperty" | "TSParenthesizedType" | "TSPropertySignature" | "TSQualifiedName" | "TSRestType" | "TSSatisfiesExpression" | "TSStringKeyword" | "TSSymbolKeyword" | "TSThisType" | "TSTupleType" | "TSTypeAliasDeclaration" | "TSTypeAnnotation" | "TSTypeAssertion" | "TSTypeLiteral" | "TSTypeOperator" | "TSTypeParameter" | "TSTypeParameterDeclaration" | "TSTypeParameterInstantiation" | "TSTypePredicate" | "TSTypeQuery" | "TSTypeReference" | "TSUndefinedKeyword" | "TSUnionType" | "TSUnknownKeyword" | "TSVoidKeyword" | "TaggedTemplateExpression" | "TemplateElement" | "TemplateLiteral" | "ThisExpression" | "ThisTypeAnnotation" | "ThrowStatement" | "TopicReference" | "TryStatement" | "TupleExpression" | "TupleTypeAnnotation" | "TypeAlias" | "TypeAnnotation" | "TypeCastExpression" | "TypeParameter" | "TypeParameterDeclaration" | "TypeParameterInstantiation" | "TypeofTypeAnnotation" | "UnaryExpression" | "UnionTypeAnnotation" | "UpdateExpression" | "V8IntrinsicIdentifier" | "VariableDeclaration" | "VariableDeclarator" | "Variance" | "VoidTypeAnnotation" | "WhileStatement" | "WithStatement" | "YieldExpression" | keyof Aliases)[];
declare const FLOWDECLARATION_TYPES: ("StringLiteral" | "NumericLiteral" | "JSXText" | "JSXIdentifier" | "AnyTypeAnnotation" | "ArgumentPlaceholder" | "ArrayExpression" | "ArrayPattern" | "ArrayTypeAnnotation" | "ArrowFunctionExpression" | "AssignmentExpression" | "AssignmentPattern" | "AwaitExpression" | "BigIntLiteral" | "BinaryExpression" | "BindExpression" | "BlockStatement" | "BooleanLiteral" | "BooleanLiteralTypeAnnotation" | "BooleanTypeAnnotation" | "BreakStatement" | "CallExpression" | "CatchClause" | "ClassAccessorProperty" | "ClassBody" | "ClassDeclaration" | "ClassExpression" | "ClassImplements" | "ClassMethod" | "ClassPrivateMethod" | "ClassPrivateProperty" | "ClassProperty" | "ConditionalExpression" | "ContinueStatement" | "DebuggerStatement" | "DecimalLiteral" | "DeclareClass" | "DeclareExportAllDeclaration" | "DeclareExportDeclaration" | "DeclareFunction" | "DeclareInterface" | "DeclareModule" | "DeclareModuleExports" | "DeclareOpaqueType" | "DeclareTypeAlias" | "DeclareVariable" | "DeclaredPredicate" | "Decorator" | "Directive" | "DirectiveLiteral" | "DoExpression" | "DoWhileStatement" | "EmptyStatement" | "EmptyTypeAnnotation" | "EnumBooleanBody" | "EnumBooleanMember" | "EnumDeclaration" | "EnumDefaultedMember" | "EnumNumberBody" | "EnumNumberMember" | "EnumStringBody" | "EnumStringMember" | "EnumSymbolBody" | "ExistsTypeAnnotation" | "ExportAllDeclaration" | "ExportDefaultDeclaration" | "ExportDefaultSpecifier" | "ExportNamedDeclaration" | "ExportNamespaceSpecifier" | "ExportSpecifier" | "ExpressionStatement" | "File" | "ForInStatement" | "ForOfStatement" | "ForStatement" | "FunctionDeclaration" | "FunctionExpression" | "FunctionTypeAnnotation" | "FunctionTypeParam" | "GenericTypeAnnotation" | "Identifier" | "IfStatement" | "Import" | "ImportAttribute" | "ImportDeclaration" | "ImportDefaultSpecifier" | "ImportExpression" | "ImportNamespaceSpecifier" | "ImportSpecifier" | "IndexedAccessType" | "InferredPredicate" | "InterfaceDeclaration" | "InterfaceExtends" | "InterfaceTypeAnnotation" | "InterpreterDirective" | "IntersectionTypeAnnotation" | "JSXAttribute" | "JSXClosingElement" | "JSXClosingFragment" | "JSXElement" | "JSXEmptyExpression" | "JSXExpressionContainer" | "JSXFragment" | "JSXMemberExpression" | "JSXNamespacedName" | "JSXOpeningElement" | "JSXOpeningFragment" | "JSXSpreadAttribute" | "JSXSpreadChild" | "LabeledStatement" | "LogicalExpression" | "MemberExpression" | "MetaProperty" | "MixedTypeAnnotation" | "ModuleExpression" | "NewExpression" | "Noop" | "NullLiteral" | "NullLiteralTypeAnnotation" | "NullableTypeAnnotation" | "NumberLiteral" | "NumberLiteralTypeAnnotation" | "NumberTypeAnnotation" | "ObjectExpression" | "ObjectMethod" | "ObjectPattern" | "ObjectProperty" | "ObjectTypeAnnotation" | "ObjectTypeCallProperty" | "ObjectTypeIndexer" | "ObjectTypeInternalSlot" | "ObjectTypeProperty" | "ObjectTypeSpreadProperty" | "OpaqueType" | "OptionalCallExpression" | "OptionalIndexedAccessType" | "OptionalMemberExpression" | "ParenthesizedExpression" | "PipelineBareFunction" | "PipelinePrimaryTopicReference" | "PipelineTopicExpression" | "Placeholder" | "PrivateName" | "Program" | "QualifiedTypeIdentifier" | "RecordExpression" | "RegExpLiteral" | "RegexLiteral" | "RestElement" | "RestProperty" | "ReturnStatement" | "SequenceExpression" | "SpreadElement" | "SpreadProperty" | "StaticBlock" | "StringLiteralTypeAnnotation" | "StringTypeAnnotation" | "Super" | "SwitchCase" | "SwitchStatement" | "SymbolTypeAnnotation" | "TSAnyKeyword" | "TSArrayType" | "TSAsExpression" | "TSBigIntKeyword" | "TSBooleanKeyword" | "TSCallSignatureDeclaration" | "TSConditionalType" | "TSConstructSignatureDeclaration" | "TSConstructorType" | "TSDeclareFunction" | "TSDeclareMethod" | "TSEnumDeclaration" | "TSEnumMember" | "TSExportAssignment" | "TSExpressionWithTypeArguments" | "TSExternalModuleReference" | "TSFunctionType" | "TSImportEqualsDeclaration" | "TSImportType" | "TSIndexSignature" | "TSIndexedAccessType" | "TSInferType" | "TSInstantiationExpression" | "TSInterfaceBody" | "TSInterfaceDeclaration" | "TSIntersectionType" | "TSIntrinsicKeyword" | "TSLiteralType" | "TSMappedType" | "TSMethodSignature" | "TSModuleBlock" | "TSModuleDeclaration" | "TSNamedTupleMember" | "TSNamespaceExportDeclaration" | "TSNeverKeyword" | "TSNonNullExpression" | "TSNullKeyword" | "TSNumberKeyword" | "TSObjectKeyword" | "TSOptionalType" | "TSParameterProperty" | "TSParenthesizedType" | "TSPropertySignature" | "TSQualifiedName" | "TSRestType" | "TSSatisfiesExpression" | "TSStringKeyword" | "TSSymbolKeyword" | "TSThisType" | "TSTupleType" | "TSTypeAliasDeclaration" | "TSTypeAnnotation" | "TSTypeAssertion" | "TSTypeLiteral" | "TSTypeOperator" | "TSTypeParameter" | "TSTypeParameterDeclaration" | "TSTypeParameterInstantiation" | "TSTypePredicate" | "TSTypeQuery" | "TSTypeReference" | "TSUndefinedKeyword" | "TSUnionType" | "TSUnknownKeyword" | "TSVoidKeyword" | "TaggedTemplateExpression" | "TemplateElement" | "TemplateLiteral" | "ThisExpression" | "ThisTypeAnnotation" | "ThrowStatement" | "TopicReference" | "TryStatement" | "TupleExpression" | "TupleTypeAnnotation" | "TypeAlias" | "TypeAnnotation" | "TypeCastExpression" | "TypeParameter" | "TypeParameterDeclaration" | "TypeParameterInstantiation" | "TypeofTypeAnnotation" | "UnaryExpression" | "UnionTypeAnnotation" | "UpdateExpression" | "V8IntrinsicIdentifier" | "VariableDeclaration" | "VariableDeclarator" | "Variance" | "VoidTypeAnnotation" | "WhileStatement" | "WithStatement" | "YieldExpression" | keyof Aliases)[];
declare const FLOWPREDICATE_TYPES: ("StringLiteral" | "NumericLiteral" | "JSXText" | "JSXIdentifier" | "AnyTypeAnnotation" | "ArgumentPlaceholder" | "ArrayExpression" | "ArrayPattern" | "ArrayTypeAnnotation" | "ArrowFunctionExpression" | "AssignmentExpression" | "AssignmentPattern" | "AwaitExpression" | "BigIntLiteral" | "BinaryExpression" | "BindExpression" | "BlockStatement" | "BooleanLiteral" | "BooleanLiteralTypeAnnotation" | "BooleanTypeAnnotation" | "BreakStatement" | "CallExpression" | "CatchClause" | "ClassAccessorProperty" | "ClassBody" | "ClassDeclaration" | "ClassExpression" | "ClassImplements" | "ClassMethod" | "ClassPrivateMethod" | "ClassPrivateProperty" | "ClassProperty" | "ConditionalExpression" | "ContinueStatement" | "DebuggerStatement" | "DecimalLiteral" | "DeclareClass" | "DeclareExportAllDeclaration" | "DeclareExportDeclaration" | "DeclareFunction" | "DeclareInterface" | "DeclareModule" | "DeclareModuleExports" | "DeclareOpaqueType" | "DeclareTypeAlias" | "DeclareVariable" | "DeclaredPredicate" | "Decorator" | "Directive" | "DirectiveLiteral" | "DoExpression" | "DoWhileStatement" | "EmptyStatement" | "EmptyTypeAnnotation" | "EnumBooleanBody" | "EnumBooleanMember" | "EnumDeclaration" | "EnumDefaultedMember" | "EnumNumberBody" | "EnumNumberMember" | "EnumStringBody" | "EnumStringMember" | "EnumSymbolBody" | "ExistsTypeAnnotation" | "ExportAllDeclaration" | "ExportDefaultDeclaration" | "ExportDefaultSpecifier" | "ExportNamedDeclaration" | "ExportNamespaceSpecifier" | "ExportSpecifier" | "ExpressionStatement" | "File" | "ForInStatement" | "ForOfStatement" | "ForStatement" | "FunctionDeclaration" | "FunctionExpression" | "FunctionTypeAnnotation" | "FunctionTypeParam" | "GenericTypeAnnotation" | "Identifier" | "IfStatement" | "Import" | "ImportAttribute" | "ImportDeclaration" | "ImportDefaultSpecifier" | "ImportExpression" | "ImportNamespaceSpecifier" | "ImportSpecifier" | "IndexedAccessType" | "InferredPredicate" | "InterfaceDeclaration" | "InterfaceExtends" | "InterfaceTypeAnnotation" | "InterpreterDirective" | "IntersectionTypeAnnotation" | "JSXAttribute" | "JSXClosingElement" | "JSXClosingFragment" | "JSXElement" | "JSXEmptyExpression" | "JSXExpressionContainer" | "JSXFragment" | "JSXMemberExpression" | "JSXNamespacedName" | "JSXOpeningElement" | "JSXOpeningFragment" | "JSXSpreadAttribute" | "JSXSpreadChild" | "LabeledStatement" | "LogicalExpression" | "MemberExpression" | "MetaProperty" | "MixedTypeAnnotation" | "ModuleExpression" | "NewExpression" | "Noop" | "NullLiteral" | "NullLiteralTypeAnnotation" | "NullableTypeAnnotation" | "NumberLiteral" | "NumberLiteralTypeAnnotation" | "NumberTypeAnnotation" | "ObjectExpression" | "ObjectMethod" | "ObjectPattern" | "ObjectProperty" | "ObjectTypeAnnotation" | "ObjectTypeCallProperty" | "ObjectTypeIndexer" | "ObjectTypeInternalSlot" | "ObjectTypeProperty" | "ObjectTypeSpreadProperty" | "OpaqueType" | "OptionalCallExpression" | "OptionalIndexedAccessType" | "OptionalMemberExpression" | "ParenthesizedExpression" | "PipelineBareFunction" | "PipelinePrimaryTopicReference" | "PipelineTopicExpression" | "Placeholder" | "PrivateName" | "Program" | "QualifiedTypeIdentifier" | "RecordExpression" | "RegExpLiteral" | "RegexLiteral" | "RestElement" | "RestProperty" | "ReturnStatement" | "SequenceExpression" | "SpreadElement" | "SpreadProperty" | "StaticBlock" | "StringLiteralTypeAnnotation" | "StringTypeAnnotation" | "Super" | "SwitchCase" | "SwitchStatement" | "SymbolTypeAnnotation" | "TSAnyKeyword" | "TSArrayType" | "TSAsExpression" | "TSBigIntKeyword" | "TSBooleanKeyword" | "TSCallSignatureDeclaration" | "TSConditionalType" | "TSConstructSignatureDeclaration" | "TSConstructorType" | "TSDeclareFunction" | "TSDeclareMethod" | "TSEnumDeclaration" | "TSEnumMember" | "TSExportAssignment" | "TSExpressionWithTypeArguments" | "TSExternalModuleReference" | "TSFunctionType" | "TSImportEqualsDeclaration" | "TSImportType" | "TSIndexSignature" | "TSIndexedAccessType" | "TSInferType" | "TSInstantiationExpression" | "TSInterfaceBody" | "TSInterfaceDeclaration" | "TSIntersectionType" | "TSIntrinsicKeyword" | "TSLiteralType" | "TSMappedType" | "TSMethodSignature" | "TSModuleBlock" | "TSModuleDeclaration" | "TSNamedTupleMember" | "TSNamespaceExportDeclaration" | "TSNeverKeyword" | "TSNonNullExpression" | "TSNullKeyword" | "TSNumberKeyword" | "TSObjectKeyword" | "TSOptionalType" | "TSParameterProperty" | "TSParenthesizedType" | "TSPropertySignature" | "TSQualifiedName" | "TSRestType" | "TSSatisfiesExpression" | "TSStringKeyword" | "TSSymbolKeyword" | "TSThisType" | "TSTupleType" | "TSTypeAliasDeclaration" | "TSTypeAnnotation" | "TSTypeAssertion" | "TSTypeLiteral" | "TSTypeOperator" | "TSTypeParameter" | "TSTypeParameterDeclaration" | "TSTypeParameterInstantiation" | "TSTypePredicate" | "TSTypeQuery" | "TSTypeReference" | "TSUndefinedKeyword" | "TSUnionType" | "TSUnknownKeyword" | "TSVoidKeyword" | "TaggedTemplateExpression" | "TemplateElement" | "TemplateLiteral" | "ThisExpression" | "ThisTypeAnnotation" | "ThrowStatement" | "TopicReference" | "TryStatement" | "TupleExpression" | "TupleTypeAnnotation" | "TypeAlias" | "TypeAnnotation" | "TypeCastExpression" | "TypeParameter" | "TypeParameterDeclaration" | "TypeParameterInstantiation" | "TypeofTypeAnnotation" | "UnaryExpression" | "UnionTypeAnnotation" | "UpdateExpression" | "V8IntrinsicIdentifier" | "VariableDeclaration" | "VariableDeclarator" | "Variance" | "VoidTypeAnnotation" | "WhileStatement" | "WithStatement" | "YieldExpression" | keyof Aliases)[];
declare const ENUMBODY_TYPES: ("StringLiteral" | "NumericLiteral" | "JSXText" | "JSXIdentifier" | "AnyTypeAnnotation" | "ArgumentPlaceholder" | "ArrayExpression" | "ArrayPattern" | "ArrayTypeAnnotation" | "ArrowFunctionExpression" | "AssignmentExpression" | "AssignmentPattern" | "AwaitExpression" | "BigIntLiteral" | "BinaryExpression" | "BindExpression" | "BlockStatement" | "BooleanLiteral" | "BooleanLiteralTypeAnnotation" | "BooleanTypeAnnotation" | "BreakStatement" | "CallExpression" | "CatchClause" | "ClassAccessorProperty" | "ClassBody" | "ClassDeclaration" | "ClassExpression" | "ClassImplements" | "ClassMethod" | "ClassPrivateMethod" | "ClassPrivateProperty" | "ClassProperty" | "ConditionalExpression" | "ContinueStatement" | "DebuggerStatement" | "DecimalLiteral" | "DeclareClass" | "DeclareExportAllDeclaration" | "DeclareExportDeclaration" | "DeclareFunction" | "DeclareInterface" | "DeclareModule" | "DeclareModuleExports" | "DeclareOpaqueType" | "DeclareTypeAlias" | "DeclareVariable" | "DeclaredPredicate" | "Decorator" | "Directive" | "DirectiveLiteral" | "DoExpression" | "DoWhileStatement" | "EmptyStatement" | "EmptyTypeAnnotation" | "EnumBooleanBody" | "EnumBooleanMember" | "EnumDeclaration" | "EnumDefaultedMember" | "EnumNumberBody" | "EnumNumberMember" | "EnumStringBody" | "EnumStringMember" | "EnumSymbolBody" | "ExistsTypeAnnotation" | "ExportAllDeclaration" | "ExportDefaultDeclaration" | "ExportDefaultSpecifier" | "ExportNamedDeclaration" | "ExportNamespaceSpecifier" | "ExportSpecifier" | "ExpressionStatement" | "File" | "ForInStatement" | "ForOfStatement" | "ForStatement" | "FunctionDeclaration" | "FunctionExpression" | "FunctionTypeAnnotation" | "FunctionTypeParam" | "GenericTypeAnnotation" | "Identifier" | "IfStatement" | "Import" | "ImportAttribute" | "ImportDeclaration" | "ImportDefaultSpecifier" | "ImportExpression" | "ImportNamespaceSpecifier" | "ImportSpecifier" | "IndexedAccessType" | "InferredPredicate" | "InterfaceDeclaration" | "InterfaceExtends" | "InterfaceTypeAnnotation" | "InterpreterDirective" | "IntersectionTypeAnnotation" | "JSXAttribute" | "JSXClosingElement" | "JSXClosingFragment" | "JSXElement" | "JSXEmptyExpression" | "JSXExpressionContainer" | "JSXFragment" | "JSXMemberExpression" | "JSXNamespacedName" | "JSXOpeningElement" | "JSXOpeningFragment" | "JSXSpreadAttribute" | "JSXSpreadChild" | "LabeledStatement" | "LogicalExpression" | "MemberExpression" | "MetaProperty" | "MixedTypeAnnotation" | "ModuleExpression" | "NewExpression" | "Noop" | "NullLiteral" | "NullLiteralTypeAnnotation" | "NullableTypeAnnotation" | "NumberLiteral" | "NumberLiteralTypeAnnotation" | "NumberTypeAnnotation" | "ObjectExpression" | "ObjectMethod" | "ObjectPattern" | "ObjectProperty" | "ObjectTypeAnnotation" | "ObjectTypeCallProperty" | "ObjectTypeIndexer" | "ObjectTypeInternalSlot" | "ObjectTypeProperty" | "ObjectTypeSpreadProperty" | "OpaqueType" | "OptionalCallExpression" | "OptionalIndexedAccessType" | "OptionalMemberExpression" | "ParenthesizedExpression" | "PipelineBareFunction" | "PipelinePrimaryTopicReference" | "PipelineTopicExpression" | "Placeholder" | "PrivateName" | "Program" | "QualifiedTypeIdentifier" | "RecordExpression" | "RegExpLiteral" | "RegexLiteral" | "RestElement" | "RestProperty" | "ReturnStatement" | "SequenceExpression" | "SpreadElement" | "SpreadProperty" | "StaticBlock" | "StringLiteralTypeAnnotation" | "StringTypeAnnotation" | "Super" | "SwitchCase" | "SwitchStatement" | "SymbolTypeAnnotation" | "TSAnyKeyword" | "TSArrayType" | "TSAsExpression" | "TSBigIntKeyword" | "TSBooleanKeyword" | "TSCallSignatureDeclaration" | "TSConditionalType" | "TSConstructSignatureDeclaration" | "TSConstructorType" | "TSDeclareFunction" | "TSDeclareMethod" | "TSEnumDeclaration" | "TSEnumMember" | "TSExportAssignment" | "TSExpressionWithTypeArguments" | "TSExternalModuleReference" | "TSFunctionType" | "TSImportEqualsDeclaration" | "TSImportType" | "TSIndexSignature" | "TSIndexedAccessType" | "TSInferType" | "TSInstantiationExpression" | "TSInterfaceBody" | "TSInterfaceDeclaration" | "TSIntersectionType" | "TSIntrinsicKeyword" | "TSLiteralType" | "TSMappedType" | "TSMethodSignature" | "TSModuleBlock" | "TSModuleDeclaration" | "TSNamedTupleMember" | "TSNamespaceExportDeclaration" | "TSNeverKeyword" | "TSNonNullExpression" | "TSNullKeyword" | "TSNumberKeyword" | "TSObjectKeyword" | "TSOptionalType" | "TSParameterProperty" | "TSParenthesizedType" | "TSPropertySignature" | "TSQualifiedName" | "TSRestType" | "TSSatisfiesExpression" | "TSStringKeyword" | "TSSymbolKeyword" | "TSThisType" | "TSTupleType" | "TSTypeAliasDeclaration" | "TSTypeAnnotation" | "TSTypeAssertion" | "TSTypeLiteral" | "TSTypeOperator" | "TSTypeParameter" | "TSTypeParameterDeclaration" | "TSTypeParameterInstantiation" | "TSTypePredicate" | "TSTypeQuery" | "TSTypeReference" | "TSUndefinedKeyword" | "TSUnionType" | "TSUnknownKeyword" | "TSVoidKeyword" | "TaggedTemplateExpression" | "TemplateElement" | "TemplateLiteral" | "ThisExpression" | "ThisTypeAnnotation" | "ThrowStatement" | "TopicReference" | "TryStatement" | "TupleExpression" | "TupleTypeAnnotation" | "TypeAlias" | "TypeAnnotation" | "TypeCastExpression" | "TypeParameter" | "TypeParameterDeclaration" | "TypeParameterInstantiation" | "TypeofTypeAnnotation" | "UnaryExpression" | "UnionTypeAnnotation" | "UpdateExpression" | "V8IntrinsicIdentifier" | "VariableDeclaration" | "VariableDeclarator" | "Variance" | "VoidTypeAnnotation" | "WhileStatement" | "WithStatement" | "YieldExpression" | keyof Aliases)[];
declare const ENUMMEMBER_TYPES: ("StringLiteral" | "NumericLiteral" | "JSXText" | "JSXIdentifier" | "AnyTypeAnnotation" | "ArgumentPlaceholder" | "ArrayExpression" | "ArrayPattern" | "ArrayTypeAnnotation" | "ArrowFunctionExpression" | "AssignmentExpression" | "AssignmentPattern" | "AwaitExpression" | "BigIntLiteral" | "BinaryExpression" | "BindExpression" | "BlockStatement" | "BooleanLiteral" | "BooleanLiteralTypeAnnotation" | "BooleanTypeAnnotation" | "BreakStatement" | "CallExpression" | "CatchClause" | "ClassAccessorProperty" | "ClassBody" | "ClassDeclaration" | "ClassExpression" | "ClassImplements" | "ClassMethod" | "ClassPrivateMethod" | "ClassPrivateProperty" | "ClassProperty" | "ConditionalExpression" | "ContinueStatement" | "DebuggerStatement" | "DecimalLiteral" | "DeclareClass" | "DeclareExportAllDeclaration" | "DeclareExportDeclaration" | "DeclareFunction" | "DeclareInterface" | "DeclareModule" | "DeclareModuleExports" | "DeclareOpaqueType" | "DeclareTypeAlias" | "DeclareVariable" | "DeclaredPredicate" | "Decorator" | "Directive" | "DirectiveLiteral" | "DoExpression" | "DoWhileStatement" | "EmptyStatement" | "EmptyTypeAnnotation" | "EnumBooleanBody" | "EnumBooleanMember" | "EnumDeclaration" | "EnumDefaultedMember" | "EnumNumberBody" | "EnumNumberMember" | "EnumStringBody" | "EnumStringMember" | "EnumSymbolBody" | "ExistsTypeAnnotation" | "ExportAllDeclaration" | "ExportDefaultDeclaration" | "ExportDefaultSpecifier" | "ExportNamedDeclaration" | "ExportNamespaceSpecifier" | "ExportSpecifier" | "ExpressionStatement" | "File" | "ForInStatement" | "ForOfStatement" | "ForStatement" | "FunctionDeclaration" | "FunctionExpression" | "FunctionTypeAnnotation" | "FunctionTypeParam" | "GenericTypeAnnotation" | "Identifier" | "IfStatement" | "Import" | "ImportAttribute" | "ImportDeclaration" | "ImportDefaultSpecifier" | "ImportExpression" | "ImportNamespaceSpecifier" | "ImportSpecifier" | "IndexedAccessType" | "InferredPredicate" | "InterfaceDeclaration" | "InterfaceExtends" | "InterfaceTypeAnnotation" | "InterpreterDirective" | "IntersectionTypeAnnotation" | "JSXAttribute" | "JSXClosingElement" | "JSXClosingFragment" | "JSXElement" | "JSXEmptyExpression" | "JSXExpressionContainer" | "JSXFragment" | "JSXMemberExpression" | "JSXNamespacedName" | "JSXOpeningElement" | "JSXOpeningFragment" | "JSXSpreadAttribute" | "JSXSpreadChild" | "LabeledStatement" | "LogicalExpression" | "MemberExpression" | "MetaProperty" | "MixedTypeAnnotation" | "ModuleExpression" | "NewExpression" | "Noop" | "NullLiteral" | "NullLiteralTypeAnnotation" | "NullableTypeAnnotation" | "NumberLiteral" | "NumberLiteralTypeAnnotation" | "NumberTypeAnnotation" | "ObjectExpression" | "ObjectMethod" | "ObjectPattern" | "ObjectProperty" | "ObjectTypeAnnotation" | "ObjectTypeCallProperty" | "ObjectTypeIndexer" | "ObjectTypeInternalSlot" | "ObjectTypeProperty" | "ObjectTypeSpreadProperty" | "OpaqueType" | "OptionalCallExpression" | "OptionalIndexedAccessType" | "OptionalMemberExpression" | "ParenthesizedExpression" | "PipelineBareFunction" | "PipelinePrimaryTopicReference" | "PipelineTopicExpression" | "Placeholder" | "PrivateName" | "Program" | "QualifiedTypeIdentifier" | "RecordExpression" | "RegExpLiteral" | "RegexLiteral" | "RestElement" | "RestProperty" | "ReturnStatement" | "SequenceExpression" | "SpreadElement" | "SpreadProperty" | "StaticBlock" | "StringLiteralTypeAnnotation" | "StringTypeAnnotation" | "Super" | "SwitchCase" | "SwitchStatement" | "SymbolTypeAnnotation" | "TSAnyKeyword" | "TSArrayType" | "TSAsExpression" | "TSBigIntKeyword" | "TSBooleanKeyword" | "TSCallSignatureDeclaration" | "TSConditionalType" | "TSConstructSignatureDeclaration" | "TSConstructorType" | "TSDeclareFunction" | "TSDeclareMethod" | "TSEnumDeclaration" | "TSEnumMember" | "TSExportAssignment" | "TSExpressionWithTypeArguments" | "TSExternalModuleReference" | "TSFunctionType" | "TSImportEqualsDeclaration" | "TSImportType" | "TSIndexSignature" | "TSIndexedAccessType" | "TSInferType" | "TSInstantiationExpression" | "TSInterfaceBody" | "TSInterfaceDeclaration" | "TSIntersectionType" | "TSIntrinsicKeyword" | "TSLiteralType" | "TSMappedType" | "TSMethodSignature" | "TSModuleBlock" | "TSModuleDeclaration" | "TSNamedTupleMember" | "TSNamespaceExportDeclaration" | "TSNeverKeyword" | "TSNonNullExpression" | "TSNullKeyword" | "TSNumberKeyword" | "TSObjectKeyword" | "TSOptionalType" | "TSParameterProperty" | "TSParenthesizedType" | "TSPropertySignature" | "TSQualifiedName" | "TSRestType" | "TSSatisfiesExpression" | "TSStringKeyword" | "TSSymbolKeyword" | "TSThisType" | "TSTupleType" | "TSTypeAliasDeclaration" | "TSTypeAnnotation" | "TSTypeAssertion" | "TSTypeLiteral" | "TSTypeOperator" | "TSTypeParameter" | "TSTypeParameterDeclaration" | "TSTypeParameterInstantiation" | "TSTypePredicate" | "TSTypeQuery" | "TSTypeReference" | "TSUndefinedKeyword" | "TSUnionType" | "TSUnknownKeyword" | "TSVoidKeyword" | "TaggedTemplateExpression" | "TemplateElement" | "TemplateLiteral" | "ThisExpression" | "ThisTypeAnnotation" | "ThrowStatement" | "TopicReference" | "TryStatement" | "TupleExpression" | "TupleTypeAnnotation" | "TypeAlias" | "TypeAnnotation" | "TypeCastExpression" | "TypeParameter" | "TypeParameterDeclaration" | "TypeParameterInstantiation" | "TypeofTypeAnnotation" | "UnaryExpression" | "UnionTypeAnnotation" | "UpdateExpression" | "V8IntrinsicIdentifier" | "VariableDeclaration" | "VariableDeclarator" | "Variance" | "VoidTypeAnnotation" | "WhileStatement" | "WithStatement" | "YieldExpression" | keyof Aliases)[];
declare const JSX_TYPES: ("StringLiteral" | "NumericLiteral" | "JSXText" | "JSXIdentifier" | "AnyTypeAnnotation" | "ArgumentPlaceholder" | "ArrayExpression" | "ArrayPattern" | "ArrayTypeAnnotation" | "ArrowFunctionExpression" | "AssignmentExpression" | "AssignmentPattern" | "AwaitExpression" | "BigIntLiteral" | "BinaryExpression" | "BindExpression" | "BlockStatement" | "BooleanLiteral" | "BooleanLiteralTypeAnnotation" | "BooleanTypeAnnotation" | "BreakStatement" | "CallExpression" | "CatchClause" | "ClassAccessorProperty" | "ClassBody" | "ClassDeclaration" | "ClassExpression" | "ClassImplements" | "ClassMethod" | "ClassPrivateMethod" | "ClassPrivateProperty" | "ClassProperty" | "ConditionalExpression" | "ContinueStatement" | "DebuggerStatement" | "DecimalLiteral" | "DeclareClass" | "DeclareExportAllDeclaration" | "DeclareExportDeclaration" | "DeclareFunction" | "DeclareInterface" | "DeclareModule" | "DeclareModuleExports" | "DeclareOpaqueType" | "DeclareTypeAlias" | "DeclareVariable" | "DeclaredPredicate" | "Decorator" | "Directive" | "DirectiveLiteral" | "DoExpression" | "DoWhileStatement" | "EmptyStatement" | "EmptyTypeAnnotation" | "EnumBooleanBody" | "EnumBooleanMember" | "EnumDeclaration" | "EnumDefaultedMember" | "EnumNumberBody" | "EnumNumberMember" | "EnumStringBody" | "EnumStringMember" | "EnumSymbolBody" | "ExistsTypeAnnotation" | "ExportAllDeclaration" | "ExportDefaultDeclaration" | "ExportDefaultSpecifier" | "ExportNamedDeclaration" | "ExportNamespaceSpecifier" | "ExportSpecifier" | "ExpressionStatement" | "File" | "ForInStatement" | "ForOfStatement" | "ForStatement" | "FunctionDeclaration" | "FunctionExpression" | "FunctionTypeAnnotation" | "FunctionTypeParam" | "GenericTypeAnnotation" | "Identifier" | "IfStatement" | "Import" | "ImportAttribute" | "ImportDeclaration" | "ImportDefaultSpecifier" | "ImportExpression" | "ImportNamespaceSpecifier" | "ImportSpecifier" | "IndexedAccessType" | "InferredPredicate" | "InterfaceDeclaration" | "InterfaceExtends" | "InterfaceTypeAnnotation" | "InterpreterDirective" | "IntersectionTypeAnnotation" | "JSXAttribute" | "JSXClosingElement" | "JSXClosingFragment" | "JSXElement" | "JSXEmptyExpression" | "JSXExpressionContainer" | "JSXFragment" | "JSXMemberExpression" | "JSXNamespacedName" | "JSXOpeningElement" | "JSXOpeningFragment" | "JSXSpreadAttribute" | "JSXSpreadChild" | "LabeledStatement" | "LogicalExpression" | "MemberExpression" | "MetaProperty" | "MixedTypeAnnotation" | "ModuleExpression" | "NewExpression" | "Noop" | "NullLiteral" | "NullLiteralTypeAnnotation" | "NullableTypeAnnotation" | "NumberLiteral" | "NumberLiteralTypeAnnotation" | "NumberTypeAnnotation" | "ObjectExpression" | "ObjectMethod" | "ObjectPattern" | "ObjectProperty" | "ObjectTypeAnnotation" | "ObjectTypeCallProperty" | "ObjectTypeIndexer" | "ObjectTypeInternalSlot" | "ObjectTypeProperty" | "ObjectTypeSpreadProperty" | "OpaqueType" | "OptionalCallExpression" | "OptionalIndexedAccessType" | "OptionalMemberExpression" | "ParenthesizedExpression" | "PipelineBareFunction" | "PipelinePrimaryTopicReference" | "PipelineTopicExpression" | "Placeholder" | "PrivateName" | "Program" | "QualifiedTypeIdentifier" | "RecordExpression" | "RegExpLiteral" | "RegexLiteral" | "RestElement" | "RestProperty" | "ReturnStatement" | "SequenceExpression" | "SpreadElement" | "SpreadProperty" | "StaticBlock" | "StringLiteralTypeAnnotation" | "StringTypeAnnotation" | "Super" | "SwitchCase" | "SwitchStatement" | "SymbolTypeAnnotation" | "TSAnyKeyword" | "TSArrayType" | "TSAsExpression" | "TSBigIntKeyword" | "TSBooleanKeyword" | "TSCallSignatureDeclaration" | "TSConditionalType" | "TSConstructSignatureDeclaration" | "TSConstructorType" | "TSDeclareFunction" | "TSDeclareMethod" | "TSEnumDeclaration" | "TSEnumMember" | "TSExportAssignment" | "TSExpressionWithTypeArguments" | "TSExternalModuleReference" | "TSFunctionType" | "TSImportEqualsDeclaration" | "TSImportType" | "TSIndexSignature" | "TSIndexedAccessType" | "TSInferType" | "TSInstantiationExpression" | "TSInterfaceBody" | "TSInterfaceDeclaration" | "TSIntersectionType" | "TSIntrinsicKeyword" | "TSLiteralType" | "TSMappedType" | "TSMethodSignature" | "TSModuleBlock" | "TSModuleDeclaration" | "TSNamedTupleMember" | "TSNamespaceExportDeclaration" | "TSNeverKeyword" | "TSNonNullExpression" | "TSNullKeyword" | "TSNumberKeyword" | "TSObjectKeyword" | "TSOptionalType" | "TSParameterProperty" | "TSParenthesizedType" | "TSPropertySignature" | "TSQualifiedName" | "TSRestType" | "TSSatisfiesExpression" | "TSStringKeyword" | "TSSymbolKeyword" | "TSThisType" | "TSTupleType" | "TSTypeAliasDeclaration" | "TSTypeAnnotation" | "TSTypeAssertion" | "TSTypeLiteral" | "TSTypeOperator" | "TSTypeParameter" | "TSTypeParameterDeclaration" | "TSTypeParameterInstantiation" | "TSTypePredicate" | "TSTypeQuery" | "TSTypeReference" | "TSUndefinedKeyword" | "TSUnionType" | "TSUnknownKeyword" | "TSVoidKeyword" | "TaggedTemplateExpression" | "TemplateElement" | "TemplateLiteral" | "ThisExpression" | "ThisTypeAnnotation" | "ThrowStatement" | "TopicReference" | "TryStatement" | "TupleExpression" | "TupleTypeAnnotation" | "TypeAlias" | "TypeAnnotation" | "TypeCastExpression" | "TypeParameter" | "TypeParameterDeclaration" | "TypeParameterInstantiation" | "TypeofTypeAnnotation" | "UnaryExpression" | "UnionTypeAnnotation" | "UpdateExpression" | "V8IntrinsicIdentifier" | "VariableDeclaration" | "VariableDeclarator" | "Variance" | "VoidTypeAnnotation" | "WhileStatement" | "WithStatement" | "YieldExpression" | keyof Aliases)[];
declare const MISCELLANEOUS_TYPES: ("StringLiteral" | "NumericLiteral" | "JSXText" | "JSXIdentifier" | "AnyTypeAnnotation" | "ArgumentPlaceholder" | "ArrayExpression" | "ArrayPattern" | "ArrayTypeAnnotation" | "ArrowFunctionExpression" | "AssignmentExpression" | "AssignmentPattern" | "AwaitExpression" | "BigIntLiteral" | "BinaryExpression" | "BindExpression" | "BlockStatement" | "BooleanLiteral" | "BooleanLiteralTypeAnnotation" | "BooleanTypeAnnotation" | "BreakStatement" | "CallExpression" | "CatchClause" | "ClassAccessorProperty" | "ClassBody" | "ClassDeclaration" | "ClassExpression" | "ClassImplements" | "ClassMethod" | "ClassPrivateMethod" | "ClassPrivateProperty" | "ClassProperty" | "ConditionalExpression" | "ContinueStatement" | "DebuggerStatement" | "DecimalLiteral" | "DeclareClass" | "DeclareExportAllDeclaration" | "DeclareExportDeclaration" | "DeclareFunction" | "DeclareInterface" | "DeclareModule" | "DeclareModuleExports" | "DeclareOpaqueType" | "DeclareTypeAlias" | "DeclareVariable" | "DeclaredPredicate" | "Decorator" | "Directive" | "DirectiveLiteral" | "DoExpression" | "DoWhileStatement" | "EmptyStatement" | "EmptyTypeAnnotation" | "EnumBooleanBody" | "EnumBooleanMember" | "EnumDeclaration" | "EnumDefaultedMember" | "EnumNumberBody" | "EnumNumberMember" | "EnumStringBody" | "EnumStringMember" | "EnumSymbolBody" | "ExistsTypeAnnotation" | "ExportAllDeclaration" | "ExportDefaultDeclaration" | "ExportDefaultSpecifier" | "ExportNamedDeclaration" | "ExportNamespaceSpecifier" | "ExportSpecifier" | "ExpressionStatement" | "File" | "ForInStatement" | "ForOfStatement" | "ForStatement" | "FunctionDeclaration" | "FunctionExpression" | "FunctionTypeAnnotation" | "FunctionTypeParam" | "GenericTypeAnnotation" | "Identifier" | "IfStatement" | "Import" | "ImportAttribute" | "ImportDeclaration" | "ImportDefaultSpecifier" | "ImportExpression" | "ImportNamespaceSpecifier" | "ImportSpecifier" | "IndexedAccessType" | "InferredPredicate" | "InterfaceDeclaration" | "InterfaceExtends" | "InterfaceTypeAnnotation" | "InterpreterDirective" | "IntersectionTypeAnnotation" | "JSXAttribute" | "JSXClosingElement" | "JSXClosingFragment" | "JSXElement" | "JSXEmptyExpression" | "JSXExpressionContainer" | "JSXFragment" | "JSXMemberExpression" | "JSXNamespacedName" | "JSXOpeningElement" | "JSXOpeningFragment" | "JSXSpreadAttribute" | "JSXSpreadChild" | "LabeledStatement" | "LogicalExpression" | "MemberExpression" | "MetaProperty" | "MixedTypeAnnotation" | "ModuleExpression" | "NewExpression" | "Noop" | "NullLiteral" | "NullLiteralTypeAnnotation" | "NullableTypeAnnotation" | "NumberLiteral" | "NumberLiteralTypeAnnotation" | "NumberTypeAnnotation" | "ObjectExpression" | "ObjectMethod" | "ObjectPattern" | "ObjectProperty" | "ObjectTypeAnnotation" | "ObjectTypeCallProperty" | "ObjectTypeIndexer" | "ObjectTypeInternalSlot" | "ObjectTypeProperty" | "ObjectTypeSpreadProperty" | "OpaqueType" | "OptionalCallExpression" | "OptionalIndexedAccessType" | "OptionalMemberExpression" | "ParenthesizedExpression" | "PipelineBareFunction" | "PipelinePrimaryTopicReference" | "PipelineTopicExpression" | "Placeholder" | "PrivateName" | "Program" | "QualifiedTypeIdentifier" | "RecordExpression" | "RegExpLiteral" | "RegexLiteral" | "RestElement" | "RestProperty" | "ReturnStatement" | "SequenceExpression" | "SpreadElement" | "SpreadProperty" | "StaticBlock" | "StringLiteralTypeAnnotation" | "StringTypeAnnotation" | "Super" | "SwitchCase" | "SwitchStatement" | "SymbolTypeAnnotation" | "TSAnyKeyword" | "TSArrayType" | "TSAsExpression" | "TSBigIntKeyword" | "TSBooleanKeyword" | "TSCallSignatureDeclaration" | "TSConditionalType" | "TSConstructSignatureDeclaration" | "TSConstructorType" | "TSDeclareFunction" | "TSDeclareMethod" | "TSEnumDeclaration" | "TSEnumMember" | "TSExportAssignment" | "TSExpressionWithTypeArguments" | "TSExternalModuleReference" | "TSFunctionType" | "TSImportEqualsDeclaration" | "TSImportType" | "TSIndexSignature" | "TSIndexedAccessType" | "TSInferType" | "TSInstantiationExpression" | "TSInterfaceBody" | "TSInterfaceDeclaration" | "TSIntersectionType" | "TSIntrinsicKeyword" | "TSLiteralType" | "TSMappedType" | "TSMethodSignature" | "TSModuleBlock" | "TSModuleDeclaration" | "TSNamedTupleMember" | "TSNamespaceExportDeclaration" | "TSNeverKeyword" | "TSNonNullExpression" | "TSNullKeyword" | "TSNumberKeyword" | "TSObjectKeyword" | "TSOptionalType" | "TSParameterProperty" | "TSParenthesizedType" | "TSPropertySignature" | "TSQualifiedName" | "TSRestType" | "TSSatisfiesExpression" | "TSStringKeyword" | "TSSymbolKeyword" | "TSThisType" | "TSTupleType" | "TSTypeAliasDeclaration" | "TSTypeAnnotation" | "TSTypeAssertion" | "TSTypeLiteral" | "TSTypeOperator" | "TSTypeParameter" | "TSTypeParameterDeclaration" | "TSTypeParameterInstantiation" | "TSTypePredicate" | "TSTypeQuery" | "TSTypeReference" | "TSUndefinedKeyword" | "TSUnionType" | "TSUnknownKeyword" | "TSVoidKeyword" | "TaggedTemplateExpression" | "TemplateElement" | "TemplateLiteral" | "ThisExpression" | "ThisTypeAnnotation" | "ThrowStatement" | "TopicReference" | "TryStatement" | "TupleExpression" | "TupleTypeAnnotation" | "TypeAlias" | "TypeAnnotation" | "TypeCastExpression" | "TypeParameter" | "TypeParameterDeclaration" | "TypeParameterInstantiation" | "TypeofTypeAnnotation" | "UnaryExpression" | "UnionTypeAnnotation" | "UpdateExpression" | "V8IntrinsicIdentifier" | "VariableDeclaration" | "VariableDeclarator" | "Variance" | "VoidTypeAnnotation" | "WhileStatement" | "WithStatement" | "YieldExpression" | keyof Aliases)[];
declare const TYPESCRIPT_TYPES: ("StringLiteral" | "NumericLiteral" | "JSXText" | "JSXIdentifier" | "AnyTypeAnnotation" | "ArgumentPlaceholder" | "ArrayExpression" | "ArrayPattern" | "ArrayTypeAnnotation" | "ArrowFunctionExpression" | "AssignmentExpression" | "AssignmentPattern" | "AwaitExpression" | "BigIntLiteral" | "BinaryExpression" | "BindExpression" | "BlockStatement" | "BooleanLiteral" | "BooleanLiteralTypeAnnotation" | "BooleanTypeAnnotation" | "BreakStatement" | "CallExpression" | "CatchClause" | "ClassAccessorProperty" | "ClassBody" | "ClassDeclaration" | "ClassExpression" | "ClassImplements" | "ClassMethod" | "ClassPrivateMethod" | "ClassPrivateProperty" | "ClassProperty" | "ConditionalExpression" | "ContinueStatement" | "DebuggerStatement" | "DecimalLiteral" | "DeclareClass" | "DeclareExportAllDeclaration" | "DeclareExportDeclaration" | "DeclareFunction" | "DeclareInterface" | "DeclareModule" | "DeclareModuleExports" | "DeclareOpaqueType" | "DeclareTypeAlias" | "DeclareVariable" | "DeclaredPredicate" | "Decorator" | "Directive" | "DirectiveLiteral" | "DoExpression" | "DoWhileStatement" | "EmptyStatement" | "EmptyTypeAnnotation" | "EnumBooleanBody" | "EnumBooleanMember" | "EnumDeclaration" | "EnumDefaultedMember" | "EnumNumberBody" | "EnumNumberMember" | "EnumStringBody" | "EnumStringMember" | "EnumSymbolBody" | "ExistsTypeAnnotation" | "ExportAllDeclaration" | "ExportDefaultDeclaration" | "ExportDefaultSpecifier" | "ExportNamedDeclaration" | "ExportNamespaceSpecifier" | "ExportSpecifier" | "ExpressionStatement" | "File" | "ForInStatement" | "ForOfStatement" | "ForStatement" | "FunctionDeclaration" | "FunctionExpression" | "FunctionTypeAnnotation" | "FunctionTypeParam" | "GenericTypeAnnotation" | "Identifier" | "IfStatement" | "Import" | "ImportAttribute" | "ImportDeclaration" | "ImportDefaultSpecifier" | "ImportExpression" | "ImportNamespaceSpecifier" | "ImportSpecifier" | "IndexedAccessType" | "InferredPredicate" | "InterfaceDeclaration" | "InterfaceExtends" | "InterfaceTypeAnnotation" | "InterpreterDirective" | "IntersectionTypeAnnotation" | "JSXAttribute" | "JSXClosingElement" | "JSXClosingFragment" | "JSXElement" | "JSXEmptyExpression" | "JSXExpressionContainer" | "JSXFragment" | "JSXMemberExpression" | "JSXNamespacedName" | "JSXOpeningElement" | "JSXOpeningFragment" | "JSXSpreadAttribute" | "JSXSpreadChild" | "LabeledStatement" | "LogicalExpression" | "MemberExpression" | "MetaProperty" | "MixedTypeAnnotation" | "ModuleExpression" | "NewExpression" | "Noop" | "NullLiteral" | "NullLiteralTypeAnnotation" | "NullableTypeAnnotation" | "NumberLiteral" | "NumberLiteralTypeAnnotation" | "NumberTypeAnnotation" | "ObjectExpression" | "ObjectMethod" | "ObjectPattern" | "ObjectProperty" | "ObjectTypeAnnotation" | "ObjectTypeCallProperty" | "ObjectTypeIndexer" | "ObjectTypeInternalSlot" | "ObjectTypeProperty" | "ObjectTypeSpreadProperty" | "OpaqueType" | "OptionalCallExpression" | "OptionalIndexedAccessType" | "OptionalMemberExpression" | "ParenthesizedExpression" | "PipelineBareFunction" | "PipelinePrimaryTopicReference" | "PipelineTopicExpression" | "Placeholder" | "PrivateName" | "Program" | "QualifiedTypeIdentifier" | "RecordExpression" | "RegExpLiteral" | "RegexLiteral" | "RestElement" | "RestProperty" | "ReturnStatement" | "SequenceExpression" | "SpreadElement" | "SpreadProperty" | "StaticBlock" | "StringLiteralTypeAnnotation" | "StringTypeAnnotation" | "Super" | "SwitchCase" | "SwitchStatement" | "SymbolTypeAnnotation" | "TSAnyKeyword" | "TSArrayType" | "TSAsExpression" | "TSBigIntKeyword" | "TSBooleanKeyword" | "TSCallSignatureDeclaration" | "TSConditionalType" | "TSConstructSignatureDeclaration" | "TSConstructorType" | "TSDeclareFunction" | "TSDeclareMethod" | "TSEnumDeclaration" | "TSEnumMember" | "TSExportAssignment" | "TSExpressionWithTypeArguments" | "TSExternalModuleReference" | "TSFunctionType" | "TSImportEqualsDeclaration" | "TSImportType" | "TSIndexSignature" | "TSIndexedAccessType" | "TSInferType" | "TSInstantiationExpression" | "TSInterfaceBody" | "TSInterfaceDeclaration" | "TSIntersectionType" | "TSIntrinsicKeyword" | "TSLiteralType" | "TSMappedType" | "TSMethodSignature" | "TSModuleBlock" | "TSModuleDeclaration" | "TSNamedTupleMember" | "TSNamespaceExportDeclaration" | "TSNeverKeyword" | "TSNonNullExpression" | "TSNullKeyword" | "TSNumberKeyword" | "TSObjectKeyword" | "TSOptionalType" | "TSParameterProperty" | "TSParenthesizedType" | "TSPropertySignature" | "TSQualifiedName" | "TSRestType" | "TSSatisfiesExpression" | "TSStringKeyword" | "TSSymbolKeyword" | "TSThisType" | "TSTupleType" | "TSTypeAliasDeclaration" | "TSTypeAnnotation" | "TSTypeAssertion" | "TSTypeLiteral" | "TSTypeOperator" | "TSTypeParameter" | "TSTypeParameterDeclaration" | "TSTypeParameterInstantiation" | "TSTypePredicate" | "TSTypeQuery" | "TSTypeReference" | "TSUndefinedKeyword" | "TSUnionType" | "TSUnknownKeyword" | "TSVoidKeyword" | "TaggedTemplateExpression" | "TemplateElement" | "TemplateLiteral" | "ThisExpression" | "ThisTypeAnnotation" | "ThrowStatement" | "TopicReference" | "TryStatement" | "TupleExpression" | "TupleTypeAnnotation" | "TypeAlias" | "TypeAnnotation" | "TypeCastExpression" | "TypeParameter" | "TypeParameterDeclaration" | "TypeParameterInstantiation" | "TypeofTypeAnnotation" | "UnaryExpression" | "UnionTypeAnnotation" | "UpdateExpression" | "V8IntrinsicIdentifier" | "VariableDeclaration" | "VariableDeclarator" | "Variance" | "VoidTypeAnnotation" | "WhileStatement" | "WithStatement" | "YieldExpression" | keyof Aliases)[];
declare const TSTYPEELEMENT_TYPES: ("StringLiteral" | "NumericLiteral" | "JSXText" | "JSXIdentifier" | "AnyTypeAnnotation" | "ArgumentPlaceholder" | "ArrayExpression" | "ArrayPattern" | "ArrayTypeAnnotation" | "ArrowFunctionExpression" | "AssignmentExpression" | "AssignmentPattern" | "AwaitExpression" | "BigIntLiteral" | "BinaryExpression" | "BindExpression" | "BlockStatement" | "BooleanLiteral" | "BooleanLiteralTypeAnnotation" | "BooleanTypeAnnotation" | "BreakStatement" | "CallExpression" | "CatchClause" | "ClassAccessorProperty" | "ClassBody" | "ClassDeclaration" | "ClassExpression" | "ClassImplements" | "ClassMethod" | "ClassPrivateMethod" | "ClassPrivateProperty" | "ClassProperty" | "ConditionalExpression" | "ContinueStatement" | "DebuggerStatement" | "DecimalLiteral" | "DeclareClass" | "DeclareExportAllDeclaration" | "DeclareExportDeclaration" | "DeclareFunction" | "DeclareInterface" | "DeclareModule" | "DeclareModuleExports" | "DeclareOpaqueType" | "DeclareTypeAlias" | "DeclareVariable" | "DeclaredPredicate" | "Decorator" | "Directive" | "DirectiveLiteral" | "DoExpression" | "DoWhileStatement" | "EmptyStatement" | "EmptyTypeAnnotation" | "EnumBooleanBody" | "EnumBooleanMember" | "EnumDeclaration" | "EnumDefaultedMember" | "EnumNumberBody" | "EnumNumberMember" | "EnumStringBody" | "EnumStringMember" | "EnumSymbolBody" | "ExistsTypeAnnotation" | "ExportAllDeclaration" | "ExportDefaultDeclaration" | "ExportDefaultSpecifier" | "ExportNamedDeclaration" | "ExportNamespaceSpecifier" | "ExportSpecifier" | "ExpressionStatement" | "File" | "ForInStatement" | "ForOfStatement" | "ForStatement" | "FunctionDeclaration" | "FunctionExpression" | "FunctionTypeAnnotation" | "FunctionTypeParam" | "GenericTypeAnnotation" | "Identifier" | "IfStatement" | "Import" | "ImportAttribute" | "ImportDeclaration" | "ImportDefaultSpecifier" | "ImportExpression" | "ImportNamespaceSpecifier" | "ImportSpecifier" | "IndexedAccessType" | "InferredPredicate" | "InterfaceDeclaration" | "InterfaceExtends" | "InterfaceTypeAnnotation" | "InterpreterDirective" | "IntersectionTypeAnnotation" | "JSXAttribute" | "JSXClosingElement" | "JSXClosingFragment" | "JSXElement" | "JSXEmptyExpression" | "JSXExpressionContainer" | "JSXFragment" | "JSXMemberExpression" | "JSXNamespacedName" | "JSXOpeningElement" | "JSXOpeningFragment" | "JSXSpreadAttribute" | "JSXSpreadChild" | "LabeledStatement" | "LogicalExpression" | "MemberExpression" | "MetaProperty" | "MixedTypeAnnotation" | "ModuleExpression" | "NewExpression" | "Noop" | "NullLiteral" | "NullLiteralTypeAnnotation" | "NullableTypeAnnotation" | "NumberLiteral" | "NumberLiteralTypeAnnotation" | "NumberTypeAnnotation" | "ObjectExpression" | "ObjectMethod" | "ObjectPattern" | "ObjectProperty" | "ObjectTypeAnnotation" | "ObjectTypeCallProperty" | "ObjectTypeIndexer" | "ObjectTypeInternalSlot" | "ObjectTypeProperty" | "ObjectTypeSpreadProperty" | "OpaqueType" | "OptionalCallExpression" | "OptionalIndexedAccessType" | "OptionalMemberExpression" | "ParenthesizedExpression" | "PipelineBareFunction" | "PipelinePrimaryTopicReference" | "PipelineTopicExpression" | "Placeholder" | "PrivateName" | "Program" | "QualifiedTypeIdentifier" | "RecordExpression" | "RegExpLiteral" | "RegexLiteral" | "RestElement" | "RestProperty" | "ReturnStatement" | "SequenceExpression" | "SpreadElement" | "SpreadProperty" | "StaticBlock" | "StringLiteralTypeAnnotation" | "StringTypeAnnotation" | "Super" | "SwitchCase" | "SwitchStatement" | "SymbolTypeAnnotation" | "TSAnyKeyword" | "TSArrayType" | "TSAsExpression" | "TSBigIntKeyword" | "TSBooleanKeyword" | "TSCallSignatureDeclaration" | "TSConditionalType" | "TSConstructSignatureDeclaration" | "TSConstructorType" | "TSDeclareFunction" | "TSDeclareMethod" | "TSEnumDeclaration" | "TSEnumMember" | "TSExportAssignment" | "TSExpressionWithTypeArguments" | "TSExternalModuleReference" | "TSFunctionType" | "TSImportEqualsDeclaration" | "TSImportType" | "TSIndexSignature" | "TSIndexedAccessType" | "TSInferType" | "TSInstantiationExpression" | "TSInterfaceBody" | "TSInterfaceDeclaration" | "TSIntersectionType" | "TSIntrinsicKeyword" | "TSLiteralType" | "TSMappedType" | "TSMethodSignature" | "TSModuleBlock" | "TSModuleDeclaration" | "TSNamedTupleMember" | "TSNamespaceExportDeclaration" | "TSNeverKeyword" | "TSNonNullExpression" | "TSNullKeyword" | "TSNumberKeyword" | "TSObjectKeyword" | "TSOptionalType" | "TSParameterProperty" | "TSParenthesizedType" | "TSPropertySignature" | "TSQualifiedName" | "TSRestType" | "TSSatisfiesExpression" | "TSStringKeyword" | "TSSymbolKeyword" | "TSThisType" | "TSTupleType" | "TSTypeAliasDeclaration" | "TSTypeAnnotation" | "TSTypeAssertion" | "TSTypeLiteral" | "TSTypeOperator" | "TSTypeParameter" | "TSTypeParameterDeclaration" | "TSTypeParameterInstantiation" | "TSTypePredicate" | "TSTypeQuery" | "TSTypeReference" | "TSUndefinedKeyword" | "TSUnionType" | "TSUnknownKeyword" | "TSVoidKeyword" | "TaggedTemplateExpression" | "TemplateElement" | "TemplateLiteral" | "ThisExpression" | "ThisTypeAnnotation" | "ThrowStatement" | "TopicReference" | "TryStatement" | "TupleExpression" | "TupleTypeAnnotation" | "TypeAlias" | "TypeAnnotation" | "TypeCastExpression" | "TypeParameter" | "TypeParameterDeclaration" | "TypeParameterInstantiation" | "TypeofTypeAnnotation" | "UnaryExpression" | "UnionTypeAnnotation" | "UpdateExpression" | "V8IntrinsicIdentifier" | "VariableDeclaration" | "VariableDeclarator" | "Variance" | "VoidTypeAnnotation" | "WhileStatement" | "WithStatement" | "YieldExpression" | keyof Aliases)[];
declare const TSTYPE_TYPES: ("StringLiteral" | "NumericLiteral" | "JSXText" | "JSXIdentifier" | "AnyTypeAnnotation" | "ArgumentPlaceholder" | "ArrayExpression" | "ArrayPattern" | "ArrayTypeAnnotation" | "ArrowFunctionExpression" | "AssignmentExpression" | "AssignmentPattern" | "AwaitExpression" | "BigIntLiteral" | "BinaryExpression" | "BindExpression" | "BlockStatement" | "BooleanLiteral" | "BooleanLiteralTypeAnnotation" | "BooleanTypeAnnotation" | "BreakStatement" | "CallExpression" | "CatchClause" | "ClassAccessorProperty" | "ClassBody" | "ClassDeclaration" | "ClassExpression" | "ClassImplements" | "ClassMethod" | "ClassPrivateMethod" | "ClassPrivateProperty" | "ClassProperty" | "ConditionalExpression" | "ContinueStatement" | "DebuggerStatement" | "DecimalLiteral" | "DeclareClass" | "DeclareExportAllDeclaration" | "DeclareExportDeclaration" | "DeclareFunction" | "DeclareInterface" | "DeclareModule" | "DeclareModuleExports" | "DeclareOpaqueType" | "DeclareTypeAlias" | "DeclareVariable" | "DeclaredPredicate" | "Decorator" | "Directive" | "DirectiveLiteral" | "DoExpression" | "DoWhileStatement" | "EmptyStatement" | "EmptyTypeAnnotation" | "EnumBooleanBody" | "EnumBooleanMember" | "EnumDeclaration" | "EnumDefaultedMember" | "EnumNumberBody" | "EnumNumberMember" | "EnumStringBody" | "EnumStringMember" | "EnumSymbolBody" | "ExistsTypeAnnotation" | "ExportAllDeclaration" | "ExportDefaultDeclaration" | "ExportDefaultSpecifier" | "ExportNamedDeclaration" | "ExportNamespaceSpecifier" | "ExportSpecifier" | "ExpressionStatement" | "File" | "ForInStatement" | "ForOfStatement" | "ForStatement" | "FunctionDeclaration" | "FunctionExpression" | "FunctionTypeAnnotation" | "FunctionTypeParam" | "GenericTypeAnnotation" | "Identifier" | "IfStatement" | "Import" | "ImportAttribute" | "ImportDeclaration" | "ImportDefaultSpecifier" | "ImportExpression" | "ImportNamespaceSpecifier" | "ImportSpecifier" | "IndexedAccessType" | "InferredPredicate" | "InterfaceDeclaration" | "InterfaceExtends" | "InterfaceTypeAnnotation" | "InterpreterDirective" | "IntersectionTypeAnnotation" | "JSXAttribute" | "JSXClosingElement" | "JSXClosingFragment" | "JSXElement" | "JSXEmptyExpression" | "JSXExpressionContainer" | "JSXFragment" | "JSXMemberExpression" | "JSXNamespacedName" | "JSXOpeningElement" | "JSXOpeningFragment" | "JSXSpreadAttribute" | "JSXSpreadChild" | "LabeledStatement" | "LogicalExpression" | "MemberExpression" | "MetaProperty" | "MixedTypeAnnotation" | "ModuleExpression" | "NewExpression" | "Noop" | "NullLiteral" | "NullLiteralTypeAnnotation" | "NullableTypeAnnotation" | "NumberLiteral" | "NumberLiteralTypeAnnotation" | "NumberTypeAnnotation" | "ObjectExpression" | "ObjectMethod" | "ObjectPattern" | "ObjectProperty" | "ObjectTypeAnnotation" | "ObjectTypeCallProperty" | "ObjectTypeIndexer" | "ObjectTypeInternalSlot" | "ObjectTypeProperty" | "ObjectTypeSpreadProperty" | "OpaqueType" | "OptionalCallExpression" | "OptionalIndexedAccessType" | "OptionalMemberExpression" | "ParenthesizedExpression" | "PipelineBareFunction" | "PipelinePrimaryTopicReference" | "PipelineTopicExpression" | "Placeholder" | "PrivateName" | "Program" | "QualifiedTypeIdentifier" | "RecordExpression" | "RegExpLiteral" | "RegexLiteral" | "RestElement" | "RestProperty" | "ReturnStatement" | "SequenceExpression" | "SpreadElement" | "SpreadProperty" | "StaticBlock" | "StringLiteralTypeAnnotation" | "StringTypeAnnotation" | "Super" | "SwitchCase" | "SwitchStatement" | "SymbolTypeAnnotation" | "TSAnyKeyword" | "TSArrayType" | "TSAsExpression" | "TSBigIntKeyword" | "TSBooleanKeyword" | "TSCallSignatureDeclaration" | "TSConditionalType" | "TSConstructSignatureDeclaration" | "TSConstructorType" | "TSDeclareFunction" | "TSDeclareMethod" | "TSEnumDeclaration" | "TSEnumMember" | "TSExportAssignment" | "TSExpressionWithTypeArguments" | "TSExternalModuleReference" | "TSFunctionType" | "TSImportEqualsDeclaration" | "TSImportType" | "TSIndexSignature" | "TSIndexedAccessType" | "TSInferType" | "TSInstantiationExpression" | "TSInterfaceBody" | "TSInterfaceDeclaration" | "TSIntersectionType" | "TSIntrinsicKeyword" | "TSLiteralType" | "TSMappedType" | "TSMethodSignature" | "TSModuleBlock" | "TSModuleDeclaration" | "TSNamedTupleMember" | "TSNamespaceExportDeclaration" | "TSNeverKeyword" | "TSNonNullExpression" | "TSNullKeyword" | "TSNumberKeyword" | "TSObjectKeyword" | "TSOptionalType" | "TSParameterProperty" | "TSParenthesizedType" | "TSPropertySignature" | "TSQualifiedName" | "TSRestType" | "TSSatisfiesExpression" | "TSStringKeyword" | "TSSymbolKeyword" | "TSThisType" | "TSTupleType" | "TSTypeAliasDeclaration" | "TSTypeAnnotation" | "TSTypeAssertion" | "TSTypeLiteral" | "TSTypeOperator" | "TSTypeParameter" | "TSTypeParameterDeclaration" | "TSTypeParameterInstantiation" | "TSTypePredicate" | "TSTypeQuery" | "TSTypeReference" | "TSUndefinedKeyword" | "TSUnionType" | "TSUnknownKeyword" | "TSVoidKeyword" | "TaggedTemplateExpression" | "TemplateElement" | "TemplateLiteral" | "ThisExpression" | "ThisTypeAnnotation" | "ThrowStatement" | "TopicReference" | "TryStatement" | "TupleExpression" | "TupleTypeAnnotation" | "TypeAlias" | "TypeAnnotation" | "TypeCastExpression" | "TypeParameter" | "TypeParameterDeclaration" | "TypeParameterInstantiation" | "TypeofTypeAnnotation" | "UnaryExpression" | "UnionTypeAnnotation" | "UpdateExpression" | "V8IntrinsicIdentifier" | "VariableDeclaration" | "VariableDeclarator" | "Variance" | "VoidTypeAnnotation" | "WhileStatement" | "WithStatement" | "YieldExpression" | keyof Aliases)[];
declare const TSBASETYPE_TYPES: ("StringLiteral" | "NumericLiteral" | "JSXText" | "JSXIdentifier" | "AnyTypeAnnotation" | "ArgumentPlaceholder" | "ArrayExpression" | "ArrayPattern" | "ArrayTypeAnnotation" | "ArrowFunctionExpression" | "AssignmentExpression" | "AssignmentPattern" | "AwaitExpression" | "BigIntLiteral" | "BinaryExpression" | "BindExpression" | "BlockStatement" | "BooleanLiteral" | "BooleanLiteralTypeAnnotation" | "BooleanTypeAnnotation" | "BreakStatement" | "CallExpression" | "CatchClause" | "ClassAccessorProperty" | "ClassBody" | "ClassDeclaration" | "ClassExpression" | "ClassImplements" | "ClassMethod" | "ClassPrivateMethod" | "ClassPrivateProperty" | "ClassProperty" | "ConditionalExpression" | "ContinueStatement" | "DebuggerStatement" | "DecimalLiteral" | "DeclareClass" | "DeclareExportAllDeclaration" | "DeclareExportDeclaration" | "DeclareFunction" | "DeclareInterface" | "DeclareModule" | "DeclareModuleExports" | "DeclareOpaqueType" | "DeclareTypeAlias" | "DeclareVariable" | "DeclaredPredicate" | "Decorator" | "Directive" | "DirectiveLiteral" | "DoExpression" | "DoWhileStatement" | "EmptyStatement" | "EmptyTypeAnnotation" | "EnumBooleanBody" | "EnumBooleanMember" | "EnumDeclaration" | "EnumDefaultedMember" | "EnumNumberBody" | "EnumNumberMember" | "EnumStringBody" | "EnumStringMember" | "EnumSymbolBody" | "ExistsTypeAnnotation" | "ExportAllDeclaration" | "ExportDefaultDeclaration" | "ExportDefaultSpecifier" | "ExportNamedDeclaration" | "ExportNamespaceSpecifier" | "ExportSpecifier" | "ExpressionStatement" | "File" | "ForInStatement" | "ForOfStatement" | "ForStatement" | "FunctionDeclaration" | "FunctionExpression" | "FunctionTypeAnnotation" | "FunctionTypeParam" | "GenericTypeAnnotation" | "Identifier" | "IfStatement" | "Import" | "ImportAttribute" | "ImportDeclaration" | "ImportDefaultSpecifier" | "ImportExpression" | "ImportNamespaceSpecifier" | "ImportSpecifier" | "IndexedAccessType" | "InferredPredicate" | "InterfaceDeclaration" | "InterfaceExtends" | "InterfaceTypeAnnotation" | "InterpreterDirective" | "IntersectionTypeAnnotation" | "JSXAttribute" | "JSXClosingElement" | "JSXClosingFragment" | "JSXElement" | "JSXEmptyExpression" | "JSXExpressionContainer" | "JSXFragment" | "JSXMemberExpression" | "JSXNamespacedName" | "JSXOpeningElement" | "JSXOpeningFragment" | "JSXSpreadAttribute" | "JSXSpreadChild" | "LabeledStatement" | "LogicalExpression" | "MemberExpression" | "MetaProperty" | "MixedTypeAnnotation" | "ModuleExpression" | "NewExpression" | "Noop" | "NullLiteral" | "NullLiteralTypeAnnotation" | "NullableTypeAnnotation" | "NumberLiteral" | "NumberLiteralTypeAnnotation" | "NumberTypeAnnotation" | "ObjectExpression" | "ObjectMethod" | "ObjectPattern" | "ObjectProperty" | "ObjectTypeAnnotation" | "ObjectTypeCallProperty" | "ObjectTypeIndexer" | "ObjectTypeInternalSlot" | "ObjectTypeProperty" | "ObjectTypeSpreadProperty" | "OpaqueType" | "OptionalCallExpression" | "OptionalIndexedAccessType" | "OptionalMemberExpression" | "ParenthesizedExpression" | "PipelineBareFunction" | "PipelinePrimaryTopicReference" | "PipelineTopicExpression" | "Placeholder" | "PrivateName" | "Program" | "QualifiedTypeIdentifier" | "RecordExpression" | "RegExpLiteral" | "RegexLiteral" | "RestElement" | "RestProperty" | "ReturnStatement" | "SequenceExpression" | "SpreadElement" | "SpreadProperty" | "StaticBlock" | "StringLiteralTypeAnnotation" | "StringTypeAnnotation" | "Super" | "SwitchCase" | "SwitchStatement" | "SymbolTypeAnnotation" | "TSAnyKeyword" | "TSArrayType" | "TSAsExpression" | "TSBigIntKeyword" | "TSBooleanKeyword" | "TSCallSignatureDeclaration" | "TSConditionalType" | "TSConstructSignatureDeclaration" | "TSConstructorType" | "TSDeclareFunction" | "TSDeclareMethod" | "TSEnumDeclaration" | "TSEnumMember" | "TSExportAssignment" | "TSExpressionWithTypeArguments" | "TSExternalModuleReference" | "TSFunctionType" | "TSImportEqualsDeclaration" | "TSImportType" | "TSIndexSignature" | "TSIndexedAccessType" | "TSInferType" | "TSInstantiationExpression" | "TSInterfaceBody" | "TSInterfaceDeclaration" | "TSIntersectionType" | "TSIntrinsicKeyword" | "TSLiteralType" | "TSMappedType" | "TSMethodSignature" | "TSModuleBlock" | "TSModuleDeclaration" | "TSNamedTupleMember" | "TSNamespaceExportDeclaration" | "TSNeverKeyword" | "TSNonNullExpression" | "TSNullKeyword" | "TSNumberKeyword" | "TSObjectKeyword" | "TSOptionalType" | "TSParameterProperty" | "TSParenthesizedType" | "TSPropertySignature" | "TSQualifiedName" | "TSRestType" | "TSSatisfiesExpression" | "TSStringKeyword" | "TSSymbolKeyword" | "TSThisType" | "TSTupleType" | "TSTypeAliasDeclaration" | "TSTypeAnnotation" | "TSTypeAssertion" | "TSTypeLiteral" | "TSTypeOperator" | "TSTypeParameter" | "TSTypeParameterDeclaration" | "TSTypeParameterInstantiation" | "TSTypePredicate" | "TSTypeQuery" | "TSTypeReference" | "TSUndefinedKeyword" | "TSUnionType" | "TSUnknownKeyword" | "TSVoidKeyword" | "TaggedTemplateExpression" | "TemplateElement" | "TemplateLiteral" | "ThisExpression" | "ThisTypeAnnotation" | "ThrowStatement" | "TopicReference" | "TryStatement" | "TupleExpression" | "TupleTypeAnnotation" | "TypeAlias" | "TypeAnnotation" | "TypeCastExpression" | "TypeParameter" | "TypeParameterDeclaration" | "TypeParameterInstantiation" | "TypeofTypeAnnotation" | "UnaryExpression" | "UnionTypeAnnotation" | "UpdateExpression" | "V8IntrinsicIdentifier" | "VariableDeclaration" | "VariableDeclarator" | "Variance" | "VoidTypeAnnotation" | "WhileStatement" | "WithStatement" | "YieldExpression" | keyof Aliases)[];
/**
 * @deprecated migrate to IMPORTOREXPORTDECLARATION_TYPES.
 */
declare const MODULEDECLARATION_TYPES: ("StringLiteral" | "NumericLiteral" | "JSXText" | "JSXIdentifier" | "AnyTypeAnnotation" | "ArgumentPlaceholder" | "ArrayExpression" | "ArrayPattern" | "ArrayTypeAnnotation" | "ArrowFunctionExpression" | "AssignmentExpression" | "AssignmentPattern" | "AwaitExpression" | "BigIntLiteral" | "BinaryExpression" | "BindExpression" | "BlockStatement" | "BooleanLiteral" | "BooleanLiteralTypeAnnotation" | "BooleanTypeAnnotation" | "BreakStatement" | "CallExpression" | "CatchClause" | "ClassAccessorProperty" | "ClassBody" | "ClassDeclaration" | "ClassExpression" | "ClassImplements" | "ClassMethod" | "ClassPrivateMethod" | "ClassPrivateProperty" | "ClassProperty" | "ConditionalExpression" | "ContinueStatement" | "DebuggerStatement" | "DecimalLiteral" | "DeclareClass" | "DeclareExportAllDeclaration" | "DeclareExportDeclaration" | "DeclareFunction" | "DeclareInterface" | "DeclareModule" | "DeclareModuleExports" | "DeclareOpaqueType" | "DeclareTypeAlias" | "DeclareVariable" | "DeclaredPredicate" | "Decorator" | "Directive" | "DirectiveLiteral" | "DoExpression" | "DoWhileStatement" | "EmptyStatement" | "EmptyTypeAnnotation" | "EnumBooleanBody" | "EnumBooleanMember" | "EnumDeclaration" | "EnumDefaultedMember" | "EnumNumberBody" | "EnumNumberMember" | "EnumStringBody" | "EnumStringMember" | "EnumSymbolBody" | "ExistsTypeAnnotation" | "ExportAllDeclaration" | "ExportDefaultDeclaration" | "ExportDefaultSpecifier" | "ExportNamedDeclaration" | "ExportNamespaceSpecifier" | "ExportSpecifier" | "ExpressionStatement" | "File" | "ForInStatement" | "ForOfStatement" | "ForStatement" | "FunctionDeclaration" | "FunctionExpression" | "FunctionTypeAnnotation" | "FunctionTypeParam" | "GenericTypeAnnotation" | "Identifier" | "IfStatement" | "Import" | "ImportAttribute" | "ImportDeclaration" | "ImportDefaultSpecifier" | "ImportExpression" | "ImportNamespaceSpecifier" | "ImportSpecifier" | "IndexedAccessType" | "InferredPredicate" | "InterfaceDeclaration" | "InterfaceExtends" | "InterfaceTypeAnnotation" | "InterpreterDirective" | "IntersectionTypeAnnotation" | "JSXAttribute" | "JSXClosingElement" | "JSXClosingFragment" | "JSXElement" | "JSXEmptyExpression" | "JSXExpressionContainer" | "JSXFragment" | "JSXMemberExpression" | "JSXNamespacedName" | "JSXOpeningElement" | "JSXOpeningFragment" | "JSXSpreadAttribute" | "JSXSpreadChild" | "LabeledStatement" | "LogicalExpression" | "MemberExpression" | "MetaProperty" | "MixedTypeAnnotation" | "ModuleExpression" | "NewExpression" | "Noop" | "NullLiteral" | "NullLiteralTypeAnnotation" | "NullableTypeAnnotation" | "NumberLiteral" | "NumberLiteralTypeAnnotation" | "NumberTypeAnnotation" | "ObjectExpression" | "ObjectMethod" | "ObjectPattern" | "ObjectProperty" | "ObjectTypeAnnotation" | "ObjectTypeCallProperty" | "ObjectTypeIndexer" | "ObjectTypeInternalSlot" | "ObjectTypeProperty" | "ObjectTypeSpreadProperty" | "OpaqueType" | "OptionalCallExpression" | "OptionalIndexedAccessType" | "OptionalMemberExpression" | "ParenthesizedExpression" | "PipelineBareFunction" | "PipelinePrimaryTopicReference" | "PipelineTopicExpression" | "Placeholder" | "PrivateName" | "Program" | "QualifiedTypeIdentifier" | "RecordExpression" | "RegExpLiteral" | "RegexLiteral" | "RestElement" | "RestProperty" | "ReturnStatement" | "SequenceExpression" | "SpreadElement" | "SpreadProperty" | "StaticBlock" | "StringLiteralTypeAnnotation" | "StringTypeAnnotation" | "Super" | "SwitchCase" | "SwitchStatement" | "SymbolTypeAnnotation" | "TSAnyKeyword" | "TSArrayType" | "TSAsExpression" | "TSBigIntKeyword" | "TSBooleanKeyword" | "TSCallSignatureDeclaration" | "TSConditionalType" | "TSConstructSignatureDeclaration" | "TSConstructorType" | "TSDeclareFunction" | "TSDeclareMethod" | "TSEnumDeclaration" | "TSEnumMember" | "TSExportAssignment" | "TSExpressionWithTypeArguments" | "TSExternalModuleReference" | "TSFunctionType" | "TSImportEqualsDeclaration" | "TSImportType" | "TSIndexSignature" | "TSIndexedAccessType" | "TSInferType" | "TSInstantiationExpression" | "TSInterfaceBody" | "TSInterfaceDeclaration" | "TSIntersectionType" | "TSIntrinsicKeyword" | "TSLiteralType" | "TSMappedType" | "TSMethodSignature" | "TSModuleBlock" | "TSModuleDeclaration" | "TSNamedTupleMember" | "TSNamespaceExportDeclaration" | "TSNeverKeyword" | "TSNonNullExpression" | "TSNullKeyword" | "TSNumberKeyword" | "TSObjectKeyword" | "TSOptionalType" | "TSParameterProperty" | "TSParenthesizedType" | "TSPropertySignature" | "TSQualifiedName" | "TSRestType" | "TSSatisfiesExpression" | "TSStringKeyword" | "TSSymbolKeyword" | "TSThisType" | "TSTupleType" | "TSTypeAliasDeclaration" | "TSTypeAnnotation" | "TSTypeAssertion" | "TSTypeLiteral" | "TSTypeOperator" | "TSTypeParameter" | "TSTypeParameterDeclaration" | "TSTypeParameterInstantiation" | "TSTypePredicate" | "TSTypeQuery" | "TSTypeReference" | "TSUndefinedKeyword" | "TSUnionType" | "TSUnknownKeyword" | "TSVoidKeyword" | "TaggedTemplateExpression" | "TemplateElement" | "TemplateLiteral" | "ThisExpression" | "ThisTypeAnnotation" | "ThrowStatement" | "TopicReference" | "TryStatement" | "TupleExpression" | "TupleTypeAnnotation" | "TypeAlias" | "TypeAnnotation" | "TypeCastExpression" | "TypeParameter" | "TypeParameterDeclaration" | "TypeParameterInstantiation" | "TypeofTypeAnnotation" | "UnaryExpression" | "UnionTypeAnnotation" | "UpdateExpression" | "V8IntrinsicIdentifier" | "VariableDeclaration" | "VariableDeclarator" | "Variance" | "VoidTypeAnnotation" | "WhileStatement" | "WithStatement" | "YieldExpression" | keyof Aliases)[];

declare const STATEMENT_OR_BLOCK_KEYS: string[];
declare const FLATTENABLE_KEYS: string[];
declare const FOR_INIT_KEYS: string[];
declare const COMMENT_KEYS: readonly ["leadingComments", "trailingComments", "innerComments"];
declare const LOGICAL_OPERATORS: string[];
declare const UPDATE_OPERATORS: string[];
declare const BOOLEAN_NUMBER_BINARY_OPERATORS: string[];
declare const EQUALITY_BINARY_OPERATORS: string[];
declare const COMPARISON_BINARY_OPERATORS: string[];
declare const BOOLEAN_BINARY_OPERATORS: string[];
declare const NUMBER_BINARY_OPERATORS: string[];
declare const BINARY_OPERATORS: string[];
declare const ASSIGNMENT_OPERATORS: string[];
declare const BOOLEAN_UNARY_OPERATORS: string[];
declare const NUMBER_UNARY_OPERATORS: string[];
declare const STRING_UNARY_OPERATORS: string[];
declare const UNARY_OPERATORS: string[];
declare const INHERIT_KEYS: {
    readonly optional: readonly ["typeAnnotation", "typeParameters", "returnType"];
    readonly force: readonly ["start", "loc", "end"];
};
declare const BLOCK_SCOPED_SYMBOL: unique symbol;
declare const NOT_LOCAL_BINDING: unique symbol;

/**
 * Ensure the `key` (defaults to "body") of a `node` is a block.
 * Casting it to a block if it is not.
 *
 * Returns the BlockStatement
 */
declare function ensureBlock(node: Node, key?: string): BlockStatement;

declare function toBindingIdentifierName(name: string): string;

declare function toBlock(node: Statement | Expression, parent?: Node): BlockStatement;

declare function toComputedKey(node: ObjectMember | ObjectProperty | ClassMethod | ClassProperty | ClassAccessorProperty | MemberExpression | OptionalMemberExpression, key?: Expression | PrivateName): PrivateName | Expression;

declare const _default$3: {
    (node: Function): FunctionExpression;
    (node: Class): ClassExpression;
    (node: ExpressionStatement | Expression | Class | Function): Expression;
};
//# sourceMappingURL=toExpression.d.ts.map

declare function toIdentifier(input: string): string;

declare function toKeyAlias(node: Method | Property, key?: Node): string;
declare namespace toKeyAlias {
    var uid: number;
    var increment: () => number;
}
//# sourceMappingURL=toKeyAlias.d.ts.map

declare const _default$2: {
    (node: AssignmentExpression, ignore?: boolean): ExpressionStatement;
    <T extends Statement>(node: T, ignore: false): T;
    <T_1 extends Statement>(node: T_1, ignore?: boolean): false | T_1;
    (node: Class, ignore: false): ClassDeclaration;
    (node: Class, ignore?: boolean): ClassDeclaration | false;
    (node: Function, ignore: false): FunctionDeclaration;
    (node: Function, ignore?: boolean): FunctionDeclaration | false;
    (node: Node, ignore: false): Statement;
    (node: Node, ignore?: boolean): Statement | false;
};
//# sourceMappingURL=toStatement.d.ts.map

declare const _default$1: {
    (value: undefined): Identifier;
    (value: boolean): BooleanLiteral;
    (value: null): NullLiteral;
    (value: string): StringLiteral;
    (value: number): NumericLiteral | BinaryExpression | UnaryExpression;
    (value: RegExp): RegExpLiteral;
    (value: ReadonlyArray<unknown>): ArrayExpression;
    (value: object): ObjectExpression;
    (value: unknown): Expression;
};
//# sourceMappingURL=valueToNode.d.ts.map

declare const VISITOR_KEYS: Record<string, string[]>;
declare const ALIAS_KEYS: Partial<Record<NodeTypesWithoutComment, string[]>>;
declare const FLIPPED_ALIAS_KEYS: Record<string, NodeTypesWithoutComment[]>;
declare const NODE_FIELDS: Record<string, FieldDefinitions>;
declare const BUILDER_KEYS: Record<string, string[]>;
declare const DEPRECATED_KEYS: Record<string, NodeTypesWithoutComment>;
declare const NODE_PARENT_VALIDATIONS: Record<string, Validator>;
declare function getType(val: any): "string" | "number" | "bigint" | "boolean" | "symbol" | "undefined" | "object" | "function" | "null" | "array";
type NodeTypesWithoutComment = Node["type"] | keyof Aliases;
type NodeTypes = NodeTypesWithoutComment | Comment["type"];
type PrimitiveTypes = ReturnType<typeof getType>;
type FieldDefinitions = {
    [x: string]: FieldOptions;
};
type Validator = ({
    type: PrimitiveTypes;
} | {
    each: Validator;
} | {
    chainOf: Validator[];
} | {
    oneOf: any[];
} | {
    oneOfNodeTypes: NodeTypes[];
} | {
    oneOfNodeOrValueTypes: (NodeTypes | PrimitiveTypes)[];
} | {
    shapeOf: {
        [x: string]: FieldOptions;
    };
} | {}) & ((node: Node, key: string, val: any) => void);
type FieldOptions = {
    default?: string | number | boolean | [];
    optional?: boolean;
    deprecated?: boolean;
    validate?: Validator;
};

declare const PLACEHOLDERS: readonly ["Identifier", "StringLiteral", "Expression", "Statement", "Declaration", "BlockStatement", "ClassBody", "Pattern"];
declare const PLACEHOLDERS_ALIAS: Record<string, string[]>;
declare const PLACEHOLDERS_FLIPPED_ALIAS: Record<string, string[]>;

declare const DEPRECATED_ALIASES: {
    ModuleDeclaration: string;
};

declare const TYPES: Array<string>;
//# sourceMappingURL=index.d.ts.map

/**
 * Append a node to a member expression.
 */
declare function appendToMemberExpression(member: MemberExpression, append: MemberExpression["property"], computed?: boolean): MemberExpression;

/**
 * Inherit all contextual properties from `parent` node to `child` node.
 */
declare function inherits<T extends Node | null | undefined>(child: T, parent: Node | null | undefined): T;

/**
 * Prepend a node to a member expression.
 */
declare function prependToMemberExpression<T extends Pick<MemberExpression, "object" | "property">>(member: T, prepend: MemberExpression["object"]): T;

type Options = {
    preserveComments?: boolean;
};
/**
 * Remove all of the _* properties from a node along with the additional metadata
 * properties like location data and raw token data.
 */
declare function removeProperties(node: Node, opts?: Options): void;

declare function removePropertiesDeep<T extends Node>(tree: T, opts?: {
    preserveComments: boolean;
} | null): T;

/**
 * Dedupe type annotations.
 */
declare function removeTypeDuplicates(nodesIn: ReadonlyArray<FlowType | false | null | undefined>): FlowType[];

declare function getBindingIdentifiers(node: Node, duplicates: true, outerOnly?: boolean, newBindingsOnly?: boolean): Record<string, Array<Identifier>>;
declare function getBindingIdentifiers(node: Node, duplicates?: false, outerOnly?: boolean, newBindingsOnly?: boolean): Record<string, Identifier>;
declare function getBindingIdentifiers(node: Node, duplicates?: boolean, outerOnly?: boolean, newBindingsOnly?: boolean): Record<string, Identifier> | Record<string, Array<Identifier>>;
declare namespace getBindingIdentifiers {
    var keys: {
        DeclareClass: string[];
        DeclareFunction: string[];
        DeclareModule: string[];
        DeclareVariable: string[];
        DeclareInterface: string[];
        DeclareTypeAlias: string[];
        DeclareOpaqueType: string[];
        InterfaceDeclaration: string[];
        TypeAlias: string[];
        OpaqueType: string[];
        CatchClause: string[];
        LabeledStatement: string[];
        UnaryExpression: string[];
        AssignmentExpression: string[];
        ImportSpecifier: string[];
        ImportNamespaceSpecifier: string[];
        ImportDefaultSpecifier: string[];
        ImportDeclaration: string[];
        ExportSpecifier: string[];
        ExportNamespaceSpecifier: string[];
        ExportDefaultSpecifier: string[];
        FunctionDeclaration: string[];
        FunctionExpression: string[];
        ArrowFunctionExpression: string[];
        ObjectMethod: string[];
        ClassMethod: string[];
        ClassPrivateMethod: string[];
        ForInStatement: string[];
        ForOfStatement: string[];
        ClassDeclaration: string[];
        ClassExpression: string[];
        RestElement: string[];
        UpdateExpression: string[];
        ObjectProperty: string[];
        AssignmentPattern: string[];
        ArrayPattern: string[];
        ObjectPattern: string[];
        VariableDeclaration: string[];
        VariableDeclarator: string[];
    };
}
//# sourceMappingURL=getBindingIdentifiers.d.ts.map

declare const _default: {
    (node: Node, duplicates: true): Record<string, Array<Identifier>>;
    (node: Node, duplicates?: false): Record<string, Identifier>;
    (node: Node, duplicates?: boolean): Record<string, Identifier> | Record<string, Array<Identifier>>;
};
//# sourceMappingURL=getOuterBindingIdentifiers.d.ts.map

type TraversalAncestors = Array<{
    node: Node;
    key: string;
    index?: number;
}>;
type TraversalHandler<T> = (this: undefined, node: Node, parent: TraversalAncestors, state: T) => void;
type TraversalHandlers<T> = {
    enter?: TraversalHandler<T>;
    exit?: TraversalHandler<T>;
};
/**
 * A general AST traversal with both prefix and postfix handlers, and a
 * state object. Exposes ancestry data to each handler so that more complex
 * AST data can be taken into account.
 */
declare function traverse<T>(node: Node, handlers: TraversalHandler<T> | TraversalHandlers<T>, state?: T): void;

/**
 * A prefix AST traversal implementation meant for simple searching
 * and processing.
 */
declare function traverseFast<Options = {}>(node: Node | null | undefined, enter: (node: Node, opts?: Options) => void, opts?: Options): void;

declare function shallowEqual<T extends object>(actual: object, expected: T): actual is T;

declare function is<T extends Node["type"]>(type: T, node: Node | null | undefined, opts?: undefined): node is Extract<Node, {
    type: T;
}>;
declare function is<T extends Node["type"], P extends Extract<Node, {
    type: T;
}>>(type: T, n: Node | null | undefined, required: Partial<P>): n is P;
declare function is<P extends Node>(type: string, node: Node | null | undefined, opts: Partial<P>): node is P;
declare function is(type: string, node: Node | null | undefined, opts?: Partial<Node>): node is Node;

/**
 * Check if the input `node` is a binding identifier.
 */
declare function isBinding(node: Node, parent: Node, grandparent?: Node): boolean;

/**
 * Check if the input `node` is block scoped.
 */
declare function isBlockScoped(node: Node): boolean;

/**
 * Check if the input `node` is definitely immutable.
 */
declare function isImmutable(node: Node): boolean;

/**
 * Check if the input `node` is a `let` variable declaration.
 */
declare function isLet(node: Node): boolean;

declare function isNode(node: any): node is Node;

/**
 * Check if two nodes are equivalent
 */
declare function isNodesEquivalent<T extends Partial<Node>>(a: T, b: any): b is T;

/**
 * Test if a `placeholderType` is a `targetType` or if `targetType` is an alias of `placeholderType`.
 */
declare function isPlaceholderType(placeholderType: string, targetType: string): boolean;

/**
 * Check if the input `node` is a reference to a bound variable.
 */
declare function isReferenced(node: Node, parent: Node, grandparent?: Node): boolean;

/**
 * Check if the input `node` is a scope.
 */
declare function isScope(node: Node, parent: Node): boolean;

/**
 * Check if the input `specifier` is a `default` import or export.
 */
declare function isSpecifierDefault(specifier: ModuleSpecifier): boolean;

declare function isType<T extends Node["type"]>(nodeType: string, targetType: T): nodeType is T;
declare function isType(nodeType: string | null | undefined, targetType: string): boolean;

/**
 * Check if the input `name` is a valid identifier name according to the ES3 specification.
 *
 * Additional ES3 reserved words are
 */
declare function isValidES3Identifier(name: string): boolean;

/**
 * Check if the input `name` is a valid identifier name
 * and isn't a reserved word.
 */
declare function isValidIdentifier(name: string, reserved?: boolean): boolean;

/**
 * Check if the input `node` is a variable declaration.
 */
declare function isVar(node: Node): boolean;

/**
 * Determines whether or not the input node `member` matches the
 * input `match`.
 *
 * For example, given the match `React.createClass` it would match the
 * parsed nodes of `React.createClass` and `React["createClass"]`.
 */
declare function matchesPattern(member: Node | null | undefined, match: string | string[], allowPartial?: boolean): boolean;

declare function validate(node: Node | undefined | null, key: string, val: any): void;

/**
 * Build a function that when called will return whether or not the
 * input `node` `MemberExpression` matches the input `match`.
 *
 * For example, given the match `React.createClass` it would match the
 * parsed nodes of `React.createClass` and `React["createClass"]`.
 */
declare function buildMatchMemberExpression(match: string, allowPartial?: boolean): (member: Node) => boolean;

type Opts<Obj> = Partial<{
    [Prop in keyof Obj]: Obj[Prop] extends Node ? Node : Obj[Prop] extends Node[] ? Node[] : Obj[Prop];
}>;
declare function isArrayExpression(node: Node | null | undefined, opts?: Opts<ArrayExpression> | null): node is ArrayExpression;
declare function isAssignmentExpression(node: Node | null | undefined, opts?: Opts<AssignmentExpression> | null): node is AssignmentExpression;
declare function isBinaryExpression(node: Node | null | undefined, opts?: Opts<BinaryExpression> | null): node is BinaryExpression;
declare function isInterpreterDirective(node: Node | null | undefined, opts?: Opts<InterpreterDirective> | null): node is InterpreterDirective;
declare function isDirective(node: Node | null | undefined, opts?: Opts<Directive> | null): node is Directive;
declare function isDirectiveLiteral(node: Node | null | undefined, opts?: Opts<DirectiveLiteral> | null): node is DirectiveLiteral;
declare function isBlockStatement(node: Node | null | undefined, opts?: Opts<BlockStatement> | null): node is BlockStatement;
declare function isBreakStatement(node: Node | null | undefined, opts?: Opts<BreakStatement> | null): node is BreakStatement;
declare function isCallExpression(node: Node | null | undefined, opts?: Opts<CallExpression> | null): node is CallExpression;
declare function isCatchClause(node: Node | null | undefined, opts?: Opts<CatchClause> | null): node is CatchClause;
declare function isConditionalExpression(node: Node | null | undefined, opts?: Opts<ConditionalExpression> | null): node is ConditionalExpression;
declare function isContinueStatement(node: Node | null | undefined, opts?: Opts<ContinueStatement> | null): node is ContinueStatement;
declare function isDebuggerStatement(node: Node | null | undefined, opts?: Opts<DebuggerStatement> | null): node is DebuggerStatement;
declare function isDoWhileStatement(node: Node | null | undefined, opts?: Opts<DoWhileStatement> | null): node is DoWhileStatement;
declare function isEmptyStatement(node: Node | null | undefined, opts?: Opts<EmptyStatement> | null): node is EmptyStatement;
declare function isExpressionStatement(node: Node | null | undefined, opts?: Opts<ExpressionStatement> | null): node is ExpressionStatement;
declare function isFile(node: Node | null | undefined, opts?: Opts<File> | null): node is File;
declare function isForInStatement(node: Node | null | undefined, opts?: Opts<ForInStatement> | null): node is ForInStatement;
declare function isForStatement(node: Node | null | undefined, opts?: Opts<ForStatement> | null): node is ForStatement;
declare function isFunctionDeclaration(node: Node | null | undefined, opts?: Opts<FunctionDeclaration> | null): node is FunctionDeclaration;
declare function isFunctionExpression(node: Node | null | undefined, opts?: Opts<FunctionExpression> | null): node is FunctionExpression;
declare function isIdentifier(node: Node | null | undefined, opts?: Opts<Identifier> | null): node is Identifier;
declare function isIfStatement(node: Node | null | undefined, opts?: Opts<IfStatement> | null): node is IfStatement;
declare function isLabeledStatement(node: Node | null | undefined, opts?: Opts<LabeledStatement> | null): node is LabeledStatement;
declare function isStringLiteral(node: Node | null | undefined, opts?: Opts<StringLiteral> | null): node is StringLiteral;
declare function isNumericLiteral(node: Node | null | undefined, opts?: Opts<NumericLiteral> | null): node is NumericLiteral;
declare function isNullLiteral(node: Node | null | undefined, opts?: Opts<NullLiteral> | null): node is NullLiteral;
declare function isBooleanLiteral(node: Node | null | undefined, opts?: Opts<BooleanLiteral> | null): node is BooleanLiteral;
declare function isRegExpLiteral(node: Node | null | undefined, opts?: Opts<RegExpLiteral> | null): node is RegExpLiteral;
declare function isLogicalExpression(node: Node | null | undefined, opts?: Opts<LogicalExpression> | null): node is LogicalExpression;
declare function isMemberExpression(node: Node | null | undefined, opts?: Opts<MemberExpression> | null): node is MemberExpression;
declare function isNewExpression(node: Node | null | undefined, opts?: Opts<NewExpression> | null): node is NewExpression;
declare function isProgram(node: Node | null | undefined, opts?: Opts<Program> | null): node is Program;
declare function isObjectExpression(node: Node | null | undefined, opts?: Opts<ObjectExpression> | null): node is ObjectExpression;
declare function isObjectMethod(node: Node | null | undefined, opts?: Opts<ObjectMethod> | null): node is ObjectMethod;
declare function isObjectProperty(node: Node | null | undefined, opts?: Opts<ObjectProperty> | null): node is ObjectProperty;
declare function isRestElement(node: Node | null | undefined, opts?: Opts<RestElement> | null): node is RestElement;
declare function isReturnStatement(node: Node | null | undefined, opts?: Opts<ReturnStatement> | null): node is ReturnStatement;
declare function isSequenceExpression(node: Node | null | undefined, opts?: Opts<SequenceExpression> | null): node is SequenceExpression;
declare function isParenthesizedExpression(node: Node | null | undefined, opts?: Opts<ParenthesizedExpression> | null): node is ParenthesizedExpression;
declare function isSwitchCase(node: Node | null | undefined, opts?: Opts<SwitchCase> | null): node is SwitchCase;
declare function isSwitchStatement(node: Node | null | undefined, opts?: Opts<SwitchStatement> | null): node is SwitchStatement;
declare function isThisExpression(node: Node | null | undefined, opts?: Opts<ThisExpression> | null): node is ThisExpression;
declare function isThrowStatement(node: Node | null | undefined, opts?: Opts<ThrowStatement> | null): node is ThrowStatement;
declare function isTryStatement(node: Node | null | undefined, opts?: Opts<TryStatement> | null): node is TryStatement;
declare function isUnaryExpression(node: Node | null | undefined, opts?: Opts<UnaryExpression> | null): node is UnaryExpression;
declare function isUpdateExpression(node: Node | null | undefined, opts?: Opts<UpdateExpression> | null): node is UpdateExpression;
declare function isVariableDeclaration(node: Node | null | undefined, opts?: Opts<VariableDeclaration> | null): node is VariableDeclaration;
declare function isVariableDeclarator(node: Node | null | undefined, opts?: Opts<VariableDeclarator> | null): node is VariableDeclarator;
declare function isWhileStatement(node: Node | null | undefined, opts?: Opts<WhileStatement> | null): node is WhileStatement;
declare function isWithStatement(node: Node | null | undefined, opts?: Opts<WithStatement> | null): node is WithStatement;
declare function isAssignmentPattern(node: Node | null | undefined, opts?: Opts<AssignmentPattern> | null): node is AssignmentPattern;
declare function isArrayPattern(node: Node | null | undefined, opts?: Opts<ArrayPattern> | null): node is ArrayPattern;
declare function isArrowFunctionExpression(node: Node | null | undefined, opts?: Opts<ArrowFunctionExpression> | null): node is ArrowFunctionExpression;
declare function isClassBody(node: Node | null | undefined, opts?: Opts<ClassBody> | null): node is ClassBody;
declare function isClassExpression(node: Node | null | undefined, opts?: Opts<ClassExpression> | null): node is ClassExpression;
declare function isClassDeclaration(node: Node | null | undefined, opts?: Opts<ClassDeclaration> | null): node is ClassDeclaration;
declare function isExportAllDeclaration(node: Node | null | undefined, opts?: Opts<ExportAllDeclaration> | null): node is ExportAllDeclaration;
declare function isExportDefaultDeclaration(node: Node | null | undefined, opts?: Opts<ExportDefaultDeclaration> | null): node is ExportDefaultDeclaration;
declare function isExportNamedDeclaration(node: Node | null | undefined, opts?: Opts<ExportNamedDeclaration> | null): node is ExportNamedDeclaration;
declare function isExportSpecifier(node: Node | null | undefined, opts?: Opts<ExportSpecifier> | null): node is ExportSpecifier;
declare function isForOfStatement(node: Node | null | undefined, opts?: Opts<ForOfStatement> | null): node is ForOfStatement;
declare function isImportDeclaration(node: Node | null | undefined, opts?: Opts<ImportDeclaration> | null): node is ImportDeclaration;
declare function isImportDefaultSpecifier(node: Node | null | undefined, opts?: Opts<ImportDefaultSpecifier> | null): node is ImportDefaultSpecifier;
declare function isImportNamespaceSpecifier(node: Node | null | undefined, opts?: Opts<ImportNamespaceSpecifier> | null): node is ImportNamespaceSpecifier;
declare function isImportSpecifier(node: Node | null | undefined, opts?: Opts<ImportSpecifier> | null): node is ImportSpecifier;
declare function isImportExpression(node: Node | null | undefined, opts?: Opts<ImportExpression> | null): node is ImportExpression;
declare function isMetaProperty(node: Node | null | undefined, opts?: Opts<MetaProperty> | null): node is MetaProperty;
declare function isClassMethod(node: Node | null | undefined, opts?: Opts<ClassMethod> | null): node is ClassMethod;
declare function isObjectPattern(node: Node | null | undefined, opts?: Opts<ObjectPattern> | null): node is ObjectPattern;
declare function isSpreadElement(node: Node | null | undefined, opts?: Opts<SpreadElement> | null): node is SpreadElement;
declare function isSuper(node: Node | null | undefined, opts?: Opts<Super> | null): node is Super;
declare function isTaggedTemplateExpression(node: Node | null | undefined, opts?: Opts<TaggedTemplateExpression> | null): node is TaggedTemplateExpression;
declare function isTemplateElement(node: Node | null | undefined, opts?: Opts<TemplateElement> | null): node is TemplateElement;
declare function isTemplateLiteral(node: Node | null | undefined, opts?: Opts<TemplateLiteral> | null): node is TemplateLiteral;
declare function isYieldExpression(node: Node | null | undefined, opts?: Opts<YieldExpression> | null): node is YieldExpression;
declare function isAwaitExpression(node: Node | null | undefined, opts?: Opts<AwaitExpression> | null): node is AwaitExpression;
declare function isImport(node: Node | null | undefined, opts?: Opts<Import> | null): node is Import;
declare function isBigIntLiteral(node: Node | null | undefined, opts?: Opts<BigIntLiteral> | null): node is BigIntLiteral;
declare function isExportNamespaceSpecifier(node: Node | null | undefined, opts?: Opts<ExportNamespaceSpecifier> | null): node is ExportNamespaceSpecifier;
declare function isOptionalMemberExpression(node: Node | null | undefined, opts?: Opts<OptionalMemberExpression> | null): node is OptionalMemberExpression;
declare function isOptionalCallExpression(node: Node | null | undefined, opts?: Opts<OptionalCallExpression> | null): node is OptionalCallExpression;
declare function isClassProperty(node: Node | null | undefined, opts?: Opts<ClassProperty> | null): node is ClassProperty;
declare function isClassAccessorProperty(node: Node | null | undefined, opts?: Opts<ClassAccessorProperty> | null): node is ClassAccessorProperty;
declare function isClassPrivateProperty(node: Node | null | undefined, opts?: Opts<ClassPrivateProperty> | null): node is ClassPrivateProperty;
declare function isClassPrivateMethod(node: Node | null | undefined, opts?: Opts<ClassPrivateMethod> | null): node is ClassPrivateMethod;
declare function isPrivateName(node: Node | null | undefined, opts?: Opts<PrivateName> | null): node is PrivateName;
declare function isStaticBlock(node: Node | null | undefined, opts?: Opts<StaticBlock> | null): node is StaticBlock;
declare function isAnyTypeAnnotation(node: Node | null | undefined, opts?: Opts<AnyTypeAnnotation> | null): node is AnyTypeAnnotation;
declare function isArrayTypeAnnotation(node: Node | null | undefined, opts?: Opts<ArrayTypeAnnotation> | null): node is ArrayTypeAnnotation;
declare function isBooleanTypeAnnotation(node: Node | null | undefined, opts?: Opts<BooleanTypeAnnotation> | null): node is BooleanTypeAnnotation;
declare function isBooleanLiteralTypeAnnotation(node: Node | null | undefined, opts?: Opts<BooleanLiteralTypeAnnotation> | null): node is BooleanLiteralTypeAnnotation;
declare function isNullLiteralTypeAnnotation(node: Node | null | undefined, opts?: Opts<NullLiteralTypeAnnotation> | null): node is NullLiteralTypeAnnotation;
declare function isClassImplements(node: Node | null | undefined, opts?: Opts<ClassImplements> | null): node is ClassImplements;
declare function isDeclareClass(node: Node | null | undefined, opts?: Opts<DeclareClass> | null): node is DeclareClass;
declare function isDeclareFunction(node: Node | null | undefined, opts?: Opts<DeclareFunction> | null): node is DeclareFunction;
declare function isDeclareInterface(node: Node | null | undefined, opts?: Opts<DeclareInterface> | null): node is DeclareInterface;
declare function isDeclareModule(node: Node | null | undefined, opts?: Opts<DeclareModule> | null): node is DeclareModule;
declare function isDeclareModuleExports(node: Node | null | undefined, opts?: Opts<DeclareModuleExports> | null): node is DeclareModuleExports;
declare function isDeclareTypeAlias(node: Node | null | undefined, opts?: Opts<DeclareTypeAlias> | null): node is DeclareTypeAlias;
declare function isDeclareOpaqueType(node: Node | null | undefined, opts?: Opts<DeclareOpaqueType> | null): node is DeclareOpaqueType;
declare function isDeclareVariable(node: Node | null | undefined, opts?: Opts<DeclareVariable> | null): node is DeclareVariable;
declare function isDeclareExportDeclaration(node: Node | null | undefined, opts?: Opts<DeclareExportDeclaration> | null): node is DeclareExportDeclaration;
declare function isDeclareExportAllDeclaration(node: Node | null | undefined, opts?: Opts<DeclareExportAllDeclaration> | null): node is DeclareExportAllDeclaration;
declare function isDeclaredPredicate(node: Node | null | undefined, opts?: Opts<DeclaredPredicate> | null): node is DeclaredPredicate;
declare function isExistsTypeAnnotation(node: Node | null | undefined, opts?: Opts<ExistsTypeAnnotation> | null): node is ExistsTypeAnnotation;
declare function isFunctionTypeAnnotation(node: Node | null | undefined, opts?: Opts<FunctionTypeAnnotation> | null): node is FunctionTypeAnnotation;
declare function isFunctionTypeParam(node: Node | null | undefined, opts?: Opts<FunctionTypeParam> | null): node is FunctionTypeParam;
declare function isGenericTypeAnnotation(node: Node | null | undefined, opts?: Opts<GenericTypeAnnotation> | null): node is GenericTypeAnnotation;
declare function isInferredPredicate(node: Node | null | undefined, opts?: Opts<InferredPredicate> | null): node is InferredPredicate;
declare function isInterfaceExtends(node: Node | null | undefined, opts?: Opts<InterfaceExtends> | null): node is InterfaceExtends;
declare function isInterfaceDeclaration(node: Node | null | undefined, opts?: Opts<InterfaceDeclaration> | null): node is InterfaceDeclaration;
declare function isInterfaceTypeAnnotation(node: Node | null | undefined, opts?: Opts<InterfaceTypeAnnotation> | null): node is InterfaceTypeAnnotation;
declare function isIntersectionTypeAnnotation(node: Node | null | undefined, opts?: Opts<IntersectionTypeAnnotation> | null): node is IntersectionTypeAnnotation;
declare function isMixedTypeAnnotation(node: Node | null | undefined, opts?: Opts<MixedTypeAnnotation> | null): node is MixedTypeAnnotation;
declare function isEmptyTypeAnnotation(node: Node | null | undefined, opts?: Opts<EmptyTypeAnnotation> | null): node is EmptyTypeAnnotation;
declare function isNullableTypeAnnotation(node: Node | null | undefined, opts?: Opts<NullableTypeAnnotation> | null): node is NullableTypeAnnotation;
declare function isNumberLiteralTypeAnnotation(node: Node | null | undefined, opts?: Opts<NumberLiteralTypeAnnotation> | null): node is NumberLiteralTypeAnnotation;
declare function isNumberTypeAnnotation(node: Node | null | undefined, opts?: Opts<NumberTypeAnnotation> | null): node is NumberTypeAnnotation;
declare function isObjectTypeAnnotation(node: Node | null | undefined, opts?: Opts<ObjectTypeAnnotation> | null): node is ObjectTypeAnnotation;
declare function isObjectTypeInternalSlot(node: Node | null | undefined, opts?: Opts<ObjectTypeInternalSlot> | null): node is ObjectTypeInternalSlot;
declare function isObjectTypeCallProperty(node: Node | null | undefined, opts?: Opts<ObjectTypeCallProperty> | null): node is ObjectTypeCallProperty;
declare function isObjectTypeIndexer(node: Node | null | undefined, opts?: Opts<ObjectTypeIndexer> | null): node is ObjectTypeIndexer;
declare function isObjectTypeProperty(node: Node | null | undefined, opts?: Opts<ObjectTypeProperty> | null): node is ObjectTypeProperty;
declare function isObjectTypeSpreadProperty(node: Node | null | undefined, opts?: Opts<ObjectTypeSpreadProperty> | null): node is ObjectTypeSpreadProperty;
declare function isOpaqueType(node: Node | null | undefined, opts?: Opts<OpaqueType> | null): node is OpaqueType;
declare function isQualifiedTypeIdentifier(node: Node | null | undefined, opts?: Opts<QualifiedTypeIdentifier> | null): node is QualifiedTypeIdentifier;
declare function isStringLiteralTypeAnnotation(node: Node | null | undefined, opts?: Opts<StringLiteralTypeAnnotation> | null): node is StringLiteralTypeAnnotation;
declare function isStringTypeAnnotation(node: Node | null | undefined, opts?: Opts<StringTypeAnnotation> | null): node is StringTypeAnnotation;
declare function isSymbolTypeAnnotation(node: Node | null | undefined, opts?: Opts<SymbolTypeAnnotation> | null): node is SymbolTypeAnnotation;
declare function isThisTypeAnnotation(node: Node | null | undefined, opts?: Opts<ThisTypeAnnotation> | null): node is ThisTypeAnnotation;
declare function isTupleTypeAnnotation(node: Node | null | undefined, opts?: Opts<TupleTypeAnnotation> | null): node is TupleTypeAnnotation;
declare function isTypeofTypeAnnotation(node: Node | null | undefined, opts?: Opts<TypeofTypeAnnotation> | null): node is TypeofTypeAnnotation;
declare function isTypeAlias(node: Node | null | undefined, opts?: Opts<TypeAlias> | null): node is TypeAlias;
declare function isTypeAnnotation(node: Node | null | undefined, opts?: Opts<TypeAnnotation> | null): node is TypeAnnotation;
declare function isTypeCastExpression(node: Node | null | undefined, opts?: Opts<TypeCastExpression> | null): node is TypeCastExpression;
declare function isTypeParameter(node: Node | null | undefined, opts?: Opts<TypeParameter> | null): node is TypeParameter;
declare function isTypeParameterDeclaration(node: Node | null | undefined, opts?: Opts<TypeParameterDeclaration> | null): node is TypeParameterDeclaration;
declare function isTypeParameterInstantiation(node: Node | null | undefined, opts?: Opts<TypeParameterInstantiation> | null): node is TypeParameterInstantiation;
declare function isUnionTypeAnnotation(node: Node | null | undefined, opts?: Opts<UnionTypeAnnotation> | null): node is UnionTypeAnnotation;
declare function isVariance(node: Node | null | undefined, opts?: Opts<Variance> | null): node is Variance;
declare function isVoidTypeAnnotation(node: Node | null | undefined, opts?: Opts<VoidTypeAnnotation> | null): node is VoidTypeAnnotation;
declare function isEnumDeclaration(node: Node | null | undefined, opts?: Opts<EnumDeclaration> | null): node is EnumDeclaration;
declare function isEnumBooleanBody(node: Node | null | undefined, opts?: Opts<EnumBooleanBody> | null): node is EnumBooleanBody;
declare function isEnumNumberBody(node: Node | null | undefined, opts?: Opts<EnumNumberBody> | null): node is EnumNumberBody;
declare function isEnumStringBody(node: Node | null | undefined, opts?: Opts<EnumStringBody> | null): node is EnumStringBody;
declare function isEnumSymbolBody(node: Node | null | undefined, opts?: Opts<EnumSymbolBody> | null): node is EnumSymbolBody;
declare function isEnumBooleanMember(node: Node | null | undefined, opts?: Opts<EnumBooleanMember> | null): node is EnumBooleanMember;
declare function isEnumNumberMember(node: Node | null | undefined, opts?: Opts<EnumNumberMember> | null): node is EnumNumberMember;
declare function isEnumStringMember(node: Node | null | undefined, opts?: Opts<EnumStringMember> | null): node is EnumStringMember;
declare function isEnumDefaultedMember(node: Node | null | undefined, opts?: Opts<EnumDefaultedMember> | null): node is EnumDefaultedMember;
declare function isIndexedAccessType(node: Node | null | undefined, opts?: Opts<IndexedAccessType> | null): node is IndexedAccessType;
declare function isOptionalIndexedAccessType(node: Node | null | undefined, opts?: Opts<OptionalIndexedAccessType> | null): node is OptionalIndexedAccessType;
declare function isJSXAttribute(node: Node | null | undefined, opts?: Opts<JSXAttribute> | null): node is JSXAttribute;
declare function isJSXClosingElement(node: Node | null | undefined, opts?: Opts<JSXClosingElement> | null): node is JSXClosingElement;
declare function isJSXElement(node: Node | null | undefined, opts?: Opts<JSXElement> | null): node is JSXElement;
declare function isJSXEmptyExpression(node: Node | null | undefined, opts?: Opts<JSXEmptyExpression> | null): node is JSXEmptyExpression;
declare function isJSXExpressionContainer(node: Node | null | undefined, opts?: Opts<JSXExpressionContainer> | null): node is JSXExpressionContainer;
declare function isJSXSpreadChild(node: Node | null | undefined, opts?: Opts<JSXSpreadChild> | null): node is JSXSpreadChild;
declare function isJSXIdentifier(node: Node | null | undefined, opts?: Opts<JSXIdentifier> | null): node is JSXIdentifier;
declare function isJSXMemberExpression(node: Node | null | undefined, opts?: Opts<JSXMemberExpression> | null): node is JSXMemberExpression;
declare function isJSXNamespacedName(node: Node | null | undefined, opts?: Opts<JSXNamespacedName> | null): node is JSXNamespacedName;
declare function isJSXOpeningElement(node: Node | null | undefined, opts?: Opts<JSXOpeningElement> | null): node is JSXOpeningElement;
declare function isJSXSpreadAttribute(node: Node | null | undefined, opts?: Opts<JSXSpreadAttribute> | null): node is JSXSpreadAttribute;
declare function isJSXText(node: Node | null | undefined, opts?: Opts<JSXText> | null): node is JSXText;
declare function isJSXFragment(node: Node | null | undefined, opts?: Opts<JSXFragment> | null): node is JSXFragment;
declare function isJSXOpeningFragment(node: Node | null | undefined, opts?: Opts<JSXOpeningFragment> | null): node is JSXOpeningFragment;
declare function isJSXClosingFragment(node: Node | null | undefined, opts?: Opts<JSXClosingFragment> | null): node is JSXClosingFragment;
declare function isNoop(node: Node | null | undefined, opts?: Opts<Noop> | null): node is Noop;
declare function isPlaceholder(node: Node | null | undefined, opts?: Opts<Placeholder> | null): node is Placeholder;
declare function isV8IntrinsicIdentifier(node: Node | null | undefined, opts?: Opts<V8IntrinsicIdentifier> | null): node is V8IntrinsicIdentifier;
declare function isArgumentPlaceholder(node: Node | null | undefined, opts?: Opts<ArgumentPlaceholder> | null): node is ArgumentPlaceholder;
declare function isBindExpression(node: Node | null | undefined, opts?: Opts<BindExpression> | null): node is BindExpression;
declare function isImportAttribute(node: Node | null | undefined, opts?: Opts<ImportAttribute> | null): node is ImportAttribute;
declare function isDecorator(node: Node | null | undefined, opts?: Opts<Decorator> | null): node is Decorator;
declare function isDoExpression(node: Node | null | undefined, opts?: Opts<DoExpression> | null): node is DoExpression;
declare function isExportDefaultSpecifier(node: Node | null | undefined, opts?: Opts<ExportDefaultSpecifier> | null): node is ExportDefaultSpecifier;
declare function isRecordExpression(node: Node | null | undefined, opts?: Opts<RecordExpression> | null): node is RecordExpression;
declare function isTupleExpression(node: Node | null | undefined, opts?: Opts<TupleExpression> | null): node is TupleExpression;
declare function isDecimalLiteral(node: Node | null | undefined, opts?: Opts<DecimalLiteral> | null): node is DecimalLiteral;
declare function isModuleExpression(node: Node | null | undefined, opts?: Opts<ModuleExpression> | null): node is ModuleExpression;
declare function isTopicReference(node: Node | null | undefined, opts?: Opts<TopicReference> | null): node is TopicReference;
declare function isPipelineTopicExpression(node: Node | null | undefined, opts?: Opts<PipelineTopicExpression> | null): node is PipelineTopicExpression;
declare function isPipelineBareFunction(node: Node | null | undefined, opts?: Opts<PipelineBareFunction> | null): node is PipelineBareFunction;
declare function isPipelinePrimaryTopicReference(node: Node | null | undefined, opts?: Opts<PipelinePrimaryTopicReference> | null): node is PipelinePrimaryTopicReference;
declare function isTSParameterProperty(node: Node | null | undefined, opts?: Opts<TSParameterProperty> | null): node is TSParameterProperty;
declare function isTSDeclareFunction(node: Node | null | undefined, opts?: Opts<TSDeclareFunction> | null): node is TSDeclareFunction;
declare function isTSDeclareMethod(node: Node | null | undefined, opts?: Opts<TSDeclareMethod> | null): node is TSDeclareMethod;
declare function isTSQualifiedName(node: Node | null | undefined, opts?: Opts<TSQualifiedName> | null): node is TSQualifiedName;
declare function isTSCallSignatureDeclaration(node: Node | null | undefined, opts?: Opts<TSCallSignatureDeclaration> | null): node is TSCallSignatureDeclaration;
declare function isTSConstructSignatureDeclaration(node: Node | null | undefined, opts?: Opts<TSConstructSignatureDeclaration> | null): node is TSConstructSignatureDeclaration;
declare function isTSPropertySignature(node: Node | null | undefined, opts?: Opts<TSPropertySignature> | null): node is TSPropertySignature;
declare function isTSMethodSignature(node: Node | null | undefined, opts?: Opts<TSMethodSignature> | null): node is TSMethodSignature;
declare function isTSIndexSignature(node: Node | null | undefined, opts?: Opts<TSIndexSignature> | null): node is TSIndexSignature;
declare function isTSAnyKeyword(node: Node | null | undefined, opts?: Opts<TSAnyKeyword> | null): node is TSAnyKeyword;
declare function isTSBooleanKeyword(node: Node | null | undefined, opts?: Opts<TSBooleanKeyword> | null): node is TSBooleanKeyword;
declare function isTSBigIntKeyword(node: Node | null | undefined, opts?: Opts<TSBigIntKeyword> | null): node is TSBigIntKeyword;
declare function isTSIntrinsicKeyword(node: Node | null | undefined, opts?: Opts<TSIntrinsicKeyword> | null): node is TSIntrinsicKeyword;
declare function isTSNeverKeyword(node: Node | null | undefined, opts?: Opts<TSNeverKeyword> | null): node is TSNeverKeyword;
declare function isTSNullKeyword(node: Node | null | undefined, opts?: Opts<TSNullKeyword> | null): node is TSNullKeyword;
declare function isTSNumberKeyword(node: Node | null | undefined, opts?: Opts<TSNumberKeyword> | null): node is TSNumberKeyword;
declare function isTSObjectKeyword(node: Node | null | undefined, opts?: Opts<TSObjectKeyword> | null): node is TSObjectKeyword;
declare function isTSStringKeyword(node: Node | null | undefined, opts?: Opts<TSStringKeyword> | null): node is TSStringKeyword;
declare function isTSSymbolKeyword(node: Node | null | undefined, opts?: Opts<TSSymbolKeyword> | null): node is TSSymbolKeyword;
declare function isTSUndefinedKeyword(node: Node | null | undefined, opts?: Opts<TSUndefinedKeyword> | null): node is TSUndefinedKeyword;
declare function isTSUnknownKeyword(node: Node | null | undefined, opts?: Opts<TSUnknownKeyword> | null): node is TSUnknownKeyword;
declare function isTSVoidKeyword(node: Node | null | undefined, opts?: Opts<TSVoidKeyword> | null): node is TSVoidKeyword;
declare function isTSThisType(node: Node | null | undefined, opts?: Opts<TSThisType> | null): node is TSThisType;
declare function isTSFunctionType(node: Node | null | undefined, opts?: Opts<TSFunctionType> | null): node is TSFunctionType;
declare function isTSConstructorType(node: Node | null | undefined, opts?: Opts<TSConstructorType> | null): node is TSConstructorType;
declare function isTSTypeReference(node: Node | null | undefined, opts?: Opts<TSTypeReference> | null): node is TSTypeReference;
declare function isTSTypePredicate(node: Node | null | undefined, opts?: Opts<TSTypePredicate> | null): node is TSTypePredicate;
declare function isTSTypeQuery(node: Node | null | undefined, opts?: Opts<TSTypeQuery> | null): node is TSTypeQuery;
declare function isTSTypeLiteral(node: Node | null | undefined, opts?: Opts<TSTypeLiteral> | null): node is TSTypeLiteral;
declare function isTSArrayType(node: Node | null | undefined, opts?: Opts<TSArrayType> | null): node is TSArrayType;
declare function isTSTupleType(node: Node | null | undefined, opts?: Opts<TSTupleType> | null): node is TSTupleType;
declare function isTSOptionalType(node: Node | null | undefined, opts?: Opts<TSOptionalType> | null): node is TSOptionalType;
declare function isTSRestType(node: Node | null | undefined, opts?: Opts<TSRestType> | null): node is TSRestType;
declare function isTSNamedTupleMember(node: Node | null | undefined, opts?: Opts<TSNamedTupleMember> | null): node is TSNamedTupleMember;
declare function isTSUnionType(node: Node | null | undefined, opts?: Opts<TSUnionType> | null): node is TSUnionType;
declare function isTSIntersectionType(node: Node | null | undefined, opts?: Opts<TSIntersectionType> | null): node is TSIntersectionType;
declare function isTSConditionalType(node: Node | null | undefined, opts?: Opts<TSConditionalType> | null): node is TSConditionalType;
declare function isTSInferType(node: Node | null | undefined, opts?: Opts<TSInferType> | null): node is TSInferType;
declare function isTSParenthesizedType(node: Node | null | undefined, opts?: Opts<TSParenthesizedType> | null): node is TSParenthesizedType;
declare function isTSTypeOperator(node: Node | null | undefined, opts?: Opts<TSTypeOperator> | null): node is TSTypeOperator;
declare function isTSIndexedAccessType(node: Node | null | undefined, opts?: Opts<TSIndexedAccessType> | null): node is TSIndexedAccessType;
declare function isTSMappedType(node: Node | null | undefined, opts?: Opts<TSMappedType> | null): node is TSMappedType;
declare function isTSLiteralType(node: Node | null | undefined, opts?: Opts<TSLiteralType> | null): node is TSLiteralType;
declare function isTSExpressionWithTypeArguments(node: Node | null | undefined, opts?: Opts<TSExpressionWithTypeArguments> | null): node is TSExpressionWithTypeArguments;
declare function isTSInterfaceDeclaration(node: Node | null | undefined, opts?: Opts<TSInterfaceDeclaration> | null): node is TSInterfaceDeclaration;
declare function isTSInterfaceBody(node: Node | null | undefined, opts?: Opts<TSInterfaceBody> | null): node is TSInterfaceBody;
declare function isTSTypeAliasDeclaration(node: Node | null | undefined, opts?: Opts<TSTypeAliasDeclaration> | null): node is TSTypeAliasDeclaration;
declare function isTSInstantiationExpression(node: Node | null | undefined, opts?: Opts<TSInstantiationExpression> | null): node is TSInstantiationExpression;
declare function isTSAsExpression(node: Node | null | undefined, opts?: Opts<TSAsExpression> | null): node is TSAsExpression;
declare function isTSSatisfiesExpression(node: Node | null | undefined, opts?: Opts<TSSatisfiesExpression> | null): node is TSSatisfiesExpression;
declare function isTSTypeAssertion(node: Node | null | undefined, opts?: Opts<TSTypeAssertion> | null): node is TSTypeAssertion;
declare function isTSEnumDeclaration(node: Node | null | undefined, opts?: Opts<TSEnumDeclaration> | null): node is TSEnumDeclaration;
declare function isTSEnumMember(node: Node | null | undefined, opts?: Opts<TSEnumMember> | null): node is TSEnumMember;
declare function isTSModuleDeclaration(node: Node | null | undefined, opts?: Opts<TSModuleDeclaration> | null): node is TSModuleDeclaration;
declare function isTSModuleBlock(node: Node | null | undefined, opts?: Opts<TSModuleBlock> | null): node is TSModuleBlock;
declare function isTSImportType(node: Node | null | undefined, opts?: Opts<TSImportType> | null): node is TSImportType;
declare function isTSImportEqualsDeclaration(node: Node | null | undefined, opts?: Opts<TSImportEqualsDeclaration> | null): node is TSImportEqualsDeclaration;
declare function isTSExternalModuleReference(node: Node | null | undefined, opts?: Opts<TSExternalModuleReference> | null): node is TSExternalModuleReference;
declare function isTSNonNullExpression(node: Node | null | undefined, opts?: Opts<TSNonNullExpression> | null): node is TSNonNullExpression;
declare function isTSExportAssignment(node: Node | null | undefined, opts?: Opts<TSExportAssignment> | null): node is TSExportAssignment;
declare function isTSNamespaceExportDeclaration(node: Node | null | undefined, opts?: Opts<TSNamespaceExportDeclaration> | null): node is TSNamespaceExportDeclaration;
declare function isTSTypeAnnotation(node: Node | null | undefined, opts?: Opts<TSTypeAnnotation> | null): node is TSTypeAnnotation;
declare function isTSTypeParameterInstantiation(node: Node | null | undefined, opts?: Opts<TSTypeParameterInstantiation> | null): node is TSTypeParameterInstantiation;
declare function isTSTypeParameterDeclaration(node: Node | null | undefined, opts?: Opts<TSTypeParameterDeclaration> | null): node is TSTypeParameterDeclaration;
declare function isTSTypeParameter(node: Node | null | undefined, opts?: Opts<TSTypeParameter> | null): node is TSTypeParameter;
declare function isStandardized(node: Node | null | undefined, opts?: Opts<Standardized> | null): node is Standardized;
declare function isExpression(node: Node | null | undefined, opts?: Opts<Expression> | null): node is Expression;
declare function isBinary(node: Node | null | undefined, opts?: Opts<Binary> | null): node is Binary;
declare function isScopable(node: Node | null | undefined, opts?: Opts<Scopable> | null): node is Scopable;
declare function isBlockParent(node: Node | null | undefined, opts?: Opts<BlockParent> | null): node is BlockParent;
declare function isBlock(node: Node | null | undefined, opts?: Opts<Block> | null): node is Block;
declare function isStatement(node: Node | null | undefined, opts?: Opts<Statement> | null): node is Statement;
declare function isTerminatorless(node: Node | null | undefined, opts?: Opts<Terminatorless> | null): node is Terminatorless;
declare function isCompletionStatement(node: Node | null | undefined, opts?: Opts<CompletionStatement> | null): node is CompletionStatement;
declare function isConditional(node: Node | null | undefined, opts?: Opts<Conditional> | null): node is Conditional;
declare function isLoop(node: Node | null | undefined, opts?: Opts<Loop> | null): node is Loop;
declare function isWhile(node: Node | null | undefined, opts?: Opts<While> | null): node is While;
declare function isExpressionWrapper(node: Node | null | undefined, opts?: Opts<ExpressionWrapper> | null): node is ExpressionWrapper;
declare function isFor(node: Node | null | undefined, opts?: Opts<For> | null): node is For;
declare function isForXStatement(node: Node | null | undefined, opts?: Opts<ForXStatement> | null): node is ForXStatement;
declare function isFunction(node: Node | null | undefined, opts?: Opts<Function> | null): node is Function;
declare function isFunctionParent(node: Node | null | undefined, opts?: Opts<FunctionParent> | null): node is FunctionParent;
declare function isPureish(node: Node | null | undefined, opts?: Opts<Pureish> | null): node is Pureish;
declare function isDeclaration(node: Node | null | undefined, opts?: Opts<Declaration> | null): node is Declaration;
declare function isPatternLike(node: Node | null | undefined, opts?: Opts<PatternLike> | null): node is PatternLike;
declare function isLVal(node: Node | null | undefined, opts?: Opts<LVal> | null): node is LVal;
declare function isTSEntityName(node: Node | null | undefined, opts?: Opts<TSEntityName> | null): node is TSEntityName;
declare function isLiteral(node: Node | null | undefined, opts?: Opts<Literal> | null): node is Literal;
declare function isUserWhitespacable(node: Node | null | undefined, opts?: Opts<UserWhitespacable> | null): node is UserWhitespacable;
declare function isMethod(node: Node | null | undefined, opts?: Opts<Method> | null): node is Method;
declare function isObjectMember(node: Node | null | undefined, opts?: Opts<ObjectMember> | null): node is ObjectMember;
declare function isProperty(node: Node | null | undefined, opts?: Opts<Property> | null): node is Property;
declare function isUnaryLike(node: Node | null | undefined, opts?: Opts<UnaryLike> | null): node is UnaryLike;
declare function isPattern(node: Node | null | undefined, opts?: Opts<Pattern> | null): node is Pattern;
declare function isClass(node: Node | null | undefined, opts?: Opts<Class> | null): node is Class;
declare function isImportOrExportDeclaration(node: Node | null | undefined, opts?: Opts<ImportOrExportDeclaration> | null): node is ImportOrExportDeclaration;
declare function isExportDeclaration(node: Node | null | undefined, opts?: Opts<ExportDeclaration> | null): node is ExportDeclaration;
declare function isModuleSpecifier(node: Node | null | undefined, opts?: Opts<ModuleSpecifier> | null): node is ModuleSpecifier;
declare function isAccessor(node: Node | null | undefined, opts?: Opts<Accessor> | null): node is Accessor;
declare function isPrivate(node: Node | null | undefined, opts?: Opts<Private> | null): node is Private;
declare function isFlow(node: Node | null | undefined, opts?: Opts<Flow> | null): node is Flow;
declare function isFlowType(node: Node | null | undefined, opts?: Opts<FlowType> | null): node is FlowType;
declare function isFlowBaseAnnotation(node: Node | null | undefined, opts?: Opts<FlowBaseAnnotation> | null): node is FlowBaseAnnotation;
declare function isFlowDeclaration(node: Node | null | undefined, opts?: Opts<FlowDeclaration> | null): node is FlowDeclaration;
declare function isFlowPredicate(node: Node | null | undefined, opts?: Opts<FlowPredicate> | null): node is FlowPredicate;
declare function isEnumBody(node: Node | null | undefined, opts?: Opts<EnumBody> | null): node is EnumBody;
declare function isEnumMember(node: Node | null | undefined, opts?: Opts<EnumMember> | null): node is EnumMember;
declare function isJSX(node: Node | null | undefined, opts?: Opts<JSX> | null): node is JSX;
declare function isMiscellaneous(node: Node | null | undefined, opts?: Opts<Miscellaneous> | null): node is Miscellaneous;
declare function isTypeScript(node: Node | null | undefined, opts?: Opts<TypeScript> | null): node is TypeScript;
declare function isTSTypeElement(node: Node | null | undefined, opts?: Opts<TSTypeElement> | null): node is TSTypeElement;
declare function isTSType(node: Node | null | undefined, opts?: Opts<TSType> | null): node is TSType;
declare function isTSBaseType(node: Node | null | undefined, opts?: Opts<TSBaseType> | null): node is TSBaseType;
/**
 * @deprecated Use `isNumericLiteral`
 */
declare function isNumberLiteral(node: Node | null | undefined, opts?: Opts<NumberLiteral> | null): boolean;
/**
 * @deprecated Use `isRegExpLiteral`
 */
declare function isRegexLiteral(node: Node | null | undefined, opts?: Opts<RegexLiteral> | null): boolean;
/**
 * @deprecated Use `isRestElement`
 */
declare function isRestProperty(node: Node | null | undefined, opts?: Opts<RestProperty> | null): boolean;
/**
 * @deprecated Use `isSpreadElement`
 */
declare function isSpreadProperty(node: Node | null | undefined, opts?: Opts<SpreadProperty> | null): boolean;
/**
 * @deprecated Use `isImportOrExportDeclaration`
 */
declare function isModuleDeclaration(node: Node | null | undefined, opts?: Opts<ModuleDeclaration> | null): node is ImportOrExportDeclaration;

interface BaseComment {
    value: string;
    start?: number;
    end?: number;
    loc?: SourceLocation;
    ignore?: boolean;
    type: "CommentBlock" | "CommentLine";
}
interface Position {
    line: number;
    column: number;
    index: number;
}
interface CommentBlock extends BaseComment {
    type: "CommentBlock";
}
interface CommentLine extends BaseComment {
    type: "CommentLine";
}
type Comment = CommentBlock | CommentLine;
interface SourceLocation {
    start: Position;
    end: Position;
    filename: string;
    identifierName: string | undefined | null;
}
interface BaseNode {
    type: Node["type"];
    leadingComments?: Comment[] | null;
    innerComments?: Comment[] | null;
    trailingComments?: Comment[] | null;
    start?: number | null;
    end?: number | null;
    loc?: SourceLocation | null;
    range?: [number, number];
    extra?: Record<string, unknown>;
}
type CommentTypeShorthand = "leading" | "inner" | "trailing";
type Node = AnyTypeAnnotation | ArgumentPlaceholder | ArrayExpression | ArrayPattern | ArrayTypeAnnotation | ArrowFunctionExpression | AssignmentExpression | AssignmentPattern | AwaitExpression | BigIntLiteral | BinaryExpression | BindExpression | BlockStatement | BooleanLiteral | BooleanLiteralTypeAnnotation | BooleanTypeAnnotation | BreakStatement | CallExpression | CatchClause | ClassAccessorProperty | ClassBody | ClassDeclaration | ClassExpression | ClassImplements | ClassMethod | ClassPrivateMethod | ClassPrivateProperty | ClassProperty | ConditionalExpression | ContinueStatement | DebuggerStatement | DecimalLiteral | DeclareClass | DeclareExportAllDeclaration | DeclareExportDeclaration | DeclareFunction | DeclareInterface | DeclareModule | DeclareModuleExports | DeclareOpaqueType | DeclareTypeAlias | DeclareVariable | DeclaredPredicate | Decorator | Directive | DirectiveLiteral | DoExpression | DoWhileStatement | EmptyStatement | EmptyTypeAnnotation | EnumBooleanBody | EnumBooleanMember | EnumDeclaration | EnumDefaultedMember | EnumNumberBody | EnumNumberMember | EnumStringBody | EnumStringMember | EnumSymbolBody | ExistsTypeAnnotation | ExportAllDeclaration | ExportDefaultDeclaration | ExportDefaultSpecifier | ExportNamedDeclaration | ExportNamespaceSpecifier | ExportSpecifier | ExpressionStatement | File | ForInStatement | ForOfStatement | ForStatement | FunctionDeclaration | FunctionExpression | FunctionTypeAnnotation | FunctionTypeParam | GenericTypeAnnotation | Identifier | IfStatement | Import | ImportAttribute | ImportDeclaration | ImportDefaultSpecifier | ImportExpression | ImportNamespaceSpecifier | ImportSpecifier | IndexedAccessType | InferredPredicate | InterfaceDeclaration | InterfaceExtends | InterfaceTypeAnnotation | InterpreterDirective | IntersectionTypeAnnotation | JSXAttribute | JSXClosingElement | JSXClosingFragment | JSXElement | JSXEmptyExpression | JSXExpressionContainer | JSXFragment | JSXIdentifier | JSXMemberExpression | JSXNamespacedName | JSXOpeningElement | JSXOpeningFragment | JSXSpreadAttribute | JSXSpreadChild | JSXText | LabeledStatement | LogicalExpression | MemberExpression | MetaProperty | MixedTypeAnnotation | ModuleExpression | NewExpression | Noop | NullLiteral | NullLiteralTypeAnnotation | NullableTypeAnnotation | NumberLiteral | NumberLiteralTypeAnnotation | NumberTypeAnnotation | NumericLiteral | ObjectExpression | ObjectMethod | ObjectPattern | ObjectProperty | ObjectTypeAnnotation | ObjectTypeCallProperty | ObjectTypeIndexer | ObjectTypeInternalSlot | ObjectTypeProperty | ObjectTypeSpreadProperty | OpaqueType | OptionalCallExpression | OptionalIndexedAccessType | OptionalMemberExpression | ParenthesizedExpression | PipelineBareFunction | PipelinePrimaryTopicReference | PipelineTopicExpression | Placeholder | PrivateName | Program | QualifiedTypeIdentifier | RecordExpression | RegExpLiteral | RegexLiteral | RestElement | RestProperty | ReturnStatement | SequenceExpression | SpreadElement | SpreadProperty | StaticBlock | StringLiteral | StringLiteralTypeAnnotation | StringTypeAnnotation | Super | SwitchCase | SwitchStatement | SymbolTypeAnnotation | TSAnyKeyword | TSArrayType | TSAsExpression | TSBigIntKeyword | TSBooleanKeyword | TSCallSignatureDeclaration | TSConditionalType | TSConstructSignatureDeclaration | TSConstructorType | TSDeclareFunction | TSDeclareMethod | TSEnumDeclaration | TSEnumMember | TSExportAssignment | TSExpressionWithTypeArguments | TSExternalModuleReference | TSFunctionType | TSImportEqualsDeclaration | TSImportType | TSIndexSignature | TSIndexedAccessType | TSInferType | TSInstantiationExpression | TSInterfaceBody | TSInterfaceDeclaration | TSIntersectionType | TSIntrinsicKeyword | TSLiteralType | TSMappedType | TSMethodSignature | TSModuleBlock | TSModuleDeclaration | TSNamedTupleMember | TSNamespaceExportDeclaration | TSNeverKeyword | TSNonNullExpression | TSNullKeyword | TSNumberKeyword | TSObjectKeyword | TSOptionalType | TSParameterProperty | TSParenthesizedType | TSPropertySignature | TSQualifiedName | TSRestType | TSSatisfiesExpression | TSStringKeyword | TSSymbolKeyword | TSThisType | TSTupleType | TSTypeAliasDeclaration | TSTypeAnnotation | TSTypeAssertion | TSTypeLiteral | TSTypeOperator | TSTypeParameter | TSTypeParameterDeclaration | TSTypeParameterInstantiation | TSTypePredicate | TSTypeQuery | TSTypeReference | TSUndefinedKeyword | TSUnionType | TSUnknownKeyword | TSVoidKeyword | TaggedTemplateExpression | TemplateElement | TemplateLiteral | ThisExpression | ThisTypeAnnotation | ThrowStatement | TopicReference | TryStatement | TupleExpression | TupleTypeAnnotation | TypeAlias | TypeAnnotation | TypeCastExpression | TypeParameter | TypeParameterDeclaration | TypeParameterInstantiation | TypeofTypeAnnotation | UnaryExpression | UnionTypeAnnotation | UpdateExpression | V8IntrinsicIdentifier | VariableDeclaration | VariableDeclarator | Variance | VoidTypeAnnotation | WhileStatement | WithStatement | YieldExpression;
interface ArrayExpression extends BaseNode {
    type: "ArrayExpression";
    elements: Array<null | Expression | SpreadElement>;
}
interface AssignmentExpression extends BaseNode {
    type: "AssignmentExpression";
    operator: string;
    left: LVal | OptionalMemberExpression;
    right: Expression;
}
interface BinaryExpression extends BaseNode {
    type: "BinaryExpression";
    operator: "+" | "-" | "/" | "%" | "*" | "**" | "&" | "|" | ">>" | ">>>" | "<<" | "^" | "==" | "===" | "!=" | "!==" | "in" | "instanceof" | ">" | "<" | ">=" | "<=" | "|>";
    left: Expression | PrivateName;
    right: Expression;
}
interface InterpreterDirective extends BaseNode {
    type: "InterpreterDirective";
    value: string;
}
interface Directive extends BaseNode {
    type: "Directive";
    value: DirectiveLiteral;
}
interface DirectiveLiteral extends BaseNode {
    type: "DirectiveLiteral";
    value: string;
}
interface BlockStatement extends BaseNode {
    type: "BlockStatement";
    body: Array<Statement>;
    directives: Array<Directive>;
}
interface BreakStatement extends BaseNode {
    type: "BreakStatement";
    label?: Identifier | null;
}
interface CallExpression extends BaseNode {
    type: "CallExpression";
    callee: Expression | Super | V8IntrinsicIdentifier;
    arguments: Array<Expression | SpreadElement | ArgumentPlaceholder>;
    optional?: true | false | null;
    typeArguments?: TypeParameterInstantiation | null;
    typeParameters?: TSTypeParameterInstantiation | null;
}
interface CatchClause extends BaseNode {
    type: "CatchClause";
    param?: Identifier | ArrayPattern | ObjectPattern | null;
    body: BlockStatement;
}
interface ConditionalExpression extends BaseNode {
    type: "ConditionalExpression";
    test: Expression;
    consequent: Expression;
    alternate: Expression;
}
interface ContinueStatement extends BaseNode {
    type: "ContinueStatement";
    label?: Identifier | null;
}
interface DebuggerStatement extends BaseNode {
    type: "DebuggerStatement";
}
interface DoWhileStatement extends BaseNode {
    type: "DoWhileStatement";
    test: Expression;
    body: Statement;
}
interface EmptyStatement extends BaseNode {
    type: "EmptyStatement";
}
interface ExpressionStatement extends BaseNode {
    type: "ExpressionStatement";
    expression: Expression;
}
interface File extends BaseNode {
    type: "File";
    program: Program;
    comments?: Array<CommentBlock | CommentLine> | null;
    tokens?: Array<any> | null;
}
interface ForInStatement extends BaseNode {
    type: "ForInStatement";
    left: VariableDeclaration | LVal;
    right: Expression;
    body: Statement;
}
interface ForStatement extends BaseNode {
    type: "ForStatement";
    init?: VariableDeclaration | Expression | null;
    test?: Expression | null;
    update?: Expression | null;
    body: Statement;
}
interface FunctionDeclaration extends BaseNode {
    type: "FunctionDeclaration";
    id?: Identifier | null;
    params: Array<Identifier | Pattern | RestElement>;
    body: BlockStatement;
    generator: boolean;
    async: boolean;
    declare?: boolean | null;
    predicate?: DeclaredPredicate | InferredPredicate | null;
    returnType?: TypeAnnotation | TSTypeAnnotation | Noop | null;
    typeParameters?: TypeParameterDeclaration | TSTypeParameterDeclaration | Noop | null;
}
interface FunctionExpression extends BaseNode {
    type: "FunctionExpression";
    id?: Identifier | null;
    params: Array<Identifier | Pattern | RestElement>;
    body: BlockStatement;
    generator: boolean;
    async: boolean;
    predicate?: DeclaredPredicate | InferredPredicate | null;
    returnType?: TypeAnnotation | TSTypeAnnotation | Noop | null;
    typeParameters?: TypeParameterDeclaration | TSTypeParameterDeclaration | Noop | null;
}
interface Identifier extends BaseNode {
    type: "Identifier";
    name: string;
    decorators?: Array<Decorator> | null;
    optional?: boolean | null;
    typeAnnotation?: TypeAnnotation | TSTypeAnnotation | Noop | null;
}
interface IfStatement extends BaseNode {
    type: "IfStatement";
    test: Expression;
    consequent: Statement;
    alternate?: Statement | null;
}
interface LabeledStatement extends BaseNode {
    type: "LabeledStatement";
    label: Identifier;
    body: Statement;
}
interface StringLiteral extends BaseNode {
    type: "StringLiteral";
    value: string;
}
interface NumericLiteral extends BaseNode {
    type: "NumericLiteral";
    value: number;
}
/**
 * @deprecated Use `NumericLiteral`
 */
interface NumberLiteral extends BaseNode {
    type: "NumberLiteral";
    value: number;
}
interface NullLiteral extends BaseNode {
    type: "NullLiteral";
}
interface BooleanLiteral extends BaseNode {
    type: "BooleanLiteral";
    value: boolean;
}
interface RegExpLiteral extends BaseNode {
    type: "RegExpLiteral";
    pattern: string;
    flags: string;
}
/**
 * @deprecated Use `RegExpLiteral`
 */
interface RegexLiteral extends BaseNode {
    type: "RegexLiteral";
    pattern: string;
    flags: string;
}
interface LogicalExpression extends BaseNode {
    type: "LogicalExpression";
    operator: "||" | "&&" | "??";
    left: Expression;
    right: Expression;
}
interface MemberExpression extends BaseNode {
    type: "MemberExpression";
    object: Expression | Super;
    property: Expression | Identifier | PrivateName;
    computed: boolean;
    optional?: true | false | null;
}
interface NewExpression extends BaseNode {
    type: "NewExpression";
    callee: Expression | Super | V8IntrinsicIdentifier;
    arguments: Array<Expression | SpreadElement | ArgumentPlaceholder>;
    optional?: true | false | null;
    typeArguments?: TypeParameterInstantiation | null;
    typeParameters?: TSTypeParameterInstantiation | null;
}
interface Program extends BaseNode {
    type: "Program";
    body: Array<Statement>;
    directives: Array<Directive>;
    sourceType: "script" | "module";
    interpreter?: InterpreterDirective | null;
}
interface ObjectExpression extends BaseNode {
    type: "ObjectExpression";
    properties: Array<ObjectMethod | ObjectProperty | SpreadElement>;
}
interface ObjectMethod extends BaseNode {
    type: "ObjectMethod";
    kind: "method" | "get" | "set";
    key: Expression | Identifier | StringLiteral | NumericLiteral | BigIntLiteral;
    params: Array<Identifier | Pattern | RestElement>;
    body: BlockStatement;
    computed: boolean;
    generator: boolean;
    async: boolean;
    decorators?: Array<Decorator> | null;
    returnType?: TypeAnnotation | TSTypeAnnotation | Noop | null;
    typeParameters?: TypeParameterDeclaration | TSTypeParameterDeclaration | Noop | null;
}
interface ObjectProperty extends BaseNode {
    type: "ObjectProperty";
    key: Expression | Identifier | StringLiteral | NumericLiteral | BigIntLiteral | DecimalLiteral | PrivateName;
    value: Expression | PatternLike;
    computed: boolean;
    shorthand: boolean;
    decorators?: Array<Decorator> | null;
}
interface RestElement extends BaseNode {
    type: "RestElement";
    argument: LVal;
    decorators?: Array<Decorator> | null;
    optional?: boolean | null;
    typeAnnotation?: TypeAnnotation | TSTypeAnnotation | Noop | null;
}
/**
 * @deprecated Use `RestElement`
 */
interface RestProperty extends BaseNode {
    type: "RestProperty";
    argument: LVal;
    decorators?: Array<Decorator> | null;
    optional?: boolean | null;
    typeAnnotation?: TypeAnnotation | TSTypeAnnotation | Noop | null;
}
interface ReturnStatement extends BaseNode {
    type: "ReturnStatement";
    argument?: Expression | null;
}
interface SequenceExpression extends BaseNode {
    type: "SequenceExpression";
    expressions: Array<Expression>;
}
interface ParenthesizedExpression extends BaseNode {
    type: "ParenthesizedExpression";
    expression: Expression;
}
interface SwitchCase extends BaseNode {
    type: "SwitchCase";
    test?: Expression | null;
    consequent: Array<Statement>;
}
interface SwitchStatement extends BaseNode {
    type: "SwitchStatement";
    discriminant: Expression;
    cases: Array<SwitchCase>;
}
interface ThisExpression extends BaseNode {
    type: "ThisExpression";
}
interface ThrowStatement extends BaseNode {
    type: "ThrowStatement";
    argument: Expression;
}
interface TryStatement extends BaseNode {
    type: "TryStatement";
    block: BlockStatement;
    handler?: CatchClause | null;
    finalizer?: BlockStatement | null;
}
interface UnaryExpression extends BaseNode {
    type: "UnaryExpression";
    operator: "void" | "throw" | "delete" | "!" | "+" | "-" | "~" | "typeof";
    argument: Expression;
    prefix: boolean;
}
interface UpdateExpression extends BaseNode {
    type: "UpdateExpression";
    operator: "++" | "--";
    argument: Expression;
    prefix: boolean;
}
interface VariableDeclaration extends BaseNode {
    type: "VariableDeclaration";
    kind: "var" | "let" | "const" | "using" | "await using";
    declarations: Array<VariableDeclarator>;
    declare?: boolean | null;
}
interface VariableDeclarator extends BaseNode {
    type: "VariableDeclarator";
    id: LVal;
    init?: Expression | null;
    definite?: boolean | null;
}
interface WhileStatement extends BaseNode {
    type: "WhileStatement";
    test: Expression;
    body: Statement;
}
interface WithStatement extends BaseNode {
    type: "WithStatement";
    object: Expression;
    body: Statement;
}
interface AssignmentPattern extends BaseNode {
    type: "AssignmentPattern";
    left: Identifier | ObjectPattern | ArrayPattern | MemberExpression | TSAsExpression | TSSatisfiesExpression | TSTypeAssertion | TSNonNullExpression;
    right: Expression;
    decorators?: Array<Decorator> | null;
    optional?: boolean | null;
    typeAnnotation?: TypeAnnotation | TSTypeAnnotation | Noop | null;
}
interface ArrayPattern extends BaseNode {
    type: "ArrayPattern";
    elements: Array<null | PatternLike | LVal>;
    decorators?: Array<Decorator> | null;
    optional?: boolean | null;
    typeAnnotation?: TypeAnnotation | TSTypeAnnotation | Noop | null;
}
interface ArrowFunctionExpression extends BaseNode {
    type: "ArrowFunctionExpression";
    params: Array<Identifier | Pattern | RestElement>;
    body: BlockStatement | Expression;
    async: boolean;
    expression: boolean;
    generator?: boolean;
    predicate?: DeclaredPredicate | InferredPredicate | null;
    returnType?: TypeAnnotation | TSTypeAnnotation | Noop | null;
    typeParameters?: TypeParameterDeclaration | TSTypeParameterDeclaration | Noop | null;
}
interface ClassBody extends BaseNode {
    type: "ClassBody";
    body: Array<ClassMethod | ClassPrivateMethod | ClassProperty | ClassPrivateProperty | ClassAccessorProperty | TSDeclareMethod | TSIndexSignature | StaticBlock>;
}
interface ClassExpression extends BaseNode {
    type: "ClassExpression";
    id?: Identifier | null;
    superClass?: Expression | null;
    body: ClassBody;
    decorators?: Array<Decorator> | null;
    implements?: Array<TSExpressionWithTypeArguments | ClassImplements> | null;
    mixins?: InterfaceExtends | null;
    superTypeParameters?: TypeParameterInstantiation | TSTypeParameterInstantiation | null;
    typeParameters?: TypeParameterDeclaration | TSTypeParameterDeclaration | Noop | null;
}
interface ClassDeclaration extends BaseNode {
    type: "ClassDeclaration";
    id?: Identifier | null;
    superClass?: Expression | null;
    body: ClassBody;
    decorators?: Array<Decorator> | null;
    abstract?: boolean | null;
    declare?: boolean | null;
    implements?: Array<TSExpressionWithTypeArguments | ClassImplements> | null;
    mixins?: InterfaceExtends | null;
    superTypeParameters?: TypeParameterInstantiation | TSTypeParameterInstantiation | null;
    typeParameters?: TypeParameterDeclaration | TSTypeParameterDeclaration | Noop | null;
}
interface ExportAllDeclaration extends BaseNode {
    type: "ExportAllDeclaration";
    source: StringLiteral;
    assertions?: Array<ImportAttribute> | null;
    attributes?: Array<ImportAttribute> | null;
    exportKind?: "type" | "value" | null;
}
interface ExportDefaultDeclaration extends BaseNode {
    type: "ExportDefaultDeclaration";
    declaration: TSDeclareFunction | FunctionDeclaration | ClassDeclaration | Expression;
    exportKind?: "value" | null;
}
interface ExportNamedDeclaration extends BaseNode {
    type: "ExportNamedDeclaration";
    declaration?: Declaration | null;
    specifiers: Array<ExportSpecifier | ExportDefaultSpecifier | ExportNamespaceSpecifier>;
    source?: StringLiteral | null;
    assertions?: Array<ImportAttribute> | null;
    attributes?: Array<ImportAttribute> | null;
    exportKind?: "type" | "value" | null;
}
interface ExportSpecifier extends BaseNode {
    type: "ExportSpecifier";
    local: Identifier;
    exported: Identifier | StringLiteral;
    exportKind?: "type" | "value" | null;
}
interface ForOfStatement extends BaseNode {
    type: "ForOfStatement";
    left: VariableDeclaration | LVal;
    right: Expression;
    body: Statement;
    await: boolean;
}
interface ImportDeclaration extends BaseNode {
    type: "ImportDeclaration";
    specifiers: Array<ImportSpecifier | ImportDefaultSpecifier | ImportNamespaceSpecifier>;
    source: StringLiteral;
    assertions?: Array<ImportAttribute> | null;
    attributes?: Array<ImportAttribute> | null;
    importKind?: "type" | "typeof" | "value" | null;
    module?: boolean | null;
    phase?: "source" | "defer" | null;
}
interface ImportDefaultSpecifier extends BaseNode {
    type: "ImportDefaultSpecifier";
    local: Identifier;
}
interface ImportNamespaceSpecifier extends BaseNode {
    type: "ImportNamespaceSpecifier";
    local: Identifier;
}
interface ImportSpecifier extends BaseNode {
    type: "ImportSpecifier";
    local: Identifier;
    imported: Identifier | StringLiteral;
    importKind?: "type" | "typeof" | "value" | null;
}
interface ImportExpression extends BaseNode {
    type: "ImportExpression";
    source: Expression;
    options?: Expression | null;
    phase?: "source" | "defer" | null;
}
interface MetaProperty extends BaseNode {
    type: "MetaProperty";
    meta: Identifier;
    property: Identifier;
}
interface ClassMethod extends BaseNode {
    type: "ClassMethod";
    kind: "get" | "set" | "method" | "constructor";
    key: Identifier | StringLiteral | NumericLiteral | BigIntLiteral | Expression;
    params: Array<Identifier | Pattern | RestElement | TSParameterProperty>;
    body: BlockStatement;
    computed: boolean;
    static: boolean;
    generator: boolean;
    async: boolean;
    abstract?: boolean | null;
    access?: "public" | "private" | "protected" | null;
    accessibility?: "public" | "private" | "protected" | null;
    decorators?: Array<Decorator> | null;
    optional?: boolean | null;
    override?: boolean;
    returnType?: TypeAnnotation | TSTypeAnnotation | Noop | null;
    typeParameters?: TypeParameterDeclaration | TSTypeParameterDeclaration | Noop | null;
}
interface ObjectPattern extends BaseNode {
    type: "ObjectPattern";
    properties: Array<RestElement | ObjectProperty>;
    decorators?: Array<Decorator> | null;
    optional?: boolean | null;
    typeAnnotation?: TypeAnnotation | TSTypeAnnotation | Noop | null;
}
interface SpreadElement extends BaseNode {
    type: "SpreadElement";
    argument: Expression;
}
/**
 * @deprecated Use `SpreadElement`
 */
interface SpreadProperty extends BaseNode {
    type: "SpreadProperty";
    argument: Expression;
}
interface Super extends BaseNode {
    type: "Super";
}
interface TaggedTemplateExpression extends BaseNode {
    type: "TaggedTemplateExpression";
    tag: Expression;
    quasi: TemplateLiteral;
    typeParameters?: TypeParameterInstantiation | TSTypeParameterInstantiation | null;
}
interface TemplateElement extends BaseNode {
    type: "TemplateElement";
    value: {
        raw: string;
        cooked?: string;
    };
    tail: boolean;
}
interface TemplateLiteral extends BaseNode {
    type: "TemplateLiteral";
    quasis: Array<TemplateElement>;
    expressions: Array<Expression | TSType>;
}
interface YieldExpression extends BaseNode {
    type: "YieldExpression";
    argument?: Expression | null;
    delegate: boolean;
}
interface AwaitExpression extends BaseNode {
    type: "AwaitExpression";
    argument: Expression;
}
interface Import extends BaseNode {
    type: "Import";
}
interface BigIntLiteral extends BaseNode {
    type: "BigIntLiteral";
    value: string;
}
interface ExportNamespaceSpecifier extends BaseNode {
    type: "ExportNamespaceSpecifier";
    exported: Identifier;
}
interface OptionalMemberExpression extends BaseNode {
    type: "OptionalMemberExpression";
    object: Expression;
    property: Expression | Identifier;
    computed: boolean;
    optional: boolean;
}
interface OptionalCallExpression extends BaseNode {
    type: "OptionalCallExpression";
    callee: Expression;
    arguments: Array<Expression | SpreadElement | ArgumentPlaceholder>;
    optional: boolean;
    typeArguments?: TypeParameterInstantiation | null;
    typeParameters?: TSTypeParameterInstantiation | null;
}
interface ClassProperty extends BaseNode {
    type: "ClassProperty";
    key: Identifier | StringLiteral | NumericLiteral | BigIntLiteral | Expression;
    value?: Expression | null;
    typeAnnotation?: TypeAnnotation | TSTypeAnnotation | Noop | null;
    decorators?: Array<Decorator> | null;
    computed: boolean;
    static: boolean;
    abstract?: boolean | null;
    accessibility?: "public" | "private" | "protected" | null;
    declare?: boolean | null;
    definite?: boolean | null;
    optional?: boolean | null;
    override?: boolean;
    readonly?: boolean | null;
    variance?: Variance | null;
}
interface ClassAccessorProperty extends BaseNode {
    type: "ClassAccessorProperty";
    key: Identifier | StringLiteral | NumericLiteral | BigIntLiteral | Expression | PrivateName;
    value?: Expression | null;
    typeAnnotation?: TypeAnnotation | TSTypeAnnotation | Noop | null;
    decorators?: Array<Decorator> | null;
    computed: boolean;
    static: boolean;
    abstract?: boolean | null;
    accessibility?: "public" | "private" | "protected" | null;
    declare?: boolean | null;
    definite?: boolean | null;
    optional?: boolean | null;
    override?: boolean;
    readonly?: boolean | null;
    variance?: Variance | null;
}
interface ClassPrivateProperty extends BaseNode {
    type: "ClassPrivateProperty";
    key: PrivateName;
    value?: Expression | null;
    decorators?: Array<Decorator> | null;
    static: boolean;
    definite?: boolean | null;
    readonly?: boolean | null;
    typeAnnotation?: TypeAnnotation | TSTypeAnnotation | Noop | null;
    variance?: Variance | null;
}
interface ClassPrivateMethod extends BaseNode {
    type: "ClassPrivateMethod";
    kind: "get" | "set" | "method";
    key: PrivateName;
    params: Array<Identifier | Pattern | RestElement | TSParameterProperty>;
    body: BlockStatement;
    static: boolean;
    abstract?: boolean | null;
    access?: "public" | "private" | "protected" | null;
    accessibility?: "public" | "private" | "protected" | null;
    async?: boolean;
    computed?: boolean;
    decorators?: Array<Decorator> | null;
    generator?: boolean;
    optional?: boolean | null;
    override?: boolean;
    returnType?: TypeAnnotation | TSTypeAnnotation | Noop | null;
    typeParameters?: TypeParameterDeclaration | TSTypeParameterDeclaration | Noop | null;
}
interface PrivateName extends BaseNode {
    type: "PrivateName";
    id: Identifier;
}
interface StaticBlock extends BaseNode {
    type: "StaticBlock";
    body: Array<Statement>;
}
interface AnyTypeAnnotation extends BaseNode {
    type: "AnyTypeAnnotation";
}
interface ArrayTypeAnnotation extends BaseNode {
    type: "ArrayTypeAnnotation";
    elementType: FlowType;
}
interface BooleanTypeAnnotation extends BaseNode {
    type: "BooleanTypeAnnotation";
}
interface BooleanLiteralTypeAnnotation extends BaseNode {
    type: "BooleanLiteralTypeAnnotation";
    value: boolean;
}
interface NullLiteralTypeAnnotation extends BaseNode {
    type: "NullLiteralTypeAnnotation";
}
interface ClassImplements extends BaseNode {
    type: "ClassImplements";
    id: Identifier;
    typeParameters?: TypeParameterInstantiation | null;
}
interface DeclareClass extends BaseNode {
    type: "DeclareClass";
    id: Identifier;
    typeParameters?: TypeParameterDeclaration | null;
    extends?: Array<InterfaceExtends> | null;
    body: ObjectTypeAnnotation;
    implements?: Array<ClassImplements> | null;
    mixins?: Array<InterfaceExtends> | null;
}
interface DeclareFunction extends BaseNode {
    type: "DeclareFunction";
    id: Identifier;
    predicate?: DeclaredPredicate | null;
}
interface DeclareInterface extends BaseNode {
    type: "DeclareInterface";
    id: Identifier;
    typeParameters?: TypeParameterDeclaration | null;
    extends?: Array<InterfaceExtends> | null;
    body: ObjectTypeAnnotation;
}
interface DeclareModule extends BaseNode {
    type: "DeclareModule";
    id: Identifier | StringLiteral;
    body: BlockStatement;
    kind?: "CommonJS" | "ES" | null;
}
interface DeclareModuleExports extends BaseNode {
    type: "DeclareModuleExports";
    typeAnnotation: TypeAnnotation;
}
interface DeclareTypeAlias extends BaseNode {
    type: "DeclareTypeAlias";
    id: Identifier;
    typeParameters?: TypeParameterDeclaration | null;
    right: FlowType;
}
interface DeclareOpaqueType extends BaseNode {
    type: "DeclareOpaqueType";
    id: Identifier;
    typeParameters?: TypeParameterDeclaration | null;
    supertype?: FlowType | null;
    impltype?: FlowType | null;
}
interface DeclareVariable extends BaseNode {
    type: "DeclareVariable";
    id: Identifier;
}
interface DeclareExportDeclaration extends BaseNode {
    type: "DeclareExportDeclaration";
    declaration?: Flow | null;
    specifiers?: Array<ExportSpecifier | ExportNamespaceSpecifier> | null;
    source?: StringLiteral | null;
    default?: boolean | null;
}
interface DeclareExportAllDeclaration extends BaseNode {
    type: "DeclareExportAllDeclaration";
    source: StringLiteral;
    exportKind?: "type" | "value" | null;
}
interface DeclaredPredicate extends BaseNode {
    type: "DeclaredPredicate";
    value: Flow;
}
interface ExistsTypeAnnotation extends BaseNode {
    type: "ExistsTypeAnnotation";
}
interface FunctionTypeAnnotation extends BaseNode {
    type: "FunctionTypeAnnotation";
    typeParameters?: TypeParameterDeclaration | null;
    params: Array<FunctionTypeParam>;
    rest?: FunctionTypeParam | null;
    returnType: FlowType;
    this?: FunctionTypeParam | null;
}
interface FunctionTypeParam extends BaseNode {
    type: "FunctionTypeParam";
    name?: Identifier | null;
    typeAnnotation: FlowType;
    optional?: boolean | null;
}
interface GenericTypeAnnotation extends BaseNode {
    type: "GenericTypeAnnotation";
    id: Identifier | QualifiedTypeIdentifier;
    typeParameters?: TypeParameterInstantiation | null;
}
interface InferredPredicate extends BaseNode {
    type: "InferredPredicate";
}
interface InterfaceExtends extends BaseNode {
    type: "InterfaceExtends";
    id: Identifier | QualifiedTypeIdentifier;
    typeParameters?: TypeParameterInstantiation | null;
}
interface InterfaceDeclaration extends BaseNode {
    type: "InterfaceDeclaration";
    id: Identifier;
    typeParameters?: TypeParameterDeclaration | null;
    extends?: Array<InterfaceExtends> | null;
    body: ObjectTypeAnnotation;
}
interface InterfaceTypeAnnotation extends BaseNode {
    type: "InterfaceTypeAnnotation";
    extends?: Array<InterfaceExtends> | null;
    body: ObjectTypeAnnotation;
}
interface IntersectionTypeAnnotation extends BaseNode {
    type: "IntersectionTypeAnnotation";
    types: Array<FlowType>;
}
interface MixedTypeAnnotation extends BaseNode {
    type: "MixedTypeAnnotation";
}
interface EmptyTypeAnnotation extends BaseNode {
    type: "EmptyTypeAnnotation";
}
interface NullableTypeAnnotation extends BaseNode {
    type: "NullableTypeAnnotation";
    typeAnnotation: FlowType;
}
interface NumberLiteralTypeAnnotation extends BaseNode {
    type: "NumberLiteralTypeAnnotation";
    value: number;
}
interface NumberTypeAnnotation extends BaseNode {
    type: "NumberTypeAnnotation";
}
interface ObjectTypeAnnotation extends BaseNode {
    type: "ObjectTypeAnnotation";
    properties: Array<ObjectTypeProperty | ObjectTypeSpreadProperty>;
    indexers?: Array<ObjectTypeIndexer>;
    callProperties?: Array<ObjectTypeCallProperty>;
    internalSlots?: Array<ObjectTypeInternalSlot>;
    exact: boolean;
    inexact?: boolean | null;
}
interface ObjectTypeInternalSlot extends BaseNode {
    type: "ObjectTypeInternalSlot";
    id: Identifier;
    value: FlowType;
    optional: boolean;
    static: boolean;
    method: boolean;
}
interface ObjectTypeCallProperty extends BaseNode {
    type: "ObjectTypeCallProperty";
    value: FlowType;
    static: boolean;
}
interface ObjectTypeIndexer extends BaseNode {
    type: "ObjectTypeIndexer";
    id?: Identifier | null;
    key: FlowType;
    value: FlowType;
    variance?: Variance | null;
    static: boolean;
}
interface ObjectTypeProperty extends BaseNode {
    type: "ObjectTypeProperty";
    key: Identifier | StringLiteral;
    value: FlowType;
    variance?: Variance | null;
    kind: "init" | "get" | "set";
    method: boolean;
    optional: boolean;
    proto: boolean;
    static: boolean;
}
interface ObjectTypeSpreadProperty extends BaseNode {
    type: "ObjectTypeSpreadProperty";
    argument: FlowType;
}
interface OpaqueType extends BaseNode {
    type: "OpaqueType";
    id: Identifier;
    typeParameters?: TypeParameterDeclaration | null;
    supertype?: FlowType | null;
    impltype: FlowType;
}
interface QualifiedTypeIdentifier extends BaseNode {
    type: "QualifiedTypeIdentifier";
    id: Identifier;
    qualification: Identifier | QualifiedTypeIdentifier;
}
interface StringLiteralTypeAnnotation extends BaseNode {
    type: "StringLiteralTypeAnnotation";
    value: string;
}
interface StringTypeAnnotation extends BaseNode {
    type: "StringTypeAnnotation";
}
interface SymbolTypeAnnotation extends BaseNode {
    type: "SymbolTypeAnnotation";
}
interface ThisTypeAnnotation extends BaseNode {
    type: "ThisTypeAnnotation";
}
interface TupleTypeAnnotation extends BaseNode {
    type: "TupleTypeAnnotation";
    types: Array<FlowType>;
}
interface TypeofTypeAnnotation extends BaseNode {
    type: "TypeofTypeAnnotation";
    argument: FlowType;
}
interface TypeAlias extends BaseNode {
    type: "TypeAlias";
    id: Identifier;
    typeParameters?: TypeParameterDeclaration | null;
    right: FlowType;
}
interface TypeAnnotation extends BaseNode {
    type: "TypeAnnotation";
    typeAnnotation: FlowType;
}
interface TypeCastExpression extends BaseNode {
    type: "TypeCastExpression";
    expression: Expression;
    typeAnnotation: TypeAnnotation;
}
interface TypeParameter extends BaseNode {
    type: "TypeParameter";
    bound?: TypeAnnotation | null;
    default?: FlowType | null;
    variance?: Variance | null;
    name: string;
}
interface TypeParameterDeclaration extends BaseNode {
    type: "TypeParameterDeclaration";
    params: Array<TypeParameter>;
}
interface TypeParameterInstantiation extends BaseNode {
    type: "TypeParameterInstantiation";
    params: Array<FlowType>;
}
interface UnionTypeAnnotation extends BaseNode {
    type: "UnionTypeAnnotation";
    types: Array<FlowType>;
}
interface Variance extends BaseNode {
    type: "Variance";
    kind: "minus" | "plus";
}
interface VoidTypeAnnotation extends BaseNode {
    type: "VoidTypeAnnotation";
}
interface EnumDeclaration extends BaseNode {
    type: "EnumDeclaration";
    id: Identifier;
    body: EnumBooleanBody | EnumNumberBody | EnumStringBody | EnumSymbolBody;
}
interface EnumBooleanBody extends BaseNode {
    type: "EnumBooleanBody";
    members: Array<EnumBooleanMember>;
    explicitType: boolean;
    hasUnknownMembers: boolean;
}
interface EnumNumberBody extends BaseNode {
    type: "EnumNumberBody";
    members: Array<EnumNumberMember>;
    explicitType: boolean;
    hasUnknownMembers: boolean;
}
interface EnumStringBody extends BaseNode {
    type: "EnumStringBody";
    members: Array<EnumStringMember | EnumDefaultedMember>;
    explicitType: boolean;
    hasUnknownMembers: boolean;
}
interface EnumSymbolBody extends BaseNode {
    type: "EnumSymbolBody";
    members: Array<EnumDefaultedMember>;
    hasUnknownMembers: boolean;
}
interface EnumBooleanMember extends BaseNode {
    type: "EnumBooleanMember";
    id: Identifier;
    init: BooleanLiteral;
}
interface EnumNumberMember extends BaseNode {
    type: "EnumNumberMember";
    id: Identifier;
    init: NumericLiteral;
}
interface EnumStringMember extends BaseNode {
    type: "EnumStringMember";
    id: Identifier;
    init: StringLiteral;
}
interface EnumDefaultedMember extends BaseNode {
    type: "EnumDefaultedMember";
    id: Identifier;
}
interface IndexedAccessType extends BaseNode {
    type: "IndexedAccessType";
    objectType: FlowType;
    indexType: FlowType;
}
interface OptionalIndexedAccessType extends BaseNode {
    type: "OptionalIndexedAccessType";
    objectType: FlowType;
    indexType: FlowType;
    optional: boolean;
}
interface JSXAttribute extends BaseNode {
    type: "JSXAttribute";
    name: JSXIdentifier | JSXNamespacedName;
    value?: JSXElement | JSXFragment | StringLiteral | JSXExpressionContainer | null;
}
interface JSXClosingElement extends BaseNode {
    type: "JSXClosingElement";
    name: JSXIdentifier | JSXMemberExpression | JSXNamespacedName;
}
interface JSXElement extends BaseNode {
    type: "JSXElement";
    openingElement: JSXOpeningElement;
    closingElement?: JSXClosingElement | null;
    children: Array<JSXText | JSXExpressionContainer | JSXSpreadChild | JSXElement | JSXFragment>;
    selfClosing?: boolean | null;
}
interface JSXEmptyExpression extends BaseNode {
    type: "JSXEmptyExpression";
}
interface JSXExpressionContainer extends BaseNode {
    type: "JSXExpressionContainer";
    expression: Expression | JSXEmptyExpression;
}
interface JSXSpreadChild extends BaseNode {
    type: "JSXSpreadChild";
    expression: Expression;
}
interface JSXIdentifier extends BaseNode {
    type: "JSXIdentifier";
    name: string;
}
interface JSXMemberExpression extends BaseNode {
    type: "JSXMemberExpression";
    object: JSXMemberExpression | JSXIdentifier;
    property: JSXIdentifier;
}
interface JSXNamespacedName extends BaseNode {
    type: "JSXNamespacedName";
    namespace: JSXIdentifier;
    name: JSXIdentifier;
}
interface JSXOpeningElement extends BaseNode {
    type: "JSXOpeningElement";
    name: JSXIdentifier | JSXMemberExpression | JSXNamespacedName;
    attributes: Array<JSXAttribute | JSXSpreadAttribute>;
    selfClosing: boolean;
    typeParameters?: TypeParameterInstantiation | TSTypeParameterInstantiation | null;
}
interface JSXSpreadAttribute extends BaseNode {
    type: "JSXSpreadAttribute";
    argument: Expression;
}
interface JSXText extends BaseNode {
    type: "JSXText";
    value: string;
}
interface JSXFragment extends BaseNode {
    type: "JSXFragment";
    openingFragment: JSXOpeningFragment;
    closingFragment: JSXClosingFragment;
    children: Array<JSXText | JSXExpressionContainer | JSXSpreadChild | JSXElement | JSXFragment>;
}
interface JSXOpeningFragment extends BaseNode {
    type: "JSXOpeningFragment";
}
interface JSXClosingFragment extends BaseNode {
    type: "JSXClosingFragment";
}
interface Noop extends BaseNode {
    type: "Noop";
}
interface Placeholder extends BaseNode {
    type: "Placeholder";
    expectedNode: "Identifier" | "StringLiteral" | "Expression" | "Statement" | "Declaration" | "BlockStatement" | "ClassBody" | "Pattern";
    name: Identifier;
}
interface V8IntrinsicIdentifier extends BaseNode {
    type: "V8IntrinsicIdentifier";
    name: string;
}
interface ArgumentPlaceholder extends BaseNode {
    type: "ArgumentPlaceholder";
}
interface BindExpression extends BaseNode {
    type: "BindExpression";
    object: Expression;
    callee: Expression;
}
interface ImportAttribute extends BaseNode {
    type: "ImportAttribute";
    key: Identifier | StringLiteral;
    value: StringLiteral;
}
interface Decorator extends BaseNode {
    type: "Decorator";
    expression: Expression;
}
interface DoExpression extends BaseNode {
    type: "DoExpression";
    body: BlockStatement;
    async: boolean;
}
interface ExportDefaultSpecifier extends BaseNode {
    type: "ExportDefaultSpecifier";
    exported: Identifier;
}
interface RecordExpression extends BaseNode {
    type: "RecordExpression";
    properties: Array<ObjectProperty | SpreadElement>;
}
interface TupleExpression extends BaseNode {
    type: "TupleExpression";
    elements: Array<Expression | SpreadElement>;
}
interface DecimalLiteral extends BaseNode {
    type: "DecimalLiteral";
    value: string;
}
interface ModuleExpression extends BaseNode {
    type: "ModuleExpression";
    body: Program;
}
interface TopicReference extends BaseNode {
    type: "TopicReference";
}
interface PipelineTopicExpression extends BaseNode {
    type: "PipelineTopicExpression";
    expression: Expression;
}
interface PipelineBareFunction extends BaseNode {
    type: "PipelineBareFunction";
    callee: Expression;
}
interface PipelinePrimaryTopicReference extends BaseNode {
    type: "PipelinePrimaryTopicReference";
}
interface TSParameterProperty extends BaseNode {
    type: "TSParameterProperty";
    parameter: Identifier | AssignmentPattern;
    accessibility?: "public" | "private" | "protected" | null;
    decorators?: Array<Decorator> | null;
    override?: boolean | null;
    readonly?: boolean | null;
}
interface TSDeclareFunction extends BaseNode {
    type: "TSDeclareFunction";
    id?: Identifier | null;
    typeParameters?: TSTypeParameterDeclaration | Noop | null;
    params: Array<Identifier | Pattern | RestElement>;
    returnType?: TSTypeAnnotation | Noop | null;
    async?: boolean;
    declare?: boolean | null;
    generator?: boolean;
}
interface TSDeclareMethod extends BaseNode {
    type: "TSDeclareMethod";
    decorators?: Array<Decorator> | null;
    key: Identifier | StringLiteral | NumericLiteral | BigIntLiteral | Expression;
    typeParameters?: TSTypeParameterDeclaration | Noop | null;
    params: Array<Identifier | Pattern | RestElement | TSParameterProperty>;
    returnType?: TSTypeAnnotation | Noop | null;
    abstract?: boolean | null;
    access?: "public" | "private" | "protected" | null;
    accessibility?: "public" | "private" | "protected" | null;
    async?: boolean;
    computed?: boolean;
    generator?: boolean;
    kind?: "get" | "set" | "method" | "constructor";
    optional?: boolean | null;
    override?: boolean;
    static?: boolean;
}
interface TSQualifiedName extends BaseNode {
    type: "TSQualifiedName";
    left: TSEntityName;
    right: Identifier;
}
interface TSCallSignatureDeclaration extends BaseNode {
    type: "TSCallSignatureDeclaration";
    typeParameters?: TSTypeParameterDeclaration | null;
    parameters: Array<ArrayPattern | Identifier | ObjectPattern | RestElement>;
    typeAnnotation?: TSTypeAnnotation | null;
}
interface TSConstructSignatureDeclaration extends BaseNode {
    type: "TSConstructSignatureDeclaration";
    typeParameters?: TSTypeParameterDeclaration | null;
    parameters: Array<ArrayPattern | Identifier | ObjectPattern | RestElement>;
    typeAnnotation?: TSTypeAnnotation | null;
}
interface TSPropertySignature extends BaseNode {
    type: "TSPropertySignature";
    key: Expression;
    typeAnnotation?: TSTypeAnnotation | null;
    computed?: boolean;
    kind: "get" | "set";
    optional?: boolean | null;
    readonly?: boolean | null;
}
interface TSMethodSignature extends BaseNode {
    type: "TSMethodSignature";
    key: Expression;
    typeParameters?: TSTypeParameterDeclaration | null;
    parameters: Array<ArrayPattern | Identifier | ObjectPattern | RestElement>;
    typeAnnotation?: TSTypeAnnotation | null;
    computed?: boolean;
    kind: "method" | "get" | "set";
    optional?: boolean | null;
}
interface TSIndexSignature extends BaseNode {
    type: "TSIndexSignature";
    parameters: Array<Identifier>;
    typeAnnotation?: TSTypeAnnotation | null;
    readonly?: boolean | null;
    static?: boolean | null;
}
interface TSAnyKeyword extends BaseNode {
    type: "TSAnyKeyword";
}
interface TSBooleanKeyword extends BaseNode {
    type: "TSBooleanKeyword";
}
interface TSBigIntKeyword extends BaseNode {
    type: "TSBigIntKeyword";
}
interface TSIntrinsicKeyword extends BaseNode {
    type: "TSIntrinsicKeyword";
}
interface TSNeverKeyword extends BaseNode {
    type: "TSNeverKeyword";
}
interface TSNullKeyword extends BaseNode {
    type: "TSNullKeyword";
}
interface TSNumberKeyword extends BaseNode {
    type: "TSNumberKeyword";
}
interface TSObjectKeyword extends BaseNode {
    type: "TSObjectKeyword";
}
interface TSStringKeyword extends BaseNode {
    type: "TSStringKeyword";
}
interface TSSymbolKeyword extends BaseNode {
    type: "TSSymbolKeyword";
}
interface TSUndefinedKeyword extends BaseNode {
    type: "TSUndefinedKeyword";
}
interface TSUnknownKeyword extends BaseNode {
    type: "TSUnknownKeyword";
}
interface TSVoidKeyword extends BaseNode {
    type: "TSVoidKeyword";
}
interface TSThisType extends BaseNode {
    type: "TSThisType";
}
interface TSFunctionType extends BaseNode {
    type: "TSFunctionType";
    typeParameters?: TSTypeParameterDeclaration | null;
    parameters: Array<ArrayPattern | Identifier | ObjectPattern | RestElement>;
    typeAnnotation?: TSTypeAnnotation | null;
}
interface TSConstructorType extends BaseNode {
    type: "TSConstructorType";
    typeParameters?: TSTypeParameterDeclaration | null;
    parameters: Array<ArrayPattern | Identifier | ObjectPattern | RestElement>;
    typeAnnotation?: TSTypeAnnotation | null;
    abstract?: boolean | null;
}
interface TSTypeReference extends BaseNode {
    type: "TSTypeReference";
    typeName: TSEntityName;
    typeParameters?: TSTypeParameterInstantiation | null;
}
interface TSTypePredicate extends BaseNode {
    type: "TSTypePredicate";
    parameterName: Identifier | TSThisType;
    typeAnnotation?: TSTypeAnnotation | null;
    asserts?: boolean | null;
}
interface TSTypeQuery extends BaseNode {
    type: "TSTypeQuery";
    exprName: TSEntityName | TSImportType;
    typeParameters?: TSTypeParameterInstantiation | null;
}
interface TSTypeLiteral extends BaseNode {
    type: "TSTypeLiteral";
    members: Array<TSTypeElement>;
}
interface TSArrayType extends BaseNode {
    type: "TSArrayType";
    elementType: TSType;
}
interface TSTupleType extends BaseNode {
    type: "TSTupleType";
    elementTypes: Array<TSType | TSNamedTupleMember>;
}
interface TSOptionalType extends BaseNode {
    type: "TSOptionalType";
    typeAnnotation: TSType;
}
interface TSRestType extends BaseNode {
    type: "TSRestType";
    typeAnnotation: TSType;
}
interface TSNamedTupleMember extends BaseNode {
    type: "TSNamedTupleMember";
    label: Identifier;
    elementType: TSType;
    optional: boolean;
}
interface TSUnionType extends BaseNode {
    type: "TSUnionType";
    types: Array<TSType>;
}
interface TSIntersectionType extends BaseNode {
    type: "TSIntersectionType";
    types: Array<TSType>;
}
interface TSConditionalType extends BaseNode {
    type: "TSConditionalType";
    checkType: TSType;
    extendsType: TSType;
    trueType: TSType;
    falseType: TSType;
}
interface TSInferType extends BaseNode {
    type: "TSInferType";
    typeParameter: TSTypeParameter;
}
interface TSParenthesizedType extends BaseNode {
    type: "TSParenthesizedType";
    typeAnnotation: TSType;
}
interface TSTypeOperator extends BaseNode {
    type: "TSTypeOperator";
    typeAnnotation: TSType;
    operator: string;
}
interface TSIndexedAccessType extends BaseNode {
    type: "TSIndexedAccessType";
    objectType: TSType;
    indexType: TSType;
}
interface TSMappedType extends BaseNode {
    type: "TSMappedType";
    typeParameter: TSTypeParameter;
    typeAnnotation?: TSType | null;
    nameType?: TSType | null;
    optional?: true | false | "+" | "-" | null;
    readonly?: true | false | "+" | "-" | null;
}
interface TSLiteralType extends BaseNode {
    type: "TSLiteralType";
    literal: NumericLiteral | StringLiteral | BooleanLiteral | BigIntLiteral | TemplateLiteral | UnaryExpression;
}
interface TSExpressionWithTypeArguments extends BaseNode {
    type: "TSExpressionWithTypeArguments";
    expression: TSEntityName;
    typeParameters?: TSTypeParameterInstantiation | null;
}
interface TSInterfaceDeclaration extends BaseNode {
    type: "TSInterfaceDeclaration";
    id: Identifier;
    typeParameters?: TSTypeParameterDeclaration | null;
    extends?: Array<TSExpressionWithTypeArguments> | null;
    body: TSInterfaceBody;
    declare?: boolean | null;
}
interface TSInterfaceBody extends BaseNode {
    type: "TSInterfaceBody";
    body: Array<TSTypeElement>;
}
interface TSTypeAliasDeclaration extends BaseNode {
    type: "TSTypeAliasDeclaration";
    id: Identifier;
    typeParameters?: TSTypeParameterDeclaration | null;
    typeAnnotation: TSType;
    declare?: boolean | null;
}
interface TSInstantiationExpression extends BaseNode {
    type: "TSInstantiationExpression";
    expression: Expression;
    typeParameters?: TSTypeParameterInstantiation | null;
}
interface TSAsExpression extends BaseNode {
    type: "TSAsExpression";
    expression: Expression;
    typeAnnotation: TSType;
}
interface TSSatisfiesExpression extends BaseNode {
    type: "TSSatisfiesExpression";
    expression: Expression;
    typeAnnotation: TSType;
}
interface TSTypeAssertion extends BaseNode {
    type: "TSTypeAssertion";
    typeAnnotation: TSType;
    expression: Expression;
}
interface TSEnumDeclaration extends BaseNode {
    type: "TSEnumDeclaration";
    id: Identifier;
    members: Array<TSEnumMember>;
    const?: boolean | null;
    declare?: boolean | null;
    initializer?: Expression | null;
}
interface TSEnumMember extends BaseNode {
    type: "TSEnumMember";
    id: Identifier | StringLiteral;
    initializer?: Expression | null;
}
interface TSModuleDeclaration extends BaseNode {
    type: "TSModuleDeclaration";
    id: Identifier | StringLiteral;
    body: TSModuleBlock | TSModuleDeclaration;
    declare?: boolean | null;
    global?: boolean | null;
}
interface TSModuleBlock extends BaseNode {
    type: "TSModuleBlock";
    body: Array<Statement>;
}
interface TSImportType extends BaseNode {
    type: "TSImportType";
    argument: StringLiteral;
    qualifier?: TSEntityName | null;
    typeParameters?: TSTypeParameterInstantiation | null;
    options?: Expression | null;
}
interface TSImportEqualsDeclaration extends BaseNode {
    type: "TSImportEqualsDeclaration";
    id: Identifier;
    moduleReference: TSEntityName | TSExternalModuleReference;
    importKind?: "type" | "value" | null;
    isExport: boolean;
}
interface TSExternalModuleReference extends BaseNode {
    type: "TSExternalModuleReference";
    expression: StringLiteral;
}
interface TSNonNullExpression extends BaseNode {
    type: "TSNonNullExpression";
    expression: Expression;
}
interface TSExportAssignment extends BaseNode {
    type: "TSExportAssignment";
    expression: Expression;
}
interface TSNamespaceExportDeclaration extends BaseNode {
    type: "TSNamespaceExportDeclaration";
    id: Identifier;
}
interface TSTypeAnnotation extends BaseNode {
    type: "TSTypeAnnotation";
    typeAnnotation: TSType;
}
interface TSTypeParameterInstantiation extends BaseNode {
    type: "TSTypeParameterInstantiation";
    params: Array<TSType>;
}
interface TSTypeParameterDeclaration extends BaseNode {
    type: "TSTypeParameterDeclaration";
    params: Array<TSTypeParameter>;
}
interface TSTypeParameter extends BaseNode {
    type: "TSTypeParameter";
    constraint?: TSType | null;
    default?: TSType | null;
    name: string;
    const?: boolean | null;
    in?: boolean | null;
    out?: boolean | null;
}
type Standardized = ArrayExpression | AssignmentExpression | BinaryExpression | InterpreterDirective | Directive | DirectiveLiteral | BlockStatement | BreakStatement | CallExpression | CatchClause | ConditionalExpression | ContinueStatement | DebuggerStatement | DoWhileStatement | EmptyStatement | ExpressionStatement | File | ForInStatement | ForStatement | FunctionDeclaration | FunctionExpression | Identifier | IfStatement | LabeledStatement | StringLiteral | NumericLiteral | NullLiteral | BooleanLiteral | RegExpLiteral | LogicalExpression | MemberExpression | NewExpression | Program | ObjectExpression | ObjectMethod | ObjectProperty | RestElement | ReturnStatement | SequenceExpression | ParenthesizedExpression | SwitchCase | SwitchStatement | ThisExpression | ThrowStatement | TryStatement | UnaryExpression | UpdateExpression | VariableDeclaration | VariableDeclarator | WhileStatement | WithStatement | AssignmentPattern | ArrayPattern | ArrowFunctionExpression | ClassBody | ClassExpression | ClassDeclaration | ExportAllDeclaration | ExportDefaultDeclaration | ExportNamedDeclaration | ExportSpecifier | ForOfStatement | ImportDeclaration | ImportDefaultSpecifier | ImportNamespaceSpecifier | ImportSpecifier | ImportExpression | MetaProperty | ClassMethod | ObjectPattern | SpreadElement | Super | TaggedTemplateExpression | TemplateElement | TemplateLiteral | YieldExpression | AwaitExpression | Import | BigIntLiteral | ExportNamespaceSpecifier | OptionalMemberExpression | OptionalCallExpression | ClassProperty | ClassAccessorProperty | ClassPrivateProperty | ClassPrivateMethod | PrivateName | StaticBlock;
type Expression = ArrayExpression | AssignmentExpression | BinaryExpression | CallExpression | ConditionalExpression | FunctionExpression | Identifier | StringLiteral | NumericLiteral | NullLiteral | BooleanLiteral | RegExpLiteral | LogicalExpression | MemberExpression | NewExpression | ObjectExpression | SequenceExpression | ParenthesizedExpression | ThisExpression | UnaryExpression | UpdateExpression | ArrowFunctionExpression | ClassExpression | ImportExpression | MetaProperty | Super | TaggedTemplateExpression | TemplateLiteral | YieldExpression | AwaitExpression | Import | BigIntLiteral | OptionalMemberExpression | OptionalCallExpression | TypeCastExpression | JSXElement | JSXFragment | BindExpression | DoExpression | RecordExpression | TupleExpression | DecimalLiteral | ModuleExpression | TopicReference | PipelineTopicExpression | PipelineBareFunction | PipelinePrimaryTopicReference | TSInstantiationExpression | TSAsExpression | TSSatisfiesExpression | TSTypeAssertion | TSNonNullExpression;
type Binary = BinaryExpression | LogicalExpression;
type Scopable = BlockStatement | CatchClause | DoWhileStatement | ForInStatement | ForStatement | FunctionDeclaration | FunctionExpression | Program | ObjectMethod | SwitchStatement | WhileStatement | ArrowFunctionExpression | ClassExpression | ClassDeclaration | ForOfStatement | ClassMethod | ClassPrivateMethod | StaticBlock | TSModuleBlock;
type BlockParent = BlockStatement | CatchClause | DoWhileStatement | ForInStatement | ForStatement | FunctionDeclaration | FunctionExpression | Program | ObjectMethod | SwitchStatement | WhileStatement | ArrowFunctionExpression | ForOfStatement | ClassMethod | ClassPrivateMethod | StaticBlock | TSModuleBlock;
type Block = BlockStatement | Program | TSModuleBlock;
type Statement = BlockStatement | BreakStatement | ContinueStatement | DebuggerStatement | DoWhileStatement | EmptyStatement | ExpressionStatement | ForInStatement | ForStatement | FunctionDeclaration | IfStatement | LabeledStatement | ReturnStatement | SwitchStatement | ThrowStatement | TryStatement | VariableDeclaration | WhileStatement | WithStatement | ClassDeclaration | ExportAllDeclaration | ExportDefaultDeclaration | ExportNamedDeclaration | ForOfStatement | ImportDeclaration | DeclareClass | DeclareFunction | DeclareInterface | DeclareModule | DeclareModuleExports | DeclareTypeAlias | DeclareOpaqueType | DeclareVariable | DeclareExportDeclaration | DeclareExportAllDeclaration | InterfaceDeclaration | OpaqueType | TypeAlias | EnumDeclaration | TSDeclareFunction | TSInterfaceDeclaration | TSTypeAliasDeclaration | TSEnumDeclaration | TSModuleDeclaration | TSImportEqualsDeclaration | TSExportAssignment | TSNamespaceExportDeclaration;
type Terminatorless = BreakStatement | ContinueStatement | ReturnStatement | ThrowStatement | YieldExpression | AwaitExpression;
type CompletionStatement = BreakStatement | ContinueStatement | ReturnStatement | ThrowStatement;
type Conditional = ConditionalExpression | IfStatement;
type Loop = DoWhileStatement | ForInStatement | ForStatement | WhileStatement | ForOfStatement;
type While = DoWhileStatement | WhileStatement;
type ExpressionWrapper = ExpressionStatement | ParenthesizedExpression | TypeCastExpression;
type For = ForInStatement | ForStatement | ForOfStatement;
type ForXStatement = ForInStatement | ForOfStatement;
type Function = FunctionDeclaration | FunctionExpression | ObjectMethod | ArrowFunctionExpression | ClassMethod | ClassPrivateMethod;
type FunctionParent = FunctionDeclaration | FunctionExpression | ObjectMethod | ArrowFunctionExpression | ClassMethod | ClassPrivateMethod | StaticBlock | TSModuleBlock;
type Pureish = FunctionDeclaration | FunctionExpression | StringLiteral | NumericLiteral | NullLiteral | BooleanLiteral | RegExpLiteral | ArrowFunctionExpression | BigIntLiteral | DecimalLiteral;
type Declaration = FunctionDeclaration | VariableDeclaration | ClassDeclaration | ExportAllDeclaration | ExportDefaultDeclaration | ExportNamedDeclaration | ImportDeclaration | DeclareClass | DeclareFunction | DeclareInterface | DeclareModule | DeclareModuleExports | DeclareTypeAlias | DeclareOpaqueType | DeclareVariable | DeclareExportDeclaration | DeclareExportAllDeclaration | InterfaceDeclaration | OpaqueType | TypeAlias | EnumDeclaration | TSDeclareFunction | TSInterfaceDeclaration | TSTypeAliasDeclaration | TSEnumDeclaration | TSModuleDeclaration;
type PatternLike = Identifier | RestElement | AssignmentPattern | ArrayPattern | ObjectPattern | TSAsExpression | TSSatisfiesExpression | TSTypeAssertion | TSNonNullExpression;
type LVal = Identifier | MemberExpression | RestElement | AssignmentPattern | ArrayPattern | ObjectPattern | TSParameterProperty | TSAsExpression | TSSatisfiesExpression | TSTypeAssertion | TSNonNullExpression;
type TSEntityName = Identifier | TSQualifiedName;
type Literal = StringLiteral | NumericLiteral | NullLiteral | BooleanLiteral | RegExpLiteral | TemplateLiteral | BigIntLiteral | DecimalLiteral;
type Immutable = StringLiteral | NumericLiteral | NullLiteral | BooleanLiteral | BigIntLiteral | JSXAttribute | JSXClosingElement | JSXElement | JSXExpressionContainer | JSXSpreadChild | JSXOpeningElement | JSXText | JSXFragment | JSXOpeningFragment | JSXClosingFragment | DecimalLiteral;
type UserWhitespacable = ObjectMethod | ObjectProperty | ObjectTypeInternalSlot | ObjectTypeCallProperty | ObjectTypeIndexer | ObjectTypeProperty | ObjectTypeSpreadProperty;
type Method = ObjectMethod | ClassMethod | ClassPrivateMethod;
type ObjectMember = ObjectMethod | ObjectProperty;
type Property = ObjectProperty | ClassProperty | ClassAccessorProperty | ClassPrivateProperty;
type UnaryLike = UnaryExpression | SpreadElement;
type Pattern = AssignmentPattern | ArrayPattern | ObjectPattern;
type Class = ClassExpression | ClassDeclaration;
type ImportOrExportDeclaration = ExportAllDeclaration | ExportDefaultDeclaration | ExportNamedDeclaration | ImportDeclaration;
type ExportDeclaration = ExportAllDeclaration | ExportDefaultDeclaration | ExportNamedDeclaration;
type ModuleSpecifier = ExportSpecifier | ImportDefaultSpecifier | ImportNamespaceSpecifier | ImportSpecifier | ExportNamespaceSpecifier | ExportDefaultSpecifier;
type Accessor = ClassAccessorProperty;
type Private = ClassPrivateProperty | ClassPrivateMethod | PrivateName;
type Flow = AnyTypeAnnotation | ArrayTypeAnnotation | BooleanTypeAnnotation | BooleanLiteralTypeAnnotation | NullLiteralTypeAnnotation | ClassImplements | DeclareClass | DeclareFunction | DeclareInterface | DeclareModule | DeclareModuleExports | DeclareTypeAlias | DeclareOpaqueType | DeclareVariable | DeclareExportDeclaration | DeclareExportAllDeclaration | DeclaredPredicate | ExistsTypeAnnotation | FunctionTypeAnnotation | FunctionTypeParam | GenericTypeAnnotation | InferredPredicate | InterfaceExtends | InterfaceDeclaration | InterfaceTypeAnnotation | IntersectionTypeAnnotation | MixedTypeAnnotation | EmptyTypeAnnotation | NullableTypeAnnotation | NumberLiteralTypeAnnotation | NumberTypeAnnotation | ObjectTypeAnnotation | ObjectTypeInternalSlot | ObjectTypeCallProperty | ObjectTypeIndexer | ObjectTypeProperty | ObjectTypeSpreadProperty | OpaqueType | QualifiedTypeIdentifier | StringLiteralTypeAnnotation | StringTypeAnnotation | SymbolTypeAnnotation | ThisTypeAnnotation | TupleTypeAnnotation | TypeofTypeAnnotation | TypeAlias | TypeAnnotation | TypeCastExpression | TypeParameter | TypeParameterDeclaration | TypeParameterInstantiation | UnionTypeAnnotation | Variance | VoidTypeAnnotation | EnumDeclaration | EnumBooleanBody | EnumNumberBody | EnumStringBody | EnumSymbolBody | EnumBooleanMember | EnumNumberMember | EnumStringMember | EnumDefaultedMember | IndexedAccessType | OptionalIndexedAccessType;
type FlowType = AnyTypeAnnotation | ArrayTypeAnnotation | BooleanTypeAnnotation | BooleanLiteralTypeAnnotation | NullLiteralTypeAnnotation | ExistsTypeAnnotation | FunctionTypeAnnotation | GenericTypeAnnotation | InterfaceTypeAnnotation | IntersectionTypeAnnotation | MixedTypeAnnotation | EmptyTypeAnnotation | NullableTypeAnnotation | NumberLiteralTypeAnnotation | NumberTypeAnnotation | ObjectTypeAnnotation | StringLiteralTypeAnnotation | StringTypeAnnotation | SymbolTypeAnnotation | ThisTypeAnnotation | TupleTypeAnnotation | TypeofTypeAnnotation | UnionTypeAnnotation | VoidTypeAnnotation | IndexedAccessType | OptionalIndexedAccessType;
type FlowBaseAnnotation = AnyTypeAnnotation | BooleanTypeAnnotation | NullLiteralTypeAnnotation | MixedTypeAnnotation | EmptyTypeAnnotation | NumberTypeAnnotation | StringTypeAnnotation | SymbolTypeAnnotation | ThisTypeAnnotation | VoidTypeAnnotation;
type FlowDeclaration = DeclareClass | DeclareFunction | DeclareInterface | DeclareModule | DeclareModuleExports | DeclareTypeAlias | DeclareOpaqueType | DeclareVariable | DeclareExportDeclaration | DeclareExportAllDeclaration | InterfaceDeclaration | OpaqueType | TypeAlias;
type FlowPredicate = DeclaredPredicate | InferredPredicate;
type EnumBody = EnumBooleanBody | EnumNumberBody | EnumStringBody | EnumSymbolBody;
type EnumMember = EnumBooleanMember | EnumNumberMember | EnumStringMember | EnumDefaultedMember;
type JSX = JSXAttribute | JSXClosingElement | JSXElement | JSXEmptyExpression | JSXExpressionContainer | JSXSpreadChild | JSXIdentifier | JSXMemberExpression | JSXNamespacedName | JSXOpeningElement | JSXSpreadAttribute | JSXText | JSXFragment | JSXOpeningFragment | JSXClosingFragment;
type Miscellaneous = Noop | Placeholder | V8IntrinsicIdentifier;
type TypeScript = TSParameterProperty | TSDeclareFunction | TSDeclareMethod | TSQualifiedName | TSCallSignatureDeclaration | TSConstructSignatureDeclaration | TSPropertySignature | TSMethodSignature | TSIndexSignature | TSAnyKeyword | TSBooleanKeyword | TSBigIntKeyword | TSIntrinsicKeyword | TSNeverKeyword | TSNullKeyword | TSNumberKeyword | TSObjectKeyword | TSStringKeyword | TSSymbolKeyword | TSUndefinedKeyword | TSUnknownKeyword | TSVoidKeyword | TSThisType | TSFunctionType | TSConstructorType | TSTypeReference | TSTypePredicate | TSTypeQuery | TSTypeLiteral | TSArrayType | TSTupleType | TSOptionalType | TSRestType | TSNamedTupleMember | TSUnionType | TSIntersectionType | TSConditionalType | TSInferType | TSParenthesizedType | TSTypeOperator | TSIndexedAccessType | TSMappedType | TSLiteralType | TSExpressionWithTypeArguments | TSInterfaceDeclaration | TSInterfaceBody | TSTypeAliasDeclaration | TSInstantiationExpression | TSAsExpression | TSSatisfiesExpression | TSTypeAssertion | TSEnumDeclaration | TSEnumMember | TSModuleDeclaration | TSModuleBlock | TSImportType | TSImportEqualsDeclaration | TSExternalModuleReference | TSNonNullExpression | TSExportAssignment | TSNamespaceExportDeclaration | TSTypeAnnotation | TSTypeParameterInstantiation | TSTypeParameterDeclaration | TSTypeParameter;
type TSTypeElement = TSCallSignatureDeclaration | TSConstructSignatureDeclaration | TSPropertySignature | TSMethodSignature | TSIndexSignature;
type TSType = TSAnyKeyword | TSBooleanKeyword | TSBigIntKeyword | TSIntrinsicKeyword | TSNeverKeyword | TSNullKeyword | TSNumberKeyword | TSObjectKeyword | TSStringKeyword | TSSymbolKeyword | TSUndefinedKeyword | TSUnknownKeyword | TSVoidKeyword | TSThisType | TSFunctionType | TSConstructorType | TSTypeReference | TSTypePredicate | TSTypeQuery | TSTypeLiteral | TSArrayType | TSTupleType | TSOptionalType | TSRestType | TSUnionType | TSIntersectionType | TSConditionalType | TSInferType | TSParenthesizedType | TSTypeOperator | TSIndexedAccessType | TSMappedType | TSLiteralType | TSExpressionWithTypeArguments | TSImportType;
type TSBaseType = TSAnyKeyword | TSBooleanKeyword | TSBigIntKeyword | TSIntrinsicKeyword | TSNeverKeyword | TSNullKeyword | TSNumberKeyword | TSObjectKeyword | TSStringKeyword | TSSymbolKeyword | TSUndefinedKeyword | TSUnknownKeyword | TSVoidKeyword | TSThisType | TSLiteralType;
type ModuleDeclaration = ExportAllDeclaration | ExportDefaultDeclaration | ExportNamedDeclaration | ImportDeclaration;
interface Aliases {
    Standardized: Standardized;
    Expression: Expression;
    Binary: Binary;
    Scopable: Scopable;
    BlockParent: BlockParent;
    Block: Block;
    Statement: Statement;
    Terminatorless: Terminatorless;
    CompletionStatement: CompletionStatement;
    Conditional: Conditional;
    Loop: Loop;
    While: While;
    ExpressionWrapper: ExpressionWrapper;
    For: For;
    ForXStatement: ForXStatement;
    Function: Function;
    FunctionParent: FunctionParent;
    Pureish: Pureish;
    Declaration: Declaration;
    PatternLike: PatternLike;
    LVal: LVal;
    TSEntityName: TSEntityName;
    Literal: Literal;
    Immutable: Immutable;
    UserWhitespacable: UserWhitespacable;
    Method: Method;
    ObjectMember: ObjectMember;
    Property: Property;
    UnaryLike: UnaryLike;
    Pattern: Pattern;
    Class: Class;
    ImportOrExportDeclaration: ImportOrExportDeclaration;
    ExportDeclaration: ExportDeclaration;
    ModuleSpecifier: ModuleSpecifier;
    Accessor: Accessor;
    Private: Private;
    Flow: Flow;
    FlowType: FlowType;
    FlowBaseAnnotation: FlowBaseAnnotation;
    FlowDeclaration: FlowDeclaration;
    FlowPredicate: FlowPredicate;
    EnumBody: EnumBody;
    EnumMember: EnumMember;
    JSX: JSX;
    Miscellaneous: Miscellaneous;
    TypeScript: TypeScript;
    TSTypeElement: TSTypeElement;
    TSType: TSType;
    TSBaseType: TSBaseType;
    ModuleDeclaration: ModuleDeclaration;
}
type DeprecatedAliases = NumberLiteral | RegexLiteral | RestProperty | SpreadProperty;
interface ParentMaps {
    AnyTypeAnnotation: ArrayTypeAnnotation | DeclareExportDeclaration | DeclareOpaqueType | DeclareTypeAlias | DeclaredPredicate | FunctionTypeAnnotation | FunctionTypeParam | IndexedAccessType | IntersectionTypeAnnotation | NullableTypeAnnotation | ObjectTypeCallProperty | ObjectTypeIndexer | ObjectTypeInternalSlot | ObjectTypeProperty | ObjectTypeSpreadProperty | OpaqueType | OptionalIndexedAccessType | TupleTypeAnnotation | TypeAlias | TypeAnnotation | TypeParameter | TypeParameterInstantiation | TypeofTypeAnnotation | UnionTypeAnnotation;
    ArgumentPlaceholder: CallExpression | NewExpression | OptionalCallExpression;
    ArrayExpression: ArrayExpression | ArrowFunctionExpression | AssignmentExpression | AssignmentPattern | AwaitExpression | BinaryExpression | BindExpression | CallExpression | ClassAccessorProperty | ClassDeclaration | ClassExpression | ClassMethod | ClassPrivateProperty | ClassProperty | ConditionalExpression | Decorator | DoWhileStatement | ExportDefaultDeclaration | ExpressionStatement | ForInStatement | ForOfStatement | ForStatement | IfStatement | ImportExpression | JSXExpressionContainer | JSXSpreadAttribute | JSXSpreadChild | LogicalExpression | MemberExpression | NewExpression | ObjectMethod | ObjectProperty | OptionalCallExpression | OptionalMemberExpression | ParenthesizedExpression | PipelineBareFunction | PipelineTopicExpression | ReturnStatement | SequenceExpression | SpreadElement | SwitchCase | SwitchStatement | TSAsExpression | TSDeclareMethod | TSEnumDeclaration | TSEnumMember | TSExportAssignment | TSImportType | TSInstantiationExpression | TSMethodSignature | TSNonNullExpression | TSPropertySignature | TSSatisfiesExpression | TSTypeAssertion | TaggedTemplateExpression | TemplateLiteral | ThrowStatement | TupleExpression | TypeCastExpression | UnaryExpression | UpdateExpression | VariableDeclarator | WhileStatement | WithStatement | YieldExpression;
    ArrayPattern: ArrayPattern | ArrowFunctionExpression | AssignmentExpression | AssignmentPattern | CatchClause | ClassMethod | ClassPrivateMethod | ForInStatement | ForOfStatement | FunctionDeclaration | FunctionExpression | ObjectMethod | ObjectProperty | RestElement | TSCallSignatureDeclaration | TSConstructSignatureDeclaration | TSConstructorType | TSDeclareFunction | TSDeclareMethod | TSFunctionType | TSMethodSignature | VariableDeclarator;
    ArrayTypeAnnotation: ArrayTypeAnnotation | DeclareExportDeclaration | DeclareOpaqueType | DeclareTypeAlias | DeclaredPredicate | FunctionTypeAnnotation | FunctionTypeParam | IndexedAccessType | IntersectionTypeAnnotation | NullableTypeAnnotation | ObjectTypeCallProperty | ObjectTypeIndexer | ObjectTypeInternalSlot | ObjectTypeProperty | ObjectTypeSpreadProperty | OpaqueType | OptionalIndexedAccessType | TupleTypeAnnotation | TypeAlias | TypeAnnotation | TypeParameter | TypeParameterInstantiation | TypeofTypeAnnotation | UnionTypeAnnotation;
    ArrowFunctionExpression: ArrayExpression | ArrowFunctionExpression | AssignmentExpression | AssignmentPattern | AwaitExpression | BinaryExpression | BindExpression | CallExpression | ClassAccessorProperty | ClassDeclaration | ClassExpression | ClassMethod | ClassPrivateProperty | ClassProperty | ConditionalExpression | Decorator | DoWhileStatement | ExportDefaultDeclaration | ExpressionStatement | ForInStatement | ForOfStatement | ForStatement | IfStatement | ImportExpression | JSXExpressionContainer | JSXSpreadAttribute | JSXSpreadChild | LogicalExpression | MemberExpression | NewExpression | ObjectMethod | ObjectProperty | OptionalCallExpression | OptionalMemberExpression | ParenthesizedExpression | PipelineBareFunction | PipelineTopicExpression | ReturnStatement | SequenceExpression | SpreadElement | SwitchCase | SwitchStatement | TSAsExpression | TSDeclareMethod | TSEnumDeclaration | TSEnumMember | TSExportAssignment | TSImportType | TSInstantiationExpression | TSMethodSignature | TSNonNullExpression | TSPropertySignature | TSSatisfiesExpression | TSTypeAssertion | TaggedTemplateExpression | TemplateLiteral | ThrowStatement | TupleExpression | TypeCastExpression | UnaryExpression | UpdateExpression | VariableDeclarator | WhileStatement | WithStatement | YieldExpression;
    AssignmentExpression: ArrayExpression | ArrowFunctionExpression | AssignmentExpression | AssignmentPattern | AwaitExpression | BinaryExpression | BindExpression | CallExpression | ClassAccessorProperty | ClassDeclaration | ClassExpression | ClassMethod | ClassPrivateProperty | ClassProperty | ConditionalExpression | Decorator | DoWhileStatement | ExportDefaultDeclaration | ExpressionStatement | ForInStatement | ForOfStatement | ForStatement | IfStatement | ImportExpression | JSXExpressionContainer | JSXSpreadAttribute | JSXSpreadChild | LogicalExpression | MemberExpression | NewExpression | ObjectMethod | ObjectProperty | OptionalCallExpression | OptionalMemberExpression | ParenthesizedExpression | PipelineBareFunction | PipelineTopicExpression | ReturnStatement | SequenceExpression | SpreadElement | SwitchCase | SwitchStatement | TSAsExpression | TSDeclareMethod | TSEnumDeclaration | TSEnumMember | TSExportAssignment | TSImportType | TSInstantiationExpression | TSMethodSignature | TSNonNullExpression | TSPropertySignature | TSSatisfiesExpression | TSTypeAssertion | TaggedTemplateExpression | TemplateLiteral | ThrowStatement | TupleExpression | TypeCastExpression | UnaryExpression | UpdateExpression | VariableDeclarator | WhileStatement | WithStatement | YieldExpression;
    AssignmentPattern: ArrayPattern | ArrowFunctionExpression | AssignmentExpression | ClassMethod | ClassPrivateMethod | ForInStatement | ForOfStatement | FunctionDeclaration | FunctionExpression | ObjectMethod | ObjectProperty | RestElement | TSDeclareFunction | TSDeclareMethod | TSParameterProperty | VariableDeclarator;
    AwaitExpression: ArrayExpression | ArrowFunctionExpression | AssignmentExpression | AssignmentPattern | AwaitExpression | BinaryExpression | BindExpression | CallExpression | ClassAccessorProperty | ClassDeclaration | ClassExpression | ClassMethod | ClassPrivateProperty | ClassProperty | ConditionalExpression | Decorator | DoWhileStatement | ExportDefaultDeclaration | ExpressionStatement | ForInStatement | ForOfStatement | ForStatement | IfStatement | ImportExpression | JSXExpressionContainer | JSXSpreadAttribute | JSXSpreadChild | LogicalExpression | MemberExpression | NewExpression | ObjectMethod | ObjectProperty | OptionalCallExpression | OptionalMemberExpression | ParenthesizedExpression | PipelineBareFunction | PipelineTopicExpression | ReturnStatement | SequenceExpression | SpreadElement | SwitchCase | SwitchStatement | TSAsExpression | TSDeclareMethod | TSEnumDeclaration | TSEnumMember | TSExportAssignment | TSImportType | TSInstantiationExpression | TSMethodSignature | TSNonNullExpression | TSPropertySignature | TSSatisfiesExpression | TSTypeAssertion | TaggedTemplateExpression | TemplateLiteral | ThrowStatement | TupleExpression | TypeCastExpression | UnaryExpression | UpdateExpression | VariableDeclarator | WhileStatement | WithStatement | YieldExpression;
    BigIntLiteral: ArrayExpression | ArrowFunctionExpression | AssignmentExpression | AssignmentPattern | AwaitExpression | BinaryExpression | BindExpression | CallExpression | ClassAccessorProperty | ClassDeclaration | ClassExpression | ClassMethod | ClassPrivateProperty | ClassProperty | ConditionalExpression | Decorator | DoWhileStatement | ExportDefaultDeclaration | ExpressionStatement | ForInStatement | ForOfStatement | ForStatement | IfStatement | ImportExpression | JSXExpressionContainer | JSXSpreadAttribute | JSXSpreadChild | LogicalExpression | MemberExpression | NewExpression | ObjectMethod | ObjectProperty | OptionalCallExpression | OptionalMemberExpression | ParenthesizedExpression | PipelineBareFunction | PipelineTopicExpression | ReturnStatement | SequenceExpression | SpreadElement | SwitchCase | SwitchStatement | TSAsExpression | TSDeclareMethod | TSEnumDeclaration | TSEnumMember | TSExportAssignment | TSImportType | TSInstantiationExpression | TSLiteralType | TSMethodSignature | TSNonNullExpression | TSPropertySignature | TSSatisfiesExpression | TSTypeAssertion | TaggedTemplateExpression | TemplateLiteral | ThrowStatement | TupleExpression | TypeCastExpression | UnaryExpression | UpdateExpression | VariableDeclarator | WhileStatement | WithStatement | YieldExpression;
    BinaryExpression: ArrayExpression | ArrowFunctionExpression | AssignmentExpression | AssignmentPattern | AwaitExpression | BinaryExpression | BindExpression | CallExpression | ClassAccessorProperty | ClassDeclaration | ClassExpression | ClassMethod | ClassPrivateProperty | ClassProperty | ConditionalExpression | Decorator | DoWhileStatement | ExportDefaultDeclaration | ExpressionStatement | ForInStatement | ForOfStatement | ForStatement | IfStatement | ImportExpression | JSXExpressionContainer | JSXSpreadAttribute | JSXSpreadChild | LogicalExpression | MemberExpression | NewExpression | ObjectMethod | ObjectProperty | OptionalCallExpression | OptionalMemberExpression | ParenthesizedExpression | PipelineBareFunction | PipelineTopicExpression | ReturnStatement | SequenceExpression | SpreadElement | SwitchCase | SwitchStatement | TSAsExpression | TSDeclareMethod | TSEnumDeclaration | TSEnumMember | TSExportAssignment | TSImportType | TSInstantiationExpression | TSMethodSignature | TSNonNullExpression | TSPropertySignature | TSSatisfiesExpression | TSTypeAssertion | TaggedTemplateExpression | TemplateLiteral | ThrowStatement | TupleExpression | TypeCastExpression | UnaryExpression | UpdateExpression | VariableDeclarator | WhileStatement | WithStatement | YieldExpression;
    BindExpression: ArrayExpression | ArrowFunctionExpression | AssignmentExpression | AssignmentPattern | AwaitExpression | BinaryExpression | BindExpression | CallExpression | ClassAccessorProperty | ClassDeclaration | ClassExpression | ClassMethod | ClassPrivateProperty | ClassProperty | ConditionalExpression | Decorator | DoWhileStatement | ExportDefaultDeclaration | ExpressionStatement | ForInStatement | ForOfStatement | ForStatement | IfStatement | ImportExpression | JSXExpressionContainer | JSXSpreadAttribute | JSXSpreadChild | LogicalExpression | MemberExpression | NewExpression | ObjectMethod | ObjectProperty | OptionalCallExpression | OptionalMemberExpression | ParenthesizedExpression | PipelineBareFunction | PipelineTopicExpression | ReturnStatement | SequenceExpression | SpreadElement | SwitchCase | SwitchStatement | TSAsExpression | TSDeclareMethod | TSEnumDeclaration | TSEnumMember | TSExportAssignment | TSImportType | TSInstantiationExpression | TSMethodSignature | TSNonNullExpression | TSPropertySignature | TSSatisfiesExpression | TSTypeAssertion | TaggedTemplateExpression | TemplateLiteral | ThrowStatement | TupleExpression | TypeCastExpression | UnaryExpression | UpdateExpression | VariableDeclarator | WhileStatement | WithStatement | YieldExpression;
    BlockStatement: ArrowFunctionExpression | BlockStatement | CatchClause | ClassMethod | ClassPrivateMethod | DeclareModule | DoExpression | DoWhileStatement | ForInStatement | ForOfStatement | ForStatement | FunctionDeclaration | FunctionExpression | IfStatement | LabeledStatement | ObjectMethod | Program | StaticBlock | SwitchCase | TSModuleBlock | TryStatement | WhileStatement | WithStatement;
    BooleanLiteral: ArrayExpression | ArrowFunctionExpression | AssignmentExpression | AssignmentPattern | AwaitExpression | BinaryExpression | BindExpression | CallExpression | ClassAccessorProperty | ClassDeclaration | ClassExpression | ClassMethod | ClassPrivateProperty | ClassProperty | ConditionalExpression | Decorator | DoWhileStatement | EnumBooleanMember | ExportDefaultDeclaration | ExpressionStatement | ForInStatement | ForOfStatement | ForStatement | IfStatement | ImportExpression | JSXExpressionContainer | JSXSpreadAttribute | JSXSpreadChild | LogicalExpression | MemberExpression | NewExpression | ObjectMethod | ObjectProperty | OptionalCallExpression | OptionalMemberExpression | ParenthesizedExpression | PipelineBareFunction | PipelineTopicExpression | ReturnStatement | SequenceExpression | SpreadElement | SwitchCase | SwitchStatement | TSAsExpression | TSDeclareMethod | TSEnumDeclaration | TSEnumMember | TSExportAssignment | TSImportType | TSInstantiationExpression | TSLiteralType | TSMethodSignature | TSNonNullExpression | TSPropertySignature | TSSatisfiesExpression | TSTypeAssertion | TaggedTemplateExpression | TemplateLiteral | ThrowStatement | TupleExpression | TypeCastExpression | UnaryExpression | UpdateExpression | VariableDeclarator | WhileStatement | WithStatement | YieldExpression;
    BooleanLiteralTypeAnnotation: ArrayTypeAnnotation | DeclareExportDeclaration | DeclareOpaqueType | DeclareTypeAlias | DeclaredPredicate | FunctionTypeAnnotation | FunctionTypeParam | IndexedAccessType | IntersectionTypeAnnotation | NullableTypeAnnotation | ObjectTypeCallProperty | ObjectTypeIndexer | ObjectTypeInternalSlot | ObjectTypeProperty | ObjectTypeSpreadProperty | OpaqueType | OptionalIndexedAccessType | TupleTypeAnnotation | TypeAlias | TypeAnnotation | TypeParameter | TypeParameterInstantiation | TypeofTypeAnnotation | UnionTypeAnnotation;
    BooleanTypeAnnotation: ArrayTypeAnnotation | DeclareExportDeclaration | DeclareOpaqueType | DeclareTypeAlias | DeclaredPredicate | FunctionTypeAnnotation | FunctionTypeParam | IndexedAccessType | IntersectionTypeAnnotation | NullableTypeAnnotation | ObjectTypeCallProperty | ObjectTypeIndexer | ObjectTypeInternalSlot | ObjectTypeProperty | ObjectTypeSpreadProperty | OpaqueType | OptionalIndexedAccessType | TupleTypeAnnotation | TypeAlias | TypeAnnotation | TypeParameter | TypeParameterInstantiation | TypeofTypeAnnotation | UnionTypeAnnotation;
    BreakStatement: BlockStatement | DoWhileStatement | ForInStatement | ForOfStatement | ForStatement | IfStatement | LabeledStatement | Program | StaticBlock | SwitchCase | TSModuleBlock | WhileStatement | WithStatement;
    CallExpression: ArrayExpression | ArrowFunctionExpression | AssignmentExpression | AssignmentPattern | AwaitExpression | BinaryExpression | BindExpression | CallExpression | ClassAccessorProperty | ClassDeclaration | ClassExpression | ClassMethod | ClassPrivateProperty | ClassProperty | ConditionalExpression | Decorator | DoWhileStatement | ExportDefaultDeclaration | ExpressionStatement | ForInStatement | ForOfStatement | ForStatement | IfStatement | ImportExpression | JSXExpressionContainer | JSXSpreadAttribute | JSXSpreadChild | LogicalExpression | MemberExpression | NewExpression | ObjectMethod | ObjectProperty | OptionalCallExpression | OptionalMemberExpression | ParenthesizedExpression | PipelineBareFunction | PipelineTopicExpression | ReturnStatement | SequenceExpression | SpreadElement | SwitchCase | SwitchStatement | TSAsExpression | TSDeclareMethod | TSEnumDeclaration | TSEnumMember | TSExportAssignment | TSImportType | TSInstantiationExpression | TSMethodSignature | TSNonNullExpression | TSPropertySignature | TSSatisfiesExpression | TSTypeAssertion | TaggedTemplateExpression | TemplateLiteral | ThrowStatement | TupleExpression | TypeCastExpression | UnaryExpression | UpdateExpression | VariableDeclarator | WhileStatement | WithStatement | YieldExpression;
    CatchClause: TryStatement;
    ClassAccessorProperty: ClassBody;
    ClassBody: ClassDeclaration | ClassExpression;
    ClassDeclaration: BlockStatement | DoWhileStatement | ExportDefaultDeclaration | ExportNamedDeclaration | ForInStatement | ForOfStatement | ForStatement | IfStatement | LabeledStatement | Program | StaticBlock | SwitchCase | TSModuleBlock | WhileStatement | WithStatement;
    ClassExpression: ArrayExpression | ArrowFunctionExpression | AssignmentExpression | AssignmentPattern | AwaitExpression | BinaryExpression | BindExpression | CallExpression | ClassAccessorProperty | ClassDeclaration | ClassExpression | ClassMethod | ClassPrivateProperty | ClassProperty | ConditionalExpression | Decorator | DoWhileStatement | ExportDefaultDeclaration | ExpressionStatement | ForInStatement | ForOfStatement | ForStatement | IfStatement | ImportExpression | JSXExpressionContainer | JSXSpreadAttribute | JSXSpreadChild | LogicalExpression | MemberExpression | NewExpression | ObjectMethod | ObjectProperty | OptionalCallExpression | OptionalMemberExpression | ParenthesizedExpression | PipelineBareFunction | PipelineTopicExpression | ReturnStatement | SequenceExpression | SpreadElement | SwitchCase | SwitchStatement | TSAsExpression | TSDeclareMethod | TSEnumDeclaration | TSEnumMember | TSExportAssignment | TSImportType | TSInstantiationExpression | TSMethodSignature | TSNonNullExpression | TSPropertySignature | TSSatisfiesExpression | TSTypeAssertion | TaggedTemplateExpression | TemplateLiteral | ThrowStatement | TupleExpression | TypeCastExpression | UnaryExpression | UpdateExpression | VariableDeclarator | WhileStatement | WithStatement | YieldExpression;
    ClassImplements: ClassDeclaration | ClassExpression | DeclareClass | DeclareExportDeclaration | DeclaredPredicate;
    ClassMethod: ClassBody;
    ClassPrivateMethod: ClassBody;
    ClassPrivateProperty: ClassBody;
    ClassProperty: ClassBody;
    CommentBlock: File;
    CommentLine: File;
    ConditionalExpression: ArrayExpression | ArrowFunctionExpression | AssignmentExpression | AssignmentPattern | AwaitExpression | BinaryExpression | BindExpression | CallExpression | ClassAccessorProperty | ClassDeclaration | ClassExpression | ClassMethod | ClassPrivateProperty | ClassProperty | ConditionalExpression | Decorator | DoWhileStatement | ExportDefaultDeclaration | ExpressionStatement | ForInStatement | ForOfStatement | ForStatement | IfStatement | ImportExpression | JSXExpressionContainer | JSXSpreadAttribute | JSXSpreadChild | LogicalExpression | MemberExpression | NewExpression | ObjectMethod | ObjectProperty | OptionalCallExpression | OptionalMemberExpression | ParenthesizedExpression | PipelineBareFunction | PipelineTopicExpression | ReturnStatement | SequenceExpression | SpreadElement | SwitchCase | SwitchStatement | TSAsExpression | TSDeclareMethod | TSEnumDeclaration | TSEnumMember | TSExportAssignment | TSImportType | TSInstantiationExpression | TSMethodSignature | TSNonNullExpression | TSPropertySignature | TSSatisfiesExpression | TSTypeAssertion | TaggedTemplateExpression | TemplateLiteral | ThrowStatement | TupleExpression | TypeCastExpression | UnaryExpression | UpdateExpression | VariableDeclarator | WhileStatement | WithStatement | YieldExpression;
    ContinueStatement: BlockStatement | DoWhileStatement | ForInStatement | ForOfStatement | ForStatement | IfStatement | LabeledStatement | Program | StaticBlock | SwitchCase | TSModuleBlock | WhileStatement | WithStatement;
    DebuggerStatement: BlockStatement | DoWhileStatement | ForInStatement | ForOfStatement | ForStatement | IfStatement | LabeledStatement | Program | StaticBlock | SwitchCase | TSModuleBlock | WhileStatement | WithStatement;
    DecimalLiteral: ArrayExpression | ArrowFunctionExpression | AssignmentExpression | AssignmentPattern | AwaitExpression | BinaryExpression | BindExpression | CallExpression | ClassAccessorProperty | ClassDeclaration | ClassExpression | ClassMethod | ClassPrivateProperty | ClassProperty | ConditionalExpression | Decorator | DoWhileStatement | ExportDefaultDeclaration | ExpressionStatement | ForInStatement | ForOfStatement | ForStatement | IfStatement | ImportExpression | JSXExpressionContainer | JSXSpreadAttribute | JSXSpreadChild | LogicalExpression | MemberExpression | NewExpression | ObjectMethod | ObjectProperty | OptionalCallExpression | OptionalMemberExpression | ParenthesizedExpression | PipelineBareFunction | PipelineTopicExpression | ReturnStatement | SequenceExpression | SpreadElement | SwitchCase | SwitchStatement | TSAsExpression | TSDeclareMethod | TSEnumDeclaration | TSEnumMember | TSExportAssignment | TSImportType | TSInstantiationExpression | TSMethodSignature | TSNonNullExpression | TSPropertySignature | TSSatisfiesExpression | TSTypeAssertion | TaggedTemplateExpression | TemplateLiteral | ThrowStatement | TupleExpression | TypeCastExpression | UnaryExpression | UpdateExpression | VariableDeclarator | WhileStatement | WithStatement | YieldExpression;
    DeclareClass: BlockStatement | DeclareExportDeclaration | DeclaredPredicate | DoWhileStatement | ExportNamedDeclaration | ForInStatement | ForOfStatement | ForStatement | IfStatement | LabeledStatement | Program | StaticBlock | SwitchCase | TSModuleBlock | WhileStatement | WithStatement;
    DeclareExportAllDeclaration: BlockStatement | DeclareExportDeclaration | DeclaredPredicate | DoWhileStatement | ExportNamedDeclaration | ForInStatement | ForOfStatement | ForStatement | IfStatement | LabeledStatement | Program | StaticBlock | SwitchCase | TSModuleBlock | WhileStatement | WithStatement;
    DeclareExportDeclaration: BlockStatement | DeclareExportDeclaration | DeclaredPredicate | DoWhileStatement | ExportNamedDeclaration | ForInStatement | ForOfStatement | ForStatement | IfStatement | LabeledStatement | Program | StaticBlock | SwitchCase | TSModuleBlock | WhileStatement | WithStatement;
    DeclareFunction: BlockStatement | DeclareExportDeclaration | DeclaredPredicate | DoWhileStatement | ExportNamedDeclaration | ForInStatement | ForOfStatement | ForStatement | IfStatement | LabeledStatement | Program | StaticBlock | SwitchCase | TSModuleBlock | WhileStatement | WithStatement;
    DeclareInterface: BlockStatement | DeclareExportDeclaration | DeclaredPredicate | DoWhileStatement | ExportNamedDeclaration | ForInStatement | ForOfStatement | ForStatement | IfStatement | LabeledStatement | Program | StaticBlock | SwitchCase | TSModuleBlock | WhileStatement | WithStatement;
    DeclareModule: BlockStatement | DeclareExportDeclaration | DeclaredPredicate | DoWhileStatement | ExportNamedDeclaration | ForInStatement | ForOfStatement | ForStatement | IfStatement | LabeledStatement | Program | StaticBlock | SwitchCase | TSModuleBlock | WhileStatement | WithStatement;
    DeclareModuleExports: BlockStatement | DeclareExportDeclaration | DeclaredPredicate | DoWhileStatement | ExportNamedDeclaration | ForInStatement | ForOfStatement | ForStatement | IfStatement | LabeledStatement | Program | StaticBlock | SwitchCase | TSModuleBlock | WhileStatement | WithStatement;
    DeclareOpaqueType: BlockStatement | DeclareExportDeclaration | DeclaredPredicate | DoWhileStatement | ExportNamedDeclaration | ForInStatement | ForOfStatement | ForStatement | IfStatement | LabeledStatement | Program | StaticBlock | SwitchCase | TSModuleBlock | WhileStatement | WithStatement;
    DeclareTypeAlias: BlockStatement | DeclareExportDeclaration | DeclaredPredicate | DoWhileStatement | ExportNamedDeclaration | ForInStatement | ForOfStatement | ForStatement | IfStatement | LabeledStatement | Program | StaticBlock | SwitchCase | TSModuleBlock | WhileStatement | WithStatement;
    DeclareVariable: BlockStatement | DeclareExportDeclaration | DeclaredPredicate | DoWhileStatement | ExportNamedDeclaration | ForInStatement | ForOfStatement | ForStatement | IfStatement | LabeledStatement | Program | StaticBlock | SwitchCase | TSModuleBlock | WhileStatement | WithStatement;
    DeclaredPredicate: ArrowFunctionExpression | DeclareExportDeclaration | DeclareFunction | DeclaredPredicate | FunctionDeclaration | FunctionExpression;
    Decorator: ArrayPattern | AssignmentPattern | ClassAccessorProperty | ClassDeclaration | ClassExpression | ClassMethod | ClassPrivateMethod | ClassPrivateProperty | ClassProperty | Identifier | ObjectMethod | ObjectPattern | ObjectProperty | RestElement | TSDeclareMethod | TSParameterProperty;
    Directive: BlockStatement | Program;
    DirectiveLiteral: Directive;
    DoExpression: ArrayExpression | ArrowFunctionExpression | AssignmentExpression | AssignmentPattern | AwaitExpression | BinaryExpression | BindExpression | CallExpression | ClassAccessorProperty | ClassDeclaration | ClassExpression | ClassMethod | ClassPrivateProperty | ClassProperty | ConditionalExpression | Decorator | DoWhileStatement | ExportDefaultDeclaration | ExpressionStatement | ForInStatement | ForOfStatement | ForStatement | IfStatement | ImportExpression | JSXExpressionContainer | JSXSpreadAttribute | JSXSpreadChild | LogicalExpression | MemberExpression | NewExpression | ObjectMethod | ObjectProperty | OptionalCallExpression | OptionalMemberExpression | ParenthesizedExpression | PipelineBareFunction | PipelineTopicExpression | ReturnStatement | SequenceExpression | SpreadElement | SwitchCase | SwitchStatement | TSAsExpression | TSDeclareMethod | TSEnumDeclaration | TSEnumMember | TSExportAssignment | TSImportType | TSInstantiationExpression | TSMethodSignature | TSNonNullExpression | TSPropertySignature | TSSatisfiesExpression | TSTypeAssertion | TaggedTemplateExpression | TemplateLiteral | ThrowStatement | TupleExpression | TypeCastExpression | UnaryExpression | UpdateExpression | VariableDeclarator | WhileStatement | WithStatement | YieldExpression;
    DoWhileStatement: BlockStatement | DoWhileStatement | ForInStatement | ForOfStatement | ForStatement | IfStatement | LabeledStatement | Program | StaticBlock | SwitchCase | TSModuleBlock | WhileStatement | WithStatement;
    EmptyStatement: BlockStatement | DoWhileStatement | ForInStatement | ForOfStatement | ForStatement | IfStatement | LabeledStatement | Program | StaticBlock | SwitchCase | TSModuleBlock | WhileStatement | WithStatement;
    EmptyTypeAnnotation: ArrayTypeAnnotation | DeclareExportDeclaration | DeclareOpaqueType | DeclareTypeAlias | DeclaredPredicate | FunctionTypeAnnotation | FunctionTypeParam | IndexedAccessType | IntersectionTypeAnnotation | NullableTypeAnnotation | ObjectTypeCallProperty | ObjectTypeIndexer | ObjectTypeInternalSlot | ObjectTypeProperty | ObjectTypeSpreadProperty | OpaqueType | OptionalIndexedAccessType | TupleTypeAnnotation | TypeAlias | TypeAnnotation | TypeParameter | TypeParameterInstantiation | TypeofTypeAnnotation | UnionTypeAnnotation;
    EnumBooleanBody: DeclareExportDeclaration | DeclaredPredicate | EnumDeclaration;
    EnumBooleanMember: DeclareExportDeclaration | DeclaredPredicate | EnumBooleanBody;
    EnumDeclaration: BlockStatement | DeclareExportDeclaration | DeclaredPredicate | DoWhileStatement | ExportNamedDeclaration | ForInStatement | ForOfStatement | ForStatement | IfStatement | LabeledStatement | Program | StaticBlock | SwitchCase | TSModuleBlock | WhileStatement | WithStatement;
    EnumDefaultedMember: DeclareExportDeclaration | DeclaredPredicate | EnumStringBody | EnumSymbolBody;
    EnumNumberBody: DeclareExportDeclaration | DeclaredPredicate | EnumDeclaration;
    EnumNumberMember: DeclareExportDeclaration | DeclaredPredicate | EnumNumberBody;
    EnumStringBody: DeclareExportDeclaration | DeclaredPredicate | EnumDeclaration;
    EnumStringMember: DeclareExportDeclaration | DeclaredPredicate | EnumStringBody;
    EnumSymbolBody: DeclareExportDeclaration | DeclaredPredicate | EnumDeclaration;
    ExistsTypeAnnotation: ArrayTypeAnnotation | DeclareExportDeclaration | DeclareOpaqueType | DeclareTypeAlias | DeclaredPredicate | FunctionTypeAnnotation | FunctionTypeParam | IndexedAccessType | IntersectionTypeAnnotation | NullableTypeAnnotation | ObjectTypeCallProperty | ObjectTypeIndexer | ObjectTypeInternalSlot | ObjectTypeProperty | ObjectTypeSpreadProperty | OpaqueType | OptionalIndexedAccessType | TupleTypeAnnotation | TypeAlias | TypeAnnotation | TypeParameter | TypeParameterInstantiation | TypeofTypeAnnotation | UnionTypeAnnotation;
    ExportAllDeclaration: BlockStatement | DoWhileStatement | ExportNamedDeclaration | ForInStatement | ForOfStatement | ForStatement | IfStatement | LabeledStatement | Program | StaticBlock | SwitchCase | TSModuleBlock | WhileStatement | WithStatement;
    ExportDefaultDeclaration: BlockStatement | DoWhileStatement | ExportNamedDeclaration | ForInStatement | ForOfStatement | ForStatement | IfStatement | LabeledStatement | Program | StaticBlock | SwitchCase | TSModuleBlock | WhileStatement | WithStatement;
    ExportDefaultSpecifier: ExportNamedDeclaration;
    ExportNamedDeclaration: BlockStatement | DoWhileStatement | ExportNamedDeclaration | ForInStatement | ForOfStatement | ForStatement | IfStatement | LabeledStatement | Program | StaticBlock | SwitchCase | TSModuleBlock | WhileStatement | WithStatement;
    ExportNamespaceSpecifier: DeclareExportDeclaration | ExportNamedDeclaration;
    ExportSpecifier: DeclareExportDeclaration | ExportNamedDeclaration;
    ExpressionStatement: BlockStatement | DoWhileStatement | ForInStatement | ForOfStatement | ForStatement | IfStatement | LabeledStatement | Program | StaticBlock | SwitchCase | TSModuleBlock | WhileStatement | WithStatement;
    File: null;
    ForInStatement: BlockStatement | DoWhileStatement | ForInStatement | ForOfStatement | ForStatement | IfStatement | LabeledStatement | Program | StaticBlock | SwitchCase | TSModuleBlock | WhileStatement | WithStatement;
    ForOfStatement: BlockStatement | DoWhileStatement | ForInStatement | ForOfStatement | ForStatement | IfStatement | LabeledStatement | Program | StaticBlock | SwitchCase | TSModuleBlock | WhileStatement | WithStatement;
    ForStatement: BlockStatement | DoWhileStatement | ForInStatement | ForOfStatement | ForStatement | IfStatement | LabeledStatement | Program | StaticBlock | SwitchCase | TSModuleBlock | WhileStatement | WithStatement;
    FunctionDeclaration: BlockStatement | DoWhileStatement | ExportDefaultDeclaration | ExportNamedDeclaration | ForInStatement | ForOfStatement | ForStatement | IfStatement | LabeledStatement | Program | StaticBlock | SwitchCase | TSModuleBlock | WhileStatement | WithStatement;
    FunctionExpression: ArrayExpression | ArrowFunctionExpression | AssignmentExpression | AssignmentPattern | AwaitExpression | BinaryExpression | BindExpression | CallExpression | ClassAccessorProperty | ClassDeclaration | ClassExpression | ClassMethod | ClassPrivateProperty | ClassProperty | ConditionalExpression | Decorator | DoWhileStatement | ExportDefaultDeclaration | ExpressionStatement | ForInStatement | ForOfStatement | ForStatement | IfStatement | ImportExpression | JSXExpressionContainer | JSXSpreadAttribute | JSXSpreadChild | LogicalExpression | MemberExpression | NewExpression | ObjectMethod | ObjectProperty | OptionalCallExpression | OptionalMemberExpression | ParenthesizedExpression | PipelineBareFunction | PipelineTopicExpression | ReturnStatement | SequenceExpression | SpreadElement | SwitchCase | SwitchStatement | TSAsExpression | TSDeclareMethod | TSEnumDeclaration | TSEnumMember | TSExportAssignment | TSImportType | TSInstantiationExpression | TSMethodSignature | TSNonNullExpression | TSPropertySignature | TSSatisfiesExpression | TSTypeAssertion | TaggedTemplateExpression | TemplateLiteral | ThrowStatement | TupleExpression | TypeCastExpression | UnaryExpression | UpdateExpression | VariableDeclarator | WhileStatement | WithStatement | YieldExpression;
    FunctionTypeAnnotation: ArrayTypeAnnotation | DeclareExportDeclaration | DeclareOpaqueType | DeclareTypeAlias | DeclaredPredicate | FunctionTypeAnnotation | FunctionTypeParam | IndexedAccessType | IntersectionTypeAnnotation | NullableTypeAnnotation | ObjectTypeCallProperty | ObjectTypeIndexer | ObjectTypeInternalSlot | ObjectTypeProperty | ObjectTypeSpreadProperty | OpaqueType | OptionalIndexedAccessType | TupleTypeAnnotation | TypeAlias | TypeAnnotation | TypeParameter | TypeParameterInstantiation | TypeofTypeAnnotation | UnionTypeAnnotation;
    FunctionTypeParam: DeclareExportDeclaration | DeclaredPredicate | FunctionTypeAnnotation;
    GenericTypeAnnotation: ArrayTypeAnnotation | DeclareExportDeclaration | DeclareOpaqueType | DeclareTypeAlias | DeclaredPredicate | FunctionTypeAnnotation | FunctionTypeParam | IndexedAccessType | IntersectionTypeAnnotation | NullableTypeAnnotation | ObjectTypeCallProperty | ObjectTypeIndexer | ObjectTypeInternalSlot | ObjectTypeProperty | ObjectTypeSpreadProperty | OpaqueType | OptionalIndexedAccessType | TupleTypeAnnotation | TypeAlias | TypeAnnotation | TypeParameter | TypeParameterInstantiation | TypeofTypeAnnotation | UnionTypeAnnotation;
    Identifier: ArrayExpression | ArrayPattern | ArrowFunctionExpression | AssignmentExpression | AssignmentPattern | AwaitExpression | BinaryExpression | BindExpression | BreakStatement | CallExpression | CatchClause | ClassAccessorProperty | ClassDeclaration | ClassExpression | ClassImplements | ClassMethod | ClassPrivateMethod | ClassPrivateProperty | ClassProperty | ConditionalExpression | ContinueStatement | DeclareClass | DeclareFunction | DeclareInterface | DeclareModule | DeclareOpaqueType | DeclareTypeAlias | DeclareVariable | Decorator | DoWhileStatement | EnumBooleanMember | EnumDeclaration | EnumDefaultedMember | EnumNumberMember | EnumStringMember | ExportDefaultDeclaration | ExportDefaultSpecifier | ExportNamespaceSpecifier | ExportSpecifier | ExpressionStatement | ForInStatement | ForOfStatement | ForStatement | FunctionDeclaration | FunctionExpression | FunctionTypeParam | GenericTypeAnnotation | IfStatement | ImportAttribute | ImportDefaultSpecifier | ImportExpression | ImportNamespaceSpecifier | ImportSpecifier | InterfaceDeclaration | InterfaceExtends | JSXExpressionContainer | JSXSpreadAttribute | JSXSpreadChild | LabeledStatement | LogicalExpression | MemberExpression | MetaProperty | NewExpression | ObjectMethod | ObjectProperty | ObjectTypeIndexer | ObjectTypeInternalSlot | ObjectTypeProperty | OpaqueType | OptionalCallExpression | OptionalMemberExpression | ParenthesizedExpression | PipelineBareFunction | PipelineTopicExpression | Placeholder | PrivateName | QualifiedTypeIdentifier | RestElement | ReturnStatement | SequenceExpression | SpreadElement | SwitchCase | SwitchStatement | TSAsExpression | TSCallSignatureDeclaration | TSConstructSignatureDeclaration | TSConstructorType | TSDeclareFunction | TSDeclareMethod | TSEnumDeclaration | TSEnumMember | TSExportAssignment | TSExpressionWithTypeArguments | TSFunctionType | TSImportEqualsDeclaration | TSImportType | TSIndexSignature | TSInstantiationExpression | TSInterfaceDeclaration | TSMethodSignature | TSModuleDeclaration | TSNamedTupleMember | TSNamespaceExportDeclaration | TSNonNullExpression | TSParameterProperty | TSPropertySignature | TSQualifiedName | TSSatisfiesExpression | TSTypeAliasDeclaration | TSTypeAssertion | TSTypePredicate | TSTypeQuery | TSTypeReference | TaggedTemplateExpression | TemplateLiteral | ThrowStatement | TupleExpression | TypeAlias | TypeCastExpression | UnaryExpression | UpdateExpression | VariableDeclarator | WhileStatement | WithStatement | YieldExpression;
    IfStatement: BlockStatement | DoWhileStatement | ForInStatement | ForOfStatement | ForStatement | IfStatement | LabeledStatement | Program | StaticBlock | SwitchCase | TSModuleBlock | WhileStatement | WithStatement;
    Import: ArrayExpression | ArrowFunctionExpression | AssignmentExpression | AssignmentPattern | AwaitExpression | BinaryExpression | BindExpression | CallExpression | ClassAccessorProperty | ClassDeclaration | ClassExpression | ClassMethod | ClassPrivateProperty | ClassProperty | ConditionalExpression | Decorator | DoWhileStatement | ExportDefaultDeclaration | ExpressionStatement | ForInStatement | ForOfStatement | ForStatement | IfStatement | ImportExpression | JSXExpressionContainer | JSXSpreadAttribute | JSXSpreadChild | LogicalExpression | MemberExpression | NewExpression | ObjectMethod | ObjectProperty | OptionalCallExpression | OptionalMemberExpression | ParenthesizedExpression | PipelineBareFunction | PipelineTopicExpression | ReturnStatement | SequenceExpression | SpreadElement | SwitchCase | SwitchStatement | TSAsExpression | TSDeclareMethod | TSEnumDeclaration | TSEnumMember | TSExportAssignment | TSImportType | TSInstantiationExpression | TSMethodSignature | TSNonNullExpression | TSPropertySignature | TSSatisfiesExpression | TSTypeAssertion | TaggedTemplateExpression | TemplateLiteral | ThrowStatement | TupleExpression | TypeCastExpression | UnaryExpression | UpdateExpression | VariableDeclarator | WhileStatement | WithStatement | YieldExpression;
    ImportAttribute: ExportAllDeclaration | ExportNamedDeclaration | ImportDeclaration;
    ImportDeclaration: BlockStatement | DoWhileStatement | ExportNamedDeclaration | ForInStatement | ForOfStatement | ForStatement | IfStatement | LabeledStatement | Program | StaticBlock | SwitchCase | TSModuleBlock | WhileStatement | WithStatement;
    ImportDefaultSpecifier: ImportDeclaration;
    ImportExpression: ArrayExpression | ArrowFunctionExpression | AssignmentExpression | AssignmentPattern | AwaitExpression | BinaryExpression | BindExpression | CallExpression | ClassAccessorProperty | ClassDeclaration | ClassExpression | ClassMethod | ClassPrivateProperty | ClassProperty | ConditionalExpression | Decorator | DoWhileStatement | ExportDefaultDeclaration | ExpressionStatement | ForInStatement | ForOfStatement | ForStatement | IfStatement | ImportExpression | JSXExpressionContainer | JSXSpreadAttribute | JSXSpreadChild | LogicalExpression | MemberExpression | NewExpression | ObjectMethod | ObjectProperty | OptionalCallExpression | OptionalMemberExpression | ParenthesizedExpression | PipelineBareFunction | PipelineTopicExpression | ReturnStatement | SequenceExpression | SpreadElement | SwitchCase | SwitchStatement | TSAsExpression | TSDeclareMethod | TSEnumDeclaration | TSEnumMember | TSExportAssignment | TSImportType | TSInstantiationExpression | TSMethodSignature | TSNonNullExpression | TSPropertySignature | TSSatisfiesExpression | TSTypeAssertion | TaggedTemplateExpression | TemplateLiteral | ThrowStatement | TupleExpression | TypeCastExpression | UnaryExpression | UpdateExpression | VariableDeclarator | WhileStatement | WithStatement | YieldExpression;
    ImportNamespaceSpecifier: ImportDeclaration;
    ImportSpecifier: ImportDeclaration;
    IndexedAccessType: ArrayTypeAnnotation | DeclareExportDeclaration | DeclareOpaqueType | DeclareTypeAlias | DeclaredPredicate | FunctionTypeAnnotation | FunctionTypeParam | IndexedAccessType | IntersectionTypeAnnotation | NullableTypeAnnotation | ObjectTypeCallProperty | ObjectTypeIndexer | ObjectTypeInternalSlot | ObjectTypeProperty | ObjectTypeSpreadProperty | OpaqueType | OptionalIndexedAccessType | TupleTypeAnnotation | TypeAlias | TypeAnnotation | TypeParameter | TypeParameterInstantiation | TypeofTypeAnnotation | UnionTypeAnnotation;
    InferredPredicate: ArrowFunctionExpression | DeclareExportDeclaration | DeclaredPredicate | FunctionDeclaration | FunctionExpression;
    InterfaceDeclaration: BlockStatement | DeclareExportDeclaration | DeclaredPredicate | DoWhileStatement | ExportNamedDeclaration | ForInStatement | ForOfStatement | ForStatement | IfStatement | LabeledStatement | Program | StaticBlock | SwitchCase | TSModuleBlock | WhileStatement | WithStatement;
    InterfaceExtends: ClassDeclaration | ClassExpression | DeclareClass | DeclareExportDeclaration | DeclareInterface | DeclaredPredicate | InterfaceDeclaration | InterfaceTypeAnnotation;
    InterfaceTypeAnnotation: ArrayTypeAnnotation | DeclareExportDeclaration | DeclareOpaqueType | DeclareTypeAlias | DeclaredPredicate | FunctionTypeAnnotation | FunctionTypeParam | IndexedAccessType | IntersectionTypeAnnotation | NullableTypeAnnotation | ObjectTypeCallProperty | ObjectTypeIndexer | ObjectTypeInternalSlot | ObjectTypeProperty | ObjectTypeSpreadProperty | OpaqueType | OptionalIndexedAccessType | TupleTypeAnnotation | TypeAlias | TypeAnnotation | TypeParameter | TypeParameterInstantiation | TypeofTypeAnnotation | UnionTypeAnnotation;
    InterpreterDirective: Program;
    IntersectionTypeAnnotation: ArrayTypeAnnotation | DeclareExportDeclaration | DeclareOpaqueType | DeclareTypeAlias | DeclaredPredicate | FunctionTypeAnnotation | FunctionTypeParam | IndexedAccessType | IntersectionTypeAnnotation | NullableTypeAnnotation | ObjectTypeCallProperty | ObjectTypeIndexer | ObjectTypeInternalSlot | ObjectTypeProperty | ObjectTypeSpreadProperty | OpaqueType | OptionalIndexedAccessType | TupleTypeAnnotation | TypeAlias | TypeAnnotation | TypeParameter | TypeParameterInstantiation | TypeofTypeAnnotation | UnionTypeAnnotation;
    JSXAttribute: JSXOpeningElement;
    JSXClosingElement: JSXElement;
    JSXClosingFragment: JSXFragment;
    JSXElement: ArrayExpression | ArrowFunctionExpression | AssignmentExpression | AssignmentPattern | AwaitExpression | BinaryExpression | BindExpression | CallExpression | ClassAccessorProperty | ClassDeclaration | ClassExpression | ClassMethod | ClassPrivateProperty | ClassProperty | ConditionalExpression | Decorator | DoWhileStatement | ExportDefaultDeclaration | ExpressionStatement | ForInStatement | ForOfStatement | ForStatement | IfStatement | ImportExpression | JSXAttribute | JSXElement | JSXExpressionContainer | JSXFragment | JSXSpreadAttribute | JSXSpreadChild | LogicalExpression | MemberExpression | NewExpression | ObjectMethod | ObjectProperty | OptionalCallExpression | OptionalMemberExpression | ParenthesizedExpression | PipelineBareFunction | PipelineTopicExpression | ReturnStatement | SequenceExpression | SpreadElement | SwitchCase | SwitchStatement | TSAsExpression | TSDeclareMethod | TSEnumDeclaration | TSEnumMember | TSExportAssignment | TSImportType | TSInstantiationExpression | TSMethodSignature | TSNonNullExpression | TSPropertySignature | TSSatisfiesExpression | TSTypeAssertion | TaggedTemplateExpression | TemplateLiteral | ThrowStatement | TupleExpression | TypeCastExpression | UnaryExpression | UpdateExpression | VariableDeclarator | WhileStatement | WithStatement | YieldExpression;
    JSXEmptyExpression: JSXExpressionContainer;
    JSXExpressionContainer: JSXAttribute | JSXElement | JSXFragment;
    JSXFragment: ArrayExpression | ArrowFunctionExpression | AssignmentExpression | AssignmentPattern | AwaitExpression | BinaryExpression | BindExpression | CallExpression | ClassAccessorProperty | ClassDeclaration | ClassExpression | ClassMethod | ClassPrivateProperty | ClassProperty | ConditionalExpression | Decorator | DoWhileStatement | ExportDefaultDeclaration | ExpressionStatement | ForInStatement | ForOfStatement | ForStatement | IfStatement | ImportExpression | JSXAttribute | JSXElement | JSXExpressionContainer | JSXFragment | JSXSpreadAttribute | JSXSpreadChild | LogicalExpression | MemberExpression | NewExpression | ObjectMethod | ObjectProperty | OptionalCallExpression | OptionalMemberExpression | ParenthesizedExpression | PipelineBareFunction | PipelineTopicExpression | ReturnStatement | SequenceExpression | SpreadElement | SwitchCase | SwitchStatement | TSAsExpression | TSDeclareMethod | TSEnumDeclaration | TSEnumMember | TSExportAssignment | TSImportType | TSInstantiationExpression | TSMethodSignature | TSNonNullExpression | TSPropertySignature | TSSatisfiesExpression | TSTypeAssertion | TaggedTemplateExpression | TemplateLiteral | ThrowStatement | TupleExpression | TypeCastExpression | UnaryExpression | UpdateExpression | VariableDeclarator | WhileStatement | WithStatement | YieldExpression;
    JSXIdentifier: JSXAttribute | JSXClosingElement | JSXMemberExpression | JSXNamespacedName | JSXOpeningElement;
    JSXMemberExpression: JSXClosingElement | JSXMemberExpression | JSXOpeningElement;
    JSXNamespacedName: JSXAttribute | JSXClosingElement | JSXOpeningElement;
    JSXOpeningElement: JSXElement;
    JSXOpeningFragment: JSXFragment;
    JSXSpreadAttribute: JSXOpeningElement;
    JSXSpreadChild: JSXElement | JSXFragment;
    JSXText: JSXElement | JSXFragment;
    LabeledStatement: BlockStatement | DoWhileStatement | ForInStatement | ForOfStatement | ForStatement | IfStatement | LabeledStatement | Program | StaticBlock | SwitchCase | TSModuleBlock | WhileStatement | WithStatement;
    LogicalExpression: ArrayExpression | ArrowFunctionExpression | AssignmentExpression | AssignmentPattern | AwaitExpression | BinaryExpression | BindExpression | CallExpression | ClassAccessorProperty | ClassDeclaration | ClassExpression | ClassMethod | ClassPrivateProperty | ClassProperty | ConditionalExpression | Decorator | DoWhileStatement | ExportDefaultDeclaration | ExpressionStatement | ForInStatement | ForOfStatement | ForStatement | IfStatement | ImportExpression | JSXExpressionContainer | JSXSpreadAttribute | JSXSpreadChild | LogicalExpression | MemberExpression | NewExpression | ObjectMethod | ObjectProperty | OptionalCallExpression | OptionalMemberExpression | ParenthesizedExpression | PipelineBareFunction | PipelineTopicExpression | ReturnStatement | SequenceExpression | SpreadElement | SwitchCase | SwitchStatement | TSAsExpression | TSDeclareMethod | TSEnumDeclaration | TSEnumMember | TSExportAssignment | TSImportType | TSInstantiationExpression | TSMethodSignature | TSNonNullExpression | TSPropertySignature | TSSatisfiesExpression | TSTypeAssertion | TaggedTemplateExpression | TemplateLiteral | ThrowStatement | TupleExpression | TypeCastExpression | UnaryExpression | UpdateExpression | VariableDeclarator | WhileStatement | WithStatement | YieldExpression;
    MemberExpression: ArrayExpression | ArrayPattern | ArrowFunctionExpression | AssignmentExpression | AssignmentPattern | AwaitExpression | BinaryExpression | BindExpression | CallExpression | ClassAccessorProperty | ClassDeclaration | ClassExpression | ClassMethod | ClassPrivateProperty | ClassProperty | ConditionalExpression | Decorator | DoWhileStatement | ExportDefaultDeclaration | ExpressionStatement | ForInStatement | ForOfStatement | ForStatement | IfStatement | ImportExpression | JSXExpressionContainer | JSXSpreadAttribute | JSXSpreadChild | LogicalExpression | MemberExpression | NewExpression | ObjectMethod | ObjectProperty | OptionalCallExpression | OptionalMemberExpression | ParenthesizedExpression | PipelineBareFunction | PipelineTopicExpression | RestElement | ReturnStatement | SequenceExpression | SpreadElement | SwitchCase | SwitchStatement | TSAsExpression | TSDeclareMethod | TSEnumDeclaration | TSEnumMember | TSExportAssignment | TSImportType | TSInstantiationExpression | TSMethodSignature | TSNonNullExpression | TSPropertySignature | TSSatisfiesExpression | TSTypeAssertion | TaggedTemplateExpression | TemplateLiteral | ThrowStatement | TupleExpression | TypeCastExpression | UnaryExpression | UpdateExpression | VariableDeclarator | WhileStatement | WithStatement | YieldExpression;
    MetaProperty: ArrayExpression | ArrowFunctionExpression | AssignmentExpression | AssignmentPattern | AwaitExpression | BinaryExpression | BindExpression | CallExpression | ClassAccessorProperty | ClassDeclaration | ClassExpression | ClassMethod | ClassPrivateProperty | ClassProperty | ConditionalExpression | Decorator | DoWhileStatement | ExportDefaultDeclaration | ExpressionStatement | ForInStatement | ForOfStatement | ForStatement | IfStatement | ImportExpression | JSXExpressionContainer | JSXSpreadAttribute | JSXSpreadChild | LogicalExpression | MemberExpression | NewExpression | ObjectMethod | ObjectProperty | OptionalCallExpression | OptionalMemberExpression | ParenthesizedExpression | PipelineBareFunction | PipelineTopicExpression | ReturnStatement | SequenceExpression | SpreadElement | SwitchCase | SwitchStatement | TSAsExpression | TSDeclareMethod | TSEnumDeclaration | TSEnumMember | TSExportAssignment | TSImportType | TSInstantiationExpression | TSMethodSignature | TSNonNullExpression | TSPropertySignature | TSSatisfiesExpression | TSTypeAssertion | TaggedTemplateExpression | TemplateLiteral | ThrowStatement | TupleExpression | TypeCastExpression | UnaryExpression | UpdateExpression | VariableDeclarator | WhileStatement | WithStatement | YieldExpression;
    MixedTypeAnnotation: ArrayTypeAnnotation | DeclareExportDeclaration | DeclareOpaqueType | DeclareTypeAlias | DeclaredPredicate | FunctionTypeAnnotation | FunctionTypeParam | IndexedAccessType | IntersectionTypeAnnotation | NullableTypeAnnotation | ObjectTypeCallProperty | ObjectTypeIndexer | ObjectTypeInternalSlot | ObjectTypeProperty | ObjectTypeSpreadProperty | OpaqueType | OptionalIndexedAccessType | TupleTypeAnnotation | TypeAlias | TypeAnnotation | TypeParameter | TypeParameterInstantiation | TypeofTypeAnnotation | UnionTypeAnnotation;
    ModuleExpression: ArrayExpression | ArrowFunctionExpression | AssignmentExpression | AssignmentPattern | AwaitExpression | BinaryExpression | BindExpression | CallExpression | ClassAccessorProperty | ClassDeclaration | ClassExpression | ClassMethod | ClassPrivateProperty | ClassProperty | ConditionalExpression | Decorator | DoWhileStatement | ExportDefaultDeclaration | ExpressionStatement | ForInStatement | ForOfStatement | ForStatement | IfStatement | ImportExpression | JSXExpressionContainer | JSXSpreadAttribute | JSXSpreadChild | LogicalExpression | MemberExpression | NewExpression | ObjectMethod | ObjectProperty | OptionalCallExpression | OptionalMemberExpression | ParenthesizedExpression | PipelineBareFunction | PipelineTopicExpression | ReturnStatement | SequenceExpression | SpreadElement | SwitchCase | SwitchStatement | TSAsExpression | TSDeclareMethod | TSEnumDeclaration | TSEnumMember | TSExportAssignment | TSImportType | TSInstantiationExpression | TSMethodSignature | TSNonNullExpression | TSPropertySignature | TSSatisfiesExpression | TSTypeAssertion | TaggedTemplateExpression | TemplateLiteral | ThrowStatement | TupleExpression | TypeCastExpression | UnaryExpression | UpdateExpression | VariableDeclarator | WhileStatement | WithStatement | YieldExpression;
    NewExpression: ArrayExpression | ArrowFunctionExpression | AssignmentExpression | AssignmentPattern | AwaitExpression | BinaryExpression | BindExpression | CallExpression | ClassAccessorProperty | ClassDeclaration | ClassExpression | ClassMethod | ClassPrivateProperty | ClassProperty | ConditionalExpression | Decorator | DoWhileStatement | ExportDefaultDeclaration | ExpressionStatement | ForInStatement | ForOfStatement | ForStatement | IfStatement | ImportExpression | JSXExpressionContainer | JSXSpreadAttribute | JSXSpreadChild | LogicalExpression | MemberExpression | NewExpression | ObjectMethod | ObjectProperty | OptionalCallExpression | OptionalMemberExpression | ParenthesizedExpression | PipelineBareFunction | PipelineTopicExpression | ReturnStatement | SequenceExpression | SpreadElement | SwitchCase | SwitchStatement | TSAsExpression | TSDeclareMethod | TSEnumDeclaration | TSEnumMember | TSExportAssignment | TSImportType | TSInstantiationExpression | TSMethodSignature | TSNonNullExpression | TSPropertySignature | TSSatisfiesExpression | TSTypeAssertion | TaggedTemplateExpression | TemplateLiteral | ThrowStatement | TupleExpression | TypeCastExpression | UnaryExpression | UpdateExpression | VariableDeclarator | WhileStatement | WithStatement | YieldExpression;
    Noop: ArrayPattern | ArrowFunctionExpression | AssignmentPattern | ClassAccessorProperty | ClassDeclaration | ClassExpression | ClassMethod | ClassPrivateMethod | ClassPrivateProperty | ClassProperty | FunctionDeclaration | FunctionExpression | Identifier | ObjectMethod | ObjectPattern | RestElement | TSDeclareFunction | TSDeclareMethod;
    NullLiteral: ArrayExpression | ArrowFunctionExpression | AssignmentExpression | AssignmentPattern | AwaitExpression | BinaryExpression | BindExpression | CallExpression | ClassAccessorProperty | ClassDeclaration | ClassExpression | ClassMethod | ClassPrivateProperty | ClassProperty | ConditionalExpression | Decorator | DoWhileStatement | ExportDefaultDeclaration | ExpressionStatement | ForInStatement | ForOfStatement | ForStatement | IfStatement | ImportExpression | JSXExpressionContainer | JSXSpreadAttribute | JSXSpreadChild | LogicalExpression | MemberExpression | NewExpression | ObjectMethod | ObjectProperty | OptionalCallExpression | OptionalMemberExpression | ParenthesizedExpression | PipelineBareFunction | PipelineTopicExpression | ReturnStatement | SequenceExpression | SpreadElement | SwitchCase | SwitchStatement | TSAsExpression | TSDeclareMethod | TSEnumDeclaration | TSEnumMember | TSExportAssignment | TSImportType | TSInstantiationExpression | TSMethodSignature | TSNonNullExpression | TSPropertySignature | TSSatisfiesExpression | TSTypeAssertion | TaggedTemplateExpression | TemplateLiteral | ThrowStatement | TupleExpression | TypeCastExpression | UnaryExpression | UpdateExpression | VariableDeclarator | WhileStatement | WithStatement | YieldExpression;
    NullLiteralTypeAnnotation: ArrayTypeAnnotation | DeclareExportDeclaration | DeclareOpaqueType | DeclareTypeAlias | DeclaredPredicate | FunctionTypeAnnotation | FunctionTypeParam | IndexedAccessType | IntersectionTypeAnnotation | NullableTypeAnnotation | ObjectTypeCallProperty | ObjectTypeIndexer | ObjectTypeInternalSlot | ObjectTypeProperty | ObjectTypeSpreadProperty | OpaqueType | OptionalIndexedAccessType | TupleTypeAnnotation | TypeAlias | TypeAnnotation | TypeParameter | TypeParameterInstantiation | TypeofTypeAnnotation | UnionTypeAnnotation;
    NullableTypeAnnotation: ArrayTypeAnnotation | DeclareExportDeclaration | DeclareOpaqueType | DeclareTypeAlias | DeclaredPredicate | FunctionTypeAnnotation | FunctionTypeParam | IndexedAccessType | IntersectionTypeAnnotation | NullableTypeAnnotation | ObjectTypeCallProperty | ObjectTypeIndexer | ObjectTypeInternalSlot | ObjectTypeProperty | ObjectTypeSpreadProperty | OpaqueType | OptionalIndexedAccessType | TupleTypeAnnotation | TypeAlias | TypeAnnotation | TypeParameter | TypeParameterInstantiation | TypeofTypeAnnotation | UnionTypeAnnotation;
    NumberLiteral: null;
    NumberLiteralTypeAnnotation: ArrayTypeAnnotation | DeclareExportDeclaration | DeclareOpaqueType | DeclareTypeAlias | DeclaredPredicate | FunctionTypeAnnotation | FunctionTypeParam | IndexedAccessType | IntersectionTypeAnnotation | NullableTypeAnnotation | ObjectTypeCallProperty | ObjectTypeIndexer | ObjectTypeInternalSlot | ObjectTypeProperty | ObjectTypeSpreadProperty | OpaqueType | OptionalIndexedAccessType | TupleTypeAnnotation | TypeAlias | TypeAnnotation | TypeParameter | TypeParameterInstantiation | TypeofTypeAnnotation | UnionTypeAnnotation;
    NumberTypeAnnotation: ArrayTypeAnnotation | DeclareExportDeclaration | DeclareOpaqueType | DeclareTypeAlias | DeclaredPredicate | FunctionTypeAnnotation | FunctionTypeParam | IndexedAccessType | IntersectionTypeAnnotation | NullableTypeAnnotation | ObjectTypeCallProperty | ObjectTypeIndexer | ObjectTypeInternalSlot | ObjectTypeProperty | ObjectTypeSpreadProperty | OpaqueType | OptionalIndexedAccessType | TupleTypeAnnotation | TypeAlias | TypeAnnotation | TypeParameter | TypeParameterInstantiation | TypeofTypeAnnotation | UnionTypeAnnotation;
    NumericLiteral: ArrayExpression | ArrowFunctionExpression | AssignmentExpression | AssignmentPattern | AwaitExpression | BinaryExpression | BindExpression | CallExpression | ClassAccessorProperty | ClassDeclaration | ClassExpression | ClassMethod | ClassPrivateProperty | ClassProperty | ConditionalExpression | Decorator | DoWhileStatement | EnumNumberMember | ExportDefaultDeclaration | ExpressionStatement | ForInStatement | ForOfStatement | ForStatement | IfStatement | ImportExpression | JSXExpressionContainer | JSXSpreadAttribute | JSXSpreadChild | LogicalExpression | MemberExpression | NewExpression | ObjectMethod | ObjectProperty | OptionalCallExpression | OptionalMemberExpression | ParenthesizedExpression | PipelineBareFunction | PipelineTopicExpression | ReturnStatement | SequenceExpression | SpreadElement | SwitchCase | SwitchStatement | TSAsExpression | TSDeclareMethod | TSEnumDeclaration | TSEnumMember | TSExportAssignment | TSImportType | TSInstantiationExpression | TSLiteralType | TSMethodSignature | TSNonNullExpression | TSPropertySignature | TSSatisfiesExpression | TSTypeAssertion | TaggedTemplateExpression | TemplateLiteral | ThrowStatement | TupleExpression | TypeCastExpression | UnaryExpression | UpdateExpression | VariableDeclarator | WhileStatement | WithStatement | YieldExpression;
    ObjectExpression: ArrayExpression | ArrowFunctionExpression | AssignmentExpression | AssignmentPattern | AwaitExpression | BinaryExpression | BindExpression | CallExpression | ClassAccessorProperty | ClassDeclaration | ClassExpression | ClassMethod | ClassPrivateProperty | ClassProperty | ConditionalExpression | Decorator | DoWhileStatement | ExportDefaultDeclaration | ExpressionStatement | ForInStatement | ForOfStatement | ForStatement | IfStatement | ImportExpression | JSXExpressionContainer | JSXSpreadAttribute | JSXSpreadChild | LogicalExpression | MemberExpression | NewExpression | ObjectMethod | ObjectProperty | OptionalCallExpression | OptionalMemberExpression | ParenthesizedExpression | PipelineBareFunction | PipelineTopicExpression | ReturnStatement | SequenceExpression | SpreadElement | SwitchCase | SwitchStatement | TSAsExpression | TSDeclareMethod | TSEnumDeclaration | TSEnumMember | TSExportAssignment | TSImportType | TSInstantiationExpression | TSMethodSignature | TSNonNullExpression | TSPropertySignature | TSSatisfiesExpression | TSTypeAssertion | TaggedTemplateExpression | TemplateLiteral | ThrowStatement | TupleExpression | TypeCastExpression | UnaryExpression | UpdateExpression | VariableDeclarator | WhileStatement | WithStatement | YieldExpression;
    ObjectMethod: ObjectExpression;
    ObjectPattern: ArrayPattern | ArrowFunctionExpression | AssignmentExpression | AssignmentPattern | CatchClause | ClassMethod | ClassPrivateMethod | ForInStatement | ForOfStatement | FunctionDeclaration | FunctionExpression | ObjectMethod | ObjectProperty | RestElement | TSCallSignatureDeclaration | TSConstructSignatureDeclaration | TSConstructorType | TSDeclareFunction | TSDeclareMethod | TSFunctionType | TSMethodSignature | VariableDeclarator;
    ObjectProperty: ObjectExpression | ObjectPattern | RecordExpression;
    ObjectTypeAnnotation: ArrayTypeAnnotation | DeclareClass | DeclareExportDeclaration | DeclareInterface | DeclareOpaqueType | DeclareTypeAlias | DeclaredPredicate | FunctionTypeAnnotation | FunctionTypeParam | IndexedAccessType | InterfaceDeclaration | InterfaceTypeAnnotation | IntersectionTypeAnnotation | NullableTypeAnnotation | ObjectTypeCallProperty | ObjectTypeIndexer | ObjectTypeInternalSlot | ObjectTypeProperty | ObjectTypeSpreadProperty | OpaqueType | OptionalIndexedAccessType | TupleTypeAnnotation | TypeAlias | TypeAnnotation | TypeParameter | TypeParameterInstantiation | TypeofTypeAnnotation | UnionTypeAnnotation;
    ObjectTypeCallProperty: DeclareExportDeclaration | DeclaredPredicate | ObjectTypeAnnotation;
    ObjectTypeIndexer: DeclareExportDeclaration | DeclaredPredicate | ObjectTypeAnnotation;
    ObjectTypeInternalSlot: DeclareExportDeclaration | DeclaredPredicate | ObjectTypeAnnotation;
    ObjectTypeProperty: DeclareExportDeclaration | DeclaredPredicate | ObjectTypeAnnotation;
    ObjectTypeSpreadProperty: DeclareExportDeclaration | DeclaredPredicate | ObjectTypeAnnotation;
    OpaqueType: BlockStatement | DeclareExportDeclaration | DeclaredPredicate | DoWhileStatement | ExportNamedDeclaration | ForInStatement | ForOfStatement | ForStatement | IfStatement | LabeledStatement | Program | StaticBlock | SwitchCase | TSModuleBlock | WhileStatement | WithStatement;
    OptionalCallExpression: ArrayExpression | ArrowFunctionExpression | AssignmentExpression | AssignmentPattern | AwaitExpression | BinaryExpression | BindExpression | CallExpression | ClassAccessorProperty | ClassDeclaration | ClassExpression | ClassMethod | ClassPrivateProperty | ClassProperty | ConditionalExpression | Decorator | DoWhileStatement | ExportDefaultDeclaration | ExpressionStatement | ForInStatement | ForOfStatement | ForStatement | IfStatement | ImportExpression | JSXExpressionContainer | JSXSpreadAttribute | JSXSpreadChild | LogicalExpression | MemberExpression | NewExpression | ObjectMethod | ObjectProperty | OptionalCallExpression | OptionalMemberExpression | ParenthesizedExpression | PipelineBareFunction | PipelineTopicExpression | ReturnStatement | SequenceExpression | SpreadElement | SwitchCase | SwitchStatement | TSAsExpression | TSDeclareMethod | TSEnumDeclaration | TSEnumMember | TSExportAssignment | TSImportType | TSInstantiationExpression | TSMethodSignature | TSNonNullExpression | TSPropertySignature | TSSatisfiesExpression | TSTypeAssertion | TaggedTemplateExpression | TemplateLiteral | ThrowStatement | TupleExpression | TypeCastExpression | UnaryExpression | UpdateExpression | VariableDeclarator | WhileStatement | WithStatement | YieldExpression;
    OptionalIndexedAccessType: ArrayTypeAnnotation | DeclareExportDeclaration | DeclareOpaqueType | DeclareTypeAlias | DeclaredPredicate | FunctionTypeAnnotation | FunctionTypeParam | IndexedAccessType | IntersectionTypeAnnotation | NullableTypeAnnotation | ObjectTypeCallProperty | ObjectTypeIndexer | ObjectTypeInternalSlot | ObjectTypeProperty | ObjectTypeSpreadProperty | OpaqueType | OptionalIndexedAccessType | TupleTypeAnnotation | TypeAlias | TypeAnnotation | TypeParameter | TypeParameterInstantiation | TypeofTypeAnnotation | UnionTypeAnnotation;
    OptionalMemberExpression: ArrayExpression | ArrowFunctionExpression | AssignmentExpression | AssignmentPattern | AwaitExpression | BinaryExpression | BindExpression | CallExpression | ClassAccessorProperty | ClassDeclaration | ClassExpression | ClassMethod | ClassPrivateProperty | ClassProperty | ConditionalExpression | Decorator | DoWhileStatement | ExportDefaultDeclaration | ExpressionStatement | ForInStatement | ForOfStatement | ForStatement | IfStatement | ImportExpression | JSXExpressionContainer | JSXSpreadAttribute | JSXSpreadChild | LogicalExpression | MemberExpression | NewExpression | ObjectMethod | ObjectProperty | OptionalCallExpression | OptionalMemberExpression | ParenthesizedExpression | PipelineBareFunction | PipelineTopicExpression | ReturnStatement | SequenceExpression | SpreadElement | SwitchCase | SwitchStatement | TSAsExpression | TSDeclareMethod | TSEnumDeclaration | TSEnumMember | TSExportAssignment | TSImportType | TSInstantiationExpression | TSMethodSignature | TSNonNullExpression | TSPropertySignature | TSSatisfiesExpression | TSTypeAssertion | TaggedTemplateExpression | TemplateLiteral | ThrowStatement | TupleExpression | TypeCastExpression | UnaryExpression | UpdateExpression | VariableDeclarator | WhileStatement | WithStatement | YieldExpression;
    ParenthesizedExpression: ArrayExpression | ArrowFunctionExpression | AssignmentExpression | AssignmentPattern | AwaitExpression | BinaryExpression | BindExpression | CallExpression | ClassAccessorProperty | ClassDeclaration | ClassExpression | ClassMethod | ClassPrivateProperty | ClassProperty | ConditionalExpression | Decorator | DoWhileStatement | ExportDefaultDeclaration | ExpressionStatement | ForInStatement | ForOfStatement | ForStatement | IfStatement | ImportExpression | JSXExpressionContainer | JSXSpreadAttribute | JSXSpreadChild | LogicalExpression | MemberExpression | NewExpression | ObjectMethod | ObjectProperty | OptionalCallExpression | OptionalMemberExpression | ParenthesizedExpression | PipelineBareFunction | PipelineTopicExpression | ReturnStatement | SequenceExpression | SpreadElement | SwitchCase | SwitchStatement | TSAsExpression | TSDeclareMethod | TSEnumDeclaration | TSEnumMember | TSExportAssignment | TSImportType | TSInstantiationExpression | TSMethodSignature | TSNonNullExpression | TSPropertySignature | TSSatisfiesExpression | TSTypeAssertion | TaggedTemplateExpression | TemplateLiteral | ThrowStatement | TupleExpression | TypeCastExpression | UnaryExpression | UpdateExpression | VariableDeclarator | WhileStatement | WithStatement | YieldExpression;
    PipelineBareFunction: ArrayExpression | ArrowFunctionExpression | AssignmentExpression | AssignmentPattern | AwaitExpression | BinaryExpression | BindExpression | CallExpression | ClassAccessorProperty | ClassDeclaration | ClassExpression | ClassMethod | ClassPrivateProperty | ClassProperty | ConditionalExpression | Decorator | DoWhileStatement | ExportDefaultDeclaration | ExpressionStatement | ForInStatement | ForOfStatement | ForStatement | IfStatement | ImportExpression | JSXExpressionContainer | JSXSpreadAttribute | JSXSpreadChild | LogicalExpression | MemberExpression | NewExpression | ObjectMethod | ObjectProperty | OptionalCallExpression | OptionalMemberExpression | ParenthesizedExpression | PipelineBareFunction | PipelineTopicExpression | ReturnStatement | SequenceExpression | SpreadElement | SwitchCase | SwitchStatement | TSAsExpression | TSDeclareMethod | TSEnumDeclaration | TSEnumMember | TSExportAssignment | TSImportType | TSInstantiationExpression | TSMethodSignature | TSNonNullExpression | TSPropertySignature | TSSatisfiesExpression | TSTypeAssertion | TaggedTemplateExpression | TemplateLiteral | ThrowStatement | TupleExpression | TypeCastExpression | UnaryExpression | UpdateExpression | VariableDeclarator | WhileStatement | WithStatement | YieldExpression;
    PipelinePrimaryTopicReference: ArrayExpression | ArrowFunctionExpression | AssignmentExpression | AssignmentPattern | AwaitExpression | BinaryExpression | BindExpression | CallExpression | ClassAccessorProperty | ClassDeclaration | ClassExpression | ClassMethod | ClassPrivateProperty | ClassProperty | ConditionalExpression | Decorator | DoWhileStatement | ExportDefaultDeclaration | ExpressionStatement | ForInStatement | ForOfStatement | ForStatement | IfStatement | ImportExpression | JSXExpressionContainer | JSXSpreadAttribute | JSXSpreadChild | LogicalExpression | MemberExpression | NewExpression | ObjectMethod | ObjectProperty | OptionalCallExpression | OptionalMemberExpression | ParenthesizedExpression | PipelineBareFunction | PipelineTopicExpression | ReturnStatement | SequenceExpression | SpreadElement | SwitchCase | SwitchStatement | TSAsExpression | TSDeclareMethod | TSEnumDeclaration | TSEnumMember | TSExportAssignment | TSImportType | TSInstantiationExpression | TSMethodSignature | TSNonNullExpression | TSPropertySignature | TSSatisfiesExpression | TSTypeAssertion | TaggedTemplateExpression | TemplateLiteral | ThrowStatement | TupleExpression | TypeCastExpression | UnaryExpression | UpdateExpression | VariableDeclarator | WhileStatement | WithStatement | YieldExpression;
    PipelineTopicExpression: ArrayExpression | ArrowFunctionExpression | AssignmentExpression | AssignmentPattern | AwaitExpression | BinaryExpression | BindExpression | CallExpression | ClassAccessorProperty | ClassDeclaration | ClassExpression | ClassMethod | ClassPrivateProperty | ClassProperty | ConditionalExpression | Decorator | DoWhileStatement | ExportDefaultDeclaration | ExpressionStatement | ForInStatement | ForOfStatement | ForStatement | IfStatement | ImportExpression | JSXExpressionContainer | JSXSpreadAttribute | JSXSpreadChild | LogicalExpression | MemberExpression | NewExpression | ObjectMethod | ObjectProperty | OptionalCallExpression | OptionalMemberExpression | ParenthesizedExpression | PipelineBareFunction | PipelineTopicExpression | ReturnStatement | SequenceExpression | SpreadElement | SwitchCase | SwitchStatement | TSAsExpression | TSDeclareMethod | TSEnumDeclaration | TSEnumMember | TSExportAssignment | TSImportType | TSInstantiationExpression | TSMethodSignature | TSNonNullExpression | TSPropertySignature | TSSatisfiesExpression | TSTypeAssertion | TaggedTemplateExpression | TemplateLiteral | ThrowStatement | TupleExpression | TypeCastExpression | UnaryExpression | UpdateExpression | VariableDeclarator | WhileStatement | WithStatement | YieldExpression;
    Placeholder: Node;
    PrivateName: BinaryExpression | ClassAccessorProperty | ClassPrivateMethod | ClassPrivateProperty | MemberExpression | ObjectProperty;
    Program: File | ModuleExpression;
    QualifiedTypeIdentifier: DeclareExportDeclaration | DeclaredPredicate | GenericTypeAnnotation | InterfaceExtends | QualifiedTypeIdentifier;
    RecordExpression: ArrayExpression | ArrowFunctionExpression | AssignmentExpression | AssignmentPattern | AwaitExpression | BinaryExpression | BindExpression | CallExpression | ClassAccessorProperty | ClassDeclaration | ClassExpression | ClassMethod | ClassPrivateProperty | ClassProperty | ConditionalExpression | Decorator | DoWhileStatement | ExportDefaultDeclaration | ExpressionStatement | ForInStatement | ForOfStatement | ForStatement | IfStatement | ImportExpression | JSXExpressionContainer | JSXSpreadAttribute | JSXSpreadChild | LogicalExpression | MemberExpression | NewExpression | ObjectMethod | ObjectProperty | OptionalCallExpression | OptionalMemberExpression | ParenthesizedExpression | PipelineBareFunction | PipelineTopicExpression | ReturnStatement | SequenceExpression | SpreadElement | SwitchCase | SwitchStatement | TSAsExpression | TSDeclareMethod | TSEnumDeclaration | TSEnumMember | TSExportAssignment | TSImportType | TSInstantiationExpression | TSMethodSignature | TSNonNullExpression | TSPropertySignature | TSSatisfiesExpression | TSTypeAssertion | TaggedTemplateExpression | TemplateLiteral | ThrowStatement | TupleExpression | TypeCastExpression | UnaryExpression | UpdateExpression | VariableDeclarator | WhileStatement | WithStatement | YieldExpression;
    RegExpLiteral: ArrayExpression | ArrowFunctionExpression | AssignmentExpression | AssignmentPattern | AwaitExpression | BinaryExpression | BindExpression | CallExpression | ClassAccessorProperty | ClassDeclaration | ClassExpression | ClassMethod | ClassPrivateProperty | ClassProperty | ConditionalExpression | Decorator | DoWhileStatement | ExportDefaultDeclaration | ExpressionStatement | ForInStatement | ForOfStatement | ForStatement | IfStatement | ImportExpression | JSXExpressionContainer | JSXSpreadAttribute | JSXSpreadChild | LogicalExpression | MemberExpression | NewExpression | ObjectMethod | ObjectProperty | OptionalCallExpression | OptionalMemberExpression | ParenthesizedExpression | PipelineBareFunction | PipelineTopicExpression | ReturnStatement | SequenceExpression | SpreadElement | SwitchCase | SwitchStatement | TSAsExpression | TSDeclareMethod | TSEnumDeclaration | TSEnumMember | TSExportAssignment | TSImportType | TSInstantiationExpression | TSMethodSignature | TSNonNullExpression | TSPropertySignature | TSSatisfiesExpression | TSTypeAssertion | TaggedTemplateExpression | TemplateLiteral | ThrowStatement | TupleExpression | TypeCastExpression | UnaryExpression | UpdateExpression | VariableDeclarator | WhileStatement | WithStatement | YieldExpression;
    RegexLiteral: null;
    RestElement: ArrayPattern | ArrowFunctionExpression | AssignmentExpression | ClassMethod | ClassPrivateMethod | ForInStatement | ForOfStatement | FunctionDeclaration | FunctionExpression | ObjectMethod | ObjectPattern | ObjectProperty | RestElement | TSCallSignatureDeclaration | TSConstructSignatureDeclaration | TSConstructorType | TSDeclareFunction | TSDeclareMethod | TSFunctionType | TSMethodSignature | VariableDeclarator;
    RestProperty: null;
    ReturnStatement: BlockStatement | DoWhileStatement | ForInStatement | ForOfStatement | ForStatement | IfStatement | LabeledStatement | Program | StaticBlock | SwitchCase | TSModuleBlock | WhileStatement | WithStatement;
    SequenceExpression: ArrayExpression | ArrowFunctionExpression | AssignmentExpression | AssignmentPattern | AwaitExpression | BinaryExpression | BindExpression | CallExpression | ClassAccessorProperty | ClassDeclaration | ClassExpression | ClassMethod | ClassPrivateProperty | ClassProperty | ConditionalExpression | Decorator | DoWhileStatement | ExportDefaultDeclaration | ExpressionStatement | ForInStatement | ForOfStatement | ForStatement | IfStatement | ImportExpression | JSXExpressionContainer | JSXSpreadAttribute | JSXSpreadChild | LogicalExpression | MemberExpression | NewExpression | ObjectMethod | ObjectProperty | OptionalCallExpression | OptionalMemberExpression | ParenthesizedExpression | PipelineBareFunction | PipelineTopicExpression | ReturnStatement | SequenceExpression | SpreadElement | SwitchCase | SwitchStatement | TSAsExpression | TSDeclareMethod | TSEnumDeclaration | TSEnumMember | TSExportAssignment | TSImportType | TSInstantiationExpression | TSMethodSignature | TSNonNullExpression | TSPropertySignature | TSSatisfiesExpression | TSTypeAssertion | TaggedTemplateExpression | TemplateLiteral | ThrowStatement | TupleExpression | TypeCastExpression | UnaryExpression | UpdateExpression | VariableDeclarator | WhileStatement | WithStatement | YieldExpression;
    SpreadElement: ArrayExpression | CallExpression | NewExpression | ObjectExpression | OptionalCallExpression | RecordExpression | TupleExpression;
    SpreadProperty: null;
    StaticBlock: ClassBody;
    StringLiteral: ArrayExpression | ArrowFunctionExpression | AssignmentExpression | AssignmentPattern | AwaitExpression | BinaryExpression | BindExpression | CallExpression | ClassAccessorProperty | ClassDeclaration | ClassExpression | ClassMethod | ClassPrivateProperty | ClassProperty | ConditionalExpression | DeclareExportAllDeclaration | DeclareExportDeclaration | DeclareModule | Decorator | DoWhileStatement | EnumStringMember | ExportAllDeclaration | ExportDefaultDeclaration | ExportNamedDeclaration | ExportSpecifier | ExpressionStatement | ForInStatement | ForOfStatement | ForStatement | IfStatement | ImportAttribute | ImportDeclaration | ImportExpression | ImportSpecifier | JSXAttribute | JSXExpressionContainer | JSXSpreadAttribute | JSXSpreadChild | LogicalExpression | MemberExpression | NewExpression | ObjectMethod | ObjectProperty | ObjectTypeProperty | OptionalCallExpression | OptionalMemberExpression | ParenthesizedExpression | PipelineBareFunction | PipelineTopicExpression | ReturnStatement | SequenceExpression | SpreadElement | SwitchCase | SwitchStatement | TSAsExpression | TSDeclareMethod | TSEnumDeclaration | TSEnumMember | TSExportAssignment | TSExternalModuleReference | TSImportType | TSInstantiationExpression | TSLiteralType | TSMethodSignature | TSModuleDeclaration | TSNonNullExpression | TSPropertySignature | TSSatisfiesExpression | TSTypeAssertion | TaggedTemplateExpression | TemplateLiteral | ThrowStatement | TupleExpression | TypeCastExpression | UnaryExpression | UpdateExpression | VariableDeclarator | WhileStatement | WithStatement | YieldExpression;
    StringLiteralTypeAnnotation: ArrayTypeAnnotation | DeclareExportDeclaration | DeclareOpaqueType | DeclareTypeAlias | DeclaredPredicate | FunctionTypeAnnotation | FunctionTypeParam | IndexedAccessType | IntersectionTypeAnnotation | NullableTypeAnnotation | ObjectTypeCallProperty | ObjectTypeIndexer | ObjectTypeInternalSlot | ObjectTypeProperty | ObjectTypeSpreadProperty | OpaqueType | OptionalIndexedAccessType | TupleTypeAnnotation | TypeAlias | TypeAnnotation | TypeParameter | TypeParameterInstantiation | TypeofTypeAnnotation | UnionTypeAnnotation;
    StringTypeAnnotation: ArrayTypeAnnotation | DeclareExportDeclaration | DeclareOpaqueType | DeclareTypeAlias | DeclaredPredicate | FunctionTypeAnnotation | FunctionTypeParam | IndexedAccessType | IntersectionTypeAnnotation | NullableTypeAnnotation | ObjectTypeCallProperty | ObjectTypeIndexer | ObjectTypeInternalSlot | ObjectTypeProperty | ObjectTypeSpreadProperty | OpaqueType | OptionalIndexedAccessType | TupleTypeAnnotation | TypeAlias | TypeAnnotation | TypeParameter | TypeParameterInstantiation | TypeofTypeAnnotation | UnionTypeAnnotation;
    Super: ArrayExpression | ArrowFunctionExpression | AssignmentExpression | AssignmentPattern | AwaitExpression | BinaryExpression | BindExpression | CallExpression | ClassAccessorProperty | ClassDeclaration | ClassExpression | ClassMethod | ClassPrivateProperty | ClassProperty | ConditionalExpression | Decorator | DoWhileStatement | ExportDefaultDeclaration | ExpressionStatement | ForInStatement | ForOfStatement | ForStatement | IfStatement | ImportExpression | JSXExpressionContainer | JSXSpreadAttribute | JSXSpreadChild | LogicalExpression | MemberExpression | NewExpression | ObjectMethod | ObjectProperty | OptionalCallExpression | OptionalMemberExpression | ParenthesizedExpression | PipelineBareFunction | PipelineTopicExpression | ReturnStatement | SequenceExpression | SpreadElement | SwitchCase | SwitchStatement | TSAsExpression | TSDeclareMethod | TSEnumDeclaration | TSEnumMember | TSExportAssignment | TSImportType | TSInstantiationExpression | TSMethodSignature | TSNonNullExpression | TSPropertySignature | TSSatisfiesExpression | TSTypeAssertion | TaggedTemplateExpression | TemplateLiteral | ThrowStatement | TupleExpression | TypeCastExpression | UnaryExpression | UpdateExpression | VariableDeclarator | WhileStatement | WithStatement | YieldExpression;
    SwitchCase: SwitchStatement;
    SwitchStatement: BlockStatement | DoWhileStatement | ForInStatement | ForOfStatement | ForStatement | IfStatement | LabeledStatement | Program | StaticBlock | SwitchCase | TSModuleBlock | WhileStatement | WithStatement;
    SymbolTypeAnnotation: ArrayTypeAnnotation | DeclareExportDeclaration | DeclareOpaqueType | DeclareTypeAlias | DeclaredPredicate | FunctionTypeAnnotation | FunctionTypeParam | IndexedAccessType | IntersectionTypeAnnotation | NullableTypeAnnotation | ObjectTypeCallProperty | ObjectTypeIndexer | ObjectTypeInternalSlot | ObjectTypeProperty | ObjectTypeSpreadProperty | OpaqueType | OptionalIndexedAccessType | TupleTypeAnnotation | TypeAlias | TypeAnnotation | TypeParameter | TypeParameterInstantiation | TypeofTypeAnnotation | UnionTypeAnnotation;
    TSAnyKeyword: TSArrayType | TSAsExpression | TSConditionalType | TSIndexedAccessType | TSIntersectionType | TSMappedType | TSNamedTupleMember | TSOptionalType | TSParenthesizedType | TSRestType | TSSatisfiesExpression | TSTupleType | TSTypeAliasDeclaration | TSTypeAnnotation | TSTypeAssertion | TSTypeOperator | TSTypeParameter | TSTypeParameterInstantiation | TSUnionType | TemplateLiteral;
    TSArrayType: TSArrayType | TSAsExpression | TSConditionalType | TSIndexedAccessType | TSIntersectionType | TSMappedType | TSNamedTupleMember | TSOptionalType | TSParenthesizedType | TSRestType | TSSatisfiesExpression | TSTupleType | TSTypeAliasDeclaration | TSTypeAnnotation | TSTypeAssertion | TSTypeOperator | TSTypeParameter | TSTypeParameterInstantiation | TSUnionType | TemplateLiteral;
    TSAsExpression: ArrayExpression | ArrayPattern | ArrowFunctionExpression | AssignmentExpression | AssignmentPattern | AwaitExpression | BinaryExpression | BindExpression | CallExpression | ClassAccessorProperty | ClassDeclaration | ClassExpression | ClassMethod | ClassPrivateProperty | ClassProperty | ConditionalExpression | Decorator | DoWhileStatement | ExportDefaultDeclaration | ExpressionStatement | ForInStatement | ForOfStatement | ForStatement | IfStatement | ImportExpression | JSXExpressionContainer | JSXSpreadAttribute | JSXSpreadChild | LogicalExpression | MemberExpression | NewExpression | ObjectMethod | ObjectProperty | OptionalCallExpression | OptionalMemberExpression | ParenthesizedExpression | PipelineBareFunction | PipelineTopicExpression | RestElement | ReturnStatement | SequenceExpression | SpreadElement | SwitchCase | SwitchStatement | TSAsExpression | TSDeclareMethod | TSEnumDeclaration | TSEnumMember | TSExportAssignment | TSImportType | TSInstantiationExpression | TSMethodSignature | TSNonNullExpression | TSPropertySignature | TSSatisfiesExpression | TSTypeAssertion | TaggedTemplateExpression | TemplateLiteral | ThrowStatement | TupleExpression | TypeCastExpression | UnaryExpression | UpdateExpression | VariableDeclarator | WhileStatement | WithStatement | YieldExpression;
    TSBigIntKeyword: TSArrayType | TSAsExpression | TSConditionalType | TSIndexedAccessType | TSIntersectionType | TSMappedType | TSNamedTupleMember | TSOptionalType | TSParenthesizedType | TSRestType | TSSatisfiesExpression | TSTupleType | TSTypeAliasDeclaration | TSTypeAnnotation | TSTypeAssertion | TSTypeOperator | TSTypeParameter | TSTypeParameterInstantiation | TSUnionType | TemplateLiteral;
    TSBooleanKeyword: TSArrayType | TSAsExpression | TSConditionalType | TSIndexedAccessType | TSIntersectionType | TSMappedType | TSNamedTupleMember | TSOptionalType | TSParenthesizedType | TSRestType | TSSatisfiesExpression | TSTupleType | TSTypeAliasDeclaration | TSTypeAnnotation | TSTypeAssertion | TSTypeOperator | TSTypeParameter | TSTypeParameterInstantiation | TSUnionType | TemplateLiteral;
    TSCallSignatureDeclaration: TSInterfaceBody | TSTypeLiteral;
    TSConditionalType: TSArrayType | TSAsExpression | TSConditionalType | TSIndexedAccessType | TSIntersectionType | TSMappedType | TSNamedTupleMember | TSOptionalType | TSParenthesizedType | TSRestType | TSSatisfiesExpression | TSTupleType | TSTypeAliasDeclaration | TSTypeAnnotation | TSTypeAssertion | TSTypeOperator | TSTypeParameter | TSTypeParameterInstantiation | TSUnionType | TemplateLiteral;
    TSConstructSignatureDeclaration: TSInterfaceBody | TSTypeLiteral;
    TSConstructorType: TSArrayType | TSAsExpression | TSConditionalType | TSIndexedAccessType | TSIntersectionType | TSMappedType | TSNamedTupleMember | TSOptionalType | TSParenthesizedType | TSRestType | TSSatisfiesExpression | TSTupleType | TSTypeAliasDeclaration | TSTypeAnnotation | TSTypeAssertion | TSTypeOperator | TSTypeParameter | TSTypeParameterInstantiation | TSUnionType | TemplateLiteral;
    TSDeclareFunction: BlockStatement | DoWhileStatement | ExportDefaultDeclaration | ExportNamedDeclaration | ForInStatement | ForOfStatement | ForStatement | IfStatement | LabeledStatement | Program | StaticBlock | SwitchCase | TSModuleBlock | WhileStatement | WithStatement;
    TSDeclareMethod: ClassBody;
    TSEnumDeclaration: BlockStatement | DoWhileStatement | ExportNamedDeclaration | ForInStatement | ForOfStatement | ForStatement | IfStatement | LabeledStatement | Program | StaticBlock | SwitchCase | TSModuleBlock | WhileStatement | WithStatement;
    TSEnumMember: TSEnumDeclaration;
    TSExportAssignment: BlockStatement | DoWhileStatement | ForInStatement | ForOfStatement | ForStatement | IfStatement | LabeledStatement | Program | StaticBlock | SwitchCase | TSModuleBlock | WhileStatement | WithStatement;
    TSExpressionWithTypeArguments: ClassDeclaration | ClassExpression | TSArrayType | TSAsExpression | TSConditionalType | TSIndexedAccessType | TSInterfaceDeclaration | TSIntersectionType | TSMappedType | TSNamedTupleMember | TSOptionalType | TSParenthesizedType | TSRestType | TSSatisfiesExpression | TSTupleType | TSTypeAliasDeclaration | TSTypeAnnotation | TSTypeAssertion | TSTypeOperator | TSTypeParameter | TSTypeParameterInstantiation | TSUnionType | TemplateLiteral;
    TSExternalModuleReference: TSImportEqualsDeclaration;
    TSFunctionType: TSArrayType | TSAsExpression | TSConditionalType | TSIndexedAccessType | TSIntersectionType | TSMappedType | TSNamedTupleMember | TSOptionalType | TSParenthesizedType | TSRestType | TSSatisfiesExpression | TSTupleType | TSTypeAliasDeclaration | TSTypeAnnotation | TSTypeAssertion | TSTypeOperator | TSTypeParameter | TSTypeParameterInstantiation | TSUnionType | TemplateLiteral;
    TSImportEqualsDeclaration: BlockStatement | DoWhileStatement | ForInStatement | ForOfStatement | ForStatement | IfStatement | LabeledStatement | Program | StaticBlock | SwitchCase | TSModuleBlock | WhileStatement | WithStatement;
    TSImportType: TSArrayType | TSAsExpression | TSConditionalType | TSIndexedAccessType | TSIntersectionType | TSMappedType | TSNamedTupleMember | TSOptionalType | TSParenthesizedType | TSRestType | TSSatisfiesExpression | TSTupleType | TSTypeAliasDeclaration | TSTypeAnnotation | TSTypeAssertion | TSTypeOperator | TSTypeParameter | TSTypeParameterInstantiation | TSTypeQuery | TSUnionType | TemplateLiteral;
    TSIndexSignature: ClassBody | TSInterfaceBody | TSTypeLiteral;
    TSIndexedAccessType: TSArrayType | TSAsExpression | TSConditionalType | TSIndexedAccessType | TSIntersectionType | TSMappedType | TSNamedTupleMember | TSOptionalType | TSParenthesizedType | TSRestType | TSSatisfiesExpression | TSTupleType | TSTypeAliasDeclaration | TSTypeAnnotation | TSTypeAssertion | TSTypeOperator | TSTypeParameter | TSTypeParameterInstantiation | TSUnionType | TemplateLiteral;
    TSInferType: TSArrayType | TSAsExpression | TSConditionalType | TSIndexedAccessType | TSIntersectionType | TSMappedType | TSNamedTupleMember | TSOptionalType | TSParenthesizedType | TSRestType | TSSatisfiesExpression | TSTupleType | TSTypeAliasDeclaration | TSTypeAnnotation | TSTypeAssertion | TSTypeOperator | TSTypeParameter | TSTypeParameterInstantiation | TSUnionType | TemplateLiteral;
    TSInstantiationExpression: ArrayExpression | ArrowFunctionExpression | AssignmentExpression | AssignmentPattern | AwaitExpression | BinaryExpression | BindExpression | CallExpression | ClassAccessorProperty | ClassDeclaration | ClassExpression | ClassMethod | ClassPrivateProperty | ClassProperty | ConditionalExpression | Decorator | DoWhileStatement | ExportDefaultDeclaration | ExpressionStatement | ForInStatement | ForOfStatement | ForStatement | IfStatement | ImportExpression | JSXExpressionContainer | JSXSpreadAttribute | JSXSpreadChild | LogicalExpression | MemberExpression | NewExpression | ObjectMethod | ObjectProperty | OptionalCallExpression | OptionalMemberExpression | ParenthesizedExpression | PipelineBareFunction | PipelineTopicExpression | ReturnStatement | SequenceExpression | SpreadElement | SwitchCase | SwitchStatement | TSAsExpression | TSDeclareMethod | TSEnumDeclaration | TSEnumMember | TSExportAssignment | TSImportType | TSInstantiationExpression | TSMethodSignature | TSNonNullExpression | TSPropertySignature | TSSatisfiesExpression | TSTypeAssertion | TaggedTemplateExpression | TemplateLiteral | ThrowStatement | TupleExpression | TypeCastExpression | UnaryExpression | UpdateExpression | VariableDeclarator | WhileStatement | WithStatement | YieldExpression;
    TSInterfaceBody: TSInterfaceDeclaration;
    TSInterfaceDeclaration: BlockStatement | DoWhileStatement | ExportNamedDeclaration | ForInStatement | ForOfStatement | ForStatement | IfStatement | LabeledStatement | Program | StaticBlock | SwitchCase | TSModuleBlock | WhileStatement | WithStatement;
    TSIntersectionType: TSArrayType | TSAsExpression | TSConditionalType | TSIndexedAccessType | TSIntersectionType | TSMappedType | TSNamedTupleMember | TSOptionalType | TSParenthesizedType | TSRestType | TSSatisfiesExpression | TSTupleType | TSTypeAliasDeclaration | TSTypeAnnotation | TSTypeAssertion | TSTypeOperator | TSTypeParameter | TSTypeParameterInstantiation | TSUnionType | TemplateLiteral;
    TSIntrinsicKeyword: TSArrayType | TSAsExpression | TSConditionalType | TSIndexedAccessType | TSIntersectionType | TSMappedType | TSNamedTupleMember | TSOptionalType | TSParenthesizedType | TSRestType | TSSatisfiesExpression | TSTupleType | TSTypeAliasDeclaration | TSTypeAnnotation | TSTypeAssertion | TSTypeOperator | TSTypeParameter | TSTypeParameterInstantiation | TSUnionType | TemplateLiteral;
    TSLiteralType: TSArrayType | TSAsExpression | TSConditionalType | TSIndexedAccessType | TSIntersectionType | TSMappedType | TSNamedTupleMember | TSOptionalType | TSParenthesizedType | TSRestType | TSSatisfiesExpression | TSTupleType | TSTypeAliasDeclaration | TSTypeAnnotation | TSTypeAssertion | TSTypeOperator | TSTypeParameter | TSTypeParameterInstantiation | TSUnionType | TemplateLiteral;
    TSMappedType: TSArrayType | TSAsExpression | TSConditionalType | TSIndexedAccessType | TSIntersectionType | TSMappedType | TSNamedTupleMember | TSOptionalType | TSParenthesizedType | TSRestType | TSSatisfiesExpression | TSTupleType | TSTypeAliasDeclaration | TSTypeAnnotation | TSTypeAssertion | TSTypeOperator | TSTypeParameter | TSTypeParameterInstantiation | TSUnionType | TemplateLiteral;
    TSMethodSignature: TSInterfaceBody | TSTypeLiteral;
    TSModuleBlock: TSModuleDeclaration;
    TSModuleDeclaration: BlockStatement | DoWhileStatement | ExportNamedDeclaration | ForInStatement | ForOfStatement | ForStatement | IfStatement | LabeledStatement | Program | StaticBlock | SwitchCase | TSModuleBlock | TSModuleDeclaration | WhileStatement | WithStatement;
    TSNamedTupleMember: TSTupleType;
    TSNamespaceExportDeclaration: BlockStatement | DoWhileStatement | ForInStatement | ForOfStatement | ForStatement | IfStatement | LabeledStatement | Program | StaticBlock | SwitchCase | TSModuleBlock | WhileStatement | WithStatement;
    TSNeverKeyword: TSArrayType | TSAsExpression | TSConditionalType | TSIndexedAccessType | TSIntersectionType | TSMappedType | TSNamedTupleMember | TSOptionalType | TSParenthesizedType | TSRestType | TSSatisfiesExpression | TSTupleType | TSTypeAliasDeclaration | TSTypeAnnotation | TSTypeAssertion | TSTypeOperator | TSTypeParameter | TSTypeParameterInstantiation | TSUnionType | TemplateLiteral;
    TSNonNullExpression: ArrayExpression | ArrayPattern | ArrowFunctionExpression | AssignmentExpression | AssignmentPattern | AwaitExpression | BinaryExpression | BindExpression | CallExpression | ClassAccessorProperty | ClassDeclaration | ClassExpression | ClassMethod | ClassPrivateProperty | ClassProperty | ConditionalExpression | Decorator | DoWhileStatement | ExportDefaultDeclaration | ExpressionStatement | ForInStatement | ForOfStatement | ForStatement | IfStatement | ImportExpression | JSXExpressionContainer | JSXSpreadAttribute | JSXSpreadChild | LogicalExpression | MemberExpression | NewExpression | ObjectMethod | ObjectProperty | OptionalCallExpression | OptionalMemberExpression | ParenthesizedExpression | PipelineBareFunction | PipelineTopicExpression | RestElement | ReturnStatement | SequenceExpression | SpreadElement | SwitchCase | SwitchStatement | TSAsExpression | TSDeclareMethod | TSEnumDeclaration | TSEnumMember | TSExportAssignment | TSImportType | TSInstantiationExpression | TSMethodSignature | TSNonNullExpression | TSPropertySignature | TSSatisfiesExpression | TSTypeAssertion | TaggedTemplateExpression | TemplateLiteral | ThrowStatement | TupleExpression | TypeCastExpression | UnaryExpression | UpdateExpression | VariableDeclarator | WhileStatement | WithStatement | YieldExpression;
    TSNullKeyword: TSArrayType | TSAsExpression | TSConditionalType | TSIndexedAccessType | TSIntersectionType | TSMappedType | TSNamedTupleMember | TSOptionalType | TSParenthesizedType | TSRestType | TSSatisfiesExpression | TSTupleType | TSTypeAliasDeclaration | TSTypeAnnotation | TSTypeAssertion | TSTypeOperator | TSTypeParameter | TSTypeParameterInstantiation | TSUnionType | TemplateLiteral;
    TSNumberKeyword: TSArrayType | TSAsExpression | TSConditionalType | TSIndexedAccessType | TSIntersectionType | TSMappedType | TSNamedTupleMember | TSOptionalType | TSParenthesizedType | TSRestType | TSSatisfiesExpression | TSTupleType | TSTypeAliasDeclaration | TSTypeAnnotation | TSTypeAssertion | TSTypeOperator | TSTypeParameter | TSTypeParameterInstantiation | TSUnionType | TemplateLiteral;
    TSObjectKeyword: TSArrayType | TSAsExpression | TSConditionalType | TSIndexedAccessType | TSIntersectionType | TSMappedType | TSNamedTupleMember | TSOptionalType | TSParenthesizedType | TSRestType | TSSatisfiesExpression | TSTupleType | TSTypeAliasDeclaration | TSTypeAnnotation | TSTypeAssertion | TSTypeOperator | TSTypeParameter | TSTypeParameterInstantiation | TSUnionType | TemplateLiteral;
    TSOptionalType: TSArrayType | TSAsExpression | TSConditionalType | TSIndexedAccessType | TSIntersectionType | TSMappedType | TSNamedTupleMember | TSOptionalType | TSParenthesizedType | TSRestType | TSSatisfiesExpression | TSTupleType | TSTypeAliasDeclaration | TSTypeAnnotation | TSTypeAssertion | TSTypeOperator | TSTypeParameter | TSTypeParameterInstantiation | TSUnionType | TemplateLiteral;
    TSParameterProperty: ArrayPattern | AssignmentExpression | ClassMethod | ClassPrivateMethod | ForInStatement | ForOfStatement | RestElement | TSDeclareMethod | VariableDeclarator;
    TSParenthesizedType: TSArrayType | TSAsExpression | TSConditionalType | TSIndexedAccessType | TSIntersectionType | TSMappedType | TSNamedTupleMember | TSOptionalType | TSParenthesizedType | TSRestType | TSSatisfiesExpression | TSTupleType | TSTypeAliasDeclaration | TSTypeAnnotation | TSTypeAssertion | TSTypeOperator | TSTypeParameter | TSTypeParameterInstantiation | TSUnionType | TemplateLiteral;
    TSPropertySignature: TSInterfaceBody | TSTypeLiteral;
    TSQualifiedName: TSExpressionWithTypeArguments | TSImportEqualsDeclaration | TSImportType | TSQualifiedName | TSTypeQuery | TSTypeReference;
    TSRestType: TSArrayType | TSAsExpression | TSConditionalType | TSIndexedAccessType | TSIntersectionType | TSMappedType | TSNamedTupleMember | TSOptionalType | TSParenthesizedType | TSRestType | TSSatisfiesExpression | TSTupleType | TSTypeAliasDeclaration | TSTypeAnnotation | TSTypeAssertion | TSTypeOperator | TSTypeParameter | TSTypeParameterInstantiation | TSUnionType | TemplateLiteral;
    TSSatisfiesExpression: ArrayExpression | ArrayPattern | ArrowFunctionExpression | AssignmentExpression | AssignmentPattern | AwaitExpression | BinaryExpression | BindExpression | CallExpression | ClassAccessorProperty | ClassDeclaration | ClassExpression | ClassMethod | ClassPrivateProperty | ClassProperty | ConditionalExpression | Decorator | DoWhileStatement | ExportDefaultDeclaration | ExpressionStatement | ForInStatement | ForOfStatement | ForStatement | IfStatement | ImportExpression | JSXExpressionContainer | JSXSpreadAttribute | JSXSpreadChild | LogicalExpression | MemberExpression | NewExpression | ObjectMethod | ObjectProperty | OptionalCallExpression | OptionalMemberExpression | ParenthesizedExpression | PipelineBareFunction | PipelineTopicExpression | RestElement | ReturnStatement | SequenceExpression | SpreadElement | SwitchCase | SwitchStatement | TSAsExpression | TSDeclareMethod | TSEnumDeclaration | TSEnumMember | TSExportAssignment | TSImportType | TSInstantiationExpression | TSMethodSignature | TSNonNullExpression | TSPropertySignature | TSSatisfiesExpression | TSTypeAssertion | TaggedTemplateExpression | TemplateLiteral | ThrowStatement | TupleExpression | TypeCastExpression | UnaryExpression | UpdateExpression | VariableDeclarator | WhileStatement | WithStatement | YieldExpression;
    TSStringKeyword: TSArrayType | TSAsExpression | TSConditionalType | TSIndexedAccessType | TSIntersectionType | TSMappedType | TSNamedTupleMember | TSOptionalType | TSParenthesizedType | TSRestType | TSSatisfiesExpression | TSTupleType | TSTypeAliasDeclaration | TSTypeAnnotation | TSTypeAssertion | TSTypeOperator | TSTypeParameter | TSTypeParameterInstantiation | TSUnionType | TemplateLiteral;
    TSSymbolKeyword: TSArrayType | TSAsExpression | TSConditionalType | TSIndexedAccessType | TSIntersectionType | TSMappedType | TSNamedTupleMember | TSOptionalType | TSParenthesizedType | TSRestType | TSSatisfiesExpression | TSTupleType | TSTypeAliasDeclaration | TSTypeAnnotation | TSTypeAssertion | TSTypeOperator | TSTypeParameter | TSTypeParameterInstantiation | TSUnionType | TemplateLiteral;
    TSThisType: TSArrayType | TSAsExpression | TSConditionalType | TSIndexedAccessType | TSIntersectionType | TSMappedType | TSNamedTupleMember | TSOptionalType | TSParenthesizedType | TSRestType | TSSatisfiesExpression | TSTupleType | TSTypeAliasDeclaration | TSTypeAnnotation | TSTypeAssertion | TSTypeOperator | TSTypeParameter | TSTypeParameterInstantiation | TSTypePredicate | TSUnionType | TemplateLiteral;
    TSTupleType: TSArrayType | TSAsExpression | TSConditionalType | TSIndexedAccessType | TSIntersectionType | TSMappedType | TSNamedTupleMember | TSOptionalType | TSParenthesizedType | TSRestType | TSSatisfiesExpression | TSTupleType | TSTypeAliasDeclaration | TSTypeAnnotation | TSTypeAssertion | TSTypeOperator | TSTypeParameter | TSTypeParameterInstantiation | TSUnionType | TemplateLiteral;
    TSTypeAliasDeclaration: BlockStatement | DoWhileStatement | ExportNamedDeclaration | ForInStatement | ForOfStatement | ForStatement | IfStatement | LabeledStatement | Program | StaticBlock | SwitchCase | TSModuleBlock | WhileStatement | WithStatement;
    TSTypeAnnotation: ArrayPattern | ArrowFunctionExpression | AssignmentPattern | ClassAccessorProperty | ClassMethod | ClassPrivateMethod | ClassPrivateProperty | ClassProperty | FunctionDeclaration | FunctionExpression | Identifier | ObjectMethod | ObjectPattern | RestElement | TSCallSignatureDeclaration | TSConstructSignatureDeclaration | TSConstructorType | TSDeclareFunction | TSDeclareMethod | TSFunctionType | TSIndexSignature | TSMethodSignature | TSPropertySignature | TSTypePredicate;
    TSTypeAssertion: ArrayExpression | ArrayPattern | ArrowFunctionExpression | AssignmentExpression | AssignmentPattern | AwaitExpression | BinaryExpression | BindExpression | CallExpression | ClassAccessorProperty | ClassDeclaration | ClassExpression | ClassMethod | ClassPrivateProperty | ClassProperty | ConditionalExpression | Decorator | DoWhileStatement | ExportDefaultDeclaration | ExpressionStatement | ForInStatement | ForOfStatement | ForStatement | IfStatement | ImportExpression | JSXExpressionContainer | JSXSpreadAttribute | JSXSpreadChild | LogicalExpression | MemberExpression | NewExpression | ObjectMethod | ObjectProperty | OptionalCallExpression | OptionalMemberExpression | ParenthesizedExpression | PipelineBareFunction | PipelineTopicExpression | RestElement | ReturnStatement | SequenceExpression | SpreadElement | SwitchCase | SwitchStatement | TSAsExpression | TSDeclareMethod | TSEnumDeclaration | TSEnumMember | TSExportAssignment | TSImportType | TSInstantiationExpression | TSMethodSignature | TSNonNullExpression | TSPropertySignature | TSSatisfiesExpression | TSTypeAssertion | TaggedTemplateExpression | TemplateLiteral | ThrowStatement | TupleExpression | TypeCastExpression | UnaryExpression | UpdateExpression | VariableDeclarator | WhileStatement | WithStatement | YieldExpression;
    TSTypeLiteral: TSArrayType | TSAsExpression | TSConditionalType | TSIndexedAccessType | TSIntersectionType | TSMappedType | TSNamedTupleMember | TSOptionalType | TSParenthesizedType | TSRestType | TSSatisfiesExpression | TSTupleType | TSTypeAliasDeclaration | TSTypeAnnotation | TSTypeAssertion | TSTypeOperator | TSTypeParameter | TSTypeParameterInstantiation | TSUnionType | TemplateLiteral;
    TSTypeOperator: TSArrayType | TSAsExpression | TSConditionalType | TSIndexedAccessType | TSIntersectionType | TSMappedType | TSNamedTupleMember | TSOptionalType | TSParenthesizedType | TSRestType | TSSatisfiesExpression | TSTupleType | TSTypeAliasDeclaration | TSTypeAnnotation | TSTypeAssertion | TSTypeOperator | TSTypeParameter | TSTypeParameterInstantiation | TSUnionType | TemplateLiteral;
    TSTypeParameter: TSInferType | TSMappedType | TSTypeParameterDeclaration;
    TSTypeParameterDeclaration: ArrowFunctionExpression | ClassDeclaration | ClassExpression | ClassMethod | ClassPrivateMethod | FunctionDeclaration | FunctionExpression | ObjectMethod | TSCallSignatureDeclaration | TSConstructSignatureDeclaration | TSConstructorType | TSDeclareFunction | TSDeclareMethod | TSFunctionType | TSInterfaceDeclaration | TSMethodSignature | TSTypeAliasDeclaration;
    TSTypeParameterInstantiation: CallExpression | ClassDeclaration | ClassExpression | JSXOpeningElement | NewExpression | OptionalCallExpression | TSExpressionWithTypeArguments | TSImportType | TSInstantiationExpression | TSTypeQuery | TSTypeReference | TaggedTemplateExpression;
    TSTypePredicate: TSArrayType | TSAsExpression | TSConditionalType | TSIndexedAccessType | TSIntersectionType | TSMappedType | TSNamedTupleMember | TSOptionalType | TSParenthesizedType | TSRestType | TSSatisfiesExpression | TSTupleType | TSTypeAliasDeclaration | TSTypeAnnotation | TSTypeAssertion | TSTypeOperator | TSTypeParameter | TSTypeParameterInstantiation | TSUnionType | TemplateLiteral;
    TSTypeQuery: TSArrayType | TSAsExpression | TSConditionalType | TSIndexedAccessType | TSIntersectionType | TSMappedType | TSNamedTupleMember | TSOptionalType | TSParenthesizedType | TSRestType | TSSatisfiesExpression | TSTupleType | TSTypeAliasDeclaration | TSTypeAnnotation | TSTypeAssertion | TSTypeOperator | TSTypeParameter | TSTypeParameterInstantiation | TSUnionType | TemplateLiteral;
    TSTypeReference: TSArrayType | TSAsExpression | TSConditionalType | TSIndexedAccessType | TSIntersectionType | TSMappedType | TSNamedTupleMember | TSOptionalType | TSParenthesizedType | TSRestType | TSSatisfiesExpression | TSTupleType | TSTypeAliasDeclaration | TSTypeAnnotation | TSTypeAssertion | TSTypeOperator | TSTypeParameter | TSTypeParameterInstantiation | TSUnionType | TemplateLiteral;
    TSUndefinedKeyword: TSArrayType | TSAsExpression | TSConditionalType | TSIndexedAccessType | TSIntersectionType | TSMappedType | TSNamedTupleMember | TSOptionalType | TSParenthesizedType | TSRestType | TSSatisfiesExpression | TSTupleType | TSTypeAliasDeclaration | TSTypeAnnotation | TSTypeAssertion | TSTypeOperator | TSTypeParameter | TSTypeParameterInstantiation | TSUnionType | TemplateLiteral;
    TSUnionType: TSArrayType | TSAsExpression | TSConditionalType | TSIndexedAccessType | TSIntersectionType | TSMappedType | TSNamedTupleMember | TSOptionalType | TSParenthesizedType | TSRestType | TSSatisfiesExpression | TSTupleType | TSTypeAliasDeclaration | TSTypeAnnotation | TSTypeAssertion | TSTypeOperator | TSTypeParameter | TSTypeParameterInstantiation | TSUnionType | TemplateLiteral;
    TSUnknownKeyword: TSArrayType | TSAsExpression | TSConditionalType | TSIndexedAccessType | TSIntersectionType | TSMappedType | TSNamedTupleMember | TSOptionalType | TSParenthesizedType | TSRestType | TSSatisfiesExpression | TSTupleType | TSTypeAliasDeclaration | TSTypeAnnotation | TSTypeAssertion | TSTypeOperator | TSTypeParameter | TSTypeParameterInstantiation | TSUnionType | TemplateLiteral;
    TSVoidKeyword: TSArrayType | TSAsExpression | TSConditionalType | TSIndexedAccessType | TSIntersectionType | TSMappedType | TSNamedTupleMember | TSOptionalType | TSParenthesizedType | TSRestType | TSSatisfiesExpression | TSTupleType | TSTypeAliasDeclaration | TSTypeAnnotation | TSTypeAssertion | TSTypeOperator | TSTypeParameter | TSTypeParameterInstantiation | TSUnionType | TemplateLiteral;
    TaggedTemplateExpression: ArrayExpression | ArrowFunctionExpression | AssignmentExpression | AssignmentPattern | AwaitExpression | BinaryExpression | BindExpression | CallExpression | ClassAccessorProperty | ClassDeclaration | ClassExpression | ClassMethod | ClassPrivateProperty | ClassProperty | ConditionalExpression | Decorator | DoWhileStatement | ExportDefaultDeclaration | ExpressionStatement | ForInStatement | ForOfStatement | ForStatement | IfStatement | ImportExpression | JSXExpressionContainer | JSXSpreadAttribute | JSXSpreadChild | LogicalExpression | MemberExpression | NewExpression | ObjectMethod | ObjectProperty | OptionalCallExpression | OptionalMemberExpression | ParenthesizedExpression | PipelineBareFunction | PipelineTopicExpression | ReturnStatement | SequenceExpression | SpreadElement | SwitchCase | SwitchStatement | TSAsExpression | TSDeclareMethod | TSEnumDeclaration | TSEnumMember | TSExportAssignment | TSImportType | TSInstantiationExpression | TSMethodSignature | TSNonNullExpression | TSPropertySignature | TSSatisfiesExpression | TSTypeAssertion | TaggedTemplateExpression | TemplateLiteral | ThrowStatement | TupleExpression | TypeCastExpression | UnaryExpression | UpdateExpression | VariableDeclarator | WhileStatement | WithStatement | YieldExpression;
    TemplateElement: TemplateLiteral;
    TemplateLiteral: ArrayExpression | ArrowFunctionExpression | AssignmentExpression | AssignmentPattern | AwaitExpression | BinaryExpression | BindExpression | CallExpression | ClassAccessorProperty | ClassDeclaration | ClassExpression | ClassMethod | ClassPrivateProperty | ClassProperty | ConditionalExpression | Decorator | DoWhileStatement | ExportDefaultDeclaration | ExpressionStatement | ForInStatement | ForOfStatement | ForStatement | IfStatement | ImportExpression | JSXExpressionContainer | JSXSpreadAttribute | JSXSpreadChild | LogicalExpression | MemberExpression | NewExpression | ObjectMethod | ObjectProperty | OptionalCallExpression | OptionalMemberExpression | ParenthesizedExpression | PipelineBareFunction | PipelineTopicExpression | ReturnStatement | SequenceExpression | SpreadElement | SwitchCase | SwitchStatement | TSAsExpression | TSDeclareMethod | TSEnumDeclaration | TSEnumMember | TSExportAssignment | TSImportType | TSInstantiationExpression | TSLiteralType | TSMethodSignature | TSNonNullExpression | TSPropertySignature | TSSatisfiesExpression | TSTypeAssertion | TaggedTemplateExpression | TemplateLiteral | ThrowStatement | TupleExpression | TypeCastExpression | UnaryExpression | UpdateExpression | VariableDeclarator | WhileStatement | WithStatement | YieldExpression;
    ThisExpression: ArrayExpression | ArrowFunctionExpression | AssignmentExpression | AssignmentPattern | AwaitExpression | BinaryExpression | BindExpression | CallExpression | ClassAccessorProperty | ClassDeclaration | ClassExpression | ClassMethod | ClassPrivateProperty | ClassProperty | ConditionalExpression | Decorator | DoWhileStatement | ExportDefaultDeclaration | ExpressionStatement | ForInStatement | ForOfStatement | ForStatement | IfStatement | ImportExpression | JSXExpressionContainer | JSXSpreadAttribute | JSXSpreadChild | LogicalExpression | MemberExpression | NewExpression | ObjectMethod | ObjectProperty | OptionalCallExpression | OptionalMemberExpression | ParenthesizedExpression | PipelineBareFunction | PipelineTopicExpression | ReturnStatement | SequenceExpression | SpreadElement | SwitchCase | SwitchStatement | TSAsExpression | TSDeclareMethod | TSEnumDeclaration | TSEnumMember | TSExportAssignment | TSImportType | TSInstantiationExpression | TSMethodSignature | TSNonNullExpression | TSPropertySignature | TSSatisfiesExpression | TSTypeAssertion | TaggedTemplateExpression | TemplateLiteral | ThrowStatement | TupleExpression | TypeCastExpression | UnaryExpression | UpdateExpression | VariableDeclarator | WhileStatement | WithStatement | YieldExpression;
    ThisTypeAnnotation: ArrayTypeAnnotation | DeclareExportDeclaration | DeclareOpaqueType | DeclareTypeAlias | DeclaredPredicate | FunctionTypeAnnotation | FunctionTypeParam | IndexedAccessType | IntersectionTypeAnnotation | NullableTypeAnnotation | ObjectTypeCallProperty | ObjectTypeIndexer | ObjectTypeInternalSlot | ObjectTypeProperty | ObjectTypeSpreadProperty | OpaqueType | OptionalIndexedAccessType | TupleTypeAnnotation | TypeAlias | TypeAnnotation | TypeParameter | TypeParameterInstantiation | TypeofTypeAnnotation | UnionTypeAnnotation;
    ThrowStatement: BlockStatement | DoWhileStatement | ForInStatement | ForOfStatement | ForStatement | IfStatement | LabeledStatement | Program | StaticBlock | SwitchCase | TSModuleBlock | WhileStatement | WithStatement;
    TopicReference: ArrayExpression | ArrowFunctionExpression | AssignmentExpression | AssignmentPattern | AwaitExpression | BinaryExpression | BindExpression | CallExpression | ClassAccessorProperty | ClassDeclaration | ClassExpression | ClassMethod | ClassPrivateProperty | ClassProperty | ConditionalExpression | Decorator | DoWhileStatement | ExportDefaultDeclaration | ExpressionStatement | ForInStatement | ForOfStatement | ForStatement | IfStatement | ImportExpression | JSXExpressionContainer | JSXSpreadAttribute | JSXSpreadChild | LogicalExpression | MemberExpression | NewExpression | ObjectMethod | ObjectProperty | OptionalCallExpression | OptionalMemberExpression | ParenthesizedExpression | PipelineBareFunction | PipelineTopicExpression | ReturnStatement | SequenceExpression | SpreadElement | SwitchCase | SwitchStatement | TSAsExpression | TSDeclareMethod | TSEnumDeclaration | TSEnumMember | TSExportAssignment | TSImportType | TSInstantiationExpression | TSMethodSignature | TSNonNullExpression | TSPropertySignature | TSSatisfiesExpression | TSTypeAssertion | TaggedTemplateExpression | TemplateLiteral | ThrowStatement | TupleExpression | TypeCastExpression | UnaryExpression | UpdateExpression | VariableDeclarator | WhileStatement | WithStatement | YieldExpression;
    TryStatement: BlockStatement | DoWhileStatement | ForInStatement | ForOfStatement | ForStatement | IfStatement | LabeledStatement | Program | StaticBlock | SwitchCase | TSModuleBlock | WhileStatement | WithStatement;
    TupleExpression: ArrayExpression | ArrowFunctionExpression | AssignmentExpression | AssignmentPattern | AwaitExpression | BinaryExpression | BindExpression | CallExpression | ClassAccessorProperty | ClassDeclaration | ClassExpression | ClassMethod | ClassPrivateProperty | ClassProperty | ConditionalExpression | Decorator | DoWhileStatement | ExportDefaultDeclaration | ExpressionStatement | ForInStatement | ForOfStatement | ForStatement | IfStatement | ImportExpression | JSXExpressionContainer | JSXSpreadAttribute | JSXSpreadChild | LogicalExpression | MemberExpression | NewExpression | ObjectMethod | ObjectProperty | OptionalCallExpression | OptionalMemberExpression | ParenthesizedExpression | PipelineBareFunction | PipelineTopicExpression | ReturnStatement | SequenceExpression | SpreadElement | SwitchCase | SwitchStatement | TSAsExpression | TSDeclareMethod | TSEnumDeclaration | TSEnumMember | TSExportAssignment | TSImportType | TSInstantiationExpression | TSMethodSignature | TSNonNullExpression | TSPropertySignature | TSSatisfiesExpression | TSTypeAssertion | TaggedTemplateExpression | TemplateLiteral | ThrowStatement | TupleExpression | TypeCastExpression | UnaryExpression | UpdateExpression | VariableDeclarator | WhileStatement | WithStatement | YieldExpression;
    TupleTypeAnnotation: ArrayTypeAnnotation | DeclareExportDeclaration | DeclareOpaqueType | DeclareTypeAlias | DeclaredPredicate | FunctionTypeAnnotation | FunctionTypeParam | IndexedAccessType | IntersectionTypeAnnotation | NullableTypeAnnotation | ObjectTypeCallProperty | ObjectTypeIndexer | ObjectTypeInternalSlot | ObjectTypeProperty | ObjectTypeSpreadProperty | OpaqueType | OptionalIndexedAccessType | TupleTypeAnnotation | TypeAlias | TypeAnnotation | TypeParameter | TypeParameterInstantiation | TypeofTypeAnnotation | UnionTypeAnnotation;
    TypeAlias: BlockStatement | DeclareExportDeclaration | DeclaredPredicate | DoWhileStatement | ExportNamedDeclaration | ForInStatement | ForOfStatement | ForStatement | IfStatement | LabeledStatement | Program | StaticBlock | SwitchCase | TSModuleBlock | WhileStatement | WithStatement;
    TypeAnnotation: ArrayPattern | ArrowFunctionExpression | AssignmentPattern | ClassAccessorProperty | ClassMethod | ClassPrivateMethod | ClassPrivateProperty | ClassProperty | DeclareExportDeclaration | DeclareModuleExports | DeclaredPredicate | FunctionDeclaration | FunctionExpression | Identifier | ObjectMethod | ObjectPattern | RestElement | TypeCastExpression | TypeParameter;
    TypeCastExpression: ArrayExpression | ArrowFunctionExpression | AssignmentExpression | AssignmentPattern | AwaitExpression | BinaryExpression | BindExpression | CallExpression | ClassAccessorProperty | ClassDeclaration | ClassExpression | ClassMethod | ClassPrivateProperty | ClassProperty | ConditionalExpression | DeclareExportDeclaration | DeclaredPredicate | Decorator | DoWhileStatement | ExportDefaultDeclaration | ExpressionStatement | ForInStatement | ForOfStatement | ForStatement | IfStatement | ImportExpression | JSXExpressionContainer | JSXSpreadAttribute | JSXSpreadChild | LogicalExpression | MemberExpression | NewExpression | ObjectMethod | ObjectProperty | OptionalCallExpression | OptionalMemberExpression | ParenthesizedExpression | PipelineBareFunction | PipelineTopicExpression | ReturnStatement | SequenceExpression | SpreadElement | SwitchCase | SwitchStatement | TSAsExpression | TSDeclareMethod | TSEnumDeclaration | TSEnumMember | TSExportAssignment | TSImportType | TSInstantiationExpression | TSMethodSignature | TSNonNullExpression | TSPropertySignature | TSSatisfiesExpression | TSTypeAssertion | TaggedTemplateExpression | TemplateLiteral | ThrowStatement | TupleExpression | TypeCastExpression | UnaryExpression | UpdateExpression | VariableDeclarator | WhileStatement | WithStatement | YieldExpression;
    TypeParameter: DeclareExportDeclaration | DeclaredPredicate | TypeParameterDeclaration;
    TypeParameterDeclaration: ArrowFunctionExpression | ClassDeclaration | ClassExpression | ClassMethod | ClassPrivateMethod | DeclareClass | DeclareExportDeclaration | DeclareInterface | DeclareOpaqueType | DeclareTypeAlias | DeclaredPredicate | FunctionDeclaration | FunctionExpression | FunctionTypeAnnotation | InterfaceDeclaration | ObjectMethod | OpaqueType | TypeAlias;
    TypeParameterInstantiation: CallExpression | ClassDeclaration | ClassExpression | ClassImplements | DeclareExportDeclaration | DeclaredPredicate | GenericTypeAnnotation | InterfaceExtends | JSXOpeningElement | NewExpression | OptionalCallExpression | TaggedTemplateExpression;
    TypeofTypeAnnotation: ArrayTypeAnnotation | DeclareExportDeclaration | DeclareOpaqueType | DeclareTypeAlias | DeclaredPredicate | FunctionTypeAnnotation | FunctionTypeParam | IndexedAccessType | IntersectionTypeAnnotation | NullableTypeAnnotation | ObjectTypeCallProperty | ObjectTypeIndexer | ObjectTypeInternalSlot | ObjectTypeProperty | ObjectTypeSpreadProperty | OpaqueType | OptionalIndexedAccessType | TupleTypeAnnotation | TypeAlias | TypeAnnotation | TypeParameter | TypeParameterInstantiation | TypeofTypeAnnotation | UnionTypeAnnotation;
    UnaryExpression: ArrayExpression | ArrowFunctionExpression | AssignmentExpression | AssignmentPattern | AwaitExpression | BinaryExpression | BindExpression | CallExpression | ClassAccessorProperty | ClassDeclaration | ClassExpression | ClassMethod | ClassPrivateProperty | ClassProperty | ConditionalExpression | Decorator | DoWhileStatement | ExportDefaultDeclaration | ExpressionStatement | ForInStatement | ForOfStatement | ForStatement | IfStatement | ImportExpression | JSXExpressionContainer | JSXSpreadAttribute | JSXSpreadChild | LogicalExpression | MemberExpression | NewExpression | ObjectMethod | ObjectProperty | OptionalCallExpression | OptionalMemberExpression | ParenthesizedExpression | PipelineBareFunction | PipelineTopicExpression | ReturnStatement | SequenceExpression | SpreadElement | SwitchCase | SwitchStatement | TSAsExpression | TSDeclareMethod | TSEnumDeclaration | TSEnumMember | TSExportAssignment | TSImportType | TSInstantiationExpression | TSLiteralType | TSMethodSignature | TSNonNullExpression | TSPropertySignature | TSSatisfiesExpression | TSTypeAssertion | TaggedTemplateExpression | TemplateLiteral | ThrowStatement | TupleExpression | TypeCastExpression | UnaryExpression | UpdateExpression | VariableDeclarator | WhileStatement | WithStatement | YieldExpression;
    UnionTypeAnnotation: ArrayTypeAnnotation | DeclareExportDeclaration | DeclareOpaqueType | DeclareTypeAlias | DeclaredPredicate | FunctionTypeAnnotation | FunctionTypeParam | IndexedAccessType | IntersectionTypeAnnotation | NullableTypeAnnotation | ObjectTypeCallProperty | ObjectTypeIndexer | ObjectTypeInternalSlot | ObjectTypeProperty | ObjectTypeSpreadProperty | OpaqueType | OptionalIndexedAccessType | TupleTypeAnnotation | TypeAlias | TypeAnnotation | TypeParameter | TypeParameterInstantiation | TypeofTypeAnnotation | UnionTypeAnnotation;
    UpdateExpression: ArrayExpression | ArrowFunctionExpression | AssignmentExpression | AssignmentPattern | AwaitExpression | BinaryExpression | BindExpression | CallExpression | ClassAccessorProperty | ClassDeclaration | ClassExpression | ClassMethod | ClassPrivateProperty | ClassProperty | ConditionalExpression | Decorator | DoWhileStatement | ExportDefaultDeclaration | ExpressionStatement | ForInStatement | ForOfStatement | ForStatement | IfStatement | ImportExpression | JSXExpressionContainer | JSXSpreadAttribute | JSXSpreadChild | LogicalExpression | MemberExpression | NewExpression | ObjectMethod | ObjectProperty | OptionalCallExpression | OptionalMemberExpression | ParenthesizedExpression | PipelineBareFunction | PipelineTopicExpression | ReturnStatement | SequenceExpression | SpreadElement | SwitchCase | SwitchStatement | TSAsExpression | TSDeclareMethod | TSEnumDeclaration | TSEnumMember | TSExportAssignment | TSImportType | TSInstantiationExpression | TSMethodSignature | TSNonNullExpression | TSPropertySignature | TSSatisfiesExpression | TSTypeAssertion | TaggedTemplateExpression | TemplateLiteral | ThrowStatement | TupleExpression | TypeCastExpression | UnaryExpression | UpdateExpression | VariableDeclarator | WhileStatement | WithStatement | YieldExpression;
    V8IntrinsicIdentifier: CallExpression | NewExpression;
    VariableDeclaration: BlockStatement | DoWhileStatement | ExportNamedDeclaration | ForInStatement | ForOfStatement | ForStatement | IfStatement | LabeledStatement | Program | StaticBlock | SwitchCase | TSModuleBlock | WhileStatement | WithStatement;
    VariableDeclarator: VariableDeclaration;
    Variance: ClassAccessorProperty | ClassPrivateProperty | ClassProperty | DeclareExportDeclaration | DeclaredPredicate | ObjectTypeIndexer | ObjectTypeProperty | TypeParameter;
    VoidTypeAnnotation: ArrayTypeAnnotation | DeclareExportDeclaration | DeclareOpaqueType | DeclareTypeAlias | DeclaredPredicate | FunctionTypeAnnotation | FunctionTypeParam | IndexedAccessType | IntersectionTypeAnnotation | NullableTypeAnnotation | ObjectTypeCallProperty | ObjectTypeIndexer | ObjectTypeInternalSlot | ObjectTypeProperty | ObjectTypeSpreadProperty | OpaqueType | OptionalIndexedAccessType | TupleTypeAnnotation | TypeAlias | TypeAnnotation | TypeParameter | TypeParameterInstantiation | TypeofTypeAnnotation | UnionTypeAnnotation;
    WhileStatement: BlockStatement | DoWhileStatement | ForInStatement | ForOfStatement | ForStatement | IfStatement | LabeledStatement | Program | StaticBlock | SwitchCase | TSModuleBlock | WhileStatement | WithStatement;
    WithStatement: BlockStatement | DoWhileStatement | ForInStatement | ForOfStatement | ForStatement | IfStatement | LabeledStatement | Program | StaticBlock | SwitchCase | TSModuleBlock | WhileStatement | WithStatement;
    YieldExpression: ArrayExpression | ArrowFunctionExpression | AssignmentExpression | AssignmentPattern | AwaitExpression | BinaryExpression | BindExpression | CallExpression | ClassAccessorProperty | ClassDeclaration | ClassExpression | ClassMethod | ClassPrivateProperty | ClassProperty | ConditionalExpression | Decorator | DoWhileStatement | ExportDefaultDeclaration | ExpressionStatement | ForInStatement | ForOfStatement | ForStatement | IfStatement | ImportExpression | JSXExpressionContainer | JSXSpreadAttribute | JSXSpreadChild | LogicalExpression | MemberExpression | NewExpression | ObjectMethod | ObjectProperty | OptionalCallExpression | OptionalMemberExpression | ParenthesizedExpression | PipelineBareFunction | PipelineTopicExpression | ReturnStatement | SequenceExpression | SpreadElement | SwitchCase | SwitchStatement | TSAsExpression | TSDeclareMethod | TSEnumDeclaration | TSEnumMember | TSExportAssignment | TSImportType | TSInstantiationExpression | TSMethodSignature | TSNonNullExpression | TSPropertySignature | TSSatisfiesExpression | TSTypeAssertion | TaggedTemplateExpression | TemplateLiteral | ThrowStatement | TupleExpression | TypeCastExpression | UnaryExpression | UpdateExpression | VariableDeclarator | WhileStatement | WithStatement | YieldExpression;
}

declare function deprecationWarning(oldName: string, newName: string, prefix?: string): void;

declare const react: {
    isReactComponent: (member: Node) => boolean;
    isCompatTag: typeof isCompatTag;
    buildChildren: typeof buildChildren;
};

export { ACCESSOR_TYPES, ALIAS_KEYS, ASSIGNMENT_OPERATORS, Accessor, Aliases, AnyTypeAnnotation, ArgumentPlaceholder, ArrayExpression, ArrayPattern, ArrayTypeAnnotation, ArrowFunctionExpression, AssignmentExpression, AssignmentPattern, AwaitExpression, BINARY_OPERATORS, BINARY_TYPES, BLOCKPARENT_TYPES, BLOCK_SCOPED_SYMBOL, BLOCK_TYPES, BOOLEAN_BINARY_OPERATORS, BOOLEAN_NUMBER_BINARY_OPERATORS, BOOLEAN_UNARY_OPERATORS, BUILDER_KEYS, BigIntLiteral, Binary, BinaryExpression, BindExpression, Block, BlockParent, BlockStatement, BooleanLiteral, BooleanLiteralTypeAnnotation, BooleanTypeAnnotation, BreakStatement, CLASS_TYPES, COMMENT_KEYS, COMPARISON_BINARY_OPERATORS, COMPLETIONSTATEMENT_TYPES, CONDITIONAL_TYPES, CallExpression, CatchClause, Class, ClassAccessorProperty, ClassBody, ClassDeclaration, ClassExpression, ClassImplements, ClassMethod, ClassPrivateMethod, ClassPrivateProperty, ClassProperty, Comment, CommentBlock, CommentLine, CommentTypeShorthand, CompletionStatement, Conditional, ConditionalExpression, ContinueStatement, DECLARATION_TYPES, DEPRECATED_ALIASES, DEPRECATED_KEYS, DebuggerStatement, DecimalLiteral, Declaration, DeclareClass, DeclareExportAllDeclaration, DeclareExportDeclaration, DeclareFunction, DeclareInterface, DeclareModule, DeclareModuleExports, DeclareOpaqueType, DeclareTypeAlias, DeclareVariable, DeclaredPredicate, Decorator, DeprecatedAliases, Directive, DirectiveLiteral, DoExpression, DoWhileStatement, ENUMBODY_TYPES, ENUMMEMBER_TYPES, EQUALITY_BINARY_OPERATORS, EXPORTDECLARATION_TYPES, EXPRESSIONWRAPPER_TYPES, EXPRESSION_TYPES, EmptyStatement, EmptyTypeAnnotation, EnumBody, EnumBooleanBody, EnumBooleanMember, EnumDeclaration, EnumDefaultedMember, EnumMember, EnumNumberBody, EnumNumberMember, EnumStringBody, EnumStringMember, EnumSymbolBody, ExistsTypeAnnotation, ExportAllDeclaration, ExportDeclaration, ExportDefaultDeclaration, ExportDefaultSpecifier, ExportNamedDeclaration, ExportNamespaceSpecifier, ExportSpecifier, Expression, ExpressionStatement, ExpressionWrapper, FLATTENABLE_KEYS, FLIPPED_ALIAS_KEYS, FLOWBASEANNOTATION_TYPES, FLOWDECLARATION_TYPES, FLOWPREDICATE_TYPES, FLOWTYPE_TYPES, FLOW_TYPES, FORXSTATEMENT_TYPES, FOR_INIT_KEYS, FOR_TYPES, FUNCTIONPARENT_TYPES, FUNCTION_TYPES, FieldOptions, File, Flow, FlowBaseAnnotation, FlowDeclaration, FlowPredicate, FlowType, For, ForInStatement, ForOfStatement, ForStatement, ForXStatement, Function, FunctionDeclaration, FunctionExpression, FunctionParent, FunctionTypeAnnotation, FunctionTypeParam, GenericTypeAnnotation, IMMUTABLE_TYPES, IMPORTOREXPORTDECLARATION_TYPES, INHERIT_KEYS, Identifier, IfStatement, Immutable, Import, ImportAttribute, ImportDeclaration, ImportDefaultSpecifier, ImportExpression, ImportNamespaceSpecifier, ImportOrExportDeclaration, ImportSpecifier, IndexedAccessType, InferredPredicate, InterfaceDeclaration, InterfaceExtends, InterfaceTypeAnnotation, InterpreterDirective, IntersectionTypeAnnotation, JSX, JSXAttribute, JSXClosingElement, JSXClosingFragment, JSXElement, JSXEmptyExpression, JSXExpressionContainer, JSXFragment, JSXIdentifier, JSXMemberExpression, JSXNamespacedName, JSXOpeningElement, JSXOpeningFragment, JSXSpreadAttribute, JSXSpreadChild, JSXText, JSX_TYPES, LITERAL_TYPES, LOGICAL_OPERATORS, LOOP_TYPES, LVAL_TYPES, LVal, LabeledStatement, Literal, LogicalExpression, Loop, METHOD_TYPES, MISCELLANEOUS_TYPES, MODULEDECLARATION_TYPES, MODULESPECIFIER_TYPES, MemberExpression, MetaProperty, Method, Miscellaneous, MixedTypeAnnotation, ModuleDeclaration, ModuleExpression, ModuleSpecifier, NODE_FIELDS, NODE_PARENT_VALIDATIONS, NOT_LOCAL_BINDING, NUMBER_BINARY_OPERATORS, NUMBER_UNARY_OPERATORS, NewExpression, Node, Noop, NullLiteral, NullLiteralTypeAnnotation, NullableTypeAnnotation, NumberLiteral, NumberLiteralTypeAnnotation, NumberTypeAnnotation, NumericLiteral, OBJECTMEMBER_TYPES, ObjectExpression, ObjectMember, ObjectMethod, ObjectPattern, ObjectProperty, ObjectTypeAnnotation, ObjectTypeCallProperty, ObjectTypeIndexer, ObjectTypeInternalSlot, ObjectTypeProperty, ObjectTypeSpreadProperty, OpaqueType, OptionalCallExpression, OptionalIndexedAccessType, OptionalMemberExpression, PATTERNLIKE_TYPES, PATTERN_TYPES, PLACEHOLDERS, PLACEHOLDERS_ALIAS, PLACEHOLDERS_FLIPPED_ALIAS, PRIVATE_TYPES, PROPERTY_TYPES, PUREISH_TYPES, ParentMaps, ParenthesizedExpression, Pattern, PatternLike, PipelineBareFunction, PipelinePrimaryTopicReference, PipelineTopicExpression, Placeholder, Private, PrivateName, Program, Property, Pureish, QualifiedTypeIdentifier, RecordExpression, RegExpLiteral, RegexLiteral, Options as RemovePropertiesOptions, RestElement, RestProperty, ReturnStatement, SCOPABLE_TYPES, STANDARDIZED_TYPES, STATEMENT_OR_BLOCK_KEYS, STATEMENT_TYPES, STRING_UNARY_OPERATORS, Scopable, SequenceExpression, SourceLocation, SpreadElement, SpreadProperty, Standardized, Statement, StaticBlock, StringLiteral, StringLiteralTypeAnnotation, StringTypeAnnotation, Super, SwitchCase, SwitchStatement, SymbolTypeAnnotation, TERMINATORLESS_TYPES, TSAnyKeyword, TSArrayType, TSAsExpression, TSBASETYPE_TYPES, TSBaseType, TSBigIntKeyword, TSBooleanKeyword, TSCallSignatureDeclaration, TSConditionalType, TSConstructSignatureDeclaration, TSConstructorType, TSDeclareFunction, TSDeclareMethod, TSENTITYNAME_TYPES, TSEntityName, TSEnumDeclaration, TSEnumMember, TSExportAssignment, TSExpressionWithTypeArguments, TSExternalModuleReference, TSFunctionType, TSImportEqualsDeclaration, TSImportType, TSIndexSignature, TSIndexedAccessType, TSInferType, TSInstantiationExpression, TSInterfaceBody, TSInterfaceDeclaration, TSIntersectionType, TSIntrinsicKeyword, TSLiteralType, TSMappedType, TSMethodSignature, TSModuleBlock, TSModuleDeclaration, TSNamedTupleMember, TSNamespaceExportDeclaration, TSNeverKeyword, TSNonNullExpression, TSNullKeyword, TSNumberKeyword, TSObjectKeyword, TSOptionalType, TSParameterProperty, TSParenthesizedType, TSPropertySignature, TSQualifiedName, TSRestType, TSSatisfiesExpression, TSStringKeyword, TSSymbolKeyword, TSTYPEELEMENT_TYPES, TSTYPE_TYPES, TSThisType, TSTupleType, TSType, TSTypeAliasDeclaration, TSTypeAnnotation, TSTypeAssertion, TSTypeElement, TSTypeLiteral, TSTypeOperator, TSTypeParameter, TSTypeParameterDeclaration, TSTypeParameterInstantiation, TSTypePredicate, TSTypeQuery, TSTypeReference, TSUndefinedKeyword, TSUnionType, TSUnknownKeyword, TSVoidKeyword, TYPES, TYPESCRIPT_TYPES, TaggedTemplateExpression, TemplateElement, TemplateLiteral, Terminatorless, ThisExpression, ThisTypeAnnotation, ThrowStatement, TopicReference, TraversalAncestors, TraversalHandler, TraversalHandlers, TryStatement, TupleExpression, TupleTypeAnnotation, TypeAlias, TypeAnnotation, TypeCastExpression, TypeParameter, TypeParameterDeclaration, TypeParameterInstantiation, TypeScript, TypeofTypeAnnotation, UNARYLIKE_TYPES, UNARY_OPERATORS, UPDATE_OPERATORS, USERWHITESPACABLE_TYPES, UnaryExpression, UnaryLike, UnionTypeAnnotation, UpdateExpression, UserWhitespacable, V8IntrinsicIdentifier, VISITOR_KEYS, VariableDeclaration, VariableDeclarator, Variance, VoidTypeAnnotation, WHILE_TYPES, While, WhileStatement, WithStatement, YieldExpression, deprecationWarning as __internal__deprecationWarning, addComment, addComments, anyTypeAnnotation, appendToMemberExpression, argumentPlaceholder, arrayExpression, arrayPattern, arrayTypeAnnotation, arrowFunctionExpression, assertAccessor, assertAnyTypeAnnotation, assertArgumentPlaceholder, assertArrayExpression, assertArrayPattern, assertArrayTypeAnnotation, assertArrowFunctionExpression, assertAssignmentExpression, assertAssignmentPattern, assertAwaitExpression, assertBigIntLiteral, assertBinary, assertBinaryExpression, assertBindExpression, assertBlock, assertBlockParent, assertBlockStatement, assertBooleanLiteral, assertBooleanLiteralTypeAnnotation, assertBooleanTypeAnnotation, assertBreakStatement, assertCallExpression, assertCatchClause, assertClass, assertClassAccessorProperty, assertClassBody, assertClassDeclaration, assertClassExpression, assertClassImplements, assertClassMethod, assertClassPrivateMethod, assertClassPrivateProperty, assertClassProperty, assertCompletionStatement, assertConditional, assertConditionalExpression, assertContinueStatement, assertDebuggerStatement, assertDecimalLiteral, assertDeclaration, assertDeclareClass, assertDeclareExportAllDeclaration, assertDeclareExportDeclaration, assertDeclareFunction, assertDeclareInterface, assertDeclareModule, assertDeclareModuleExports, assertDeclareOpaqueType, assertDeclareTypeAlias, assertDeclareVariable, assertDeclaredPredicate, assertDecorator, assertDirective, assertDirectiveLiteral, assertDoExpression, assertDoWhileStatement, assertEmptyStatement, assertEmptyTypeAnnotation, assertEnumBody, assertEnumBooleanBody, assertEnumBooleanMember, assertEnumDeclaration, assertEnumDefaultedMember, assertEnumMember, assertEnumNumberBody, assertEnumNumberMember, assertEnumStringBody, assertEnumStringMember, assertEnumSymbolBody, assertExistsTypeAnnotation, assertExportAllDeclaration, assertExportDeclaration, assertExportDefaultDeclaration, assertExportDefaultSpecifier, assertExportNamedDeclaration, assertExportNamespaceSpecifier, assertExportSpecifier, assertExpression, assertExpressionStatement, assertExpressionWrapper, assertFile, assertFlow, assertFlowBaseAnnotation, assertFlowDeclaration, assertFlowPredicate, assertFlowType, assertFor, assertForInStatement, assertForOfStatement, assertForStatement, assertForXStatement, assertFunction, assertFunctionDeclaration, assertFunctionExpression, assertFunctionParent, assertFunctionTypeAnnotation, assertFunctionTypeParam, assertGenericTypeAnnotation, assertIdentifier, assertIfStatement, assertImmutable, assertImport, assertImportAttribute, assertImportDeclaration, assertImportDefaultSpecifier, assertImportExpression, assertImportNamespaceSpecifier, assertImportOrExportDeclaration, assertImportSpecifier, assertIndexedAccessType, assertInferredPredicate, assertInterfaceDeclaration, assertInterfaceExtends, assertInterfaceTypeAnnotation, assertInterpreterDirective, assertIntersectionTypeAnnotation, assertJSX, assertJSXAttribute, assertJSXClosingElement, assertJSXClosingFragment, assertJSXElement, assertJSXEmptyExpression, assertJSXExpressionContainer, assertJSXFragment, assertJSXIdentifier, assertJSXMemberExpression, assertJSXNamespacedName, assertJSXOpeningElement, assertJSXOpeningFragment, assertJSXSpreadAttribute, assertJSXSpreadChild, assertJSXText, assertLVal, assertLabeledStatement, assertLiteral, assertLogicalExpression, assertLoop, assertMemberExpression, assertMetaProperty, assertMethod, assertMiscellaneous, assertMixedTypeAnnotation, assertModuleDeclaration, assertModuleExpression, assertModuleSpecifier, assertNewExpression, assertNode, assertNoop, assertNullLiteral, assertNullLiteralTypeAnnotation, assertNullableTypeAnnotation, assertNumberLiteral, assertNumberLiteralTypeAnnotation, assertNumberTypeAnnotation, assertNumericLiteral, assertObjectExpression, assertObjectMember, assertObjectMethod, assertObjectPattern, assertObjectProperty, assertObjectTypeAnnotation, assertObjectTypeCallProperty, assertObjectTypeIndexer, assertObjectTypeInternalSlot, assertObjectTypeProperty, assertObjectTypeSpreadProperty, assertOpaqueType, assertOptionalCallExpression, assertOptionalIndexedAccessType, assertOptionalMemberExpression, assertParenthesizedExpression, assertPattern, assertPatternLike, assertPipelineBareFunction, assertPipelinePrimaryTopicReference, assertPipelineTopicExpression, assertPlaceholder, assertPrivate, assertPrivateName, assertProgram, assertProperty, assertPureish, assertQualifiedTypeIdentifier, assertRecordExpression, assertRegExpLiteral, assertRegexLiteral, assertRestElement, assertRestProperty, assertReturnStatement, assertScopable, assertSequenceExpression, assertSpreadElement, assertSpreadProperty, assertStandardized, assertStatement, assertStaticBlock, assertStringLiteral, assertStringLiteralTypeAnnotation, assertStringTypeAnnotation, assertSuper, assertSwitchCase, assertSwitchStatement, assertSymbolTypeAnnotation, assertTSAnyKeyword, assertTSArrayType, assertTSAsExpression, assertTSBaseType, assertTSBigIntKeyword, assertTSBooleanKeyword, assertTSCallSignatureDeclaration, assertTSConditionalType, assertTSConstructSignatureDeclaration, assertTSConstructorType, assertTSDeclareFunction, assertTSDeclareMethod, assertTSEntityName, assertTSEnumDeclaration, assertTSEnumMember, assertTSExportAssignment, assertTSExpressionWithTypeArguments, assertTSExternalModuleReference, assertTSFunctionType, assertTSImportEqualsDeclaration, assertTSImportType, assertTSIndexSignature, assertTSIndexedAccessType, assertTSInferType, assertTSInstantiationExpression, assertTSInterfaceBody, assertTSInterfaceDeclaration, assertTSIntersectionType, assertTSIntrinsicKeyword, assertTSLiteralType, assertTSMappedType, assertTSMethodSignature, assertTSModuleBlock, assertTSModuleDeclaration, assertTSNamedTupleMember, assertTSNamespaceExportDeclaration, assertTSNeverKeyword, assertTSNonNullExpression, assertTSNullKeyword, assertTSNumberKeyword, assertTSObjectKeyword, assertTSOptionalType, assertTSParameterProperty, assertTSParenthesizedType, assertTSPropertySignature, assertTSQualifiedName, assertTSRestType, assertTSSatisfiesExpression, assertTSStringKeyword, assertTSSymbolKeyword, assertTSThisType, assertTSTupleType, assertTSType, assertTSTypeAliasDeclaration, assertTSTypeAnnotation, assertTSTypeAssertion, assertTSTypeElement, assertTSTypeLiteral, assertTSTypeOperator, assertTSTypeParameter, assertTSTypeParameterDeclaration, assertTSTypeParameterInstantiation, assertTSTypePredicate, assertTSTypeQuery, assertTSTypeReference, assertTSUndefinedKeyword, assertTSUnionType, assertTSUnknownKeyword, assertTSVoidKeyword, assertTaggedTemplateExpression, assertTemplateElement, assertTemplateLiteral, assertTerminatorless, assertThisExpression, assertThisTypeAnnotation, assertThrowStatement, assertTopicReference, assertTryStatement, assertTupleExpression, assertTupleTypeAnnotation, assertTypeAlias, assertTypeAnnotation, assertTypeCastExpression, assertTypeParameter, assertTypeParameterDeclaration, assertTypeParameterInstantiation, assertTypeScript, assertTypeofTypeAnnotation, assertUnaryExpression, assertUnaryLike, assertUnionTypeAnnotation, assertUpdateExpression, assertUserWhitespacable, assertV8IntrinsicIdentifier, assertVariableDeclaration, assertVariableDeclarator, assertVariance, assertVoidTypeAnnotation, assertWhile, assertWhileStatement, assertWithStatement, assertYieldExpression, assignmentExpression, assignmentPattern, awaitExpression, bigIntLiteral, binaryExpression, bindExpression, blockStatement, booleanLiteral, booleanLiteralTypeAnnotation, booleanTypeAnnotation, breakStatement, buildMatchMemberExpression, buildUndefinedNode, callExpression, catchClause, classAccessorProperty, classBody, classDeclaration, classExpression, classImplements, classMethod, classPrivateMethod, classPrivateProperty, classProperty, clone, cloneDeep, cloneDeepWithoutLoc, cloneNode, cloneWithoutLoc, conditionalExpression, continueStatement, createFlowUnionType, createTSUnionType, _default$4 as createTypeAnnotationBasedOnTypeof, createFlowUnionType as createUnionTypeAnnotation, debuggerStatement, decimalLiteral, declareClass, declareExportAllDeclaration, declareExportDeclaration, declareFunction, declareInterface, declareModule, declareModuleExports, declareOpaqueType, declareTypeAlias, declareVariable, declaredPredicate, decorator, directive, directiveLiteral, doExpression, doWhileStatement, emptyStatement, emptyTypeAnnotation, ensureBlock, enumBooleanBody, enumBooleanMember, enumDeclaration, enumDefaultedMember, enumNumberBody, enumNumberMember, enumStringBody, enumStringMember, enumSymbolBody, existsTypeAnnotation, exportAllDeclaration, exportDefaultDeclaration, exportDefaultSpecifier, exportNamedDeclaration, exportNamespaceSpecifier, exportSpecifier, expressionStatement, file, forInStatement, forOfStatement, forStatement, functionDeclaration, functionExpression, functionTypeAnnotation, functionTypeParam, genericTypeAnnotation, getBindingIdentifiers, _default as getOuterBindingIdentifiers, identifier, ifStatement, _import as import, importAttribute, importDeclaration, importDefaultSpecifier, importExpression, importNamespaceSpecifier, importSpecifier, indexedAccessType, inferredPredicate, inheritInnerComments, inheritLeadingComments, inheritTrailingComments, inherits, inheritsComments, interfaceDeclaration, interfaceExtends, interfaceTypeAnnotation, interpreterDirective, intersectionTypeAnnotation, is, isAccessor, isAnyTypeAnnotation, isArgumentPlaceholder, isArrayExpression, isArrayPattern, isArrayTypeAnnotation, isArrowFunctionExpression, isAssignmentExpression, isAssignmentPattern, isAwaitExpression, isBigIntLiteral, isBinary, isBinaryExpression, isBindExpression, isBinding, isBlock, isBlockParent, isBlockScoped, isBlockStatement, isBooleanLiteral, isBooleanLiteralTypeAnnotation, isBooleanTypeAnnotation, isBreakStatement, isCallExpression, isCatchClause, isClass, isClassAccessorProperty, isClassBody, isClassDeclaration, isClassExpression, isClassImplements, isClassMethod, isClassPrivateMethod, isClassPrivateProperty, isClassProperty, isCompletionStatement, isConditional, isConditionalExpression, isContinueStatement, isDebuggerStatement, isDecimalLiteral, isDeclaration, isDeclareClass, isDeclareExportAllDeclaration, isDeclareExportDeclaration, isDeclareFunction, isDeclareInterface, isDeclareModule, isDeclareModuleExports, isDeclareOpaqueType, isDeclareTypeAlias, isDeclareVariable, isDeclaredPredicate, isDecorator, isDirective, isDirectiveLiteral, isDoExpression, isDoWhileStatement, isEmptyStatement, isEmptyTypeAnnotation, isEnumBody, isEnumBooleanBody, isEnumBooleanMember, isEnumDeclaration, isEnumDefaultedMember, isEnumMember, isEnumNumberBody, isEnumNumberMember, isEnumStringBody, isEnumStringMember, isEnumSymbolBody, isExistsTypeAnnotation, isExportAllDeclaration, isExportDeclaration, isExportDefaultDeclaration, isExportDefaultSpecifier, isExportNamedDeclaration, isExportNamespaceSpecifier, isExportSpecifier, isExpression, isExpressionStatement, isExpressionWrapper, isFile, isFlow, isFlowBaseAnnotation, isFlowDeclaration, isFlowPredicate, isFlowType, isFor, isForInStatement, isForOfStatement, isForStatement, isForXStatement, isFunction, isFunctionDeclaration, isFunctionExpression, isFunctionParent, isFunctionTypeAnnotation, isFunctionTypeParam, isGenericTypeAnnotation, isIdentifier, isIfStatement, isImmutable, isImport, isImportAttribute, isImportDeclaration, isImportDefaultSpecifier, isImportExpression, isImportNamespaceSpecifier, isImportOrExportDeclaration, isImportSpecifier, isIndexedAccessType, isInferredPredicate, isInterfaceDeclaration, isInterfaceExtends, isInterfaceTypeAnnotation, isInterpreterDirective, isIntersectionTypeAnnotation, isJSX, isJSXAttribute, isJSXClosingElement, isJSXClosingFragment, isJSXElement, isJSXEmptyExpression, isJSXExpressionContainer, isJSXFragment, isJSXIdentifier, isJSXMemberExpression, isJSXNamespacedName, isJSXOpeningElement, isJSXOpeningFragment, isJSXSpreadAttribute, isJSXSpreadChild, isJSXText, isLVal, isLabeledStatement, isLet, isLiteral, isLogicalExpression, isLoop, isMemberExpression, isMetaProperty, isMethod, isMiscellaneous, isMixedTypeAnnotation, isModuleDeclaration, isModuleExpression, isModuleSpecifier, isNewExpression, isNode, isNodesEquivalent, isNoop, isNullLiteral, isNullLiteralTypeAnnotation, isNullableTypeAnnotation, isNumberLiteral, isNumberLiteralTypeAnnotation, isNumberTypeAnnotation, isNumericLiteral, isObjectExpression, isObjectMember, isObjectMethod, isObjectPattern, isObjectProperty, isObjectTypeAnnotation, isObjectTypeCallProperty, isObjectTypeIndexer, isObjectTypeInternalSlot, isObjectTypeProperty, isObjectTypeSpreadProperty, isOpaqueType, isOptionalCallExpression, isOptionalIndexedAccessType, isOptionalMemberExpression, isParenthesizedExpression, isPattern, isPatternLike, isPipelineBareFunction, isPipelinePrimaryTopicReference, isPipelineTopicExpression, isPlaceholder, isPlaceholderType, isPrivate, isPrivateName, isProgram, isProperty, isPureish, isQualifiedTypeIdentifier, isRecordExpression, isReferenced, isRegExpLiteral, isRegexLiteral, isRestElement, isRestProperty, isReturnStatement, isScopable, isScope, isSequenceExpression, isSpecifierDefault, isSpreadElement, isSpreadProperty, isStandardized, isStatement, isStaticBlock, isStringLiteral, isStringLiteralTypeAnnotation, isStringTypeAnnotation, isSuper, isSwitchCase, isSwitchStatement, isSymbolTypeAnnotation, isTSAnyKeyword, isTSArrayType, isTSAsExpression, isTSBaseType, isTSBigIntKeyword, isTSBooleanKeyword, isTSCallSignatureDeclaration, isTSConditionalType, isTSConstructSignatureDeclaration, isTSConstructorType, isTSDeclareFunction, isTSDeclareMethod, isTSEntityName, isTSEnumDeclaration, isTSEnumMember, isTSExportAssignment, isTSExpressionWithTypeArguments, isTSExternalModuleReference, isTSFunctionType, isTSImportEqualsDeclaration, isTSImportType, isTSIndexSignature, isTSIndexedAccessType, isTSInferType, isTSInstantiationExpression, isTSInterfaceBody, isTSInterfaceDeclaration, isTSIntersectionType, isTSIntrinsicKeyword, isTSLiteralType, isTSMappedType, isTSMethodSignature, isTSModuleBlock, isTSModuleDeclaration, isTSNamedTupleMember, isTSNamespaceExportDeclaration, isTSNeverKeyword, isTSNonNullExpression, isTSNullKeyword, isTSNumberKeyword, isTSObjectKeyword, isTSOptionalType, isTSParameterProperty, isTSParenthesizedType, isTSPropertySignature, isTSQualifiedName, isTSRestType, isTSSatisfiesExpression, isTSStringKeyword, isTSSymbolKeyword, isTSThisType, isTSTupleType, isTSType, isTSTypeAliasDeclaration, isTSTypeAnnotation, isTSTypeAssertion, isTSTypeElement, isTSTypeLiteral, isTSTypeOperator, isTSTypeParameter, isTSTypeParameterDeclaration, isTSTypeParameterInstantiation, isTSTypePredicate, isTSTypeQuery, isTSTypeReference, isTSUndefinedKeyword, isTSUnionType, isTSUnknownKeyword, isTSVoidKeyword, isTaggedTemplateExpression, isTemplateElement, isTemplateLiteral, isTerminatorless, isThisExpression, isThisTypeAnnotation, isThrowStatement, isTopicReference, isTryStatement, isTupleExpression, isTupleTypeAnnotation, isType, isTypeAlias, isTypeAnnotation, isTypeCastExpression, isTypeParameter, isTypeParameterDeclaration, isTypeParameterInstantiation, isTypeScript, isTypeofTypeAnnotation, isUnaryExpression, isUnaryLike, isUnionTypeAnnotation, isUpdateExpression, isUserWhitespacable, isV8IntrinsicIdentifier, isValidES3Identifier, isValidIdentifier, isVar, isVariableDeclaration, isVariableDeclarator, isVariance, isVoidTypeAnnotation, isWhile, isWhileStatement, isWithStatement, isYieldExpression, jsxAttribute as jSXAttribute, jsxClosingElement as jSXClosingElement, jsxClosingFragment as jSXClosingFragment, jsxElement as jSXElement, jsxEmptyExpression as jSXEmptyExpression, jsxExpressionContainer as jSXExpressionContainer, jsxFragment as jSXFragment, jsxIdentifier as jSXIdentifier, jsxMemberExpression as jSXMemberExpression, jsxNamespacedName as jSXNamespacedName, jsxOpeningElement as jSXOpeningElement, jsxOpeningFragment as jSXOpeningFragment, jsxSpreadAttribute as jSXSpreadAttribute, jsxSpreadChild as jSXSpreadChild, jsxText as jSXText, jsxAttribute, jsxClosingElement, jsxClosingFragment, jsxElement, jsxEmptyExpression, jsxExpressionContainer, jsxFragment, jsxIdentifier, jsxMemberExpression, jsxNamespacedName, jsxOpeningElement, jsxOpeningFragment, jsxSpreadAttribute, jsxSpreadChild, jsxText, labeledStatement, logicalExpression, matchesPattern, memberExpression, metaProperty, mixedTypeAnnotation, moduleExpression, newExpression, noop, nullLiteral, nullLiteralTypeAnnotation, nullableTypeAnnotation, NumberLiteral$1 as numberLiteral, numberLiteralTypeAnnotation, numberTypeAnnotation, numericLiteral, objectExpression, objectMethod, objectPattern, objectProperty, objectTypeAnnotation, objectTypeCallProperty, objectTypeIndexer, objectTypeInternalSlot, objectTypeProperty, objectTypeSpreadProperty, opaqueType, optionalCallExpression, optionalIndexedAccessType, optionalMemberExpression, parenthesizedExpression, pipelineBareFunction, pipelinePrimaryTopicReference, pipelineTopicExpression, placeholder, prependToMemberExpression, privateName, program, qualifiedTypeIdentifier, react, recordExpression, regExpLiteral, RegexLiteral$1 as regexLiteral, removeComments, removeProperties, removePropertiesDeep, removeTypeDuplicates, restElement, RestProperty$1 as restProperty, returnStatement, sequenceExpression, shallowEqual, spreadElement, SpreadProperty$1 as spreadProperty, staticBlock, stringLiteral, stringLiteralTypeAnnotation, stringTypeAnnotation, _super as super, switchCase, switchStatement, symbolTypeAnnotation, tsAnyKeyword as tSAnyKeyword, tsArrayType as tSArrayType, tsAsExpression as tSAsExpression, tsBigIntKeyword as tSBigIntKeyword, tsBooleanKeyword as tSBooleanKeyword, tsCallSignatureDeclaration as tSCallSignatureDeclaration, tsConditionalType as tSConditionalType, tsConstructSignatureDeclaration as tSConstructSignatureDeclaration, tsConstructorType as tSConstructorType, tsDeclareFunction as tSDeclareFunction, tsDeclareMethod as tSDeclareMethod, tsEnumDeclaration as tSEnumDeclaration, tsEnumMember as tSEnumMember, tsExportAssignment as tSExportAssignment, tsExpressionWithTypeArguments as tSExpressionWithTypeArguments, tsExternalModuleReference as tSExternalModuleReference, tsFunctionType as tSFunctionType, tsImportEqualsDeclaration as tSImportEqualsDeclaration, tsImportType as tSImportType, tsIndexSignature as tSIndexSignature, tsIndexedAccessType as tSIndexedAccessType, tsInferType as tSInferType, tsInstantiationExpression as tSInstantiationExpression, tsInterfaceBody as tSInterfaceBody, tsInterfaceDeclaration as tSInterfaceDeclaration, tsIntersectionType as tSIntersectionType, tsIntrinsicKeyword as tSIntrinsicKeyword, tsLiteralType as tSLiteralType, tsMappedType as tSMappedType, tsMethodSignature as tSMethodSignature, tsModuleBlock as tSModuleBlock, tsModuleDeclaration as tSModuleDeclaration, tsNamedTupleMember as tSNamedTupleMember, tsNamespaceExportDeclaration as tSNamespaceExportDeclaration, tsNeverKeyword as tSNeverKeyword, tsNonNullExpression as tSNonNullExpression, tsNullKeyword as tSNullKeyword, tsNumberKeyword as tSNumberKeyword, tsObjectKeyword as tSObjectKeyword, tsOptionalType as tSOptionalType, tsParameterProperty as tSParameterProperty, tsParenthesizedType as tSParenthesizedType, tsPropertySignature as tSPropertySignature, tsQualifiedName as tSQualifiedName, tsRestType as tSRestType, tsSatisfiesExpression as tSSatisfiesExpression, tsStringKeyword as tSStringKeyword, tsSymbolKeyword as tSSymbolKeyword, tsThisType as tSThisType, tsTupleType as tSTupleType, tsTypeAliasDeclaration as tSTypeAliasDeclaration, tsTypeAnnotation as tSTypeAnnotation, tsTypeAssertion as tSTypeAssertion, tsTypeLiteral as tSTypeLiteral, tsTypeOperator as tSTypeOperator, tsTypeParameter as tSTypeParameter, tsTypeParameterDeclaration as tSTypeParameterDeclaration, tsTypeParameterInstantiation as tSTypeParameterInstantiation, tsTypePredicate as tSTypePredicate, tsTypeQuery as tSTypeQuery, tsTypeReference as tSTypeReference, tsUndefinedKeyword as tSUndefinedKeyword, tsUnionType as tSUnionType, tsUnknownKeyword as tSUnknownKeyword, tsVoidKeyword as tSVoidKeyword, taggedTemplateExpression, templateElement, templateLiteral, thisExpression, thisTypeAnnotation, throwStatement, toBindingIdentifierName, toBlock, toComputedKey, _default$3 as toExpression, toIdentifier, toKeyAlias, _default$2 as toStatement, topicReference, traverse, traverseFast, tryStatement, tsAnyKeyword, tsArrayType, tsAsExpression, tsBigIntKeyword, tsBooleanKeyword, tsCallSignatureDeclaration, tsConditionalType, tsConstructSignatureDeclaration, tsConstructorType, tsDeclareFunction, tsDeclareMethod, tsEnumDeclaration, tsEnumMember, tsExportAssignment, tsExpressionWithTypeArguments, tsExternalModuleReference, tsFunctionType, tsImportEqualsDeclaration, tsImportType, tsIndexSignature, tsIndexedAccessType, tsInferType, tsInstantiationExpression, tsInterfaceBody, tsInterfaceDeclaration, tsIntersectionType, tsIntrinsicKeyword, tsLiteralType, tsMappedType, tsMethodSignature, tsModuleBlock, tsModuleDeclaration, tsNamedTupleMember, tsNamespaceExportDeclaration, tsNeverKeyword, tsNonNullExpression, tsNullKeyword, tsNumberKeyword, tsObjectKeyword, tsOptionalType, tsParameterProperty, tsParenthesizedType, tsPropertySignature, tsQualifiedName, tsRestType, tsSatisfiesExpression, tsStringKeyword, tsSymbolKeyword, tsThisType, tsTupleType, tsTypeAliasDeclaration, tsTypeAnnotation, tsTypeAssertion, tsTypeLiteral, tsTypeOperator, tsTypeParameter, tsTypeParameterDeclaration, tsTypeParameterInstantiation, tsTypePredicate, tsTypeQuery, tsTypeReference, tsUndefinedKeyword, tsUnionType, tsUnknownKeyword, tsVoidKeyword, tupleExpression, tupleTypeAnnotation, typeAlias, typeAnnotation, typeCastExpression, typeParameter, typeParameterDeclaration, typeParameterInstantiation, typeofTypeAnnotation, unaryExpression, unionTypeAnnotation, updateExpression, v8IntrinsicIdentifier, validate, _default$1 as valueToNode, variableDeclaration, variableDeclarator, variance, voidTypeAnnotation, whileStatement, withStatement, yieldExpression };
