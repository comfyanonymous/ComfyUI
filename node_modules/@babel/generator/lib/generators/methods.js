"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.ArrowFunctionExpression = ArrowFunctionExpression;
exports.FunctionDeclaration = exports.FunctionExpression = FunctionExpression;
exports._functionHead = _functionHead;
exports._methodHead = _methodHead;
exports._param = _param;
exports._parameters = _parameters;
exports._params = _params;
exports._predicate = _predicate;
var _t = require("@babel/types");
const {
  isIdentifier
} = _t;
function _params(node, idNode, parentNode) {
  this.print(node.typeParameters, node);
  const nameInfo = _getFuncIdName.call(this, idNode, parentNode);
  if (nameInfo) {
    this.sourceIdentifierName(nameInfo.name, nameInfo.pos);
  }
  this.tokenChar(40);
  this._parameters(node.params, node);
  this.tokenChar(41);
  const noLineTerminator = node.type === "ArrowFunctionExpression";
  this.print(node.returnType, node, noLineTerminator);
  this._noLineTerminator = noLineTerminator;
}
function _parameters(parameters, parent) {
  const paramLength = parameters.length;
  for (let i = 0; i < paramLength; i++) {
    this._param(parameters[i], parent);
    if (i < parameters.length - 1) {
      this.tokenChar(44);
      this.space();
    }
  }
}
function _param(parameter, parent) {
  this.printJoin(parameter.decorators, parameter);
  this.print(parameter, parent);
  if (parameter.optional) {
    this.tokenChar(63);
  }
  this.print(parameter.typeAnnotation, parameter);
}
function _methodHead(node) {
  const kind = node.kind;
  const key = node.key;
  if (kind === "get" || kind === "set") {
    this.word(kind);
    this.space();
  }
  if (node.async) {
    this.word("async", true);
    this.space();
  }
  if (kind === "method" || kind === "init") {
    if (node.generator) {
      this.tokenChar(42);
    }
  }
  if (node.computed) {
    this.tokenChar(91);
    this.print(key, node);
    this.tokenChar(93);
  } else {
    this.print(key, node);
  }
  if (node.optional) {
    this.tokenChar(63);
  }
  this._params(node, node.computed && node.key.type !== "StringLiteral" ? undefined : node.key, undefined);
}
function _predicate(node, noLineTerminatorAfter) {
  if (node.predicate) {
    if (!node.returnType) {
      this.tokenChar(58);
    }
    this.space();
    this.print(node.predicate, node, noLineTerminatorAfter);
  }
}
function _functionHead(node, parent) {
  if (node.async) {
    this.word("async");
    this._endsWithInnerRaw = false;
    this.space();
  }
  this.word("function");
  if (node.generator) {
    this._endsWithInnerRaw = false;
    this.tokenChar(42);
  }
  this.space();
  if (node.id) {
    this.print(node.id, node);
  }
  this._params(node, node.id, parent);
  if (node.type !== "TSDeclareFunction") {
    this._predicate(node);
  }
}
function FunctionExpression(node, parent) {
  this._functionHead(node, parent);
  this.space();
  this.print(node.body, node);
}
function ArrowFunctionExpression(node, parent) {
  if (node.async) {
    this.word("async", true);
    this.space();
  }
  let firstParam;
  if (!this.format.retainLines && node.params.length === 1 && isIdentifier(firstParam = node.params[0]) && !hasTypesOrComments(node, firstParam)) {
    this.print(firstParam, node, true);
  } else {
    this._params(node, undefined, parent);
  }
  this._predicate(node, true);
  this.space();
  this.printInnerComments();
  this.token("=>");
  this.space();
  this.print(node.body, node);
}
function hasTypesOrComments(node, param) {
  var _param$leadingComment, _param$trailingCommen;
  return !!(node.typeParameters || node.returnType || node.predicate || param.typeAnnotation || param.optional || (_param$leadingComment = param.leadingComments) != null && _param$leadingComment.length || (_param$trailingCommen = param.trailingComments) != null && _param$trailingCommen.length);
}
function _getFuncIdName(idNode, parent) {
  let id = idNode;
  if (!id && parent) {
    const parentType = parent.type;
    if (parentType === "VariableDeclarator") {
      id = parent.id;
    } else if (parentType === "AssignmentExpression" || parentType === "AssignmentPattern") {
      id = parent.left;
    } else if (parentType === "ObjectProperty" || parentType === "ClassProperty") {
      if (!parent.computed || parent.key.type === "StringLiteral") {
        id = parent.key;
      }
    } else if (parentType === "ClassPrivateProperty" || parentType === "ClassAccessorProperty") {
      id = parent.key;
    }
  }
  if (!id) return;
  let nameInfo;
  if (id.type === "Identifier") {
    var _id$loc, _id$loc2;
    nameInfo = {
      pos: (_id$loc = id.loc) == null ? void 0 : _id$loc.start,
      name: ((_id$loc2 = id.loc) == null ? void 0 : _id$loc2.identifierName) || id.name
    };
  } else if (id.type === "PrivateName") {
    var _id$loc3;
    nameInfo = {
      pos: (_id$loc3 = id.loc) == null ? void 0 : _id$loc3.start,
      name: "#" + id.id.name
    };
  } else if (id.type === "StringLiteral") {
    var _id$loc4;
    nameInfo = {
      pos: (_id$loc4 = id.loc) == null ? void 0 : _id$loc4.start,
      name: id.value
    };
  }
  return nameInfo;
}

//# sourceMappingURL=methods.js.map
