"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = void 0;
var _helperEnvironmentVisitor = require("@babel/helper-environment-visitor");
var _helperMemberExpressionToFunctions = require("@babel/helper-member-expression-to-functions");
var _helperOptimiseCallExpression = require("@babel/helper-optimise-call-expression");
var _core = require("@babel/core");
const {
  assignmentExpression,
  booleanLiteral,
  callExpression,
  cloneNode,
  identifier,
  memberExpression,
  sequenceExpression,
  stringLiteral,
  thisExpression
} = _core.types;
{
  const ns = require("@babel/helper-environment-visitor");
  exports.environmentVisitor = ns.default;
  exports.skipAllButComputedKey = ns.skipAllButComputedKey;
}
function getPrototypeOfExpression(objectRef, isStatic, file, isPrivateMethod) {
  objectRef = cloneNode(objectRef);
  const targetRef = isStatic || isPrivateMethod ? objectRef : memberExpression(objectRef, identifier("prototype"));
  return callExpression(file.addHelper("getPrototypeOf"), [targetRef]);
}
const visitor = _core.traverse.visitors.merge([_helperEnvironmentVisitor.default, {
  Super(path, state) {
    const {
      node,
      parentPath
    } = path;
    if (!parentPath.isMemberExpression({
      object: node
    })) return;
    state.handle(parentPath);
  }
}]);
const unshadowSuperBindingVisitor = _core.traverse.visitors.merge([_helperEnvironmentVisitor.default, {
  Scopable(path, {
    refName
  }) {
    const binding = path.scope.getOwnBinding(refName);
    if (binding && binding.identifier.name === refName) {
      path.scope.rename(refName);
    }
  }
}]);
const specHandlers = {
  memoise(superMember, count) {
    const {
      scope,
      node
    } = superMember;
    const {
      computed,
      property
    } = node;
    if (!computed) {
      return;
    }
    const memo = scope.maybeGenerateMemoised(property);
    if (!memo) {
      return;
    }
    this.memoiser.set(property, memo, count);
  },
  prop(superMember) {
    const {
      computed,
      property
    } = superMember.node;
    if (this.memoiser.has(property)) {
      return cloneNode(this.memoiser.get(property));
    }
    if (computed) {
      return cloneNode(property);
    }
    return stringLiteral(property.name);
  },
  get(superMember) {
    return this._get(superMember, this._getThisRefs());
  },
  _get(superMember, thisRefs) {
    const proto = getPrototypeOfExpression(this.getObjectRef(), this.isStatic, this.file, this.isPrivateMethod);
    return callExpression(this.file.addHelper("get"), [thisRefs.needAccessFirst ? sequenceExpression([thisRefs.this, proto]) : proto, this.prop(superMember), thisRefs.this]);
  },
  _getThisRefs() {
    return {
      needAccessFirst: this.isDerivedConstructor,
      this: thisExpression()
    };
  },
  set(superMember, value) {
    const thisRefs = this._getThisRefs();
    const proto = getPrototypeOfExpression(this.getObjectRef(), this.isStatic, this.file, this.isPrivateMethod);
    return callExpression(this.file.addHelper("set"), [thisRefs.needAccessFirst ? sequenceExpression([thisRefs.this, proto]) : proto, this.prop(superMember), value, thisRefs.this, booleanLiteral(superMember.isInStrictMode())]);
  },
  destructureSet(superMember) {
    throw superMember.buildCodeFrameError(`Destructuring to a super field is not supported yet.`);
  },
  call(superMember, args) {
    const thisRefs = this._getThisRefs();
    return (0, _helperOptimiseCallExpression.default)(this._get(superMember, thisRefs), cloneNode(thisRefs.this), args, false);
  },
  optionalCall(superMember, args) {
    const thisRefs = this._getThisRefs();
    return (0, _helperOptimiseCallExpression.default)(this._get(superMember, thisRefs), cloneNode(thisRefs.this), args, true);
  },
  delete(superMember) {
    if (superMember.node.computed) {
      return sequenceExpression([callExpression(this.file.addHelper("toPropertyKey"), [cloneNode(superMember.node.property)]), _core.template.expression.ast`
          function () { throw new ReferenceError("'delete super[expr]' is invalid"); }()
        `]);
    } else {
      return _core.template.expression.ast`
        function () { throw new ReferenceError("'delete super.prop' is invalid"); }()
      `;
    }
  }
};
const looseHandlers = Object.assign({}, specHandlers, {
  prop(superMember) {
    const {
      property
    } = superMember.node;
    if (this.memoiser.has(property)) {
      return cloneNode(this.memoiser.get(property));
    }
    return cloneNode(property);
  },
  get(superMember) {
    const {
      isStatic,
      getSuperRef
    } = this;
    const {
      computed
    } = superMember.node;
    const prop = this.prop(superMember);
    let object;
    if (isStatic) {
      var _getSuperRef;
      object = (_getSuperRef = getSuperRef()) != null ? _getSuperRef : memberExpression(identifier("Function"), identifier("prototype"));
    } else {
      var _getSuperRef2;
      object = memberExpression((_getSuperRef2 = getSuperRef()) != null ? _getSuperRef2 : identifier("Object"), identifier("prototype"));
    }
    return memberExpression(object, prop, computed);
  },
  set(superMember, value) {
    const {
      computed
    } = superMember.node;
    const prop = this.prop(superMember);
    return assignmentExpression("=", memberExpression(thisExpression(), prop, computed), value);
  },
  destructureSet(superMember) {
    const {
      computed
    } = superMember.node;
    const prop = this.prop(superMember);
    return memberExpression(thisExpression(), prop, computed);
  },
  call(superMember, args) {
    return (0, _helperOptimiseCallExpression.default)(this.get(superMember), thisExpression(), args, false);
  },
  optionalCall(superMember, args) {
    return (0, _helperOptimiseCallExpression.default)(this.get(superMember), thisExpression(), args, true);
  }
});
class ReplaceSupers {
  constructor(opts) {
    var _opts$constantSuper;
    const path = opts.methodPath;
    this.methodPath = path;
    this.isDerivedConstructor = path.isClassMethod({
      kind: "constructor"
    }) && !!opts.superRef;
    this.isStatic = path.isObjectMethod() || path.node.static || (path.isStaticBlock == null ? void 0 : path.isStaticBlock());
    this.isPrivateMethod = path.isPrivate() && path.isMethod();
    this.file = opts.file;
    this.constantSuper = (_opts$constantSuper = opts.constantSuper) != null ? _opts$constantSuper : opts.isLoose;
    this.opts = opts;
  }
  getObjectRef() {
    return cloneNode(this.opts.objectRef || this.opts.getObjectRef());
  }
  getSuperRef() {
    if (this.opts.superRef) return cloneNode(this.opts.superRef);
    if (this.opts.getSuperRef) {
      return cloneNode(this.opts.getSuperRef());
    }
  }
  replace() {
    const {
      methodPath
    } = this;
    if (this.opts.refToPreserve) {
      methodPath.traverse(unshadowSuperBindingVisitor, {
        refName: this.opts.refToPreserve.name
      });
    }
    const handler = this.constantSuper ? looseHandlers : specHandlers;
    visitor.shouldSkip = path => {
      if (path.parentPath === methodPath) {
        if (path.parentKey === "decorators" || path.parentKey === "key") {
          return true;
        }
      }
    };
    (0, _helperMemberExpressionToFunctions.default)(methodPath, visitor, Object.assign({
      file: this.file,
      scope: this.methodPath.scope,
      isDerivedConstructor: this.isDerivedConstructor,
      isStatic: this.isStatic,
      isPrivateMethod: this.isPrivateMethod,
      getObjectRef: this.getObjectRef.bind(this),
      getSuperRef: this.getSuperRef.bind(this),
      boundGet: handler.get
    }, handler));
  }
}
exports.default = ReplaceSupers;

//# sourceMappingURL=index.js.map
