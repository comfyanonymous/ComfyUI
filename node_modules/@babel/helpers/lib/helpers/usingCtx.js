"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = _usingCtx;
function _usingCtx() {
  var _disposeSuppressedError = typeof SuppressedError === "function" ? SuppressedError : function (error, suppressed) {
      var err = new Error();
      err.name = "SuppressedError";
      err.error = error;
      err.suppressed = suppressed;
      return err;
    },
    empty = {},
    stack = [];
  function using(isAwait, value) {
    if (value != null) {
      if (Object(value) !== value) {
        throw new TypeError("using declarations can only be used with objects, functions, null, or undefined.");
      }
      if (isAwait) {
        var dispose = value[Symbol.asyncDispose || Symbol.for("Symbol.asyncDispose")];
      }
      if (dispose == null) {
        dispose = value[Symbol.dispose || Symbol.for("Symbol.dispose")];
      }
      if (typeof dispose !== "function") {
        throw new TypeError(`Property [Symbol.dispose] is not a function.`);
      }
      stack.push({
        v: value,
        d: dispose,
        a: isAwait
      });
    } else if (isAwait) {
      stack.push({
        d: value,
        a: isAwait
      });
    }
    return value;
  }
  return {
    e: empty,
    u: using.bind(null, false),
    a: using.bind(null, true),
    d: function () {
      var error = this.e;
      function next() {
        while (resource = stack.pop()) {
          try {
            var resource,
              disposalResult = resource.d && resource.d.call(resource.v);
            if (resource.a) {
              return Promise.resolve(disposalResult).then(next, err);
            }
          } catch (e) {
            return err(e);
          }
        }
        if (error !== empty) throw error;
      }
      function err(e) {
        error = error !== empty ? new _disposeSuppressedError(e, error) : e;
        return next();
      }
      return next();
    }
  };
}

//# sourceMappingURL=usingCtx.js.map
