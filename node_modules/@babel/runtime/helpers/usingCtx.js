function _usingCtx() {
  var r = "function" == typeof SuppressedError ? SuppressedError : function (r, n) {
      var e = Error();
      return e.name = "SuppressedError", e.error = r, e.suppressed = n, e;
    },
    n = {},
    e = [];
  function using(r, n) {
    if (null != n) {
      if (Object(n) !== n) throw new TypeError("using declarations can only be used with objects, functions, null, or undefined.");
      if (r) var o = n[Symbol.asyncDispose || Symbol["for"]("Symbol.asyncDispose")];
      if (null == o && (o = n[Symbol.dispose || Symbol["for"]("Symbol.dispose")]), "function" != typeof o) throw new TypeError("Property [Symbol.dispose] is not a function.");
      e.push({
        v: n,
        d: o,
        a: r
      });
    } else r && e.push({
      d: n,
      a: r
    });
    return n;
  }
  return {
    e: n,
    u: using.bind(null, !1),
    a: using.bind(null, !0),
    d: function d() {
      var o = this.e;
      function next() {
        for (; r = e.pop();) try {
          var r,
            t = r.d && r.d.call(r.v);
          if (r.a) return Promise.resolve(t).then(next, err);
        } catch (r) {
          return err(r);
        }
        if (o !== n) throw o;
      }
      function err(e) {
        return o = o !== n ? new r(e, o) : e, next();
      }
      return next();
    }
  };
}
module.exports = _usingCtx, module.exports.__esModule = true, module.exports["default"] = module.exports;