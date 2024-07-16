"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = _importDeferProxy;
function _importDeferProxy(init) {
  var ns = null;
  var constValue = function (v) {
    return function () {
      return v;
    };
  };
  var proxy = function (run) {
    return function (arg1, arg2, arg3) {
      if (ns === null) ns = init();
      return run(ns, arg2, arg3);
    };
  };
  return new Proxy({}, {
    defineProperty: constValue(false),
    deleteProperty: constValue(false),
    get: proxy(Reflect.get),
    getOwnPropertyDescriptor: proxy(Reflect.getOwnPropertyDescriptor),
    getPrototypeOf: constValue(null),
    isExtensible: constValue(false),
    has: proxy(Reflect.has),
    ownKeys: proxy(Reflect.ownKeys),
    preventExtensions: constValue(true),
    set: constValue(false),
    setPrototypeOf: constValue(false)
  });
}

//# sourceMappingURL=importDeferProxy.js.map
