"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.clear = clear;
exports.clearPath = clearPath;
exports.clearScope = clearScope;
exports.getCachedPaths = getCachedPaths;
exports.getOrCreateCachedPaths = getOrCreateCachedPaths;
exports.scope = exports.path = void 0;
let pathsCache = exports.path = new WeakMap();
let scope = exports.scope = new WeakMap();
function clear() {
  clearPath();
  clearScope();
}
function clearPath() {
  exports.path = pathsCache = new WeakMap();
}
function clearScope() {
  exports.scope = scope = new WeakMap();
}
const nullHub = Object.freeze({});
function getCachedPaths(hub, parent) {
  var _pathsCache$get, _hub;
  {
    hub = null;
  }
  return (_pathsCache$get = pathsCache.get((_hub = hub) != null ? _hub : nullHub)) == null ? void 0 : _pathsCache$get.get(parent);
}
function getOrCreateCachedPaths(hub, parent) {
  var _hub2, _hub3;
  {
    hub = null;
  }
  let parents = pathsCache.get((_hub2 = hub) != null ? _hub2 : nullHub);
  if (!parents) pathsCache.set((_hub3 = hub) != null ? _hub3 : nullHub, parents = new WeakMap());
  let paths = parents.get(parent);
  if (!paths) parents.set(parent, paths = new Map());
  return paths;
}

//# sourceMappingURL=cache.js.map
