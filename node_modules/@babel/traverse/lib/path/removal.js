"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports._assertUnremoved = _assertUnremoved;
exports._callRemovalHooks = _callRemovalHooks;
exports._markRemoved = _markRemoved;
exports._remove = _remove;
exports._removeFromScope = _removeFromScope;
exports.remove = remove;
var _removalHooks = require("./lib/removal-hooks.js");
var _cache = require("../cache.js");
var _index = require("./index.js");
var _t = require("@babel/types");
const {
  getBindingIdentifiers
} = _t;
function remove() {
  var _this$opts;
  this._assertUnremoved();
  this.resync();
  if (!((_this$opts = this.opts) != null && _this$opts.noScope)) {
    this._removeFromScope();
  }
  if (this._callRemovalHooks()) {
    this._markRemoved();
    return;
  }
  this.shareCommentsWithSiblings();
  this._remove();
  this._markRemoved();
}
function _removeFromScope() {
  const bindings = getBindingIdentifiers(this.node, false, false, true);
  Object.keys(bindings).forEach(name => this.scope.removeBinding(name));
}
function _callRemovalHooks() {
  if (this.parentPath) {
    for (const fn of _removalHooks.hooks) {
      if (fn(this, this.parentPath)) return true;
    }
  }
}
function _remove() {
  if (Array.isArray(this.container)) {
    this.container.splice(this.key, 1);
    this.updateSiblingKeys(this.key, -1);
  } else {
    this._replaceWith(null);
  }
}
function _markRemoved() {
  this._traverseFlags |= _index.SHOULD_SKIP | _index.REMOVED;
  if (this.parent) {
    (0, _cache.getCachedPaths)(this.hub, this.parent).delete(this.node);
  }
  this.node = null;
}
function _assertUnremoved() {
  if (this.removed) {
    throw this.buildCodeFrameError("NodePath has been removed so is read-only.");
  }
}

//# sourceMappingURL=removal.js.map
