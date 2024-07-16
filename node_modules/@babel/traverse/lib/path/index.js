"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = exports.SHOULD_STOP = exports.SHOULD_SKIP = exports.REMOVED = void 0;
var virtualTypes = require("./lib/virtual-types.js");
var _debug = require("debug");
var _index = require("../index.js");
var _index2 = require("../scope/index.js");
var _t = require("@babel/types");
var t = _t;
var cache = require("../cache.js");
var _generator = require("@babel/generator");
var NodePath_ancestry = require("./ancestry.js");
var NodePath_inference = require("./inference/index.js");
var NodePath_replacement = require("./replacement.js");
var NodePath_evaluation = require("./evaluation.js");
var NodePath_conversion = require("./conversion.js");
var NodePath_introspection = require("./introspection.js");
var NodePath_context = require("./context.js");
var NodePath_removal = require("./removal.js");
var NodePath_modification = require("./modification.js");
var NodePath_family = require("./family.js");
var NodePath_comments = require("./comments.js");
var NodePath_virtual_types_validator = require("./lib/virtual-types-validator.js");
const {
  validate
} = _t;
const debug = _debug("babel");
const REMOVED = exports.REMOVED = 1 << 0;
const SHOULD_STOP = exports.SHOULD_STOP = 1 << 1;
const SHOULD_SKIP = exports.SHOULD_SKIP = 1 << 2;
const NodePath_Final = exports.default = class NodePath {
  constructor(hub, parent) {
    this.contexts = [];
    this.state = null;
    this.opts = null;
    this._traverseFlags = 0;
    this.skipKeys = null;
    this.parentPath = null;
    this.container = null;
    this.listKey = null;
    this.key = null;
    this.node = null;
    this.type = null;
    this.parent = parent;
    this.hub = hub;
    this.data = null;
    this.context = null;
    this.scope = null;
  }
  static get({
    hub,
    parentPath,
    parent,
    container,
    listKey,
    key
  }) {
    if (!hub && parentPath) {
      hub = parentPath.hub;
    }
    if (!parent) {
      throw new Error("To get a node path the parent needs to exist");
    }
    const targetNode = container[key];
    const paths = cache.getOrCreateCachedPaths(hub, parent);
    let path = paths.get(targetNode);
    if (!path) {
      path = new NodePath(hub, parent);
      if (targetNode) paths.set(targetNode, path);
    }
    path.setup(parentPath, container, listKey, key);
    return path;
  }
  getScope(scope) {
    return this.isScope() ? new _index2.default(this) : scope;
  }
  setData(key, val) {
    if (this.data == null) {
      this.data = Object.create(null);
    }
    return this.data[key] = val;
  }
  getData(key, def) {
    if (this.data == null) {
      this.data = Object.create(null);
    }
    let val = this.data[key];
    if (val === undefined && def !== undefined) val = this.data[key] = def;
    return val;
  }
  hasNode() {
    return this.node != null;
  }
  buildCodeFrameError(msg, Error = SyntaxError) {
    return this.hub.buildError(this.node, msg, Error);
  }
  traverse(visitor, state) {
    (0, _index.default)(this.node, visitor, this.scope, state, this);
  }
  set(key, node) {
    validate(this.node, key, node);
    this.node[key] = node;
  }
  getPathLocation() {
    const parts = [];
    let path = this;
    do {
      let key = path.key;
      if (path.inList) key = `${path.listKey}[${key}]`;
      parts.unshift(key);
    } while (path = path.parentPath);
    return parts.join(".");
  }
  debug(message) {
    if (!debug.enabled) return;
    debug(`${this.getPathLocation()} ${this.type}: ${message}`);
  }
  toString() {
    return (0, _generator.default)(this.node).code;
  }
  get inList() {
    return !!this.listKey;
  }
  set inList(inList) {
    if (!inList) {
      this.listKey = null;
    }
  }
  get parentKey() {
    return this.listKey || this.key;
  }
  get shouldSkip() {
    return !!(this._traverseFlags & SHOULD_SKIP);
  }
  set shouldSkip(v) {
    if (v) {
      this._traverseFlags |= SHOULD_SKIP;
    } else {
      this._traverseFlags &= ~SHOULD_SKIP;
    }
  }
  get shouldStop() {
    return !!(this._traverseFlags & SHOULD_STOP);
  }
  set shouldStop(v) {
    if (v) {
      this._traverseFlags |= SHOULD_STOP;
    } else {
      this._traverseFlags &= ~SHOULD_STOP;
    }
  }
  get removed() {
    return !!(this._traverseFlags & REMOVED);
  }
  set removed(v) {
    if (v) {
      this._traverseFlags |= REMOVED;
    } else {
      this._traverseFlags &= ~REMOVED;
    }
  }
};
Object.assign(NodePath_Final.prototype, NodePath_ancestry, NodePath_inference, NodePath_replacement, NodePath_evaluation, NodePath_conversion, NodePath_introspection, NodePath_context, NodePath_removal, NodePath_modification, NodePath_family, NodePath_comments);
{
  NodePath_Final.prototype._guessExecutionStatusRelativeToDifferentFunctions = NodePath_introspection._guessExecutionStatusRelativeTo;
}
for (const type of t.TYPES) {
  const typeKey = `is${type}`;
  const fn = t[typeKey];
  NodePath_Final.prototype[typeKey] = function (opts) {
    return fn(this.node, opts);
  };
  NodePath_Final.prototype[`assert${type}`] = function (opts) {
    if (!fn(this.node, opts)) {
      throw new TypeError(`Expected node path of type ${type}`);
    }
  };
}
Object.assign(NodePath_Final.prototype, NodePath_virtual_types_validator);
for (const type of Object.keys(virtualTypes)) {
  if (type[0] === "_") continue;
  if (!t.TYPES.includes(type)) t.TYPES.push(type);
}

//# sourceMappingURL=index.js.map
