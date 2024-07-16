var superPropBase = require("./superPropBase.js");
var defineProperty = require("./defineProperty.js");
function set(target, property, value, receiver) {
  if (typeof Reflect !== "undefined" && Reflect.set) {
    set = Reflect.set;
  } else {
    set = function set(target, property, value, receiver) {
      var base = superPropBase(target, property);
      var desc;
      if (base) {
        desc = Object.getOwnPropertyDescriptor(base, property);
        if (desc.set) {
          desc.set.call(receiver, value);
          return true;
        } else if (!desc.writable) {
          return false;
        }
      }
      desc = Object.getOwnPropertyDescriptor(receiver, property);
      if (desc) {
        if (!desc.writable) {
          return false;
        }
        desc.value = value;
        Object.defineProperty(receiver, property, desc);
      } else {
        defineProperty(receiver, property, value);
      }
      return true;
    };
  }
  return set(target, property, value, receiver);
}
function _set(target, property, value, receiver, isStrict) {
  var s = set(target, property, value, receiver || target);
  if (!s && isStrict) {
    throw new TypeError('failed to set property');
  }
  return value;
}
module.exports = _set, module.exports.__esModule = true, module.exports["default"] = module.exports;