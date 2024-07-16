"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = _iterableToArrayLimit;
function _iterableToArrayLimit(arr, i) {
  var _i = arr == null ? null : typeof Symbol !== "undefined" && arr[Symbol.iterator] || arr["@@iterator"];
  if (_i == null) return;
  var _arr = [];
  var _n = true;
  var _d = false;
  var _s, _e, _x, _r;
  try {
    _x = (_i = _i.call(arr)).next;
    if (i === 0) {
      if (Object(_i) !== _i) return;
      _n = false;
    } else {
      for (; !(_n = (_s = _x.call(_i)).done); _n = true) {
        _arr.push(_s.value);
        if (_arr.length === i) break;
      }
    }
  } catch (err) {
    _d = true;
    _e = err;
  } finally {
    try {
      if (!_n && _i["return"] != null) {
        _r = _i["return"]();
        if (Object(_r) !== _r) return;
      }
    } finally {
      if (_d) throw _e;
    }
  }
  return _arr;
}

//# sourceMappingURL=iterableToArrayLimit.js.map
