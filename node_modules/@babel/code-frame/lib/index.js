"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.codeFrameColumns = codeFrameColumns;
exports.default = _default;
var _highlight = require("@babel/highlight");
var _picocolors = _interopRequireWildcard(require("picocolors"), true);
function _getRequireWildcardCache(e) { if ("function" != typeof WeakMap) return null; var r = new WeakMap(), t = new WeakMap(); return (_getRequireWildcardCache = function (e) { return e ? t : r; })(e); }
function _interopRequireWildcard(e, r) { if (!r && e && e.__esModule) return e; if (null === e || "object" != typeof e && "function" != typeof e) return { default: e }; var t = _getRequireWildcardCache(r); if (t && t.has(e)) return t.get(e); var n = { __proto__: null }, a = Object.defineProperty && Object.getOwnPropertyDescriptor; for (var u in e) if ("default" !== u && Object.prototype.hasOwnProperty.call(e, u)) { var i = a ? Object.getOwnPropertyDescriptor(e, u) : null; i && (i.get || i.set) ? Object.defineProperty(n, u, i) : n[u] = e[u]; } return n.default = e, t && t.set(e, n), n; }
const colors = typeof process === "object" && (process.env.FORCE_COLOR === "0" || process.env.FORCE_COLOR === "false") ? (0, _picocolors.createColors)(false) : _picocolors.default;
const compose = (f, g) => v => f(g(v));
let pcWithForcedColor = undefined;
function getColors(forceColor) {
  if (forceColor) {
    var _pcWithForcedColor;
    (_pcWithForcedColor = pcWithForcedColor) != null ? _pcWithForcedColor : pcWithForcedColor = (0, _picocolors.createColors)(true);
    return pcWithForcedColor;
  }
  return colors;
}
let deprecationWarningShown = false;
function getDefs(colors) {
  return {
    gutter: colors.gray,
    marker: compose(colors.red, colors.bold),
    message: compose(colors.red, colors.bold)
  };
}
const NEWLINE = /\r\n|[\n\r\u2028\u2029]/;
function getMarkerLines(loc, source, opts) {
  const startLoc = Object.assign({
    column: 0,
    line: -1
  }, loc.start);
  const endLoc = Object.assign({}, startLoc, loc.end);
  const {
    linesAbove = 2,
    linesBelow = 3
  } = opts || {};
  const startLine = startLoc.line;
  const startColumn = startLoc.column;
  const endLine = endLoc.line;
  const endColumn = endLoc.column;
  let start = Math.max(startLine - (linesAbove + 1), 0);
  let end = Math.min(source.length, endLine + linesBelow);
  if (startLine === -1) {
    start = 0;
  }
  if (endLine === -1) {
    end = source.length;
  }
  const lineDiff = endLine - startLine;
  const markerLines = {};
  if (lineDiff) {
    for (let i = 0; i <= lineDiff; i++) {
      const lineNumber = i + startLine;
      if (!startColumn) {
        markerLines[lineNumber] = true;
      } else if (i === 0) {
        const sourceLength = source[lineNumber - 1].length;
        markerLines[lineNumber] = [startColumn, sourceLength - startColumn + 1];
      } else if (i === lineDiff) {
        markerLines[lineNumber] = [0, endColumn];
      } else {
        const sourceLength = source[lineNumber - i].length;
        markerLines[lineNumber] = [0, sourceLength];
      }
    }
  } else {
    if (startColumn === endColumn) {
      if (startColumn) {
        markerLines[startLine] = [startColumn, 0];
      } else {
        markerLines[startLine] = true;
      }
    } else {
      markerLines[startLine] = [startColumn, endColumn - startColumn];
    }
  }
  return {
    start,
    end,
    markerLines
  };
}
function codeFrameColumns(rawLines, loc, opts = {}) {
  const highlighted = (opts.highlightCode || opts.forceColor) && (0, _highlight.shouldHighlight)(opts);
  const colors = getColors(opts.forceColor);
  const defs = getDefs(colors);
  const maybeHighlight = (fmt, string) => {
    return highlighted ? fmt(string) : string;
  };
  const lines = rawLines.split(NEWLINE);
  const {
    start,
    end,
    markerLines
  } = getMarkerLines(loc, lines, opts);
  const hasColumns = loc.start && typeof loc.start.column === "number";
  const numberMaxWidth = String(end).length;
  const highlightedLines = highlighted ? (0, _highlight.default)(rawLines, opts) : rawLines;
  let frame = highlightedLines.split(NEWLINE, end).slice(start, end).map((line, index) => {
    const number = start + 1 + index;
    const paddedNumber = ` ${number}`.slice(-numberMaxWidth);
    const gutter = ` ${paddedNumber} |`;
    const hasMarker = markerLines[number];
    const lastMarkerLine = !markerLines[number + 1];
    if (hasMarker) {
      let markerLine = "";
      if (Array.isArray(hasMarker)) {
        const markerSpacing = line.slice(0, Math.max(hasMarker[0] - 1, 0)).replace(/[^\t]/g, " ");
        const numberOfMarkers = hasMarker[1] || 1;
        markerLine = ["\n ", maybeHighlight(defs.gutter, gutter.replace(/\d/g, " ")), " ", markerSpacing, maybeHighlight(defs.marker, "^").repeat(numberOfMarkers)].join("");
        if (lastMarkerLine && opts.message) {
          markerLine += " " + maybeHighlight(defs.message, opts.message);
        }
      }
      return [maybeHighlight(defs.marker, ">"), maybeHighlight(defs.gutter, gutter), line.length > 0 ? ` ${line}` : "", markerLine].join("");
    } else {
      return ` ${maybeHighlight(defs.gutter, gutter)}${line.length > 0 ? ` ${line}` : ""}`;
    }
  }).join("\n");
  if (opts.message && !hasColumns) {
    frame = `${" ".repeat(numberMaxWidth + 1)}${opts.message}\n${frame}`;
  }
  if (highlighted) {
    return colors.reset(frame);
  } else {
    return frame;
  }
}
function _default(rawLines, lineNumber, colNumber, opts = {}) {
  if (!deprecationWarningShown) {
    deprecationWarningShown = true;
    const message = "Passing lineNumber and colNumber is deprecated to @babel/code-frame. Please use `codeFrameColumns`.";
    if (process.emitWarning) {
      process.emitWarning(message, "DeprecationWarning");
    } else {
      const deprecationError = new Error(message);
      deprecationError.name = "DeprecationWarning";
      console.warn(new Error(message));
    }
  }
  colNumber = Math.max(colNumber, 0);
  const location = {
    start: {
      column: colNumber,
      line: lineNumber
    }
  };
  return codeFrameColumns(rawLines, location, opts);
}

//# sourceMappingURL=index.js.map
