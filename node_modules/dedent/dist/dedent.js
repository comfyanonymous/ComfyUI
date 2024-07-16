"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = void 0;
const dedent = createDedent({});
var _default = exports.default = dedent;
function createDedent(options) {
  dedent.withOptions = newOptions => createDedent({
    ...options,
    ...newOptions
  });
  return dedent;
  function dedent(strings, ...values) {
    const raw = typeof strings === "string" ? [strings] : strings.raw;
    const {
      escapeSpecialCharacters = Array.isArray(strings)
    } = options;

    // first, perform interpolation
    let result = "";
    for (let i = 0; i < raw.length; i++) {
      let next = raw[i];
      if (escapeSpecialCharacters) {
        // handle escaped newlines, backticks, and interpolation characters
        next = next.replace(/\\\n[ \t]*/g, "").replace(/\\`/g, "`").replace(/\\\$/g, "$").replace(/\\\{/g, "{");
      }
      result += next;
      if (i < values.length) {
        // eslint-disable-next-line @typescript-eslint/restrict-plus-operands
        result += values[i];
      }
    }

    // now strip indentation
    const lines = result.split("\n");
    let mindent = null;
    for (const l of lines) {
      const m = l.match(/^(\s+)\S+/);
      if (m) {
        const indent = m[1].length;
        if (!mindent) {
          // this is the first indented line
          mindent = indent;
        } else {
          mindent = Math.min(mindent, indent);
        }
      }
    }
    if (mindent !== null) {
      const m = mindent; // appease TypeScript
      result = lines
      // https://github.com/typescript-eslint/typescript-eslint/issues/7140
      // eslint-disable-next-line @typescript-eslint/prefer-string-starts-ends-with
      .map(l => l[0] === " " || l[0] === "\t" ? l.slice(m) : l).join("\n");
    }

    // dedent eats leading and trailing whitespace too
    result = result.trim();
    if (escapeSpecialCharacters) {
      // handle escaped newlines at the end to ensure they don't get stripped too
      result = result.replace(/\\n/g, "\n");
    }
    return result;
  }
}
module.exports = exports.default;
module.exports.default = exports.default;
