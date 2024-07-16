"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = generate;
var _sourceMap = require("./source-map.js");
var _printer = require("./printer.js");
function normalizeOptions(code, opts) {
  var _opts$recordAndTupleS;
  const format = {
    auxiliaryCommentBefore: opts.auxiliaryCommentBefore,
    auxiliaryCommentAfter: opts.auxiliaryCommentAfter,
    shouldPrintComment: opts.shouldPrintComment,
    retainLines: opts.retainLines,
    retainFunctionParens: opts.retainFunctionParens,
    comments: opts.comments == null || opts.comments,
    compact: opts.compact,
    minified: opts.minified,
    concise: opts.concise,
    indent: {
      adjustMultilineComment: true,
      style: "  "
    },
    jsescOption: Object.assign({
      quotes: "double",
      wrap: true,
      minimal: false
    }, opts.jsescOption),
    recordAndTupleSyntaxType: (_opts$recordAndTupleS = opts.recordAndTupleSyntaxType) != null ? _opts$recordAndTupleS : "hash",
    topicToken: opts.topicToken,
    importAttributesKeyword: opts.importAttributesKeyword
  };
  {
    format.decoratorsBeforeExport = opts.decoratorsBeforeExport;
    format.jsescOption.json = opts.jsonCompatibleStrings;
  }
  if (format.minified) {
    format.compact = true;
    format.shouldPrintComment = format.shouldPrintComment || (() => format.comments);
  } else {
    format.shouldPrintComment = format.shouldPrintComment || (value => format.comments || value.includes("@license") || value.includes("@preserve"));
  }
  if (format.compact === "auto") {
    format.compact = typeof code === "string" && code.length > 500000;
    if (format.compact) {
      console.error("[BABEL] Note: The code generator has deoptimised the styling of " + `${opts.filename} as it exceeds the max of ${"500KB"}.`);
    }
  }
  if (format.compact) {
    format.indent.adjustMultilineComment = false;
  }
  const {
    auxiliaryCommentBefore,
    auxiliaryCommentAfter,
    shouldPrintComment
  } = format;
  if (auxiliaryCommentBefore && !shouldPrintComment(auxiliaryCommentBefore)) {
    format.auxiliaryCommentBefore = undefined;
  }
  if (auxiliaryCommentAfter && !shouldPrintComment(auxiliaryCommentAfter)) {
    format.auxiliaryCommentAfter = undefined;
  }
  return format;
}
{
  exports.CodeGenerator = class CodeGenerator {
    constructor(ast, opts = {}, code) {
      this._ast = void 0;
      this._format = void 0;
      this._map = void 0;
      this._ast = ast;
      this._format = normalizeOptions(code, opts);
      this._map = opts.sourceMaps ? new _sourceMap.default(opts, code) : null;
    }
    generate() {
      const printer = new _printer.default(this._format, this._map);
      return printer.generate(this._ast);
    }
  };
}
function generate(ast, opts = {}, code) {
  const format = normalizeOptions(code, opts);
  const map = opts.sourceMaps ? new _sourceMap.default(opts, code) : null;
  const printer = new _printer.default(format, map);
  return printer.generate(ast);
}

//# sourceMappingURL=index.js.map
