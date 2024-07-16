/* eslint-disable import/no-extraneous-dependencies */

import fs from "fs";
import { createRequire } from "module";

const [parse, generate] = await Promise.all([
  import("@babel/parser").then(ns => ns.parse),
  import("@babel/generator").then(ns => ns.default.default || ns.default),
]).catch(error => {
  console.error(error);
  throw new Error(
    "Before running generate-helpers.js you must compile @babel/parser and @babel/generator.",
    { cause: error }
  );
});

const REGENERATOR_RUNTIME_IN_FILE = fs.readFileSync(
  createRequire(import.meta.url).resolve("regenerator-runtime"),
  "utf8"
);

const MIN_VERSION = "7.18.0";

const HEADER = `/* @minVersion ${MIN_VERSION} */
/*
 * This file is auto-generated! Do not modify it directly.
 * To re-generate, update the regenerator-runtime dependency of
 * @babel/helpers and run 'yarn gulp generate-runtime-helpers'.
 */

/* eslint-disable */
`;

const COPYRIGHT = `/*! regenerator-runtime -- Copyright (c) 2014-present, Facebook, Inc. -- license (MIT): https://github.com/facebook/regenerator/blob/main/LICENSE */`;

export default function generateRegeneratorRuntimeHelper() {
  const ast = parse(REGENERATOR_RUNTIME_IN_FILE, { sourceType: "script" });

  const factoryFunction = ast.program.body[0].declarations[0].init.callee;
  factoryFunction.type = "FunctionDeclaration";
  factoryFunction.id = { type: "Identifier", name: "_regeneratorRuntime" };
  factoryFunction.params = [];
  factoryFunction.body.body.unshift(
    ...stmts(`
      ${COPYRIGHT}
      _regeneratorRuntime = function () { return exports; };
      var exports = {};
    `)
  );

  const { code } = generate({
    type: "ExportDefaultDeclaration",
    declaration: factoryFunction,
  });

  return HEADER + code;
}

function stmts(code) {
  return parse(`function _() { ${code} }`, {
    sourceType: "script",
  }).program.body[0].body.body;
}
