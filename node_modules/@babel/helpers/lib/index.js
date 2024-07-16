"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = void 0;
exports.ensure = ensure;
exports.get = get;
exports.getDependencies = getDependencies;
exports.list = void 0;
exports.minVersion = minVersion;
var _traverse = require("@babel/traverse");
var _t = require("@babel/types");
var _helpers = require("./helpers.js");
const {
  assignmentExpression,
  cloneNode,
  expressionStatement,
  file,
  identifier
} = _t;
function makePath(path) {
  const parts = [];
  for (; path.parentPath; path = path.parentPath) {
    parts.push(path.key);
    if (path.inList) parts.push(path.listKey);
  }
  return parts.reverse().join(".");
}
let FileClass = undefined;
function getHelperMetadata(file) {
  const globals = new Set();
  const localBindingNames = new Set();
  const dependencies = new Map();
  let exportName;
  let exportPath;
  const exportBindingAssignments = [];
  const importPaths = [];
  const importBindingsReferences = [];
  const dependencyVisitor = {
    ImportDeclaration(child) {
      const name = child.node.source.value;
      if (!_helpers.default[name]) {
        throw child.buildCodeFrameError(`Unknown helper ${name}`);
      }
      if (child.get("specifiers").length !== 1 || !child.get("specifiers.0").isImportDefaultSpecifier()) {
        throw child.buildCodeFrameError("Helpers can only import a default value");
      }
      const bindingIdentifier = child.node.specifiers[0].local;
      dependencies.set(bindingIdentifier, name);
      importPaths.push(makePath(child));
    },
    ExportDefaultDeclaration(child) {
      const decl = child.get("declaration");
      if (!decl.isFunctionDeclaration() || !decl.node.id) {
        throw decl.buildCodeFrameError("Helpers can only export named function declarations");
      }
      exportName = decl.node.id.name;
      exportPath = makePath(child);
    },
    ExportAllDeclaration(child) {
      throw child.buildCodeFrameError("Helpers can only export default");
    },
    ExportNamedDeclaration(child) {
      throw child.buildCodeFrameError("Helpers can only export default");
    },
    Statement(child) {
      if (child.isImportDeclaration() || child.isExportDeclaration()) return;
      child.skip();
    }
  };
  const referenceVisitor = {
    Program(path) {
      const bindings = path.scope.getAllBindings();
      Object.keys(bindings).forEach(name => {
        if (name === exportName) return;
        if (dependencies.has(bindings[name].identifier)) return;
        localBindingNames.add(name);
      });
    },
    ReferencedIdentifier(child) {
      const name = child.node.name;
      const binding = child.scope.getBinding(name);
      if (!binding) {
        if (name !== "arguments" || child.scope.path.isProgram()) {
          globals.add(name);
        }
      } else if (dependencies.has(binding.identifier)) {
        importBindingsReferences.push(makePath(child));
      }
    },
    AssignmentExpression(child) {
      const left = child.get("left");
      if (!(exportName in left.getBindingIdentifiers())) return;
      if (!left.isIdentifier()) {
        throw left.buildCodeFrameError("Only simple assignments to exports are allowed in helpers");
      }
      const binding = child.scope.getBinding(exportName);
      if (binding != null && binding.scope.path.isProgram()) {
        exportBindingAssignments.push(makePath(child));
      }
    }
  };
  (0, _traverse.default)(file.ast, dependencyVisitor, file.scope);
  (0, _traverse.default)(file.ast, referenceVisitor, file.scope);
  if (!exportPath) throw new Error("Helpers must have a default export.");
  exportBindingAssignments.reverse();
  return {
    globals: Array.from(globals),
    localBindingNames: Array.from(localBindingNames),
    dependencies,
    exportBindingAssignments,
    exportPath,
    exportName,
    importBindingsReferences,
    importPaths
  };
}
function permuteHelperAST(file, metadata, id, localBindings, getDependency) {
  if (localBindings && !id) {
    throw new Error("Unexpected local bindings for module-based helpers.");
  }
  if (!id) return;
  const {
    localBindingNames,
    dependencies,
    exportBindingAssignments,
    exportPath,
    exportName,
    importBindingsReferences,
    importPaths
  } = metadata;
  const dependenciesRefs = {};
  dependencies.forEach((name, id) => {
    dependenciesRefs[id.name] = typeof getDependency === "function" && getDependency(name) || id;
  });
  const toRename = {};
  const bindings = new Set(localBindings || []);
  if (id.type === "Identifier") bindings.add(id.name);
  localBindingNames.forEach(name => {
    let newName = name;
    while (bindings.has(newName)) newName = "_" + newName;
    if (newName !== name) toRename[name] = newName;
  });
  if (id.type === "Identifier" && exportName !== id.name) {
    toRename[exportName] = id.name;
  }
  const {
    path
  } = file;
  const exp = path.get(exportPath);
  const imps = importPaths.map(p => path.get(p));
  const impsBindingRefs = importBindingsReferences.map(p => path.get(p));
  const decl = exp.get("declaration");
  if (id.type === "Identifier") {
    exp.replaceWith(decl);
  } else if (id.type === "MemberExpression") {
    exportBindingAssignments.forEach(assignPath => {
      const assign = path.get(assignPath);
      assign.replaceWith(assignmentExpression("=", id, assign.node));
    });
    exp.replaceWith(decl);
    path.pushContainer("body", expressionStatement(assignmentExpression("=", id, identifier(exportName))));
  } else {
    throw new Error("Unexpected helper format.");
  }
  Object.keys(toRename).forEach(name => {
    path.scope.rename(name, toRename[name]);
  });
  for (const path of imps) path.remove();
  for (const path of impsBindingRefs) {
    const node = cloneNode(dependenciesRefs[path.node.name]);
    path.replaceWith(node);
  }
}
const helperData = Object.create(null);
function loadHelper(name) {
  if (!helperData[name]) {
    const helper = _helpers.default[name];
    if (!helper) {
      throw Object.assign(new ReferenceError(`Unknown helper ${name}`), {
        code: "BABEL_HELPER_UNKNOWN",
        helper: name
      });
    }
    const fn = () => {
      {
        if (!FileClass) {
          const fakeFile = {
            ast: file(helper.ast()),
            path: null
          };
          (0, _traverse.default)(fakeFile.ast, {
            Program: path => (fakeFile.path = path).stop()
          });
          return fakeFile;
        }
      }
      return new FileClass({
        filename: `babel-helper://${name}`
      }, {
        ast: file(helper.ast()),
        code: "[internal Babel helper code]",
        inputMap: null
      });
    };
    let metadata = null;
    helperData[name] = {
      minVersion: helper.minVersion,
      build(getDependency, id, localBindings) {
        const file = fn();
        metadata || (metadata = getHelperMetadata(file));
        permuteHelperAST(file, metadata, id, localBindings, getDependency);
        return {
          nodes: file.ast.program.body,
          globals: metadata.globals
        };
      },
      getDependencies() {
        metadata || (metadata = getHelperMetadata(fn()));
        return Array.from(metadata.dependencies.values());
      }
    };
  }
  return helperData[name];
}
function get(name, getDependency, id, localBindings) {
  return loadHelper(name).build(getDependency, id, localBindings);
}
function minVersion(name) {
  return loadHelper(name).minVersion;
}
function getDependencies(name) {
  return loadHelper(name).getDependencies();
}
function ensure(name, newFileClass) {
  FileClass || (FileClass = newFileClass);
  loadHelper(name);
}
const list = exports.list = Object.keys(_helpers.default).map(name => name.replace(/^_/, ""));
var _default = exports.default = get;

//# sourceMappingURL=index.js.map
