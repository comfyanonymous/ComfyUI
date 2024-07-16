"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.BreakStatement = BreakStatement;
exports.CatchClause = CatchClause;
exports.ContinueStatement = ContinueStatement;
exports.DebuggerStatement = DebuggerStatement;
exports.DoWhileStatement = DoWhileStatement;
exports.ForOfStatement = exports.ForInStatement = void 0;
exports.ForStatement = ForStatement;
exports.IfStatement = IfStatement;
exports.LabeledStatement = LabeledStatement;
exports.ReturnStatement = ReturnStatement;
exports.SwitchCase = SwitchCase;
exports.SwitchStatement = SwitchStatement;
exports.ThrowStatement = ThrowStatement;
exports.TryStatement = TryStatement;
exports.VariableDeclaration = VariableDeclaration;
exports.VariableDeclarator = VariableDeclarator;
exports.WhileStatement = WhileStatement;
exports.WithStatement = WithStatement;
var _t = require("@babel/types");
const {
  isFor,
  isForStatement,
  isIfStatement,
  isStatement
} = _t;
function WithStatement(node) {
  this.word("with");
  this.space();
  this.tokenChar(40);
  this.print(node.object, node);
  this.tokenChar(41);
  this.printBlock(node);
}
function IfStatement(node) {
  this.word("if");
  this.space();
  this.tokenChar(40);
  this.print(node.test, node);
  this.tokenChar(41);
  this.space();
  const needsBlock = node.alternate && isIfStatement(getLastStatement(node.consequent));
  if (needsBlock) {
    this.tokenChar(123);
    this.newline();
    this.indent();
  }
  this.printAndIndentOnComments(node.consequent, node);
  if (needsBlock) {
    this.dedent();
    this.newline();
    this.tokenChar(125);
  }
  if (node.alternate) {
    if (this.endsWith(125)) this.space();
    this.word("else");
    this.space();
    this.printAndIndentOnComments(node.alternate, node);
  }
}
function getLastStatement(statement) {
  const {
    body
  } = statement;
  if (isStatement(body) === false) {
    return statement;
  }
  return getLastStatement(body);
}
function ForStatement(node) {
  this.word("for");
  this.space();
  this.tokenChar(40);
  this.inForStatementInitCounter++;
  this.print(node.init, node);
  this.inForStatementInitCounter--;
  this.tokenChar(59);
  if (node.test) {
    this.space();
    this.print(node.test, node);
  }
  this.tokenChar(59);
  if (node.update) {
    this.space();
    this.print(node.update, node);
  }
  this.tokenChar(41);
  this.printBlock(node);
}
function WhileStatement(node) {
  this.word("while");
  this.space();
  this.tokenChar(40);
  this.print(node.test, node);
  this.tokenChar(41);
  this.printBlock(node);
}
function ForXStatement(node) {
  this.word("for");
  this.space();
  const isForOf = node.type === "ForOfStatement";
  if (isForOf && node.await) {
    this.word("await");
    this.space();
  }
  this.noIndentInnerCommentsHere();
  this.tokenChar(40);
  this.print(node.left, node);
  this.space();
  this.word(isForOf ? "of" : "in");
  this.space();
  this.print(node.right, node);
  this.tokenChar(41);
  this.printBlock(node);
}
const ForInStatement = exports.ForInStatement = ForXStatement;
const ForOfStatement = exports.ForOfStatement = ForXStatement;
function DoWhileStatement(node) {
  this.word("do");
  this.space();
  this.print(node.body, node);
  this.space();
  this.word("while");
  this.space();
  this.tokenChar(40);
  this.print(node.test, node);
  this.tokenChar(41);
  this.semicolon();
}
function printStatementAfterKeyword(printer, node, parent, isLabel) {
  if (node) {
    printer.space();
    printer.printTerminatorless(node, parent, isLabel);
  }
  printer.semicolon();
}
function BreakStatement(node) {
  this.word("break");
  printStatementAfterKeyword(this, node.label, node, true);
}
function ContinueStatement(node) {
  this.word("continue");
  printStatementAfterKeyword(this, node.label, node, true);
}
function ReturnStatement(node) {
  this.word("return");
  printStatementAfterKeyword(this, node.argument, node, false);
}
function ThrowStatement(node) {
  this.word("throw");
  printStatementAfterKeyword(this, node.argument, node, false);
}
function LabeledStatement(node) {
  this.print(node.label, node);
  this.tokenChar(58);
  this.space();
  this.print(node.body, node);
}
function TryStatement(node) {
  this.word("try");
  this.space();
  this.print(node.block, node);
  this.space();
  if (node.handlers) {
    this.print(node.handlers[0], node);
  } else {
    this.print(node.handler, node);
  }
  if (node.finalizer) {
    this.space();
    this.word("finally");
    this.space();
    this.print(node.finalizer, node);
  }
}
function CatchClause(node) {
  this.word("catch");
  this.space();
  if (node.param) {
    this.tokenChar(40);
    this.print(node.param, node);
    this.print(node.param.typeAnnotation, node);
    this.tokenChar(41);
    this.space();
  }
  this.print(node.body, node);
}
function SwitchStatement(node) {
  this.word("switch");
  this.space();
  this.tokenChar(40);
  this.print(node.discriminant, node);
  this.tokenChar(41);
  this.space();
  this.tokenChar(123);
  this.printSequence(node.cases, node, {
    indent: true,
    addNewlines(leading, cas) {
      if (!leading && node.cases[node.cases.length - 1] === cas) return -1;
    }
  });
  this.rightBrace(node);
}
function SwitchCase(node) {
  if (node.test) {
    this.word("case");
    this.space();
    this.print(node.test, node);
    this.tokenChar(58);
  } else {
    this.word("default");
    this.tokenChar(58);
  }
  if (node.consequent.length) {
    this.newline();
    this.printSequence(node.consequent, node, {
      indent: true
    });
  }
}
function DebuggerStatement() {
  this.word("debugger");
  this.semicolon();
}
function VariableDeclaration(node, parent) {
  if (node.declare) {
    this.word("declare");
    this.space();
  }
  const {
    kind
  } = node;
  this.word(kind, kind === "using" || kind === "await using");
  this.space();
  let hasInits = false;
  if (!isFor(parent)) {
    for (const declar of node.declarations) {
      if (declar.init) {
        hasInits = true;
      }
    }
  }
  this.printList(node.declarations, node, {
    separator: hasInits ? function () {
      this.tokenChar(44);
      this.newline();
    } : undefined,
    indent: node.declarations.length > 1 ? true : false
  });
  if (isFor(parent)) {
    if (isForStatement(parent)) {
      if (parent.init === node) return;
    } else {
      if (parent.left === node) return;
    }
  }
  this.semicolon();
}
function VariableDeclarator(node) {
  this.print(node.id, node);
  if (node.definite) this.tokenChar(33);
  this.print(node.id.typeAnnotation, node);
  if (node.init) {
    this.space();
    this.tokenChar(61);
    this.space();
    this.print(node.init, node);
  }
}

//# sourceMappingURL=statements.js.map
