"""Pylint checker for ensuring main_pre is imported first."""

from typing import TYPE_CHECKING, Optional, Union

import astroid
from astroid import nodes
from pylint.checkers import BaseChecker

if TYPE_CHECKING:
    from pylint.lint import PyLinter


class MainPreImportOrderChecker(BaseChecker):
    """
    Ensures that imports from 'comfy.cmd.main_pre' or similar setup modules
    occur before any other relative imports or imports from the 'comfy' package.

    This is important for code that relies on setup being performed by 'main_pre'
    before other modules from the same package are imported.
    """

    name = 'main-pre-import-order'
    msgs = {
        'W0002': (
            'Setup import %s must be placed before other package imports like %s.',
            'main-pre-import-not-first',
            "To ensure necessary setup is performed, 'comfy.cmd.main_pre' or similar "
            "setup imports must precede other relative imports or imports from the "
            "'comfy' family of packages."
        ),
    }

    def _is_main_pre_import(self, stmt: Union[nodes.Import, nodes.ImportFrom]) -> bool:
        """Checks if an import statement is for 'comfy.cmd.main_pre'."""
        if isinstance(stmt, nodes.Import):
            for name, _ in stmt.names:
                if name == 'comfy.cmd.main_pre' or name.startswith('comfy.cmd.main_pre.'):
                    return True
            return False

        if isinstance(stmt, nodes.ImportFrom):
            qname: Optional[str] = None
            if stmt.level == 0:
                qname = stmt.modname
            else:
                try:
                    # Attempt to resolve the relative import to a fully qualified name
                    imported_module = stmt.do_import_module()
                    qname = imported_module.qname()
                except astroid.AstroidError:
                    # Fallback for unresolved relative imports, check the literal module name
                    if stmt.modname and stmt.modname.endswith('.main_pre'):
                        return True
                    # Heuristic for `from ..cmd import main_pre` in `comfy/entrypoints/*`
                    if stmt.modname == 'cmd' and stmt.root().qname().startswith('comfy'):
                        for name, _ in stmt.names:
                            if name == 'main_pre':
                                return True

            if not qname:
                return False

            # from comfy.cmd import main_pre
            if qname == 'comfy.cmd':
                for name, _ in stmt.names:
                    if name == 'main_pre':
                        return True

            # from comfy.cmd.main_pre import ... OR from a.b.c.main_pre import ...
            if qname == 'comfy.cmd.main_pre' or qname.endswith('.main_pre'):
                return True

        return False

    def _is_other_relevant_import(self, stmt: Union[nodes.Import, nodes.ImportFrom]) -> bool:
        """
        Checks if an import is a relative import or an import from
        the 'comfy' package family, and is not a 'main_pre' import.
        """
        if self._is_main_pre_import(stmt):
            return False

        if isinstance(stmt, nodes.ImportFrom):
            if stmt.level and stmt.level > 0:  # Any relative import
                return True
            if stmt.modname and stmt.modname.startswith('comfy'):
                return True

        if isinstance(stmt, nodes.Import):
            for name, _ in stmt.names:
                if name.startswith('comfy'):
                    return True

        return False

    def visit_module(self, node: nodes.Module) -> None:
        """Checks the order of imports within a module."""
        imports = [
            stmt for stmt in node.body
            if isinstance(stmt, (nodes.Import, nodes.ImportFrom))
        ]

        main_pre_import_node = None
        for stmt in imports:
            if self._is_main_pre_import(stmt):
                main_pre_import_node = stmt
                break

        # If there's no main_pre import, there's nothing to check.
        if not main_pre_import_node:
            return

        for stmt in imports:
            # We only care about imports that appear before the main_pre import
            if stmt.fromlineno >= main_pre_import_node.fromlineno:
                break

            if self._is_other_relevant_import(stmt):
                self.add_message(
                    'main-pre-import-not-first',
                    node=main_pre_import_node,
                    args=(main_pre_import_node.as_string(), stmt.as_string())
                )
                return  # Report once per file and exit to avoid spam


def register(linter: "PyLinter") -> None:
    """This function is required for a Pylint plugin."""
    linter.register_checker(MainPreImportOrderChecker(linter))
