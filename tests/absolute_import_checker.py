import os
from typing import TYPE_CHECKING, Optional

from astroid import nodes
from pylint.checkers import BaseChecker

if TYPE_CHECKING:
    from pylint.lint import PyLinter


class AbsoluteImportChecker(BaseChecker):
    """Checker for detecting absolute imports within the same package."""

    name = 'absolute-import'
    msgs = {
        'W0001': (
            'Absolute import from same package used: %s',
            'absolute-import-used',
            'Use relative imports instead of absolute imports from the same package.'
        ),
    }

    def __init__(self, linter: Optional["PyLinter"] = None) -> None:
        super().__init__(linter)

    def visit_importfrom(self, node: nodes.ImportFrom) -> None:
        """
        Check for absolute imports from the same top-level package.

        This method is called for every `from ... import ...` statement.
        It checks if a module within 'comfy' or 'comfy_extras' packages
        is using an absolute import from its own package, which should
        be a relative import instead.

        For example, inside `comfy/nodes/logic.py`, an import like
        `from comfy.utils import some_function` will be flagged.
        The preferred way would be `from ..utils import some_function`.
        """
        # An import is relative if its level is greater than 0.
        # e.g., from . import foo (level=1), from .. import bar (level=2)
        # We only want to check absolute imports, so we skip relative ones.
        if node.level and node.level > 0:
            return

        # Get the fully qualified name of the module being linted.
        # For a file at '.../comfy/nodes/common.py', this will be 'comfy.nodes.common'.
        module_qname = node.root().qname()

        # `node.modname` is the module name in the `from` statement.
        # For `from comfy.utils import x`, `modname` is `comfy.utils`.
        imported_modname = node.modname
        if not imported_modname:
            return

        # We are only interested in modules within 'comfy' or 'comfy_extras'.
        # We determine this by looking at the first part of the qualified name.
        current_top_package = module_qname.split('.')[0]
        if current_top_package not in ['comfy', 'comfy_extras']:
            return

        imported_top_package = imported_modname.split('.')[0]

        # If the top-level package of the imported module is the same as the
        # current module's top-level package, it's an internal absolute import.
        if imported_top_package == current_top_package:
            self.add_message('absolute-import-used', node=node, args=(imported_modname,))


def register(linter: "PyLinter") -> None:
    linter.register_checker(AbsoluteImportChecker(linter))