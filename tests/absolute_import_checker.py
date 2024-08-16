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
        current_file = node.root().file
        if current_file is None:
            return

        package_path = os.path.dirname(current_file)
        package_name = os.path.basename(package_path)

        if node.modname.startswith(package_name) and package_name in ['comfy', 'comfy_extras']:
            import_parts = node.modname.split('.')

            if import_parts[0] == package_name:
                self.add_message('absolute-import-used', node=node, args=(node.modname,))


def register(linter: "PyLinter") -> None:
    linter.register_checker(AbsoluteImportChecker(linter))
