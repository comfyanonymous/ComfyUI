import os
from typing import TYPE_CHECKING, Optional
from pylint.checkers import BaseChecker

if TYPE_CHECKING:
    from pylint.lint import PyLinter

class MissingInitChecker(BaseChecker):
    name = 'missing-init'
    priority = -1
    msgs = {
        'W8001': (
            'Directory %s has .py files but missing __init__.py',
            'missing-init',
            'All directories containing .py files should have an __init__.py file.',
        ),
    }

    def __init__(self, linter: Optional["PyLinter"] = None) -> None:
        super().__init__(linter)

    def visit_module(self, node):
        if not node.file:
            return
        
        # Only check .py files
        if not node.file.endswith('.py'):
            return
        
        # Skip __init__.py itself
        if os.path.basename(node.file) == '__init__.py':
            return

        directory = os.path.dirname(os.path.abspath(node.file))
        init_file = os.path.join(directory, '__init__.py')

        if not os.path.exists(init_file):
            self.add_message('missing-init', args=directory, node=node)

def register(linter: "PyLinter") -> None:
    linter.register_checker(MissingInitChecker(linter))
