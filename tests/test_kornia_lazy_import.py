import ast
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).parents[1]
KORNIA_NODE_MODULES = (
    "comfy_extras/nodes_canny.py",
    "comfy_extras/nodes_morphology.py",
    "comfy_extras/nodes_post_processing.py",
)


@pytest.mark.parametrize("relative_path", KORNIA_NODE_MODULES)
def test_kornia_nodes_do_not_import_kornia_at_module_load(relative_path):
    source_path = REPOSITORY_ROOT / relative_path
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))

    top_level_imports = [
        node
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]

    for node in top_level_imports:
        modules = [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module]
        assert not any(module == "kornia" or module.startswith("kornia.") for module in modules), (
            f"{relative_path} imports Kornia while node modules are loaded"
        )
