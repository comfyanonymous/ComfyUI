from typing import TYPE_CHECKING, Optional
from pylint.checkers import BaseChecker

if TYPE_CHECKING:
    from pylint.lint import PyLinter
    from astroid import nodes

class SDClipModelInitChecker(BaseChecker):

    name = 'sd-clip-model-init-checker'
    priority = -1
    msgs = {
        'W9001': (
            'Class %s inheriting from SDClipModel must have textmodel_json_config in __init__ arguments',
            'sd-clip-model-missing-config',
            'Classes inheriting from comfy.sd1_clip.SDClipModel must accept textmodel_json_config in their __init__ method.',
        ),
    }

    def __init__(self, linter: Optional["PyLinter"] = None) -> None:
        super().__init__(linter)

    def visit_classdef(self, node: "nodes.ClassDef") -> None:
        # Check if class inherits from SDClipModel
        is_sd_clip_model = False
        for base in node.bases:
            # Check for direct name 'SDClipModel' or fully qualified 'comfy.sd1_clip.SDClipModel'
            # Simple name check is usually sufficient for this targeted rule
            if getattr(base, 'name', '') == 'SDClipModel':
                is_sd_clip_model = True
                break
            if getattr(base, 'attrname', '') == 'SDClipModel': # for attribute access like module.SDClipModel
                 is_sd_clip_model = True
                 break

        if not is_sd_clip_model:
            return

        # Check __init__ arguments
        if '__init__' not in node.locals:
            return # Uses parent init, assuming parent is compliant or we can't check easily

        init_methods = node.locals['__init__']
        if not init_methods:
            return
            
        # method could be a list of inferred values, usually we just want the definition
        # node.locals returns a list of nodes for that name
        init_method = init_methods[0]
        
        if not hasattr(init_method, 'args'):
            return # Might not be a function definition

        args = init_method.args
        arg_names = [arg.name for arg in args.args]
        
        # Check keyword-only arguments too if present
        if args.kwonlyargs:
            arg_names.extend([arg.name for arg in args.kwonlyargs])

        # We need check usage of *args or **kwargs? 
        # The prompt specifically says "have `textmodel_json_config` in the args".
        # Usually this means explicit argument.
        
        if 'textmodel_json_config' not in arg_names:
            self.add_message('sd-clip-model-missing-config', args=node.name, node=node)

def register(linter: "PyLinter") -> None:
    linter.register_checker(SDClipModelInitChecker(linter))
