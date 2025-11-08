import re
import traceback
import types

from comfy.execution_context import current_execution_context
from comfy.node_helpers import export_package_as_web_directory, export_custom_nodes
from comfy.nodes.package_typing import CustomNode

remove_type_name = re.compile(r"(\{.*\})", re.I | re.M)


# Hack: string type that is always equal in not equal comparisons, thanks pythongosssss
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False


PY_CODE = AnyType("*")
IDEs_DICT = {}


# - Thank you very much for the class -> Trung0246 -
# - https://github.com/Trung0246/ComfyUI-0246/blob/main/utils.py#L51
class TautologyStr(str):
    def __ne__(self, other):
        return False


class ByPassTypeTuple(tuple):
    def __getitem__(self, index):
        if index > 0:
            index = 0
        item = super().__getitem__(index)
        if isinstance(item, str):
            return TautologyStr(item)
        return item


# ---------------------------


class KY_Eval_Python(CustomNode):
    @classmethod
    def INPUT_TYPES(s):

        return {
            "required": {
                "pycode": (
                    "PYCODE",
                    {
                        "default": """import re, json, os, traceback
from time import strftime

def runCode():
    nowDataTime = strftime("%Y-%m-%d %H:%M:%S")
    return f"Hello ComfyUI with us today {nowDataTime}!"
r0_str = runCode() + unique_id
"""
                    },
                ),
            },
            "hidden": {"unique_id": "UNIQUE_ID", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ByPassTypeTuple((PY_CODE,))
    RETURN_NAMES = ("r0_str",)
    FUNCTION = "exec_py"
    DESCRIPTION = "IDE Node is an node that allows you to run code written in Python or Javascript directly in the node."
    CATEGORY = "KYNode/Code"

    def exec_py(self, pycode, unique_id, extra_pnginfo, **kwargs):
        ctx = current_execution_context()
        if ctx.configuration.enable_eval is not True:
            raise ValueError("Python eval is disabled")

        if unique_id not in IDEs_DICT:
            IDEs_DICT[unique_id] = self

        outputs = {unique_id: unique_id}
        if extra_pnginfo and 'workflow' in extra_pnginfo and extra_pnginfo['workflow']:
            for node in extra_pnginfo['workflow']['nodes']:
                if node['id'] == int(unique_id):
                    outputs_valid = [ouput for ouput in node.get('outputs', []) if ouput.get('name', '') != '' and ouput.get('type', '') != '']
                    outputs = {ouput['name']: None for ouput in outputs_valid}
                    self.RETURN_TYPES = ByPassTypeTuple(out["type"] for out in outputs_valid)
                    self.RETURN_NAMES = tuple(name for name in outputs.keys())
        my_namespace = types.SimpleNamespace()
        # 从 prompt 对象中提取 prompt_id
        # if extra_data and 'extra_data' in extra_data and 'prompt_id' in extra_data['extra_data']:
        #     prompt_id = prompt['extra_data']['prompt_id']
        # outputs['p0_str'] = p0_str

        my_namespace.__dict__.update(outputs)
        my_namespace.__dict__.update({prop: kwargs[prop] for prop in kwargs})
        # my_namespace.__dict__.setdefault("r0_str", "The r0 variable is not assigned")

        try:
            exec(pycode, my_namespace.__dict__)
        except Exception as e:
            err = traceback.format_exc()
            mc = re.search(r'line (\d+), in <module>([\w\W]+)$', err, re.MULTILINE)
            msg = mc[1] + ':' + mc[2]
            my_namespace.r0 = f"Error Line{msg}"

        new_dict = {key: my_namespace.__dict__[key] for key in my_namespace.__dict__ if key not in ['__builtins__', *kwargs.keys()] and not callable(my_namespace.__dict__[key])}
        return (*new_dict.values(),)


export_custom_nodes()
export_package_as_web_directory("comfy_extras.eval_web")
