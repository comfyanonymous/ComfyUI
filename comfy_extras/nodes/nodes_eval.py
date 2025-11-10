import logging

from comfy.comfy_types import IO
from comfy.execution_context import current_execution_context
from comfy.node_helpers import export_package_as_web_directory, export_custom_nodes
from comfy.nodes.package_typing import CustomNode

logger = logging.getLogger(__name__)


def eval_python(inputs=5, outputs=5, name=None, input_is_list=None, output_is_list=None):
    """
    Factory function to create EvalPython node classes with configurable input/output counts.

    Args:
        inputs: Number of input value slots (default: 5)
        outputs: Number of output item slots (default: 5)
        name: Class name (default: f"EvalPython_{inputs}_{outputs}")
        input_is_list: Optional list of bools indicating which inputs accept lists (default: None, meaning all scalar)
        output_is_list: Optional tuple of bools indicating which outputs return lists (default: None, meaning all scalar)

    Returns:
        A CustomNode subclass configured with the specified inputs/outputs
    """
    if name is None:
        name = f"EvalPython_{inputs}_{outputs}"

    default_code = f"""
print("Hello World!")
return {", ".join([f"value{i}" for i in range(inputs)])}
"""

    class EvalPythonNode(CustomNode):
        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {
                    "pycode": (
                        "CODE_BLOCK_PYTHON",
                        {
                            "default": default_code
                        },
                    ),
                },
                "optional": {f"value{i}": (IO.ANY, {}) for i in range(inputs)},
            }

        RETURN_TYPES = tuple(IO.ANY for _ in range(outputs))
        RETURN_NAMES = tuple(f"item{i}" for i in range(outputs))
        FUNCTION = "exec_py"
        DESCRIPTION = ""
        CATEGORY = "eval"

        @classmethod
        def VALIDATE_INPUTS(cls, *args, **kwargs):
            ctx = current_execution_context()

            return ctx.configuration.enable_eval

        def exec_py(self, pycode, **kwargs):
            ctx = current_execution_context()

            kwargs = {
                **{f"value{i}": None for i in range(inputs)},
                **kwargs,
            }

            def print(*args):
                ctx.server.send_progress_text(" ".join(map(str, args)), ctx.node_id)

            if not ctx.configuration.enable_eval:
                raise ValueError("Python eval is disabled")

            value_args = [kwargs.pop(f"value{i}") for i in range(inputs)]
            arg_names = ", ".join(f"value{i}=None" for i in range(inputs))

            wrapped_code = f"def _eval_func({arg_names}):\n"
            for line in pycode.splitlines():
                wrapped_code += "    " + line + "\n"

            globals_for_eval = {
                **kwargs,
                "logger": logger,
                "print": print,
            }

            exec(wrapped_code, globals_for_eval)
            results = globals_for_eval["_eval_func"](*value_args)
            if not isinstance(results, tuple):
                results = (results,)

            if len(results) < outputs:
                results += (None,) * (outputs - len(results))
            elif len(results) > outputs:
                results = results[:outputs]

            return results

    # todo: interact better with the weird comfyui machinery for this
    if input_is_list is not None:
        setattr(EvalPythonNode, "INPUT_IS_LIST", input_is_list)

    if output_is_list is not None:
        setattr(EvalPythonNode, "OUTPUT_IS_LIST", output_is_list)

    EvalPythonNode.__name__ = name
    EvalPythonNode.__qualname__ = name

    return EvalPythonNode


EvalPython_1_1 = eval_python(inputs=1, outputs=1, name="EvalPython_1_1")
EvalPython_5_5 = eval_python(inputs=5, outputs=5, name="EvalPython_5_5")
EvalPython_List_1 = eval_python(inputs=1, outputs=1, name="EvalPython_List_1", input_is_list=True, output_is_list=None)
EvalPython_1_List = eval_python(inputs=1, outputs=1, name="EvalPython_1_List", input_is_list=None, output_is_list=(True,))
EvalPython_List_List = eval_python(inputs=1, outputs=1, name="EvalPython_List_List", input_is_list=True, output_is_list=(True,))

export_custom_nodes()
export_package_as_web_directory("comfy_extras.eval_web")
