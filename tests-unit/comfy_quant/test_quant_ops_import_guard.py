import builtins
import importlib.util
import os
import unittest
from unittest import mock

QUANT_OPS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "comfy", "quant_ops.py"
)


class TestQuantOpsImportGuard(unittest.TestCase):
    def test_survives_comfy_kitchen_schema_value_error(self):
        """comfy_kitchen can raise ValueError (not ImportError) at import time when its
        custom ops use PEP 585 generics (e.g. list[int]) that torch.library.infer_schema
        only accepts from torch>=2.7. quant_ops must degrade instead of crashing startup."""
        real_import = builtins.__import__
        triggered = []

        def fake_import(name, *args, **kwargs):
            if name == "comfy_kitchen" and not triggered:
                triggered.append(True)
                raise ValueError(
                    "infer_schema(func): Parameter kernel_size has unsupported type list[int]"
                )
            return real_import(name, *args, **kwargs)

        spec = importlib.util.spec_from_file_location(
            "quant_ops_value_error_test", QUANT_OPS_PATH
        )
        module = importlib.util.module_from_spec(spec)
        with mock.patch("builtins.__import__", side_effect=fake_import):
            spec.loader.exec_module(module)

        self.assertFalse(module._CK_AVAILABLE)


if __name__ == "__main__":
    unittest.main()
