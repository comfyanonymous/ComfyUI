import importlib
import sys
import types
import unittest
from unittest.mock import patch

import torch
from PIL import Image


class _DummyPort:
    @staticmethod
    def Input(*args, **kwargs):
        return None

    @staticmethod
    def Output(*args, **kwargs):
        return None


class _DummyIO:
    ComfyNode = object

    @staticmethod
    def Schema(*args, **kwargs):
        return None

    @staticmethod
    def NodeOutput(*args, **kwargs):
        return args

    def __getattr__(self, name):
        return _DummyPort


class _DummyTypes:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


dummy_comfy_api_latest = types.SimpleNamespace(
    ComfyExtension=object,
    IO=_DummyIO(),
    Types=_DummyTypes(),
)

dummy_sparse_tensor = type("SparseTensor", (), {})
dummy_trellis_vae = types.SimpleNamespace(SparseTensor=dummy_sparse_tensor)

with patch.dict(sys.modules, {
    "comfy_api.latest": dummy_comfy_api_latest,
    "comfy.ldm.trellis2.vae": dummy_trellis_vae,
}):
    nodes_trellis2 = importlib.import_module("comfy_extras.nodes_trellis2")


class DummyInnerModel:
    def __init__(self, image_size=..., fail_on_call=None):
        self.call_count = 0
        self.fail_on_call = fail_on_call
        if image_size is not ...:
            self.image_size = image_size

    def __call__(self, input_tensor, skip_norm_elementwise=True):
        self.call_count += 1
        if self.fail_on_call == self.call_count:
            raise RuntimeError("expected conditioning failure")
        return (torch.ones((1, 4), dtype=torch.float32),)


class DummyModel:
    def __init__(self, inner_model):
        self.model = inner_model


class TestRunConditioningRestore(unittest.TestCase):
    def setUp(self):
        self.intermediate_patch = patch.object(
            nodes_trellis2.comfy.model_management, "intermediate_device", lambda: "cpu"
        )
        self.torch_device_patch = patch.object(
            nodes_trellis2.comfy.model_management, "get_torch_device", lambda: "cpu"
        )
        self.intermediate_patch.start()
        self.torch_device_patch.start()

    def tearDown(self):
        self.intermediate_patch.stop()
        self.torch_device_patch.stop()

    @staticmethod
    def make_test_image():
        return Image.new("RGB", (8, 8), color="white")

    def test_restores_existing_image_size_after_success(self):
        inner_model = DummyInnerModel(image_size=777)

        nodes_trellis2.run_conditioning(DummyModel(inner_model), self.make_test_image(), include_1024=True)

        self.assertEqual(inner_model.image_size, 777)

    def test_deletes_missing_image_size_after_success(self):
        inner_model = DummyInnerModel()

        nodes_trellis2.run_conditioning(DummyModel(inner_model), self.make_test_image(), include_1024=True)

        self.assertFalse(hasattr(inner_model, "image_size"))

    def test_restores_existing_image_size_after_512_failure(self):
        inner_model = DummyInnerModel(image_size=777, fail_on_call=1)

        with self.assertRaisesRegex(RuntimeError, "expected conditioning failure"):
            nodes_trellis2.run_conditioning(DummyModel(inner_model), self.make_test_image(), include_1024=True)

        self.assertEqual(inner_model.image_size, 777)

    def test_deletes_missing_image_size_after_1024_failure(self):
        inner_model = DummyInnerModel(fail_on_call=2)

        with self.assertRaisesRegex(RuntimeError, "expected conditioning failure"):
            nodes_trellis2.run_conditioning(DummyModel(inner_model), self.make_test_image(), include_1024=True)

        self.assertFalse(hasattr(inner_model, "image_size"))


if __name__ == "__main__":
    unittest.main()
