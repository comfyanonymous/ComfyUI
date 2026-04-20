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


class DummyCloneModel:
    def __init__(self):
        self.model_options = {}

    def clone(self):
        cloned = DummyCloneModel()
        cloned.model_options = self.model_options.copy()
        return cloned


class TestTrellisBatchSemantics(unittest.TestCase):
    def test_empty_structure_latent_is_deterministic_and_propagates_sample_indices(self):
        batch_output = nodes_trellis2.EmptyStructureLatentTrellis2.execute(2, 0, 17)[0]
        single_output = nodes_trellis2.EmptyStructureLatentTrellis2.execute(1, 5, 17)[0]

        expected_batch = torch.zeros(2, 8, 16, 16, 16)
        expected_batch[0] = torch.randn(1, 8, 16, 16, 16, generator=torch.Generator(device="cpu").manual_seed(17))[0]
        expected_batch[1] = torch.randn(1, 8, 16, 16, 16, generator=torch.Generator(device="cpu").manual_seed(18))[0]
        expected_single = torch.randn(1, 8, 16, 16, 16, generator=torch.Generator(device="cpu").manual_seed(22))

        self.assertTrue(torch.equal(batch_output["samples"], expected_batch))
        self.assertEqual(batch_output["batch_index"], [0, 1])
        self.assertTrue(torch.equal(single_output["samples"], expected_single))
        self.assertEqual(single_output["batch_index"], [5])

    def test_empty_shape_latent_is_deterministic_and_propagates_batch_index(self):
        coords = torch.tensor(
            [
                [1, 5, 5, 5],
                [0, 1, 1, 1],
                [1, 6, 6, 6],
                [0, 2, 2, 2],
                [1, 7, 7, 7],
            ],
            dtype=torch.int32,
        )
        structure = {
            "coords": coords,
            "coord_counts": torch.tensor([2, 3], dtype=torch.int64),
            "batch_index": [4, 9],
        }

        output, _ = nodes_trellis2.EmptyShapeLatentTrellis2.execute(structure, DummyCloneModel(), 23)

        expected = torch.zeros(2, 32, 3, 1)
        expected[0, :, :2, :] = torch.randn(1, 32, 2, 1, generator=torch.Generator(device="cpu").manual_seed(27))[0]
        expected[1, :, :3, :] = torch.randn(1, 32, 3, 1, generator=torch.Generator(device="cpu").manual_seed(32))[0]

        self.assertTrue(torch.equal(output["samples"], expected))
        self.assertTrue(torch.equal(output["coord_counts"], torch.tensor([2, 3], dtype=torch.int64)))
        self.assertEqual(output["batch_index"], [4, 9])

    def test_empty_shape_latent_keeps_singleton_coord_counts(self):
        structure = {
            "coords": torch.tensor(
                [
                    [0, 1, 1, 1],
                    [0, 2, 2, 2],
                ],
                dtype=torch.int32,
            ),
        }

        output, _ = nodes_trellis2.EmptyShapeLatentTrellis2.execute(structure, DummyCloneModel(), 11)

        self.assertTrue(torch.equal(output["coord_counts"], torch.tensor([2], dtype=torch.int64)))

    def test_empty_shape_latent_rejects_multi_index_singleton(self):
        structure = {
            "coords": torch.tensor(
                [
                    [0, 1, 1, 1],
                    [0, 2, 2, 2],
                ],
                dtype=torch.int32,
            ),
            "batch_index": [5, 6],
        }

        with self.assertRaises(ValueError):
            nodes_trellis2.EmptyShapeLatentTrellis2.execute(structure, DummyCloneModel(), 11)

    def test_empty_texture_latent_rejects_multi_index_singleton(self):
        coords = torch.tensor(
            [
                [0, 1, 1, 1],
                [0, 2, 2, 2],
            ],
            dtype=torch.int32,
        )
        structure = {"coords": coords, "batch_index": [7, 8]}
        shape_latent = {"samples": torch.zeros(1, 32, 2, 1)}

        with self.assertRaises(ValueError):
            nodes_trellis2.EmptyTextureLatentTrellis2.execute(
                structure,
                shape_latent,
                DummyCloneModel(),
                13,
            )

    def test_empty_texture_latent_rejects_invalid_structure_input(self):
        with self.assertRaises(ValueError):
            nodes_trellis2.EmptyTextureLatentTrellis2.execute(
                "bad-input",
                {"samples": torch.zeros(1, 32, 2, 1)},
                DummyCloneModel(),
                13,
            )

    def test_flatten_batched_sparse_latent_validates_coord_counts(self):
        samples = torch.zeros(2, 32, 3, 1)
        coords = torch.tensor(
            [
                [0, 1, 1, 1],
                [1, 2, 2, 2],
                [1, 3, 3, 3],
            ],
            dtype=torch.int32,
        )
        coord_counts = torch.tensor([2, 1], dtype=torch.int64)

        with self.assertRaises(ValueError):
            nodes_trellis2.flatten_batched_sparse_latent(samples, coords, coord_counts)

    def test_infer_batched_coord_layout_rejects_negative_batch_ids(self):
        coords = torch.tensor(
            [
                [-1, 1, 1, 1],
                [0, 2, 2, 2],
            ],
            dtype=torch.int32,
        )

        with self.assertRaises(ValueError):
            nodes_trellis2.infer_batched_coord_layout(coords)

    def test_split_batched_coords_validates_total_count(self):
        coords = torch.tensor(
            [
                [0, 1, 1, 1],
                [1, 2, 2, 2],
                [1, 3, 3, 3],
            ],
            dtype=torch.int32,
        )
        coord_counts = torch.tensor([1, 1], dtype=torch.int64)

        with self.assertRaises(ValueError):
            nodes_trellis2.split_batched_coords(coords, coord_counts)

    def test_empty_shape_latent_preserves_resolutions_key(self):
        structure = {
            "coords": torch.tensor(
                [
                    [0, 1, 1, 1],
                    [0, 2, 2, 2],
                ],
                dtype=torch.int32,
            ),
            "resolutions": torch.tensor([1024], dtype=torch.int64),
        }

        output, model = nodes_trellis2.EmptyShapeLatentTrellis2.execute(structure, DummyCloneModel(), 11)

        self.assertTrue(torch.equal(output["resolutions"], torch.tensor([1024], dtype=torch.int64)))
        self.assertNotIn("coord_resolutions", model.model_options["transformer_options"])


if __name__ == "__main__":
    unittest.main()
