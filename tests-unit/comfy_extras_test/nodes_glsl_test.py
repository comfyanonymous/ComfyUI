import importlib
from unittest.mock import MagicMock, patch

import numpy as np
import torch

mock_nodes = MagicMock()
mock_nodes.MAX_RESOLUTION = 16384

_glsl_module = None


def _get_glsl_module():
    """Import nodes_glsl with the heavyweight ``nodes`` module mocked out."""
    global _glsl_module
    if _glsl_module is None:
        with patch.dict("sys.modules", {"nodes": mock_nodes}):
            _glsl_module = importlib.import_module("comfy_extras.nodes_glsl")
    return _glsl_module


class _FakeCurve:
    def __init__(self, lut):
        self._lut = np.asarray(lut, dtype=np.float32)

    def to_lut(self):
        return self._lut


class TestAutogrowSlotMapping:
    @staticmethod
    def _fake_render(captured, width, height, batch_size):
        def render(fragment_code, out_width, out_height, image_batches, floats, ints, bools, curves, **kwargs):
            captured.update(
                fragment_code=fragment_code,
                image_batches=image_batches,
                floats=floats,
                ints=ints,
                bools=bools,
                curves=curves,
                **kwargs,
            )
            black = np.zeros((out_height, out_width, 4), dtype=np.float32)
            return [[black] * 4 for _ in range(batch_size)]

        return render

    def test_execute_preserves_autogrow_slot_indices(self, monkeypatch):
        nodes_glsl = _get_glsl_module()
        GLSLShader = nodes_glsl.GLSLShader

        image_a = torch.zeros(1, 3, 4, 3)
        image_b = torch.ones(1, 3, 4, 3)
        curve = _FakeCurve([0.0, 0.5, 1.0])
        captured = {}
        render = self._fake_render(captured, width=4, height=3, batch_size=1)

        monkeypatch.setattr(nodes_glsl, "_render_shader_batch", render)
        monkeypatch.setattr(
            GLSLShader,
            "_build_ui_output",
            classmethod(lambda cls, image_list, output_batch: {}),
        )

        result = GLSLShader.execute(
            fragment_shader="frag",
            size_mode={"size_mode": "custom", "width": 4, "height": 3},
            images={"image2": image_b, "image0": image_a},
            floats={"u_float0": 0.25, "u_float2": 0.75},
            ints={"u_int1": 3},
            bools={"u_bool2": True},
            curves={"u_curve1": curve},
        )

        assert len(result.args) == 4
        assert captured["floats"] == [0.25, 0.0, 0.75]
        assert captured["ints"] == [0, 3]
        assert captured["bools"] == [False, False, True]
        assert captured["image_slots"] == [0, 2]
        assert captured["image_batches"][0][0].shape == (3, 4, 3)
        assert np.allclose(captured["image_batches"][0][0], image_a[0].numpy())
        assert np.allclose(captured["image_batches"][0][1], image_b[0].numpy())
        assert [slot for slot, _ in captured["curves"]] == [1]
        assert np.allclose(captured["curves"][0][1], curve.to_lut())


class TestAutogrowSlotHelpers:
    def test_dense_values_fill_gaps_and_none_with_default(self):
        _dense_autogrow_values = getattr(_get_glsl_module(), "_dense_autogrow_values")

        values = {"u_float0": 0.25, "u_float1": None, "u_float3": 0.75}
        assert _dense_autogrow_values(values, "u_float", 0.0) == [0.25, 0.0, 0.0, 0.75]

    def test_dense_values_empty_and_other_keys(self):
        _dense_autogrow_values = getattr(_get_glsl_module(), "_dense_autogrow_values")

        assert _dense_autogrow_values({}, "u_int", 0) == []
        assert _dense_autogrow_values({"image0": 1.0}, "u_int", 0) == []

    def test_indexed_values_sort_and_drop_none(self):
        _indexed_autogrow_values = getattr(_get_glsl_module(), "_indexed_autogrow_values")

        values = {"image2": "b", "image0": "a", "image1": None, "other": "c"}
        assert _indexed_autogrow_values(values, "image") == [(0, "a"), (2, "b")]
