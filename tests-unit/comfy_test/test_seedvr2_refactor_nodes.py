import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import comfy_extras.nodes_seedvr as nodes_seedvr
import nodes


def test_seedvr2_postprocessing_restores_flat_decoded_batch_time():
    decoded = torch.arange(6 * 4 * 6 * 1, dtype=torch.float32).reshape(6, 4, 6, 1)
    original = torch.ones((2, 3, 4, 6, 1), dtype=torch.float32)

    output = nodes_seedvr.SeedVR2PostProcessing.execute(decoded, original, 4, "none").result[0]

    assert output.shape == (6, 4, 6, 1)
    torch.testing.assert_close(output, decoded)


def test_seedvr2_postprocessing_crops_to_resized_original_size():
    decoded = torch.ones((1, 128, 176, 3), dtype=torch.float32)
    original = torch.full((1, 1, 120, 169, 3), 0.25, dtype=torch.float32)

    output = nodes_seedvr.SeedVR2PostProcessing.execute(decoded, original, 120, "none").result[0]

    assert output.shape == (1, 120, 168, 3)


def test_seedvr2_postprocessing_uses_decoded_size_when_resized_original_is_larger():
    decoded = torch.ones((1, 128, 160, 3), dtype=torch.float32)
    original = torch.full((1, 1, 480, 640, 3), 0.25, dtype=torch.float32)

    output = nodes_seedvr.SeedVR2PostProcessing.execute(decoded, original, 480, "none").result[0]

    assert output.shape == (1, 128, 160, 3)


def test_seedvr2_postprocessing_does_not_trim_real_black_original_edges():
    decoded = torch.ones((1, 128, 176, 3), dtype=torch.float32)
    original = torch.zeros((1, 1, 128, 176, 3), dtype=torch.float32)

    output = nodes_seedvr.SeedVR2PostProcessing.execute(decoded, original, 128, "none").result[0]

    assert output.shape == (1, 128, 176, 3)


def test_seedvr2_postprocessing_crops_height_only_to_resized_original_size():
    decoded = torch.ones((1, 128, 176, 3), dtype=torch.float32)
    original = torch.full((1, 1, 120, 176, 3), 0.25, dtype=torch.float32)

    output = nodes_seedvr.SeedVR2PostProcessing.execute(decoded, original, 120, "none").result[0]

    assert output.shape == (1, 120, 176, 3)


def test_seedvr2_postprocessing_lab_uses_resized_original_size(monkeypatch):
    decoded = torch.ones((1, 128, 176, 3), dtype=torch.float32)
    original = torch.full((1, 1, 120, 169, 3), 0.25, dtype=torch.float32)
    calls = []

    def fake_lab_color_transfer(decoded_flat, reference_flat):
        calls.append((tuple(decoded_flat.shape), tuple(reference_flat.shape)))
        return decoded_flat

    monkeypatch.setattr(nodes_seedvr, "lab_color_transfer", fake_lab_color_transfer)

    output = nodes_seedvr.SeedVR2PostProcessing.execute(decoded, original, 120, "lab").result[0]

    assert calls == [((1, 3, 120, 169), (1, 3, 120, 169))]
    assert output.shape == (1, 120, 168, 3)


def test_seedvr2_tiled_decode_node_ignores_seedvr2_sideband_metadata():
    class FakeVAE:
        def __init__(self):
            self.decode_call = None

        def temporal_compression_decode(self):
            return 4

        def spacial_compression_decode(self):
            return 8

        def decode_tiled(self, samples, **kwargs):
            self.decode_call = kwargs
            return torch.zeros((1, 1, 2, 2, 3), dtype=torch.float32)

    vae = FakeVAE()
    samples = {
        "samples": torch.zeros((1, 16, 4, 4, 16), dtype=torch.float32),
        "seedvr2_channel_last": True,
    }

    nodes.VAEDecodeTiled().decode(
        vae,
        samples,
        tile_size=64,
        overlap=0,
        temporal_size=64,
        temporal_overlap=8,
    )

    assert "seedvr2_channel_last" not in vae.decode_call


def test_seedvr2_decode_node_ignores_seedvr2_sideband_metadata():
    class FakeVAE:
        def __init__(self):
            self.decode_call = None

        def decode(self, samples, **kwargs):
            self.decode_call = kwargs
            return torch.zeros((1, 1, 2, 2, 3), dtype=torch.float32)

    vae = FakeVAE()
    samples = {
        "samples": torch.zeros((1, 16, 4, 4, 16), dtype=torch.float32),
        "seedvr2_channel_last": True,
    }

    nodes.VAEDecode().decode(vae, samples)

    assert "seedvr2_channel_last" not in vae.decode_call


def test_seedvr2_decode_node_leaves_unmarked_ambiguous_latent_unforced():
    class FakeVAE:
        def __init__(self):
            self.decode_call = None

        def decode(self, samples, **kwargs):
            self.decode_call = kwargs
            return torch.zeros((1, 1, 2, 2, 3), dtype=torch.float32)

    vae = FakeVAE()
    samples = {"samples": torch.zeros((1, 16, 4, 4, 16), dtype=torch.float32)}

    nodes.VAEDecode().decode(vae, samples)

    assert "seedvr2_channel_last" not in vae.decode_call


def test_seedvr2_encode_node_does_not_mark_model_specific_layout_metadata():
    class FakeVAE:
        def encode(self, pixels):
            return torch.zeros((1, 16, 2, 3, 4), dtype=torch.float32)

    output = nodes.VAEEncode().encode(FakeVAE(), torch.zeros((1, 8, 8, 3)))[0]

    assert set(output) == {"samples"}


def test_seedvr2_tiled_encode_node_does_not_mark_model_specific_layout_metadata():
    class FakeVAE:
        def encode_tiled(self, pixels, **kwargs):
            return torch.zeros((1, 16, 2, 3, 4), dtype=torch.float32)

    output = nodes.VAEEncodeTiled().encode(FakeVAE(), torch.zeros((1, 8, 8, 3)), 64, 0)[0]

    assert set(output) == {"samples"}


def test_seedvr2_saved_latent_does_not_persist_model_specific_layout_metadata(monkeypatch):
    saved = {}

    def fake_save_image_path(filename_prefix, output_dir):
        return output_dir, filename_prefix, 1, "", filename_prefix

    def fake_save_torch_file(output, file, metadata=None):
        saved.update(output)

    monkeypatch.setattr(nodes.folder_paths, "get_save_image_path", fake_save_image_path)
    monkeypatch.setattr(nodes.comfy.utils, "save_torch_file", fake_save_torch_file)
    monkeypatch.setattr(nodes.folder_paths, "get_annotated_filepath", lambda latent: latent)
    monkeypatch.setattr(nodes.safetensors.torch, "load_file", lambda latent_path, device="cpu": saved)

    original = torch.zeros((1, 16, 4, 4, 16), dtype=torch.float32)
    nodes.SaveLatent().save({"samples": original, "seedvr2_channel_last": True}, "seedvr2_latent")
    loaded = nodes.LoadLatent().load("seedvr2_latent")[0]

    assert "seedvr2_channel_last" not in saved
    assert "seedvr2_channel_last" not in loaded
    torch.testing.assert_close(loaded["samples"], original)


def test_seedvr2_tiled_decode_node_preserves_legacy_decode_tiled_signature():
    class FakeVAE:
        def __init__(self):
            self.decode_call = None

        def temporal_compression_decode(self):
            return 4

        def spacial_compression_decode(self):
            return 8

        def decode_tiled(self, samples, tile_x, tile_y, overlap, tile_t, overlap_t):
            self.decode_call = {
                "tile_x": tile_x,
                "tile_y": tile_y,
                "overlap": overlap,
                "tile_t": tile_t,
                "overlap_t": overlap_t,
            }
            return torch.zeros((1, 1, 2, 2, 3), dtype=torch.float32)

    vae = FakeVAE()
    samples = {"samples": torch.zeros((1, 16, 4, 4, 16), dtype=torch.float32)}

    nodes.VAEDecodeTiled().decode(
        vae,
        samples,
        tile_size=64,
        overlap=0,
        temporal_size=64,
        temporal_overlap=8,
    )

    assert vae.decode_call == {
        "tile_x": 8,
        "tile_y": 8,
        "overlap": 0,
        "tile_t": 16,
        "overlap_t": 2,
    }
