import torch
import safetensors.torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import comfy.nested_tensor
import comfy.utils
import folder_paths
import nodes


def configure_latent_output_directory(tmp_path, monkeypatch):
    output_path = tmp_path / "latent_00001_.latent"
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path))
    monkeypatch.setattr(
        folder_paths,
        "get_save_image_path",
        lambda filename_prefix, output_dir: (tmp_path, "latent", 1, "", filename_prefix),
    )
    return output_path


def test_save_and_load_nested_latent_roundtrip(tmp_path, monkeypatch):
    output_path = configure_latent_output_directory(tmp_path, monkeypatch)

    video = torch.arange(4, dtype=torch.float32).reshape(1, 2, 1, 2, 1)
    audio = torch.arange(4, dtype=torch.float32).reshape(1, 2, 1, 2)
    samples = {"samples": comfy.nested_tensor.NestedTensor((video, audio))}

    nodes.SaveLatent().save(samples, prompt=None, extra_pnginfo=None)

    saved = safetensors.torch.load_file(str(output_path))
    expected, _ = comfy.utils.pack_latents((video, audio))
    assert torch.equal(saved["latent_tensor"], expected)
    assert torch.equal(saved["latent_shape_0"], torch.tensor(video.shape, dtype=torch.int64))
    assert torch.equal(saved["latent_shape_1"], torch.tensor(audio.shape, dtype=torch.int64))

    monkeypatch.setattr(folder_paths, "get_annotated_filepath", lambda filename: str(output_path))
    loaded = nodes.LoadLatent().load(str(output_path))[0]["samples"]
    assert torch.equal(loaded.tensors[0], video)
    assert torch.equal(loaded.tensors[1], audio)


def test_save_and_load_one_stream_nested_latent_roundtrip(tmp_path, monkeypatch):
    output_path = configure_latent_output_directory(tmp_path, monkeypatch)
    video = torch.arange(4, dtype=torch.float32).reshape(1, 2, 1, 2)
    samples = {"samples": comfy.nested_tensor.NestedTensor((video,))}

    nodes.SaveLatent().save(samples, prompt=None, extra_pnginfo=None)

    monkeypatch.setattr(folder_paths, "get_annotated_filepath", lambda filename: str(output_path))
    loaded = nodes.LoadLatent().load(str(output_path))[0]["samples"]
    assert len(loaded.tensors) == 1
    assert loaded.tensors[0].shape == video.shape
    assert torch.equal(loaded.tensors[0], video)
