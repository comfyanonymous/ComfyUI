import numpy as np
import pytest
import torch

from comfy.nodes.base_nodes import ImagePadForOutpaint, ImageBatch, ImageInvert, ImageScaleBy, ImageScale, LatentCrop, \
    LatentComposite, LatentFlip, LatentRotate, LatentUpscaleBy, LatentUpscale, InpaintModelConditioning, CLIPTextEncode, \
    VAEEncodeForInpaint, VAEEncode, VAEDecode, ConditioningSetMask, ConditioningSetArea, ConditioningCombine, \
    EmptyImage

torch.set_grad_enabled(False)

_image_1x1_px = np.array([[[255, 0, 0]]], dtype=np.uint8)
_image_1x1 = torch.ones((1, 1, 1, 3), dtype=torch.float32, device="cpu")
_image_512x512 = torch.randn((1, 512, 512, 3), dtype=torch.float32, device="cpu")

_cond = torch.randn((1, 4, 77, 768))
_cond_with_pooled = (_cond, {"pooled_output": torch.zeros((1, 1, 768))})

_latent = {"samples": torch.randn((1, 4, 64, 64))}


def test_clip_text_encode(clip):
    cond, = CLIPTextEncode().encode(clip, "test prompt")
    assert len(cond) == 1
    assert cond[0][0].shape == (1, 77, 768)
    assert "pooled_output" in cond[0][1]
    assert cond[0][1]["pooled_output"].shape == (1, 768)


def test_conditioning_combine():
    cond = ConditioningCombine().combine(_cond_with_pooled, _cond_with_pooled)
    assert len(cond) == 1
    assert cond[0][0].shape == (1, 4, 77, 768)


def test_conditioning_set_area(clip):
    cond, = CLIPTextEncode().encode(clip, "test prompt")
    cond, = ConditioningSetArea().append(cond, 64, 64, 0, 0, 1.0)
    assert len(cond) == 1
    assert cond[0][1]["area"] == (8, 8, 0, 0)
    assert cond[0][1]["strength"] == 1.0


def test_conditioning_set_mask(clip):
    cond, = CLIPTextEncode().encode(clip, "test prompt")
    mask = torch.ones((1, 64, 64))
    cond, = ConditioningSetMask().append(cond, mask, "default", 1.0)
    assert len(cond) == 1
    assert torch.equal(cond[0][1]["mask"], mask)
    assert cond[0][1]["mask_strength"] == 1.0


def test_vae_decode(vae, has_gpu):
    if not has_gpu:
        pytest.skip("requires gpu for performant testing")
    decoded, = VAEDecode().decode(vae, _latent)
    assert decoded.shape == (1, 512, 512, 3)


def test_vae_encode(has_gpu, vae):
    if not has_gpu:
        pytest.skip("requires gpu for performant testing")
    latent, = VAEEncode().encode(vae, _image_512x512)
    assert "samples" in latent
    assert latent["samples"].shape == (1, 4, 64, 64)


def test_vae_encode_for_inpaint(has_gpu, vae):
    if not has_gpu:
        pytest.skip("requires gpu for performant testing")
    mask = torch.ones((1, 512, 512))
    latent, = VAEEncodeForInpaint().encode(vae, _image_512x512, mask)
    assert "samples" in latent
    assert latent["samples"].shape == (1, 4, 64, 64)
    assert "noise_mask" in latent
    assert torch.allclose(latent["noise_mask"], mask)


def test_inpaint_model_conditioning(model, vae, clip, has_gpu):
    if not has_gpu:
        pytest.skip("requires gpu for performant testing")
    cond_pos, = CLIPTextEncode().encode(clip, "test prompt")
    cond_neg, = CLIPTextEncode().encode(clip, "test negative prompt")
    pos, neg, latent = InpaintModelConditioning().encode(cond_pos, cond_neg, _image_512x512, vae, torch.ones((1, 512, 512)), noise_mask=True)
    assert len(pos) == len(cond_pos)
    assert len(neg) == len(cond_neg)
    assert "samples" in latent
    assert "noise_mask" in latent


def test_latent_upscale():
    latent, = LatentUpscale().upscale(_latent, "nearest-exact", 1024, 1024, "disabled")
    assert latent["samples"].shape == (1, 4, 128, 128)


def test_latent_upscale_by():
    latent, = LatentUpscaleBy().upscale(_latent, "nearest-exact", 2.0)
    assert latent["samples"].shape == (1, 4, 128, 128)


def test_latent_rotate():
    latent, = LatentRotate().rotate(_latent, "90 degrees")
    assert latent["samples"].shape == (1, 4, 64, 64)


def test_latent_flip():
    latent, = LatentFlip().flip(_latent, "y-axis: horizontally")
    assert latent["samples"].shape == (1, 4, 64, 64)


def test_latent_composite():
    latent, = LatentComposite().composite(_latent, _latent, 0, 0)
    assert latent["samples"].shape == (1, 4, 64, 64)


def test_latent_crop():
    latent, = LatentCrop().crop(_latent, 32, 32, 0, 0)
    assert latent["samples"].shape == (1, 4, 4, 4)


def test_image_scale():
    image, = ImageScale().upscale(_image_1x1, "nearest-exact", 64, 64, "disabled")
    assert image.shape == (1, 64, 64, 3)


def test_image_scale_by():
    image, = ImageScaleBy().upscale(_image_1x1, "nearest-exact", 2.0)
    assert image.shape == (1, 2, 2, 3)


def test_image_invert():
    image, = ImageInvert().invert(_image_1x1)
    assert image.shape == (1, 1, 1, 3)
    assert torch.allclose(image, 1.0 - _image_1x1)


def test_image_batch():
    image, = ImageBatch().batch(_image_1x1, _image_1x1)
    assert image.shape == (2, 1, 1, 3)


def test_image_pad_for_outpaint():
    padded, mask = ImagePadForOutpaint().expand_image(_image_1x1, 1, 1, 1, 1, 0)
    assert padded.shape == (1, 3, 3, 3)
    # the mask should now be batched
    assert mask.shape == (1, 3, 3)


def test_image_pad_for_outpaint_batched():
    padded, mask = ImagePadForOutpaint().expand_image(_image_1x1.expand(2, -1, -1, -1), 1, 1, 1, 1, 0)
    assert padded.shape == (2, 3, 3, 3)
    # the mask should now be batched
    assert mask.shape == (2, 3, 3)


def test_empty_image():
    image, = EmptyImage().generate(64, 64, 1, 0xFF0000)
    assert image.shape == (1, 64, 64, 3)
    assert torch.allclose(image[0, 0, 0], torch.tensor([1.0, 0.0, 0.0]))
