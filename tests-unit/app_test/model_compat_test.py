from app.model_compat import group_shards, judge, parse_repo, precision, role, summarize_repo

GIB = 1024 ** 3
HW = {
    "type": "cuda",
    "vram_total": 16 * GIB,
    "ram_total": 32 * GIB,
    "supports": {"bf16": True, "fp8": True, "fp4": True},
}
OLD_GPU = {**HW, "supports": {"bf16": False, "fp8": False, "fp4": False}}


def test_precision_from_file_name():
    assert precision("z_image_turbo_nvfp4.safetensors") == "FP4"
    assert precision("flux1-dev-fp8-e4m3fn.safetensors") == "FP8"
    assert precision("boogu_image_edit_int8_convrot.safetensors") == "INT8"
    assert precision("lumina_2_model_bf16.safetensors") == "BF16"
    assert precision("flux1-dev-Q4_K_S.gguf") == "GGUF Q4_K_S"
    assert precision("qwen-image-IQ4_XS.gguf") == "GGUF IQ4_XS"
    assert precision("ae.safetensors") is None


def test_role_prefers_folder_over_name():
    assert role("diffusion_models/qwen_image_bf16.safetensors") == "diffusion"
    assert role("split_files/text_encoders/qwen_3_4b.safetensors") == "text_encoder"
    assert role("text_encoder_2/model-00001-of-00002.safetensors") == "text_encoder"
    assert role("embeddings/minimaxh3_four_seasons.safetensors") == "embedding"
    assert role("ae.safetensors") == "vae"
    assert role("flux1-dev.safetensors") == "model"


def test_judge_levels():
    assert judge(11 * GIB, "BF16", HW)["level"] == "fits"
    assert judge(15 * GIB, "BF16", HW)["level"] == "offload"
    assert judge(40 * GIB, "BF16", HW)["level"] == "no"


def test_judge_notes_unsupported_precision():
    notes = judge(4 * GIB, "FP4", OLD_GPU)["notes"]
    assert any("FP4" in note for note in notes)


def test_judge_without_gpu_never_fits():
    cpu = {**HW, "type": "cpu"}
    assert judge(1 * GIB, None, cpu)["level"] == "offload"


def test_group_shards_sums_parts():
    files = group_shards([
        ("transformer/diffusion_pytorch_model-00001-of-00002.safetensors", 5 * GIB),
        ("transformer/diffusion_pytorch_model-00002-of-00002.safetensors", 3 * GIB),
        ("vae/ae.safetensors", GIB),
    ])
    assert ("transformer/diffusion_pytorch_model.safetensors (2 shards)", 8 * GIB) in files
    assert len(files) == 2


def test_summarize_picks_main_model_over_small_files():
    files = [
        ("checkpoints/sdpose_wholebody_fp16.safetensors", 2 * GIB),
        ("diffusion_models/rt_detr_v4-x-hgnet_fp32.safetensors", GIB // 4),
        ("embeddings/tiny.safetensors", 1024),
    ]
    summary = summarize_repo("Comfy-Org/SDPose", files, HW)
    assert summary["level"] == "fits"
    assert summary["best"]["name"] == "sdpose_wholebody_fp16.safetensors"


def test_summarize_shows_smallest_when_nothing_fits():
    files = [
        ("diffusion_models/big_bf16.safetensors", 60 * GIB),
        ("diffusion_models/big_int8.safetensors", 35 * GIB),
    ]
    summary = summarize_repo("org/big", files, HW)
    assert summary["level"] == "no"
    assert summary["best"]["name"] == "big_int8.safetensors"


def test_parse_repo():
    assert parse_repo("Comfy-Org/z_image_turbo") == "Comfy-Org/z_image_turbo"
    assert parse_repo("https://huggingface.co/Comfy-Org/z_image_turbo/tree/main") == "Comfy-Org/z_image_turbo"
    assert parse_repo("not a repo") is None
