from comfy.ldm.modules import attention as comfy_attention
import logging
import comfy.model_patcher
import comfy.utils
import comfy.sd
import torch
import folder_paths
import comfy.model_management as mm
from comfy.cli_args import args

sageattn_modes = ["disabled", "auto", "sageattn_qk_int8_pv_fp16_cuda", "sageattn_qk_int8_pv_fp16_triton", "sageattn_qk_int8_pv_fp8_cuda", "sageattn_qk_int8_pv_fp8_cuda++"]

_initialized = False
_original_functions = {}

if not _initialized:
    _original_functions["orig_attention"] = comfy_attention.optimized_attention
    _original_functions["original_patch_model"] = comfy.model_patcher.ModelPatcher.patch_model
    _original_functions["original_load_lora_for_models"] = comfy.sd.load_lora_for_models
    _initialized = True

class BaseLoaderKJ:
    original_linear = None
    cublas_patched = False

    @torch.compiler.disable()
    def _patch_modules(self, patch_cublaslinear, sage_attention):
        from comfy.ops import disable_weight_init, CastWeightBiasOp, cast_bias_weight

        if sage_attention != "disabled":
            print("Patching comfy attention to use sageattn")
            from sageattention import sageattn
            def set_sage_func(sage_attention):
                if sage_attention == "auto":
                    def func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
                        return sageattn(q, k, v, is_causal=is_causal, attn_mask=attn_mask, tensor_layout=tensor_layout)
                    return func
                elif sage_attention == "sageattn_qk_int8_pv_fp16_cuda":
                    from sageattention import sageattn_qk_int8_pv_fp16_cuda
                    def func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
                        return sageattn_qk_int8_pv_fp16_cuda(q, k, v, is_causal=is_causal, attn_mask=attn_mask, pv_accum_dtype="fp32", tensor_layout=tensor_layout)
                    return func
                elif sage_attention == "sageattn_qk_int8_pv_fp16_triton":
                    from sageattention import sageattn_qk_int8_pv_fp16_triton
                    def func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
                        return sageattn_qk_int8_pv_fp16_triton(q, k, v, is_causal=is_causal, attn_mask=attn_mask, tensor_layout=tensor_layout)
                    return func
                elif sage_attention == "sageattn_qk_int8_pv_fp8_cuda":
                    from sageattention import sageattn_qk_int8_pv_fp8_cuda
                    def func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
                        return sageattn_qk_int8_pv_fp8_cuda(q, k, v, is_causal=is_causal, attn_mask=attn_mask, pv_accum_dtype="fp32+fp32", tensor_layout=tensor_layout)
                    return func
                elif sage_attention == "sageattn_qk_int8_pv_fp8_cuda++":
                    from sageattention import sageattn_qk_int8_pv_fp8_cuda
                    def func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
                        return sageattn_qk_int8_pv_fp8_cuda(q, k, v, is_causal=is_causal, attn_mask=attn_mask, pv_accum_dtype="fp32+fp16", tensor_layout=tensor_layout)
                    return func

            sage_func = set_sage_func(sage_attention)

            @torch.compiler.disable()
            def attention_sage(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False):
                if skip_reshape:
                    b, _, _, dim_head = q.shape
                    tensor_layout="HND"
                else:
                    b, _, dim_head = q.shape
                    dim_head //= heads
                    q, k, v = map(
                        lambda t: t.view(b, -1, heads, dim_head),
                        (q, k, v),
                    )
                    tensor_layout="NHD"
                if mask is not None:
                    # add a batch dimension if there isn't already one
                    if mask.ndim == 2:
                        mask = mask.unsqueeze(0)
                    # add a heads dimension if there isn't already one
                    if mask.ndim == 3:
                        mask = mask.unsqueeze(1)
                out = sage_func(q, k, v, attn_mask=mask, is_causal=False, tensor_layout=tensor_layout)
                if tensor_layout == "HND":
                    if not skip_output_reshape:
                        out = (
                            out.transpose(1, 2).reshape(b, -1, heads * dim_head)
                        )
                else:
                    if skip_output_reshape:
                        out = out.transpose(1, 2)
                    else:
                        out = out.reshape(b, -1, heads * dim_head)
                return out

            comfy_attention.optimized_attention = attention_sage
            comfy.ldm.hunyuan_video.model.optimized_attention = attention_sage
            comfy.ldm.flux.math.optimized_attention = attention_sage
            comfy.ldm.genmo.joint_model.asymm_models_joint.optimized_attention = attention_sage
            comfy.ldm.cosmos.blocks.optimized_attention = attention_sage
            comfy.ldm.wan.model.optimized_attention = attention_sage

        else:
            print("Restoring initial comfy attention")
            comfy_attention.optimized_attention = _original_functions.get("orig_attention")
            comfy.ldm.hunyuan_video.model.optimized_attention = _original_functions.get("orig_attention")
            comfy.ldm.flux.math.optimized_attention = _original_functions.get("orig_attention")
            comfy.ldm.genmo.joint_model.asymm_models_joint.optimized_attention = _original_functions.get("orig_attention")
            comfy.ldm.cosmos.blocks.optimized_attention = _original_functions.get("orig_attention")
            comfy.ldm.wan.model.optimized_attention = _original_functions.get("orig_attention")

        if patch_cublaslinear:
            if not BaseLoaderKJ.cublas_patched:
                BaseLoaderKJ.original_linear = disable_weight_init.Linear
                try:
                    from cublas_ops import CublasLinear
                except ImportError:
                    raise Exception("Can't import 'torch-cublas-hgemm', install it from here https://github.com/aredden/torch-cublas-hgemm")

                class PatchedLinear(CublasLinear, CastWeightBiasOp):
                    def reset_parameters(self):
                        pass

                    def forward_comfy_cast_weights(self, input):
                        weight, bias = cast_bias_weight(self, input)
                        return torch.nn.functional.linear(input, weight, bias)

                    def forward(self, *args, **kwargs):
                        if self.comfy_cast_weights:
                            return self.forward_comfy_cast_weights(*args, **kwargs)
                        else:
                            return super().forward(*args, **kwargs)

                disable_weight_init.Linear = PatchedLinear
                BaseLoaderKJ.cublas_patched = True
        else:
            if BaseLoaderKJ.cublas_patched:
                disable_weight_init.Linear = BaseLoaderKJ.original_linear
                BaseLoaderKJ.cublas_patched = False

from comfy.patcher_extension import CallbacksMP
class PathchSageAttentionKJ(BaseLoaderKJ):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("MODEL",),
            "sage_attention": (sageattn_modes, {"default": False, "tooltip": "Global patch comfy attention to use sageattn, once patched to revert back to normal you would need to run this node again with disabled option."}),
        }}

    RETURN_TYPES = ("MODEL", )
    FUNCTION = "patch"
    DESCRIPTION = "Experimental node for patching attention mode. This doesn't use the model patching system and thus can't be disabled without running the node again with 'disabled' option."
    EXPERIMENTAL = True
    CATEGORY = "KJNodes/experimental"

    def patch(self, model, sage_attention):
        model_clone = model.clone()
        @torch.compiler.disable()
        def patch_attention_enable(model):
            self._patch_modules(False, sage_attention)
        @torch.compiler.disable()
        def patch_attention_disable(model):
            self._patch_modules(False, "disabled")
        
        model_clone.add_callback(CallbacksMP.ON_PRE_RUN, patch_attention_enable)
        model_clone.add_callback(CallbacksMP.ON_CLEANUP, patch_attention_disable)
        
        return model_clone,
 
class CheckpointLoaderKJ(BaseLoaderKJ):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "ckpt_name": (folder_paths.get_filename_list("checkpoints"), {"tooltip": "The name of the checkpoint (model) to load."}),
            "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2", "fp16", "bf16", "fp32"],),
            "compute_dtype": (["default", "fp16", "bf16", "fp32"], {"default": "default", "tooltip": "The compute dtype to use for the model."}),
            "patch_cublaslinear": ("BOOLEAN", {"default": False, "tooltip": "Enable or disable the patching, won't take effect on already loaded models!"}),
            "sage_attention": (sageattn_modes, {"default": False, "tooltip": "Patch comfy attention to use sageattn."}),
            "enable_fp16_accumulation": ("BOOLEAN", {"default": False, "tooltip": "Enable torch.backends.cuda.matmul.allow_fp16_accumulation, requires pytorch 2.7.0 nightly."}),
        }}

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "patch"
    DESCRIPTION = "Experimental node for patching torch.nn.Linear with CublasLinear."
    EXPERIMENTAL = True
    CATEGORY = "KJNodes/experimental"

    def patch(self, ckpt_name, weight_dtype, compute_dtype, patch_cublaslinear, sage_attention, enable_fp16_accumulation):
        DTYPE_MAP = {
            "fp8_e4m3fn": torch.float8_e4m3fn,
            "fp8_e5m2": torch.float8_e5m2,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32
        }
        model_options = {}
        if dtype := DTYPE_MAP.get(weight_dtype):
            model_options["dtype"] = dtype
            print(f"Setting {ckpt_name} weight dtype to {dtype}")

        if weight_dtype == "fp8_e4m3fn_fast":
            model_options["dtype"] = torch.float8_e4m3fn
            model_options["fp8_optimizations"] = True

        ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
        sd, metadata = comfy.utils.load_torch_file(ckpt_path, return_metadata=True)
    
        model, clip, vae = self.load_state_dict_guess_config(
            sd,
            output_vae=True, 
            output_clip=True, 
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            metadata=metadata,
            model_options=model_options)

        if dtype := DTYPE_MAP.get(compute_dtype):
            model.set_model_compute_dtype(dtype)
            model.force_cast_weights = False
            print(f"Setting {ckpt_name} compute dtype to {dtype}")

        if enable_fp16_accumulation:
            if hasattr(torch.backends.cuda.matmul, "allow_fp16_accumulation"):
                torch.backends.cuda.matmul.allow_fp16_accumulation = True
            else:
                raise RuntimeError("Failed to set fp16 accumulation, this requires pytorch 2.7.0 nightly currently")
        else:
            if hasattr(torch.backends.cuda.matmul, "allow_fp16_accumulation"):
                torch.backends.cuda.matmul.allow_fp16_accumulation = False

        def patch_attention(model):
            self._patch_modules(patch_cublaslinear, sage_attention)
        model.add_callback(CallbacksMP.ON_PRE_RUN,patch_attention)
        return model, clip, vae
    
    def load_state_dict_guess_config(self, sd, output_vae=True, output_clip=True, embedding_directory=None, output_model=True, model_options={}, te_model_options={}, metadata=None):
        from comfy.sd import load_diffusion_model_state_dict, model_detection, VAE, CLIP
        clip = None
        vae = None
        model = None
        model_patcher = None

        diffusion_model_prefix = model_detection.unet_prefix_from_state_dict(sd)
        parameters = comfy.utils.calculate_parameters(sd, diffusion_model_prefix)
        weight_dtype = comfy.utils.weight_dtype(sd, diffusion_model_prefix)
        load_device = mm.get_torch_device()

        model_config = model_detection.model_config_from_unet(sd, diffusion_model_prefix, metadata=metadata)
        if model_config is None:
            logging.warning("Warning, This is not a checkpoint file, trying to load it as a diffusion model only.")
            diffusion_model = load_diffusion_model_state_dict(sd, model_options={})
            if diffusion_model is None:
                return None
            return (diffusion_model, None, VAE(sd={}), None)  # The VAE object is there to throw an exception if it's actually used'


        unet_weight_dtype = list(model_config.supported_inference_dtypes)
        if model_config.scaled_fp8 is not None:
            weight_dtype = None

        model_config.custom_operations = model_options.get("custom_operations", None)
        unet_dtype = model_options.get("dtype", model_options.get("weight_dtype", None))

        if unet_dtype is None:
            unet_dtype = mm.unet_dtype(model_params=parameters, supported_dtypes=unet_weight_dtype, weight_dtype=weight_dtype)

        manual_cast_dtype = mm.unet_manual_cast(unet_dtype, load_device, model_config.supported_inference_dtypes)
        model_config.set_inference_dtype(unet_dtype, manual_cast_dtype)

        if output_model:
            inital_load_device = mm.unet_inital_load_device(parameters, unet_dtype)
            model = model_config.get_model(sd, diffusion_model_prefix, device=inital_load_device)
            model.load_model_weights(sd, diffusion_model_prefix)

        if output_vae:
            vae_sd = comfy.utils.state_dict_prefix_replace(sd, {k: "" for k in model_config.vae_key_prefix}, filter_keys=True)
            vae_sd = model_config.process_vae_state_dict(vae_sd)
            vae = VAE(sd=vae_sd, metadata=metadata)

        if output_clip:
            clip_target = model_config.clip_target(state_dict=sd)
            if clip_target is not None:
                clip_sd = model_config.process_clip_state_dict(sd)
                if len(clip_sd) > 0:
                    parameters = comfy.utils.calculate_parameters(clip_sd)
                    clip = CLIP(clip_target, embedding_directory=embedding_directory, tokenizer_data=clip_sd, parameters=parameters, model_options=te_model_options)
                    m, u = clip.load_sd(clip_sd, full_model=True)
                    if len(m) > 0:
                        m_filter = list(filter(lambda a: ".logit_scale" not in a and ".transformer.text_projection.weight" not in a, m))
                        if len(m_filter) > 0:
                            logging.warning("clip missing: {}".format(m))
                        else:
                            logging.debug("clip missing: {}".format(m))

                    if len(u) > 0:
                        logging.debug("clip unexpected {}:".format(u))
                else:
                    logging.warning("no CLIP/text encoder weights in checkpoint, the text encoder model will not be loaded.")

        left_over = sd.keys()
        if len(left_over) > 0:
            logging.debug("left over keys: {}".format(left_over))

        if output_model:
            model_patcher = comfy.model_patcher.ModelPatcher(model, load_device=load_device, offload_device=mm.unet_offload_device())
            if inital_load_device != torch.device("cpu"):
                logging.info("loaded diffusion model directly to GPU")
                mm.load_models_gpu([model_patcher], force_full_load=True)

        return (model_patcher, clip, vae)

class DiffusionModelLoaderKJ(BaseLoaderKJ):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model_name": (folder_paths.get_filename_list("diffusion_models"), {"tooltip": "The name of the checkpoint (model) to load."}),
            "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2", "fp16", "bf16", "fp32"],),
            "compute_dtype": (["default", "fp16", "bf16", "fp32"], {"default": "default", "tooltip": "The compute dtype to use for the model."}),
            "patch_cublaslinear": ("BOOLEAN", {"default": False, "tooltip": "Enable or disable the patching, won't take effect on already loaded models!"}),
            "sage_attention": (sageattn_modes, {"default": False, "tooltip": "Patch comfy attention to use sageattn."}),
            "enable_fp16_accumulation": ("BOOLEAN", {"default": False, "tooltip": "Enable torch.backends.cuda.matmul.allow_fp16_accumulation, requires pytorch 2.7.0 nightly."}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch_and_load"
    DESCRIPTION = "Node for patching torch.nn.Linear with CublasLinear."
    EXPERIMENTAL = True
    CATEGORY = "KJNodes/experimental"

    def patch_and_load(self, model_name, weight_dtype, compute_dtype, patch_cublaslinear, sage_attention, enable_fp16_accumulation):        
        DTYPE_MAP = {
            "fp8_e4m3fn": torch.float8_e4m3fn,
            "fp8_e5m2": torch.float8_e5m2,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32
        }
        model_options = {}
        if dtype := DTYPE_MAP.get(weight_dtype):
            model_options["dtype"] = dtype
            print(f"Setting {model_name} weight dtype to {dtype}")
        
        if weight_dtype == "fp8_e4m3fn_fast":
            model_options["dtype"] = torch.float8_e4m3fn
            model_options["fp8_optimizations"] = True
        
        if enable_fp16_accumulation:
            if hasattr(torch.backends.cuda.matmul, "allow_fp16_accumulation"):
                torch.backends.cuda.matmul.allow_fp16_accumulation = True
            else:
                raise RuntimeError("Failed to set fp16 accumulation, this requires pytorch 2.7.0 nightly currently")
        else:
            if hasattr(torch.backends.cuda.matmul, "allow_fp16_accumulation"):
                torch.backends.cuda.matmul.allow_fp16_accumulation = False

        unet_path = folder_paths.get_full_path_or_raise("diffusion_models", model_name)
        model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)
        if dtype := DTYPE_MAP.get(compute_dtype):
            model.set_model_compute_dtype(dtype)
            model.force_cast_weights = False
            print(f"Setting {model_name} compute dtype to {dtype}")

        def patch_attention(model):
            self._patch_modules(patch_cublaslinear, sage_attention)
        model.add_callback(CallbacksMP.ON_PRE_RUN,patch_attention)
        
        return (model,)

class ModelPatchTorchSettings:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("MODEL",),
            "enable_fp16_accumulation": ("BOOLEAN", {"default": False, "tooltip": "Enable torch.backends.cuda.matmul.allow_fp16_accumulation, requires pytorch 2.7.0 nightly."}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    DESCRIPTION = "Adds callbacks to model to set torch settings before and after running the model."
    EXPERIMENTAL = True
    CATEGORY = "KJNodes/experimental"

    def patch(self, model, enable_fp16_accumulation):        
        model_clone = model.clone()

        def patch_enable_fp16_accum(model):
            print("Patching torch settings: torch.backends.cuda.matmul.allow_fp16_accumulation = True")
            torch.backends.cuda.matmul.allow_fp16_accumulation = True
        def patch_disable_fp16_accum(model):
            print("Patching torch settings: torch.backends.cuda.matmul.allow_fp16_accumulation = False")
            torch.backends.cuda.matmul.allow_fp16_accumulation = False
        
        if enable_fp16_accumulation:
            if hasattr(torch.backends.cuda.matmul, "allow_fp16_accumulation"):
                model_clone.add_callback(CallbacksMP.ON_PRE_RUN, patch_enable_fp16_accum)
                model_clone.add_callback(CallbacksMP.ON_CLEANUP, patch_disable_fp16_accum)
            else:
                raise RuntimeError("Failed to set fp16 accumulation, this requires pytorch 2.7.0 nightly currently")
        else:
            if hasattr(torch.backends.cuda.matmul, "allow_fp16_accumulation"):
                model_clone.add_callback(CallbacksMP.ON_PRE_RUN, patch_disable_fp16_accum)
            else:
                raise RuntimeError("Failed to set fp16 accumulation, this requires pytorch 2.7.0 nightly currently")
                
        return (model_clone,)
    
def patched_patch_model(self, device_to=None, lowvram_model_memory=0, load_weights=True, force_patch_weights=False):
    with self.use_ejected():

        device_to = mm.get_torch_device()

        full_load_override = getattr(self.model, "full_load_override", "auto")
        if full_load_override in ["enabled", "disabled"]:
            full_load = full_load_override == "enabled"
        else:
            full_load = lowvram_model_memory == 0

        self.load(device_to, lowvram_model_memory=lowvram_model_memory, force_patch_weights=force_patch_weights, full_load=full_load)

        for k in self.object_patches:
            old = comfy.utils.set_attr(self.model, k, self.object_patches[k])
            if k not in self.object_patches_backup:
                self.object_patches_backup[k] = old
       
    self.inject_model()
    return self.model

def patched_load_lora_for_models(model, clip, lora, strength_model, strength_clip):

    patch_keys = list(model.object_patches_backup.keys())
    for k in patch_keys:
        #print("backing up object patch: ", k)
        comfy.utils.set_attr(model.model, k, model.object_patches_backup[k])

    key_map = {}
    if model is not None:
        key_map = comfy.lora.model_lora_keys_unet(model.model, key_map)
    if clip is not None:
        key_map = comfy.lora.model_lora_keys_clip(clip.cond_stage_model, key_map)

    lora = comfy.lora_convert.convert_lora(lora)
    loaded = comfy.lora.load_lora(lora, key_map)
    #print(temp_object_patches_backup)
   
    if model is not None:
        new_modelpatcher = model.clone()
        k = new_modelpatcher.add_patches(loaded, strength_model)
    else:
        k = ()
        new_modelpatcher = None

    if clip is not None:
        new_clip = clip.clone()
        k1 = new_clip.add_patches(loaded, strength_clip)
    else:
        k1 = ()
        new_clip = None
    k = set(k)
    k1 = set(k1)
    for x in loaded:
        if (x not in k) and (x not in k1):
            print("NOT LOADED {}".format(x))

    if patch_keys:
        if hasattr(model.model, "compile_settings"):
            compile_settings = getattr(model.model, "compile_settings")
            print("compile_settings: ", compile_settings)
            for k in patch_keys:
                if "diffusion_model." in k:
                    # Remove the prefix to get the attribute path
                    key = k.replace('diffusion_model.', '')
                    attributes = key.split('.')
                    # Start with the diffusion_model object
                    block = model.get_model_object("diffusion_model")
                    # Navigate through the attributes to get to the block
                    for attr in attributes:
                        if attr.isdigit():
                            block = block[int(attr)]
                        else:
                            block = getattr(block, attr)
                    # Compile the block
                    compiled_block = torch.compile(block, mode=compile_settings["mode"], dynamic=compile_settings["dynamic"], fullgraph=compile_settings["fullgraph"], backend=compile_settings["backend"])
                    # Add the compiled block back as an object patch
                    model.add_object_patch(k, compiled_block)
    return (new_modelpatcher, new_clip)

class PatchModelPatcherOrder:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
                    "model": ("MODEL",),
                    "patch_order": (["object_patch_first", "weight_patch_first"], {"default": "weight_patch_first", "tooltip": "Patch the comfy patch_model function to load weight patches (LoRAs) before compiling the model"}),
                    "full_load": (["enabled", "disabled", "auto"], {"default": "auto", "tooltip": "Disabling may help with memory issues when loading large models, when changing this you should probably force model reload to avoid issues!"}),
                }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "KJNodes/experimental"
    DESCRIPTION = "Patch the comfy patch_model function patching order, useful for torch.compile (used as object_patch) as it should come last if you want to use LoRAs with compile"
    EXPERIMENTAL = True

    def patch(self, model, patch_order, full_load):
        comfy.model_patcher.ModelPatcher.temp_object_patches_backup = {}
        setattr(model.model, "full_load_override", full_load)
        if patch_order == "weight_patch_first":
            comfy.model_patcher.ModelPatcher.patch_model = patched_patch_model
            comfy.sd.load_lora_for_models = patched_load_lora_for_models
        else:
            comfy.model_patcher.ModelPatcher.patch_model = _original_functions.get("original_patch_model")
            comfy.sd.load_lora_for_models = _original_functions.get("original_load_lora_for_models")
        
        return model,

class TorchCompileModelFluxAdvanced:
    def __init__(self):
        self._compiled = False

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
                    "model": ("MODEL",),
                    "backend": (["inductor", "cudagraphs"],),
                    "fullgraph": ("BOOLEAN", {"default": False, "tooltip": "Enable full graph mode"}),
                    "mode": (["default", "max-autotune", "max-autotune-no-cudagraphs", "reduce-overhead"], {"default": "default"}),
                    "double_blocks": ("STRING", {"default": "0-18", "multiline": True}),
                    "single_blocks": ("STRING", {"default": "0-37", "multiline": True}),
                    "dynamic": ("BOOLEAN", {"default": False, "tooltip": "Enable dynamic mode"}),
                },
                "optional": {
                    "dynamo_cache_size_limit": ("INT", {"default": 64, "min": 0, "max": 1024, "step": 1, "tooltip": "torch._dynamo.config.cache_size_limit"}),
                }
                }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "KJNodes/torchcompile"
    EXPERIMENTAL = True
    DEPRECATED = True

    def parse_blocks(self, blocks_str):
        blocks = []
        for part in blocks_str.split(','):
            part = part.strip()
            if '-' in part:
                start, end = map(int, part.split('-'))
                blocks.extend(range(start, end + 1))
            else:
                blocks.append(int(part))
        return blocks

    def patch(self, model, backend, mode, fullgraph, single_blocks, double_blocks, dynamic, dynamo_cache_size_limit):
        single_block_list = self.parse_blocks(single_blocks)
        double_block_list = self.parse_blocks(double_blocks)
        m = model.clone()
        diffusion_model = m.get_model_object("diffusion_model")
        torch._dynamo.config.cache_size_limit = dynamo_cache_size_limit
        
        if not self._compiled:
            try:
                for i, block in enumerate(diffusion_model.double_blocks):
                    if i in double_block_list:
                        #print("Compiling double_block", i)
                        m.add_object_patch(f"diffusion_model.double_blocks.{i}", torch.compile(block, mode=mode, dynamic=dynamic, fullgraph=fullgraph, backend=backend))
                for i, block in enumerate(diffusion_model.single_blocks):
                    if i in single_block_list:
                        #print("Compiling single block", i)
                        m.add_object_patch(f"diffusion_model.single_blocks.{i}", torch.compile(block, mode=mode, dynamic=dynamic, fullgraph=fullgraph, backend=backend))
                self._compiled = True
                compile_settings = {
                    "backend": backend,
                    "mode": mode,
                    "fullgraph": fullgraph,
                    "dynamic": dynamic,
                }
                setattr(m.model, "compile_settings", compile_settings)
            except:
                raise RuntimeError("Failed to compile model")
        
        return (m, )
        # rest of the layers that are not patched
        # diffusion_model.final_layer = torch.compile(diffusion_model.final_layer, mode=mode, fullgraph=fullgraph, backend=backend)
        # diffusion_model.guidance_in = torch.compile(diffusion_model.guidance_in, mode=mode, fullgraph=fullgraph, backend=backend)
        # diffusion_model.img_in = torch.compile(diffusion_model.img_in, mode=mode, fullgraph=fullgraph, backend=backend)
        # diffusion_model.time_in = torch.compile(diffusion_model.time_in, mode=mode, fullgraph=fullgraph, backend=backend)
        # diffusion_model.txt_in = torch.compile(diffusion_model.txt_in, mode=mode, fullgraph=fullgraph, backend=backend)
        # diffusion_model.vector_in = torch.compile(diffusion_model.vector_in, mode=mode, fullgraph=fullgraph, backend=backend)

class TorchCompileModelFluxAdvancedV2:
    def __init__(self):
        self._compiled = False

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
                    "model": ("MODEL",),
                    "backend": (["inductor", "cudagraphs"],),
                    "fullgraph": ("BOOLEAN", {"default": False, "tooltip": "Enable full graph mode"}),
                    "mode": (["default", "max-autotune", "max-autotune-no-cudagraphs", "reduce-overhead"], {"default": "default"}),
                    "double_blocks": ("BOOLEAN", {"default": True, "tooltip": "Compile double blocks"}),
                    "single_blocks": ("BOOLEAN", {"default": True, "tooltip": "Compile single blocks"}),
                    "dynamic": ("BOOLEAN", {"default": False, "tooltip": "Enable dynamic mode"}),
                },
                "optional": {
                    "dynamo_cache_size_limit": ("INT", {"default": 64, "min": 0, "max": 1024, "step": 1, "tooltip": "torch._dynamo.config.cache_size_limit"}),
                }
                }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "KJNodes/torchcompile"
    EXPERIMENTAL = True

    def patch(self, model, backend, mode, fullgraph, single_blocks, double_blocks, dynamic, dynamo_cache_size_limit):
        from comfy_api.torch_helpers import set_torch_compile_wrapper
        m = model.clone()
        diffusion_model = m.get_model_object("diffusion_model")
        torch._dynamo.config.cache_size_limit = dynamo_cache_size_limit

        compile_key_list = []
        
        try:
            if double_blocks:
                for i, block in enumerate(diffusion_model.double_blocks):
                    compile_key_list.append(f"diffusion_model.double_blocks.{i}")
            if single_blocks:
                for i, block in enumerate(diffusion_model.single_blocks):
                    compile_key_list.append(f"diffusion_model.single_blocks.{i}")

            set_torch_compile_wrapper(model=m, keys=compile_key_list, backend=backend, mode=mode, dynamic=dynamic, fullgraph=fullgraph)           
        except:
            raise RuntimeError("Failed to compile model")
        
        return (m, )
        # rest of the layers that are not patched
        # diffusion_model.final_layer = torch.compile(diffusion_model.final_layer, mode=mode, fullgraph=fullgraph, backend=backend)
        # diffusion_model.guidance_in = torch.compile(diffusion_model.guidance_in, mode=mode, fullgraph=fullgraph, backend=backend)
        # diffusion_model.img_in = torch.compile(diffusion_model.img_in, mode=mode, fullgraph=fullgraph, backend=backend)
        # diffusion_model.time_in = torch.compile(diffusion_model.time_in, mode=mode, fullgraph=fullgraph, backend=backend)
        # diffusion_model.txt_in = torch.compile(diffusion_model.txt_in, mode=mode, fullgraph=fullgraph, backend=backend)
        # diffusion_model.vector_in = torch.compile(diffusion_model.vector_in, mode=mode, fullgraph=fullgraph, backend=backend)

    
class TorchCompileModelHyVideo:
    def __init__(self):
        self._compiled = False

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "backend": (["inductor","cudagraphs"], {"default": "inductor"}),
                "fullgraph": ("BOOLEAN", {"default": False, "tooltip": "Enable full graph mode"}),
                "mode": (["default", "max-autotune", "max-autotune-no-cudagraphs", "reduce-overhead"], {"default": "default"}),
                "dynamic": ("BOOLEAN", {"default": False, "tooltip": "Enable dynamic mode"}),
                "dynamo_cache_size_limit": ("INT", {"default": 64, "min": 0, "max": 1024, "step": 1, "tooltip": "torch._dynamo.config.cache_size_limit"}),
                "compile_single_blocks": ("BOOLEAN", {"default": True, "tooltip": "Compile single blocks"}),
                "compile_double_blocks": ("BOOLEAN", {"default": True, "tooltip": "Compile double blocks"}),
                "compile_txt_in": ("BOOLEAN", {"default": False, "tooltip": "Compile txt_in layers"}),
                "compile_vector_in": ("BOOLEAN", {"default": False, "tooltip": "Compile vector_in layers"}),
                "compile_final_layer": ("BOOLEAN", {"default": False, "tooltip": "Compile final layer"}),

            },
        }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "KJNodes/torchcompile"
    EXPERIMENTAL = True

    def patch(self, model, backend, fullgraph, mode, dynamic, dynamo_cache_size_limit, compile_single_blocks, compile_double_blocks, compile_txt_in, compile_vector_in, compile_final_layer):
        m = model.clone()
        diffusion_model = m.get_model_object("diffusion_model")
        torch._dynamo.config.cache_size_limit = dynamo_cache_size_limit
        if not self._compiled:
            try:
                if compile_single_blocks:
                    for i, block in enumerate(diffusion_model.single_blocks):
                        compiled_block = torch.compile(block, fullgraph=fullgraph, dynamic=dynamic, backend=backend, mode=mode)
                        m.add_object_patch(f"diffusion_model.single_blocks.{i}", compiled_block)
                if compile_double_blocks:
                    for i, block in enumerate(diffusion_model.double_blocks):
                        compiled_block = torch.compile(block, fullgraph=fullgraph, dynamic=dynamic, backend=backend, mode=mode)
                        m.add_object_patch(f"diffusion_model.double_blocks.{i}", compiled_block)
                if compile_txt_in:
                    compiled_block = torch.compile(diffusion_model.txt_in, fullgraph=fullgraph, dynamic=dynamic, backend=backend, mode=mode)
                    m.add_object_patch("diffusion_model.txt_in", compiled_block)
                if compile_vector_in:
                    compiled_block = torch.compile(diffusion_model.vector_in, fullgraph=fullgraph, dynamic=dynamic, backend=backend, mode=mode)
                    m.add_object_patch("diffusion_model.vector_in", compiled_block)
                if compile_final_layer:
                    compiled_block = torch.compile(diffusion_model.final_layer, fullgraph=fullgraph, dynamic=dynamic, backend=backend, mode=mode)
                    m.add_object_patch("diffusion_model.final_layer", compiled_block)
                self._compiled = True
                compile_settings = {
                    "backend": backend,
                    "mode": mode,
                    "fullgraph": fullgraph,
                    "dynamic": dynamic,
                }
                setattr(m.model, "compile_settings", compile_settings)
            except:
                raise RuntimeError("Failed to compile model")
        return (m, )
    
class TorchCompileModelWanVideo:
    def __init__(self):
        self._compiled = False

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "backend": (["inductor","cudagraphs"], {"default": "inductor"}),
                "fullgraph": ("BOOLEAN", {"default": False, "tooltip": "Enable full graph mode"}),
                "mode": (["default", "max-autotune", "max-autotune-no-cudagraphs", "reduce-overhead"], {"default": "default"}),
                "dynamic": ("BOOLEAN", {"default": False, "tooltip": "Enable dynamic mode"}),
                "dynamo_cache_size_limit": ("INT", {"default": 64, "min": 0, "max": 1024, "step": 1, "tooltip": "torch._dynamo.config.cache_size_limit"}),
                "compile_transformer_blocks_only": ("BOOLEAN", {"default": False, "tooltip": "Compile only transformer blocks"}),
            },
        }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "KJNodes/torchcompile"
    EXPERIMENTAL = True
    DEPRECATED = True

    def patch(self, model, backend, fullgraph, mode, dynamic, dynamo_cache_size_limit, compile_transformer_blocks_only):
        m = model.clone()
        diffusion_model = m.get_model_object("diffusion_model")
        torch._dynamo.config.cache_size_limit = dynamo_cache_size_limit            
        try:
            if compile_transformer_blocks_only:
                for i, block in enumerate(diffusion_model.blocks):
                    if hasattr(block, "_orig_mod"):
                        block = block._orig_mod
                    compiled_block = torch.compile(block, fullgraph=fullgraph, dynamic=dynamic, backend=backend, mode=mode)
                    m.add_object_patch(f"diffusion_model.blocks.{i}", compiled_block)
            else:
                compiled_model = torch.compile(diffusion_model, fullgraph=fullgraph, dynamic=dynamic, backend=backend, mode=mode)
                m.add_object_patch("diffusion_model", compiled_model)

            compile_settings = {
                "backend": backend,
                "mode": mode,
                "fullgraph": fullgraph,
                "dynamic": dynamic,
            }
            setattr(m.model, "compile_settings", compile_settings)
        except:
            raise RuntimeError("Failed to compile model")
        return (m, )
    
class TorchCompileModelWanVideoV2:
    def __init__(self):
        self._compiled = False

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "backend": (["inductor","cudagraphs"], {"default": "inductor"}),
                "fullgraph": ("BOOLEAN", {"default": False, "tooltip": "Enable full graph mode"}),
                "mode": (["default", "max-autotune", "max-autotune-no-cudagraphs", "reduce-overhead"], {"default": "default"}),
                "dynamic": ("BOOLEAN", {"default": False, "tooltip": "Enable dynamic mode"}),
                "compile_transformer_blocks_only": ("BOOLEAN", {"default": True, "tooltip": "Compile only transformer blocks, faster compile and less error prone"}),
                "dynamo_cache_size_limit": ("INT", {"default": 64, "min": 0, "max": 1024, "step": 1, "tooltip": "torch._dynamo.config.cache_size_limit"}),
            },
        }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "KJNodes/torchcompile"
    EXPERIMENTAL = True

    def patch(self, model, backend, fullgraph, mode, dynamic, dynamo_cache_size_limit, compile_transformer_blocks_only):
        from comfy_api.torch_helpers import set_torch_compile_wrapper
        m = model.clone()
        diffusion_model = m.get_model_object("diffusion_model")
        torch._dynamo.config.cache_size_limit = dynamo_cache_size_limit            
        try:
            if compile_transformer_blocks_only:
                compile_key_list = []
                for i, block in enumerate(diffusion_model.blocks):
                    compile_key_list.append(f"diffusion_model.blocks.{i}")
            else:
                compile_key_list =["diffusion_model"]

            set_torch_compile_wrapper(model=m, keys=compile_key_list, backend=backend, mode=mode, dynamic=dynamic, fullgraph=fullgraph)           
        except:
            raise RuntimeError("Failed to compile model")

        return (m, )

class TorchCompileVAE:
    def __init__(self):
        self._compiled_encoder = False
        self._compiled_decoder = False

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
                    "vae": ("VAE",),
                    "backend": (["inductor", "cudagraphs"],),
                    "fullgraph": ("BOOLEAN", {"default": False, "tooltip": "Enable full graph mode"}),
                    "mode": (["default", "max-autotune", "max-autotune-no-cudagraphs", "reduce-overhead"], {"default": "default"}),
                    "compile_encoder": ("BOOLEAN", {"default": True, "tooltip": "Compile encoder"}),
                    "compile_decoder": ("BOOLEAN", {"default": True, "tooltip": "Compile decoder"}),
                }}
    RETURN_TYPES = ("VAE",)
    FUNCTION = "compile"

    CATEGORY = "KJNodes/torchcompile"
    EXPERIMENTAL = True

    def compile(self, vae, backend, mode, fullgraph, compile_encoder, compile_decoder):
        if compile_encoder:
            if not self._compiled_encoder:
                encoder_name = "encoder"
                if hasattr(vae.first_stage_model, "taesd_encoder"):
                    encoder_name = "taesd_encoder"

                try:
                    setattr(
                        vae.first_stage_model,
                        encoder_name,
                        torch.compile(
                            getattr(vae.first_stage_model, encoder_name),
                            mode=mode,
                            fullgraph=fullgraph,
                            backend=backend,
                        ),
                    )
                    self._compiled_encoder = True
                except:
                    raise RuntimeError("Failed to compile model")
        if compile_decoder:
            if not self._compiled_decoder:
                decoder_name = "decoder"
                if hasattr(vae.first_stage_model, "taesd_decoder"):
                    decoder_name = "taesd_decoder"

                try:
                    setattr(
                        vae.first_stage_model,
                        decoder_name,
                        torch.compile(
                            getattr(vae.first_stage_model, decoder_name),
                            mode=mode,
                            fullgraph=fullgraph,
                            backend=backend,
                        ),
                    )
                    self._compiled_decoder = True
                except:
                    raise RuntimeError("Failed to compile model")
        return (vae, )

class TorchCompileControlNet:
    def __init__(self):
        self._compiled= False

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
                    "controlnet": ("CONTROL_NET",),
                    "backend": (["inductor", "cudagraphs"],),
                    "fullgraph": ("BOOLEAN", {"default": False, "tooltip": "Enable full graph mode"}),
                    "mode": (["default", "max-autotune", "max-autotune-no-cudagraphs", "reduce-overhead"], {"default": "default"}),
                }}
    RETURN_TYPES = ("CONTROL_NET",)
    FUNCTION = "compile"

    CATEGORY = "KJNodes/torchcompile"
    EXPERIMENTAL = True

    def compile(self, controlnet, backend, mode, fullgraph):
        if not self._compiled:
            try:
                # for i, block in enumerate(controlnet.control_model.double_blocks):
                #     print("Compiling controlnet double_block", i)
                #     controlnet.control_model.double_blocks[i] = torch.compile(block, mode=mode, fullgraph=fullgraph, backend=backend)
                controlnet.control_model = torch.compile(controlnet.control_model, mode=mode, fullgraph=fullgraph, backend=backend)
                self._compiled = True
            except:
                self._compiled = False
                raise RuntimeError("Failed to compile model")
       
        return (controlnet, )

class TorchCompileLTXModel:
    def __init__(self):
        self._compiled = False

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
                    "model": ("MODEL",),
                    "backend": (["inductor", "cudagraphs"],),
                    "fullgraph": ("BOOLEAN", {"default": False, "tooltip": "Enable full graph mode"}),
                    "mode": (["default", "max-autotune", "max-autotune-no-cudagraphs", "reduce-overhead"], {"default": "default"}),
                    "dynamic": ("BOOLEAN", {"default": False, "tooltip": "Enable dynamic mode"}),
                }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "KJNodes/torchcompile"
    EXPERIMENTAL = True

    def patch(self, model, backend, mode, fullgraph, dynamic):
        m = model.clone()
        diffusion_model = m.get_model_object("diffusion_model")
        
        if not self._compiled:
            try:
                for i, block in enumerate(diffusion_model.transformer_blocks):
                        compiled_block = torch.compile(block, mode=mode, dynamic=dynamic, fullgraph=fullgraph, backend=backend)
                        m.add_object_patch(f"diffusion_model.transformer_blocks.{i}", compiled_block)
                self._compiled = True
                compile_settings = {
                    "backend": backend,
                    "mode": mode,
                    "fullgraph": fullgraph,
                    "dynamic": dynamic,
                }
                setattr(m.model, "compile_settings", compile_settings)
               
            except:
                raise RuntimeError("Failed to compile model")           
        
        return (m, )
      
class TorchCompileCosmosModel:
    def __init__(self):
        self._compiled = False

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
                    "model": ("MODEL",),
                    "backend": (["inductor", "cudagraphs"],),
                    "fullgraph": ("BOOLEAN", {"default": False, "tooltip": "Enable full graph mode"}),
                    "mode": (["default", "max-autotune", "max-autotune-no-cudagraphs", "reduce-overhead"], {"default": "default"}),
                    "dynamic": ("BOOLEAN", {"default": False, "tooltip": "Enable dynamic mode"}),
                    "dynamo_cache_size_limit": ("INT", {"default": 64, "tooltip": "Set the dynamo cache size limit"}),
                }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "KJNodes/torchcompile"
    EXPERIMENTAL = True

    def patch(self, model, backend, mode, fullgraph, dynamic, dynamo_cache_size_limit):
        
        m = model.clone()
        diffusion_model = m.get_model_object("diffusion_model")
        torch._dynamo.config.cache_size_limit = dynamo_cache_size_limit
        
        if not self._compiled:
            try:
                for name, block in diffusion_model.blocks.items():
                    #print(f"Compiling block {name}")
                    compiled_block = torch.compile(block, mode=mode, dynamic=dynamic, fullgraph=fullgraph, backend=backend)
                    m.add_object_patch(f"diffusion_model.blocks.{name}", compiled_block)
                    #diffusion_model.blocks[name] = compiled_block

                self._compiled = True
                compile_settings = {
                    "backend": backend,
                    "mode": mode,
                    "fullgraph": fullgraph,
                    "dynamic": dynamic,
                }
                setattr(m.model, "compile_settings", compile_settings)
               
            except:
                raise RuntimeError("Failed to compile model")           
        
        return (m, )


#teacache

try:
    from comfy.ldm.wan.model import sinusoidal_embedding_1d
except:
    pass
from einops import repeat
from unittest.mock import patch
from contextlib import nullcontext
import numpy as np

def relative_l1_distance(last_tensor, current_tensor):
    l1_distance = torch.abs(last_tensor - current_tensor).mean()
    norm = torch.abs(last_tensor).mean()
    relative_l1_distance = l1_distance / norm
    return relative_l1_distance.to(torch.float32)

@torch.compiler.disable()
def tea_cache(self, x, e0, e, transformer_options):
    #teacache for cond and uncond separately
    rel_l1_thresh = transformer_options["rel_l1_thresh"]
    
    is_cond = True if transformer_options["cond_or_uncond"] == [0] else False

    should_calc = True
    suffix = "cond" if is_cond else "uncond"

    # Init cache dict if not exists
    if not hasattr(self, 'teacache_state'):
        self.teacache_state = {
            'cond': {'accumulated_rel_l1_distance': 0, 'prev_input': None, 
                    'teacache_skipped_steps': 0, 'previous_residual': None},
            'uncond': {'accumulated_rel_l1_distance': 0, 'prev_input': None,
                    'teacache_skipped_steps': 0, 'previous_residual': None}
        }
        logging.info("\nTeaCache: Initialized")

    cache = self.teacache_state[suffix]

    if cache['prev_input'] is not None:
        if transformer_options["coefficients"] == []:
            temb_relative_l1 = relative_l1_distance(cache['prev_input'], e0)
            curr_acc_dist = cache['accumulated_rel_l1_distance'] + temb_relative_l1
        else:
            rescale_func = np.poly1d(transformer_options["coefficients"])
            curr_acc_dist = cache['accumulated_rel_l1_distance'] + rescale_func(((e-cache['prev_input']).abs().mean() / cache['prev_input'].abs().mean()).cpu().item())
        try:
            if curr_acc_dist < rel_l1_thresh:
                should_calc = False
                cache['accumulated_rel_l1_distance'] = curr_acc_dist
            else:
                should_calc = True
                cache['accumulated_rel_l1_distance'] = 0
        except:
            should_calc = True
            cache['accumulated_rel_l1_distance'] = 0

    if transformer_options["coefficients"] == []:
        cache['prev_input'] = e0.clone().detach()
    else:
        cache['prev_input'] = e.clone().detach()

    if not should_calc:
        x += cache['previous_residual'].to(x.device)
        cache['teacache_skipped_steps'] += 1
        #print(f"TeaCache: Skipping {suffix} step")
    return should_calc, cache

def teacache_wanvideo_vace_forward_orig(self, x, t, context, vace_context, vace_strength, clip_fea=None, freqs=None, transformer_options={}, **kwargs):
        # embeddings
        x = self.patch_embedding(x.float()).to(x.dtype)
        grid_sizes = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)

        # time embeddings
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t).to(dtype=x[0].dtype))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))

        # context
        context = self.text_embedding(context)

        context_img_len = None
        if clip_fea is not None:
            if self.img_emb is not None:
                context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
                context = torch.concat([context_clip, context], dim=1)
            context_img_len = clip_fea.shape[-2]

        orig_shape = list(vace_context.shape)
        vace_context = vace_context.movedim(0, 1).reshape([-1] + orig_shape[2:])
        c = self.vace_patch_embedding(vace_context.float()).to(vace_context.dtype)
        c = c.flatten(2).transpose(1, 2)
        c = list(c.split(orig_shape[0], dim=0))

        if not transformer_options:
            raise RuntimeError("Can't access transformer_options, this requires ComfyUI nightly version from Mar 14, 2025 or later")

        teacache_enabled = transformer_options.get("teacache_enabled", False)
        if not teacache_enabled:
            should_calc = True
        else:
            should_calc, cache = tea_cache(self, x, e0, e, transformer_options)
        
        if should_calc:
            original_x = x.clone().detach()
            patches_replace = transformer_options.get("patches_replace", {})
            blocks_replace = patches_replace.get("dit", {})
            for i, block in enumerate(self.blocks):
                if ("double_block", i) in blocks_replace:
                    def block_wrap(args):
                        out = {}
                        out["img"] = block(args["img"], context=args["txt"], e=args["vec"], freqs=args["pe"], context_img_len=context_img_len)
                        return out
                    out = blocks_replace[("double_block", i)]({"img": x, "txt": context, "vec": e0, "pe": freqs}, {"original_block": block_wrap, "transformer_options": transformer_options})
                    x = out["img"]
                else:
                    x = block(x, e=e0, freqs=freqs, context=context, context_img_len=context_img_len)

                ii = self.vace_layers_mapping.get(i, None)
                if ii is not None:
                    for iii in range(len(c)):
                        c_skip, c[iii] = self.vace_blocks[ii](c[iii], x=original_x, e=e0, freqs=freqs, context=context, context_img_len=context_img_len)
                        x += c_skip * vace_strength[iii]
                    del c_skip

            if teacache_enabled:
                cache['previous_residual']  = (x - original_x).to(transformer_options["teacache_device"])
          
        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return x

def teacache_wanvideo_forward_orig(self, x, t, context, clip_fea=None, freqs=None, transformer_options={}, **kwargs):
        # embeddings
        x = self.patch_embedding(x.float()).to(x.dtype)
        grid_sizes = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)

        # time embeddings
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t).to(dtype=x[0].dtype))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))

        # context
        context = self.text_embedding(context)

        context_img_len = None
        if clip_fea is not None:
            if self.img_emb is not None:
                context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
                context = torch.concat([context_clip, context], dim=1)
            context_img_len = clip_fea.shape[-2]


        teacache_enabled = transformer_options.get("teacache_enabled", False)
        if not teacache_enabled:
            should_calc = True
        else:
            should_calc, cache = tea_cache(self, x, e0, e, transformer_options)
        
        if should_calc:
            original_x = x.clone().detach()
            patches_replace = transformer_options.get("patches_replace", {})
            blocks_replace = patches_replace.get("dit", {})
            for i, block in enumerate(self.blocks):
                if ("double_block", i) in blocks_replace:
                    def block_wrap(args):
                        out = {}
                        out["img"] = block(args["img"], context=args["txt"], e=args["vec"], freqs=args["pe"], context_img_len=context_img_len)
                        return out
                    out = blocks_replace[("double_block", i)]({"img": x, "txt": context, "vec": e0, "pe": freqs}, {"original_block": block_wrap, "transformer_options": transformer_options})
                    x = out["img"]
                else:
                    x = block(x, e=e0, freqs=freqs, context=context, context_img_len=context_img_len)

            if teacache_enabled:
                cache['previous_residual']  = (x - original_x).to(transformer_options["teacache_device"])
          
        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return x

class WanVideoTeaCacheKJ:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "rel_l1_thresh": ("FLOAT", {"default": 0.275, "min": 0.0, "max": 10.0, "step": 0.001, "tooltip": "Threshold for to determine when to apply the cache, compromise between speed and accuracy. When using coefficients a good value range is something between 0.2-0.4 for all but 1.3B model, which should be about 10 times smaller, same as when not using coefficients."}),
                "start_percent": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The start percentage of the steps to use with TeaCache."}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The end percentage of the steps to use with TeaCache."}),
                "cache_device": (["main_device", "offload_device"], {"default": "offload_device", "tooltip": "Device to cache to"}),
                "coefficients": (["disabled", "1.3B", "14B", "i2v_480", "i2v_720"], {"default": "i2v_480", "tooltip": "Coefficients for rescaling the relative l1 distance, if disabled the threshold value should be about 10 times smaller than the value used with coefficients."}),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch_teacache"
    CATEGORY = "KJNodes/teacache"
    DESCRIPTION = """
Patch WanVideo model to use TeaCache. Speeds up inference by caching the output and  
applying it instead of doing the step.  Best results are achieved by choosing the  
appropriate coefficients for the model. Early steps should never be skipped, with too  
aggressive values this can happen and the motion suffers. Starting later can help with that too.   
When NOT using coefficients, the threshold value should be  
about 10 times smaller than the value used with coefficients.  

Official recommended values https://github.com/ali-vilab/TeaCache/tree/main/TeaCache4Wan2.1:


<pre style='font-family:monospace'>
+-------------------+--------+---------+--------+
|       Model       |  Low   | Medium  |  High  |
+-------------------+--------+---------+--------+
| Wan2.1 t2v 1.3B  |  0.05  |  0.07   |  0.08  |
| Wan2.1 t2v 14B   |  0.14  |  0.15   |  0.20  |
| Wan2.1 i2v 480P  |  0.13  |  0.19   |  0.26  |
| Wan2.1 i2v 720P  |  0.18  |  0.20   |  0.30  |
+-------------------+--------+---------+--------+
</pre> 
"""
    EXPERIMENTAL = True

    def patch_teacache(self, model, rel_l1_thresh, start_percent, end_percent, cache_device, coefficients):
        if rel_l1_thresh == 0:
            return (model,)

        if coefficients == "disabled" and rel_l1_thresh > 0.1:
            logging.warning("Threshold value is too high for TeaCache without coefficients, consider using coefficients for better results.")
        if coefficients != "disabled" and rel_l1_thresh < 0.1 and "1.3B" not in coefficients:
            logging.warning("Threshold value is too low for TeaCache with coefficients, consider using higher threshold value for better results.")
        
        # type_str = str(type(model.model.model_config).__name__)
        #if model.model.diffusion_model.dim == 1536:
        #    model_type ="1.3B"
        # else:
        #     if "WAN21_T2V" in type_str:
        #         model_type = "14B"
        #     elif "WAN21_I2V" in type_str:
        #         model_type = "i2v_480"
        #     else:
        #         model_type = "i2v_720" #how to detect this?
  
       
        teacache_coefficients_map = {
            "disabled": [],
            "1.3B": [2.39676752e+03, -1.31110545e+03, 2.01331979e+02, -8.29855975e+00, 1.37887774e-01],
            "14B": [-5784.54975374, 5449.50911966, -1811.16591783, 256.27178429, -13.02252404],
            "i2v_480": [-3.02331670e+02, 2.23948934e+02, -5.25463970e+01, 5.87348440e+00, -2.01973289e-01],
            "i2v_720": [-114.36346466, 65.26524496, -18.82220707, 4.91518089, -0.23412683],
        }
        coefficients = teacache_coefficients_map[coefficients]
        
        teacache_device = mm.get_torch_device() if cache_device == "main_device" else mm.unet_offload_device()

        model_clone = model.clone()
        if 'transformer_options' not in model_clone.model_options:
            model_clone.model_options['transformer_options'] = {}
        model_clone.model_options["transformer_options"]["rel_l1_thresh"] = rel_l1_thresh
        model_clone.model_options["transformer_options"]["teacache_device"] = teacache_device
        model_clone.model_options["transformer_options"]["coefficients"] = coefficients
        diffusion_model = model_clone.get_model_object("diffusion_model")
                
        def outer_wrapper(start_percent, end_percent):        
            def unet_wrapper_function(model_function, kwargs):
                input = kwargs["input"]
                timestep = kwargs["timestep"]
                c = kwargs["c"]
                sigmas = c["transformer_options"]["sample_sigmas"]
                cond_or_uncond = kwargs["cond_or_uncond"]
                last_step = (len(sigmas) - 1)
             
                matched_step_index = (sigmas == timestep[0] ).nonzero()
                if len(matched_step_index) > 0:
                    current_step_index = matched_step_index.item()
                else:
                    for i in range(len(sigmas) - 1):
                        # walk from beginning of steps until crossing the timestep
                        if (sigmas[i] - timestep[0]) * (sigmas[i + 1] - timestep[0]) <= 0:
                            current_step_index = i
                            break
                    else:
                        current_step_index = 0

                if current_step_index == 0:
                    if (len(cond_or_uncond) == 1 and cond_or_uncond[0] == 1) or len(cond_or_uncond) == 2:
                        if hasattr(diffusion_model, "teacache_state"):
                            delattr(diffusion_model, "teacache_state")
                            logging.info("\nResetting TeaCache state")
                
                current_percent = current_step_index / (len(sigmas) - 1)
                c["transformer_options"]["current_percent"] = current_percent
                if start_percent <= current_percent <= end_percent:
                    c["transformer_options"]["teacache_enabled"] = True
                
                forward_function = teacache_wanvideo_vace_forward_orig if hasattr(diffusion_model, "vace_layers") else teacache_wanvideo_forward_orig
                context = patch.multiple(
                    diffusion_model, 
                    forward_orig=forward_function.__get__(diffusion_model, diffusion_model.__class__)
                )

                with context:
                    out = model_function(input, timestep, **c)
                    if current_step_index+1 == last_step and hasattr(diffusion_model, "teacache_state"):
                        if len(cond_or_uncond) == 1 and cond_or_uncond[0] == 0:
                            skipped_steps_cond = diffusion_model.teacache_state["cond"]["teacache_skipped_steps"]
                            skipped_steps_uncond = diffusion_model.teacache_state["uncond"]["teacache_skipped_steps"]
                            logging.info("-----------------------------------")
                            logging.info(f"TeaCache skipped:")
                            logging.info(f"{skipped_steps_cond} cond steps")
                            logging.info(f"{skipped_steps_uncond} uncond step")
                            logging.info(f"out of {last_step} steps")
                            logging.info("-----------------------------------")
                        elif len(cond_or_uncond) == 2:
                            skipped_steps_cond = diffusion_model.teacache_state["uncond"]["teacache_skipped_steps"]
                            logging.info("-----------------------------------")
                            logging.info(f"TeaCache skipped:")
                            logging.info(f"{skipped_steps_cond} cond steps")
                            logging.info(f"out of {last_step} steps")
                            logging.info("-----------------------------------")
                        
                    return out
            return unet_wrapper_function

        model_clone.set_model_unet_function_wrapper(outer_wrapper(start_percent=start_percent, end_percent=end_percent))

        return (model_clone,)




from comfy.ldm.flux.math import apply_rope

def modified_wan_self_attention_forward(self, x, freqs):
    r"""
    Args:
        x(Tensor): Shape [B, L, num_heads, C / num_heads]
        freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
    """
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

    # query, key, value function
    def qkv_fn(x):
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n * d)
        return q, k, v

    q, k, v = qkv_fn(x)

    q, k = apply_rope(q, k, freqs)

    feta_scores = get_feta_scores(q, k, self.num_frames, self.enhance_weight)

    x = comfy.ldm.modules.attention.optimized_attention(
        q.view(b, s, n * d),
        k.view(b, s, n * d),
        v,
        heads=self.num_heads,
    )

    x = self.o(x)

    x *= feta_scores

    return x
    
from einops import rearrange
def get_feta_scores(query, key, num_frames, enhance_weight):
    img_q, img_k = query, key #torch.Size([2, 9216, 12, 128])
    
    _, ST, num_heads, head_dim = img_q.shape
    spatial_dim = ST / num_frames
    spatial_dim = int(spatial_dim)

    query_image = rearrange(
        img_q, "B (T S) N C -> (B S) N T C", T=num_frames, S=spatial_dim, N=num_heads, C=head_dim
    )
    key_image = rearrange(
        img_k, "B (T S) N C -> (B S) N T C", T=num_frames, S=spatial_dim, N=num_heads, C=head_dim
    )

    return feta_score(query_image, key_image, head_dim, num_frames, enhance_weight)

def feta_score(query_image, key_image, head_dim, num_frames, enhance_weight):
    scale = head_dim**-0.5
    query_image = query_image * scale
    attn_temp = query_image @ key_image.transpose(-2, -1)  # translate attn to float32
    attn_temp = attn_temp.to(torch.float32)
    attn_temp = attn_temp.softmax(dim=-1)

    # Reshape to [batch_size * num_tokens, num_frames, num_frames]
    attn_temp = attn_temp.reshape(-1, num_frames, num_frames)

    # Create a mask for diagonal elements
    diag_mask = torch.eye(num_frames, device=attn_temp.device).bool()
    diag_mask = diag_mask.unsqueeze(0).expand(attn_temp.shape[0], -1, -1)

    # Zero out diagonal elements
    attn_wo_diag = attn_temp.masked_fill(diag_mask, 0)

    # Calculate mean for each token's attention matrix
    # Number of off-diagonal elements per matrix is n*n - n
    num_off_diag = num_frames * num_frames - num_frames
    mean_scores = attn_wo_diag.sum(dim=(1, 2)) / num_off_diag

    enhance_scores = mean_scores.mean() * (num_frames + enhance_weight)
    enhance_scores = enhance_scores.clamp(min=1)
    return enhance_scores

import types
class WanAttentionPatch:
    def __init__(self, num_frames, weight):
        self.num_frames = num_frames
        self.enhance_weight = weight
        
    def __get__(self, obj, objtype=None):
        # Create bound method with stored parameters
        def wrapped_attention(self_module, *args, **kwargs):
            self_module.num_frames = self.num_frames
            self_module.enhance_weight = self.enhance_weight
            return modified_wan_self_attention_forward(self_module, *args, **kwargs)
        return types.MethodType(wrapped_attention, obj)

class WanVideoEnhanceAVideoKJ:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "latent": ("LATENT", {"tooltip": "Only used to get the latent count"}),
                "weight": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 10.0, "step": 0.001, "tooltip": "Strength of the enhance effect"}),
           }
        }
    
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "enhance"
    CATEGORY = "KJNodes/experimental"
    DESCRIPTION = "https://github.com/NUS-HPC-AI-Lab/Enhance-A-Video"
    EXPERIMENTAL = True

    def enhance(self, model, weight, latent):
        if weight == 0:
            return (model,)
        
        num_frames = latent["samples"].shape[2]

        model_clone = model.clone()
        if 'transformer_options' not in model_clone.model_options:
            model_clone.model_options['transformer_options'] = {}
        model_clone.model_options["transformer_options"]["enhance_weight"] = weight
        diffusion_model = model_clone.get_model_object("diffusion_model")

        compile_settings = getattr(model.model, "compile_settings", None)
        for idx, block in enumerate(diffusion_model.blocks):
            patched_attn = WanAttentionPatch(num_frames, weight).__get__(block.self_attn, block.__class__)
            if compile_settings is not None:
                patched_attn = torch.compile(patched_attn, mode=compile_settings["mode"], dynamic=compile_settings["dynamic"], fullgraph=compile_settings["fullgraph"], backend=compile_settings["backend"])
            
            model_clone.add_object_patch(f"diffusion_model.blocks.{idx}.self_attn.forward", patched_attn)
            
        return (model_clone,)
    
def normalized_attention_guidance(self, query, context_positive, context_negative):
    nag_scale = self.nag_scale
    nag_alpha = self.nag_alpha
    nag_tau = self.nag_tau

    k_positive = self.norm_k(self.k(context_positive))
    v_positive = self.v(context_positive)
    k_negative = self.norm_k(self.k(context_negative))
    v_negative = self.v(context_negative)

    x_positive = comfy.ldm.modules.attention.optimized_attention(query, k_positive, v_positive, heads=self.num_heads)
    x_positive = x_positive.flatten(2)

    x_negative = comfy.ldm.modules.attention.optimized_attention(query, k_negative, v_negative, heads=self.num_heads)
    x_negative = x_negative.flatten(2)

    nag_guidance = x_positive * nag_scale - x_negative * (nag_scale - 1)
    
    norm_positive = torch.norm(x_positive, p=1, dim=-1, keepdim=True).expand_as(x_positive)
    norm_guidance = torch.norm(nag_guidance, p=1, dim=-1, keepdim=True).expand_as(nag_guidance)
    
    scale = norm_guidance / norm_positive
    scale = torch.nan_to_num(scale, nan=10.0)
    
    mask = scale > nag_tau
    adjustment = (norm_positive * nag_tau) / (norm_guidance + 1e-7)
    nag_guidance = torch.where(mask, nag_guidance * adjustment, nag_guidance)
    
    x = nag_guidance * nag_alpha + x_positive * (1 - nag_alpha)
    del nag_guidance

    return x

#region NAG
def wan_crossattn_forward_nag(self, x, context, **kwargs):
    r"""
    Args:
        x(Tensor): Shape [B, L1, C]
        context(Tensor): Shape [B, L2, C]
    """
 
    if context.shape[0] == 2:
        x, x_real_negative = torch.chunk(x, 2, dim=0)
        context_positive, context_negative = torch.chunk(context, 2, dim=0)
    else:
        context_positive = context
        context_negative = None

    q = self.norm_q(self.q(x))

    x = normalized_attention_guidance(self, q, context_positive, self.nag_context)

    if context_negative is not None:
        q_real_negative = self.norm_q(self.q(x_real_negative))
        k_real_negative = self.norm_k(self.k(context_negative))
        v_real_negative = self.v(context_negative)
        x_real_negative = comfy.ldm.modules.attention.optimized_attention(q_real_negative, k_real_negative, v_real_negative, heads=self.num_heads)
        x = torch.cat([x, x_real_negative], dim=0)

    x = self.o(x)
    return x


def wan_i2v_crossattn_forward_nag(self, x, context, context_img_len):
    r"""
    Args:
        x(Tensor): Shape [B, L1, C]
        context(Tensor): Shape [B, L2, C]
    """
    context_img = context[:, :context_img_len]
    context = context[:, context_img_len:]

    q_img = self.norm_q(self.q(x))    
    k_img = self.norm_k_img(self.k_img(context_img))
    v_img = self.v_img(context_img)
    img_x = comfy.ldm.modules.attention.optimized_attention(q_img, k_img, v_img, heads=self.num_heads)

    if context.shape[0] == 2:
        x, x_real_negative = torch.chunk(x, 2, dim=0)
        context_positive, context_negative = torch.chunk(context, 2, dim=0)
    else:
        context_positive = context
        context_negative = None
    
    q = self.norm_q(self.q(x))

    x = normalized_attention_guidance(self, q, context_positive, self.nag_context)

    if context_negative is not None:
        q_real_negative = self.norm_q(self.q(x_real_negative))
        k_real_negative = self.norm_k(self.k(context_negative))
        v_real_negative = self.v(context_negative)
        x_real_negative = comfy.ldm.modules.attention.optimized_attention(q_real_negative, k_real_negative, v_real_negative, heads=self.num_heads)
        x = torch.cat([x, x_real_negative], dim=0)

    # output
    x = x + img_x
    x = self.o(x)
    return x

class WanCrossAttentionPatch:
    def __init__(self, context, nag_scale, nag_alpha, nag_tau, i2v=False):
        self.nag_context = context
        self.nag_scale = nag_scale
        self.nag_alpha = nag_alpha
        self.nag_tau = nag_tau
        self.i2v = i2v
    def __get__(self, obj, objtype=None):
        # Create bound method with stored parameters
        def wrapped_attention(self_module, *args, **kwargs):
            self_module.nag_context = self.nag_context
            self_module.nag_scale = self.nag_scale
            self_module.nag_alpha = self.nag_alpha
            self_module.nag_tau = self.nag_tau
            if self.i2v:
                return wan_i2v_crossattn_forward_nag(self_module, *args, **kwargs)
            else:
                return wan_crossattn_forward_nag(self_module, *args, **kwargs)
        return types.MethodType(wrapped_attention, obj)
    
class WanVideoNAG:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "conditioning": ("CONDITIONING",),
                "nag_scale": ("FLOAT", {"default": 11.0, "min": 0.0, "max": 100.0, "step": 0.001, "tooltip": "Strength of negative guidance effect"}),
                "nag_alpha": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.001, "tooltip": "Mixing coefficient in that controls the balance between the normalized guided representation and the original positive representation."}),
                "nag_tau": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 10.0, "step": 0.001, "tooltip": "Clipping threshold that controls how much the guided attention can deviate from the positive attention."}),
           }
        }
    
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = "KJNodes/experimental"
    DESCRIPTION = "https://github.com/ChenDarYen/Normalized-Attention-Guidance"
    EXPERIMENTAL = True

    def patch(self, model, conditioning, nag_scale, nag_alpha, nag_tau):
        if nag_scale == 0:
            return (model,)
        
        device = mm.get_torch_device()
        dtype = mm.unet_dtype()

        model_clone = model.clone()

        diffusion_model = model_clone.get_model_object("diffusion_model")

        diffusion_model.text_embedding.to(device)
        context = diffusion_model.text_embedding(conditioning[0][0].to(device, dtype))

        type_str = str(type(model.model.model_config).__name__)
        i2v = True if "WAN21_I2V" in type_str else False
    
        for idx, block in enumerate(diffusion_model.blocks):
            patched_attn = WanCrossAttentionPatch(context, nag_scale, nag_alpha, nag_tau, i2v).__get__(block.cross_attn, block.__class__)
          
            model_clone.add_object_patch(f"diffusion_model.blocks.{idx}.cross_attn.forward", patched_attn)
            
        return (model_clone,)
    
class SkipLayerGuidanceWanVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": ("MODEL", ),
                             "blocks": ("STRING", {"default": "10", "multiline": False}),
                             "start_percent": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.001}),
                             "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                            }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "slg"
    EXPERIMENTAL = True
    DESCRIPTION = "Simplified skip layer guidance that only skips the uncond on selected blocks"

    CATEGORY = "advanced/guidance"

    def slg(self, model, start_percent, end_percent, blocks):
        def skip(args, extra_args):
            transformer_options = extra_args.get("transformer_options", {})
            original_block = extra_args["original_block"]

            if not transformer_options:
                raise ValueError("transformer_options not found in extra_args, currently SkipLayerGuidanceWanVideo only works with TeaCacheKJ")
            if start_percent <= transformer_options["current_percent"] <= end_percent:
                if args["img"].shape[0] == 2:
                    prev_img_uncond = args["img"][0].unsqueeze(0)

                    new_args = {
                        "img": args["img"][1].unsqueeze(0),
                        "txt": args["txt"][1].unsqueeze(0),
                        "vec": args["vec"][1].unsqueeze(0),
                        "pe": args["pe"][1].unsqueeze(0)
                    }
                    
                    block_out = original_block(new_args)

                    out = {
                        "img": torch.cat([prev_img_uncond, block_out["img"]], dim=0),
                        "txt": args["txt"],
                        "vec": args["vec"],
                        "pe": args["pe"]
                    }
                else:
                    if transformer_options.get("cond_or_uncond") == [0]:
                        out = original_block(args)
                    else:
                        out = args
            else:
                out = original_block(args)
            return out

        block_list = [int(x.strip()) for x in blocks.split(",")]
        blocks = [int(i) for i in block_list]
        logging.info(f"Selected blocks to skip uncond on: {blocks}")

        m = model.clone()

        for b in blocks:
            #m.set_model_patch_replace(skip, "dit", "double_block", b)
            model_options = m.model_options["transformer_options"].copy()
            if "patches_replace" not in model_options:
                model_options["patches_replace"] = {}
            else:
                model_options["patches_replace"] = model_options["patches_replace"].copy()

            if "dit" not in model_options["patches_replace"]:
                model_options["patches_replace"]["dit"] = {}
            else:
                model_options["patches_replace"]["dit"] = model_options["patches_replace"]["dit"].copy()

            block = ("double_block", b)

            model_options["patches_replace"]["dit"][block] = skip
            m.model_options["transformer_options"] = model_options
            

        return (m, )

class CFGZeroStarAndInit:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
                    "model": ("MODEL",),
                    "use_zero_init": ("BOOLEAN", {"default": True}),
                    "zero_init_steps": ("INT", {"default": 0, "min": 0, "tooltip": "for zero init, starts from 0 so first step is always zeroed out if use_zero_init enabled"}),
                }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    DESCRIPTION = "https://github.com/WeichenFan/CFG-Zero-star"
    CATEGORY = "KJNodes/experimental"
    EXPERIMENTAL = True

    def patch(self, model, use_zero_init, zero_init_steps):
        def cfg_zerostar(args):
            #zero init
            cond = args["cond"]
            timestep = args["timestep"]
            sigmas = args["model_options"]["transformer_options"]["sample_sigmas"]
            matched_step_index = (sigmas == timestep[0]).nonzero()
            if len(matched_step_index) > 0:
                current_step_index = matched_step_index.item()
            else:
                for i in range(len(sigmas) - 1):
                    if (sigmas[i] - timestep[0]) * (sigmas[i + 1] - timestep[0]) <= 0:
                        current_step_index = i
                        break
                else:
                    current_step_index = 0

            if (current_step_index <= zero_init_steps) and use_zero_init:
                return cond * 0
                        
            uncond = args["uncond"]
            cond_scale = args["cond_scale"]
                
            batch_size = cond.shape[0]

            positive_flat = cond.view(batch_size, -1)
            negative_flat = uncond.view(batch_size, -1)

            dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)
            squared_norm = torch.sum(negative_flat ** 2, dim=1, keepdim=True) + 1e-8
            alpha = dot_product / squared_norm
            alpha = alpha.view(batch_size, *([1] * (len(cond.shape) - 1)))

            noise_pred = uncond * alpha + cond_scale * (cond - uncond * alpha)
            return noise_pred

        m = model.clone()
        m.set_model_sampler_cfg_function(cfg_zerostar)
        return (m, )