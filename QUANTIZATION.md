# The Comfy guide to Quantization


## How does quantization work?

Quantization aims to map a high-precision value x_f to a lower precision format with minimal loss in accuracy. These smaller formats then serve to reduce the models memory footprint and increase throughput by using specialized hardware.

When simply converting a value from FP16 to FP8 using the round-nearest method we might hit two issues:
- The dynamic range of FP16 (-65,504, 65,504) far exceeds FP8 formats like E4M3 (-448, 448) or E5M2 (-57,344, 57,344), potentially resulting in clipped values
- The original values are concentrated in a small range (e.g. -1,1) leaving many FP8-bits "unused"

By using a scaling factor, we aim to map these values into the quantized-dtype range, making use of the full spectrum. One of the easiest approaches, and common, is using per-tensor absolute-maximum scaling.

```
absmax = max(abs(tensor))
scale = amax / max_dynamic_range_low_precision

# Quantization
tensor_q = (tensor / scale).to(low_precision_dtype)

# De-Quantization
tensor_dq = tensor_q.to(fp16) * scale

tensor_dq ~ tensor
```

Given that additional information (scaling factor) is needed to "interpret" the quantized values, we describe those as derived datatypes.


## Quantization in Comfy

```
QuantizedTensor (torch.Tensor subclass)
  ↓ __torch_dispatch__
Two-Level Registry (generic + layout handlers)
  ↓
MixedPrecisionOps + Metadata Detection
```

### Representation

To represent these derived datatypes, ComfyUI uses a subclass of torch.Tensor to implements these using the `QuantizedTensor` class found in `comfy/quant_ops.py`

A `Layout` class defines how a specific quantization format behaves:
- Required parameters
- Quantize method
- De-Quantize method

```python
from comfy.quant_ops import QuantizedLayout

class MyLayout(QuantizedLayout):
    @classmethod
    def quantize(cls, tensor, **kwargs):
        # Convert to quantized format
        qdata = ...
        params = {'scale': ..., 'orig_dtype': tensor.dtype}
        return qdata, params
    
    @staticmethod
    def dequantize(qdata, scale, orig_dtype, **kwargs):
        return qdata.to(orig_dtype) * scale
```

To then run operations using these QuantizedTensors we use two registry systems to define supported operations. 
The first is a **generic registry** that handles operations common to all quantized formats (e.g., `.to()`, `.clone()`, `.reshape()`).

The second registry is layout-specific and allows to implement fast-paths like nn.Linear.
```python
from comfy.quant_ops import register_layout_op

@register_layout_op(torch.ops.aten.linear.default, MyLayout)
def my_linear(func, args, kwargs):
    # Extract tensors, call optimized kernel
    ...
```
When `torch.nn.functional.linear()` is called with QuantizedTensor arguments, `__torch_dispatch__` automatically routes to the registered implementation.
For any unsupported operation, QuantizedTensor will fallback to call `dequantize` and dispatch using the high-precision implementation.


### Mixed Precision

The `MixedPrecisionOps` class (lines 542-648 in `comfy/ops.py`) enables per-layer quantization decisions, allowing different layers in a model to use different precisions. This is activated when a model config contains a `layer_quant_config` dictionary that specifies which layers should be quantized and how.

**Architecture:**

```python
class MixedPrecisionOps(disable_weight_init):
    _layer_quant_config = {}  # Maps layer names to quantization configs
    _compute_dtype = torch.bfloat16  # Default compute / dequantize precision
```

**Key mechanism:**

The custom `Linear._load_from_state_dict()` method inspects each layer during model loading:
- If the layer name is **not** in `_layer_quant_config`: load weight as regular tensor in `_compute_dtype`
- If the layer name **is** in `_layer_quant_config`: 
  - Load weight as `QuantizedTensor` with the specified layout (e.g., `TensorCoreFP8Layout`)
  - Load associated quantization parameters (scales, block_size, etc.)

**Why it's needed:**

Not all layers tolerate quantization equally. Sensitive operations like final projections can be kept in higher precision, while compute-heavy matmuls are quantized. This provides most of the performance benefits while maintaining quality.

The system is selected in `pick_operations()` when `model_config.layer_quant_config` is present, making it the highest-priority operation mode.


## Checkpoint Format

Quantized checkpoints are stored as standard safetensors files with quantized weight tensors and associated scaling parameters, plus a `_quantization_metadata` JSON entry describing the quantization scheme.

The quantized checkpoint will contain the same layers as the original checkpoint but:
- The weights are stored as quantized values, sometimes using a different storage datatype. E.g. uint8 container for fp8.
- For each quantized weight a number of additional scaling parameters are stored alongside depending on the recipe.
- We store a metadata.json in the metadata of the final safetensor containing the `_quantization_metadata` describing which layers are quantized and what layout has been used.

### Scaling Parameters details
We define 4 possible scaling parameters that should cover most recipes in the near-future:
- **weight_scale**: quantization scalers for the weights
- **weight_scale_2**: global scalers in the context of double scaling
- **pre_quant_scale**: scalers used for smoothing salient weights
- **input_scale**: quantization scalers for the activations

| Format | Storage dtype | weight_scale | weight_scale_2 | pre_quant_scale | input_scale |
|--------|---------------|--------------|----------------|-----------------|-------------|
| float8_e4m3fn | float32 | float32 (scalar) | - | - | float32 (scalar) |
| svdquant_int4 | int8 (packed 4-bit) | - | - | - | - |
| svdquant_nvfp4 | int8 (packed 4-bit) | - | - | - | - |
| awq_int4 | int32 (packed 4-bit) | - | - | - | - |

For SVDQuant formats, additional parameters are stored:
- **wscales**: Weight quantization scales (shape: in_features // group_size, out_features)
- **smooth_factor**: Smoothing factors for inputs (shape: in_features)
- **smooth_factor_orig**: Original smoothing factors (shape: in_features)
- **proj_down**: Low-rank down projection (shape: in_features, rank)
- **proj_up**: Low-rank up projection (shape: out_features, rank)
- **wtscale**: Global weight scale (nvfp4 only, scalar float)
- **wcscales**: Channel-wise weight scales (nvfp4 only, shape: out_features)

For AWQ format, the following parameters are stored:
- **wscales**: Weight quantization scales (shape: in_features // group_size, out_features)
- **wzeros**: Weight zero points (shape: in_features // group_size, out_features)

You can find the defined formats in `comfy/quant_ops.py` (QUANT_ALGOS).

### Quantization Metadata

The metadata stored alongside the checkpoint contains:
- **format_version**: String to define a version of the standard
- **layers**: A dictionary mapping layer names to their quantization format. The format string maps to the definitions found in `QUANT_ALGOS`. 

Example:
```json
{
  "_quantization_metadata": {
    "format_version": "1.0",
    "layers": {
      "model.layers.0.mlp.up_proj": {"format": "float8_e4m3fn"},
      "model.layers.0.mlp.down_proj": {"format": "float8_e4m3fn"},
      "model.layers.1.mlp.up_proj": {"format": "float8_e4m3fn"}
    }
  }
}
```


## Creating Quantized Checkpoints

To create compatible checkpoints, use any quantization tool provided the output follows the checkpoint format described above and uses a layout defined in `QUANT_ALGOS`.

### Weight Quantization

Weight quantization is straightforward - compute the scaling factor directly from the weight tensor using the absolute maximum method described earlier. Each layer's weights are quantized independently and stored with their corresponding `weight_scale` parameter.

### Calibration (for Activation Quantization)

Activation quantization (e.g., for FP8 Tensor Core operations) requires `input_scale` parameters that cannot be determined from static weights alone. Since activation values depend on actual inputs, we use **post-training calibration (PTQ)**:

1. **Collect statistics**: Run inference on N representative samples
2. **Track activations**: Record the absolute maximum (`amax`) of inputs to each quantized layer
3. **Compute scales**: Derive `input_scale` from collected statistics
4. **Store in checkpoint**: Save `input_scale` parameters alongside weights

The calibration dataset should be representative of your target use case. For diffusion models, this typically means a diverse set of prompts and generation parameters.


## SVDQuant

SVDQuant is an advanced 4-bit quantization scheme that decomposes linear operations using low-rank factorization combined with residual quantization:

```
X*W = X * proj_down * proj_up + quantize(X) * quantize(R)
```

Where:
- `proj_down`, `proj_up`: Low-rank factorization matrices of the original weights
- `R`: Residual weights (quantized to 4-bit)
- `quantize()`: 4-bit quantization with smoothing factors

### Key Features

1. **Asymmetric Quantization**: Unlike FP8 where both weights and activations are quantized offline or use the same quantization scheme, SVDQuant:
   - Quantizes weights offline with multiple parameters stored in the checkpoint
   - Quantizes activations on-the-fly during forward pass using smoothing factors

2. **Two Precision Modes**:
   - `svdquant_int4`: 4-bit integer quantization with group_size=64
   - `svdquant_nvfp4`: 4-bit floating-point (NVIDIA FP4) with group_size=16, includes additional channel-wise scales

3. **Low-Rank Optimization**: Separates the easy-to-approximate low-rank component from the hard-to-quantize residual, improving accuracy.

### Implementation

SVDQuant requires the `nunchaku` library for optimized CUDA kernels:
```bash
pip install nunchaku
```

The implementation uses two main operations:
- `svdq_quantize_w4a4_act_fuse_lora_cuda`: Quantizes activations and computes low-rank hidden states
- `svdq_gemm_w4a4_cuda`: Performs the quantized GEMM with low-rank residual addition

### Checkpoint Format

SVDQuant checkpoints contain the standard weight tensor (packed 4-bit residuals in int8) plus additional parameters per quantized layer:

```python
{
  "layer_name.weight": tensor,  # Packed 4-bit residual weights (out_features, in_features // 2)
  "layer_name.wscales": tensor,  # Weight scales (in_features // group_size, out_features)
  "layer_name.smooth_factor": tensor,  # Smoothing factors (in_features,)
  "layer_name.smooth_factor_orig": tensor,  # Original smoothing factors (in_features,)
  "layer_name.proj_down": tensor,  # Low-rank down projection (in_features, rank)
  "layer_name.proj_up": tensor,  # Low-rank up projection (out_features, rank)
  
  # For nvfp4 only:
  "layer_name.wtscale": float,  # Global weight scale
  "layer_name.wcscales": tensor,  # Channel-wise scales (out_features,)
}
```

The quantization metadata specifies which layers use SVDQuant:

```json
{
  "_quantization_metadata": {
    "format_version": "1.0",
    "layers": {
      "model.layers.0.mlp.up_proj": {"format": "svdquant_int4"},
      "model.layers.0.mlp.down_proj": {"format": "svdquant_int4"}
    }
  }
}
```

## AWQ

AWQ (Activation-aware Weight Quantization) is a 4-bit weight quantization scheme that keeps activations in 16-bit precision (W4A16):

```
Y = X @ W_quantized
```

Where:
- `X`: 16-bit activations (float16/bfloat16)
- `W_quantized`: 4-bit quantized weights with per-group scales and zero points

### Key Features

1. **W4A16 Quantization**: 
   - Quantizes weights to 4-bit while keeping activations in 16-bit
   - Uses per-group quantization with configurable group size (typically 64)
   - Stores zero points for asymmetric quantization

2. **Activation-Aware**: 
   - Quantization is calibrated based on activation statistics
   - Protects salient weights that are important for accuracy

3. **Hardware Efficient**:
   - Optimized for GPU inference
   - Significantly reduces memory footprint
   - Increases throughput with specialized kernels

### Implementation

AWQ requires the `nunchaku` library for optimized CUDA kernels:
```bash
pip install nunchaku
```

The implementation uses the `awq_gemv_w4a16_cuda` kernel for efficient W4A16 matrix multiplication.

### Checkpoint Format

AWQ checkpoints contain the standard weight tensor (packed 4-bit weights in int32) plus additional parameters per quantized layer:

```python
{
  "layer_name.weight": tensor,  # Packed 4-bit weights (out_features // 4, in_features // 2)
  "layer_name.wscales": tensor,  # Weight scales (in_features // group_size, out_features)
  "layer_name.wzeros": tensor,   # Zero points (in_features // group_size, out_features)
}
```

The quantization metadata specifies which layers use AWQ:

```json
{
  "_quantization_metadata": {
    "format_version": "1.0",
    "layers": {
      "model.layers.0.mlp.up_proj": {"format": "awq_int4"},
      "model.layers.0.mlp.down_proj": {"format": "awq_int4"}
    }
  }
}
```