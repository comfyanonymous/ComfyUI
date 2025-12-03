# Inference Scaling Implementation

This document describes the inference scaling feature implementation for ComfyUI.

## Overview

The inference scaling system monitors image quality during the generation loop (not just at the end) and can adjust sampling parameters based on quality metrics. It consists of two main nodes:

1. **VerifierSelectionNode** - Creates and configures an image quality verifier
2. **InferenceScalingNode** - Wraps the standard KSampler with quality monitoring

## Architecture

### VerifierSelectionNode

This node allows users to select and configure a quality verifier:

- **Inputs:**
  - `verifier_type`: Type of verifier ("simple" or "custom")
  - `quality_threshold`: Minimum quality score threshold (0.0-1.0)

- **Outputs:**
  - `verifier_id`: String identifier for the verifier (used by InferenceScalingNode)

### InferenceScalingNode

This node wraps the standard sampling process with quality monitoring:

- **Inputs:**
  - `verifier_id`: Verifier identifier from VerifierSelectionNode
  - `model`: The diffusion model
  - `positive`: Positive conditioning
  - `negative`: Negative conditioning
  - `latent_image`: Initial latent image
  - `vae`: VAE model for decoding latents during quality checks
  - `seed`: Random seed
  - `steps`: Number of sampling steps
  - `cfg`: CFG scale
  - `sampler_name`: Sampler algorithm name
  - `scheduler`: Scheduler name
  - `check_interval`: Check quality every N steps (default: 5)
  - `quality_threshold`: Quality threshold for adjustments
  - `scale_factor`: Factor to scale parameters when quality is low

- **Outputs:**
  - `latent`: The denoised latent image

## How It Works

1. **During Sampling:**
   - The node creates a callback function that is called at each sampling step
   - At intervals specified by `check_interval`, the callback:
     - Decodes the current latent representation to an image using VAE
     - Uses the verifier to assess image quality
     - Logs quality metrics
     - Can adjust parameters based on quality (currently limited - see Limitations)

2. **Quality Verification:**
   - The `SimpleVerifier` class uses:
     - Image variance (higher = more detail)
     - Edge detection strength (measures structure)
     - Combined into a quality score (0.0-1.0)

3. **Scaling Logic:**
   - When quality is below threshold: Can increase CFG (though this is limited - see Limitations)
   - Quality history is tracked for analysis

## Limitations

### CFG Adjustment

**Important:** Currently, CFG cannot be dynamically adjusted during sampling because:
- CFG is set at the beginning of sampling and used throughout
- The sampling process doesn't support mid-run parameter changes

**Workarounds:**
- Quality checking still works and provides valuable feedback
- Quality history can be used to inform future runs
- Early stopping could be implemented (not yet done)

### Future Improvements

1. **Dynamic Step Adjustment:** Implement adaptive step count based on quality
2. **Early Stopping:** Stop sampling early if quality is consistently good
3. **Better Verifiers:** Add more sophisticated quality metrics (e.g., perceptual metrics, CLIP-based scoring)
4. **Parameter Tuning:** Implement more sophisticated parameter adjustment strategies

## Usage Example

```
1. Create VerifierSelectionNode
   - Set verifier_type: "simple"
   - Set quality_threshold: 0.5
   - Connect output to InferenceScalingNode's verifier_id input

2. Create InferenceScalingNode
   - Connect model, positive, negative, latent_image, vae
   - Set steps: 20
   - Set cfg: 7.0
   - Set check_interval: 5 (check every 5 steps)
   - Set quality_threshold: 0.5
   - Set scale_factor: 1.2
   - Connect verifier_id from VerifierSelectionNode
```

## Files

- `comfy_api_nodes/nodes_inference_scaling.py` - Main implementation
- `INFERENCE_SCALING_README.md` - This file

## Testing

To test the implementation:

1. Start ComfyUI
2. Create a workflow with:
   - Model loading
   - Text encoding (positive/negative)
   - Empty latent image
   - VerifierSelectionNode
   - InferenceScalingNode
   - VAE decode (to see final result)
3. Run the workflow and observe quality checks in logs

## Notes

- VAE decoding during sampling adds computational overhead
- Quality checks are performed at intervals to balance performance and monitoring
- The verifier registry stores verifiers in memory (simple approach for now)
