
### Create PTQ Artefact / Run Calibration 
Only needs to be done once per model
```
python -m tools.ptq.quantize \
    --model_type flux_schnell 
    --unet_path <path>/flux1-schnell.safetensors 
    --clip_path clip_l.safetensors 
    --t5_path t5xxl_fp16.safetensors  
    --output flux_schnell_debug.json 
    --calib_steps 16
```

### Create Quantized Checkpoint
Uses the artefact from before + checkpoint to generate a quantized checkpoint based of a yml configuration. 
This file defines which layers to quantize and what dtype they should use using regex. See `tools/ptq/configs/` for examples.
```yaml
# config.yml
disable_list: ["*img_in*", "*final_layer*", "*norm*"]  # Keep these in BF16
per_layer_dtype: {"*": "float8_e4m3fn"}                # Everything else to FP8
```

```
python -m tools.ptq.checkpoint_merger 
    --artefact flux_dev_debug.json 
    --checkpoint <path>/flux1-dev.safetensors 
    --config tools/ptq/configs/flux_nvfp4.yml 
    --output <out_path>/flux1-nvfp4.safetensors
    --debug
```

