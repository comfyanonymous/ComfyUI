
import argparse
import logging
import sys
import torch.utils.data

import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from tools.ptq.models import get_recipe_class, list_recipes
from tools.ptq.quantizer import PTQPipeline
from tools.ptq.utils import register_comfy_ops, FP8_CFG
from tools.ptq.dataset.t2i import PromptDataset

def main():
    """Main entry point for PTQ CLI."""

    # Step 1: Parse model_type first to determine which recipe to use
    parser = argparse.ArgumentParser(
        description="Quantize ComfyUI models using NVIDIA ModelOptimizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--model_type",
        required=True,
        choices=list_recipes(),
        help="Model recipe to use"
    )

    # Parse just model_type first to get recipe class
    args, remaining = parser.parse_known_args()

    # Step 2: Get recipe class and add its model-specific arguments
    recipe_cls = get_recipe_class(args.model_type)
    recipe_cls.add_model_args(parser)

    # Step 3: Add common arguments
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for amax artefact"
    )
    parser.add_argument(
        "--calib_steps",
        type=int,
        help="Override default calibration steps"
    )
    parser.add_argument(
        "--calib_data",
        default="tools/ptq/data/calib_prompts.txt",
        help="Path to calibration prompts"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (sets logging to DEBUG and calib_steps to 1)"
    )

    # Step 4: Parse all arguments
    args = parser.parse_args()

    # Configure logging
    if args.debug:
        logging.basicConfig(
            level=logging.DEBUG,
            format='[%(levelname)s] %(name)s: %(message)s'
        )
        logging.info("Debug mode enabled")
    else:
        logging.basicConfig(
            level=logging.INFO,
            format='[%(levelname)s] %(message)s'
        )

    # Step 5: Create recipe instance
    try:
        recipe = recipe_cls(args)
    except Exception as e:
        logging.error(f"Failed to initialize recipe: {e}")
        sys.exit(1)

    # Debug mode overrides calibration steps
    if args.debug:
        calib_steps = 1
        logging.debug("Debug mode: forcing calib_steps=1")
    elif args.calib_steps:
        calib_steps = args.calib_steps
    else:
        calib_steps = recipe.get_default_calib_steps()

    # Print header
    if args.debug:
        pass

    # Step 6: Register ComfyUI ops with ModelOptimizer
    logging.info("Registering ComfyUI ops with ModelOptimizer...")
    register_comfy_ops()

    # Step 7: Load model
    logging.info("[1/6] Loading model...")
    try:
        model_components = recipe.load_model()
        model_patcher = model_components[0]
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        sys.exit(1)

    # Step 8: Create PTQ pipeline
    logging.info("[2/6] Preparing quantization...")
    try:
        pipeline = PTQPipeline(
            model_patcher,
            quant_config=FP8_CFG,
            filter_func=recipe.get_filter_func()
        )
    except Exception as e:
        logging.error(f"Failed to prepare quantization: {e}")
        sys.exit(1)

    # Step 9: Create calibration pipeline
    logging.info("[3/6] Creating calibration pipeline...")
    try:
        calib_pipeline = recipe.create_calibration_pipeline(model_components)
    except Exception as e:
        logging.error(f"Failed to create calibration pipeline: {e}")
        sys.exit(1)

    # Step 10: Load calibration data
    logging.info(f"[4/6] Loading calibration data from {args.calib_data}")
    try:
        dataset = PromptDataset(args.calib_data)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
        logging.info(f"Loaded {len(dataset)} prompts")
    except Exception as e:
        logging.error(f"Failed to load calibration data: {e}")
        sys.exit(1)

    # Step 11: Run calibration
    logging.info(f"[5/6] Running calibration ({calib_steps} steps)...")
    try:
        pipeline.calibrate_with_pipeline(
            calib_pipeline,
            dataloader,
            num_steps=calib_steps,
            get_forward_loop=recipe.get_forward_loop
        )
    except Exception as e:
        logging.error(f"Calibration failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Only save amax values
    logging.info("[6/6] Extracting and saving amax values...")
    try:
        # Build metadata
        metadata = {
            "model_type": recipe.name(),
            "calibration_steps": calib_steps,
            "calibration_data": args.calib_data,
            "quantization_format": "FP8_E4M3",
            "debug_mode": args.debug
        }

        # Add checkpoint path if available
        if hasattr(args, 'ckpt_path') and args.ckpt_path:
            metadata["checkpoint_path"] = args.ckpt_path

        pipeline.save_amax_values(args.output, metadata=metadata)

        # Success!
    except Exception as e:
        logging.error(f"Failed to save amax values: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

