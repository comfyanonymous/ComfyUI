#!/usr/bin/env python3
"""
Standalone script to generate TTS audio using VibeVoice.
This script has NO ComfyUI dependencies and uses the models directly from HuggingFace.

Based on Microsoft's VibeVoice: https://github.com/microsoft/VibeVoice

Requirements:
    pip install torch transformers numpy scipy soundfile librosa huggingface-hub

Usage:
    python generate_vibevoice_standalone.py
"""

import torch
import numpy as np
import soundfile as sf
import os
import random
import re
import logging
from typing import Optional, List, Tuple
from huggingface_hub import snapshot_download

logging.basicConfig(level=logging.INFO, format='[VibeVoice] %(message)s')
logger = logging.getLogger(__name__)

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    logger.warning("librosa not available - resampling will not work")
    LIBROSA_AVAILABLE = False


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    if seed == 0:
        seed = random.randint(1, 0xffffffffffffffff)
    
    MAX_NUMPY_SEED = 2**32 - 1
    numpy_seed = seed % MAX_NUMPY_SEED
    
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(numpy_seed)
    random.seed(seed)
    
    return seed


def parse_script(script: str) -> Tuple[List[Tuple[int, str]], List[int]]:
    """
    Parse speaker script into (speaker_id, text) tuples.
    
    Supports formats:
        [1] Some text...
        Speaker 1: Some text...
    
    Returns:
        parsed_lines: List of (0-based speaker_id, text) tuples
        speaker_ids: List of unique 1-based speaker IDs in order of appearance
    """
    parsed_lines = []
    speaker_ids_in_script = []
    
    line_format_regex = re.compile(r'^(?:Speaker\s+(\d+)\s*:|\[(\d+)\])\s*(.*)$', re.IGNORECASE)
    
    for line in script.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        
        match = line_format_regex.match(line)
        if match:
            speaker_id_str = match.group(1) or match.group(2)
            speaker_id = int(speaker_id_str)
            text_content = match.group(3)
            
            if match.group(1) is None and text_content.lstrip().startswith(':'):
                colon_index = text_content.find(':')
                text_content = text_content[colon_index + 1:]
            
            if speaker_id < 1:
                logger.warning(f"Speaker ID must be 1 or greater. Skipping line: '{line}'")
                continue
            
            text = text_content.strip()
            internal_speaker_id = speaker_id - 1
            parsed_lines.append((internal_speaker_id, text))
            
            if speaker_id not in speaker_ids_in_script:
                speaker_ids_in_script.append(speaker_id)
        else:
            logger.warning(f"Could not parse speaker marker, ignoring: '{line}'")
    
    if not parsed_lines and script.strip():
        logger.info("No speaker markers found. Treating entire text as Speaker 1.")
        parsed_lines.append((0, script.strip()))
        speaker_ids_in_script.append(1)
    
    return parsed_lines, sorted(list(set(speaker_ids_in_script)))


def load_audio_file(audio_path: str, target_sr: int = 24000) -> Optional[np.ndarray]:
    """Load audio file and convert to mono at target sample rate"""
    if not os.path.exists(audio_path):
        logger.error(f"Audio file not found: {audio_path}")
        return None
    
    logger.info(f"Loading audio: {audio_path}")
    
    try:
        # Load audio using soundfile
        waveform, sr = sf.read(audio_path)
        
        # Convert to mono if stereo
        if waveform.ndim > 1:
            waveform = np.mean(waveform, axis=1)
        
        # Resample if needed
        if sr != target_sr:
            if not LIBROSA_AVAILABLE:
                raise ImportError("librosa is required for resampling. Install with: pip install librosa")
            logger.info(f"Resampling from {sr}Hz to {target_sr}Hz")
            waveform = librosa.resample(y=waveform, orig_sr=sr, target_sr=target_sr)
        
        # Validate audio
        if np.any(np.isnan(waveform)) or np.any(np.isinf(waveform)):
            logger.error("Audio contains NaN or Inf values, replacing with zeros")
            waveform = np.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0)
        
        if np.all(waveform == 0):
            logger.warning("Audio waveform is completely silent")
        
        # Normalize extreme values
        max_val = np.abs(waveform).max()
        if max_val > 10.0:
            logger.warning(f"Audio values are very large (max: {max_val}), normalizing")
            waveform = waveform / max_val
        
        return waveform.astype(np.float32)
        
    except Exception as e:
        logger.error(f"Error loading audio: {e}")
        return None


def download_model(model_name: str = "VibeVoice-1.5B", cache_dir: str = "./models"):
    """Download VibeVoice model from HuggingFace"""
    
    repo_mapping = {
        "VibeVoice-1.5B": "microsoft/VibeVoice-1.5B",
        "VibeVoice-Large": "aoi-ot/VibeVoice-Large"
    }
    
    if model_name not in repo_mapping:
        raise ValueError(f"Unknown model: {model_name}. Choose from: {list(repo_mapping.keys())}")
    
    repo_id = repo_mapping[model_name]
    model_path = os.path.join(cache_dir, model_name)
    
    if os.path.exists(os.path.join(model_path, "config.json")):
        logger.info(f"Model already downloaded: {model_path}")
        return model_path
    
    logger.info(f"Downloading model from {repo_id}...")
    os.makedirs(cache_dir, exist_ok=True)
    
    model_path = snapshot_download(
        repo_id=repo_id,
        local_dir=model_path,
        local_dir_use_symlinks=False
    )
    
    logger.info(f"Model downloaded to: {model_path}")
    return model_path


def generate_tts(
    text: str,
    model_name: str = "VibeVoice-Large",
    speaker_audio_paths: Optional[dict] = None,
    output_path: str = "output.wav",
    cfg_scale: float = 1.3,
    inference_steps: int = 10,
    seed: int = 42,
    temperature: float = 0.95,
    top_p: float = 0.95,
    top_k: int = 0,
    cache_dir: str = "./models",
    device: str = "auto"
):
    """
    Generate TTS audio using VibeVoice
    
    Args:
        text: Text script with speaker markers like "[1] text" or "Speaker 1: text"
        model_name: Model to use ("VibeVoice-1.5B" or "VibeVoice-Large")
        speaker_audio_paths: Dict mapping speaker IDs to audio file paths for voice cloning
                            e.g., {1: "voice1.wav", 2: "voice2.wav"}
        output_path: Where to save the generated audio
        cfg_scale: Classifier-Free Guidance scale (higher = more adherence to prompt)
        inference_steps: Number of diffusion steps
        seed: Random seed for reproducibility
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        top_k: Top-K sampling parameter
        cache_dir: Directory to cache downloaded models
        device: Device to use ("cuda", "mps", "cpu", or "auto" for automatic detection)
    """
    
    # Set seed
    actual_seed = set_seed(seed)
    logger.info(f"Using seed: {actual_seed}")
    
    # Determine device - with MPS support for Mac
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
            logger.info("MPS (Metal Performance Shaders) detected - using Mac GPU acceleration")
        else:
            device = "cpu"
    logger.info(f"Using device: {device}")
    
    # Download model if needed
    model_path = download_model(model_name, cache_dir)
    
    # Import VibeVoice components
    logger.info("Loading VibeVoice model...")
    try:
        # Add the VibeVoice custom model code to path
        import sys
        vibevoice_custom_path = os.path.join(os.path.dirname(__file__), "custom_nodes", "ComfyUI-VibeVoice")
        if vibevoice_custom_path not in sys.path:
            sys.path.insert(0, vibevoice_custom_path)
        
        # Import custom VibeVoice model
        from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
        from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
        from vibevoice.processor.vibevoice_tokenizer_processor import VibeVoiceTokenizerProcessor
        from vibevoice.modular.modular_vibevoice_text_tokenizer import VibeVoiceTextTokenizerFast
        from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig
        import json
        
        # Load config
        config_path = os.path.join(model_path, "config.json")
        config = VibeVoiceConfig.from_pretrained(config_path)
        
        # Load tokenizer - download if not present
        tokenizer_file = os.path.join(model_path, "tokenizer.json")
        if not os.path.exists(tokenizer_file):
            logger.info(f"tokenizer.json not found, downloading from HuggingFace...")
            from huggingface_hub import hf_hub_download
            
            # Determine which Qwen model to use based on model size
            qwen_repo = "Qwen/Qwen2.5-1.5B" if "1.5B" in model_name else "Qwen/Qwen2.5-7B"
            
            try:
                hf_hub_download(
                    repo_id=qwen_repo,
                    filename="tokenizer.json",
                    local_dir=model_path,
                    local_dir_use_symlinks=False
                )
                logger.info("tokenizer.json downloaded successfully")
            except Exception as e:
                logger.error(f"Failed to download tokenizer.json: {e}")
                raise FileNotFoundError(f"Could not download tokenizer.json from {qwen_repo}")
        
        tokenizer = VibeVoiceTextTokenizerFast(tokenizer_file=tokenizer_file)
        
        # Load processor config
        preprocessor_config_path = os.path.join(model_path, "preprocessor_config.json")
        processor_config_data = {}
        if os.path.exists(preprocessor_config_path):
            with open(preprocessor_config_path, 'r') as f:
                processor_config_data = json.load(f)
        
        audio_processor = VibeVoiceTokenizerProcessor()
        processor = VibeVoiceProcessor(
            tokenizer=tokenizer,
            audio_processor=audio_processor,
            speech_tok_compress_ratio=processor_config_data.get("speech_tok_compress_ratio", 3200),
            db_normalize=processor_config_data.get("db_normalize", True)
        )
        
        # Load model
        # MPS doesn't support bfloat16 well, use float16
        if device == "mps":
            dtype = torch.float16
            logger.info("Using float16 for MPS device")
        elif torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float16
        
        model = VibeVoiceForConditionalGenerationInference.from_pretrained(
            model_path,
            config=config,
            torch_dtype=dtype,
            device_map=device,
            attn_implementation="sdpa"
        )
        
        model.eval()
        logger.info("Model loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Parse script
    parsed_lines, speaker_ids = parse_script(text)
    if not parsed_lines:
        raise ValueError("Script is empty or invalid")
    
    logger.info(f"Parsed {len(parsed_lines)} lines with speakers: {speaker_ids}")
    
    # Load speaker audio samples
    voice_samples = []
    if speaker_audio_paths is None:
        speaker_audio_paths = {}
    
    for speaker_id in speaker_ids:
        audio_path = speaker_audio_paths.get(speaker_id)
        if audio_path:
            audio = load_audio_file(audio_path, target_sr=24000)
            if audio is None:
                logger.warning(f"Could not load audio for speaker {speaker_id}, using zero-shot TTS")
                voice_samples.append(None)
            else:
                voice_samples.append(audio)
        else:
            logger.info(f"No reference audio for speaker {speaker_id}, using zero-shot TTS")
            voice_samples.append(None)
    
    # Prepare inputs
    logger.info("Processing inputs...")
    try:
        inputs = processor(
            parsed_scripts=[parsed_lines],
            voice_samples=[voice_samples],
            speaker_ids_for_prompt=[speaker_ids],
            padding=True,
            return_tensors="pt",
            return_attention_mask=True
        )
        
        # Move to device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
    except Exception as e:
        logger.error(f"Error processing inputs: {e}")
        raise
    
    # Configure generation
    model.set_ddpm_inference_steps(num_steps=inference_steps)
    
    generation_config = {
        'do_sample': True,
        'temperature': temperature,
        'top_p': top_p,
    }
    if top_k > 0:
        generation_config['top_k'] = top_k
    
    # Generate
    logger.info(f"Generating audio ({inference_steps} steps)...")
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=None,
                cfg_scale=cfg_scale,
                tokenizer=processor.tokenizer,
                generation_config=generation_config,
                verbose=False
            )
        
        # Extract waveform
        waveform = outputs.speech_outputs[0].cpu().numpy()
        
        # Ensure correct shape
        if waveform.ndim == 1:
            waveform = waveform.reshape(1, -1)
        elif waveform.ndim == 2 and waveform.shape[0] > 1:
            # If multiple channels, take first
            waveform = waveform[0:1, :]
        
        # Convert to float32 for soundfile compatibility
        waveform = waveform.astype(np.float32)
        
        # Save audio
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        sf.write(output_path, waveform.T, 24000)
        logger.info(f"Audio saved to: {output_path}")
        
        return waveform
        
    except Exception as e:
        logger.error(f"Error during generation: {e}")
        raise


def main():
    """Example usage"""
    
    # Configuration
    model_name = "VibeVoice-Large"  # or "VibeVoice-Large"
    
    # Text to generate - supports multiple speakers
    text = """
    [1] Hello, this is speaker one. How are you today?
    [2] Hi there! This is speaker two responding to you. It's great to meet you.
    [1] Likewise! Let's generate some amazing speech together.
    [2] Absolutely! VibeVoice makes it so easy to create diverse voices.
    """
    
    # Reference audio for voice cloning (optional)
    # If not provided, will use zero-shot TTS
    speaker_audio_paths = {
        1: "input/audio1.wav",  # Path to reference audio for speaker 1
        2: "input/laundry.mp3",  # Uncomment to provide reference for speaker 2
    }
    
    # Generation parameters
    output_path = "output/vibevoice_generated.wav"
    cfg_scale = 1.3
    inference_steps = 10
    seed = 42  # or 0 for random
    temperature = 0.95
    top_p = 0.95
    top_k = 0
    
    print("=" * 60)
    print("VibeVoice TTS - Standalone Script")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Text: {text[:100]}...")
    print("=" * 60)
    
    try:
        generate_tts(
            text=text,
            model_name=model_name,
            speaker_audio_paths=speaker_audio_paths,
            output_path=output_path,
            cfg_scale=cfg_scale,
            inference_steps=inference_steps,
            seed=seed,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            cache_dir="./models",
            device="auto"
        )
        
        print("=" * 60)
        print("Generation complete!")
        print(f"Audio saved to: {output_path}")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
