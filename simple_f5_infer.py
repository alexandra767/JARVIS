#!/usr/bin/env python3
"""Simple F5-TTS inference with memory-efficient GPU usage."""

import sys
import os
import gc
import torch
import soundfile as sf
import numpy as np
from importlib.resources import files
from omegaconf import OmegaConf

# Disable transformers pipeline import in f5_tts
class DummyPipeline:
    pass

import transformers
transformers.pipeline = lambda *args, **kwargs: None

# Now import f5_tts components
from f5_tts.infer.utils_infer import (
    load_vocoder,
    load_model,
    preprocess_ref_audio_text,
    infer_process,
)
from hydra.utils import get_class

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="F5TTS_v1_Base")
    parser.add_argument("--ckpt_file", required=True)
    parser.add_argument("--ref_audio", required=True)
    parser.add_argument("--ref_text", required=True)
    parser.add_argument("--gen_text", required=True)
    parser.add_argument("--output_dir", default="/tmp")
    parser.add_argument("--output_file", default="output.wav")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--vocab_file", default="", help="Path to custom vocab.txt file")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    print(f"Checkpoint: {args.ckpt_file}")
    print(f"Speed: {args.speed}")

    # Clear GPU cache before loading
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    # Try CUDA first, fall back to CPU if out of memory
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load vocoder - try GPU, fall back to CPU
    try:
        vocoder = load_vocoder(device=device)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" in str(e).lower() or "CUDA" in str(e):
            print(f"GPU out of memory, falling back to CPU")
            device = "cpu"
            torch.cuda.empty_cache()
            gc.collect()
            vocoder = load_vocoder(device=device)
        else:
            raise

    # Load model config
    model_cfg_path = str(files("f5_tts").joinpath(f"configs/{args.model}.yaml"))
    model_cfg = OmegaConf.load(model_cfg_path)

    # Get model class and architecture
    model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
    model_arc = model_cfg.model.arch

    # Load F5-TTS model - with CPU fallback
    vocab_file = args.vocab_file if args.vocab_file else ""
    if vocab_file:
        print(f"Using custom vocab: {vocab_file}")

    try:
        model = load_model(
            model_cls,
            model_arc,
            args.ckpt_file,
            vocab_file=vocab_file,
            device=device,
        )
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" in str(e).lower() or "CUDA" in str(e):
            print(f"GPU out of memory for model, falling back to CPU")
            device = "cpu"
            torch.cuda.empty_cache()
            gc.collect()
            vocoder = load_vocoder(device=device)
            model = load_model(
                model_cls,
                model_arc,
                args.ckpt_file,
                vocab_file=vocab_file,
                device=device,
            )
        else:
            raise

    print(f"Loading reference audio: {args.ref_audio}")

    # Preprocess reference
    ref_audio, ref_text = preprocess_ref_audio_text(
        args.ref_audio,
        args.ref_text,
    )

    print(f"Generating speech for: {args.gen_text}")

    # Generate with speed parameter
    audio, sr, _ = infer_process(
        ref_audio,
        ref_text,
        args.gen_text,
        model,
        vocoder,
        device=device,
        speed=args.speed,
    )

    # Save output
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_file)
    sf.write(output_path, audio, sr)
    print(f"Saved to: {output_path}")

    # Cleanup
    del model, vocoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    main()
