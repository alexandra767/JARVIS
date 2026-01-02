"""
Alexandra AI - Vision Module
Image analysis using Qwen2.5-VL-72B (local, 4-bit quantized)
"""

import os
import torch
from PIL import Image
import numpy as np

# Model path - local model
VISION_MODEL_PATH = "/workspace/host/models/Qwen2.5-VL-72B-Instruct"

# Global model state
vision_model = None
vision_processor = None
VISION_AVAILABLE = False

def load_vision_model():
    """Load Qwen2.5-VL-72B model for vision analysis"""
    global vision_model, vision_processor, VISION_AVAILABLE

    if vision_model is not None:
        return "Vision model already loaded"

    # Check if model exists
    if not os.path.exists(VISION_MODEL_PATH):
        return f"Vision model not found at {VISION_MODEL_PATH}. Download in progress?"

    try:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

        print("[VISION] Loading Qwen2.5-VL-72B (4-bit quantized)...")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=True,
        )

        # Allow using up to 115GB of GPU memory
        max_memory = {0: "115GB", "cpu": "48GB"}

        vision_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            VISION_MODEL_PATH,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            max_memory=max_memory,
            trust_remote_code=True,
        )

        vision_processor = AutoProcessor.from_pretrained(VISION_MODEL_PATH)

        VISION_AVAILABLE = True
        mem = torch.cuda.memory_allocated() / 1024**3
        print(f"[VISION] Loaded! GPU: {mem:.1f}GB")
        return f"Vision model loaded - GPU: {mem:.1f}GB"

    except Exception as e:
        print(f"[VISION] Load failed: {e}")
        VISION_AVAILABLE = False
        return f"Failed to load vision: {e}"


def unload_vision_model():
    """Unload vision model to free GPU memory"""
    global vision_model, vision_processor, VISION_AVAILABLE

    if vision_model is not None:
        del vision_model
        del vision_processor
        vision_model = None
        vision_processor = None
        torch.cuda.empty_cache()
        VISION_AVAILABLE = False
        print("[VISION] Unloaded")
        return "Vision model unloaded"
    return "No vision model loaded"


def analyze_image(image, prompt="Describe this image in detail."):
    """Analyze image using Qwen2-VL

    Args:
        image: PIL Image, numpy array, or file path
        prompt: Question or instruction about the image

    Returns:
        Analysis text
    """
    global vision_model, vision_processor, VISION_AVAILABLE

    if not VISION_AVAILABLE or vision_model is None:
        return "Vision model not loaded. Click 'Load Vision Model' first."

    try:
        # Convert to PIL Image if needed
        if isinstance(image, str):
            pil_image = Image.open(image)
        elif isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            pil_image = image
        else:
            return f"Unsupported image type: {type(image)}"

        # Ensure RGB
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        # Resize if too large (Qwen2-VL handles up to 1280 well)
        max_size = 1280
        if max(pil_image.size) > max_size:
            ratio = max_size / max(pil_image.size)
            new_size = (int(pil_image.width * ratio), int(pil_image.height * ratio))
            pil_image = pil_image.resize(new_size, Image.LANCZOS)

        # Prepare message format for Qwen2-VL
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        # Process inputs
        text = vision_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = vision_processor(
            text=[text],
            images=[pil_image],
            padding=True,
            return_tensors="pt"
        ).to(vision_model.device)

        # Generate response
        with torch.no_grad():
            output_ids = vision_model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )

        # Decode response
        generated_ids = output_ids[0][inputs.input_ids.shape[1]:]
        response = vision_processor.decode(generated_ids, skip_special_tokens=True)

        return response.strip()

    except Exception as e:
        return f"Vision analysis error: {e}"


def get_vision_status():
    """Get vision system status"""
    if VISION_AVAILABLE and vision_model is not None:
        mem = torch.cuda.memory_allocated() / 1024**3
        return f"Qwen2.5-VL-72B Ready - GPU: {mem:.1f}GB"
    if os.path.exists(VISION_MODEL_PATH):
        return "Vision model available (not loaded)"
    return "Vision model not downloaded"
