"""
Alexandra AI - Model Loading Module
Shared between alexandra_ui.py and jarvis_dashboard.py
Supports both text-only and vision-language models
"""

import torch
import os
import sys
from PIL import Image
import base64
import io

# Model configuration
MODELS = {
    "Qwen2.5-VL-72B (Vision+Chat)": {
        "base": "/workspace/host/models/Qwen2.5-VL-72B-Instruct",
        "lora": None,
        "type": "vision",  # Vision-language model (uses bitsandbytes 4bit)
    },
    "Qwen2.5-VL + Coding LoRA": {
        "base": "/workspace/host/models/Qwen2.5-VL-72B-Instruct",
        "lora": "/workspace/host/coding-lora-final",
        "type": "vision",
    },
    "Alexandra (Personal)": {
        "base": "/workspace/host/models/Qwen2.5-72B-Instruct",
        "lora": "/workspace/host/ai-clone-training/my-output/alexandra-personal-lora-final",
        "type": "text",
    },
    "Alexandra (Original)": {
        "base": "/workspace/host/models/Qwen2.5-72B-Instruct",
        "lora": "/workspace/host/ai-clone-training/my-output/alexandra-qwen72b-lora-final",
        "type": "text",
    },
    "Qwen 72B (Base)": {
        "base": "/workspace/host/models/Qwen2.5-72B-Instruct",
        "lora": None,
        "type": "text",
    },
}

# Check which models exist
AVAILABLE_MODELS = {}
for name, paths in MODELS.items():
    if os.path.exists(paths["base"]):
        if paths["lora"] is None or os.path.exists(paths["lora"]):
            AVAILABLE_MODELS[name] = paths

DEFAULT_MODEL = list(AVAILABLE_MODELS.keys())[0] if AVAILABLE_MODELS else None

# Global model state
model = None
tokenizer = None
processor = None  # For vision models
current_model_name = None
current_model_type = None  # "text" or "vision"


def load_model(model_name=None):
    """Load selected model (text or vision)"""
    global model, tokenizer, processor, current_model_name, current_model_type

    if model_name is None:
        model_name = DEFAULT_MODEL

    if not AVAILABLE_MODELS:
        return "No local models available"

    if model_name not in AVAILABLE_MODELS:
        return f"Model '{model_name}' not found"

    if current_model_name == model_name and model is not None:
        return f"Model {model_name} already loaded"

    # Unload current model
    if model is not None:
        del model
        if tokenizer is not None:
            del tokenizer
        if processor is not None:
            del processor
        torch.cuda.empty_cache()
        model = None
        tokenizer = None
        processor = None

    print(f"Loading {model_name}...")
    paths = AVAILABLE_MODELS[model_name]
    model_type = paths.get("type", "text")

    if model_type == "vision":
        # Load Vision-Language model with bitsandbytes 4-bit quantization
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=True,
        )

        max_memory = {0: "115GB", "cpu": "48GB"}

        print("Loading Qwen2.5-VL vision model (4-bit quantized)...")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            paths["base"],
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            max_memory=max_memory,
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(paths["base"])
        tokenizer = None  # VL models use processor instead

        # Note: LoRA for VL models would need special handling
        if paths["lora"] and os.path.exists(paths["lora"]):
            print(f"Note: LoRA adapter at {paths['lora']} - attempting to load...")
            try:
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, paths["lora"])
                print("LoRA loaded successfully!")
            except Exception as e:
                print(f"Could not load LoRA on VL model: {e}")

        current_model_type = "vision"
    else:
        # Load text-only model
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        model = AutoModelForCausalLM.from_pretrained(
            paths["base"],
            quantization_config=bnb_config,
            device_map={"": 0},
            torch_dtype=torch.bfloat16,
        )

        if paths["lora"]:
            model = PeftModel.from_pretrained(model, paths["lora"])
            tokenizer = AutoTokenizer.from_pretrained(paths["lora"])
        else:
            tokenizer = AutoTokenizer.from_pretrained(paths["base"])

        processor = None
        current_model_type = "text"

    current_model_name = model_name
    mem = torch.cuda.memory_allocated() / 1024**3
    print(f"Loaded {model_name} ({model_type}) - GPU Memory: {mem:.1f}GB")
    return f"Loaded: {model_name} ({model_type})"


def generate_response(user_input, history=None, system_prompt="", temperature=0.7, max_tokens=2048, image=None):
    """Generate response using local model (text or vision)"""
    global model, tokenizer, processor, current_model_type

    if model is None:
        return "Model not loaded. Please load a model first."

    # If image provided and we have a vision model, use vision generation
    if image is not None and current_model_type == "vision":
        return generate_vision_response(user_input, image, system_prompt, temperature, max_tokens)

    # Text-only generation
    if current_model_type == "vision":
        # Use processor for VL model text-only
        return _generate_vl_text_only(user_input, history, system_prompt, temperature, max_tokens)
    else:
        # Use tokenizer for standard text model
        return _generate_text_response(user_input, history, system_prompt, temperature, max_tokens)


def _generate_text_response(user_input, history, system_prompt, temperature, max_tokens):
    """Generate text response using standard text model"""
    global model, tokenizer

    if tokenizer is None:
        return "Tokenizer not loaded."

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": str(system_prompt)})

    if history:
        for msg in history[-100:]:
            if isinstance(msg, dict):
                messages.append(msg)
            elif isinstance(msg, tuple) and len(msg) == 2:
                messages.append({"role": "user", "content": str(msg[0])})
                messages.append({"role": "assistant", "content": str(msg[1])})

    messages.append({"role": "user", "content": str(user_input)})

    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=min(max_tokens, 4096),
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return response.strip()
    except Exception as e:
        return f"Error generating response: {e}"


def _generate_vl_text_only(user_input, history, system_prompt, temperature, max_tokens):
    """Generate text response using VL model (no image)"""
    global model, processor

    if processor is None:
        return "Processor not loaded."

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": str(system_prompt)})

    if history:
        for msg in history[-100:]:
            if isinstance(msg, dict):
                messages.append(msg)
            elif isinstance(msg, tuple) and len(msg) == 2:
                messages.append({"role": "user", "content": str(msg[0])})
                messages.append({"role": "assistant", "content": str(msg[1])})

    messages.append({"role": "user", "content": str(user_input)})

    try:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=min(max_tokens, 4096),
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
            )

        response = processor.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return response.strip()
    except Exception as e:
        return f"Error generating response: {e}"


def generate_vision_response(user_input, image, system_prompt="", temperature=0.7, max_tokens=2048):
    """Generate response with image analysis using VL model"""
    global model, processor, current_model_type

    if model is None or processor is None:
        return "Vision model not loaded."

    if current_model_type != "vision":
        return "Current model doesn't support vision. Please load a VL model."

    try:
        # Handle different image input types
        if isinstance(image, str):
            # File path or base64
            if os.path.exists(image):
                pil_image = Image.open(image).convert("RGB")
            elif image.startswith("data:image"):
                # Base64 data URL
                base64_data = image.split(",")[1]
                pil_image = Image.open(io.BytesIO(base64.b64decode(base64_data))).convert("RGB")
            else:
                # Assume raw base64
                pil_image = Image.open(io.BytesIO(base64.b64decode(image))).convert("RGB")
        elif isinstance(image, Image.Image):
            pil_image = image.convert("RGB")
        elif hasattr(image, 'read'):
            # File-like object
            pil_image = Image.open(image).convert("RGB")
        else:
            # Assume numpy array (from camera)
            import numpy as np
            if isinstance(image, np.ndarray):
                pil_image = Image.fromarray(image).convert("RGB")
            else:
                return "Unsupported image format"

        # Build message with image
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": str(system_prompt)})

        # Create vision message
        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": str(user_input)}
            ]
        })

        # Process with Qwen2.5-VL
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text],
            images=[pil_image],
            return_tensors="pt",
            padding=True
        ).to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=min(max_tokens, 4096),
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
            )

        response = processor.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return response.strip()

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error analyzing image: {e}"


def get_model_status():
    """Get current model status"""
    if model is None:
        return "No model loaded"
    mem = torch.cuda.memory_allocated() / 1024**3
    model_type = current_model_type or "unknown"
    return f"{current_model_name} ({model_type}) - GPU: {mem:.1f}GB"


def is_vision_model():
    """Check if current model supports vision"""
    return current_model_type == "vision"


def get_available_models():
    """Get list of available models"""
    return list(AVAILABLE_MODELS.keys())

