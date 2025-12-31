"""
Alexandra AI - Model Loading Module
Shared between alexandra_ui.py and jarvis_dashboard.py
"""

import torch
import os
import sys

# Model configuration
MODELS = {
    "Alexandra (Personal)": {
        "base": "/workspace/models/Qwen2.5-72B-Instruct",
        "lora": "/workspace/ai-clone-training/my-output/alexandra-personal-lora-final",
    },
    "Alexandra (Original)": {
        "base": "/workspace/models/Qwen2.5-72B-Instruct",
        "lora": "/workspace/ai-clone-training/my-output/alexandra-qwen72b-lora-final",
    },
    "Qwen 72B (Base)": {
        "base": "/workspace/models/Qwen2.5-72B-Instruct",
        "lora": None,
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
current_model_name = None


def load_model(model_name=None):
    """Load selected model"""
    global model, tokenizer, current_model_name

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
        del tokenizer
        torch.cuda.empty_cache()

    print(f"Loading {model_name}...")
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    paths = AVAILABLE_MODELS[model_name]

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # GB10 unified memory - load directly to CUDA
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

    current_model_name = model_name
    print(f"Loaded {model_name} - GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.1f}GB")
    return f"Loaded: {model_name}"


def generate_response(user_input, history=None, system_prompt="", temperature=0.7, max_tokens=2048):
    """Generate response using local model"""
    global model, tokenizer

    if model is None or tokenizer is None:
        return "Model not loaded. Please load a model first."

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": str(system_prompt)})

    # Add history
    if history:
        for msg in history[-10:]:
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


def get_model_status():
    """Get current model status"""
    if model is None:
        return "No model loaded"
    mem = torch.cuda.memory_allocated() / 1024**3
    return f"{current_model_name} - GPU: {mem:.1f}GB"

