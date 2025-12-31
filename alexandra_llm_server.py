#!/usr/bin/env python3
"""
Alexandra LLM Server - Serves Qwen2.5-72B with LoRA using 4-bit quantization.
Provides an OpenAI-compatible API endpoint.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from flask import Flask, request, jsonify
import threading
import time

# Configuration
BASE_MODEL = "/workspace/models/Qwen2.5-72B-Instruct"
LORA_PATH = "/workspace/ai-clone-training/my-output/alexandra-qwen72b-lora/checkpoint-20608"
PORT = 8000

app = Flask(__name__)
model = None
tokenizer = None
model_lock = threading.Lock()

def load_model():
    global model, tokenizer
    print(f"Loading tokenizer from {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    print("Setting up 4-bit quantization config...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    print(f"Loading base model with 4-bit quantization...")
    print("This may take 10-20 minutes for a 72B model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"Loading LoRA adapter from {LORA_PATH}...")
    model = PeftModel.from_pretrained(model, LORA_PATH)

    print("Model loaded successfully!")
    return True

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """OpenAI-compatible chat completions endpoint."""
    global model, tokenizer

    if model is None:
        return jsonify({"error": "Model not loaded yet"}), 503

    data = request.json
    messages = data.get('messages', [])
    max_tokens = data.get('max_tokens', 512)
    temperature = data.get('temperature', 0.7)

    # Format messages for Qwen
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    with model_lock:
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.15,
                no_repeat_ngram_size=3,
            )

        response_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    # Return OpenAI-compatible response
    return jsonify({
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "alexandra-72b",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": response_text
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": inputs['input_ids'].shape[1],
            "completion_tokens": len(outputs[0]) - inputs['input_ids'].shape[1],
            "total_tokens": len(outputs[0])
        }
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "ready" if model is not None else "loading",
        "model": "alexandra-72b"
    })

if __name__ == '__main__':
    print("=" * 60)
    print("Alexandra LLM Server")
    print("=" * 60)
    print(f"Base model: {BASE_MODEL}")
    print(f"LoRA: {LORA_PATH}")
    print(f"Port: {PORT}")
    print("=" * 60)

    # Load model in background
    load_thread = threading.Thread(target=load_model)
    load_thread.start()

    # Start server
    app.run(host='0.0.0.0', port=PORT, threaded=True)
