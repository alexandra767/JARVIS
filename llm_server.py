#!/usr/bin/env python3
"""
Simple LLM Server - Keeps model in memory for fast inference.
Runs on port 8000 with OpenAI-compatible API.
"""

import os
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

app = FastAPI(title="Alexandra LLM Server")

# Global model
model = None
tokenizer = None

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9

class ChatResponse(BaseModel):
    choices: List[dict]

def load_model():
    global model, tokenizer

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    MODEL_PATH = "/home/alexandratitus767/models/Qwen2.5-72B-Instruct"
    LORA_PATH = "/home/alexandratitus767/ai-clone-training/my-output/alexandra-personal-lora-final"

    print("Loading model...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
    )

    if os.path.exists(LORA_PATH):
        print(f"Loading LoRA from {LORA_PATH}")
        model = PeftModel.from_pretrained(model, LORA_PATH)
        tokenizer = AutoTokenizer.from_pretrained(LORA_PATH)
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    print("Model loaded and ready!")
    return model, tokenizer

@app.on_event("startup")
async def startup():
    global model, tokenizer
    model, tokenizer = load_model()

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    global model, tokenizer

    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response_text = tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    return {
        "choices": [
            {"message": {"role": "assistant", "content": response_text}}
        ]
    }

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
