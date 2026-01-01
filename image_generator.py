#!/usr/bin/env python3
"""
Smart Image Generator - Dual Model with Selection
Juggernaut (nature/action) + Pony Realism (people/explicit)
"""

import torch
import gradio as gr
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
import time
import gc

# Global pipelines
pipe_juggernaut = None
pipe_pony_realism = None
models_loaded = False

def load_models():
    global pipe_juggernaut, pipe_pony_realism, models_loaded

    if models_loaded:
        return "Models already loaded!"

    print("Loading Juggernaut XL V9 (Nature/Action)...", flush=True)
    pipe_juggernaut = StableDiffusionXLPipeline.from_pretrained(
        "/workspace/models/juggernaut-xl-v9",
        torch_dtype=torch.float16,
        variant="fp16"
    ).to("cuda")
    pipe_juggernaut.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe_juggernaut.scheduler.config,
        use_karras_sigmas=True,
        algorithm_type="dpmsolver++"
    )
    pipe_juggernaut.enable_attention_slicing()
    print("Juggernaut ready!", flush=True)

    print("Loading Pony Realism V2.2 (People/Explicit)...", flush=True)
    pipe_pony_realism = StableDiffusionXLPipeline.from_single_file(
        "/workspace/models/pony-realism.safetensors",
        torch_dtype=torch.float16,
    ).to("cuda")
    pipe_pony_realism.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe_pony_realism.scheduler.config,
        use_karras_sigmas=True,
        algorithm_type="dpmsolver++"
    )
    pipe_pony_realism.enable_attention_slicing()
    print("Pony Realism ready!", flush=True)

    models_loaded = True
    return "Both models loaded!"

def unload_models():
    global pipe_juggernaut, pipe_pony_realism, models_loaded

    print("Unloading models...", flush=True)

    if pipe_juggernaut is not None:
        del pipe_juggernaut
        pipe_juggernaut = None

    if pipe_pony_realism is not None:
        del pipe_pony_realism
        pipe_pony_realism = None

    models_loaded = False
    gc.collect()
    torch.cuda.empty_cache()

    print("Models unloaded, VRAM freed!", flush=True)
    return "Models unloaded! VRAM freed."

def generate(prompt, negative_prompt, model_choice, use_trigger, width, height, steps, guidance, seed):
    global pipe_juggernaut, pipe_pony_realism, models_loaded

    if not models_loaded:
        load_models()

    # Select model based on choice
    if model_choice == "Pony Realism (People/Explicit)":
        pipe = pipe_pony_realism
        model_name = "Pony Realism"
        # Add score tags for Pony if not present
        if "score_9" not in prompt.lower():
            prompt = f"score_9, score_8_up, score_7_up, {prompt}"
        if not negative_prompt:
            negative_prompt = "score_4, score_5, score_6, blurry, bad anatomy, deformed, ugly, watermark"
    else:
        pipe = pipe_juggernaut
        model_name = "Juggernaut"
        if not negative_prompt:
            negative_prompt = "blurry, low quality, bad anatomy, deformed, ugly, watermark, text, cartoon, anime"

    if use_trigger and "alexandra" not in prompt.lower():
        prompt = f"alexandratitus767 woman, {prompt}"

    if seed == -1:
        seed = int(time.time()) % 2147483647

    generator = torch.Generator("cuda").manual_seed(seed)

    print(f"[{model_name}] Generating: {prompt[:60]}...", flush=True)

    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator,
        height=height,
        width=width,
    ).images[0]

    return image, f"Seed: {seed} | {model_name}"

# Pre-load models at startup
print("=" * 50, flush=True)
print("LOADING BOTH MODELS...", flush=True)
print("=" * 50, flush=True)
load_models()

# UI
print("Starting Gradio UI...", flush=True)
with gr.Blocks(title="Dual Model Image Generator") as demo:
    gr.Markdown("# Image Generator")
    gr.Markdown("**Juggernaut** = Nature, Fishing, Landscapes | **Pony Realism** = People, Explicit")

    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt", lines=3,
                placeholder="Describe what you want...")
            negative = gr.Textbox(label="Negative Prompt",
                value="blurry, bad anatomy, deformed, ugly, watermark")

            model_choice = gr.Radio(
                choices=["Juggernaut (Nature/Action)", "Pony Realism (People/Explicit)"],
                value="Pony Realism (People/Explicit)",
                label="Select Model"
            )

            use_trigger = gr.Checkbox(label="Add Alexandra (your face)", value=False)

            with gr.Row():
                width = gr.Slider(512, 1536, value=1024, step=64, label="Width")
                height = gr.Slider(512, 1536, value=1024, step=64, label="Height")

            with gr.Row():
                steps = gr.Slider(10, 50, value=30, step=1, label="Steps")
                guidance = gr.Slider(3, 12, value=7.0, step=0.5, label="Guidance")

            seed = gr.Number(value=-1, label="Seed (-1=random)")

            with gr.Row():
                btn = gr.Button("Generate", variant="primary", size="lg")
                unload_btn = gr.Button("Unload Models", variant="secondary")
                load_btn = gr.Button("Load Models", variant="secondary")

        with gr.Column():
            output = gr.Image(label="Generated Image", height=512)
            info = gr.Textbox(label="Info")
            status = gr.Textbox(label="Model Status", value="Both models loaded")

    gr.Markdown("### Quick Guide:")
    gr.Markdown("- **Fishing, nature, landscapes, action** → Use Juggernaut")
    gr.Markdown("- **People, portraits, explicit content** → Use Pony Realism")

    btn.click(generate, [prompt, negative, model_choice, use_trigger, width, height, steps, guidance, seed], [output, info], api_name="generate")
    unload_btn.click(unload_models, outputs=[status], api_name="unload_models")
    load_btn.click(load_models, outputs=[status], api_name="load_models")

print("Launching demo...", flush=True)
demo.launch(server_name="0.0.0.0", server_port=7865, share=False, show_error=True, prevent_thread_lock=True)

# Keep running
import time
while True:
    time.sleep(1)
