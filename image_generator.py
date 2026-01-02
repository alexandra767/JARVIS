#!/usr/bin/env python3
"""
Smart Image Generator - Triple Model with Selection
Juggernaut (nature/action) + Pony Realism (people/explicit) + FLUX (best faces)
"""

import torch
import gradio as gr
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, DPMSolverMultistepScheduler, FluxPipeline
from PIL import Image
import time
import gc
import os

# Global pipelines
pipe_juggernaut = None
pipe_pony_realism = None
pipe_flux = None
pipe_juggernaut_img2img = None
pipe_pony_img2img = None
current_model = None  # Track which model is loaded to save VRAM

def load_sdxl_models():
    global pipe_juggernaut, pipe_pony_realism

    print("Loading Juggernaut XL V9 (Nature/Action)...", flush=True)
    pipe_juggernaut = StableDiffusionXLPipeline.from_pretrained(
        "RunDiffusion/Juggernaut-XL-v9",
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
        "/workspace/host/pony-realism.safetensors",
        torch_dtype=torch.float16,
    ).to("cuda")
    pipe_pony_realism.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe_pony_realism.scheduler.config,
        use_karras_sigmas=True,
        algorithm_type="dpmsolver++"
    )
    pipe_pony_realism.enable_attention_slicing()

    # Load Alexandra LoRA v2 for Pony (but disable by default - enabled per-request)
    lora_path = "/workspace/loras/alexandra-pony-lora-v2.safetensors"
    if os.path.exists(lora_path):
        print(f"Loading Alexandra LoRA v2 for Pony (inactive until requested)...", flush=True)
        pipe_pony_realism.load_lora_weights(lora_path, adapter_name="alexandra")
        pipe_pony_realism.disable_lora()  # Start with LoRA disabled
        print("Alexandra LoRA v2 loaded (disabled by default)!", flush=True)

    print("Pony Realism ready!", flush=True)
    return "SDXL models loaded!"

def load_flux():
    global pipe_flux

    if pipe_flux is not None:
        return "FLUX already loaded!"

    print("Loading FLUX.1-dev (first time downloads ~24GB, then cached)...", flush=True)

    # Use from_pretrained - downloads once then uses HuggingFace cache
    pipe_flux = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16
    ).to("cuda")

    # Load Alexandra FLUX LoRA (your trained face model)
    lora_path = "/workspace/host/ComfyUI/models/loras/flux_dreambooth.safetensors"
    uncensored_path = "/workspace/host/ComfyUI/models/loras/flux-uncensored.safetensors"

    adapters = []
    weights = []

    if os.path.exists(lora_path):
        print(f"Loading Alexandra FLUX LoRA (your trained face)...", flush=True)
        pipe_flux.load_lora_weights(lora_path, adapter_name="alexandra")
        adapters.append("alexandra")
        weights.append(1.0)
        print("Alexandra FLUX LoRA loaded!", flush=True)
    else:
        print(f"WARNING: Face LoRA not found at {lora_path}", flush=True)

    if os.path.exists(uncensored_path):
        print(f"Loading FLUX uncensored LoRA...", flush=True)
        pipe_flux.load_lora_weights(uncensored_path, adapter_name="uncensored")
        adapters.append("uncensored")
        weights.append(0.8)
        print("FLUX uncensored LoRA loaded!", flush=True)

    if adapters:
        pipe_flux.set_adapters(adapters, adapter_weights=weights)
        print(f"Active adapters: {adapters} with weights {weights}", flush=True)

    pipe_flux.enable_attention_slicing()
    print("FLUX ready!", flush=True)
    return "FLUX loaded!"

def load_img2img_pipelines():
    """Load img2img pipelines from existing text2img pipelines"""
    global pipe_juggernaut_img2img, pipe_pony_img2img, pipe_juggernaut, pipe_pony_realism

    if pipe_juggernaut is None:
        load_sdxl_models()

    print("Creating img2img pipelines...", flush=True)

    # Create img2img from existing pipelines (shares weights, minimal extra VRAM)
    pipe_juggernaut_img2img = StableDiffusionXLImg2ImgPipeline(
        vae=pipe_juggernaut.vae,
        text_encoder=pipe_juggernaut.text_encoder,
        text_encoder_2=pipe_juggernaut.text_encoder_2,
        tokenizer=pipe_juggernaut.tokenizer,
        tokenizer_2=pipe_juggernaut.tokenizer_2,
        unet=pipe_juggernaut.unet,
        scheduler=pipe_juggernaut.scheduler,
    )

    pipe_pony_img2img = StableDiffusionXLImg2ImgPipeline(
        vae=pipe_pony_realism.vae,
        text_encoder=pipe_pony_realism.text_encoder,
        text_encoder_2=pipe_pony_realism.text_encoder_2,
        tokenizer=pipe_pony_realism.tokenizer,
        tokenizer_2=pipe_pony_realism.tokenizer_2,
        unet=pipe_pony_realism.unet,
        scheduler=pipe_pony_realism.scheduler,
    )

    print("Img2img pipelines ready!", flush=True)
    return "Img2img ready!"

def generate_img2img(input_image, prompt, negative_prompt, model_choice, strength, steps, guidance, seed):
    """Generate image from input image using Pony or Juggernaut"""
    global pipe_juggernaut_img2img, pipe_pony_img2img

    if input_image is None:
        return None, "Please upload an image first!"

    if pipe_juggernaut_img2img is None:
        load_img2img_pipelines()

    if seed == -1:
        seed = int(time.time()) % 2147483647

    generator = torch.Generator("cuda").manual_seed(seed)

    # Convert input image to PIL if needed
    if not isinstance(input_image, Image.Image):
        input_image = Image.fromarray(input_image)

    # Resize to valid dimensions
    input_image = input_image.convert("RGB")
    w, h = input_image.size
    # Round to nearest 64
    w = (w // 64) * 64
    h = (h // 64) * 64
    if w < 512: w = 512
    if h < 512: h = 512
    if w > 1536: w = 1536
    if h > 1536: h = 1536
    input_image = input_image.resize((w, h), Image.LANCZOS)

    if "Pony" in model_choice:
        pipe = pipe_pony_img2img
        model_name = "Pony Realism"
        # Add score tags for Pony
        if "score_9" not in prompt.lower():
            prompt = f"score_9, score_8_up, score_7_up, {prompt}"
        if not negative_prompt:
            negative_prompt = "score_4, score_5, score_6, blurry, bad anatomy, deformed, ugly, watermark"
    else:
        pipe = pipe_juggernaut_img2img
        model_name = "Juggernaut"
        if not negative_prompt:
            negative_prompt = "blurry, low quality, bad anatomy, deformed, ugly, watermark, text, cartoon, anime"

    print(f"[{model_name} img2img] Strength: {strength}, Generating: {prompt[:50]}...", flush=True)

    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=input_image,
        strength=strength,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator,
    ).images[0]

    return image, f"Seed: {seed} | {model_name} img2img | Strength: {strength}"

def unload_all():
    global pipe_juggernaut, pipe_pony_realism, pipe_flux

    print("Unloading all models...", flush=True)

    if pipe_juggernaut is not None:
        del pipe_juggernaut
        pipe_juggernaut = None

    if pipe_pony_realism is not None:
        del pipe_pony_realism
        pipe_pony_realism = None

    if pipe_flux is not None:
        del pipe_flux
        pipe_flux = None

    gc.collect()
    torch.cuda.empty_cache()

    print("All models unloaded, VRAM freed!", flush=True)
    return "All models unloaded! VRAM freed."

def generate(prompt, negative_prompt, model_choice, use_trigger, width, height, steps, guidance, seed):
    global pipe_juggernaut, pipe_pony_realism, pipe_flux

    # Add trigger word if requested
    if use_trigger and "alexandra" not in prompt.lower():
        prompt = f"alexandratitus767 woman, {prompt}"

    if seed == -1:
        seed = int(time.time()) % 2147483647

    generator = torch.Generator("cuda").manual_seed(seed)

    # Select model based on choice
    if model_choice == "FLUX (Best Faces)":
        if pipe_flux is None:
            load_flux()
        pipe = pipe_flux
        model_name = "FLUX"

        print(f"[{model_name}] Generating: {prompt[:60]}...", flush=True)

        # FLUX uses different parameters
        image = pipe(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
            height=height,
            width=width,
        ).images[0]

    elif model_choice == "Pony Realism (People/Explicit)":
        if pipe_pony_realism is None:
            load_sdxl_models()
        pipe = pipe_pony_realism
        model_name = "Pony Realism"

        # Enable/disable Alexandra LoRA based on checkbox
        if use_trigger:
            pipe.enable_lora()
            model_name = "Pony Realism + Alexandra LoRA"
        else:
            pipe.disable_lora()

        # Add score tags for Pony if not present
        if "score_9" not in prompt.lower():
            prompt = f"score_9, score_8_up, score_7_up, {prompt}"
        if not negative_prompt:
            negative_prompt = "score_4, score_5, score_6, blurry, bad anatomy, deformed, ugly, watermark"

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

    else:  # Juggernaut
        if pipe_juggernaut is None:
            load_sdxl_models()
        pipe = pipe_juggernaut
        model_name = "Juggernaut"

        if not negative_prompt:
            negative_prompt = "blurry, low quality, bad anatomy, deformed, ugly, watermark, text, cartoon, anime"

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

# Pre-load SDXL models at startup (FLUX loads on demand to save VRAM)
print("=" * 50, flush=True)
print("LOADING SDXL MODELS...", flush=True)
print("(FLUX loads on first use to save VRAM)", flush=True)
print("=" * 50, flush=True)
load_sdxl_models()

# UI
print("Starting Gradio UI...", flush=True)
with gr.Blocks(title="Triple Model Image Generator") as demo:
    gr.Markdown("# Image Generator")

    with gr.Tabs():
        # ============ TEXT TO IMAGE TAB ============
        with gr.TabItem("Text to Image"):
            gr.Markdown("**Juggernaut** = Nature, Landscapes | **Pony Realism** = Explicit | **FLUX** = Best Faces")

            with gr.Row():
                with gr.Column():
                    prompt = gr.Textbox(label="Prompt", lines=3,
                        placeholder="Describe what you want...")
                    negative = gr.Textbox(label="Negative Prompt",
                        value="blurry, bad anatomy, deformed, ugly, watermark")

                    model_choice = gr.Radio(
                        choices=["Juggernaut (Nature/Action)", "Pony Realism (People/Explicit)", "FLUX (Best Faces)"],
                        value="FLUX (Best Faces)",
                        label="Select Model"
                    )

                    use_trigger = gr.Checkbox(label="Add Alexandra (your face)", value=True)

                    with gr.Row():
                        width = gr.Slider(512, 1536, value=1024, step=64, label="Width")
                        height = gr.Slider(512, 1536, value=1024, step=64, label="Height")

                    with gr.Row():
                        steps = gr.Slider(10, 50, value=30, step=1, label="Steps")
                        guidance = gr.Slider(1, 12, value=3.5, step=0.5, label="Guidance (FLUX: 3-4, SDXL: 7-8)")

                    seed = gr.Number(value=-1, label="Seed (-1=random)")

                    with gr.Row():
                        btn = gr.Button("Generate", variant="primary", size="lg")
                        unload_btn = gr.Button("Unload All", variant="secondary")
                        load_flux_btn = gr.Button("Load FLUX", variant="secondary")

                with gr.Column():
                    output = gr.Image(label="Generated Image", height=512)
                    info = gr.Textbox(label="Info")
                    status = gr.Textbox(label="Model Status", value="SDXL loaded, FLUX loads on demand")

            gr.Markdown("### Quick Guide:")
            gr.Markdown("- **FLUX** → Best facial likeness (trigger: alexandratitus767)")
            gr.Markdown("- **Pony Realism** → Explicit content, decent likeness")
            gr.Markdown("- **Juggernaut** → Fishing, nature, landscapes, action")

            btn.click(generate, [prompt, negative, model_choice, use_trigger, width, height, steps, guidance, seed], [output, info], api_name="generate")
            unload_btn.click(unload_all, outputs=[status], api_name="unload_models")
            load_flux_btn.click(load_flux, outputs=[status], api_name="load_flux")

        # ============ IMG2IMG TAB ============
        with gr.TabItem("Image to Image"):
            gr.Markdown("### Upload a FLUX image → Transform with Pony/Juggernaut")
            gr.Markdown("Use FLUX for best face, then modify with other models while keeping likeness")

            with gr.Row():
                with gr.Column():
                    img2img_input = gr.Image(label="Upload Image (from FLUX)", type="pil")
                    img2img_prompt = gr.Textbox(label="Prompt", lines=3,
                        placeholder="Describe modifications...")
                    img2img_negative = gr.Textbox(label="Negative Prompt",
                        value="blurry, bad anatomy, deformed, ugly, watermark")

                    img2img_model = gr.Radio(
                        choices=["Pony Realism (Explicit)", "Juggernaut (Nature/Action)"],
                        value="Pony Realism (Explicit)",
                        label="Select Model"
                    )

                    img2img_strength = gr.Slider(0.1, 1.0, value=0.5, step=0.05,
                        label="Strength (0.3=subtle, 0.5=balanced, 0.8=major changes)")

                    with gr.Row():
                        img2img_steps = gr.Slider(10, 50, value=30, step=1, label="Steps")
                        img2img_guidance = gr.Slider(1, 12, value=7.0, step=0.5, label="Guidance")

                    img2img_seed = gr.Number(value=-1, label="Seed (-1=random)")

                    img2img_btn = gr.Button("Transform Image", variant="primary", size="lg")

                with gr.Column():
                    img2img_output = gr.Image(label="Transformed Image", height=512)
                    img2img_info = gr.Textbox(label="Info")

            gr.Markdown("### Tips:")
            gr.Markdown("- **Low strength (0.3-0.4)**: Keep face, change style/lighting")
            gr.Markdown("- **Medium strength (0.5-0.6)**: Modify pose/outfit while keeping likeness")
            gr.Markdown("- **High strength (0.7-0.8)**: Major changes, face may drift")

            img2img_btn.click(generate_img2img,
                [img2img_input, img2img_prompt, img2img_negative, img2img_model, img2img_strength, img2img_steps, img2img_guidance, img2img_seed],
                [img2img_output, img2img_info], api_name="img2img")

print("Launching demo...", flush=True)
demo.launch(server_name="0.0.0.0", server_port=7865, share=False, show_error=True, prevent_thread_lock=True)

# Keep running
import time
while True:
    time.sleep(1)
