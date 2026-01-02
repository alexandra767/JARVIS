#!/usr/bin/env python3
"""
SDXL LoRA Training Script for Alexandra
Simplified DreamBooth-style training using diffusers
"""

import os
import sys
import torch
import gc
import json
from pathlib import Path

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

# Status file for JARVIS to read
STATUS_FILE = "/tmp/lora_training_status.json"

def update_status(step, total, loss, message="Training..."):
    """Update training status for JARVIS to display"""
    status = {
        "step": step,
        "total": total,
        "loss": loss,
        "message": message,
        "progress": round(step / total * 100, 1) if total > 0 else 0
    }
    with open(STATUS_FILE, "w") as f:
        json.dump(status, f)
    print(f"Step {step}/{total} - Loss: {loss:.4f}", flush=True)
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# Training config
CONFIG = {
    "model_path": "/workspace/models/juggernaut-xl-v9",
    "train_data_dir": "/workspace/alexandra-training-images",
    "output_dir": "/workspace/models/alexandra-lora",
    "trigger_word": "alexandra woman",
    "resolution": 1024,
    "train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "learning_rate": 1e-4,
    "max_train_steps": 800,
    "checkpointing_steps": 200,
    "seed": 42,
    "rank": 16,
}

def check_images():
    """Check training images exist"""
    data_dir = Path(CONFIG["train_data_dir"])
    if not data_dir.exists():
        print(f"ERROR: Training directory not found: {data_dir}")
        return False, []

    images = list(data_dir.glob("*.jpg")) + list(data_dir.glob("*.JPG")) + \
             list(data_dir.glob("*.jpeg")) + list(data_dir.glob("*.png"))

    if len(images) == 0:
        print(f"ERROR: No images found in {data_dir}")
        return False, []

    print(f"Found {len(images)} training images")
    return True, images

def create_captions(images):
    """Create caption files for each image"""
    trigger = CONFIG["trigger_word"]
    for img in images:
        caption_file = img.with_suffix(".txt")
        if not caption_file.exists():
            caption = f"a photo of {trigger}, ultra realistic photograph, detailed face"
            caption_file.write_text(caption)
    print(f"Captions ready for {len(images)} images")

class SimpleDataset(Dataset):
    def __init__(self, images, resolution=1024, trigger_word="alexandra woman"):
        self.images = images
        self.trigger = trigger_word
        self.transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        caption_path = img_path.with_suffix(".txt")
        if caption_path.exists():
            caption = caption_path.read_text().strip()
        else:
            caption = f"a photo of {self.trigger}"

        return {"pixel_values": image, "caption": caption}

def train():
    """Run LoRA training with simple loop"""
    from diffusers import StableDiffusionXLPipeline, DDPMScheduler
    from peft import LoraConfig, get_peft_model

    print("\n" + "="*60, flush=True)
    print("SDXL LoRA Training - Alexandra", flush=True)
    print("="*60, flush=True)
    print(f"Model: {CONFIG['model_path']}", flush=True)
    print(f"Training images: {CONFIG['train_data_dir']}", flush=True)
    print(f"Trigger word: '{CONFIG['trigger_word']}'", flush=True)
    print(f"Output: {CONFIG['output_dir']}", flush=True)
    print("="*60 + "\n", flush=True)
    update_status(0, CONFIG['max_train_steps'], 0.0, "Starting training...")

    # Check images
    ok, images = check_images()
    if not ok:
        return False

    create_captions(images)

    # Load base model components separately
    print("Loading SDXL components...")
    from diffusers import AutoencoderKL, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

    # Load VAE (use fp16 variant)
    vae = AutoencoderKL.from_pretrained(
        CONFIG["model_path"], subfolder="vae",
        torch_dtype=torch.float16,
        variant="fp16"
    ).to("cuda")
    vae.requires_grad_(False)

    # Load text encoders (use fp16 variant)
    tokenizer = CLIPTokenizer.from_pretrained(CONFIG["model_path"], subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(CONFIG["model_path"], subfolder="tokenizer_2")

    text_encoder = CLIPTextModel.from_pretrained(
        CONFIG["model_path"], subfolder="text_encoder",
        torch_dtype=torch.float16,
        variant="fp16"
    ).to("cuda")
    text_encoder.requires_grad_(False)

    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        CONFIG["model_path"], subfolder="text_encoder_2",
        torch_dtype=torch.float16,
        variant="fp16"
    ).to("cuda")
    text_encoder_2.requires_grad_(False)

    # Load UNet (use fp16 variant)
    unet = UNet2DConditionModel.from_pretrained(
        CONFIG["model_path"], subfolder="unet",
        torch_dtype=torch.float16,
        variant="fp16"
    ).to("cuda")

    # Load scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(CONFIG["model_path"], subfolder="scheduler")

    print("Applying LoRA to UNet...")

    # Configure and apply LoRA
    lora_config = LoraConfig(
        r=CONFIG["rank"],
        lora_alpha=CONFIG["rank"],
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )

    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    # Optimizer
    params = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=CONFIG["learning_rate"])

    # Dataset
    dataset = SimpleDataset(images, CONFIG["resolution"], CONFIG["trigger_word"])
    dataloader = DataLoader(dataset, batch_size=CONFIG["train_batch_size"], shuffle=True)

    print(f"\nTraining on {len(dataset)} images for {CONFIG['max_train_steps']} steps...")
    print("="*60)

    # Training
    unet.train()
    global_step = 0

    while global_step < CONFIG["max_train_steps"]:
        for batch in dataloader:
            if global_step >= CONFIG["max_train_steps"]:
                break

            # Get pixel values and captions
            pixel_values = batch["pixel_values"].to("cuda", dtype=torch.float16)
            captions = batch["caption"]

            # Encode images to latents
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            # Sample noise
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device="cuda")
            timesteps = timesteps.long()

            # Add noise to latents
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Get text embeddings
            with torch.no_grad():
                # Tokenize
                text_inputs = tokenizer(
                    captions, padding="max_length", max_length=77,
                    truncation=True, return_tensors="pt"
                ).to("cuda")
                text_inputs_2 = tokenizer_2(
                    captions, padding="max_length", max_length=77,
                    truncation=True, return_tensors="pt"
                ).to("cuda")

                # Encode with hidden states
                prompt_embeds = text_encoder(text_inputs.input_ids, output_hidden_states=True)
                prompt_embeds = prompt_embeds.hidden_states[-2]

                prompt_embeds_2 = text_encoder_2(text_inputs_2.input_ids, output_hidden_states=True)
                pooled_prompt_embeds = prompt_embeds_2.text_embeds
                prompt_embeds_2 = prompt_embeds_2.hidden_states[-2]

                # Concatenate
                prompt_embeds = torch.cat([prompt_embeds, prompt_embeds_2], dim=-1)

            # Create time embeddings for SDXL
            original_size = (CONFIG["resolution"], CONFIG["resolution"])
            target_size = (CONFIG["resolution"], CONFIG["resolution"])
            crops_coords_top_left = (0, 0)

            add_time_ids = torch.tensor([
                original_size[0], original_size[1],
                crops_coords_top_left[0], crops_coords_top_left[1],
                target_size[0], target_size[1]
            ]).unsqueeze(0).to("cuda", dtype=torch.float16)

            add_time_ids = add_time_ids.repeat(bsz, 1)

            # Forward pass
            added_cond_kwargs = {"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids}
            model_pred = unet(
                noisy_latents, timesteps, prompt_embeds,
                added_cond_kwargs=added_cond_kwargs
            ).sample

            # Calculate loss
            loss = torch.nn.functional.mse_loss(model_pred.float(), noise.float(), reduction="mean")

            # Backward
            loss.backward()

            # Step optimizer
            if (global_step + 1) % CONFIG["gradient_accumulation_steps"] == 0:
                optimizer.step()
                optimizer.zero_grad()

            global_step += 1

            if global_step % 10 == 0:
                update_status(global_step, CONFIG['max_train_steps'], loss.item())

            # Checkpoint
            if global_step % CONFIG["checkpointing_steps"] == 0:
                ckpt_dir = Path(CONFIG["output_dir"]) / f"checkpoint-{global_step}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                unet.save_pretrained(ckpt_dir)
                print(f"💾 Saved checkpoint: {ckpt_dir}")

    # Save final
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    unet.save_pretrained(output_dir)

    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    print(f"LoRA saved to: {output_dir}")
    print(f"\nTo use in prompts: 'a photo of {CONFIG['trigger_word']} smiling'")
    print("="*60)

    # Cleanup
    del unet, vae, text_encoder, text_encoder_2
    gc.collect()
    torch.cuda.empty_cache()

    return True

if __name__ == "__main__":
    train()
