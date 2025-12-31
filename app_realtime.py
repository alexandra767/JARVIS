"""
Real-Time Alexandra AI Avatar
Uses MuseTalk 1.5 for faster lip sync + F5-TTS voice
"""

import gradio as gr
import subprocess
import os
import sys
import uuid
import re
import glob
import time
import shutil

# Add MuseTalk to path
MUSETALK_DIR = os.path.expanduser("~/MuseTalk")
sys.path.insert(0, MUSETALK_DIR)

# Paths
AVATAR_IMAGE = os.path.expanduser("~/ComfyUI/output/ComfyUI_00077_.png")
F5_TTS_DIR = os.path.expanduser("~/voice_training/F5-TTS")
F5_TTS_CHECKPOINT = os.path.join(F5_TTS_DIR, "ckpts/alexandra/model_last.pt")
F5_TTS_REF_AUDIO = os.path.join(F5_TTS_DIR, "data/alexandra_char/wavs/clip_001.wav")
F5_TTS_REF_TEXT = "I love about technology, it keeps surprising me."
MUSETALK_VENV = os.path.expanduser("~/MuseTalk/venv/bin/python")
OUTPUT_DIR = "/tmp/alexandra_realtime"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def sanitize_text(text):
    """Clean text for TTS"""
    text = text.replace(chr(8212), "-").replace(chr(8211), "-")
    text = re.sub(r"[()\\[\\]{}]", "", text)
    text = re.sub(r"[^\w\s.,!?'-]", "", text)
    return text.strip()

def generate_voice(text, output_path):
    """Generate voice using F5-TTS"""
    clean_text = sanitize_text(text)
    if len(clean_text) < 5:
        return None

    voice_cmd = [
        "python", "-m", "f5_tts.infer",
        "--model", "F5TTS_v1_Base",
        "--ref_audio", F5_TTS_REF_AUDIO,
        "--ref_text", F5_TTS_REF_TEXT,
        "--gen_text", clean_text,
        "--output", output_path,
        "--vocoder_name", "vocos",
        "--load_vocoder_from_local",
        "--ckpt_file", F5_TTS_CHECKPOINT
    ]

    result = subprocess.run(voice_cmd, capture_output=True, cwd=F5_TTS_DIR, timeout=120)

    if os.path.exists(output_path):
        return output_path
    return None

def generate_lipsync_musetalk(audio_path, image_path, output_dir):
    """Generate lip-synced video using MuseTalk 1.5"""

    # Run MuseTalk inference
    musetalk_cmd = [
        MUSETALK_VENV, "-c",
        f"""
import sys
sys.path.insert(0, '{MUSETALK_DIR}')
os.chdir('{MUSETALK_DIR}')

import os
import torch
import numpy as np
import cv2
import glob
import imageio
from argparse import Namespace
from omegaconf import OmegaConf
import subprocess

# Set paths
audio_path = '{audio_path}'
image_path = '{image_path}'
output_dir = '{output_dir}'

# Load configs
from musetalk.utils.utils import get_file_type, get_video_fps, datagen, load_all_model
from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs, coord_placeholder
from musetalk.utils.blending import get_image
from musetalk.utils.face_parsing import FaceParsing

# Load models
audio_processor, vae, unet, pe = load_all_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weight_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
timesteps = torch.tensor([0], device=device)

# Read image
frame = cv2.imread(image_path)
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Get face coords
coord_list, frame_list = get_landmark_and_bbox([image_path], 0)
bbox = coord_list[0]

if bbox == coord_placeholder:
    print("ERROR: No face detected")
    exit(1)

# Process audio
from musetalk.whisper.whisper import load_model
whisper = load_model('tiny', '{MUSETALK_DIR}/models/whisper', device)
whisper_input_features, librosa_length = audio_processor.get_audio_feature(audio_path)
whisper_chunks = audio_processor.get_whisper_chunk(
    whisper_input_features, device, weight_dtype, whisper, librosa_length, fps=25
)

# Prepare face
fp = FaceParsing()
x1, y1, x2, y2 = bbox
crop_frame = frame[y1:y2, x1:x2]
crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
latents = vae.get_latents_for_unet(crop_frame)

# Generate frames
os.makedirs(output_dir, exist_ok=True)
batch_size = 8
video_num = len(whisper_chunks)
gen = datagen(whisper_chunks, [latents]*video_num, batch_size, 0, device)

res_frames = []
for whisper_batch, latent_batch in gen:
    audio_feature = pe(whisper_batch)
    latent_batch = latent_batch.to(dtype=weight_dtype)
    pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature).sample
    recon = vae.decode_latents(pred_latents)
    res_frames.extend(recon)

# Write frames and create video
for i, res_frame in enumerate(res_frames):
    res_frame = cv2.resize(res_frame.astype(np.uint8), (x2-x1, y2-y1))
    combine_frame = get_image(frame.copy(), res_frame, [x1, y1, x2, y2], mode='jaw', fp=fp)
    cv2.imwrite(f'{output_dir}/{{i:08d}}.png', combine_frame)

# Create video
output_video = f'{output_dir}/output.mp4'
subprocess.run([
    'ffmpeg', '-y', '-framerate', '25',
    '-i', f'{output_dir}/%08d.png',
    '-i', audio_path,
    '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
    '-c:a', 'aac', '-shortest',
    output_video
], capture_output=True)

print(f"SUCCESS: {{output_video}}")
"""
    ]

    result = subprocess.run(musetalk_cmd, capture_output=True, timeout=180)
    output_video = os.path.join(output_dir, "output.mp4")

    if os.path.exists(output_video):
        return output_video
    return None

def generate_lipsync_sadtalker(audio_path, image_path, output_dir):
    """Fallback to SadTalker if MuseTalk fails"""
    sad_cmd = [
        "python", "inference.py",
        "--driven_audio", audio_path,
        "--source_image", image_path,
        "--result_dir", output_dir,
        "--enhancer", "gfpgan"
    ]

    result = subprocess.run(
        sad_cmd, capture_output=True,
        cwd=os.path.expanduser("~/SadTalker"),
        timeout=180
    )

    videos = glob.glob(f"{output_dir}/*.mp4")
    return videos[0] if videos else None

def respond(user_input, use_musetalk=True, progress=gr.Progress()):
    """Generate avatar response"""
    if not user_input.strip():
        return "", None, "Please enter a message"

    session_id = uuid.uuid4().hex[:8]
    session_dir = os.path.join(OUTPUT_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)

    status_log = []

    try:
        # Step 1: Generate LLM response (placeholder - integrate with your LLM)
        progress(0.1, desc="Generating response...")
        response_text = f"Hello! You said: {user_input}. This is a test response from Alexandra."
        status_log.append(f"[LLM] Generated response: {len(response_text)} chars")

        # Step 2: Generate voice
        progress(0.3, desc="Generating voice...")
        start_time = time.time()
        voice_path = os.path.join(session_dir, "voice.wav")
        voice_result = generate_voice(response_text, voice_path)
        voice_time = time.time() - start_time

        if not voice_result:
            return response_text, None, "Voice generation failed"
        status_log.append(f"[VOICE] Generated in {voice_time:.1f}s")

        # Step 3: Generate lip-synced video
        progress(0.5, desc="Animating avatar...")
        start_time = time.time()

        if use_musetalk:
            video_path = generate_lipsync_musetalk(voice_path, AVATAR_IMAGE, session_dir)
            method = "MuseTalk"
        else:
            video_path = generate_lipsync_sadtalker(voice_path, AVATAR_IMAGE, session_dir)
            method = "SadTalker"

        lipsync_time = time.time() - start_time

        if not video_path:
            # Fallback
            if use_musetalk:
                status_log.append("[LIPSYNC] MuseTalk failed, trying SadTalker...")
                video_path = generate_lipsync_sadtalker(voice_path, AVATAR_IMAGE, session_dir)
                method = "SadTalker (fallback)"
                lipsync_time = time.time() - start_time

        if not video_path:
            return response_text, None, "\n".join(status_log) + "\n[ERROR] Lip sync failed"

        status_log.append(f"[LIPSYNC] {method} completed in {lipsync_time:.1f}s")

        progress(1.0, desc="Done!")
        total_time = voice_time + lipsync_time
        status_log.append(f"[TOTAL] {total_time:.1f}s")

        return response_text, video_path, "\n".join(status_log)

    except Exception as e:
        return "", None, f"Error: {str(e)}"

def test_components():
    """Test that all components are available"""
    tests = []

    # Check avatar image
    if os.path.exists(AVATAR_IMAGE):
        tests.append(f"Avatar image: Found")
    else:
        tests.append(f"Avatar image: MISSING ({AVATAR_IMAGE})")

    # Check F5-TTS
    if os.path.exists(F5_TTS_CHECKPOINT):
        tests.append(f"F5-TTS voice model: Found")
    else:
        tests.append(f"F5-TTS voice model: MISSING")

    # Check MuseTalk
    musetalk_model = os.path.join(MUSETALK_DIR, "models/musetalkV15/unet.pth")
    if os.path.exists(musetalk_model):
        tests.append(f"MuseTalk 1.5 model: Found")
    else:
        tests.append(f"MuseTalk 1.5 model: MISSING")

    # Check SadTalker
    sadtalker_path = os.path.expanduser("~/SadTalker/inference.py")
    if os.path.exists(sadtalker_path):
        tests.append(f"SadTalker (fallback): Found")
    else:
        tests.append(f"SadTalker (fallback): MISSING")

    return "\n".join(tests)

# Gradio UI
with gr.Blocks(title="Alexandra AI - Real-Time Avatar", theme=gr.themes.Soft()) as app:
    gr.Markdown("# Alexandra AI - Real-Time Avatar")
    gr.Markdown("*Using MuseTalk 1.5 for faster lip sync*")

    with gr.Row():
        with gr.Column(scale=2):
            video = gr.Video(label="Alexandra", autoplay=True, height=512)
        with gr.Column(scale=1):
            response_text = gr.Textbox(label="Response", lines=4)
            status = gr.Textbox(label="Status", lines=6)

    user_input = gr.Textbox(
        label="You:",
        placeholder="Type your message...",
        lines=2
    )

    with gr.Row():
        use_musetalk = gr.Checkbox(label="Use MuseTalk (faster)", value=True)
        submit_btn = gr.Button("Send", variant="primary")
        test_btn = gr.Button("Test Components")

    submit_btn.click(
        respond,
        inputs=[user_input, use_musetalk],
        outputs=[response_text, video, status]
    )
    user_input.submit(
        respond,
        inputs=[user_input, use_musetalk],
        outputs=[response_text, video, status]
    )
    test_btn.click(test_components, outputs=[status])

if __name__ == "__main__":
    print("="*50)
    print("Alexandra AI - Real-Time Avatar")
    print("="*50)
    print(test_components())
    print("="*50)
    app.launch(server_name="0.0.0.0", server_port=7861)
