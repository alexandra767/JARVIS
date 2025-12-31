import gradio as gr
import ollama
import subprocess
import os
import uuid
import re
import glob

SYSTEM_PROMPT = "You are Alexandra AI clone. Keep responses concise, 1-2 sentences."
history = []
AVATAR_IMAGE = os.path.expanduser("~/ComfyUI/output/ComfyUI_00077_.png")

def sanitize_text(text):
    text = text.replace(chr(8212), "-").replace(chr(8211), "-")
    text = re.sub(r"[()\[\]{}]", "", text)
    text = re.sub(r"[^\w\s.,!?-]", "", text)
    return text.strip()

def respond(text, progress=gr.Progress()):
    if not text.strip():
        return "", None
    
    progress(0.1, desc="Thinking...")
    history.append({"role": "user", "content": text})
    r = ollama.chat(model="gpt-oss:120b", messages=[{"role": "system", "content": SYSTEM_PROMPT}] + history, keep_alive=0)
    msg = r["message"]["content"]
    history.append({"role": "assistant", "content": msg})
    print(f"[AI] {msg}")
    
    progress(0.3, desc="Generating voice...")
    safe = sanitize_text(msg)
    audio_file = f"/tmp/voice_{uuid.uuid4().hex[:6]}.wav"
    
    ref_text = "I love about technology, it keeps surprising me. Every time I think I have seen it all, something new comes along that"
    
    cmd = [
        "python", "-m", "f5_tts.infer.infer_cli",
        "--model", "F5TTS_v1_Base",
        "--ckpt_file", os.path.expanduser("~/voice_training/F5-TTS/ckpts/alexandra/model_last.pt"),
        "--vocab_file", os.path.expanduser("~/voice_training/F5-TTS/data/Emilia_ZH_EN_pinyin/vocab.txt"),
        "--ref_audio", os.path.expanduser("~/voice_training/F5-TTS/data/alexandra_char/wavs/clip_001.wav"),
        "--ref_text", ref_text,
        "--gen_text", safe,
        "--output_dir", "/tmp",
        "--output_file", os.path.basename(audio_file)
    ]
    subprocess.run(cmd, capture_output=True, timeout=90, cwd=os.path.expanduser("~/voice_training/F5-TTS"))
    print(f"[VOICE] {audio_file}")
    
    progress(0.5, desc="Creating lip-sync video (~60s)...")
    output_dir = f"/tmp/sadtalker_{uuid.uuid4().hex[:6]}"
    
    sadtalker_cmd = [
        "python", "inference.py",
        "--driven_audio", audio_file,
        "--source_image", AVATAR_IMAGE,
        "--result_dir", output_dir,
        "--still",
        "--preprocess", "crop"
    ]
    
    result = subprocess.run(sadtalker_cmd, capture_output=True, timeout=180, cwd=os.path.expanduser("~/SadTalker"))
    print(f"[SADTALKER] Return code: {result.returncode}")
    
    video_files = glob.glob(f"{output_dir}/*.mp4")
    if video_files:
        print(f"[VIDEO] {video_files[0]}")
        progress(1.0, desc="Done!")
        return msg, video_files[0]
    else:
        print(f"[ERROR] {result.stderr.decode()[-500:]}")
        return msg, None

with gr.Blocks() as demo:
    gr.Markdown("# Alexandra AI Clone")
    with gr.Row():
        video_output = gr.Video(label="Alexandra Speaking", autoplay=True, height=512)
        with gr.Column():
            txt = gr.Textbox(label="You:", lines=2)
            btn = gr.Button("Send", variant="primary", size="lg")
            response = gr.Textbox(label="Alexandra says:", lines=4)
    btn.click(respond, txt, [response, video_output])
    txt.submit(respond, txt, [response, video_output])

demo.launch(server_name="0.0.0.0", server_port=7860)
