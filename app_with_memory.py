import gradio as gr
import ollama
import subprocess
import os
import uuid
import re
import glob
from memory_system import AlexandraMemory, extract_facts_from_exchange

# Initialize memory
memory = AlexandraMemory()

AVATAR_IMAGE = os.path.expanduser("~/ComfyUI/output/ComfyUI_00077_.png")

def get_system_prompt(user_input):
    """Build system prompt with relevant memories"""
    base_prompt = "You are Alexandra, a warm and friendly AI companion. Keep responses concise, 1-2 sentences."
    
    # Recall relevant memories
    memories = memory.recall(user_input, n_results=5)
    
    if memories:
        memory_context = "\n".join(memories)
        return f"""{base_prompt}

You have these memories about the user:
{memory_context}

Use these naturally in conversation when relevant - don't list them out."""
    
    return base_prompt

def simple_llm_call(prompt):
    """Simple LLM call for fact extraction"""
    r = ollama.chat(model="gpt-oss:120b", messages=[{"role": "user", "content": prompt}])
    return r['message']['content']

def sanitize_text(text):
    text = text.replace(chr(8212), "-").replace(chr(8211), "-")
    text = re.sub(r"[()\\[\\]{}]", "", text)
    text = re.sub(r"[^\\w\\s.,!?'-]", "", text)
    return text.strip()

def respond(user_input, progress=gr.Progress()):
    if not user_input.strip():
        return "", None
    
    progress(0.1, desc="Remembering...")
    
    # Get system prompt with memories
    system_prompt = get_system_prompt(user_input)
    
    progress(0.2, desc="Thinking...")
    
    # Generate response
    r = ollama.chat(
        model="gpt-oss:120b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]
    )
    response_text = r['message']['content']
    print(f"[AI] {response_text}")
    
    # Save exchange to memory
    memory.save_exchange(user_input, response_text)
    
    # Extract and save facts (background)
    try:
        facts = extract_facts_from_exchange(user_input, response_text, simple_llm_call)
        for fact in facts:
            memory.save_fact(fact)
            print(f"[MEMORY] Saved fact: {fact}")
    except Exception as e:
        print(f"[MEMORY] Fact extraction skipped: {e}")
    
    progress(0.4, desc="Generating voice...")
    
    # Voice synthesis
    clean_text = sanitize_text(response_text)
    voice_out = f"/tmp/voice_{uuid.uuid4().hex[:6]}.wav"
    
    voice_cmd = [
        "python", "-m", "f5_tts.infer",
        "--model", "F5TTS_v1_Base",
        "--ref_audio", os.path.expanduser("~/voice_training/F5-TTS/data/alexandra_char/wavs/clip_001.wav"),
        "--ref_text", "I love about technology, it keeps surprising me. Every time I think I have seen it all, something new comes along that",
        "--gen_text", clean_text,
        "--output", voice_out,
        "--vocoder_name", "vocos",
        "--load_vocoder_from_local",
        "--ckpt_file", os.path.expanduser("~/voice_training/F5-TTS/ckpts/alexandra/model_last.pt")
    ]
    
    subprocess.run(voice_cmd, capture_output=True, cwd=os.path.expanduser("~/voice_training/F5-TTS"))
    print(f"[VOICE] {voice_out}")
    
    progress(0.6, desc="Animating face...")
    
    # Lip sync
    out_dir = f"/tmp/sadtalker_{uuid.uuid4().hex[:6]}"
    os.makedirs(out_dir, exist_ok=True)
    
    sad_cmd = [
        "python", "inference.py",
        "--driven_audio", voice_out,
        "--source_image", AVATAR_IMAGE,
        "--result_dir", out_dir,
        "--enhancer", "gfpgan"
    ]
    
    result = subprocess.run(sad_cmd, capture_output=True, cwd=os.path.expanduser("~/SadTalker"))
    print(f"[SADTALKER] Return code: {result.returncode}")
    
    progress(1.0, desc="Done!")
    
    videos = glob.glob(f"{out_dir}/*.mp4")
    video_path = videos[0] if videos else None
    print(f"[VIDEO] {video_path}")
    
    return response_text, video_path

def show_memory_stats():
    stats = memory.stats()
    recent = memory.get_recent(5)
    return f"Facts: {stats['facts']} | Conversations: {stats['conversations']}\n\nRecent:\n" + "\n".join(recent[:5])

# UI
with gr.Blocks(title="Alexandra AI") as app:
    gr.Markdown("# Alexandra AI Clone")
    
    with gr.Row():
        with gr.Column(scale=2):
            video = gr.Video(label="Alexandra", autoplay=True, height=512)
        with gr.Column(scale=1):
            response_text = gr.Textbox(label="Response", lines=4)
            memory_display = gr.Textbox(label="Memory Stats", lines=6)
    
    user_input = gr.Textbox(label="You:", placeholder="Say something...", lines=2)
    
    with gr.Row():
        submit_btn = gr.Button("Send", variant="primary")
        memory_btn = gr.Button("Show Memories")
    
    submit_btn.click(respond, inputs=[user_input], outputs=[response_text, video])
    user_input.submit(respond, inputs=[user_input], outputs=[response_text, video])
    memory_btn.click(show_memory_stats, outputs=[memory_display])

app.launch(server_name="0.0.0.0", server_port=7860)
