"""
Alexandra AI - Full Featured Application
Complete avatar system with all features
"""

import gradio as gr
import os
import sys
import uuid
import time
import json
import subprocess
import glob
import re
from datetime import datetime
from pathlib import Path

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from alexandra_config import *
from emotion_detector import detect_emotion, AdvancedEmotionDetector
from voice_input import VoiceInput, transcribe_audio

# Try to import memory system
try:
    from enhanced_memory import EnhancedAlexandraMemory
    memory = EnhancedAlexandraMemory()
    HAS_MEMORY = True
except ImportError:
    memory = None
    HAS_MEMORY = False
    print("[WARNING] Memory system not available")

# Initialize components
emotion_detector = AdvancedEmotionDetector()
voice_input = None  # Lazy load

# Conversation history
conversation_history = []

# ============== Core Functions ==============

def sanitize_text(text):
    """Clean text for TTS"""
    text = text.replace(chr(8212), "-").replace(chr(8211), "-")
    text = re.sub(r"[()\\[\\]{}]", "", text)
    text = re.sub(r"[^\w\s.,!?'-]", "", text)
    return text.strip()

def get_llm_response(user_input, personality="default", response_length="medium"):
    """Get response from LLM via Ollama"""
    try:
        import ollama

        # Build system prompt
        system_prompt = PERSONALITY_MODES.get(personality, PERSONALITY_MODES["default"])["system_prompt"]
        length_instruction = RESPONSE_LENGTHS.get(response_length, "")

        if length_instruction:
            system_prompt += f"\n\n{length_instruction}"

        # Add memory context if available
        if HAS_MEMORY:
            context = memory.build_context(user_input, include_web=False)
            if context:
                system_prompt += f"\n\nRelevant context:\n{context}"

        # Call Ollama
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ]
        )

        return response['message']['content']

    except Exception as e:
        # Fallback response
        return f"I received your message: '{user_input}'. (LLM unavailable: {str(e)[:50]})"

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

    try:
        result = subprocess.run(voice_cmd, capture_output=True, cwd=F5_TTS_DIR, timeout=120)
        if os.path.exists(output_path):
            return output_path
    except Exception as e:
        print(f"[VOICE] Error: {e}")

    return None

def generate_lipsync(audio_path, image_path, output_dir, use_musetalk=False):
    """Generate lip-synced video"""
    os.makedirs(output_dir, exist_ok=True)

    if use_musetalk:
        # MuseTalk (faster but may need debugging)
        # For now, fall back to SadTalker
        pass

    # SadTalker (reliable)
    sad_cmd = [
        "python", "inference.py",
        "--driven_audio", audio_path,
        "--source_image", image_path,
        "--result_dir", output_dir,
        "--enhancer", "gfpgan"
    ]

    try:
        result = subprocess.run(
            sad_cmd, capture_output=True,
            cwd=SADTALKER_DIR, timeout=180
        )

        videos = glob.glob(f"{output_dir}/*.mp4")
        return videos[0] if videos else None
    except Exception as e:
        print(f"[LIPSYNC] Error: {e}")
        return None

# ============== Main Chat Function ==============

def chat(user_input, personality, response_length, generate_video, use_musetalk, progress=gr.Progress()):
    """Main chat function"""
    global conversation_history

    if not user_input.strip():
        return "", None, "", conversation_history

    session_id = uuid.uuid4().hex[:8]
    session_dir = os.path.join(OUTPUT_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)

    status_log = []
    start_time = time.time()

    try:
        # Step 1: Get LLM response
        progress(0.1, desc="Thinking...")
        response_text = get_llm_response(user_input, personality, response_length)
        status_log.append(f"[LLM] {time.time() - start_time:.1f}s")

        # Detect emotion
        emotion = detect_emotion(response_text)
        status_log.append(f"[EMOTION] {emotion['emotion']} ({emotion['confidence']:.0%})")

        # Save to memory
        if HAS_MEMORY:
            memory.save_exchange(user_input, response_text)

        # Add to conversation history
        timestamp = datetime.now().strftime("%H:%M")
        conversation_history.append({
            "time": timestamp,
            "user": user_input,
            "assistant": response_text,
            "emotion": emotion['emotion']
        })

        # Keep only last 20 exchanges
        if len(conversation_history) > 20:
            conversation_history = conversation_history[-20:]

        video_path = None

        if generate_video:
            # Step 2: Generate voice
            progress(0.3, desc="Generating voice...")
            voice_start = time.time()
            voice_path = os.path.join(session_dir, "voice.wav")
            voice_result = generate_voice(response_text, voice_path)

            if voice_result:
                status_log.append(f"[VOICE] {time.time() - voice_start:.1f}s")

                # Step 3: Generate lip-synced video
                progress(0.5, desc="Animating avatar...")
                lipsync_start = time.time()

                # Select avatar based on emotion
                avatar_key = emotion.get('avatar_suggestion', 'default')
                avatar_image = AVATAR_IMAGES.get(avatar_key, AVATAR_IMAGES['default'])

                video_path = generate_lipsync(
                    voice_path, avatar_image, session_dir, use_musetalk
                )

                if video_path:
                    status_log.append(f"[VIDEO] {time.time() - lipsync_start:.1f}s")
                else:
                    status_log.append("[VIDEO] Failed")
            else:
                status_log.append("[VOICE] Failed")

        progress(1.0, desc="Done!")
        total_time = time.time() - start_time
        status_log.append(f"[TOTAL] {total_time:.1f}s")

        # Format history for display
        history_display = format_history(conversation_history)

        return response_text, video_path, "\n".join(status_log), history_display

    except Exception as e:
        return f"Error: {str(e)}", None, str(e), format_history(conversation_history)

def format_history(history):
    """Format conversation history for display"""
    if not history:
        return "No conversation history yet."

    output = ""
    for item in history:
        output += f"**[{item['time']}]** You: {item['user'][:100]}{'...' if len(item['user']) > 100 else ''}\n"
        output += f"**Alexandra** ({item['emotion']}): {item['assistant'][:150]}{'...' if len(item['assistant']) > 150 else ''}\n\n"

    return output

# ============== Voice Input ==============

def process_voice_input(audio_path, personality, response_length, generate_video, use_musetalk, progress=gr.Progress()):
    """Process voice input"""
    global voice_input

    if audio_path is None:
        return "", None, "No audio recorded", format_history(conversation_history)

    # Lazy load voice input
    if voice_input is None:
        progress(0.05, desc="Loading Whisper...")
        voice_input = VoiceInput(WHISPER_MODEL)

    # Transcribe
    progress(0.1, desc="Transcribing...")
    result = voice_input.transcribe(audio_path)

    if result.get("error"):
        return "", None, f"Transcription error: {result['error']}", format_history(conversation_history)

    transcribed_text = result.get("text", "")
    if not transcribed_text:
        return "", None, "Could not understand audio", format_history(conversation_history)

    # Process as normal chat
    return chat(transcribed_text, personality, response_length, generate_video, use_musetalk, progress)

# ============== Batch Video Generation ==============

def batch_generate(script_text, progress=gr.Progress()):
    """Generate multiple videos from a script (one per line)"""
    lines = [l.strip() for l in script_text.strip().split("\n") if l.strip()]

    if not lines:
        return "No lines to process", []

    results = []
    total = len(lines)

    for i, line in enumerate(lines):
        progress((i + 1) / total, desc=f"Processing {i+1}/{total}...")

        session_id = uuid.uuid4().hex[:8]
        session_dir = os.path.join(OUTPUT_DIR, f"batch_{session_id}")
        os.makedirs(session_dir, exist_ok=True)

        # Generate voice
        voice_path = os.path.join(session_dir, "voice.wav")
        voice_result = generate_voice(line, voice_path)

        if voice_result:
            # Generate video
            video_path = generate_lipsync(
                voice_path, AVATAR_IMAGES['default'], session_dir, False
            )

            if video_path:
                results.append({
                    "text": line,
                    "video": video_path,
                    "status": "success"
                })
            else:
                results.append({"text": line, "status": "video_failed"})
        else:
            results.append({"text": line, "status": "voice_failed"})

    # Summary
    success = len([r for r in results if r.get("status") == "success"])
    summary = f"Generated {success}/{total} videos"

    # Return video paths
    videos = [r.get("video") for r in results if r.get("video")]

    return summary, videos

# ============== Utility Functions ==============

def clear_history():
    """Clear conversation history"""
    global conversation_history
    conversation_history = []
    return "History cleared", ""

def get_memory_stats():
    """Get memory system stats"""
    if HAS_MEMORY:
        stats = memory.stats()
        return f"""**Memory Stats:**
- Facts: {stats.get('facts', 0)}
- Conversations: {stats.get('conversations', 0)}
- Knowledge: {stats.get('knowledge', 0)}
- News: {stats.get('news', 0)}"""
    return "Memory system not available"

def test_components():
    """Test all components"""
    tests = []

    # Avatar
    for name, path in AVATAR_IMAGES.items():
        status = "Found" if os.path.exists(path) else "MISSING"
        tests.append(f"Avatar ({name}): {status}")

    # Voice model
    status = "Found" if os.path.exists(F5_TTS_CHECKPOINT) else "MISSING"
    tests.append(f"F5-TTS Model: {status}")

    # SadTalker
    status = "Found" if os.path.exists(os.path.join(SADTALKER_DIR, "inference.py")) else "MISSING"
    tests.append(f"SadTalker: {status}")

    # MuseTalk
    status = "Found" if os.path.exists(os.path.join(MUSETALK_DIR, "models/musetalkV15/unet.pth")) else "MISSING"
    tests.append(f"MuseTalk 1.5: {status}")

    # Memory
    tests.append(f"Memory System: {'Available' if HAS_MEMORY else 'Not Available'}")

    return "\n".join(tests)

# ============== Gradio UI ==============

with gr.Blocks(title="Alexandra AI - Full", theme=gr.themes.Soft(), css="""
    .status-box { font-family: monospace; font-size: 12px; }
    .history-box { max-height: 400px; overflow-y: auto; }
""") as app:

    gr.Markdown("# Alexandra AI - Full Featured")
    gr.Markdown("*Your AI companion with voice, video, and memory*")

    with gr.Tabs():
        # ============== Chat Tab ==============
        with gr.Tab("Chat"):
            with gr.Row():
                with gr.Column(scale=2):
                    video_output = gr.Video(label="Alexandra", autoplay=True, height=400)
                    response_output = gr.Textbox(label="Response", lines=3)

                with gr.Column(scale=1):
                    status_output = gr.Textbox(label="Status", lines=8, elem_classes=["status-box"])
                    with gr.Accordion("Settings", open=False):
                        personality = gr.Dropdown(
                            choices=list(PERSONALITY_MODES.keys()),
                            value="default",
                            label="Personality"
                        )
                        response_length = gr.Dropdown(
                            choices=["short", "medium", "long"],
                            value="medium",
                            label="Response Length"
                        )
                        generate_video = gr.Checkbox(label="Generate Video", value=True)
                        use_musetalk = gr.Checkbox(label="Use MuseTalk (faster)", value=False)

            # Text input
            text_input = gr.Textbox(label="Type your message:", placeholder="Hello Alexandra!", lines=2)

            with gr.Row():
                send_btn = gr.Button("Send", variant="primary")
                clear_btn = gr.Button("Clear")

            # Voice input
            with gr.Accordion("Voice Input", open=False):
                audio_input = gr.Audio(sources=["microphone"], type="filepath", label="Record your voice")
                voice_btn = gr.Button("Send Voice")

            send_btn.click(
                chat,
                inputs=[text_input, personality, response_length, generate_video, use_musetalk],
                outputs=[response_output, video_output, status_output, gr.State()]
            )

            text_input.submit(
                chat,
                inputs=[text_input, personality, response_length, generate_video, use_musetalk],
                outputs=[response_output, video_output, status_output, gr.State()]
            )

            voice_btn.click(
                process_voice_input,
                inputs=[audio_input, personality, response_length, generate_video, use_musetalk],
                outputs=[response_output, video_output, status_output, gr.State()]
            )

            clear_btn.click(lambda: ("", None, ""), outputs=[text_input, video_output, status_output])

        # ============== History Tab ==============
        with gr.Tab("History"):
            history_display = gr.Markdown(elem_classes=["history-box"])
            with gr.Row():
                refresh_history_btn = gr.Button("Refresh")
                clear_history_btn = gr.Button("Clear History", variant="stop")

            refresh_history_btn.click(
                lambda: format_history(conversation_history),
                outputs=[history_display]
            )
            clear_history_btn.click(clear_history, outputs=[status_output, history_display])

        # ============== Batch Video Tab ==============
        with gr.Tab("Batch Video"):
            gr.Markdown("### Generate multiple videos from a script")
            gr.Markdown("Enter one line per video to generate:")

            script_input = gr.Textbox(
                label="Script",
                placeholder="Hello, welcome to my channel!\nToday we'll be talking about AI.\nThanks for watching!",
                lines=10
            )
            batch_btn = gr.Button("Generate All Videos", variant="primary")
            batch_status = gr.Textbox(label="Status")
            batch_gallery = gr.Gallery(label="Generated Videos", columns=3)

            batch_btn.click(
                batch_generate,
                inputs=[script_input],
                outputs=[batch_status, batch_gallery]
            )

        # ============== Memory Tab ==============
        with gr.Tab("Memory"):
            memory_stats = gr.Markdown()
            refresh_memory_btn = gr.Button("Refresh Stats")
            refresh_memory_btn.click(get_memory_stats, outputs=[memory_stats])

        # ============== System Tab ==============
        with gr.Tab("System"):
            system_status = gr.Textbox(label="Component Status", lines=10)
            test_btn = gr.Button("Test Components")
            test_btn.click(test_components, outputs=[system_status])

            gr.Markdown("### Personality Modes")
            for name, config in PERSONALITY_MODES.items():
                gr.Markdown(f"**{config['name']}**: {config['system_prompt'][:100]}...")

    # Load initial status
    app.load(test_components, outputs=[system_status])

# ============== Main ==============

if __name__ == "__main__":
    print("="*60)
    print("Alexandra AI - Full Featured Application")
    print("="*60)
    print(test_components())
    print("="*60)
    print("Starting on http://0.0.0.0:7862")
    print("="*60)

    app.launch(server_name="0.0.0.0", server_port=7862)
