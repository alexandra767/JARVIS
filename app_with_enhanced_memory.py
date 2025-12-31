import gradio as gr
import ollama
import subprocess
import os
import uuid
import re
import glob
from enhanced_memory import EnhancedAlexandraMemory

# Initialize enhanced memory
memory = EnhancedAlexandraMemory()

AVATAR_IMAGE = os.path.expanduser("~/ComfyUI/output/ComfyUI_00077_.png")

# Keywords that trigger web search for current info
CURRENT_EVENT_KEYWORDS = [
    "news", "today", "latest", "recent", "current", "happening",
    "update", "2024", "2025", "election", "president", "stock",
    "weather", "trending", "breaking"
]

def needs_current_info(text):
    """Check if query needs current/live information"""
    text_lower = text.lower()
    return any(kw in text_lower for kw in CURRENT_EVENT_KEYWORDS)

def get_system_prompt(user_input):
    """Build system prompt with relevant memories and context"""
    base_prompt = """You are Alexandra, a warm and friendly AI companion.
You stay informed on current events and can discuss news, politics, technology, and more.
Keep responses conversational and concise (2-3 sentences unless more detail is needed)."""

    # Check if we need current info
    include_web = needs_current_info(user_input)

    # Build context from RAG
    context = memory.build_context(user_input, include_web=include_web)

    if context.strip():
        return f"""{base_prompt}

Here is relevant context to help you respond:
{context}

Use this information naturally in conversation - don't list it out mechanically."""

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

def extract_facts_from_exchange(user_msg, assistant_msg, llm_func):
    """Use LLM to extract facts from conversation"""
    try:
        prompt = f"""Extract any key facts about the user from this exchange.
Return ONLY a JSON list of facts, or empty list if none.
Examples of facts: name, job, pets, preferences, family, hobbies, important events.

User: {user_msg}
Assistant: {assistant_msg}

Return format: ["fact 1", "fact 2"] or []"""

        response = llm_func(prompt)
        import json
        match = re.search(r'\[.*?\]', response, re.DOTALL)
        if match:
            facts = json.loads(match.group())
            return facts if isinstance(facts, list) else []
    except:
        pass
    return []

def respond(user_input, progress=gr.Progress()):
    if not user_input.strip():
        return "", None

    progress(0.1, desc="Gathering context...")

    # Get system prompt with memories + current info if needed
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

def update_news():
    """Manually update news feeds"""
    count = memory.update_news_feeds()
    return f"Updated! Fetched {count} new articles.\n\nTotal news in database: {memory.stats()['news']}"

def show_memory_stats():
    stats = memory.stats()
    return f"""📊 Memory Stats:
• Facts about user: {stats['facts']}
• Conversations: {stats['conversations']}
• Knowledge base: {stats['knowledge']}
• News articles: {stats['news']}
"""

def search_web_now(query):
    """Manual web search"""
    if not query.strip():
        return "Enter a search query"

    results = memory.search_web_duckduckgo(query, max_results=5)
    if not results:
        return "No results found"

    output = f"🔍 Web results for: {query}\n\n"
    for r in results:
        output += f"**{r['title']}**\n{r['snippet']}\n{r['url']}\n\n"
    return output

# UI
with gr.Blocks(title="Alexandra AI", theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🤖 Alexandra AI Clone")
    gr.Markdown("*Now with current events & enhanced memory!*")

    with gr.Tabs():
        with gr.Tab("💬 Chat"):
            with gr.Row():
                with gr.Column(scale=2):
                    video = gr.Video(label="Alexandra", autoplay=True, height=512)
                with gr.Column(scale=1):
                    response_text = gr.Textbox(label="Response", lines=4)
                    memory_display = gr.Textbox(label="Memory Stats", lines=6)

            user_input = gr.Textbox(label="You:", placeholder="Ask me anything - including current events!", lines=2)

            with gr.Row():
                submit_btn = gr.Button("Send", variant="primary")
                memory_btn = gr.Button("📊 Memory Stats")

            submit_btn.click(respond, inputs=[user_input], outputs=[response_text, video])
            user_input.submit(respond, inputs=[user_input], outputs=[response_text, video])
            memory_btn.click(show_memory_stats, outputs=[memory_display])

        with gr.Tab("📰 News & Search"):
            gr.Markdown("### Update news feeds or search the web")

            with gr.Row():
                update_btn = gr.Button("🔄 Update News Feeds", variant="secondary")
                news_status = gr.Textbox(label="Status", lines=2)

            update_btn.click(update_news, outputs=[news_status])

            gr.Markdown("---")

            search_input = gr.Textbox(label="Web Search", placeholder="Search for current information...")
            search_btn = gr.Button("🔍 Search", variant="primary")
            search_results = gr.Markdown(label="Results")

            search_btn.click(search_web_now, inputs=[search_input], outputs=[search_results])

app.launch(server_name="0.0.0.0", server_port=7860)
