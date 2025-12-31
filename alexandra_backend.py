"""Alexandra Backend - Voice/Video on GPU with smaller LLM"""

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn, ollama, json, re, os, uuid, subprocess, glob
from threading import Thread
from memory_system import AlexandraMemory

app = FastAPI()
memory = AlexandraMemory()
ollama_client = ollama.Client(host="http://localhost:11434")

OLLAMA_MODEL = "qwen2.5:72b"  # Changed from gpt-oss:120b
AVATAR_IMAGE = os.path.expanduser("~/ComfyUI/output/ComfyUI_00077_.png")
VIDEO_DIR = "/tmp/alexandra_videos"
F5_TTS_DIR = os.path.expanduser("~/voice_training/F5-TTS")
VENV_PYTHON = os.path.expanduser("~/comfyui-env/bin/python")
os.makedirs(VIDEO_DIR, exist_ok=True)

app.mount("/videos", StaticFiles(directory=VIDEO_DIR), name="videos")
latest_videos = {}

def should_generate_video(text):
    first_100 = text[:100].lower()
    skip_words = ['follow_up', 'title', 'tags', 'summary', 'keywords', 'suggested', '{']
    for word in skip_words:
        if word in first_100:
            return False
    return len(text.strip()) >= 15

def sanitize_for_voice(text):
    text = text.replace('\n', ' ').strip()
    for end in ['. ', '! ', '? ']:
        if end in text[:200]:
            text = text[:text.index(end)+1]
            break
    else:
        text = text[:150]
    text = re.sub(r'```[\s\S]*?```', '', text)
    text = re.sub(r'[^\w\s.,!?\'-]', '', text)
    return text.strip()

def generate_video(text, video_id):
    try:
        if not should_generate_video(text):
            print(f"[VIDEO] Skip: {text[:30]}...")
            return
            
        voice_text = sanitize_for_voice(text)
        if len(voice_text) < 5:
            return
        
        print(f"[VIDEO] Voice ({len(voice_text)} chars): {voice_text[:50]}...")
        voice_out = f"/tmp/voice_{video_id}.wav"
        
        # GPU enabled now!
        voice_cmd = [VENV_PYTHON, "-c", "from f5_tts.infer.infer_cli import main; main()",
            "-m", "F5TTS_v1_Base", "-r", f"{F5_TTS_DIR}/data/alexandra_char/wavs/clip_001.wav",
            "-s", "I love about technology, it keeps surprising me.", "-t", voice_text,
            "-w", voice_out, "-p", f"{F5_TTS_DIR}/ckpts/alexandra/model_last.pt"]
        
        result = subprocess.run(voice_cmd, capture_output=True, cwd=F5_TTS_DIR, timeout=60)
        print(f"[VOICE] Done: {result.returncode}")
        
        if not os.path.exists(voice_out):
            print(f"[VIDEO] Voice failed"); return
        
        print(f"[VIDEO] SadTalker...")
        out_dir = f"/tmp/sadtalker_{video_id}"
        os.makedirs(out_dir, exist_ok=True)
        subprocess.run([VENV_PYTHON, "inference.py", "--driven_audio", voice_out, "--source_image", AVATAR_IMAGE, "--result_dir", out_dir, "--enhancer", "gfpgan"], capture_output=True, cwd=os.path.expanduser("~/SadTalker"), timeout=120)
        
        videos = glob.glob(f"{out_dir}/*.mp4")
        if videos:
            final_path = f"{VIDEO_DIR}/{video_id}.mp4"
            subprocess.run(["cp", videos[0], final_path])
            latest_videos["current"] = video_id
            print(f"[VIDEO] Done: {final_path}")
        else:
            print(f"[VIDEO] No video generated")
    except Exception as e:
        print(f"[VIDEO] Error: {e}")

@app.post("/api/chat")
async def chat(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    user_msg = messages[-1].get("content", "") if messages else ""
    return StreamingResponse(stream_response(messages, user_msg, body.get("model", OLLAMA_MODEL)), media_type="application/x-ndjson")

async def stream_response(messages, user_msg, model):
    full_response = ""
    for chunk in ollama_client.chat(model=model, messages=messages, stream=True):
        full_response += chunk["message"]["content"]
        yield json.dumps({"model": model, "message": {"role": "assistant", "content": chunk["message"]["content"]}, "done": chunk.get("done", False)}) + "\n"
    memory.save_exchange(user_msg, full_response)
    Thread(target=generate_video, args=(full_response, uuid.uuid4().hex[:8]), daemon=True).start()

@app.get("/api/tags")
async def list_models():
    return JSONResponse({"models": [{"name": m.model, "model": m.model, "size": m.size, "digest": m.digest} for m in ollama_client.list().models]})

@app.get("/api/version")
async def version():
    return JSONResponse({"version": "alexandra-1.0"})

@app.get("/latest-video")
async def get_latest_video():
    vid = latest_videos.get("current")
    if vid and os.path.exists(f"{VIDEO_DIR}/{vid}.mp4"):
        return JSONResponse({"video_url": f"/videos/{vid}.mp4", "video_id": vid})
    return JSONResponse({"video_url": None})

@app.get("/video-player")
async def video_player():
    return HTMLResponse("""<!DOCTYPE html><html><head><title>Alexandra</title><style>body{background:#1a1a1a;display:flex;justify-content:center;align-items:center;height:100vh;margin:0}video{max-width:512px;border-radius:12px}</style></head><body><video id="p" autoplay controls></video><script>let l=null;setInterval(async()=>{let r=await fetch('/latest-video');let d=await r.json();if(d.video_url&&d.video_id!==l){l=d.video_id;document.getElementById('p').src=d.video_url}},2000)</script></body></html>""")

if __name__ == "__main__":
    print("="*50 + "\nAlexandra Backend (qwen2.5:72b + GPU voice)\n" + "="*50)
    uvicorn.run(app, host="0.0.0.0", port=5000)
