"""
Alexandra AI - REST API
FastAPI endpoint for avatar generation
"""

import os
import sys
import uuid
import asyncio
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import aiofiles
import uvicorn
import requests

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from alexandra_config import *
from emotion_detector import detect_emotion
from voice_input import transcribe_audio

app = FastAPI(
    title="Alexandra AI API",
    description="API for generating AI avatar videos",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for video serving
os.makedirs(VIDEO_DIR, exist_ok=True)
app.mount("/videos", StaticFiles(directory=VIDEO_DIR), name="videos")

# ============== Request/Response Models ==============

class ChatRequest(BaseModel):
    message: str
    personality: str = "default"
    response_length: str = "medium"
    use_musetalk: bool = True

class ChatResponse(BaseModel):
    text: str
    video_url: Optional[str]
    emotion: str
    processing_time: float
    job_id: str

class GenerateVideoRequest(BaseModel):
    text: str
    avatar: str = "default"
    emotion: Optional[str] = None

class JobStatus(BaseModel):
    job_id: str
    status: str  # pending, processing, completed, failed
    progress: float
    result: Optional[dict]

class AskRequest(BaseModel):
    message: str
    system_prompt: str = "You are Alexandra, a helpful and friendly AI assistant."

class AskResponse(BaseModel):
    response: str
    model: str

# ============== Ollama Integration ==============

def call_ollama(message: str, system_prompt: str = "", max_tokens: int = 1024) -> str:
    """Call Ollama API for text generation"""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": message})

    try:
        resp = requests.post(
            f"{OLLAMA_HOST}/v1/chat/completions",
            json={
                "model": OLLAMA_MODEL,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.7,
            },
            timeout=120,
        )
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"]
        else:
            return f"Error: Ollama returned {resp.status_code}"
    except requests.exceptions.ConnectionError:
        return "Error: Cannot connect to Ollama. Is it running?"
    except Exception as e:
        return f"Error: {str(e)}"

# ============== Job Queue ==============

jobs = {}

async def process_avatar_job(job_id: str, text: str, avatar: str, emotion: str):
    """Background task to generate avatar video"""
    import subprocess
    import time

    jobs[job_id]["status"] = "processing"
    jobs[job_id]["progress"] = 0.1
    start_time = time.time()

    try:
        session_dir = os.path.join(VIDEO_DIR, job_id)
        os.makedirs(session_dir, exist_ok=True)

        # Step 1: Generate voice
        jobs[job_id]["progress"] = 0.2
        voice_path = os.path.join(session_dir, "voice.wav")

        voice_cmd = [
            "python", "-m", "f5_tts.infer",
            "--model", "F5TTS_v1_Base",
            "--ref_audio", F5_TTS_REF_AUDIO,
            "--ref_text", F5_TTS_REF_TEXT,
            "--gen_text", text,
            "--output", voice_path,
            "--vocoder_name", "vocos",
            "--load_vocoder_from_local",
            "--ckpt_file", F5_TTS_CHECKPOINT
        ]

        subprocess.run(voice_cmd, capture_output=True, cwd=F5_TTS_DIR, timeout=120)

        if not os.path.exists(voice_path):
            raise Exception("Voice generation failed")

        jobs[job_id]["progress"] = 0.5

        # Step 2: Generate lip-synced video
        avatar_image = AVATAR_IMAGES.get(avatar, AVATAR_IMAGES["default"])

        # Use SadTalker (more reliable for now)
        sad_cmd = [
            "python", "inference.py",
            "--driven_audio", voice_path,
            "--source_image", avatar_image,
            "--result_dir", session_dir,
            "--enhancer", "gfpgan"
        ]

        subprocess.run(sad_cmd, capture_output=True, cwd=SADTALKER_DIR, timeout=180)

        jobs[job_id]["progress"] = 0.9

        # Find generated video
        import glob
        videos = glob.glob(f"{session_dir}/*.mp4")
        if not videos:
            raise Exception("Video generation failed")

        # Move to final location
        final_video = os.path.join(VIDEO_DIR, f"{job_id}.mp4")
        os.rename(videos[0], final_video)

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["progress"] = 1.0
        jobs[job_id]["result"] = {
            "video_url": f"/videos/{job_id}.mp4",
            "processing_time": time.time() - start_time
        }

    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["result"] = {"error": str(e)}

# ============== API Endpoints ==============

@app.get("/")
async def root():
    """API info"""
    return {
        "name": "Alexandra AI API",
        "version": "1.0.0",
        "endpoints": {
            "ask": "/ask (simple text - perfect for Siri)",
            "chat": "/chat",
            "generate_video": "/generate-video",
            "transcribe": "/transcribe",
            "job_status": "/job/{job_id}",
            "personalities": "/personalities",
            "avatars": "/avatars"
        }
    }

@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    """
    Simple text-only chat - perfect for Siri Shortcuts / Action Button

    Just send a message, get a response. No video, no bells and whistles.
    """
    response = call_ollama(request.message, request.system_prompt)
    return AskResponse(response=response, model=OLLAMA_MODEL)

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, background_tasks: BackgroundTasks):
    """
    Chat with Alexandra and get video response

    This is an async endpoint - returns a job_id for polling
    """
    import time
    start_time = time.time()

    # Detect emotion from input
    emotion_result = detect_emotion(request.message)

    # Generate LLM response via Ollama
    response_text = call_ollama(request.message)

    # Create job for video generation
    job_id = uuid.uuid4().hex[:12]
    jobs[job_id] = {
        "status": "pending",
        "progress": 0,
        "result": None
    }

    # Start background video generation
    background_tasks.add_task(
        process_avatar_job,
        job_id,
        response_text,
        "default",
        emotion_result["avatar_suggestion"]
    )

    return ChatResponse(
        text=response_text,
        video_url=None,  # Will be available when job completes
        emotion=emotion_result["emotion"],
        processing_time=time.time() - start_time,
        job_id=job_id
    )

@app.post("/generate-video")
async def generate_video(request: GenerateVideoRequest, background_tasks: BackgroundTasks):
    """
    Generate avatar video from text

    Returns job_id for polling status
    """
    # Detect emotion if not provided
    if not request.emotion:
        emotion_result = detect_emotion(request.text)
        emotion = emotion_result["avatar_suggestion"]
    else:
        emotion = request.emotion

    # Create job
    job_id = uuid.uuid4().hex[:12]
    jobs[job_id] = {
        "status": "pending",
        "progress": 0,
        "result": None
    }

    # Start background generation
    background_tasks.add_task(
        process_avatar_job,
        job_id,
        request.text,
        request.avatar,
        emotion
    )

    return {"job_id": job_id, "status": "pending"}

@app.get("/job/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get status of a video generation job"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    return JobStatus(
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        result=job["result"]
    )

@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    """Transcribe audio to text using Whisper"""
    # Save uploaded file
    temp_path = f"/tmp/whisper_{uuid.uuid4().hex[:8]}.wav"

    async with aiofiles.open(temp_path, 'wb') as f:
        content = await audio.read()
        await f.write(content)

    # Transcribe
    result = transcribe_audio(temp_path)

    # Cleanup
    os.unlink(temp_path)

    return result

@app.get("/personalities")
async def list_personalities():
    """List available personality modes"""
    return {
        name: {"name": config["name"], "description": config["system_prompt"][:100] + "..."}
        for name, config in PERSONALITY_MODES.items()
    }

@app.get("/avatars")
async def list_avatars():
    """List available avatar images"""
    return {
        name: {"path": path, "exists": os.path.exists(path)}
        for name, path in AVATAR_IMAGES.items()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "components": {
            "avatar_image": os.path.exists(AVATAR_IMAGES.get("default", "")),
            "voice_model": os.path.exists(F5_TTS_CHECKPOINT),
            "sadtalker": os.path.exists(os.path.join(SADTALKER_DIR, "inference.py")),
            "musetalk": os.path.exists(os.path.join(MUSETALK_DIR, "models/musetalkV15/unet.pth"))
        }
    }

# ============== Batch Processing ==============

class BatchRequest(BaseModel):
    texts: list[str]
    avatar: str = "default"

@app.post("/batch-generate")
async def batch_generate(request: BatchRequest, background_tasks: BackgroundTasks):
    """Generate multiple videos from a list of texts"""
    job_ids = []

    for text in request.texts:
        job_id = uuid.uuid4().hex[:12]
        jobs[job_id] = {
            "status": "pending",
            "progress": 0,
            "result": None
        }

        emotion_result = detect_emotion(text)
        background_tasks.add_task(
            process_avatar_job,
            job_id,
            text,
            request.avatar,
            emotion_result["avatar_suggestion"]
        )

        job_ids.append(job_id)

    return {"job_ids": job_ids, "count": len(job_ids)}

@app.get("/batch-status")
async def batch_status(job_ids: str):
    """Get status of multiple jobs (comma-separated IDs)"""
    ids = job_ids.split(",")
    results = {}

    for job_id in ids:
        if job_id in jobs:
            results[job_id] = jobs[job_id]
        else:
            results[job_id] = {"status": "not_found"}

    return results


if __name__ == "__main__":
    print("="*50)
    print("Alexandra AI API Server")
    print("="*50)
    print(f"Starting on http://{API_HOST}:{API_PORT}")
    print(f"Docs available at http://{API_HOST}:{API_PORT}/docs")
    print("="*50)

    uvicorn.run(app, host=API_HOST, port=API_PORT)
