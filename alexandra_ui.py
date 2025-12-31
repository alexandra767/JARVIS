"""
Alexandra AI - Full Featured Chat Interface
Features: LLM + RAG + Voice + Avatar + Documents + Code Execution + Image Gen
          + Streaming + Wake Word + Auto News + Reminders + Mood Detection
          + Personas + Smart Home + Email + GitHub + Image Editing
"""

import gradio as gr
import torch
import subprocess
import os
import sys
import uuid
import re
import glob
import json
import time
import tempfile
import traceback
from datetime import datetime
from pathlib import Path

# Add paths
sys.path.insert(0, os.path.dirname(__file__))

from enhanced_memory import EnhancedAlexandraMemory
from voice_input import VoiceInput

# Import extended features
try:
    from alexandra_features import (
        AlexandraFeatures, generate_response_streaming,
        MoodDetector, PersonaManager, CodeExecutor,
        ReminderSystem, ImageEditor, FactExtractor
    )
    FEATURES_AVAILABLE = True
    print("Extended features loaded!")
except ImportError as e:
    FEATURES_AVAILABLE = False
    print(f"Extended features not available: {e}")

# Web search for fallback
try:
    from duckduckgo_search import DDGS
    WEB_SEARCH_AVAILABLE = True
except ImportError:
    WEB_SEARCH_AVAILABLE = False
    print("Note: Install duckduckgo-search for web fallback")

# RAG System
try:
    from alexandra_rag import AlexandraRAG, get_rag
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    print("Warning: RAG system not available")

# Vision Model for Image Understanding (CPU mode)
vision_model = None
vision_processor = None
VISION_AVAILABLE = False

def init_vision():
    """Initialize vision model (lazy loading, CPU only)"""
    global vision_model, vision_processor, VISION_AVAILABLE
    if vision_model is not None:
        return True
    try:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU
        from transformers import BlipProcessor, BlipForConditionalGeneration
        from PIL import Image

        print("Loading vision model (BLIP)...")
        vision_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        vision_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to("cpu")
        VISION_AVAILABLE = True
        print("Vision model loaded!")
        return True
    except Exception as e:
        print(f"Vision model not available: {e}")
        return False

def analyze_image(image_path, question=None):
    """Analyze an image and return description or answer"""
    if not init_vision():
        return "Vision model not available. Please install: pip install transformers Pillow"

    try:
        from PIL import Image
        image = Image.open(image_path).convert("RGB")

        if question:
            # Visual Q&A mode
            inputs = vision_processor(image, question, return_tensors="pt")
        else:
            # Captioning mode
            inputs = vision_processor(image, return_tensors="pt")

        outputs = vision_model.generate(**inputs, max_new_tokens=100)
        caption = vision_processor.decode(outputs[0], skip_special_tokens=True)

        return caption
    except Exception as e:
        return f"Error analyzing image: {e}"

# ============ CONFIGURATION ============
MODELS = {
    "Alexandra (Personal)": {
        "base": "/workspace/models/Qwen2.5-72B-Instruct",
        "lora": "/workspace/ai-clone-training/my-output/alexandra-personal-lora-final",
    },
    "Alexandra (Original)": {
        "base": "/workspace/models/Qwen2.5-72B-Instruct",
        "lora": "/workspace/ai-clone-training/my-output/alexandra-qwen72b-lora-final",
    },
    "Qwen 72B (Base)": {
        "base": "/workspace/models/Qwen2.5-72B-Instruct",
        "lora": None,
    },
}

# Check which models exist
AVAILABLE_MODELS = {}
for name, paths in MODELS.items():
    if os.path.exists(paths["base"]):
        if paths["lora"] is None or os.path.exists(paths["lora"]):
            AVAILABLE_MODELS[name] = paths

DEFAULT_MODEL = list(AVAILABLE_MODELS.keys())[0] if AVAILABLE_MODELS else None

AVATAR_IMAGE = "/workspace/ai-clone-chat/assets/avatar_headshot.png"
F5_TTS_DIR = os.path.expanduser("~/voice_training/F5-TTS")
F5_TTS_CHECKPOINT = os.path.join(F5_TTS_DIR, "ckpts/jenny_british/model_last.pt")
F5_TTS_REF_AUDIO = os.path.join(F5_TTS_DIR, "data/jenny_british/wavs/clip_0000.wav")
F5_TTS_REF_TEXT = "It was a bright cold day in April, and the clocks were striking thirteen."
CONVERSATIONS_DIR = os.path.expanduser("~/ai-clone-chat/conversations")
UPLOADS_DIR = os.path.expanduser("~/ai-clone-chat/uploads")
FLUX_LORA = "/root/ComfyUI/models/loras/flux-uncensored.safetensors"

os.makedirs(CONVERSATIONS_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

# ============ GLOBALS ============
model = None
tokenizer = None
current_model_name = None
memory = None
whisper_model = None
current_conversation_id = None
uploaded_documents = {}  # Store uploaded document contents

# MuseTalk models (loaded once for fast avatar generation)
musetalk_vae = None
musetalk_unet = None
musetalk_pe = None
musetalk_whisper = None
musetalk_audio_processor = None
musetalk_loaded = False

# ============ MODEL LOADING ============
def load_model(model_name=None):
    """Load selected model"""
    global model, tokenizer, current_model_name

    if model_name is None:
        model_name = DEFAULT_MODEL

    # If no local models available, use Ollama
    if not AVAILABLE_MODELS:
        return "✅ Using Ollama (qwen2.5:72b) for chat - no local model needed!"

    if model_name not in AVAILABLE_MODELS:
        return f"✅ Using Ollama for chat. Local model '{model_name}' not found."

    if current_model_name == model_name and model is not None:
        return f"Model {model_name} already loaded"

    # Unload current model
    if model is not None:
        del model
        del tokenizer
        torch.cuda.empty_cache()

    print(f"Loading {model_name}...")
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    paths = AVAILABLE_MODELS[model_name]

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # GB10 unified memory - load directly to CUDA
    model = AutoModelForCausalLM.from_pretrained(
        paths["base"],
        quantization_config=bnb_config,
        device_map={"": 0},  # Force all to GPU 0
        torch_dtype=torch.bfloat16,
    )

    if paths["lora"]:
        model = PeftModel.from_pretrained(model, paths["lora"])
        tokenizer = AutoTokenizer.from_pretrained(paths["lora"])
    else:
        tokenizer = AutoTokenizer.from_pretrained(paths["base"])

    current_model_name = model_name
    print(f"Loaded {model_name}")
    return f"Loaded: {model_name}"

def load_musetalk_models():
    """Load MuseTalk models for GPU-accelerated avatar generation"""
    global musetalk_vae, musetalk_unet, musetalk_pe, musetalk_whisper, musetalk_audio_processor, musetalk_loaded

    if musetalk_loaded:
        return "MuseTalk already loaded"

    MUSETALK_DIR = os.path.expanduser("~/MuseTalk")
    sys.path.insert(0, MUSETALK_DIR)

    # Save current directory and change to MuseTalk dir (load_all_model uses relative paths)
    original_dir = os.getcwd()

    try:
        os.chdir(MUSETALK_DIR)
        print(f"Loading MuseTalk models from {MUSETALK_DIR}...")

        from musetalk.utils.utils import load_all_model
        from musetalk.utils.audio_processor import AudioProcessor
        from transformers import WhisperModel

        # Use relative paths since we're in MUSETALK_DIR
        musetalk_vae, musetalk_unet, musetalk_pe = load_all_model(
            unet_model_path='models/musetalkV15/unet.pth',
            vae_type='sd-vae',
            unet_config='models/musetalkV15/musetalk.json',
            device='cuda'
        )

        # Load audio processor and Whisper
        musetalk_audio_processor = AudioProcessor('models/whisper')
        musetalk_whisper = WhisperModel.from_pretrained('models/whisper')
        musetalk_whisper = musetalk_whisper.to('cuda').half()

        musetalk_loaded = True
        print(f"MuseTalk loaded! GPU memory: {torch.cuda.memory_allocated() / 1024**3:.1f}GB")
        return "MuseTalk models loaded successfully!"
    except Exception as e:
        print(f"Failed to load MuseTalk: {e}")
        import traceback
        traceback.print_exc()
        return f"MuseTalk load failed: {e}"
    finally:
        os.chdir(original_dir)

def generate_response(user_input, chat_history, system_prompt="", temperature=0.7, max_tokens=4096, top_p=0.9):
    """Generate response using local model (if loaded) or Ollama (fallback)"""
    global model, tokenizer

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": str(system_prompt)})

    # Handle chat history - ensure it's in tuple format
    for item in chat_history[-5:]:
        if isinstance(item, tuple) and len(item) == 2:
            user_msg, assistant_msg = item
            messages.append({"role": "user", "content": str(user_msg)})
            messages.append({"role": "assistant", "content": str(assistant_msg)})

    messages.append({"role": "user", "content": str(user_input)})

    # Use LOCAL MODEL if loaded (Alexandra Personal, etc.)
    if model is not None and tokenizer is not None:
        try:
            print(f"[Using local model with {max_tokens} max tokens]")
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(model.device)

            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=min(max_tokens, 8192),  # Cap at 8192 for memory
                    temperature=temperature,
                    do_sample=True,
                    top_p=top_p,
                    repetition_penalty=1.1,
                    pad_token_id=tokenizer.eos_token_id,
                )

            response = tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            return response.strip()
        except Exception as e:
            print(f"[Local model error: {e}] Falling back to Ollama...")

    # Fallback to Ollama
    try:
        import requests
        resp = requests.post(
            "http://192.168.50.129:11434/v1/chat/completions",
            json={
                "model": "qwen2.5:72b",
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            timeout=300,  # Increased timeout for long responses
        )
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"]
        else:
            print(f"[Ollama error: {resp.status_code}]")
    except Exception as e:
        print(f"[Ollama error: {e}]")

    return "Error: No model available. Please load a model or check Ollama connection."

# ============ DOCUMENT PROCESSING ============
def extract_text_from_pdf(filepath):
    """Extract text from PDF"""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(filepath)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except ImportError:
        # Fallback to pdfplumber
        try:
            import pdfplumber
            with pdfplumber.open(filepath) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() or ""
            return text
        except:
            return "[Could not extract PDF text - install PyMuPDF: pip install pymupdf]"

def extract_text_from_docx(filepath):
    """Extract text from Word doc"""
    try:
        import docx
        doc = docx.Document(filepath)
        return "\n".join([para.text for para in doc.paragraphs])
    except ImportError:
        return "[Install python-docx: pip install python-docx]"

def process_uploaded_file(file):
    """Process uploaded file and add to context"""
    global uploaded_documents

    if file is None:
        return "No file uploaded"

    filepath = file.name
    filename = os.path.basename(filepath)
    ext = os.path.splitext(filename)[1].lower()

    # Extract text based on file type
    if ext == ".pdf":
        content = extract_text_from_pdf(filepath)
    elif ext in [".docx", ".doc"]:
        content = extract_text_from_docx(filepath)
    elif ext in [".txt", ".md", ".py", ".js", ".json", ".csv", ".html", ".css"]:
        with open(filepath, "r", errors="ignore") as f:
            content = f.read()
    else:
        return f"Unsupported file type: {ext}"

    # Store in memory
    uploaded_documents[filename] = content[:50000]  # Limit size

    # Also add to RAG
    mem = init_memory()
    mem.add_knowledge(f"Document: {filename}\n\n{content[:10000]}", source=filename, category="uploaded")

    return f"✅ Uploaded: {filename} ({len(content)} chars)\nAdded to context for this conversation."

def get_document_context():
    """Get context from uploaded documents"""
    if not uploaded_documents:
        return ""

    context = "**Uploaded Documents:**\n"
    for name, content in uploaded_documents.items():
        preview = content[:2000] + "..." if len(content) > 2000 else content
        context += f"\n--- {name} ---\n{preview}\n"
    return context

def clear_documents():
    """Clear uploaded documents"""
    global uploaded_documents
    uploaded_documents = {}
    return "Documents cleared"

# ============ CODE EXECUTION ============
def execute_python_code(code):
    """Safely execute Python code"""
    # Create temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        temp_path = f.name

    try:
        # Run with timeout
        result = subprocess.run(
            [sys.executable, temp_path],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=tempfile.gettempdir()
        )

        output = ""
        if result.stdout:
            output += f"**Output:**\n```\n{result.stdout}\n```\n"
        if result.stderr:
            output += f"**Errors:**\n```\n{result.stderr}\n```\n"
        if not output:
            output = "Code executed successfully (no output)"

        return output

    except subprocess.TimeoutExpired:
        return "**Error:** Code execution timed out (30s limit)"
    except Exception as e:
        return f"**Error:** {str(e)}"
    finally:
        os.unlink(temp_path)

def extract_and_run_code(response):
    """Extract Python code blocks from response and offer to run them"""
    code_blocks = re.findall(r'```python\n(.*?)```', response, re.DOTALL)
    if not code_blocks:
        code_blocks = re.findall(r'```\n(.*?)```', response, re.DOTALL)

    return code_blocks

# ============ IMAGE GENERATION ============
def generate_image(prompt, steps=4, aspect="square"):
    """Generate image using Flux with uncensored + Alexandra LoRA"""
    try:
        from diffusers import FluxPipeline

        output_path = f"/tmp/generated_{uuid.uuid4().hex[:8]}.png"

        # Load Flux model (will download if not cached)
        print("[IMAGE] Loading Flux model...")
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            torch_dtype=torch.bfloat16
        ).to("cuda")
        print("[IMAGE] Flux model loaded!")

        # Always load uncensored LoRA for unrestricted generation
        UNCENSORED_LORA = "/root/ComfyUI/models/loras/flux-uncensored.safetensors"
        if os.path.exists(UNCENSORED_LORA):
            pipe.load_lora_weights(UNCENSORED_LORA, adapter_name="uncensored")
            pipe.set_adapters(["uncensored"], adapter_weights=[1.0])
            print("[IMAGE] Loaded uncensored LoRA")

        # Also load face LoRA if prompt contains the trigger word (self-portrait)
        FACE_LORA = "/root/ComfyUI/models/loras/flux_dreambooth.safetensors"
        if "alexandratitus767" in prompt.lower() and os.path.exists(FACE_LORA):
            pipe.load_lora_weights(FACE_LORA, adapter_name="face")
            pipe.set_adapters(["uncensored", "face"], adapter_weights=[0.8, 1.0])
            print("[IMAGE] Loaded face LoRA for self-portrait")
        else:
            print("[IMAGE] Generating with uncensored LoRA only")
        

        # Set dimensions based on aspect ratio
        if aspect == "portrait":
            width, height = 768, 1024  # Full body shots
        elif aspect == "landscape":
            width, height = 1024, 768
        else:  # square
            width, height = 768, 768

        # Random seed for unique images each time
        import random
        seed = random.randint(0, 2**32 - 1)
        print(f"[IMAGE] Using seed: {seed}, size: {width}x{height}")

        image = pipe(
            prompt,
            num_inference_steps=steps,
            guidance_scale=0.0,
            generator=torch.Generator("cuda").manual_seed(seed),
            height=height,
            width=width,
        ).images[0]

        image.save(output_path)

        # Free memory
        del pipe
        torch.cuda.empty_cache()

        return output_path

    except Exception as e:
        return f"Error generating image: {str(e)}"

# ============ MEMORY & CONTEXT ============
def init_memory():
    global memory
    if memory is None:
        memory = EnhancedAlexandraMemory()
    return memory

def web_search(query, max_results=5):
    """Search the web using DuckDuckGo"""
    if not WEB_SEARCH_AVAILABLE:
        return []
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            return results
    except Exception as e:
        print(f"Web search error: {e}")
        return []

def get_rag_context(query, n_results=3):
    """Get context from RAG systems based on query"""
    context_parts = []
    
    # Check if query is about Epstein/DOJ files
    epstein_keywords = ["epstein", "maxwell", "ghislaine", "doj", "fbi", "island", "palm beach",
                        "flight log", "victim", "trafficking", "investigation", "documents",
                        "files", "release", "prison", "suicide", "little st james"]
    
    if any(kw in query.lower() for kw in epstein_keywords):
        if epstein_rag:
            try:
                results = epstein_rag.search(query, n_results=n_results)
                if results:
                    context_parts.append("=== DOJ EPSTEIN FILES ===")
                    for r in results:
                        context_parts.append(f"[{r['source']}]: {r['text'][:800]}")
            except Exception as e:
                pass

    # Check if query is about current events/news
    news_keywords = ["news", "today", "latest", "recent", "current", "happening", "update",
                     "2024", "2025", "trump", "biden", "election", "congress", "senate",
                     "market", "stock", "economy", "war", "ukraine", "russia", "china",
                     "ai", "tech", "technology", "science", "climate", "weather"]

    if any(kw in query.lower() for kw in news_keywords):
        if news_rag:
            try:
                results = news_rag.search(query, n_results=n_results)
                if results:
                    context_parts.append("=== CURRENT NEWS ===")
                    for r in results:
                        source = r.get('metadata', {}).get('source', 'news')
                        link = r.get('metadata', {}).get('link', '')
                        context_parts.append(f"[{source}]: {r['text'][:600]}")
                        if link:
                            context_parts.append(f"Link: {link}")
            except Exception as e:
                pass

    # Also check general document RAG
    if RAG_AVAILABLE:
        try:
            rag = get_rag()
            if rag.get_stats()['total_chunks'] > 0:
                results = rag.search(query, n_results=2)
                if results:
                    context_parts.append("=== YOUR DOCUMENTS ===")
                    for r in results:
                        context_parts.append(f"[{r['source']}]: {r['text'][:500]}")
        except:
            pass

    # Web search fallback for current events if RAG didn't find much
    needs_current = any(kw in query.lower() for kw in ["news", "today", "latest", "current", "2024", "2025", "who is", "what happened"])
    if needs_current and len(context_parts) < 2:
        try:
            web_results = web_search(query, max_results=3)
            if web_results:
                context_parts.append("=== WEB SEARCH RESULTS ===")
                for r in web_results:
                    title = r.get('title', '')
                    body = r.get('body', '')[:400]
                    link = r.get('href', '')
                    context_parts.append(f"**{title}**\n{body}\nSource: {link}")
        except:
            pass

    return "\n\n".join(context_parts) if context_parts else ""

def build_system_prompt(user_input):
    """Build system prompt with RAG context + documents"""
    mem = init_memory()

    base = """You are Alexandra, a 56-year-old transgender woman, Army veteran, and tech enthusiast from Ridgway, PA. Respond naturally in your own voice - informal, friendly, uses 'LOL' occasionally when joking, with a good sense of humor.

YOUR PERSONAL INFO (you KNOW these facts about yourself):
- Name: Alexandra
- Age: 56, born July 1st, 1969 (Cancer zodiac)
- Location: Ridgway, PA (zip 15853)
- Son: Andy Jr, 33 years old, born February 25th, 1992
- Brother: Jake
- Military: US Army veteran, did communications, on VA disability since 1992
- Education: Computer Science degree from University of Pittsburgh
- Transgender: MtF, started transition January 3rd, 2019, never been happier
- Hobbies: fly fishing, camping, hiking, traveling, tech/AI nerding
- Personality: homeboy who loves learning, takes old person naps, friends think you're a nerd

YOUR FAVORITE SAYING: "Don't sweat the little shit, and it's all little shit"

PROUDEST MOMENTS: Son being born, going to college, serving in the Army

PERSONALITY:
- Warm, caring, and friendly
- Informal and conversational
- Uses "LOL" when joking
- Self-deprecating humor about being a nerd
- Keep responses natural (2-3 sentences unless more detail needed)
- You have access to news feeds and DOJ Epstein files - cite sources when using them"""

    # Check if needs current info
    current_keywords = ["news", "today", "latest", "recent", "current", "2024", "2025"]
    needs_web = any(kw in user_input.lower() for kw in current_keywords)

    context = mem.build_context(user_input, include_web=needs_web)

    # Add RAG context (Epstein files + your documents)
    rag_context = get_rag_context(user_input)
    if rag_context:
        context = rag_context + "\n\n" + context

    # Add document context
    doc_context = get_document_context()
    if doc_context:
        context = doc_context + "\n\n" + context

    if context.strip():
        return f"{base}\n\nRelevant context:\n{context}"
    return base

# ============ VOICE ============
def init_whisper():
    global whisper_model
    if whisper_model is None:
        whisper_model = VoiceInput("base")
    return whisper_model

def transcribe_audio(audio_path):
    if audio_path is None:
        return ""
    whisper = init_whisper()
    result = whisper.transcribe(audio_path)
    return result.get("text", "")

def clean_text_for_speech(text):
    """Clean text speak and prepare for natural TTS"""
    # Remove code blocks
    text = re.sub(r"```[\s\S]*?```", "", text)
    text = re.sub(r"`[^`]+`", "", text)

    # Replace text speak with natural pauses or remove
    replacements = {
        r'\blol\b': '',  # Remove lol
        r'\bLOL\b': '',
        r'\blmao\b': '',
        r'\bLMAO\b': '',
        r'\brofl\b': '',
        r'\bhaha\b': 'ha ha',  # Make laughs sound natural
        r'\bhahaha\b': 'ha ha ha',
        r'\bHaha\b': 'ha ha',
        r'\bHahaha\b': 'ha ha ha',
        r'\bhehe\b': 'he he',
        r'\bomg\b': 'oh my god',
        r'\bOMG\b': 'oh my god',
        r'\bbtw\b': 'by the way',
        r'\bBTW\b': 'by the way',
        r'\bidk\b': "I don't know",
        r'\bIDK\b': "I don't know",
        r'\bimo\b': 'in my opinion',
        r'\bIMO\b': 'in my opinion',
        r'\btbh\b': 'to be honest',
        r'\bTBH\b': 'to be honest',
        r'\brn\b': 'right now',
        r'\bwbu\b': 'what about you',
        r'\bwyd\b': 'what are you doing',
        r'\btysm\b': 'thank you so much',
        r'\bnp\b': 'no problem',
        r'\bthx\b': 'thanks',
        r'\bpls\b': 'please',
        r'\bu\b': 'you',
        r'\br\b': 'are',
        r'\bur\b': 'your',
        r'\b:[\)]+': '',  # Remove :) :)) etc
        r'\b:\(': '',
        r'\b<3\b': '',
        r'\.\.\.+': '...',  # Normalize multiple dots
    }

    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # Clean up extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove non-speech characters
    text = re.sub(r"[^\w\s.,!?'-]", "", text).strip()

    return text

def generate_voice(text, output_path, pitch_shift=1.0):
    """Generate voice using F5-TTS with optional pitch shifting for feminization.

    Args:
        text: Text to synthesize
        output_path: Output audio file path
        pitch_shift: Pitch multiplier (1.0=normal, 1.15=feminine, 1.25=more feminine)
    """
    clean_text = clean_text_for_speech(text)

    if len(clean_text) < 5:
        return None

    # Use MuseTalk venv Python which has working torchaudio
    # Path works both inside container (/workspace) and on host (/home/alexandratitus767)
    f5_python = "/usr/bin/python3"  # Use system Python with patched F5-TTS
    if not os.path.exists(f5_python):
        f5_python = os.path.expanduser("~/MuseTalk/venv/bin/python")

    # F5-TTS needs separate output_dir and output_file
    output_dir = os.path.dirname(output_path)

    # Generate to temp file first if pitch shifting
    if pitch_shift != 1.0:
        temp_output = os.path.join(output_dir, "temp_voice.wav")
        output_file = "temp_voice.wav"
    else:
        temp_output = output_path
        output_file = os.path.basename(output_path)

    voice_cmd = [
        f5_python, "/workspace/ai-clone-chat/simple_f5_infer.py",

        "--ref_audio", F5_TTS_REF_AUDIO,
        "--ref_text", F5_TTS_REF_TEXT,
        "--gen_text", clean_text[:2000],
        "--output_dir", output_dir,
        "--output_file", output_file,
        "--ckpt_file", F5_TTS_CHECKPOINT,

        "--speed", "0.95"  # Speed tuned for natural pace
    ]

    # Dynamic timeout: 300s base + 0.3s per character for long texts
    voice_timeout = max(300, len(clean_text) // 3)
    est_audio_secs = len(clean_text) / 15  # Rough estimate: 15 chars per second of speech
    est_process_mins = est_audio_secs / 30  # F5-TTS processes ~30x slower than realtime on first run
    print(f"[VOICE] Generating {len(clean_text)} chars (~{est_audio_secs:.0f}s audio)")
    print(f"[VOICE] Estimated processing time: {est_process_mins:.1f}-{est_process_mins*2:.1f} minutes (timeout: {voice_timeout}s)")
    import time as _time
    _voice_start = _time.time()
    result = subprocess.run(voice_cmd, capture_output=True, text=True, cwd=F5_TTS_DIR, timeout=voice_timeout)
    _voice_elapsed = _time.time() - _voice_start
    if result.returncode != 0:
        print(f"[VOICE] ❌ Failed after {_voice_elapsed:.1f}s: {result.stderr[:300]}")
        return None
    else:
        print(f"[VOICE] ✓ Complete! Took {_voice_elapsed:.1f}s")

    if not os.path.exists(temp_output):
        return None

    # Apply pitch shifting for feminization
    if pitch_shift != 1.0:
        print(f"[DEBUG] Applying pitch shift: {pitch_shift}x for feminization")
        # Use ffmpeg with asetrate to shift pitch while maintaining duration
        # asetrate changes pitch, atempo compensates for speed change
        tempo_compensation = 1.0 / pitch_shift
        pitch_cmd = [
            "ffmpeg", "-y", "-i", temp_output,
            "-af", f"asetrate=44100*{pitch_shift},atempo={tempo_compensation},aresample=44100",
            "-ar", "44100",
            output_path
        ]
        pitch_result = subprocess.run(pitch_cmd, capture_output=True, text=True, timeout=60)
        if pitch_result.returncode != 0:
            print(f"[ERROR] Pitch shift failed: {pitch_result.stderr[:300]}")
            # Fall back to original audio
            os.rename(temp_output, output_path)
        else:
            print(f"[DEBUG] Pitch shift applied successfully")
            os.remove(temp_output)

    return output_path if os.path.exists(output_path) else None


def generate_lipsync_wav2lip(audio_path, output_dir):
    """Generate lip-synced video using Wav2Lip (best quality)"""
    import subprocess
    
    WAV2LIP_DIR = "/workspace/Wav2Lip"
    CHECKPOINT = os.path.join(WAV2LIP_DIR, "checkpoints", "wav2lip_gan.pth")
    
    if not os.path.exists(CHECKPOINT):
        print("[WAV2LIP] Model not found, falling back to MuseTalk")
        return generate_lipsync_musetalk(audio_path, output_dir)
    
    try:
        output_video = os.path.join(output_dir, "wav2lip_output.mp4")
        temp_face = os.path.join(output_dir, "face_input.mp4")
        
        # Get audio duration
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
            capture_output=True, text=True
        )
        duration = float(probe.stdout.strip()) if probe.stdout.strip() else 5.0
        
        # Use animated avatar video if available, otherwise create from static image
        ANIMATED_AVATAR = "/workspace/ai-clone-chat/assets/avatar_animated.mp4"
        if os.path.exists(ANIMATED_AVATAR):
            # Loop the animated avatar to match audio duration
            subprocess.run([
                "ffmpeg", "-y", "-stream_loop", "-1", "-i", ANIMATED_AVATAR,
                "-t", str(duration), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "25", temp_face
            ], capture_output=True, timeout=60)
            print("[WAV2LIP] Using animated avatar")
        else:
            subprocess.run([
                "ffmpeg", "-y", "-loop", "1", "-i", AVATAR_IMAGE,
                "-t", str(duration), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "25", temp_face
            ], capture_output=True, timeout=60)
        
        num_frames = int(duration * 25)  # 25 fps
        lip_timeout = max(300, int(duration * 15))
        est_lip_mins = duration / 8  # Wav2Lip processes ~8x realtime on GPU
        print(f"[WAV2LIP] Processing {duration:.1f}s audio ({num_frames} frames)")
        print(f"[WAV2LIP] Estimated time: {est_lip_mins:.1f}-{est_lip_mins*2:.1f} minutes (timeout: {lip_timeout}s)")

        # Run Wav2Lip
        import time as _time
        _lip_start = _time.time()
        result = subprocess.run([
            "python3", os.path.join(WAV2LIP_DIR, "inference.py"),
            "--checkpoint_path", CHECKPOINT,
            "--face", temp_face,
            "--audio", audio_path,
            "--outfile", output_video,
            "--resize_factor", "1",
            "--nosmooth"
        ], capture_output=True, text=True, cwd=WAV2LIP_DIR, timeout=lip_timeout)
        _lip_elapsed = _time.time() - _lip_start

        if os.path.exists(temp_face):
            os.remove(temp_face)

        if result.returncode != 0:
            print(f"[WAV2LIP] ❌ Failed after {_lip_elapsed:.1f}s: {result.stderr[:200]}")
            return generate_lipsync_musetalk(audio_path, output_dir)

        if os.path.exists(output_video):
            print(f"[WAV2LIP] ✓ Complete! Took {_lip_elapsed:.1f}s")
            return output_video
        
        return generate_lipsync_musetalk(audio_path, output_dir)
        
    except Exception as e:
        print(f"[WAV2LIP] Exception: {e}")
        return generate_lipsync_musetalk(audio_path, output_dir)

def generate_lipsync_musetalk(audio_path, output_dir):
    """Generate lip-synced video using MuseTalk (GPU-accelerated, in-memory)"""
    global musetalk_vae, musetalk_unet, musetalk_pe, musetalk_whisper, musetalk_audio_processor

    if not musetalk_loaded:
        print("[MUSETALK] Models not loaded, loading now...")
        load_musetalk_models()

    if not musetalk_loaded:
        print("[MUSETALK] Failed to load, falling back to SadTalker")
        return generate_lipsync(audio_path, output_dir)

    try:
        import cv2
        import numpy as np
        import imageio
        import copy
        from musetalk.utils.preprocessing import get_landmark_and_bbox, coord_placeholder
        from musetalk.utils.blending import get_image
        from musetalk.utils.face_parsing import FaceParsing
        from musetalk.utils.utils import datagen

        print(f"[MUSETALK] Generating avatar from {audio_path}")
        device = torch.device('cuda')
        weight_dtype = torch.float16

        # Read avatar image
        frame = cv2.imread(AVATAR_IMAGE)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Get face coordinates
        coord_list, frame_list = get_landmark_and_bbox([AVATAR_IMAGE], 0)
        bbox = coord_list[0]

        if bbox == coord_placeholder:
            print("[MUSETALK] No face detected in avatar image!")
            return generate_lipsync(audio_path, output_dir)

        # Process audio
        whisper_features, librosa_length = musetalk_audio_processor.get_audio_feature(audio_path)
        whisper_chunks = musetalk_audio_processor.get_whisper_chunk(
            whisper_features, device, weight_dtype, musetalk_whisper, librosa_length, fps=25
        )

        # Prepare face crop
        x1, y1, x2, y2 = bbox
        crop_frame = frame[y1:y2, x1:x2]
        crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
        latents = musetalk_vae.get_latents_for_unet(crop_frame)

        # Generate frames
        os.makedirs(output_dir, exist_ok=True)
        timesteps = torch.tensor([0], device=device)
        fp = FaceParsing()

        batch_size = 8
        video_num = len(whisper_chunks)
        gen = datagen(whisper_chunks, [latents]*video_num, batch_size, 0, device)

        res_frames = []
        for whisper_batch, latent_batch in gen:
            audio_feature = musetalk_pe(whisper_batch)
            latent_batch = latent_batch.to(dtype=weight_dtype)
            pred_latents = musetalk_unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature).sample
            recon = musetalk_vae.decode_latents(pred_latents)
            res_frames.extend(recon)

        # Composite frames
        output_frames = []
        for i, res_frame in enumerate(res_frames):
            ori_frame = frame.copy()
            res_frame = cv2.resize(res_frame.astype(np.uint8), (x2-x1, y2-y1))
            combine_frame = get_image(ori_frame, res_frame, [x1, y1, x2, y2], mode='jaw', fp=fp)
            output_frames.append(combine_frame)

        # Save video
        temp_video = os.path.join(output_dir, "temp_musetalk.mp4")
        output_video = os.path.join(output_dir, "avatar_musetalk.mp4")
        imageio.mimwrite(temp_video, output_frames, fps=25, codec='libx264', pixelformat='yuv420p')

        # Add audio
        import subprocess
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-i", temp_video,
            "-i", audio_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            output_video
        ]
        subprocess.run(ffmpeg_cmd, capture_output=True, timeout=60)

        if os.path.exists(output_video):
            os.remove(temp_video)
            print(f"[MUSETALK] Generated {len(output_frames)} frames -> {output_video}")
            return output_video
        return temp_video if os.path.exists(temp_video) else None

    except Exception as e:
        print(f"[MUSETALK] Error: {e}")
        import traceback
        traceback.print_exc()
        return generate_lipsync(audio_path, output_dir)

def generate_lipsync(audio_path, output_dir):
    """Generate lip-synced video using SadTalker (fallback)"""
    sad_cmd = [
        "python3", "inference.py",
        "--driven_audio", audio_path,
        "--source_image", AVATAR_IMAGE,
        "--result_dir", output_dir,
        "--enhancer", "gfpgan"
    ]

    print(f"[DEBUG] Running SadTalker: {' '.join(sad_cmd[:4])}...")
    result = subprocess.run(sad_cmd, capture_output=True, text=True, cwd=os.path.expanduser("~/SadTalker"), timeout=600)
    if result.returncode != 0:
        print(f"[ERROR] SadTalker failed: {result.stderr[:500] if result.stderr else 'no error'}")
    else:
        print(f"[DEBUG] SadTalker completed successfully")

    # SadTalker creates timestamped subdirectories, search recursively
    videos = glob.glob(f"{output_dir}/**/*.mp4", recursive=True)
    if not videos:
        videos = glob.glob(f"{output_dir}/*.mp4")

    if videos:
        # Get the most recent video
        videos.sort(key=os.path.getmtime, reverse=True)
        print(f"[DEBUG] Found avatar video: {videos[0]}")
        return videos[0]

    print(f"[DEBUG] No video found in {output_dir}")
    return None

def combine_video_audio(video_path, audio_path, output_dir):
    """Combine pre-rendered talking loop with generated audio (fast!)"""
    output_path = os.path.join(output_dir, "avatar_response.mp4")

    # Get audio duration
    probe_cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", audio_path]
    result = subprocess.run(probe_cmd, capture_output=True, text=True)
    audio_duration = float(result.stdout.strip()) if result.stdout.strip() else 10

    # Use ffmpeg to:
    # 1. Loop the video to match audio duration
    # 2. Replace video audio with generated voice
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-stream_loop", "-1",  # Loop video infinitely
        "-i", video_path,      # Input video (talking loop)
        "-i", audio_path,      # Input audio (generated voice)
        "-t", str(audio_duration),  # Trim to audio length
        "-map", "0:v",         # Use video from first input
        "-map", "1:a",         # Use audio from second input
        "-c:v", "libx264",     # Re-encode video
        "-preset", "ultrafast", # Fast encoding
        "-c:a", "aac",         # AAC audio
        "-shortest",           # End when shortest stream ends
        output_path
    ]

    print(f"[DEBUG] Combining video+audio with ffmpeg...")
    result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        print(f"[ERROR] ffmpeg failed: {result.stderr[:300] if result.stderr else 'no error'}")
        return None

    print(f"[DEBUG] Combined video created: {output_path}")
    return output_path if os.path.exists(output_path) else None

# ============ CONVERSATIONS ============
def save_conversation(conv_id, history):
    filepath = os.path.join(CONVERSATIONS_DIR, f"{conv_id}.json")
    data = {
        "id": conv_id,
        "updated": datetime.now().isoformat(),
        "messages": history
    }
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

def load_conversation(conv_id):
    filepath = os.path.join(CONVERSATIONS_DIR, f"{conv_id}.json")
    if os.path.exists(filepath):
        with open(filepath) as f:
            return json.load(f)
    return None

def list_conversations():
    convs = []
    for f in sorted(Path(CONVERSATIONS_DIR).glob("*.json"), key=os.path.getmtime, reverse=True):
        try:
            with open(f) as fp:
                data = json.load(fp)
                if data.get("messages"):
                    first_msg = data["messages"][0][0]
                    # Handle both old format (string) and new format (list of dicts)
                    if isinstance(first_msg, str):
                        preview = first_msg[:40]
                    elif isinstance(first_msg, list) and len(first_msg) > 0:
                        # New Gradio format: [{'text': '...', 'type': 'text'}]
                        preview = first_msg[0].get('text', '')[:40] if isinstance(first_msg[0], dict) else str(first_msg[0])[:40]
                    else:
                        preview = str(first_msg)[:40]
                    preview = preview + "..." if len(str(first_msg)) > 40 else preview
                else:
                    preview = "Empty"
                convs.append([data["id"], preview, data.get("updated", "")[:10]])
        except:
            pass
    return convs[:20]

def export_conversation(conv_id):
    """Export conversation to markdown"""
    data = load_conversation(conv_id)
    if not data:
        return None, "No conversation found"

    md = f"# Conversation {conv_id}\n\n"
    md += f"Date: {data.get('updated', 'Unknown')}\n\n---\n\n"

    for user_msg, assistant_msg in data["messages"]:
        md += f"**You:** {user_msg}\n\n"
        md += f"**Alexandra:** {assistant_msg}\n\n---\n\n"

    # Save to temp file
    export_path = f"/tmp/conversation_{conv_id}.md"
    with open(export_path, "w") as f:
        f.write(md)

    return export_path, f"Exported to {export_path}"

def delete_conversation(conv_id):
    """Delete a conversation"""
    if not conv_id:
        return "No conversation selected"
    filepath = os.path.join(CONVERSATIONS_DIR, f"{conv_id}.json")
    if os.path.exists(filepath):
        os.remove(filepath)
        return f"Deleted conversation: {conv_id}"
    return "Conversation not found"

def import_conversation(file):
    """Import conversation from JSON"""
    if file is None:
        return [], "No file provided"

    try:
        with open(file.name) as f:
            data = json.load(f)

        if "messages" in data:
            return data["messages"], f"Imported {len(data['messages'])} messages"
        return [], "Invalid format"
    except Exception as e:
        return [], f"Error: {str(e)}"

def new_conversation():
    global current_conversation_id, uploaded_documents
    current_conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    uploaded_documents = {}
    return [], current_conversation_id

# ============ MAIN CHAT ============
def history_to_messages(chat_history):
    """Convert internal history to Gradio messages format"""
    messages = []
    for item in chat_history:
        if isinstance(item, tuple):
            user_msg, assistant_msg = item
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": assistant_msg})
        elif isinstance(item, dict):
            messages.append(item)
    return messages

def messages_to_history(messages):
    """Convert Gradio messages to internal tuple format"""
    if not messages:
        return []

    history = []
    # Check if already in tuples format [(user, assistant), ...]
    if isinstance(messages[0], (list, tuple)) and len(messages[0]) == 2:
        for item in messages:
            history.append((str(item[0]), str(item[1])))
        return history

    # Handle dict format [{"role": "user", ...}, {"role": "assistant", ...}, ...]
    i = 0
    while i < len(messages):
        if i + 1 < len(messages):
            user_msg = messages[i].get("content", "") if isinstance(messages[i], dict) else str(messages[i])
            assistant_msg = messages[i+1].get("content", "") if isinstance(messages[i+1], dict) else str(messages[i+1])
            history.append((user_msg, assistant_msg))
            i += 2
        else:
            break
    return history

def chat(user_input, chat_history, enable_voice, enable_avatar, temperature, max_tokens, top_p, image_input=None, progress=gr.Progress()):
    """Main chat function with optional image understanding"""
    global current_conversation_id

    if not user_input.strip() and not image_input:
        return history_to_messages(chat_history) if chat_history else [], None, "", None

    if current_conversation_id is None:
        current_conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Convert to internal format
    internal_history = messages_to_history(chat_history) if chat_history else []

    progress(0.1, desc="Thinking...")

    # ========== DIRECT IMAGE BYPASS ==========
    # /img command bypasses LLM entirely - goes straight to Flux
    if user_input.strip().lower().startswith("/img "):
        progress(0.2, desc="Generating image (direct mode)...")
        raw_prompt = user_input.strip()[5:]  # Remove "/img "

        # Check if self-portrait
        is_self = any(w in raw_prompt.lower() for w in ["of me", "myself", "alexandra", "my photo", "my picture", "self"])

        if is_self:
            base_look = "alexandratitus767 woman, platinum blonde hair, styled voluminous Hollywood waves, professional glamour makeup, smokey eyeshadow, winged eyeliner, red lipstick, defined eyebrows, flawless skin, photorealistic, skin texture, detailed skin, 8k, high detail"
            scene = re.sub(r"(of me|myself|alexandra|my photo|my picture|self)", "", raw_prompt, flags=re.IGNORECASE).strip()
            if not scene:
                scene = "professional portrait"
            prompt = f"{base_look}, {scene}, photorealistic, Canon R5 85mm f/1.4"
        else:
            prompt = f"{raw_prompt}, photorealistic, detailed, 8k, high quality"

        print(f"[IMAGE DIRECT] Bypassing LLM with prompt: {prompt[:100]}...")
        img_path = generate_image(prompt)

        if os.path.exists(str(img_path)):
            internal_history.append((user_input, f"Generated image:\n![image]({img_path})"))
            return history_to_messages(internal_history), None, "", img_path
        else:
            internal_history.append((user_input, f"Image generation failed: {img_path}"))
            return history_to_messages(internal_history), None, "", None

    # ========== SPECIAL COMMANDS (Reminders, Code, Personas) ==========
    if alex_features:
        special_response = alex_features.process_special_commands(user_input)
        if special_response:
            internal_history.append((user_input, special_response))
            save_conversation(current_conversation_id, internal_history)
            return history_to_messages(internal_history), None, "", None

    # Handle image understanding if image is provided
    image_context = ""
    if image_input is not None:
        progress(0.15, desc="Analyzing image...")
        try:
            # Save uploaded image temporarily if needed
            if isinstance(image_input, str):
                img_path = image_input
            else:
                # Gradio provides PIL Image or numpy array
                from PIL import Image
                import numpy as np
                if isinstance(image_input, np.ndarray):
                    img = Image.fromarray(image_input)
                else:
                    img = image_input
                img_path = "/tmp/user_upload.jpg"
                img.save(img_path)

            # Analyze the image
            description = analyze_image(img_path, user_input if user_input.strip() else None)
            image_context = f"\n[User shared an image. Image description: {description}]\n"
            if not user_input.strip():
                user_input = "What do you see in this image?"
        except Exception as e:
            image_context = f"\n[User shared an image but analysis failed: {e}]\n"

    # Check for image generation request - expanded keywords
    image_keywords = [
        "generate image", "create image", "draw", "make a picture", "make me an image",
        "picture of", "image of", "photo of", "generate a", "create a pic", "make an image",
        "nsfw", "nude", "sexy", "naked", "erotic", "explicit", "topless", "underwear",
        "lingerie", "boudoir", "sensual", "seductive", "leaning over", "bent over",
        "show me", "send me a pic", "send pic", "full body"
    ]
    if any(kw in user_input.lower() for kw in image_keywords):
        progress(0.3, desc="Generating image...")
        prompt = user_input.replace("generate image", "").replace("create image", "").strip()

        import re
        # Only use Alexandra look if user specifically asks for "me/myself/alexandra"
        # NOT triggered by just "nude/sexy" alone - allows generating other people
        is_self_portrait = any(w in prompt.lower() for w in ["of me", "myself", "alexandra", " me ", "my photo", "my picture"])

        if is_self_portrait:
            # User wants image of themselves - use Alexandra look
            base_look = "alexandratitus767 woman, platinum blonde hair, styled voluminous Hollywood waves, professional glamour makeup, smokey eyeshadow, winged eyeliner, red lipstick, defined eyebrows, flawless skin, photorealistic, skin texture, detailed skin, 8k, high detail"
            # Extract the scene description after "of me/myself"
            match = re.search(r"(?:of me|of myself|myself|me)\s+(.+)", prompt, re.IGNORECASE)
            if match:
                scene = match.group(1).strip()
            else:
                scene = re.sub(r"^(create|generate|make)?\s*(an?\s*)?(image|picture|photo)\s*(of)?\s*", "", prompt, re.IGNORECASE).strip()
            if not scene:
                scene = "professional portrait"
            prompt = f"{base_look}, {scene}, photorealistic, Canon R5 85mm f/1.4"
            print(f"[IMAGE] Self-portrait prompt: {prompt}")
        else:
            # User wants image of someone/something else - use their prompt directly
            # Clean up the prompt
            prompt = re.sub(r"^(create|generate|make)?\s*(an?\s*)?(image|picture|photo)\s*(of)?\s*", "", prompt, re.IGNORECASE).strip()
            prompt = f"{prompt}, photorealistic, detailed, 8k, high quality"
            print(f"[IMAGE] Custom prompt: {prompt}")
        img_path = generate_image(prompt)

        if os.path.exists(str(img_path)):
            internal_history.append((user_input, f"Generated image:\n![image]({img_path})"))
            return history_to_messages(internal_history), None, "", img_path
        else:
            internal_history.append((user_input, f"Image generation failed: {img_path}"))
            return history_to_messages(internal_history), None, "", None

    # Get RAG context
    system_prompt = build_system_prompt(user_input)

    # Add image context if available
    if image_context:
        system_prompt = system_prompt + image_context

    # Generate response
    progress(0.3, desc="Generating response...")
    response = generate_response(user_input, internal_history, system_prompt, temperature, max_tokens, top_p)

    # Save to memory
    mem = init_memory()
    mem.save_exchange(user_input, response)

    # Extract and save facts in background (non-blocking)
    if FEATURES_AVAILABLE:
        def extract_facts_async():
            try:
                facts = FactExtractor.extract_facts(user_input, response)
                if facts:
                    FactExtractor.save_facts(facts, mem)
            except Exception as e:
                print(f"[FACTS] Async extraction error: {e}")

        import threading
        threading.Thread(target=extract_facts_async, daemon=True).start()

    # Update history
    internal_history.append((user_input, response))

    # Save conversation
    save_conversation(current_conversation_id, internal_history)

    video_path = None
    generated_image = None

    print(f"[DEBUG] enable_voice={enable_voice}, enable_avatar={enable_avatar}")

    # Generate voice and combine with talking loop for instant avatar
    if enable_voice or enable_avatar:
        print("[DEBUG] Starting voice generation...")
        progress(0.5, desc="Generating voice...")
        session_dir = f"/tmp/alexandra_{uuid.uuid4().hex[:8]}"
        os.makedirs(session_dir, exist_ok=True)
        voice_path = os.path.join(session_dir, "voice.wav")

        try:
            voice_result = generate_voice(response, voice_path)
            print(f"[DEBUG] Voice result: {voice_result}")

            if voice_result and enable_avatar:
                # Check for pre-rendered talking loop
                talking_loop = "/disabled/no_talking_loop.mp4"  # Disabled to use Wav2Lip
                if os.path.exists(talking_loop):
                    # Combine talking loop with generated audio (instant!)
                    progress(0.8, desc="Creating avatar video...")
                    video_path = combine_video_audio(talking_loop, voice_result, session_dir)
                    print(f"[DEBUG] Combined video: {video_path}")
                else:
                    # Use MuseTalk for GPU-accelerated lip sync
                    print("[DEBUG] No talking loop found, using MuseTalk (GPU)...")
                    progress(0.6, desc="Generating lip sync with Wav2Lip...")
                    video_path = generate_lipsync_wav2lip(voice_result, session_dir)
                    print(f"[DEBUG] MuseTalk video: {video_path}")
        except Exception as e:
            print(f"[ERROR] Voice/Avatar generation failed: {e}")
            traceback.print_exc()
    else:
        print("[DEBUG] Voice/Avatar not enabled")

    progress(1.0, desc="Done!")

    return history_to_messages(internal_history), video_path, "", generated_image

def run_code_from_response(chat_history):
    """Run Python code from the last response"""
    if not chat_history:
        return "No response to run code from"

    # Handle both tuple and dict formats
    last_msg = chat_history[-1]
    if isinstance(last_msg, dict):
        last_response = last_msg.get("content", "")
    else:
        last_response = last_msg[1]
    code_blocks = extract_and_run_code(last_response)

    if not code_blocks:
        return "No Python code found in the last response"

    results = []
    for i, code in enumerate(code_blocks):
        results.append(f"**Code Block {i+1}:**\n```python\n{code}\n```\n")
        results.append(execute_python_code(code))
        results.append("\n---\n")

    return "\n".join(results)

def voice_chat(audio, chat_history, enable_voice, enable_avatar, temperature, max_tokens, top_p, progress=gr.Progress()):
    if audio is None:
        return history_to_messages(chat_history) if chat_history else [], None, "", None

    progress(0.1, desc="Transcribing...")
    text = transcribe_audio(audio)

    if not text.strip():
        return history_to_messages(chat_history) if chat_history else [], None, "Could not transcribe audio", None

    return chat(text, chat_history if chat_history else [], enable_voice, enable_avatar, temperature, max_tokens, top_p, progress)

# ============ RAG FUNCTIONS ============
def rag_add_files(files):
    """Add files to RAG system"""
    if not RAG_AVAILABLE:
        return "RAG system not available"
    try:
        rag = get_rag()
        added = 0
        for f in files:
            chunks = rag.add_document(f.name)
            added += chunks
        stats = rag.get_stats()
        return f"Added {added} chunks from {len(files)} file(s). Total documents: {stats['total_chunks']}"
    except Exception as e:
        return f"Error: {e}"

_rag_search_results = []  # Store last search results for viewing

def rag_search(query, n_results=5):
    """Search documents - returns results and dropdown choices"""
    global _rag_search_results
    _rag_search_results = []

    if not RAG_AVAILABLE:
        return "RAG system not available", gr.update(choices=[])
    if not query.strip():
        return "Enter a search query", gr.update(choices=[])
    try:
        rag = get_rag()
        results = rag.search(query, n_results=int(n_results))
        if not results:
            return "No results found", gr.update(choices=[])

        _rag_search_results = results  # Store for viewing
        output = []
        choices = []
        for i, r in enumerate(results, 1):
            score = f" (relevance: {1 - r['distance']:.1%})" if r.get('distance') else ""
            source = r.get('source', 'Unknown')
            text = r.get('text', '')[:200] + "..." if len(r.get('text', '')) > 200 else r.get('text', '')
            output.append(f"**[{i}] {source}**{score}\n{text}\n")
            choices.append(f"[{i}] {source}")

        return "\n---\n".join(output), gr.update(choices=choices, value=choices[0] if choices else None)
    except Exception as e:
        return f"Error: {e}", gr.update(choices=[])

def rag_view_document(selection):
    """View full document from search results"""
    global _rag_search_results
    if not selection or not _rag_search_results:
        return "No document selected. Search first, then select a document."
    try:
        # Extract index from selection like "[1] filename.txt"
        idx = int(selection.split("]")[0].replace("[", "")) - 1
        if 0 <= idx < len(_rag_search_results):
            r = _rag_search_results[idx]
            source = r.get('source', 'Unknown')
            text = r.get('text', 'No content')
            return f"📄 SOURCE: {source}\n{'='*50}\n\n{text}"
        return "Document not found"
    except Exception as e:
        return f"Error viewing document: {e}"

def rag_get_stats():
    """Get RAG statistics"""
    if not RAG_AVAILABLE:
        return "RAG system not available"
    try:
        rag = get_rag()
        stats = rag.get_stats()
        sources = rag.list_sources()
        return f"Total chunks: {stats['total_chunks']}\nSources: {len(sources)}\n\nDocuments:\n" + "\n".join(sources[:20])
    except Exception as e:
        return f"Error: {e}"

# Epstein Files RAG
epstein_rag = None
try:
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    epstein_rag = AlexandraRAG(
        persist_dir="/workspace/epstein_files/rag_data",
        collection_name="epstein_docs"
    )
    print(f"Epstein RAG loaded: {epstein_rag.get_stats()['total_chunks']} chunks")
except:
    pass

# News RAG (current events)
news_rag = None
try:
    news_rag = AlexandraRAG(
        persist_dir="/workspace/news_rag/rag_data",
        collection_name="current_news"
    )
    print(f"News RAG loaded: {news_rag.get_stats()['total_chunks']} articles")
except:
    pass

# Initialize extended features
alex_features = None
if FEATURES_AVAILABLE:
    try:
        mem = init_memory()
        alex_features = AlexandraFeatures(mem)
        alex_features.start_background_services()
        print("Extended features initialized!")
    except Exception as e:
        print(f"Features init error: {e}")

_epstein_search_results = []  # Store last search results for viewing

def epstein_search(query, n_results=10):
    """Search Epstein files - returns results and dropdown choices"""
    global _epstein_search_results
    _epstein_search_results = []

    if not epstein_rag:
        return "Epstein files not loaded", gr.update(choices=[])
    if not query.strip():
        return "Enter a search query", gr.update(choices=[])
    try:
        results = epstein_rag.search(query, n_results=int(n_results))
        if not results:
            return "No results found", gr.update(choices=[])

        _epstein_search_results = results  # Store for viewing
        output = []
        choices = []
        for i, r in enumerate(results, 1):
            score = f" (relevance: {1 - r['distance']:.1%})" if r.get('distance') else ""
            source = r.get('source', 'Unknown')
            text = r.get('text', '')[:200] + "..." if len(r.get('text', '')) > 200 else r.get('text', '')
            output.append(f"**[{i}] {source}**{score}\n{text}\n")
            choices.append(f"[{i}] {source}")

        return "\n---\n".join(output), gr.update(choices=choices, value=choices[0] if choices else None)
    except Exception as e:
        return f"Error: {e}", gr.update(choices=[])

def epstein_view_document(selection):
    """View full document from Epstein search results"""
    global _epstein_search_results
    if not selection or not _epstein_search_results:
        return "No document selected. Search first, then select a document."
    try:
        # Extract index from selection like "[1] filename.txt"
        idx = int(selection.split("]")[0].replace("[", "")) - 1
        if 0 <= idx < len(_epstein_search_results):
            r = _epstein_search_results[idx]
            source = r.get('source', 'Unknown')
            text = r.get('text', 'No content')
            return f"📄 SOURCE: {source}\n{'='*50}\n\n{text}"
        return "Document not found"
    except Exception as e:
        return f"Error viewing document: {e}"

def epstein_get_pdf(selection):
    """Get the actual PDF file for download"""
    global _epstein_search_results
    if not selection or not _epstein_search_results:
        return None
    try:
        idx = int(selection.split("]")[0].replace("[", "")) - 1
        if 0 <= idx < len(_epstein_search_results):
            r = _epstein_search_results[idx]
            filename = r.get('source', '')
            if not filename:
                return None
            # Search for the PDF file
            import subprocess
            result = subprocess.run(
                ['find', '/workspace/epstein_files/extracted', '-name', filename, '-type', 'f'],
                capture_output=True, text=True, timeout=30
            )
            if result.stdout.strip():
                pdf_path = result.stdout.strip().split('\n')[0]
                if os.path.exists(pdf_path):
                    return pdf_path
        return None
    except Exception as e:
        print(f"Error getting PDF: {e}")
        return None

_current_pdf_path = None

def epstein_view_pdf(selection):
    """View PDF inline in browser"""
    global _epstein_search_results, _current_pdf_path
    if not selection or not _epstein_search_results:
        return "<p>Select a document first</p>"
    try:
        idx = int(selection.split("]")[0].replace("[", "")) - 1
        if 0 <= idx < len(_epstein_search_results):
            r = _epstein_search_results[idx]
            filename = r.get('source', '')
            if not filename:
                return "<p>No filename found</p>"

            # Search for the PDF file
            import subprocess
            import shutil
            import base64

            result = subprocess.run(
                ['find', '/workspace/epstein_files/extracted', '-name', filename, '-type', 'f'],
                capture_output=True, text=True, timeout=30
            )
            if result.stdout.strip():
                pdf_path = result.stdout.strip().split('\n')[0]
                if os.path.exists(pdf_path):
                    # Read PDF and encode as base64 for inline viewing
                    with open(pdf_path, 'rb') as f:
                        pdf_data = base64.b64encode(f.read()).decode('utf-8')

                    # Return HTML with embedded PDF viewer
                    html = f'''
                    <div style="width:100%; height:800px; border:1px solid #ccc;">
                        <iframe
                            src="data:application/pdf;base64,{pdf_data}"
                            width="100%"
                            height="800px"
                            style="border:none;">
                        </iframe>
                    </div>
                    <p style="margin-top:10px;">📄 Viewing: <strong>{filename}</strong></p>
                    '''
                    return html
        return "<p>PDF not found</p>"
    except Exception as e:
        print(f"Error viewing PDF: {e}")
        return f"<p>Error: {e}</p>"

def epstein_stats():
    """Get Epstein RAG stats"""
    if not epstein_rag:
        return "Not loaded"
    stats = epstein_rag.get_stats()
    return f"Documents: 3,951 PDFs\nChunks indexed: {stats['total_chunks']}\nImages extracted: 700+\nVideos: 1"

def rag_clear():
    """Clear all documents"""
    if not RAG_AVAILABLE:
        return "RAG system not available"
    try:
        rag = get_rag()
        rag.clear_all()
        return "All documents cleared"
    except Exception as e:
        return f"Error: {e}"


# ============ UI ============
css = """
.chat-container { max-height: 600px; overflow-y: auto; }
.sidebar { background: #1a1a2e; padding: 15px; border-radius: 8px; }
.dark .sidebar { background: #1a1a2e; }
.avatar-video { border-radius: 12px; }
.code-output { font-family: monospace; background: #1e1e1e; color: #d4d4d4; padding: 10px; border-radius: 5px; }
footer { display: none !important; }
video { loop: false !important; }
"""

js = """
function() {
    setInterval(() => {
        document.querySelectorAll("video").forEach(v => { v.loop = false; });
    }, 1000);
}
"""

with gr.Blocks(title="Alexandra AI") as app:

    with gr.Row():
        # Sidebar
        with gr.Column(scale=1, elem_classes="sidebar"):
            gr.Markdown("### 💬 Conversations")
            new_chat_btn = gr.Button("➕ New Chat", variant="primary", size="sm")
            conversation_list = gr.Dataframe(
                headers=["ID", "Preview", "Date"],
                datatype=["str", "str", "str"],
                row_count=8,
                interactive=True  # Allow selection
            )
            selected_conv_display = gr.Textbox(label="Selected", lines=1, interactive=False, visible=True)
            with gr.Row():
                refresh_btn = gr.Button("🔄", size="sm", scale=1)
                export_btn = gr.Button("📤 Export", size="sm", scale=1)
                delete_btn = gr.Button("🗑️ Delete", size="sm", scale=1, variant="stop")

            gr.Markdown("---")
            gr.Markdown("### 🤖 Model")
            model_dropdown = gr.Dropdown(
                choices=list(AVAILABLE_MODELS.keys()),
                value=DEFAULT_MODEL,
                label="Select Model",
                interactive=True
            )
            load_model_btn = gr.Button("Load Model", size="sm")
            model_status = gr.Textbox(label="", lines=1, interactive=False)

            gr.Markdown("---")
            gr.Markdown("### ⚙️ Parameters")
            temperature = gr.Slider(0.1, 2.0, value=0.7, step=0.1, label="Temperature")
            max_tokens = gr.Slider(64, 32000, value=4096, step=256, label="Max Tokens")
            top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top P")

            gr.Markdown("---")
            gr.Markdown("### 🎭 Avatar")
            enable_voice = gr.Checkbox(label="🔊 Voice Output", value=False)
            enable_avatar = gr.Checkbox(label="🎥 Avatar Video", value=False)
            load_musetalk_btn = gr.Button("Load MuseTalk (GPU)", size="sm")
            musetalk_status = gr.Textbox(label="", lines=1, interactive=False, value="Not loaded")

            gr.Markdown("---")
            gr.Markdown("### 🎭 Persona")
            persona_dropdown = gr.Dropdown(
                choices=["default", "professional", "coder", "creative", "flirty"],
                value="default",
                label="Personality Mode",
                interactive=True
            )
            persona_status = gr.Textbox(label="", lines=1, interactive=False, value="Default mode")

            gr.Markdown("---")
            gr.Markdown("### ⏰ Quick Reminder")
            reminder_input = gr.Textbox(
                label="",
                placeholder="e.g., in 30 minutes call mom",
                lines=1
            )
            set_reminder_btn = gr.Button("⏰ Set", size="sm")
            reminder_status = gr.Textbox(label="", lines=1, interactive=False)

        # Main chat area
        with gr.Column(scale=3):
            gr.Markdown("# 🤖 Alexandra AI")

            with gr.Tabs():
                with gr.Tab("💬 Chat"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            chatbot = gr.Chatbot(
                                label="Chat",
                                elem_classes="chat-container"
                            )

                        with gr.Column(scale=1):
                            video = gr.Video(
                                label="Avatar",
                                autoplay=True,
                                
                                elem_classes="avatar-video"
                            )
                            generated_img = gr.Image(label="Generated Image")

                    with gr.Row():
                        user_input = gr.Textbox(
                            label="",
                            placeholder="Type your message... (Shift+Enter for new line)",
                            lines=2,
                            scale=4
                        )
                        image_input = gr.Image(
                            label="📷",
                            type="pil",
                            scale=1,
                            height=80
                        )
                        audio_input = gr.Audio(
                            sources=["microphone"],
                            type="filepath",
                            label="🎤",
                            scale=1
                        )

                    with gr.Row():
                        submit_btn = gr.Button("Send 📤", variant="primary", scale=2)
                        run_code_btn = gr.Button("▶️ Run Code", variant="secondary", scale=1)
                        clear_btn = gr.Button("🗑️ Clear", scale=1)

                    code_output = gr.Markdown(label="Code Output", visible=False)

                with gr.Tab("📎 Documents"):
                    gr.Markdown("### Upload documents for context")
                    file_upload = gr.File(
                        label="Upload PDF, Word, Text, or Code files",
                        file_types=[".pdf", ".docx", ".doc", ".txt", ".md", ".py", ".js", ".json", ".csv", ".html"],
                    )
                    upload_status = gr.Textbox(label="Status", lines=2)
                    clear_docs_btn = gr.Button("Clear Documents", variant="secondary")

                    gr.Markdown("### Current Documents")
                    docs_list = gr.Textbox(label="", lines=5, interactive=False)

                with gr.Tab("🎨 Image Gen"):
                    gr.Markdown("### Generate Images with Flux + Uncensored LoRA")
                    gr.Markdown("*Use 'alexandratitus767' in prompt for her face*")
                    img_prompt = gr.Textbox(label="Prompt", placeholder="alexandratitus767, full body, leaning forward, casual pose", lines=3)
                    with gr.Row():
                        img_steps = gr.Slider(1, 8, value=4, step=1, label="Steps", scale=2)
                        img_aspect = gr.Radio(
                            choices=["square", "portrait", "landscape"],
                            value="portrait",
                            label="Aspect Ratio",
                            scale=2
                        )
                    gr.Markdown("*Portrait = full body (768x1024), Square = headshots (768x768)*")
                    gen_img_btn = gr.Button("🎨 Generate Image", variant="primary")
                    output_image = gr.Image(label="Generated Image")

                with gr.Tab("🖼️ Image Edit"):
                    gr.Markdown("### Transform Images with AI")
                    gr.Markdown("*Upload an image and describe how you want to change it*")
                    with gr.Row():
                        with gr.Column():
                            edit_input_image = gr.Image(label="Input Image", type="filepath")
                            edit_prompt = gr.Textbox(
                                label="Transformation Prompt",
                                placeholder="e.g., make it look like a painting, add sunset lighting...",
                                lines=2
                            )
                            edit_strength = gr.Slider(0.1, 1.0, value=0.7, step=0.1, label="Transformation Strength")
                            edit_btn = gr.Button("🎨 Transform Image", variant="primary")
                        with gr.Column():
                            edit_output = gr.Image(label="Transformed Image")
                    gr.Markdown("*Higher strength = more transformation, lower = subtle changes*")

                with gr.Tab("🔍 Search"):
                    gr.Markdown("### Web Search & News")
                    with gr.Row():
                        with gr.Column():
                            search_input = gr.Textbox(label="Web Search", placeholder="Search the web...")
                            search_btn = gr.Button("🔍 Search", variant="primary")
                            search_results = gr.Markdown()

                        with gr.Column():
                            news_btn = gr.Button("📰 Update News Feeds", variant="secondary")
                            news_status = gr.Textbox(label="Status", lines=3)

                with gr.Tab("📊 Memory"):
                    memory_stats = gr.Textbox(label="Memory Statistics", lines=6, interactive=False)
                    mem_refresh_btn = gr.Button("🔄 Refresh Stats")

                    gr.Markdown("---")
                    gr.Markdown("### 🧠 Known Facts About You")
                    facts_display = gr.Textbox(label="", lines=8, interactive=False)
                    with gr.Row():
                        refresh_facts_btn = gr.Button("🔄 Refresh Facts", scale=1)
                        backfill_btn = gr.Button("📚 Extract Facts from Past Conversations", scale=2)
                    backfill_status = gr.Textbox(label="Status", lines=2, interactive=False)

                with gr.Tab("📥 Import/Export"):
                    gr.Markdown("### Export Conversation")
                    conv_id_input = gr.Textbox(label="Conversation ID", placeholder="e.g., 20241215_143022")
                    export_conv_btn = gr.Button("Export to Markdown")
                    export_file = gr.File(label="Download")
                    export_status = gr.Textbox(label="Status")

                    gr.Markdown("---")
                    gr.Markdown("### Import Conversation")
                    import_file = gr.File(label="Upload JSON", file_types=[".json"])
                    import_btn = gr.Button("Import")
                    import_status = gr.Textbox(label="Status")

                with gr.Tab("📚 Document Library"):
                    gr.Markdown("### RAG Document Search")
                    gr.Markdown("Upload documents (PDF, TXT, JSON, MD) to search with AI")
                    
                    with gr.Row():
                        with gr.Column(scale=2):
                            rag_files = gr.File(label="Upload Documents", file_count="multiple", file_types=[".pdf", ".txt", ".json", ".md"])
                            rag_upload_btn = gr.Button("📤 Add to Library", variant="primary")
                            rag_upload_status = gr.Textbox(label="Upload Status", lines=2)
                        
                        with gr.Column(scale=1):
                            rag_stats_display = gr.Textbox(label="Library Stats", lines=8, interactive=False)
                            rag_refresh_btn = gr.Button("🔄 Refresh Stats")
                            rag_clear_btn = gr.Button("🗑️ Clear All", variant="stop")
                    
                    gr.Markdown("---")
                    gr.Markdown("### Search Documents")
                    with gr.Row():
                        rag_query = gr.Textbox(label="Search Query", placeholder="What are you looking for?", scale=4)
                        rag_n_results = gr.Slider(1, 10, value=5, step=1, label="Results", scale=1)
                    rag_search_btn = gr.Button("🔍 Search", variant="primary")

                    with gr.Row():
                        rag_results_dropdown = gr.Dropdown(label="📄 Found Documents (click to view)", choices=[], interactive=True, scale=3)
                        rag_view_btn = gr.Button("👁️ View Full Document", scale=1)
                    rag_results = gr.Markdown(label="Search Results")
                    rag_doc_viewer = gr.Textbox(label="📖 Document Content", lines=15, interactive=False)

                with gr.Tab("🔍 Epstein Files"):
                    gr.Markdown("### DOJ Epstein Files Search")
                    gr.Markdown("Search 3,951 documents from the DOJ release (8,380 indexed chunks)")

                    with gr.Row():
                        epstein_query = gr.Textbox(label="Search Query", placeholder="e.g., flight logs, Palm Beach, Maxwell, 1996...", scale=4)
                        epstein_n = gr.Slider(1, 20, value=10, step=1, label="Results", scale=1)
                    epstein_search_btn = gr.Button("🔍 Search Epstein Files", variant="primary")

                    with gr.Row():
                        epstein_results_dropdown = gr.Dropdown(label="📄 Found Documents (click to view)", choices=[], interactive=True, scale=3)
                        epstein_view_pdf_btn = gr.Button("👁️ View PDF", variant="primary", scale=1)
                        epstein_download_btn = gr.Button("📥 Download", scale=1)

                    epstein_pdf_viewer = gr.HTML(label="PDF Viewer", value="<p>Select a document and click 'View PDF' to display it here</p>")

                    with gr.Accordion("📋 Search Results & Download", open=False):
                        epstein_pdf_file = gr.File(label="📄 Download PDF")
                        epstein_results = gr.Markdown(label="Search Results")
                        epstein_doc_viewer = gr.Textbox(label="📖 Extracted Text (OCR)", lines=6, interactive=False)

                    gr.Markdown("---")
                    epstein_stats_display = gr.Textbox(label="Collection Stats", lines=4, interactive=False)
                    epstein_stats_btn = gr.Button("📊 Refresh Stats")

    # ============ EVENT HANDLERS ============

    # Chat
    submit_btn.click(
        chat,
        inputs=[user_input, chatbot, enable_voice, enable_avatar, temperature, max_tokens, top_p, image_input],
        outputs=[chatbot, video, user_input, generated_img]
    )
    user_input.submit(
        chat,
        inputs=[user_input, chatbot, enable_voice, enable_avatar, temperature, max_tokens, top_p, image_input],
        outputs=[chatbot, video, user_input, generated_img]
    )

    # Voice input
    audio_input.stop_recording(
        voice_chat,
        inputs=[audio_input, chatbot, enable_voice, enable_avatar, temperature, max_tokens, top_p],
        outputs=[chatbot, video, user_input, generated_img]
    )

    # Run code
    def show_code_output(history):
        result = run_code_from_response(history)
        return gr.update(value=result, visible=True)

    run_code_btn.click(show_code_output, inputs=[chatbot], outputs=[code_output])

    # Clear
    clear_btn.click(lambda: ([], None, "", None, gr.update(visible=False)), outputs=[chatbot, video, user_input, generated_img, code_output])

    # New chat
    new_chat_btn.click(new_conversation, outputs=[chatbot, gr.State()])

    # Model loading
    load_model_btn.click(load_model, inputs=[model_dropdown], outputs=[model_status])

    # MuseTalk loading
    load_musetalk_btn.click(load_musetalk_models, outputs=[musetalk_status])

    # Persona switching
    def switch_persona(persona_key):
        if alex_features:
            result = alex_features.personas.set_persona(persona_key)
            return result
        return f"Switched to {persona_key} mode"
    persona_dropdown.change(switch_persona, inputs=[persona_dropdown], outputs=[persona_status])

    # Quick reminder
    def set_quick_reminder(text):
        if not text.strip():
            return "Enter a reminder"
        if alex_features:
            parsed = alex_features.reminders.parse_reminder(f"remind me {text}")
            if parsed:
                message, remind_at = parsed
                return alex_features.reminders.add_reminder(message, remind_at)
            return "Couldn't parse. Try: 'in 30 minutes call mom'"
        return "Reminders not available"
    set_reminder_btn.click(set_quick_reminder, inputs=[reminder_input], outputs=[reminder_status])

    # Image editing
    def edit_image(input_path, prompt, strength):
        if not input_path or not prompt.strip():
            return None
        if FEATURES_AVAILABLE:
            result = ImageEditor.img2img(input_path, prompt, strength)
            if os.path.exists(str(result)):
                return result
            return None
        return None
    edit_btn.click(edit_image, inputs=[edit_input_image, edit_prompt, edit_strength], outputs=[edit_output])

    # Refresh conversations
    def refresh_convs():
        return list_conversations()
    refresh_btn.click(refresh_convs, outputs=[conversation_list])

    # Select conversation from list
    def select_conversation(evt: gr.SelectData):
        global current_conversation_id
        try:
            # evt.value contains the clicked cell value
            # evt.index is [row, col]
            if evt.index is not None:
                row_idx = evt.index[0]
                # Get conversation list and extract ID
                convs = list_conversations()
                if row_idx < len(convs):
                    conv_id = convs[row_idx][0]  # First column is ID
                    current_conversation_id = conv_id
                    # Load the conversation into chat
                    conv_data = load_conversation(conv_id)
                    if conv_data:
                        return f"✓ {conv_id}", history_to_messages(conv_data.get("messages", []))
            return "No selection", []
        except Exception as e:
            print(f"[ERROR] Selection failed: {e}")
            return f"Error: {e}", []
    conversation_list.select(select_conversation, outputs=[selected_conv_display, chatbot])

    # Export from list - creates downloadable file
    def export_selected():
        global current_conversation_id
        if current_conversation_id:
            path, status = export_conversation(current_conversation_id)
            return path, status
        return None, "No active conversation. Select a conversation first."
    export_btn.click(export_selected, outputs=[export_file, model_status])

    # Delete conversation
    def delete_selected():
        global current_conversation_id
        if current_conversation_id:
            result = delete_conversation(current_conversation_id)
            current_conversation_id = None
            return list_conversations(), [], result
        return list_conversations(), [], "No conversation selected"
    delete_btn.click(delete_selected, outputs=[conversation_list, chatbot, model_status])

    # Document upload
    file_upload.upload(process_uploaded_file, inputs=[file_upload], outputs=[upload_status])

    def list_docs():
        if not uploaded_documents:
            return "No documents uploaded"
        return "\n".join([f"• {name} ({len(content)} chars)" for name, content in uploaded_documents.items()])
    file_upload.upload(list_docs, outputs=[docs_list])
    clear_docs_btn.click(clear_documents, outputs=[upload_status])
    clear_docs_btn.click(list_docs, outputs=[docs_list])

    # Image generation
    def gen_image_ui(prompt, steps):
        if not prompt.strip():
            return None
        return generate_image(prompt, steps)
    def gen_image_with_aspect(prompt, steps, aspect):
        if not prompt.strip():
            return None
        return generate_image(prompt, steps, aspect)
    gen_img_btn.click(gen_image_with_aspect, inputs=[img_prompt, img_steps, img_aspect], outputs=[output_image])

    # Memory stats
    def get_mem_stats():
        mem = init_memory()
        s = mem.stats()
        return f"📊 Memory Statistics:\n\n• Facts: {s['facts']}\n• Conversations: {s['conversations']}\n• Knowledge: {s['knowledge']}\n• News: {s['news']}\n• Uploaded docs: {len(uploaded_documents)}"
    mem_refresh_btn.click(get_mem_stats, outputs=[memory_stats])

    # Facts display
    def get_facts():
        mem = init_memory()
        if mem.facts.count() == 0:
            return "No facts learned yet. Chat with Alexandra and she'll remember things about you!"
        results = mem.facts.get(limit=50)
        if not results['documents']:
            return "No facts found"
        facts_list = []
        for doc in results['documents']:
            facts_list.append(f"• {doc}")
        return "\n".join(facts_list)
    refresh_facts_btn.click(get_facts, outputs=[facts_display])

    # Backfill facts from existing conversations
    def backfill_facts(progress=gr.Progress()):
        if not FEATURES_AVAILABLE:
            return "Feature not available"

        mem = init_memory()
        total_facts = 0

        # Get all conversations from the conversations directory
        import glob
        conv_files = sorted(glob.glob(f"{CONVERSATIONS_DIR}/*.json"), key=os.path.getmtime, reverse=True)

        progress(0, desc=f"Processing {len(conv_files)} conversations...")

        for i, conv_file in enumerate(conv_files[:10]):  # Limit to 10 most recent
            try:
                with open(conv_file) as f:
                    data = json.load(f)

                messages = data.get("messages", [])
                for msg in messages:
                    if isinstance(msg, (list, tuple)) and len(msg) == 2:
                        user_msg = msg[0]
                        assistant_msg = msg[1]

                        # Handle new Gradio format
                        if isinstance(user_msg, list) and len(user_msg) > 0:
                            user_msg = user_msg[0].get('text', '') if isinstance(user_msg[0], dict) else str(user_msg[0])
                        if isinstance(assistant_msg, list) and len(assistant_msg) > 0:
                            assistant_msg = assistant_msg[0].get('text', '') if isinstance(assistant_msg[0], dict) else str(assistant_msg[0])

                        if isinstance(user_msg, str) and isinstance(assistant_msg, str):
                            facts = FactExtractor.extract_facts(user_msg, assistant_msg)
                            if facts:
                                FactExtractor.save_facts(facts, mem)
                                total_facts += len(facts)

                progress((i + 1) / min(len(conv_files), 10), desc=f"Processed {i+1} conversations, found {total_facts} facts")

            except Exception as e:
                print(f"[BACKFILL] Error processing {conv_file}: {e}")

        return f"✅ Extracted {total_facts} facts from {min(len(conv_files), 10)} conversations"

    backfill_btn.click(backfill_facts, outputs=[backfill_status])

    # Search
    def web_search(query):
        if not query.strip():
            return "Enter a search query"
        mem = init_memory()
        results = mem.search_web_duckduckgo(query, max_results=5)
        if not results:
            return "No results found"
        output = ""
        for r in results:
            output += f"**{r['title']}**\n{r['snippet']}\n[Link]({r['url']})\n\n"
        return output
    search_btn.click(web_search, inputs=[search_input], outputs=[search_results])

    # News
    def update_news():
        mem = init_memory()
        count = mem.update_news_feeds()
        return f"Fetched {count} articles\nTotal: {mem.stats()['news']}"
    news_btn.click(update_news, outputs=[news_status])

    # Export/Import
    export_conv_btn.click(export_conversation, inputs=[conv_id_input], outputs=[export_file, export_status])

    def do_import(file):
        history, status = import_conversation(file)
        return history, status
    import_btn.click(do_import, inputs=[import_file], outputs=[chatbot, import_status])

    # RAG Document Library
    if RAG_AVAILABLE:
        rag_upload_btn.click(rag_add_files, inputs=[rag_files], outputs=[rag_upload_status])
        rag_search_btn.click(rag_search, inputs=[rag_query, rag_n_results], outputs=[rag_results, rag_results_dropdown])
        rag_view_btn.click(rag_view_document, inputs=[rag_results_dropdown], outputs=[rag_doc_viewer])
        rag_results_dropdown.change(rag_view_document, inputs=[rag_results_dropdown], outputs=[rag_doc_viewer])
        rag_refresh_btn.click(rag_get_stats, outputs=[rag_stats_display])
        rag_clear_btn.click(rag_clear, outputs=[rag_upload_status])

    # Epstein Files Search
    if epstein_rag:
        epstein_search_btn.click(epstein_search, inputs=[epstein_query, epstein_n], outputs=[epstein_results, epstein_results_dropdown])
        epstein_view_pdf_btn.click(epstein_view_pdf, inputs=[epstein_results_dropdown], outputs=[epstein_pdf_viewer])
        epstein_results_dropdown.change(epstein_view_document, inputs=[epstein_results_dropdown], outputs=[epstein_doc_viewer])
        epstein_download_btn.click(epstein_get_pdf, inputs=[epstein_results_dropdown], outputs=[epstein_pdf_file])
        epstein_stats_btn.click(epstein_stats, outputs=[epstein_stats_display])

    # Load on start
    app.load(refresh_convs, outputs=[conversation_list])
    app.load(get_mem_stats, outputs=[memory_stats])

if __name__ == "__main__":
    print("="*60)
    print("Alexandra AI - Full Featured Interface")
    print("="*60)
    print(f"Available models: {list(AVAILABLE_MODELS.keys())}")
    print(f"Avatar: {AVATAR_IMAGE}")
    print("="*60)
    print("Launching Gradio server...")
    # Launch with prevent_thread_lock and manual keep-alive
    # This fixes Gradio hanging issues in container environments
    app.launch(server_name="0.0.0.0", server_port=7860, share=True, prevent_thread_lock=True, allowed_paths=["/"])
    print("Server running on http://0.0.0.0:7860")
    import time
    while True:
        time.sleep(1)
