"""
Alexandra AI - Telegram Bot
Chat with Alexandra on Telegram
"""

import os
import sys
import asyncio
import aiohttp
import logging
from io import BytesIO

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    CallbackQueryHandler, ContextTypes, filters
)

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from alexandra_config import TELEGRAM_TOKEN, API_HOST, API_PORT, PERSONALITY_MODES

# Logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# API base URL
API_URL = f"http://localhost:{API_PORT}"

# User preferences (in-memory, use database for production)
user_prefs = {}

# ============== Helper Functions ==============

async def call_api(endpoint, method="GET", data=None):
    """Call Alexandra API"""
    async with aiohttp.ClientSession() as session:
        url = f"{API_URL}{endpoint}"
        try:
            if method == "GET":
                async with session.get(url) as resp:
                    return await resp.json()
            elif method == "POST":
                async with session.post(url, json=data) as resp:
                    return await resp.json()
        except Exception as e:
            logger.error(f"API call failed: {e}")
            return None

async def wait_for_video(job_id, timeout=120):
    """Poll for video completion"""
    for _ in range(timeout):
        result = await call_api(f"/job/{job_id}")
        if result and result.get("status") == "completed":
            return result.get("result")
        elif result and result.get("status") == "failed":
            return None
        await asyncio.sleep(1)
    return None

async def download_video(video_url):
    """Download video from API"""
    full_url = f"{API_URL}{video_url}"
    async with aiohttp.ClientSession() as session:
        async with session.get(full_url) as resp:
            if resp.status == 200:
                return await resp.read()
    return None

def get_user_personality(user_id):
    """Get user's preferred personality"""
    return user_prefs.get(user_id, {}).get("personality", "default")

# ============== Command Handlers ==============

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command"""
    welcome_text = """
Hello! I'm **Alexandra AI** - your personal AI companion!

**Commands:**
- Just send me a message to chat!
- /video <text> - Generate a video of me saying something
- /personality - Change my personality mode
- /status - Check my system status
- /help - Show this help message

I can respond with both text and video! Video generation takes a bit longer, so be patient!
    """
    await update.message.reply_text(welcome_text, parse_mode='Markdown')

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /help command"""
    help_text = """
**Alexandra AI Help**

**Chat:** Just send me any message!

**Commands:**
• /video <text> - Make me say something in a video
• /personality - Choose my personality mode
• /status - Check if all systems are working
• /help - Show this help

**Personality Modes:**
• Default - Friendly and balanced
• Professional - Formal and precise
• Casual - Fun and relaxed
• Creative - Imaginative and expressive
• Teacher - Patient and educational

**Tips:**
• I remember our conversation context!
• Videos take 30-60 seconds to generate
• You can send voice messages too!
    """
    await update.message.reply_text(help_text, parse_mode='Markdown')

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /status command"""
    await update.message.reply_text("Checking systems...")

    health = await call_api("/health")
    if not health:
        await update.message.reply_text("Unable to reach API server. Is it running?")
        return

    components = health.get("components", {})
    status_text = "**System Status**\n\n"
    status_text += f"{'✅' if components.get('avatar_image') else '❌'} Avatar Image\n"
    status_text += f"{'✅' if components.get('voice_model') else '❌'} Voice Model (F5-TTS)\n"
    status_text += f"{'✅' if components.get('sadtalker') else '❌'} SadTalker\n"
    status_text += f"{'✅' if components.get('musetalk') else '❌'} MuseTalk\n"

    await update.message.reply_text(status_text, parse_mode='Markdown')

async def personality_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /personality command"""
    keyboard = [
        [InlineKeyboardButton(config["name"], callback_data=f"personality_{name}")]
        for name, config in PERSONALITY_MODES.items()
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(
        "Choose a personality mode:",
        reply_markup=reply_markup
    )

async def personality_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle personality selection"""
    query = update.callback_query
    await query.answer()

    personality = query.data.replace("personality_", "")
    user_id = query.from_user.id

    if user_id not in user_prefs:
        user_prefs[user_id] = {}
    user_prefs[user_id]["personality"] = personality

    personality_name = PERSONALITY_MODES.get(personality, {}).get("name", personality)
    await query.edit_message_text(f"Personality set to: **{personality_name}**", parse_mode='Markdown')

async def video_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /video command"""
    if not context.args:
        await update.message.reply_text("Please provide text for the video.\nUsage: /video Hello, this is a test!")
        return

    text = " ".join(context.args)
    await update.message.reply_text(f"Generating video for: \"{text[:50]}{'...' if len(text) > 50 else ''}\"\nThis may take a minute...")

    # Generate video
    response = await call_api("/generate-video", method="POST", data={
        "text": text,
        "avatar": "default"
    })

    if not response or not response.get("job_id"):
        await update.message.reply_text("Failed to start video generation.")
        return

    # Wait for video
    video_result = await wait_for_video(response["job_id"], timeout=120)

    if video_result and video_result.get("video_url"):
        video_data = await download_video(video_result["video_url"])
        if video_data:
            await update.message.reply_video(
                video=BytesIO(video_data),
                filename="alexandra_video.mp4",
                caption="Here's your video!"
            )
        else:
            await update.message.reply_text("Failed to download video.")
    else:
        await update.message.reply_text("Video generation failed or timed out.")

# ============== Message Handlers ==============

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle regular text messages"""
    user_id = update.message.from_user.id
    text = update.message.text

    # Send typing indicator
    await update.message.chat.send_action("typing")

    # Get user's personality preference
    personality = get_user_personality(user_id)

    # Call API
    response = await call_api("/chat", method="POST", data={
        "message": text,
        "personality": personality,
        "response_length": "medium"
    })

    if not response:
        await update.message.reply_text("Sorry, I couldn't process your message. Is the API server running?")
        return

    # Send text response
    await update.message.reply_text(response.get("text", "I'm not sure how to respond to that."))

    # Optionally generate video (can be slow)
    if response.get("job_id"):
        # Check if video generation completes quickly
        video_result = await wait_for_video(response["job_id"], timeout=30)
        if video_result and video_result.get("video_url"):
            video_data = await download_video(video_result["video_url"])
            if video_data:
                await update.message.reply_video(
                    video=BytesIO(video_data),
                    filename="response.mp4"
                )

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle voice messages"""
    await update.message.reply_text("Processing your voice message...")

    # Download voice file
    voice_file = await update.message.voice.get_file()
    voice_data = await voice_file.download_as_bytearray()

    # Save temporarily
    temp_path = f"/tmp/telegram_voice_{update.message.message_id}.ogg"
    with open(temp_path, "wb") as f:
        f.write(voice_data)

    # Transcribe using API
    async with aiohttp.ClientSession() as session:
        with open(temp_path, "rb") as f:
            form = aiohttp.FormData()
            form.add_field('audio', f, filename='voice.ogg')
            async with session.post(f"{API_URL}/transcribe", data=form) as resp:
                result = await resp.json()

    # Clean up
    os.unlink(temp_path)

    if result.get("text"):
        transcribed_text = result["text"]
        await update.message.reply_text(f"I heard: \"{transcribed_text}\"")

        # Process as regular message
        await update.message.chat.send_action("typing")

        response = await call_api("/chat", method="POST", data={
            "message": transcribed_text,
            "personality": get_user_personality(update.message.from_user.id)
        })

        if response:
            await update.message.reply_text(response.get("text", "I couldn't understand that."))
    else:
        await update.message.reply_text("Sorry, I couldn't understand the voice message.")

# ============== Main ==============

def run_bot():
    """Run the Telegram bot"""
    if not TELEGRAM_TOKEN:
        print("[Telegram] Error: TELEGRAM_TOKEN not set!")
        print("[Telegram] Set it with: export TELEGRAM_TOKEN='your_token_here'")
        print("[Telegram] Get a token from @BotFather on Telegram")
        return

    print("[Telegram] Starting Alexandra Telegram Bot...")
    print("[Telegram] Make sure the API server is running!")

    # Create application
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    # Add handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(CommandHandler("personality", personality_command))
    application.add_handler(CommandHandler("video", video_command))
    application.add_handler(CallbackQueryHandler(personality_callback, pattern="^personality_"))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(MessageHandler(filters.VOICE, handle_voice))

    # Start bot
    print("[Telegram] Bot is running! Press Ctrl+C to stop.")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    run_bot()
