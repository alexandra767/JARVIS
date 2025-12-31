"""
Alexandra AI - Discord Bot
Chat with Alexandra in Discord servers
"""

import os
import sys
import asyncio
import aiohttp
import discord
from discord.ext import commands
from discord import app_commands

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from alexandra_config import DISCORD_TOKEN, API_HOST, API_PORT

# Bot setup
intents = discord.Intents.default()
intents.message_content = True
intents.voice_states = True

bot = commands.Bot(command_prefix="!", intents=intents)

# API base URL
API_URL = f"http://localhost:{API_PORT}"

# ============== Helper Functions ==============

async def call_api(endpoint, method="GET", data=None):
    """Call Alexandra API"""
    async with aiohttp.ClientSession() as session:
        url = f"{API_URL}{endpoint}"
        if method == "GET":
            async with session.get(url) as resp:
                return await resp.json()
        elif method == "POST":
            async with session.post(url, json=data) as resp:
                return await resp.json()

async def wait_for_video(job_id, timeout=120):
    """Poll for video completion"""
    for _ in range(timeout):
        result = await call_api(f"/job/{job_id}")
        if result["status"] == "completed":
            return result["result"]
        elif result["status"] == "failed":
            return None
        await asyncio.sleep(1)
    return None

# ============== Events ==============

@bot.event
async def on_ready():
    print(f"[Discord] Logged in as {bot.user}")
    print(f"[Discord] Connected to {len(bot.guilds)} servers")
    try:
        synced = await bot.tree.sync()
        print(f"[Discord] Synced {len(synced)} commands")
    except Exception as e:
        print(f"[Discord] Failed to sync commands: {e}")

@bot.event
async def on_message(message):
    """Respond when mentioned"""
    if message.author == bot.user:
        return

    # Check if bot is mentioned
    if bot.user in message.mentions:
        # Remove mention from message
        content = message.content.replace(f"<@{bot.user.id}>", "").strip()

        if not content:
            await message.reply("Hey! How can I help you? Just mention me with your question!")
            return

        # Show typing indicator
        async with message.channel.typing():
            try:
                # Call API
                response = await call_api("/chat", method="POST", data={
                    "message": content,
                    "personality": "casual",
                    "response_length": "medium"
                })

                # Send text response immediately
                await message.reply(response["text"])

                # Wait for video in background
                if response.get("job_id"):
                    video_result = await wait_for_video(response["job_id"])
                    if video_result and video_result.get("video_url"):
                        # Download and send video
                        video_url = f"{API_URL}{video_result['video_url']}"
                        async with aiohttp.ClientSession() as session:
                            async with session.get(video_url) as resp:
                                if resp.status == 200:
                                    video_data = await resp.read()
                                    await message.channel.send(
                                        file=discord.File(
                                            fp=asyncio.BytesIO(video_data),
                                            filename="alexandra_response.mp4"
                                        )
                                    )

            except Exception as e:
                await message.reply(f"Sorry, I encountered an error: {str(e)}")

    await bot.process_commands(message)

# ============== Slash Commands ==============

@bot.tree.command(name="ask", description="Ask Alexandra a question")
@app_commands.describe(question="Your question for Alexandra")
async def ask_command(interaction: discord.Interaction, question: str):
    """Ask Alexandra a question"""
    await interaction.response.defer(thinking=True)

    try:
        response = await call_api("/chat", method="POST", data={
            "message": question,
            "personality": "default",
            "response_length": "medium"
        })

        await interaction.followup.send(response["text"])

        # Send video if available
        if response.get("job_id"):
            video_result = await wait_for_video(response["job_id"], timeout=60)
            if video_result and video_result.get("video_url"):
                video_url = f"{API_URL}{video_result['video_url']}"
                async with aiohttp.ClientSession() as session:
                    async with session.get(video_url) as resp:
                        if resp.status == 200:
                            video_data = await resp.read()
                            await interaction.channel.send(
                                file=discord.File(
                                    fp=asyncio.BytesIO(video_data),
                                    filename="alexandra_response.mp4"
                                )
                            )

    except Exception as e:
        await interaction.followup.send(f"Error: {str(e)}")

@bot.tree.command(name="personality", description="Change Alexandra's personality mode")
@app_commands.describe(mode="Personality mode to use")
@app_commands.choices(mode=[
    app_commands.Choice(name="Default", value="default"),
    app_commands.Choice(name="Professional", value="professional"),
    app_commands.Choice(name="Casual", value="casual"),
    app_commands.Choice(name="Creative", value="creative"),
    app_commands.Choice(name="Teacher", value="teacher"),
])
async def personality_command(interaction: discord.Interaction, mode: str):
    """Change personality mode"""
    # Store in user preferences (simplified - would use database in production)
    await interaction.response.send_message(f"Personality mode set to: **{mode}**")

@bot.tree.command(name="video", description="Generate a video of Alexandra saying something")
@app_commands.describe(text="What should Alexandra say?")
async def video_command(interaction: discord.Interaction, text: str):
    """Generate a video"""
    await interaction.response.defer(thinking=True)

    try:
        response = await call_api("/generate-video", method="POST", data={
            "text": text,
            "avatar": "default"
        })

        await interaction.followup.send(f"Generating video... (Job ID: {response['job_id']})")

        video_result = await wait_for_video(response["job_id"], timeout=120)

        if video_result and video_result.get("video_url"):
            video_url = f"{API_URL}{video_result['video_url']}"
            async with aiohttp.ClientSession() as session:
                async with session.get(video_url) as resp:
                    if resp.status == 200:
                        video_data = await resp.read()
                        await interaction.channel.send(
                            content=f"Here's your video!",
                            file=discord.File(
                                fp=asyncio.BytesIO(video_data),
                                filename="alexandra_video.mp4"
                            )
                        )
        else:
            await interaction.channel.send("Sorry, video generation failed.")

    except Exception as e:
        await interaction.followup.send(f"Error: {str(e)}")

@bot.tree.command(name="status", description="Check Alexandra's status")
async def status_command(interaction: discord.Interaction):
    """Check system status"""
    try:
        health = await call_api("/health")
        components = health.get("components", {})

        status_text = "**Alexandra AI Status**\n"
        status_text += f"- Avatar: {'Available' if components.get('avatar_image') else 'Missing'}\n"
        status_text += f"- Voice Model: {'Available' if components.get('voice_model') else 'Missing'}\n"
        status_text += f"- SadTalker: {'Available' if components.get('sadtalker') else 'Missing'}\n"
        status_text += f"- MuseTalk: {'Available' if components.get('musetalk') else 'Missing'}\n"

        await interaction.response.send_message(status_text)
    except Exception as e:
        await interaction.response.send_message(f"Error checking status: {str(e)}")

# ============== Text Commands ==============

@bot.command(name="ask")
async def ask_text_command(ctx, *, question: str):
    """Text command version of ask"""
    async with ctx.typing():
        try:
            response = await call_api("/chat", method="POST", data={
                "message": question,
                "personality": "default"
            })
            await ctx.reply(response["text"])
        except Exception as e:
            await ctx.reply(f"Error: {str(e)}")

# ============== Main ==============

def run_bot():
    """Run the Discord bot"""
    if not DISCORD_TOKEN:
        print("[Discord] Error: DISCORD_TOKEN not set!")
        print("[Discord] Set it with: export DISCORD_TOKEN='your_token_here'")
        return

    print("[Discord] Starting Alexandra Discord Bot...")
    print("[Discord] Make sure the API server is running!")
    bot.run(DISCORD_TOKEN)

if __name__ == "__main__":
    run_bot()
