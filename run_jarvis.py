#!/usr/bin/env python3
"""
Simple launcher for Jarvis Mode
Run this to start the full Jarvis experience
"""

import asyncio
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║       ██╗ █████╗ ██████╗ ██╗   ██╗██╗███████╗                   ║
║       ██║██╔══██╗██╔══██╗██║   ██║██║██╔════╝                   ║
║       ██║███████║██████╔╝██║   ██║██║███████╗                   ║
║  ██   ██║██╔══██║██╔══██╗╚██╗ ██╔╝██║╚════██║                   ║
║  ╚█████╔╝██║  ██║██║  ██║ ╚████╔╝ ██║███████║                   ║
║   ╚════╝ ╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝  ╚═╝╚══════╝                   ║
║                                                                  ║
║               Alexandra AI - Jarvis Mode                         ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
    """)

def print_controls():
    print("""
┌─────────────────────────────────────────────────────────────────┐
│  CONTROLS                                                        │
├─────────────────────────────────────────────────────────────────┤
│  Voice:                                                          │
│    • Just speak - always listening                               │
│    • Say "Hey Alexandra" or "Hey Jarvis" to wake                │
│    • Speak while she's talking to interrupt                      │
│                                                                  │
│  Hand Gestures (if camera enabled):                              │
│    • Point index finger - move cursor                            │
│    • Pinch (thumb + index) - click                              │
│    • Open palm - stop/interrupt                                  │
│    • Thumbs up - confirm                                         │
│    • Thumbs down - cancel                                        │
│                                                                  │
│  Commands:                                                       │
│    • "What do you see?" - camera vision                         │
│    • "What am I holding?" - object identification               │
│    • "Turn on/off the lights" - smart home                      │
│    • "Go to amazon and..." - browser automation                 │
│                                                                  │
│  Press Ctrl+C to exit                                            │
└─────────────────────────────────────────────────────────────────┘
    """)


async def main():
    print_banner()

    # Configuration options
    print("\nConfiguration:")
    print("  1. Full mode (voice + vision + hands + browser)")
    print("  2. Voice only (minimal resources)")
    print("  3. Voice + Vision")
    print("  4. Custom")
    print("")

    choice = input("Select mode [1]: ").strip() or "1"

    from jarvis_mode import JarvisMode, JarvisConfig

    if choice == "1":
        config = JarvisConfig(
            whisper_model="base",
            vision_enabled=True,
            hand_tracking_enabled=True,
            browser_enabled=True,
            always_listening=True,
        )
    elif choice == "2":
        config = JarvisConfig(
            whisper_model="base",
            vision_enabled=False,
            hand_tracking_enabled=False,
            browser_enabled=False,
            always_listening=True,
        )
    elif choice == "3":
        config = JarvisConfig(
            whisper_model="base",
            vision_enabled=True,
            hand_tracking_enabled=False,
            browser_enabled=False,
            always_listening=True,
        )
    else:
        # Custom configuration
        print("\nCustom configuration:")
        vision = input("Enable vision? [y/N]: ").lower() == 'y'
        hands = input("Enable hand tracking? [y/N]: ").lower() == 'y'
        browser = input("Enable browser? [y/N]: ").lower() == 'y'
        whisper = input("Whisper model (tiny/base/small/medium/large) [base]: ").strip() or "base"

        config = JarvisConfig(
            whisper_model=whisper,
            vision_enabled=vision,
            hand_tracking_enabled=hands,
            browser_enabled=browser,
            always_listening=True,
        )

    print_controls()

    print("\nStarting Jarvis Mode...")
    print("(This may take a moment to load models)")
    print("")

    jarvis = JarvisMode(config)

    # Setup callbacks for feedback
    def on_listening():
        print("🎤 Listening...", end='\r')

    def on_thinking():
        print("🤔 Thinking...  ", end='\r')

    def on_speaking():
        print("🗣️  Speaking... ", end='\r')

    def on_response(text):
        print(f"\n💬 Alexandra: {text}\n")

    jarvis.on_listening = on_listening
    jarvis.on_thinking = on_thinking
    jarvis.on_speaking = on_speaking
    jarvis.on_response = on_response

    await jarvis.start()


def run_test_mode():
    """Run quick tests of individual components"""
    print("\nTest Mode - Select component to test:")
    print("  1. Voice Input (Whisper)")
    print("  2. Voice Output (TTS)")
    print("  3. Vision (Camera)")
    print("  4. Hand Tracking")
    print("  5. Full Voice Loop")
    print("")

    choice = input("Select [1]: ").strip() or "1"

    if choice == "1":
        from jarvis_voice import test_voice_input
        asyncio.run(test_voice_input())
    elif choice == "2":
        from jarvis_voice import test_voice_output
        asyncio.run(test_voice_output())
    elif choice == "3":
        from jarvis_vision import test_vision
        asyncio.run(test_vision())
    elif choice == "4":
        from jarvis_hands import test_hand_tracking
        asyncio.run(test_hand_tracking())
    elif choice == "5":
        from jarvis_voice import test_full_system
        asyncio.run(test_full_system())


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        run_test_mode()
    else:
        try:
            asyncio.run(main())
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
