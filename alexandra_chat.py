#!/usr/bin/env python3
"""
Alexandra AI - Unified Chat Interface
Automatically routes questions to the best specialized knowledge domain
"""

import os
import sys
import argparse
import json
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from alexandra_router import (
    AlexandraRouter, Domain, TopicClassifier,
    get_router, classify, test_classifier
)


class AlexandraChat:
    """Interactive chat interface for Alexandra with automatic routing"""

    def __init__(self,
                 base_model_path: str = "/models/Qwen2.5-72B-Instruct",
                 lora_base_dir: str = "/home/alexandratitus767/ai-clone-training/my-output",
                 auto_load: bool = True,
                 show_routing: bool = True):
        """
        Initialize Alexandra Chat

        Args:
            base_model_path: Path to base Qwen model
            lora_base_dir: Directory containing domain-specific LoRA adapters
            auto_load: Whether to load the model immediately
            show_routing: Whether to show routing decisions
        """
        self.show_routing = show_routing
        self.router = AlexandraRouter(
            base_model_path=base_model_path,
            lora_base_dir=lora_base_dir,
            auto_load_model=auto_load
        )
        self.chat_log = []
        self.session_start = datetime.now()

    def print_header(self):
        """Print welcome header"""
        print("\n" + "=" * 60)
        print("   Alexandra AI - Intelligent Multi-Domain Assistant")
        print("=" * 60)
        print("\nI can help you with:")
        print("  - Travel advice and trip planning")
        print("  - Coding and programming questions")
        print("  - Legal information")
        print("  - Health and medical information")
        print("  - News and current events")
        print("  - General questions")
        print("\nCommands:")
        print("  /status  - Show current status")
        print("  /domain  - Show/set current domain")
        print("  /history - Show conversation history")
        print("  /clear   - Clear conversation")
        print("  /save    - Save conversation to file")
        print("  /help    - Show this help")
        print("  /quit    - Exit")
        print("-" * 60)

    def print_status(self):
        """Print current status"""
        status = self.router.get_status()
        print("\n--- Alexandra Status ---")
        print(f"Model loaded: {'Yes' if status['model_loaded'] else 'No'}")
        print(f"Current domain: {status['current_domain']}")
        print(f"Current LoRA: {status['current_lora'] or 'None (base model)'}")
        print(f"Conversation length: {status['conversation_length']} messages")
        print("\nAvailable domain experts:")
        for domain, available in status['available_loras'].items():
            icon = "[READY]" if available else "[TRAIN]"
            print(f"  {icon} {domain}")
        print("-" * 25)

    def print_history(self):
        """Print conversation history"""
        if not self.chat_log:
            print("\nNo conversation history yet.")
            return

        print("\n--- Conversation History ---")
        for entry in self.chat_log:
            role = "You" if entry['role'] == 'user' else "Alexandra"
            domain = f" [{entry.get('domain', 'general')}]" if entry['role'] == 'assistant' else ""
            print(f"\n{role}{domain}:")
            print(f"  {entry['content'][:200]}{'...' if len(entry['content']) > 200 else ''}")
        print("-" * 25)

    def save_conversation(self, filename: str = None):
        """Save conversation to file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"alexandra_chat_{timestamp}.json"

        filepath = Path(filename)
        data = {
            "session_start": self.session_start.isoformat(),
            "session_end": datetime.now().isoformat(),
            "messages": self.chat_log
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\nConversation saved to: {filepath}")

    def handle_command(self, command: str) -> bool:
        """
        Handle a slash command

        Returns True if should continue, False if should exit
        """
        cmd = command.lower().strip()

        if cmd in ['/quit', '/exit', '/q']:
            print("\nGoodbye! Thanks for chatting with Alexandra.")
            return False

        elif cmd == '/status':
            self.print_status()

        elif cmd == '/help':
            self.print_header()

        elif cmd == '/history':
            self.print_history()

        elif cmd == '/clear':
            self.router.reset_conversation()
            self.chat_log = []
            print("\nConversation cleared.")

        elif cmd.startswith('/save'):
            parts = cmd.split(maxsplit=1)
            filename = parts[1] if len(parts) > 1 else None
            self.save_conversation(filename)

        elif cmd.startswith('/domain'):
            parts = cmd.split(maxsplit=1)
            if len(parts) == 1:
                print(f"\nCurrent domain: {self.router.current_domain.value}")
                print("Available domains: general, travel, coding, legal, medical, news")
                print("Usage: /domain <name> to force a domain")
            else:
                domain_name = parts[1].lower()
                try:
                    domain = Domain(domain_name)
                    self.router.current_domain = domain
                    print(f"\nDomain set to: {domain.value}")
                except ValueError:
                    print(f"\nUnknown domain: {domain_name}")
                    print("Available: general, travel, coding, legal, medical, news")

        elif cmd == '/routing':
            self.show_routing = not self.show_routing
            print(f"\nRouting display: {'ON' if self.show_routing else 'OFF'}")

        else:
            print(f"\nUnknown command: {command}")
            print("Type /help for available commands")

        return True

    def chat(self, question: str) -> str:
        """Process a question and return response"""
        # First classify
        classifier = TopicClassifier()
        domain, confidence = classifier.classify(question)

        if self.show_routing:
            confidence_bar = "█" * int(confidence * 10) + "░" * (10 - int(confidence * 10))
            print(f"\n[Routing to: {domain.value}] [{confidence_bar}] {confidence:.0%}")

        # Get response
        result = self.router.route_and_respond(question)

        # Log
        self.chat_log.append({
            "role": "user",
            "content": question,
            "timestamp": datetime.now().isoformat()
        })
        self.chat_log.append({
            "role": "assistant",
            "content": result['response'],
            "domain": result['domain'],
            "confidence": result['confidence'],
            "timestamp": datetime.now().isoformat()
        })

        return result['response']

    def run_interactive(self):
        """Run interactive chat session"""
        self.print_header()
        self.print_status()

        print("\n")

        while True:
            try:
                user_input = input("You: ").strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.startswith('/'):
                    if not self.handle_command(user_input):
                        break
                    continue

                # Get response
                print("\nAlexandra: ", end="", flush=True)
                response = self.chat(user_input)
                print(response)
                print()

            except KeyboardInterrupt:
                print("\n\nInterrupted. Type /quit to exit or continue chatting.")

            except Exception as e:
                print(f"\nError: {e}")
                print("Type /help for commands or continue chatting.")


class OfflineChat:
    """Lightweight chat interface that doesn't require the model loaded"""

    def __init__(self):
        self.classifier = TopicClassifier()

    def classify_interactive(self):
        """Interactive classification mode"""
        print("\n" + "=" * 50)
        print("Alexandra Router - Classification Mode")
        print("=" * 50)
        print("\nThis mode tests question routing without loading the model.")
        print("Type questions to see how they would be classified.")
        print("Type 'quit' to exit.\n")

        while True:
            try:
                question = input("Test: ").strip()
                if question.lower() in ['quit', 'exit', 'q']:
                    break
                if not question:
                    continue

                result = self.classifier.classify_with_details(question)

                print(f"\n  Domain: {result['classified_domain']}")
                print(f"  Confidence: {result['confidence']:.0%}")

                # Show breakdown
                if result['confidence'] > 0:
                    print("\n  Score breakdown:")
                    for domain, score in sorted(result['scores'].items(),
                                               key=lambda x: x[1], reverse=True)[:3]:
                        if score > 0:
                            print(f"    {domain}: {score:.2f}")

                    # Show matched keywords
                    matched_kw = result['matched_keywords'].get(result['classified_domain'], [])
                    if matched_kw:
                        print(f"\n  Matched keywords: {', '.join(matched_kw[:5])}")

                print()

            except KeyboardInterrupt:
                break

        print("\nGoodbye!")


def main():
    parser = argparse.ArgumentParser(
        description="Alexandra AI - Intelligent Multi-Domain Chat",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python alexandra_chat.py                    # Interactive chat
  python alexandra_chat.py --classify         # Classification mode (no model)
  python alexandra_chat.py --test             # Test classifier
  python alexandra_chat.py --status           # Show status
  python alexandra_chat.py --no-routing       # Hide routing info
  python alexandra_chat.py --ask "question"   # Single question mode
        """
    )

    parser.add_argument('--classify', action='store_true',
                       help='Run in classification-only mode (no model needed)')
    parser.add_argument('--test', action='store_true',
                       help='Run classifier tests')
    parser.add_argument('--status', action='store_true',
                       help='Show router status and exit')
    parser.add_argument('--no-routing', action='store_true',
                       help='Hide routing information during chat')
    parser.add_argument('--ask', type=str,
                       help='Ask a single question and exit')
    parser.add_argument('--model', type=str,
                       default="/models/Qwen2.5-72B-Instruct",
                       help='Path to base model')
    parser.add_argument('--lora-dir', type=str,
                       default="/home/alexandratitus767/ai-clone-training/my-output",
                       help='Directory containing LoRA adapters')

    args = parser.parse_args()

    # Classification mode
    if args.classify:
        offline = OfflineChat()
        offline.classify_interactive()
        return

    # Test mode
    if args.test:
        test_classifier()
        return

    # Status mode
    if args.status:
        router = AlexandraRouter(auto_load_model=False)
        status = router.get_status()
        print("\n" + "=" * 40)
        print("Alexandra Router Status")
        print("=" * 40)
        print(f"Model path: {args.model}")
        print(f"LoRA directory: {args.lora_dir}")
        print(f"\nAvailable LoRA Adapters:")
        for domain, available in status['available_loras'].items():
            icon = "[READY]" if available else "[NOT TRAINED]"
            print(f"  {icon} {domain}")
        return

    # Single question mode
    if args.ask:
        print("\nLoading Alexandra...")
        chat = AlexandraChat(
            base_model_path=args.model,
            lora_base_dir=args.lora_dir,
            auto_load=True,
            show_routing=not args.no_routing
        )
        response = chat.chat(args.ask)
        print(f"\nAlexandra: {response}")
        return

    # Interactive mode
    print("\nInitializing Alexandra AI...")
    print("(This may take a few minutes to load the model)\n")

    try:
        chat = AlexandraChat(
            base_model_path=args.model,
            lora_base_dir=args.lora_dir,
            auto_load=True,
            show_routing=not args.no_routing
        )
        chat.run_interactive()

    except Exception as e:
        print(f"\nError initializing chat: {e}")
        print("\nTry running in classification mode instead:")
        print("  python alexandra_chat.py --classify")


if __name__ == "__main__":
    main()
