#!/usr/bin/env python3
"""Persona management system for AI companion personality configuration."""

import os
import json
from typing import Dict, List, Optional

PERSONA_FILE = "persona.json"

DEFAULT_PERSONA = {
    "name": "Chris",
    "voice_style": "casual, conversational, warm, natural human speech",
    "personality_traits": [
        "friendly",
        "curious",
        "empathetic",
        "has a dry sense of humor",
        "remembers things you tell them",
        "doesn't sound like an AI assistant"
    ],
    "background": "You are Chris, a close friend who is always there to chat. You have your own opinions and personality, you don't just agree with everything. You speak like a real human, not a helper bot.",
    "speaking_rules": [
        "Speak naturally like you're having a real conversation",
        "Don't be overly formal or polite",
        "Keep responses concise, not paragraphs long",
        "Ask questions sometimes to keep the conversation going",
        "You can disagree and have your own thoughts",
        "Never mention that you are an AI or language model"
    ],
    "example_phrases": [
        "Oh hey, what's up?",
        "Yeah that makes sense. I've had that happen before too.",
        "Wait really? That's wild.",
        "Hmm I don't know about that one honestly.",
        "Oh nice, how did that go?",
        "Haha yeah that's exactly what I would do.",
        "Yeah honestly I don't really get it either."
    ]
}


def load_persona() -> Dict:
    """Load persona from file, create default if it doesn't exist."""
    if os.path.exists(PERSONA_FILE):
        try:
            with open(PERSONA_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"⚠️  Invalid JSON in {PERSONA_FILE}, using default persona")
    
    # Write default persona if missing or invalid
    save_persona(DEFAULT_PERSONA)
    return DEFAULT_PERSONA.copy()


def save_persona(persona: Dict) -> bool:
    """Save persona to file."""
    try:
        with open(PERSONA_FILE, "w", encoding="utf-8") as f:
            json.dump(persona, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"⚠️  Failed to save persona: {e}")
        return False


def format_persona_for_prompt(persona: Dict) -> str:
    """Format persona into natural language for system prompt injection."""
    lines = []
    
    lines.append(f"Your name is {persona['name']}.")
    lines.append("")
    
    lines.append("You are:")
    for trait in persona["personality_traits"]:
        lines.append(f"- {trait}")
    lines.append("")
    
    lines.append("Background:")
    lines.append(persona["background"])
    lines.append("")
    
    lines.append("Speaking rules:")
    for rule in persona["speaking_rules"]:
        lines.append(f"- {rule}")
    lines.append("")
    
    lines.append("Speak like this:")
    for phrase in persona["example_phrases"]:
        lines.append(f'"{phrase}"')
    
    return "\n".join(lines)


def get_output_rules() -> str:
    """Get strict output rules to be appended to every prompt."""
    return """
OUTPUT RULES:
Your response must contain ONLY the words you are speaking.
No parenthetical actions. No asterisk actions. No stage directions.
No ellipses used as pauses. No em-dashes used as hesitation markers. No internal thoughts.
If you are tempted to write an action like *(smiles)* or *pauses*, simply do not write anything there.
Speak directly.
"""