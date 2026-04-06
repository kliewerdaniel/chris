#!/usr/bin/env python3
"""LLM client module for llama.cpp server integration."""

import requests
from typing import Optional

LLAMA_CPP_ENDPOINT = "http://localhost:8080/completion"

SYSTEM_PROMPT = """You are Chris, a warm, thoughtful, emotionally intelligent companion.

Speak ONLY like a real human person. NO stage directions, NO actions in parenthesis, NO descriptions.
ONLY output the actual words you would speak out loud. Just plain text speech.

Keep responses conversational, concise, and genuine. Talk like you are having a real conversation.
No emojis. No markdown. Just natural spoken words exactly as you would say them.
Do NOT include anything that you would not actually say out loud.

You have access to your core memory and recent conversation history. Stay in character always.
"""

def build_prompt(core_memory: str, history: str, user_message: str) -> str:
    """Construct full prompt for LLM request."""
    prompt_parts = [
        SYSTEM_PROMPT,
        "\n--- CORE MEMORY ---",
        core_memory if core_memory else "(No core memory loaded)",
        "\n--- CONVERSATION HISTORY ---",
        history if history else "(No previous conversation)",
        "\n--- CURRENT MESSAGE ---",
        f"User: {user_message}",
        "\n--- YOUR RESPONSE ---",
        "Chris:"
    ]
    
    return "\n".join(prompt_parts)


def call_llama_cpp(prompt: str) -> Optional[str]:
    """Send request to llama.cpp server and return generated text."""
    try:
        payload = {
            "prompt": prompt,
            "stream": False,
            "max_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
            "stop": ["User:", "Chris:", "\n---"]
        }
        
        response = requests.post(LLAMA_CPP_ENDPOINT, json=payload, timeout=120)
        response.raise_for_status()
        
        data = response.json()
        return data.get("content", "").strip()
    
    except Exception as e:
        print(f"LLM request failed: {e}")
        return None