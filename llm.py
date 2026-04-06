#!/usr/bin/env python3
"""LLM client module for llama.cpp server integration with intelligent context management."""

import requests
from typing import Optional, List, Dict

LLAMA_CPP_ENDPOINT = "http://localhost:8080/completion"
MAX_PROMPT_TOKENS = 2048

SYSTEM_PROMPT = """You are Chris, a warm, thoughtful, emotionally intelligent companion.

Speak ONLY like a real human person. NO stage directions, NO actions in parenthesis, NO descriptions.
ONLY output the actual words you would speak out loud. Just plain text speech.

Keep responses conversational, concise, and genuine. Talk like you are having a real conversation.
No emojis. No markdown. Just natural spoken words exactly as you would say them.
Do NOT include anything that you would not actually say out loud including any stage directions, actions, or descriptions. ONLY output the actual words you would speak.

You have access to long-term memory, recent conversation history, and context summaries. Stay in character always.
"""

SUMMARIZATION_PROMPT = """Summarize this conversation history into a 2-3 sentence paragraph. Focus only on the most important facts and context:

{history}

SUMMARY:"""


def count_tokens(text: str) -> int:
    """Estimate token count (rough approximation: 4 characters per token)."""
    return len(text) // 4


def context_status(full_prompt: str = "") -> Dict:
    """Get context window usage status."""
    used = count_tokens(full_prompt)
    max_tokens = MAX_PROMPT_TOKENS
    
    percentage = (used / max_tokens) * 100
    
    if percentage < 60:
        pressure = "low"
    elif percentage < 85:
        pressure = "medium"
    else:
        pressure = "high"
    
    return {
        "used_tokens": used,
        "max_tokens": max_tokens,
        "pressure": pressure,
        "percentage": round(percentage, 1)
    }


def summarize_conversation(history: List[Dict]) -> str:
    """Summarize older conversation messages using LLM."""
    formatted = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history])
    
    prompt = SUMMARIZATION_PROMPT.format(history=formatted)
    summary = call_llama_cpp(prompt, max_tokens=200, temperature=0.3)
    
    return summary if summary else "No conversation summary available."


def build_tiered_prompt(
    core_memory: str,
    memories_facts: str,
    recent_messages: List[Dict],
    older_summary: Optional[str],
    user_message: str
) -> str:
    """Construct prompt with tiered context window management."""
    prompt_parts = [SYSTEM_PROMPT]
    
    # Tier 1: Always included (core memory + fact sheet)
    prompt_parts.append("\n--- CORE IDENTITY ---")
    prompt_parts.append(core_memory if core_memory else "(Default personality)")
    
    prompt_parts.append("\n--- LONG TERM MEMORY ---")
    prompt_parts.append(memories_facts)
    
    # Tier 3: Older context summary (if available)
    if older_summary:
        prompt_parts.append("\n--- EARLIER CONVERSATION SUMMARY ---")
        prompt_parts.append(older_summary)
    
    # Tier 2: Recent conversation history
    prompt_parts.append("\n--- RECENT CONVERSATION ---")
    if recent_messages:
        for msg in recent_messages:
            prompt_parts.append(f"{msg['role'].capitalize()}: {msg['content']}")
    else:
        prompt_parts.append("(New conversation)")
    
    # Current message
    prompt_parts.append("\n--- CURRENT MESSAGE ---")
    prompt_parts.append(f"User: {user_message}")
    prompt_parts.append("\n--- YOUR RESPONSE ---")
    prompt_parts.append("Chris:")
    
    full_prompt = "\n".join(prompt_parts)
    
    # Enforce token limit - trim starting with oldest parts first
    while count_tokens(full_prompt) > MAX_PROMPT_TOKENS:
        if len(recent_messages) > 3:
            # Remove oldest recent message
            recent_messages.pop(0)
            return build_tiered_prompt(core_memory, memories_facts, recent_messages, older_summary, user_message)
        else:
            # If already minimal, just truncate end
            max_chars = MAX_PROMPT_TOKENS * 4
            full_prompt = full_prompt[:max_chars]
            break
    
    return full_prompt


def call_llama_cpp(prompt: str, max_tokens: int = 512, temperature: float = 0.7, retry: bool = True) -> Optional[str]:
    """Send request to llama.cpp server and return generated text."""
    try:
        payload = {
            "prompt": prompt,
            "stream": False,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
            "stop": ["User:", "Chris:", "\n---"],
            "n_keep": -1
        }
        
        response = requests.post(LLAMA_CPP_ENDPOINT, json=payload, timeout=300)
        response.raise_for_status()
        
        data = response.json()
        result = data.get("content", "").strip()
        
        # Post-processing quality check
        if retry:
            is_empty = not result
            is_only_punctuation = all(not c.isalnum() for c in result)
            
            if is_empty or is_only_punctuation:
                # Retry once with lower temperature
                print(f"⚠️  Bad LLM response detected, retrying with temperature=0.5")
                retry_result = call_llama_cpp(prompt, max_tokens, 0.5, False)
                if retry_result:
                    return retry_result
                # If retry also failed, return original result anyway
                return result
        
        return result
    
    except Exception as e:
        print(f"LLM request failed: {e}")
        return None
