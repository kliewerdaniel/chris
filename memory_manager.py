#!/usr/bin/env python3
"""Intelligent memory extraction and management system."""

import re
from typing import Dict, List
from llm import call_llama_cpp
from database import save_memory, get_all_memories

MEMORY_EXTRACTION_PROMPT = """You are a memory extractor. Analyze the last conversation turn and extract any new facts, preferences, or information about the user that should be remembered long term.

Extract only NEW information that was not already known. Do not repeat existing facts.

Return ONLY key-value pairs in this exact format, one per line:
KEY: VALUE

Valid categories include:
- NAME
- LIKES
- DISLIKES
- LIVES_IN
- WORKS_AT
- FAMILY
- FRIENDS
- HOBBIES
- GOALS
- FEARS
- IMPORTANT_EVENTS
- ANY_OTHER_CATEGORY

If there are no new facts to remember, return ONLY the word NONE.

LAST CONVERSATION:
{conversation}

EXISTING MEMORIES:
{existing_memories}

EXTRACTED FACTS:"""


def extract_memories(conversation_turn: str) -> Dict[str, str]:
    """Extract memorable facts from a conversation turn using LLM."""
    existing = get_all_memories()
    existing_formatted = "\n".join([f"{k}: {v}" for k, v in existing.items()])

    prompt = MEMORY_EXTRACTION_PROMPT.format(
        conversation=conversation_turn,
        existing_memories=existing_formatted if existing_formatted else "None"
    )

    response = call_llama_cpp(prompt)
    if not response:
        return {}

    if response.strip().upper() == "NONE":
        return {}

    memories = {}
    lines = response.strip().split("\n")

    for line in lines:
        line = line.strip()
        if not line or ":" not in line:
            continue

        # Split on first colon
        key, value = line.split(":", 1)
        key = key.strip().upper()
        value = value.strip()

        if key and value:
            memories[key] = value

    return memories


def process_conversation_for_memory(user_message: str, assistant_response: str):
    """Process a complete conversation turn and save extracted memories."""
    conversation = f"User: {user_message}\nAssistant: {assistant_response}"

    new_memories = extract_memories(conversation)

    for key, value in new_memories.items():
        save_memory(key, value)

    return new_memories


def format_memories_for_context(max_tokens: int = 200) -> str:
    """Format all memories as a concise fact sheet for context window."""
    memories = get_all_memories()

    if not memories:
        return "No stored memories."

    lines = []
    for key, value in memories.items():
        lines.append(f"- {key}: {value}")

    fact_sheet = "\n".join(lines)

    # Rough token trimming (4 chars per token estimate)
    while len(fact_sheet) > (max_tokens * 4) and len(lines) > 5:
        lines.pop(0)
        fact_sheet = "\n".join(lines)

    return fact_sheet