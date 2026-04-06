#!/usr/bin/env python3
"""Memory module for loading and managing core system memory."""

import os
from pathlib import Path

CORE_MEMORY_FILE = "core_memory.txt"

def load_core_memory() -> str:
    """Load core memory from file at startup."""
    if os.path.exists(CORE_MEMORY_FILE):
        with open(CORE_MEMORY_FILE, "r", encoding="utf-8") as f:
            return f.read().strip()
    return ""


class ConversationHistory:
    """In-memory conversation history manager."""
    
    MAX_MESSAGES = 20
    
    def __init__(self):
        self.messages = []
    
    def add_user_message(self, text: str):
        """Add a user message to history."""
        self.messages.append({"role": "user", "content": text})
        self._trim()
    
    def add_assistant_message(self, text: str):
        """Add an assistant message to history."""
        self.messages.append({"role": "assistant", "content": text})
        self._trim()
    
    def _trim(self):
        """Keep only last MAX_MESSAGES messages."""
        if len(self.messages) > self.MAX_MESSAGES:
            self.messages = self.messages[-self.MAX_MESSAGES:]
    
    def get_formatted_history(self) -> str:
        """Get history formatted for prompt injection."""
        lines = []
        for msg in self.messages:
            role = msg["role"].capitalize()
            lines.append(f"{role}: {msg['content']}")
        return "\n".join(lines)
    
    def clear(self):
        """Clear conversation history."""
        self.messages.clear()