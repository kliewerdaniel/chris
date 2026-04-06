#!/usr/bin/env python3
"""Utility functions for sanitization and helpers."""

import re
from typing import List

# Stage direction indicators
STAGE_WORDS = {
    "thinks", "pauses", "smiles", "nods", "laughs", "sighs",
    "whispers", "looks", "glances", "smirks", "grins", "frowns",
    "shrugs", "blinks", "chuckles", "groans", "hesitates", "pauses"
}


def sanitize_response(text: str) -> str:
    """
    Sanitize LLM response by removing actions, stage directions, and formatting artifacts.
    Returns clean text suitable for TTS and conversation history.
    """
    if not text:
        return ""
    
    cleaned = text
    
    # Remove anything in parentheses, asterisks, brackets
    cleaned = re.sub(r'\([^)]*\)', '', cleaned)
    cleaned = re.sub(r'\*[^*]*\*', '', cleaned)
    cleaned = re.sub(r'\[[^\]]*\]', '', cleaned)
    
    lines = cleaned.splitlines()
    filtered_lines: List[str] = []
    
    for line in lines:
        stripped = line.strip()
        
        # Skip empty lines for now
        if not stripped:
            filtered_lines.append("")
            continue
        
        # Skip lines starting with action markers
        if stripped.startswith(('*', '[', '(', '—', '-', '>')):
            continue
        
        # Check if this line is just a stage direction
        words = re.split(r'\W+', stripped.lower())
        line_words = set(w for w in words if w)
        
        # If the line only contains stage direction words and nothing else
        if len(line_words) <= 3 and any(word in STAGE_WORDS for word in line_words):
            continue
        
        # Keep this line
        filtered_lines.append(line)
    
    # Collapse multiple blank lines
    collapsed: List[str] = []
    last_blank = False
    
    for line in filtered_lines:
        if not line.strip():
            if not last_blank:
                collapsed.append("")
                last_blank = True
        else:
            collapsed.append(line)
            last_blank = False
    
    # Join and trim
    result = '\n'.join(collapsed).strip()
    
    return result