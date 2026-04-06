#!/usr/bin/env python3
"""Utility functions for speech validation and helpers."""

import re
from typing import Tuple

# Stage direction indicators
STAGE_WORDS = {
    "thinks", "pauses", "smiles", "nods", "laughs", "sighs",
    "whispers", "looks", "glances", "smirks", "grins", "frowns",
    "shrugs", "blinks", "chuckles", "groans", "hesitates", "pauses"
}


def validate_speech_only(text: str) -> Tuple[bool, str]:
    """
    Validate that response contains only spoken words, no actions or stage directions.
    Returns (is_valid, reason) tuple.
    """
    if not text:
        return False, "Empty response"
    
    # Check for forbidden markers
    if '*' in text:
        return False, "Contains asterisk actions"
    if '(' in text or ')' in text:
        return False, "Contains parenthetical actions"
    if '[' in text or ']' in text:
        return False, "Contains bracket actions"
    
    lines = text.splitlines()
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(('—', '-', '>')):
            return False, "Contains line starting with action marker"
        
        # Check for standalone stage direction words
        words = re.split(r'\W+', stripped.lower())
        line_words = set(w for w in words if w)
        if len(line_words) <= 3 and any(word in STAGE_WORDS for word in line_words):
            return False, f"Contains standalone stage direction: {line_words & STAGE_WORDS}"
    
    return True, "OK"
