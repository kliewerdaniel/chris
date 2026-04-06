#!/usr/bin/env python3
"""SQLite database for conversation history, JSON file for memory storage."""

import sqlite3
import time
import json
import os
from typing import List, Dict, Optional
from pathlib import Path

DB_FILE = "companion.db"
MEMORIES_FILE = "memories.json"
MEMORIES_TMP_FILE = "memories.json.tmp"


def init_db():
    """Initialize database tables if they don't exist."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # Conversations table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS conversations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        timestamp REAL NOT NULL,
        session_id TEXT DEFAULT 'default'
    )
    ''')

    conn.commit()
    conn.close()

    # Initialize memories file if doesn't exist
    if not Path(MEMORIES_FILE).exists():
        with open(MEMORIES_FILE, 'w', encoding='utf-8') as f:
            json.dump({}, f, indent=2)


def _load_memories() -> Dict[str, str]:
    """Load all memories from JSON file."""
    try:
        with open(MEMORIES_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return {}


def _save_memories(memories: Dict[str, str]):
    """Save memories atomically to JSON file to prevent corruption."""
    # Write to temporary file first
    with open(MEMORIES_TMP_FILE, 'w', encoding='utf-8') as f:
        json.dump(memories, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    
    # Atomic rename
    os.replace(MEMORIES_TMP_FILE, MEMORIES_FILE)


def save_message(role: str, content: str, session_id: str = 'default'):
    """Save a single conversation message to database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    timestamp = time.time()
    cursor.execute(
        "INSERT INTO conversations (role, content, timestamp, session_id) VALUES (?, ?, ?, ?)",
        (role, content, timestamp, session_id)
    )

    conn.commit()
    conn.close()


def get_recent_messages(n: int = 20, session_id: str = 'default') -> List[Dict]:
    """Get last N messages from conversation history."""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute('''
    SELECT role, content, timestamp FROM conversations
    WHERE session_id = ?
    ORDER BY timestamp DESC
    LIMIT ?
    ''', (session_id, n))

    rows = cursor.fetchall()
    conn.close()

    # Return in chronological order (oldest first)
    messages = [dict(row) for row in reversed(rows)]
    return messages


def clear_conversation_history(session_id: str = 'default'):
    """Clear all conversation messages for given session."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute("DELETE FROM conversations WHERE session_id = ?", (session_id,))

    conn.commit()
    conn.close()


def save_memory(key: str, value: str):
    """Save or update a memory entry (upsert)."""
    memories = _load_memories()
    memories[key.upper()] = value
    _save_memories(memories)


def get_all_memories() -> Dict[str, str]:
    """Get all stored memories as a dictionary."""
    return _load_memories()


def delete_memory(key: str) -> bool:
    """Delete a memory by key. Returns True if deleted."""
    memories = _load_memories()
    key_upper = key.upper()
    if key_upper in memories:
        del memories[key_upper]
        _save_memories(memories)
        return True
    return False


def search_memories(query: str) -> Dict[str, str]:
    """Search memories for matching key or value."""
    memories = _load_memories()
    query_lower = query.lower()
    results = {}
    
    for key, value in memories.items():
        if query_lower in key.lower() or query_lower in value.lower():
            results[key] = value
    
    return results


def clear_all_memories():
    """Delete all stored memories."""
    _save_memories({})
