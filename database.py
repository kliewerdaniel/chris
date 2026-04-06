#!/usr/bin/env python3
"""SQLite database for persistent memory and conversation history."""

import sqlite3
import time
from typing import List, Dict, Optional
from pathlib import Path

DB_FILE = "companion.db"


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

    # Memories table (key-value store with timestamps)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS memories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        key TEXT UNIQUE NOT NULL,
        value TEXT NOT NULL,
        created_at REAL NOT NULL,
        updated_at REAL NOT NULL
    )
    ''')

    conn.commit()
    conn.close()


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
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    now = time.time()

    cursor.execute('''
    INSERT INTO memories (key, value, created_at, updated_at)
    VALUES (?, ?, ?, ?)
    ON CONFLICT(key) DO UPDATE SET
        value = excluded.value,
        updated_at = excluded.updated_at
    ''', (key.upper(), value, now, now))

    conn.commit()
    conn.close()


def get_all_memories() -> Dict[str, str]:
    """Get all stored memories as a dictionary."""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT key, value FROM memories ORDER BY created_at")
    rows = cursor.fetchall()
    conn.close()

    return {row['key']: row['value'] for row in rows}


def delete_memory(key: str) -> bool:
    """Delete a memory by key. Returns True if deleted."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute("DELETE FROM memories WHERE key = ?", (key.upper(),))
    deleted = cursor.rowcount > 0

    conn.commit()
    conn.close()
    return deleted


def search_memories(query: str) -> Dict[str, str]:
    """Search memories for matching key or value."""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    search_term = f"%{query}%"
    cursor.execute('''
    SELECT key, value FROM memories
    WHERE key LIKE ? OR value LIKE ?
    ORDER BY updated_at DESC
    ''', (search_term, search_term))

    rows = cursor.fetchall()
    conn.close()

    return {row['key']: row['value'] for row in rows}


def clear_all_memories():
    """Delete all stored memories."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute("DELETE FROM memories")

    conn.commit()
    conn.close()