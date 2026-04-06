#!/usr/bin/env python3
"""FastAPI backend for AI companion system with persistent memory."""

import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager

from memory import load_core_memory
from database import init_db, save_message, get_recent_messages, clear_conversation_history, clear_all_memories, get_all_memories, save_memory
from memory_manager import format_memories_for_context, process_conversation_for_memory
from llm import build_tiered_prompt, call_llama_cpp, summarize_conversation
from tts import generate_speech, OUTPUT_DIR


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    text: str
    audio_url: str


# Global state
core_memory = ""
cached_summary = None
summary_turn_counter = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load resources at startup."""
    global core_memory
    
    # Initialize database
    init_db()
    print("✅ Database initialized")
    
    # Load core memory
    core_memory = load_core_memory()
    
    # Seed core memory into database if empty
    existing_memories = get_all_memories()
    if not existing_memories and core_memory:
        save_memory("CORE_IDENTITY", core_memory)
    
    print(f"✅ Core memory loaded: {len(core_memory)} characters")
    print(f"✅ Loaded {len(existing_memories)} stored memories")
    
    # Load last 20 messages on startup
    recent = get_recent_messages(20)
    print(f"✅ Loaded {len(recent)} previous conversation messages")
    
    yield


app = FastAPI(title="Chris AI Companion", lifespan=lifespan)

# Mount audio files
app.mount("/audio", StaticFiles(directory=OUTPUT_DIR), name="audio")


@app.get("/")
async def index():
    """Serve frontend interface."""
    return FileResponse("index.html")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Handle chat request with text + audio response."""
    global cached_summary, summary_turn_counter
    
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    # 1. Save user message
    save_message("user", request.message)
    
    # 2. Get context windows
    all_messages = get_recent_messages(30)
    recent_messages = all_messages[-10:] if len(all_messages) > 10 else all_messages
    older_messages = all_messages[:-10] if len(all_messages) > 10 else []
    
    # 3. Refresh summary every 5 turns or when needed
    if older_messages and (summary_turn_counter % 5 == 0 or not cached_summary):
        cached_summary = summarize_conversation(older_messages)
    
    summary_turn_counter += 1
    
    # 4. Build tiered prompt
    memories_facts = format_memories_for_context()
    prompt = build_tiered_prompt(
        core_memory=core_memory,
        memories_facts=memories_facts,
        recent_messages=recent_messages,
        older_summary=cached_summary,
        user_message=request.message
    )
    
    # 5. Call LLM
    llm_response = call_llama_cpp(prompt)
    if not llm_response:
        raise HTTPException(status_code=500, detail="Failed to get LLM response")
    
    # 6. Save assistant response
    save_message("assistant", llm_response)
    
    # 7. Extract memories from this conversation turn
    process_conversation_for_memory(request.message, llm_response)
    
    # 8. Generate TTS
    try:
        audio_path = generate_speech(llm_response)
        audio_filename = os.path.basename(audio_path)
        audio_url = f"/audio/{audio_filename}"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS failed: {str(e)}")
    
    return ChatResponse(
        text=llm_response,
        audio_url=audio_url
    )


@app.delete("/reset")
async def reset_conversation():
    """Reset conversation history (preserves stored memories)."""
    global cached_summary, summary_turn_counter
    clear_conversation_history()
    cached_summary = None
    summary_turn_counter = 0
    return {"success": True, "message": "Conversation history cleared. Memories preserved."}


@app.delete("/reset_memories")
async def reset_memories():
    """Clear ALL stored memories (irreversible)."""
    clear_all_memories()
    return {"success": True, "message": "All stored memories have been deleted."}


@app.get("/memories")
async def list_memories():
    """List all currently stored memories."""
    return get_all_memories()


if __name__ == "__main__":
    import uvicorn
    print("Starting Chris server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
