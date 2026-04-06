#!/usr/bin/env python3
"""FastAPI backend for AI companion system with persistent memory."""

import os
import json
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager

import requests
from persona import load_persona, save_persona, format_persona_for_prompt, get_output_rules, PERSONA_FILE
from database import init_db, save_message, get_recent_messages, clear_conversation_history, clear_all_memories, get_all_memories, save_memory
from memory_manager import format_memories_for_context, process_conversation_for_memory
from llm import build_tiered_prompt, call_llama_cpp, summarize_conversation, LLAMA_CPP_ENDPOINT, context_status, count_tokens
from tts import generate_speech, OUTPUT_DIR, REFERENCE_VOICE
from utils import validate_speech_only


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    text: str
    audio_url: str


# Global state
persona = {}
cached_summary = None
summary_turn_counter = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load resources at startup."""
    global persona
    
    # Initialize database
    init_db()
    print("✅ Database initialized")
    
    # Load persona
    persona = load_persona()
    print(f"\n📝 Persona file: ./{PERSONA_FILE} — edit this file to change Chris's personality")
    print(f"✅ Persona loaded: {persona['name']} with {len(persona['personality_traits'])} traits")
    
    # Seed core identity into database if empty
    existing_memories = get_all_memories()
    if not existing_memories:
        save_memory("CORE_IDENTITY", format_persona_for_prompt(persona))
    
    print(f"🧠 Loaded {len(existing_memories)} stored memories")
    
    # Load last 20 messages on startup
    recent = get_recent_messages(20)
    print(f"💬 Loaded {len(recent)} previous conversation messages")
    
    # Check voice file
    if os.path.exists(REFERENCE_VOICE):
        print(f"✅ Reference voice file '{REFERENCE_VOICE}' found")
    else:
        print(f"⚠️  Reference voice file '{REFERENCE_VOICE}' not found - using default voice")
    
    # Check llama.cpp server connectivity
    llama_ok = False
    try:
        response = requests.post(LLAMA_CPP_ENDPOINT, json={"prompt": "", "max_tokens": 1}, timeout=3)
        llama_ok = response.status_code < 500
        print("✅ llama.cpp server is reachable")
    except Exception:
        print("⚠️  llama.cpp server is not reachable")
    
    print("\n📊 Startup summary:")
    print(f"   Memories: {len(existing_memories)}")
    print(f"   Messages: {len(recent)}")
    print(f"   Voice file: {'✓' if os.path.exists(REFERENCE_VOICE) else '✗'}")
    print(f"   LLM server: {'✓' if llama_ok else '✗'}")
    print()
    
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
    
    # Check context pressure and auto-summarize if needed
    test_prompt = build_tiered_prompt(
        core_memory=format_persona_for_prompt(persona),
        memories_facts=format_memories_for_context(),
        recent_messages=recent_messages,
        older_summary=cached_summary,
        user_message=request.message
    )
    
    status = context_status(test_prompt)
    if status["pressure"] == "high" and len(older_messages) > 4:
        # Summarize oldest half of conversation
        split_point = len(older_messages) // 2
        cached_summary = summarize_conversation(older_messages[:split_point])
        recent_messages = older_messages[split_point:] + recent_messages
        print(f"🔄 Context pressure high, auto-summarized oldest {split_point} messages")
    
    # 4. Build final tiered prompt
    memories_facts = format_memories_for_context()
    prompt = build_tiered_prompt(
        core_memory=format_persona_for_prompt(persona),
        memories_facts=memories_facts,
        recent_messages=recent_messages,
        older_summary=cached_summary,
        user_message=request.message
    )
    
    # Append output rules to every prompt right before response
    prompt += get_output_rules()
    prompt += "\nChris:"
    
    # 5. Call LLM
    llm_response = call_llama_cpp(prompt)
    if not llm_response:
        raise HTTPException(status_code=500, detail="Failed to get LLM response")
    
    # Validate response is speech only
    is_valid, reason = validate_speech_only(llm_response)
    clean_response = llm_response
    
    if not is_valid:
        print(f"⚠️  Invalid response detected ({reason}), retrying once")
        retry_prompt = prompt.replace("\nChris:", f"\nIMPORTANT: Your previous response contained non-speech content. Reply with spoken words ONLY:\nChris:")
        retry_response = call_llama_cpp(retry_prompt, retry=False)
        if retry_response:
            clean_response = retry_response

    # 6. Save assistant response
    save_message("assistant", clean_response)

    # 7. Extract memories from this conversation turn
    process_conversation_for_memory(request.message, clean_response)

    # 8. Generate TTS
    try:
        audio_path = generate_speech(clean_response)
        audio_filename = os.path.basename(audio_path)
        audio_url = f"/audio/{audio_filename}"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS failed: {str(e)}")
    
    return ChatResponse(
        text=clean_response,
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


@app.get("/persona")
async def get_persona():
    """Get current persona configuration."""
    return persona


@app.put("/persona")
async def update_persona(request: Request):
    """Update persona configuration."""
    global persona
    try:
        new_persona = await request.json()
        if save_persona(new_persona):
            persona = new_persona
            return {"success": True, "message": "Persona updated successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to save persona")
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON")


@app.get("/context_status")
async def get_context_status():
    """Get context window usage status."""
    all_messages = get_recent_messages(30)
    test_prompt = build_tiered_prompt(
        core_memory=format_persona_for_prompt(persona),
        memories_facts=format_memories_for_context(),
        recent_messages=all_messages[-10:],
        older_summary=cached_summary,
        user_message=""
    )
    return context_status(test_prompt)


@app.get("/export")
async def export_conversation():
    """Export full conversation history as plain text."""
    messages = get_recent_messages(10000)
    
    lines = []
    for msg in messages:
        timestamp = datetime.fromtimestamp(msg.get("timestamp", datetime.now().timestamp()))
        time_str = timestamp.strftime("%H:%M")
        role = msg["role"].capitalize()
        lines.append(f"[{time_str}] {role}: {msg['content']}")
    
    content = "\n".join(lines)
    return PlainTextResponse(content, headers={
        "Content-Disposition": "attachment; filename=conversation.txt"
    })


@app.get("/health")
async def health_check():
    """Health check endpoint with system status."""
    memory_count = len(get_all_memories())
    conversation_count = len(get_recent_messages(10000))
    
    # Check if llama.cpp server is reachable
    llama_ok = False
    try:
        # Simple ping request
        response = requests.post(LLAMA_CPP_ENDPOINT, json={"prompt": "", "max_tokens": 1}, timeout=2)
        llama_ok = response.status_code < 500
    except Exception:
        pass
    
    return {
        "status": "ok",
        "memories": memory_count,
        "conversation_messages": conversation_count,
        "llama_cpp_connected": llama_ok
    }


if __name__ == "__main__":
    import uvicorn
    print("Starting Chris server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
