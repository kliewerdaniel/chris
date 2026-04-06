#!/usr/bin/env python3
"""FastAPI backend for AI companion system."""

import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager

from memory import load_core_memory, ConversationHistory
from llm import build_prompt, call_llama_cpp
from tts import generate_speech, OUTPUT_DIR


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    text: str
    audio_url: str


# Global state
core_memory = ""
conversation_history = ConversationHistory()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load resources at startup."""
    global core_memory
    core_memory = load_core_memory()
    print(f"✅ Core memory loaded: {len(core_memory)} characters")
    print(f"✅ Conversation history initialized (max {conversation_history.MAX_MESSAGES} messages)")
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
    
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    # 1. Add user message to history
    conversation_history.add_user_message(request.message)
    
    # 2. Build prompt
    prompt = build_prompt(
        core_memory=core_memory,
        history=conversation_history.get_formatted_history(),
        user_message=request.message
    )
    
    # 3. Call LLM
    llm_response = call_llama_cpp(prompt)
    if not llm_response:
        raise HTTPException(status_code=500, detail="Failed to get LLM response")
    
    # 4. Add assistant response to history
    conversation_history.add_assistant_message(llm_response)
    
    # 5. Generate TTS
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
    """Reset conversation history."""
    conversation_history.clear()
    return {"success": True, "message": "Conversation history cleared"}


if __name__ == "__main__":
    import uvicorn
    print("Starting Chris server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)