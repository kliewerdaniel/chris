#!/usr/bin/env python3
"""TTS module wrapping Qwen3-TTS pipeline."""

import os
import uuid
import numpy as np
import soundfile as sf
from pathlib import Path

OUTPUT_DIR = "audio_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

QWEN_MODEL_ID = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-4bit"
REFERENCE_VOICE = "chris.wav"

_model = None
_ref_text = None

def _get_model():
    """Lazy load TTS model."""
    global _model
    if _model is None:
        from mlx_audio.tts.utils import load_model
        _model = load_model(QWEN_MODEL_ID)
    return _model


def _get_reference_text():
    """Lazy transcribe reference voice once."""
    global _ref_text
    if _ref_text is None and os.path.exists(REFERENCE_VOICE):
        try:
            from mlx_audio.stt import load as load_stt
            stt_model = load_stt("mlx-community/whisper-large-v3-turbo-asr-fp16")
            _ref_text = stt_model.generate(REFERENCE_VOICE).text
            print(f"✅ Reference voice transcribed: {_ref_text[:60]}...")
        except Exception as e:
            print(f"⚠️  Could not transcribe reference voice: {e}")
            _ref_text = ""
    return _ref_text


def generate_speech(text: str, speed: float = 1.0) -> str:
    """
    Generate speech audio from text using Qwen3-TTS with Chris voice cloning.
    
    Returns:
        Path to generated wav file
    """
    try:
        model = _get_model()
        ref_audio = REFERENCE_VOICE if os.path.exists(REFERENCE_VOICE) else None
        ref_text = _get_reference_text() if ref_audio else None
        
        all_audio = []
        sample_rate = 24000
        
        for result in model.generate(
            text=text,
            ref_audio=ref_audio,
            ref_text=ref_text,
            speed=speed,
            verbose=False,
        ):
            all_audio.append(np.array(result.audio))
            sample_rate = getattr(result, 'sample_rate', sample_rate)
        
        if not all_audio:
            raise Exception("No audio generated")
        
        audio = np.concatenate(all_audio) if len(all_audio) > 1 else all_audio[0]
        
        # Save to unique file
        file_id = str(uuid.uuid4())
        output_path = os.path.join(OUTPUT_DIR, f"{file_id}.wav")
        sf.write(output_path, audio, sample_rate)
        
        return output_path
    
    except Exception as e:
        print(f"TTS generation failed: {e}")
        raise