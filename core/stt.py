"""
Speech-to-Text engines for Orthos.

Whisper  – offline transcription via faster-whisper (VAD-buffered)
Vosk     – offline streaming transcription (lighter)
Gemini   – cloud transcription via Google Gemini API (no local model)
"""
import io
import json

import numpy as np
import soundfile as sf


class GroqSTT:
    """Cloud STT via Groq Whisper API. No local model needed — ideal for low-resource systems."""

    def __init__(self, api_key: str, language: str = "auto"):
        from groq import Groq
        self._client = Groq(api_key=api_key)
        self._language = language if language and language.strip().lower() != "auto" else None

    def transcribe(self, audio: np.ndarray) -> str:
        buf = io.BytesIO()
        sf.write(buf, audio, 16000, format="WAV")
        buf.seek(0)

        kwargs = dict(
            model="whisper-large-v3-turbo",
            file=("audio.wav", buf),
        )
        if self._language:
            kwargs["language"] = self._language

        response = self._client.audio.transcriptions.create(**kwargs)
        return response.text.strip()


class GeminiSTT:
    """Cloud STT via Google Gemini API. No local model needed — ideal for low-resource systems."""

    def __init__(self, api_key: str, language: str = "auto"):
        if not api_key:
            raise RuntimeError(
                "Gemini STT requires 'gemini_voice_api_key' — "
                "set it in Configure → Text-to-Speech → Voice API Key."
            )
        from google import genai
        self._client = genai.Client(api_key=api_key)
        self._language = language

    def transcribe(self, audio: np.ndarray) -> str:
        from google.genai import types
        buf = io.BytesIO()
        sf.write(buf, audio, 16000, format="WAV")
        wav_bytes = buf.getvalue()

        response = self._client.models.generate_content(
            model="gemini-2.0-flash-001",
            contents=[
                types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav"),
                "Transcribe this audio exactly. Return only the transcribed text, nothing else.",
            ]
        )
        return response.text.strip()


class WhisperSTT:
    """Offline transcription using faster-whisper."""

    def __init__(self, model_name: str = "base", language: str | None = None):
        import os
        from faster_whisper import WhisperModel
        print(f"[STT] Loading Whisper '{model_name}'…")
        try:
            import torch
            device  = "cuda" if torch.cuda.is_available() else "cpu"
            compute = "float16" if device == "cuda" else "int8"
        except Exception:
            device, compute = "cpu", "int8"

        try:
            self._model = WhisperModel(model_name, device=device, compute_type=compute)
        except Exception as _first_err:
            # Offline flag set but model not cached yet → download once, then offline forever
            _e = str(_first_err).lower()
            if any(k in _e for k in ("offline", "not found", "cache", "localentry", "does not exist")):
                print(f"[STT] '{model_name}' not cached — downloading (internet required for first run)…")
                os.environ.pop("HF_HUB_OFFLINE",      None)
                os.environ.pop("TRANSFORMERS_OFFLINE", None)
                os.environ.pop("HF_DATASETS_OFFLINE",  None)
                self._model = WhisperModel(model_name, device=device, compute_type=compute)
            else:
                raise

        self._language = None if (not language or language.strip().lower() == "auto") else language.strip().lower()
        print(f"[STT] Whisper '{model_name}' ready ({device})")

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe a float32 mono 16 kHz numpy array. Returns transcript string."""
        try:
            segments, _ = self._model.transcribe(
                audio,
                language=self._language,
                beam_size=1,                       # greedy — 2-3x faster
                best_of=1,
                condition_on_previous_text=False,  # no hallucinations, faster
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 300},
            )
            return " ".join(s.text for s in segments).strip()
        except Exception as e:
            print(f"[STT] Transcription error: {e}")
            raise


class VoskSTT:
    """Streaming transcription using Vosk."""

    def __init__(self, model_path: str | None = None, language: str = "en-us"):
        from vosk import Model, KaldiRecognizer
        print("[STT] Loading Vosk model…")
        if model_path:
            model = Model(model_path)
        else:
            lang  = language.strip().lower() if language and language.strip().lower() != "auto" else "en-us"
            model = Model(lang=lang)
        self._rec = KaldiRecognizer(model, 16000)
        print("[STT] Vosk ready.")

    def process_chunk(self, audio_bytes: bytes) -> tuple[str, bool]:
        """Feed raw int16 LE PCM bytes. Returns (text, is_final)."""
        if self._rec.AcceptWaveform(audio_bytes):
            result = json.loads(self._rec.Result())
            return result.get("text", ""), True
        partial = json.loads(self._rec.PartialResult())
        return partial.get("partial", ""), False
