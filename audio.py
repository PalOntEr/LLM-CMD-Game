"""Audio and text-to-speech capabilities for the story game engine.

Uses Kokoro TTS to generate voice audio for characters in memory, with zero disk
clutter, and plays audio through a background thread so the terminal animation
and player input remain smooth and responsive.
"""
import io
import os
import re
import threading
import warnings
from typing import Optional

# Suppress known non-critical torch / huggingface warnings during audio generation
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

_pipelines = {}
_pipeline_lock = threading.Lock()


def get_lang_code(voice: str) -> str:
    """Infer Kokoro language code from the voice name (e.g. 'bm_george' -> 'b', 'am_adam' -> 'a')."""
    if voice and len(voice) >= 2 and voice[1] in ("m", "f"):
        return voice[0].lower()
    return "a"


def get_pipeline(lang_code: str = "a"):
    """Get or lazily create a Kokoro KPipeline for the given language code."""
    with _pipeline_lock:
        if lang_code not in _pipelines:
            from kokoro import KPipeline
            _pipelines[lang_code] = KPipeline(lang_code=lang_code)
        return _pipelines[lang_code]


def synthesize_wav(text: str, voice: str, speed: float = 1.0, sample_rate: int = 24000) -> Optional[bytes]:
    """Synthesize text into in-memory 16-bit PCM WAV bytes using Kokoro.
    
    Returns None if text is empty or synthesis fails.
    """
    clean_text = text.strip()
    if not clean_text or not voice:
        return None

    # Remove markdown / stage directions like *mumbles* or (chuckles) before speaking
    spoken_text = re.sub(r"\*[^*]+\*", "", clean_text)
    spoken_text = re.sub(r"\([^)]+\)", "", spoken_text)
    spoken_text = re.sub(r"\s+", " ", spoken_text).strip()
    if not spoken_text:
        spoken_text = clean_text

    try:
        import numpy as np
        import soundfile as sf

        lang_code = get_lang_code(voice)
        pipeline = get_pipeline(lang_code)
        generator = pipeline(spoken_text, voice=voice, speed=speed, split_pattern=r"\n+")
        
        chunks = [audio for _, _, audio in generator]
        if not chunks:
            return None

        combined_audio = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]

        buf = io.BytesIO()
        sf.write(buf, combined_audio, sample_rate, format="WAV", subtype="PCM_16")
        return buf.getvalue()
    except Exception as e:
        warnings.warn(f"Speech synthesis error for voice '{voice}': {e}")
        return None


class VoiceSpeaker:
    """Manages audio generation and background playback for a game character."""

    def __init__(self, voice: Optional[str] = None, speed: float = 1.0):
        self.voice = voice
        self.speed = speed
        self._pending_audio: Optional[bytes] = None
        self._current_thread: Optional[threading.Thread] = None

    def prepare(self, text: str) -> Optional[bytes]:
        """Synthesize speech ahead of time (e.g. while the model thinking spinner is active)."""
        if not self.voice:
            return None
        self._pending_audio = synthesize_wav(text, self.voice, self.speed)
        return self._pending_audio

    def play(self, wav_bytes: Optional[bytes] = None, block: bool = False):
        """Play the synthesized WAV audio. Defaults to playing the previously prepared audio."""
        data = wav_bytes or self._pending_audio
        self._pending_audio = None
        if not data:
            return

        self.stop()

        def _worker():
            if os.name == "nt":
                import winsound
                try:
                    winsound.PlaySound(data, winsound.SND_MEMORY)
                except Exception:
                    pass
            else:
                # Fallback for non-Windows platforms if soundfile / sounddevice available
                try:
                    import sounddevice as sd
                    import soundfile as sf
                    audio, sr = sf.read(io.BytesIO(data))
                    sd.play(audio, sr)
                    sd.wait()
                except Exception:
                    pass

        self._current_thread = threading.Thread(target=_worker, daemon=True)
        self._current_thread.start()

        if block:
            self._current_thread.join()

    def speak(self, text: str, block: bool = False):
        """Convenience method to synthesize and immediately play audio."""
        if self.prepare(text):
            self.play(block=block)

    def stop(self):
        """Immediately stop any currently playing audio."""
        if os.name == "nt":
            try:
                import winsound
                winsound.PlaySound(None, 0)
            except Exception:
                pass
        self._current_thread = None

