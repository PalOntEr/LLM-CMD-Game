# LLM CMD Game

A terminal-based, story-driven conversational game framework powered by local LLMs via [Ollama](https://ollama.com) and local text-to-speech voice synthesis via [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M).

Features real-time animated ASCII characters that react with mood changes, fidgets, transitions, thinking loops, and lip-synced speaking animations while you chat in the terminal, complete with voiced dialogue spoken in character.

---

## Features

- **Living ASCII Animations**: Characters idle, blink, look around, chuckle, and speak dynamically. Mouths animate with speech cadence while audio plays, updating live on screen even while the player is typing.
- **Local Character Voice (Kokoro TTS)**: Characters speak their dialogue aloud with fully local, in-memory voice synthesis. Non-blocking audio playback lets ASCII animations and mouth movements run seamlessly while speech plays.

- **Dual Input Modes (Speech & Actions)**: Switch at any time between speaking to the character (`[say]`) and describing physical gestures or actions (`[action]`). The active mode is always clearly displayed on the input prompt.
- **Attributed Memory & Compaction**: Maintains rolling memory notes that explicitly track who said or did what (`[Player]` vs `[Character]`). When notes exceed a threshold, a background thread summarizes and compacts older context without freezing gameplay.
- **Ollama Powered**: Runs fully locally with any Ollama model (defaults to `gemma4:e2b`).
- **Structured Game Loop**: Enforces JSON response contracts with schema validation and automatic JSON repair fallbacks.
- **Dynamic Story Progression**: Discrete story points (`StoryPoint`) guide the scene. The LLM decides when the player has satisfied objectives before advancing.
- **Modular Game Design**: Separate game logic (`wizard_game.py`) from the reusable core engine (`engine.py`) and audio layer (`audio.py`).

---

## Included Game: *The Old Wizard*

Speak with **Aldrin**, a grumpy, ancient wizard living alone in his stone tower. The player must convince Aldrin to teach them the quiet, patient craft of the *Old Magic* instead of flashy street performer tricks.

- **Voiced Dialogue**: Voiced using Kokoro's distinguished British scholarly voice (`bm_george`) at a deliberate, contemplative pace.
- **Multi-layered Procedural ASCII Art**:
  - Flapping hat and trailing beard rippling in fluctuating wind gusts
  - A cape billowing to the side
  - A pivoting staff with an orb that twinkles, glows, sparks, or flares depending on Aldrin's mood
  - Expressive brows, eyelids, and mouth changes across 9 moods (`neutral`, `pleased`, `amused`, `suspicious`, `stern`, `angry`, `thoughtful`, `impressed`, `warm`)

---

## Requirements

- **Python 3.10 – 3.12** recommended
- **[Ollama](https://ollama.com)** running locally
- *(Optional, for voice synthesis)*:
  ```bash
  pip install kokoro soundfile
  ```
  *(If you prefer to run silently without installing audio dependencies, pass `--no-voice`)*

---

## Quick Start

1. **Start Ollama** (if not already running as a service):
   ```bash
   ollama serve
   ```

2. **Pull the default model** (or any model of your choice):
   ```bash
   ollama pull gemma4:e2b
   ```

3. **Install audio dependencies** (optional for voice):
   ```bash
   pip install kokoro soundfile
   ```

4. **Launch the game**:
   ```bash
   python wizard_game.py
   ```

### Command-Line Options

```bash
python wizard_game.py [OPTIONS]

Options:
  --model MODEL   Ollama model name (default: gemma4:e2b)
  --host HOST     Ollama API host (default: http://localhost:11434)
  --no-anim       Disable live animations and show still ASCII art
  --no-voice      Disable character voice audio
  -h, --help      Show help message
```

---

## In-Game Controls & Commands

### Input Modes
Your current input mode is always shown directly on the prompt:
- `You [say] > ` – Spoken dialogue sent directly to the character (e.g. `Who built this tower?`).
- `You [action] > ` – Physical gestures, movements, or actions (e.g. `bows deeply and points to the rune on the door`).

### Commands
Type these commands at the prompt at any time:

- `/switch [say|action]` – Toggle or switch input mode between speech and physical actions.
- `/notes` – Display all facts and notes the character has recorded about you and their own dialogue (`[Player]` vs `[Character]`).
- `/stats` – View turn count, token generation rate (tok/s), latency, and compaction statistics.
- `/restart` – Restart the current story from the beginning.
- `/help` – Show list of available commands.
- `/quit` – Exit the game.

---

## Building Your Own Game

To create a new game using `engine.py`, define a `Game` dataclass and run it with `main`:

```python
from engine import Game, StoryPoint, main

STORY = [
    StoryPoint(
        name="Arrival",
        goal="Greet the player and ask why they've entered your shop.",
        done="The player introduces themselves and states what they seek.",
    ),
]

MOODS = ["neutral", "curious", "grumpy"]

def static_art(mood: str) -> str:
    return r"""
       /\_/\
      ( o.o )
       > ^ <
    """

GAME = Game(
    title="The Alchemist's Den",
    character="Grimm",
    persona="You are Grimm, an alchemist running an apothecary. Speak in 2-3 sentences.",
    story=STORY,
    moods=MOODS,
    voice="am_adam",      # Kokoro voice name (e.g. bm_george, am_adam, af_nicole)
    voice_speed=0.95,     # Speech speed multiplier
    art=static_art,
)

if __name__ == "__main__":
    main(GAME)
```

You can optionally pass an `Animations` definition for full character animations, fidgets, transitions, and thinking loops.

---

## License

MIT License. See [LICENSE](LICENSE) for details.
