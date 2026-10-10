# LLM CMD Game

A terminal-based, story-driven conversational game framework powered by local LLMs via [Ollama](https://ollama.com) and local text-to-speech voice synthesis via [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M).

Features real-time animated ASCII characters that react with mood changes, fidgets, transitions, thinking loops, and lip-synced speaking animations while you chat in the terminal, complete with voiced dialogue spoken in character.

---

## Features

- **Living ASCII Animations**: Characters idle, blink, look around, chuckle, and speak dynamically. Mouths animate with speech cadence while audio plays, updating live on screen even while the player is typing.
- **Local Character Voice (Kokoro TTS)**: Characters speak their dialogue aloud with fully local, in-memory voice synthesis. Non-blocking audio playback lets ASCII animations and mouth movements run seamlessly while speech plays.
- **Dual Input Modes (Speech & Actions)**: Switch at any time between speaking to the character (`[say]`) and describing physical gestures or actions (`[action]`). The active mode is always clearly displayed on the input prompt.
- **Save & Load System**: Save complete game state and exact model request payloads to JSON with `/save`, reload state and resume waiting for player input with `/load` or `/saves`, or launch directly into a save using `--load`.
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
  --load FILE     Load saved game JSON file on startup
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

- `/save [filename]` – Save the full game session state, conversation history, memory notes, and the exact model payload to a JSON file (defaults to `savegame.json`).
- `/load [filename]` – Load a saved game from JSON, restore the character's visage, mood, and previous dialogue on screen, and resume waiting for the player's next message without triggering an automatic model turn.
- `/saves` – List all available `.json` save files in the current directory with save timestamps and story progress.
- `/switch [say|action]` – Toggle or switch input mode between speech and physical actions.
- `/notes` – Display all facts and notes the character has recorded about you and their own dialogue (`[Player]` vs `[Character]`).
- `/stats` – View turn count, token generation rate (tok/s), latency, and compaction statistics.
- `/restart` – Restart the current story from the beginning.
- `/help` – Show list of available commands.
- `/quit` – Exit the game.

> [!NOTE]
> **Automatic Completion Save**: When a playthrough finishes (all story points are completed), the engine automatically executes the save routine before closing the program. It writes an immutable session log to `completed_<timestamp>.json` and updates `savegame.json`, preserving the complete conversation, memory notes, and final model payload for future reference.

---

## What Is Sent to the Model on Each Call

`engine.py` manages all communication with local Ollama models via HTTP `POST /api/chat`. On every turn (and when using `/save`), the engine constructs and sends a strictly structured JSON payload.

### 1. API Request Structure (`model_payload`)

```json
{
  "model": "gemma4:e2b",
  "messages": [
    { "role": "system", "content": "..." },
    { "role": "user", "content": "[Player Speech] \"...\"" },
    { "role": "assistant", "content": "{\"speak\": \"...\", \"mood\": \"...\", \"action\": \"stay\", \"notes\": \"...\"}" }
  ],
  "stream": false,
  "think": false,
  "format": {
    "type": "object",
    "properties": {
      "speak": { "type": "string" },
      "mood": { "type": "string", "enum": ["..."] },
      "action": { "type": "string", "enum": ["stay", "advance"] },
      "notes": { "type": "string" }
    },
    "required": ["speak", "mood", "action", "notes"]
  },
  "options": {
    "temperature": 0.7,
    "num_predict": 400
  }
}
```

- **`stream: false`**: Responses are received as complete JSON payloads for validation and atomic rendering.
- **`think: false`**: Disables extended thinking tokens, speeding up model turn latency by ~25x.
- **`format`**: Enforces strict structured output. The model is constrained to valid moods defined in `game.moods` and valid actions (`"stay"` or `"advance"`).
- **`options`**: Clamped to `temperature: 0.7` and `num_predict: 400` to keep responses concise and creative without runaway generation.

---

### 2. Dynamic System Prompt (`build_system_prompt`)

The `system` message is dynamically reassembled on every turn from several layers of game state:

1. **Character Persona (`game.persona`)**:
   Defines the character's personality, lore, speaking mannerisms, and rules of engagement.
2. **Current Story Point & Progression Objectives**:
   - Injects the total story length and the active story point index (`idx + 1` of `len(game.story)`).
   - Injects the active scene name and immediate character goal (`sp.goal`).
   - Injects the completion condition (`sp.done`), strictly instructing the model:
     > *"Set action to 'advance' ONLY when this is true: {sp.done}. Otherwise set action to 'stay'. Do not rush; never advance on the first reply of a story point unless clearly satisfied."*
     >
3. **Attributed Memory Notes Snapshot**:
   When active memory notes exist, they are injected directly into the prompt:
   ```text
   YOUR NOTES SO FAR about this player and the conversation (older messages are not shown, rely on these):
   - [Player] asked about the tower
   - [Aldrin] warned that old magic requires patience
   ```
4. **Player Modality Guidance**:
   Instructs the model how to interpret spoken words (`[Player Speech] "..."`) versus physical actions and gestures (`[Player Action] ...`).
5. **Note Generation Protocol**:
   Forces the model to produce comma-separated notes for key facts, requiring explicit `[Player]` or `[{Character}]` attribution tags.

---

### 3. Sliding Conversation Window (`MAX_HISTORY`)

To maintain ultra-fast inference speed and prevent context degradation, `engine.py` applies a sliding window:

- Only the **most recent `MAX_HISTORY` messages** (default: 4 messages, or 2 complete conversation turns) are sent in the `messages` array alongside the `system` message.
- Older messages that slide out of the window are **not lost**; their key decisions, names, and revelations are continuously tracked by the rolling attributed `Memory` system and injected into the dynamic system prompt.
- Full un-truncated dialogue history is always preserved in `engine.history` and written to disk when using `/save`.

---

### 4. Background Memory Compaction

When accumulated memory notes exceed `NOTES_LIMIT` (default: 20 notes), `engine.py` automatically spawns a non-blocking background thread:

- A separate lightweight LLM call merges duplicates, removes obsolete trivia, and compresses the notes into at most `NOTES_TARGET` (default: 10) high-density facts while preserving character and player attributions.
- Gameplay continues without stuttering or input delay while compaction occurs in the background.

---

### 5. Inspecting What Was Sent (`/save`)

Whenever you type `/save` (or `/save <filename>`), the engine writes:

- The exact `model_payload` dictionary sent over the wire
- The exact raw `system_prompt` text
- The `messages_sent_to_model` array
- The full unbroken `history` and `notes`
- The current `story_idx`, `input_mode`, and character `current_mood`

You can open the resulting `.json` file at any time to inspect the exact prompt and token configuration delivered to the model for that turn. When a game completes, this same routine is automatically triggered to create `completed_<timestamp>.json` as an archived post-game session log.

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
