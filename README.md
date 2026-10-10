# LLM CMD Game

A terminal-based, story-driven conversational game framework powered by local LLMs via [Ollama](https://ollama.com).

Features real-time animated ASCII characters that react with mood changes, fidgets, transitions, and thinking loops while you chat in the terminal. No external Python packages are required—it runs purely on the standard library.

---

## Features

- **Living ASCII Animations**: Characters idle, blink, look around, chuckle, and react dynamically. Animations update live on screen even while the player is typing.
- **Ollama Powered**: Runs fully locally with any Ollama model (defaults to `gemma4:e2b`).
- **Structured Game Loop**: Enforces JSON response contracts with schema validation and automatic JSON repair fallbacks.
- **Dynamic Story Progression**: Discrete story points (`StoryPoint`) guide the scene. The LLM decides when the player has satisfied objectives before advancing.
- **Rolling Memory & Compaction**: Maintains running player notes. When notes exceed a threshold, a background thread summarizes and compacts older context without freezing gameplay.
- **Modular Game Design**: Separate game logic (`wizard_game.py`) from the reusable core engine (`engine.py`).

---

## Included Game: *The Old Wizard*

Speak with **Aldrin**, a grumpy, ancient wizard living alone in his stone tower. The player must convince Aldrin to teach them the quiet, patient craft of the *Old Magic* instead of flashy street performer tricks.

The wizard features multi-layered procedural ASCII art rendered on a canvas:
- Flapping hat and trailing beard rippling in fluctuating wind gusts
- A cape billowing to the side
- A pivoting staff with an orb that twinkles, glows, sparks, or flares depending on Aldrin's mood
- Expressive brows, eyelids, and mouth changes across 9 moods (`neutral`, `pleased`, `amused`, `suspicious`, `stern`, `angry`, `thoughtful`, `impressed`, `warm`)

---

## Requirements

- **Python 3.10+** (only standard library modules: `urllib`, `dataclasses`, `argparse`, `threading`, etc.)
- **[Ollama](https://ollama.com)** running locally

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

3. **Launch the game**:
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
  -h, --help      Show help message
```

---

## In-Game Commands

While playing, you can type special commands at the prompt:

- `/notes` – Display all facts and notes the character has recorded about you.
- `/stats` – View turn count, token generation rate (tok/s), latency, and compaction statistics.
- `/restart` – Restart the current story from the beginning.
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
    art=static_art,
)

if __name__ == "__main__":
    main(GAME)
```

You can optionally pass an `Animations` definition for full character animations, fidgets, transitions, and thinking loops.

---

## License

MIT License. See [LICENSE](LICENSE) for details.
