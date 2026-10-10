"""Reusable engine for story-driven chat games powered by a local Ollama model.

A game is a plain `Game` definition (character, story points, moods, ASCII art) living in its own file.
The engine takes care of everything else: the Ollama client, the JSON reply contract, notes-based memory
with background compaction, story progression, speed stats and a ready-made terminal UI.

Minimal game file:

    from engine import Game, StoryPoint, main

    GAME = Game(title="...", character="...", persona="...", story=[StoryPoint(...)], moods=[...], art=my_art)

    if __name__ == "__main__":
        main(GAME)

Animated characters pass `animations=Animations(...)` instead of (or as well as) `art`: a looping idle Clip per
mood, one-shot transition Clips between moods, random Fidgets (blinks, glances) and a thinking loop shown while
the model generates. The terminal UI keeps the art moving in place while the player types (--no-anim turns it
off).

Commands in the terminal UI: /quit, /restart, /stats, /notes
"""
import argparse
import json
import os
import random
import shutil
import sys
import textwrap
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable

DEFAULT_MODEL = "gemma4:e2b"
DEFAULT_HOST = "http://localhost:11434"
MAX_HISTORY = 4  # recent messages sent verbatim; older context lives in the notes
NOTES_LIMIT = 20  # compact the notes in the background once they exceed this
NOTES_TARGET = 10  # roughly how many notes a compaction should leave
ACTIONS = ["stay", "advance"]


# ------------------------------------------------------------------ animation
@dataclass
class Cue:
    """Everything a frame function needs to draw itself at one instant."""
    t: float  # seconds since the animator started; drive never-ending effects (wind, glow) with this
    local: float  # seconds since the current clip started
    progress: float  # 0..1 through a one-shot clip (always 0 for looping clips)
    mood: str  # current mood
    prev_mood: str  # mood before the latest change, handy for transitions


@dataclass
class Clip:
    """An animation: a list of frames played at `fps`, or a function Cue -> frame.

    Looping clips are idles and thinking loops. One-shot clips (loop=False) are transitions and fidgets;
    a one-shot function clip needs its `length` in seconds. All frames of a character should share one size."""
    frames: object  # list[str] | Callable[[Cue], str]
    fps: float = 8.0
    loop: bool = True
    length: float = 0.0

    def __post_init__(self):
        if not callable(self.frames) and not self.frames:
            raise ValueError("A Clip needs at least one frame")
        if not self.loop and self.duration <= 0:
            raise ValueError("A one-shot function Clip needs a length in seconds")

    @property
    def duration(self) -> float:
        if self.loop:
            return float("inf")
        return self.length if callable(self.frames) else len(self.frames) / self.fps

    def render(self, cue: Cue) -> str:
        if callable(self.frames):
            return self.frames(cue)
        i = int(cue.local * self.fps)
        return self.frames[i % len(self.frames) if self.loop else min(i, len(self.frames) - 1)]


@dataclass
class Fidget:
    """A one-shot clip played now and then while the character idles (a blink, a glance around)."""
    clip: Clip
    every: tuple = (4.0, 10.0)  # random wait in seconds before it plays (again)


@dataclass
class Animations:
    idle: dict  # mood -> looping Clip; "*" covers moods without their own
    transitions: dict = field(default_factory=dict)  # (from_mood, to_mood) -> one-shot Clip; either may be "*"
    fidgets: dict = field(default_factory=dict)  # mood -> list[Fidget]; "*" covers moods without their own
    thinking: object = None  # looping Clip shown while the model generates a reply; None keeps idling

    def __post_init__(self):
        if not self.idle:
            raise ValueError("Animations need at least one idle clip")
        oneshots = list(self.transitions.values()) + [f.clip for fs in self.fidgets.values() for f in fs]
        if any(c.loop for c in oneshots):
            raise ValueError("Transition and fidget clips must be one-shot (loop=False)")

    def idle_for(self, mood: str) -> Clip:
        return self.idle.get(mood) or self.idle.get("*") or next(iter(self.idle.values()))

    def fidgets_for(self, mood: str) -> list:
        return self.fidgets.get(mood, self.fidgets.get("*", []))

    def transition(self, old: str, new: str):
        for key in ((old, new), (old, "*"), ("*", new), ("*", "*")):
            if key in self.transitions:
                return self.transitions[key]
        return None


class Animator:
    """Decides what the character shows over time: transitions, the mood's idle loop, random fidgets and the
    thinking loop. UI-free and not thread-safe: call set_mood()/set_thinking() as the game changes and frame()
    whenever you want to draw (from one thread, or under a lock)."""

    def __init__(self, anims: Animations, mood: str, clock=time.monotonic, rng=None):
        self.anims, self.clock = anims, clock
        self.rng = rng or random.Random()
        self.t0 = clock()
        self.mood = self.prev_mood = mood
        self.thinking = False
        self.due = {}  # fidget index -> time it should play, for the current mood's fidgets
        self._settle(self.t0)

    def set_mood(self, mood: str):
        if mood == self.mood:
            return
        self.prev_mood, self.mood = self.mood, mood
        self.due = {}
        now = self.clock()
        clip = self.anims.transition(self.prev_mood, mood)
        if clip:
            self._play(clip, "transition", now)
        else:
            self._settle(now)

    def set_thinking(self, on: bool):
        if on != self.thinking:
            self.thinking = on
            if self.kind != "transition":  # a transition finishes first, then settles into the right loop
                self._settle(self.clock())

    def frame(self, now=None) -> str:
        now = self.clock() if now is None else now
        if now - self.start >= self.clip.duration:
            self._settle(now)
        elif self.kind == "idle":
            ready = [i for i, at in self.due.items() if at <= now]
            if ready:
                i = self.rng.choice(ready)
                del self.due[i]
                self._play(self.anims.fidgets_for(self.mood)[i].clip, "fidget", now)
        local = now - self.start
        progress = 0.0 if self.clip.loop else min(1.0, local / self.clip.duration)
        return self.clip.render(Cue(now - self.t0, local, progress, self.mood, self.prev_mood))

    def _play(self, clip: Clip, kind: str, now: float):
        self.clip, self.kind, self.start = clip, kind, now

    def _settle(self, now: float):
        """Go back to the resting loop: thinking if the model is busy, otherwise the mood's idle."""
        if self.thinking and self.anims.thinking:
            self._play(self.anims.thinking, "thinking", now)
            return
        self._play(self.anims.idle_for(self.mood), "idle", now)
        for i, f in enumerate(self.anims.fidgets_for(self.mood)):
            if self.due.get(i, 0) <= now:  # unscheduled or overdue: (re)schedule so fidgets don't pile up
                self.due[i] = now + self.rng.uniform(*f.every)


class Canvas:
    """Fixed-size character grid for building a frame out of layered sprites; later draws cover earlier ones."""

    def __init__(self, width: int, height: int):
        self.width, self.height = width, height
        self.grid = [[" "] * width for _ in range(height)]

    def draw(self, sprite: str, row=0, col=0, solid=True) -> "Canvas":
        """Stamp a multi-line sprite with its top-left at (row, col); anything off the canvas is cut.
        Spaces before and after the text on each line are see-through; with solid=False inner spaces are too."""
        for r, line in enumerate(sprite.split("\n"), row):
            if not 0 <= r < self.height:
                continue
            first = len(line) - len(line.lstrip(" "))
            for c in range(first, len(line.rstrip(" "))):
                x = col + c
                if 0 <= x < self.width and (solid or line[c] != " "):
                    self.grid[r][x] = line[c]
        return self

    def __str__(self) -> str:
        return "\n".join("".join(row).rstrip() for row in self.grid)


# ------------------------------------------------------------ game definition
@dataclass
class StoryPoint:
    name: str
    goal: str  # what the character is trying to do / what the scene is about
    done: str  # the condition under which the model may answer action="advance"


@dataclass
class Game:
    title: str
    character: str  # display name of the character the player talks to
    persona: str  # system-prompt text: who the character is, the world, and what the player wants
    story: list  # list[StoryPoint], played in order
    moods: list  # strict list of mood strings the model may use; moods[0] is the default
    art: Callable[[str], str] = None  # mood -> still ASCII art (derived from `animations` if omitted)
    animations: Animations = None  # animated art (derived from `art`, standing still, if omitted)
    opening_nudge: str = "(The player approaches. Speak first, in character.)"
    next_nudge: str = "(The story moves on to the next point. Speak first, in character.)"
    ending_text: str = "THE END."
    farewell: str = "Goodbye."  # printed when the program exits
    fallback_line: str = "...Say that again?"  # shown if the model fails twice in a row
    keep_hint: str = "names, promises, motives, key events, opinions"  # what compaction must preserve

    def __post_init__(self):
        if not self.story or not self.moods:
            raise ValueError("A Game needs at least one story point and one mood")
        if self.art is None and self.animations is None:
            raise ValueError("A Game needs art, animations, or both")
        if self.animations is None:
            art = self.art
            self.animations = Animations(idle={"*": Clip(lambda cue: art(cue.mood))})
        if self.art is None:
            anims = self.animations
            self.art = lambda mood: anims.idle_for(mood).render(Cue(0.0, 0.0, 0.0, mood, mood))


# ------------------------------------------------------------------- protocol
def build_schema(moods) -> dict:
    return {
        "type": "object",
        "properties": {
            "speak": {"type": "string"},
            "mood": {"type": "string", "enum": list(moods)},
            "action": {"type": "string", "enum": ACTIONS},
            "notes": {"type": "string"},
        },
        "required": ["speak", "mood", "action", "notes"],
    }


COMPACT_SCHEMA = {
    "type": "object",
    "properties": {"notes": {"type": "string"}},
    "required": ["notes"],
}


def build_system_prompt(game: Game, idx: int, notes=()) -> str:
    sp = game.story[idx]
    notes_block = ""
    if notes:
        notes_block = ("\nYOUR NOTES SO FAR about this player and the conversation (older messages are not shown, "
                       "rely on these):\n" + "\n".join(f"- {n}" for n in notes) + "\n")
    return f"""{game.persona}

The story has {len(game.story)} points. CURRENT STORY POINT ({idx + 1}/{len(game.story)}): {sp.name}
Your goal now: {sp.goal}
Set action to "advance" ONLY when this is true: {sp.done}
Otherwise set action to "stay". Do not rush; never advance on the first reply of a story point unless clearly satisfied.
{notes_block}
You MUST reply with ONLY a JSON object, exactly in this format:
{{"speak": "what you say", "mood": "<one of: {', '.join(game.moods)}>", "action": "<stay or advance>", \
"notes": "<comma-separated short notes>"}}
"notes" = important facts from the player's last message and from your own reply that you must remember later \
(player's name, their motives, claims, promises, what you revealed or did). Short phrases separated by commas, \
no commas inside a single note. Use "" if nothing important."""


def split_notes(raw) -> list:
    """Turn a comma-separated string (or a list, if the model ignores the schema) into clean notes."""
    if isinstance(raw, list):
        raw = ",".join(str(x) for x in raw)
    if not isinstance(raw, str):
        return []
    return [n.strip() for n in raw.split(",") if n.strip()]


def parse_reply(raw: str, moods):
    """Return (dict, ok). Always returns a usable dict; ok=False if it had to be repaired."""
    try:
        data = json.loads(raw)
        if not isinstance(data, dict) or not isinstance(data.get("speak"), str) or not data["speak"].strip():
            raise ValueError("bad speak")
    except (json.JSONDecodeError, ValueError):
        return {"speak": "", "mood": moods[0], "action": "stay", "notes": ""}, False
    ok = data.get("mood") in moods and data.get("action") in ACTIONS and "notes" in data
    data["notes"] = ",".join(split_notes(data.get("notes")))
    if data.get("mood") not in moods:
        data["mood"] = moods[0]
    if data.get("action") not in ACTIONS:
        data["action"] = "stay"
    return data, ok


def chat(host: str, model: str, messages: list, schema: dict, num_predict=400):
    body = json.dumps({
        "model": model, "messages": messages, "stream": False, "format": schema,
        "think": False,  # thinking tokens made each turn ~25x slower
        "options": {"temperature": 0.7, "num_predict": num_predict},
    }).encode()
    req = urllib.request.Request(f"{host}/api/chat", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read())


def check_ollama(host: str, model: str):
    """Raise RuntimeError with a friendly message if Ollama or the model is unavailable."""
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=5) as r:
            names = [m["name"] for m in json.loads(r.read()).get("models", [])]
    except (urllib.error.URLError, OSError):
        raise RuntimeError(f"Cannot reach Ollama at {host}. Start it with 'ollama serve'.")
    if model not in names and f"{model}:latest" not in names:
        raise RuntimeError(f"Model '{model}' not found. Run: ollama pull {model}\n"
                           f"Installed: {', '.join(names) or 'none'}")


