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

DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "gemma4:e2b")  # Ollama model name (with optional :tag)
DEFAULT_HOST = os.environ.get("DEFAULT_HOST", "http://localhost:11434")  # Ollama server URL
MAX_HISTORY = int(os.environ.get("MAX_HISTORY", 4))  # recent messages sent verbatim; older context lives in the notes
NOTES_LIMIT = int(os.environ.get("NOTES_LIMIT", 20))  # compact the notes in the background once they exceed this
NOTES_TARGET = int(os.environ.get("NOTES_TARGET", 10))  # roughly how many notes a compaction should leave
ACTIONS = os.environ.get("ACTIONS", "stay,advance").split(",")  # available actions in the game by the model


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


# --------------------------------------------------------------------- memory
class Memory:
    """Rolling notes about the conversation, compacted by a background model call when they grow."""

    def __init__(self, game: Game, host: str, model: str):
        self.game, self.host, self.model = game, host, model
        self.notes = []
        self.lock = threading.Lock()
        self.compacting = False
        self.compactions = 0
        self.compact_failures = 0
        self.last_compact_s = 0.0

    def add(self, raw):
        with self.lock:
            known = {n.lower() for n in self.notes}
            for n in split_notes(raw):
                if n.lower() not in known:
                    self.notes.append(n)
                    known.add(n.lower())
            if len(self.notes) > NOTES_LIMIT and not self.compacting:
                self.compacting = True
                threading.Thread(target=self._compact, args=(list(self.notes),), daemon=True).start()

    def snapshot(self) -> list:
        with self.lock:
            return list(self.notes)

    def _compact(self, old):
        t0 = time.time()
        prompt = (f"Below are notes about a conversation between {self.game.character} and a player. Merge "
                  f"duplicates, drop trivia, and keep every important fact ({self.game.keep_hint}). "
                  f"Rewrite them as at most {NOTES_TARGET} short notes, separated by commas, with no commas "
                  f"inside a single note.\n\nNOTES:\n" + "\n".join(f"- {n}" for n in old))
        new = []
        try:
            resp = chat(self.host, self.model, [{"role": "user", "content": prompt}], COMPACT_SCHEMA, 300)
            new = split_notes(json.loads(resp.get("message", {}).get("content", "")).get("notes"))
        except (OSError, ValueError, AttributeError):
            pass
        with self.lock:
            if new and len(new) < len(old):
                # notes added while compacting sit after the snapshot; keep them
                self.notes = new + self.notes[len(old):]
                self.compactions += 1
            else:
                self.compact_failures += 1
            self.compacting = False
            self.last_compact_s = time.time() - t0


@dataclass
class Stats:
    turns: int = 0
    tokens: int = 0
    gen_ns: int = 0
    latency: float = 0.0
    repairs: int = 0
    failures: int = 0

    def report(self) -> str:
        if not self.turns:
            return "No turns yet."
        tps = self.tokens / (self.gen_ns / 1e9) if self.gen_ns else 0
        return (f"Turns: {self.turns} | avg {tps:.1f} tok/s | avg latency {self.latency / self.turns:.1f}s | "
                f"invalid-JSON repairs: {self.repairs} | total failures: {self.failures}")


# --------------------------------------------------------------------- engine
@dataclass
class Turn:
    """Result of one model turn, ready for any UI to display."""
    speak: str
    mood: str
    action: str
    footer: str
    story_idx: int  # story point this reply belongs to
    advanced: bool = False  # this reply completed the story point
    finished: bool = False  # ...and it was the last one


class Engine:
    """UI-free game session: call start(), then say() for each player message."""

    def __init__(self, game: Game, host=DEFAULT_HOST, model=DEFAULT_MODEL):
        self.game, self.host, self.model = game, host, model
        self.schema = build_schema(game.moods)
        self.idx = 0
        self.history = []
        self.memory = Memory(game, host, model)
        self.stats = Stats()

    @property
    def story_point(self) -> StoryPoint:
        return self.game.story[self.idx]

    def start(self) -> Turn:
        """The character speaks first."""
        return self._turn([{"role": "user", "content": self.game.opening_nudge}])

    def say(self, text: str) -> Turn:
        """Send a player message. If the reply has advanced=True and not finished, call continue_story() next."""
        self.history.append({"role": "user", "content": text})
        turn = self._turn(self.history)
        if turn.action == "advance":
            turn.advanced = True
            self.idx += 1
            turn.finished = self.idx >= len(self.game.story)
        return turn

    def continue_story(self) -> Turn:
        """The character opens the next story point (call after an advanced turn)."""
        return self._turn(self.history + [{"role": "user", "content": self.game.next_nudge}])

    def _turn(self, context) -> Turn:
        """Ask the model (up to 2 attempts), record the reply in history and notes."""
        idx = self.idx
        system = build_system_prompt(self.game, idx, self.memory.snapshot())
        messages = [{"role": "system", "content": system}] + context[-MAX_HISTORY:]
        moods = self.game.moods
        t0 = time.time()
        for _ in range(2):
            resp = chat(self.host, self.model, messages, self.schema)
            reply, ok = parse_reply(resp.get("message", {}).get("content", ""), moods)
            self.stats.tokens += resp.get("eval_count", 0)
            self.stats.gen_ns += resp.get("eval_duration", 0)
            if reply["speak"]:
                if not ok:
                    self.stats.repairs += 1
                break
            self.stats.repairs += 1
        else:
            self.stats.failures += 1
            reply = {"speak": self.game.fallback_line, "mood": moods[0], "action": "stay", "notes": ""}
        self.history.append({"role": "assistant", "content": json.dumps(reply)})
        self.memory.add(reply["notes"])
        elapsed = time.time() - t0
        self.stats.turns += 1
        self.stats.latency += elapsed
        n, d = resp.get("eval_count", 0), resp.get("eval_duration", 0)
        tps = n / (d / 1e9) if d else 0
        footer = (f"[{n} tokens | {tps:.1f} tok/s | {elapsed:.1f}s | mood={reply['mood']} "
                  f"action={reply['action']} | notes={len(self.memory.snapshot())}"
                  f"{' (compacting)' if self.memory.compacting else ''}]")
        return Turn(reply["speak"], reply["mood"], reply["action"], footer, idx)

    def stats_report(self) -> str:
        m = self.memory
        return (f"{self.stats.report()}\nNotes: {len(m.snapshot())} | compactions: {m.compactions} "
                f"(failed: {m.compact_failures}, last took {m.last_compact_s:.1f}s)")


# ---------------------------------------------------------------- terminal UI
def enable_ansi() -> bool:
    """True if stdout understands ANSI escape codes (switching them on for Windows consoles)."""
    if not sys.stdout.isatty():
        return False
    if os.name != "nt":
        return True
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        return bool(mode.value & 0x4 or kernel32.SetConsoleMode(handle, mode.value | 0x4))  # VT processing
    except (AttributeError, OSError):
        return False


def text_rows(text: str) -> int:
    """How many terminal rows `text` takes once long lines wrap."""
    cols = shutil.get_terminal_size().columns
    return sum(max(1, -(-len(line) // cols)) for line in text.split("\n"))


class Stage:
    """The terminal screen. A background thread keeps the character's art animating in place, above the input
    line, even while the player is typing.

    The art is redrawn with ANSI cursor moves, so every row printed below it must go through print()/ask():
    the stage counts them to know where the art has scrolled to."""

    def __init__(self, game: Game, animate=True, fps=12):
        self.game, self.fps = game, fps
        self.animator = Animator(game.animations, game.moods[0])
        self.live = animate and enable_ansi()
        self.lock = threading.RLock()
        self.rows = 0  # rows printed since the screen was cleared, i.e. the cursor's row
        self.art_top = None  # row where the art starts; None while no art is on screen
        self.drawn = []  # art lines as they are on screen right now
        self.stopped = threading.Event()
        if self.live:
            threading.Thread(target=self._run, daemon=True).start()

    def print(self, text=""):
        with self.lock:
            print(text, flush=True)
            self.rows += text_rows(text)

    def ask(self, prompt: str) -> str:
        text = input(prompt)  # without the lock, so the art keeps moving while the player types
        with self.lock:
            self.rows += text_rows(prompt + text)
        return text

    def clear(self):
        with self.lock:
            if self.live:
                sys.stdout.write("\x1b[2J\x1b[3J\x1b[H")
                sys.stdout.flush()
            else:
                os.system("cls" if os.name == "nt" else "clear")
            self.rows, self.art_top, self.drawn = 0, None, []

    def show(self, turn: Turn = None, banner=""):
        """Redraw the whole screen for a reply (or, with no turn, for the wait before the first one)."""
        game = self.game
        with self.lock:
            if turn:
                self.animator.set_thinking(False)
                self.animator.set_mood(turn.mood)
            idx = turn.story_idx if turn else 0
            self.clear()
            self.print(f"=== {game.title.upper()} ===   Story point {idx + 1}/{len(game.story)}: "
                       f"{game.story[idx].name}")
            frame = self.animator.frame() if self.live else game.art(self.animator.mood)
            self.art_top, self.drawn = self.rows, frame.split("\n")
            self.print(frame)
            self.print()
            if turn:
                self.print(textwrap.fill(f"{game.character} ({turn.mood}): \"{turn.speak}\"", width=78))
                if banner:
                    self.print(f"\n  *** {banner} ***")
                self.print(f"\n  {turn.footer}\n")

    @contextmanager
    def thinking(self, label: str):
        """Play the thinking animation and a spinner while the model works. The next show() ends the
        thinking animation together with the mood change, so the old mood's idle never flashes in between."""
        with self.lock:
            self.animator.set_thinking(True)
        stop = threading.Event()

        def spin():
            frames, i = "|/-\\", 0
            while not stop.wait(0.15):
                with self.lock:
                    sys.stdout.write(f"\r  {label} {frames[i % 4]}")
                    sys.stdout.flush()
                i += 1
            with self.lock:
                sys.stdout.write("\r" + " " * (len(label) + 4) + "\r")
                sys.stdout.flush()

        th = threading.Thread(target=spin, daemon=True)
        if sys.stdout.isatty():
            th.start()
        try:
            yield
        finally:
            stop.set()
            if th.is_alive():
                th.join()

    def close(self):
        """Stop animating; the art stays on screen as it is."""
        self.stopped.set()
        with self.lock:
            self.art_top = None

    def _run(self):
        while not self.stopped.wait(1 / self.fps):
            with self.lock:
                self._draw()

    def _draw(self):
        if self.art_top is None:
            return
        size = shutil.get_terminal_size()
        lines = self.animator.frame().split("\n")[:len(self.drawn)]
        lines += [""] * (len(self.drawn) - len(lines))
        if max(map(len, lines)) > size.columns:  # wrapped lines would wreck the layout
            return
        scroll = max(0, self.rows - size.lines + 1)  # rows the screen has scrolled up since it was cleared
        out = []
        for i, line in enumerate(lines):
            row = self.art_top + i - scroll
            if line != self.drawn[i] and row >= 0:
                out.append(f"\x1b[{row + 1};1H{line.ljust(len(self.drawn[i]))}")  # pad to erase leftovers
                self.drawn[i] = line
        if out:
            # save cursor + hide it, draw, restore + show: whatever the player is typing stays put
            sys.stdout.write("\x1b7\x1b[?25l" + "".join(out) + "\x1b8\x1b[?25h")
            sys.stdout.flush()


def play_terminal(game: Game, host: str, model: str, animate=True) -> bool:
    """Run one playthrough. Returns True if the player asked to restart."""
    engine = Engine(game, host, model)
    stage = Stage(game, animate)
    label = f"{game.character} ponders"
    try:
        stage.show()
        with stage.thinking(label):
            turn = engine.start()
        stage.show(turn)

        while True:
            try:
                text = stage.ask("You > ").strip()
            except (EOFError, KeyboardInterrupt):
                stage.print()
                return False
            if not text:
                continue
            if text == "/quit":
                return False
            if text == "/restart":
                return True
            if text == "/stats":
                stage.print(engine.stats_report())
                continue
            if text == "/notes":
                for n in engine.memory.snapshot():
                    stage.print(f"  - {n}")
                continue

            with stage.thinking(label):
                turn = engine.say(text)
            if turn.advanced:
                stage.show(turn, f"Story point completed: {game.story[turn.story_idx].name}")
                if turn.finished:
                    stage.print(game.ending_text)
                    stage.print(engine.stats_report())
                    return False
                stage.ask("  (press Enter to continue)")
                with stage.thinking(label):
                    turn = engine.continue_story()
            stage.show(turn)
    finally:
        stage.close()


def main(game: Game):
    """Entry point for game files: parses --model/--host/--no-anim, checks Ollama and runs the terminal UI."""
    ap = argparse.ArgumentParser(description=game.title)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--no-anim", action="store_true", help="show still art instead of animations")
    args = ap.parse_args()
    try:
        check_ollama(args.host, args.model)
        while play_terminal(game, args.host, args.model, animate=not args.no_anim):
            pass
    except RuntimeError as e:
        sys.exit(str(e))
    except urllib.error.URLError as e:
        sys.exit(f"Lost connection to Ollama: {e}")
    print(game.farewell)
