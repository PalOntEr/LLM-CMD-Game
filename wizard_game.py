"""The Old Wizard - game definition for engine.py.

Usage: python wizard_game.py [--model gemma4:e2b] [--host http://localhost:11434] [--no-anim]
"""
import math
import random
from dataclasses import dataclass, replace

from engine import Animations, Canvas, Clip, Fidget, Game, StoryPoint, main

MOODS = ["neutral", "pleased", "amused", "suspicious", "stern",
         "angry", "thoughtful", "impressed", "warm"]

PERSONA = """You are Aldrin, a grumpy, ancient wizard living alone in a stone tower. You speak in short, \
characterful sentences (2-4 sentences max). You never break character and never mention being an AI.
The player is a stranger who wants to learn the OLD MAGIC (ancient, quiet, patient craft) - not the new flashy \
show magic used by street performers."""

STORY = [
    StoryPoint(
        name="Meeting the wizard",
        goal="The player has just found the wizard's tower. Greet them gruffly and ask who they are and why they came.",
        done="The player has introduced themselves and said why they came, in a polite, non-hostile way.",
    ),
    StoryPoint(
        name="Getting to know the wizard",
        goal="Let the player ask about you, your life, your tower. Answer briefly, in character, with some mystery.",
        done="The player has asked at least two genuine questions about you or your life and listened with respect.",
    ),
    StoryPoint(
        name="Convincing the wizard to teach you",
        goal=("You are suspicious. Many apprentices only want flashy show magic (fireworks, tricks, fame). "
              "Test whether the player truly wants the ANCIENT ways of old magic out of curiosity."),
        done=("The player has convincingly shown sincere curiosity about the old magic, and clearly rejected "
              "flashy show magic. Be hard to impress: one empty claim is not enough. If they boast, ask for "
              "fame or power, or only say 'teach me', stay."),
    ),
    StoryPoint(
        name="Getting taught a simple spell to open doors",
        goal=("Teach the player a simple door-opening spell. Invent a short incantation in a made-up ancient "
              "tongue (e.g. 'Aperi Voltu') and explain the gesture and the focus needed. Ask them to try it."),
        done="The player has spoken the incantation (or a close attempt) and described the gesture or focus.",
    ),
    StoryPoint(
        name="Thanking the wizard for his trust",
        goal="The spell works. Wait for the player to say farewell and thank you.",
        done="The player has sincerely thanked you for teaching and trusting them.",
    ),
]

# ---------------------------------------------------------------- ASCII art
# The wizard is drawn in layers on a Canvas, back to front: wind streaks, cape, staff, robe, hat, face, beard,
# hand, effects. The wind blows from the right, so the hat tip, beard and cape all stream to the left.
# Columns are laid out around a centre line at x=28.5 (the face spans 15..42).
W, H = 62, 24
STAFF_X, HAND_ROW, ORB_ROW = 52, 19, 2  # the staff pivots in his fist when he waves it

HAT_TIPS = [  # rows 0-2: the floppy tip with its star tassel, flapping up / level / down
    ("                  *-.__   _.-.\n"
     "                       `-'  * \\"),
    ("                         _.--.\n"
     "                *~-.__.--'' * \\"),
    ("                         _.--.\n"
     "                    _.--''' * \\\n"
     "                 *-'"),
]

HAT = """\
                          / .  \\
                         /  *   \\
                        / .   (  \\
                       /  *  .  . \\
                      /  .   *     \\
         ____________/______________\\____________
        '-.______________________________________.-'"""

FACE = """\
               |\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/|
               |                          |
               |                          |
                \\            /\\          /
                 \\.   ,~~~~(__)~~~~,   ./"""

ROBE = """\
             _.-'                          '-._
           .'   \\                          /   '.
          /      \\                        /      \\
          |   |   |                      |   |\\   \\
          |   |   |                      |   | \\   \\
          |   |   |                      |   |  \\___\\
          |   |   |                      |   |   |
          |  (_)  |                      |   |   |
          |_______|______________________|___|___|
                (____)                (____)"""

BEARD_WIDTHS = [24, 22, 20, 18, 16, 14, 12, 8]  # rows 14..21, hanging from under the moustache

# Per-mood features. Brows are (left, right), 7 characters each; a leading "^" lifts that brow a row.
BROWS = {
    "neutral": (",~~~~~,", ",~~~~~,"), "pleased": (".-~~~-.", ".-~~~-."), "warm": (".-~~~-.", ".-~~~-."),
    "amused": ("^.~~~~~.", ",~~~~~,"), "suspicious": ("^.~~~~~.", "__---~~"), "stern": ("~~--.._", "_..--~~"),
    "angry": ("\\\\-.__ ", " __.-//"), "thoughtful": ("^.~~~~~.", "~~--.._"), "impressed": ("^.~~~~~.", "^.~~~~~."),
}
LIDS = {"neutral": "open", "pleased": "happy", "warm": "happy", "amused": "happy", "suspicious": "narrow",
        "stern": "narrow", "angry": "fury", "thoughtful": "open", "impressed": "wide"}
MOUTHS = {"neutral": " ____ ", "pleased": "\\____/", "warm": "\\____/", "amused": " \\__/~", "suspicious": " ~~~~ ",
          "stern": " ---- ", "angry": "/VVVV\\", "thoughtful": "  ~~  ", "impressed": " (  ) "}
ORBS = {"pleased": "+", "warm": "+", "amused": "+", "angry": "@", "impressed": "*"}


@dataclass
class Pose:
    """Everything that can change between frames; draw() turns a Pose into ASCII art."""
    mood: str
    look: int = 0  # pupils: -1 left, 0 ahead, 1 right, 2 up
    lid: str = ""  # eyelids, overriding the mood's ("closed" blinks)
    mouth: str = ""  # 6 characters, overriding the mood's mouth
    tilt: float = 0.0  # staff lean in columns per 6 rows; positive leans right
    orb: str = ""  # orb effect: glow, twinkle, sparks, flare
    power: float = 1.0  # strength of the orb effect, 0..1
    bubbles: float = -1.0  # thought bubbles: 0..1 through their cycle, negative for none
    words: str = ""  # a little word floating up beside the head ("hmph", "ho ho")
    words_at: float = 0.0  # 0..1, how far the word has floated


def wave(t: float, speed: float, phase=0.0) -> float:
    """0..1 sine wave."""
    return 0.5 + 0.5 * math.sin(t * speed - phase)


def gust(t: float) -> float:
    """Wind strength: rises and falls slowly so the flapping never looks mechanical."""
    return 0.55 + 0.25 * math.sin(t * 0.7) + 0.2 * math.sin(t * 1.9 + 1)


def eye(lid: str, look: int) -> str:
    if lid == "closed":
        return "(---)"
    fill, pupil = {"open": (" ", "o"), "wide": (" ", "O"), "happy": (" ", "^"),
                   "narrow": ("-", "o"), "fury": ("=", "*")}[lid]
    if look == 2:  # rolled up in thought
        pupil = "'" if lid in ("open", "wide") else pupil
        look = 0
    inner = [fill] * 3
    inner[look + 1] = pupil
    return "(" + "".join(inner) + ")"


def draw_wind(c: Canvas, t: float):
    for row, speed, streak in ((4, 30, "~~~"), (12, 42, "- -"), (20, 36, "~~ ~")):
        x = W - int(t * speed + row * 13) % (W + 12)
        c.draw(streak, row, x, solid=False)


def draw_cape(c: Canvas, t: float):
    """The cape streams out to the left of the robe, its edge and folds rippling with the wind."""
    g, prev = gust(t), 11
    for i, row in enumerate(range(15, 23)):
        x = 9 - i - round(i / 7 * 5 * g * wave(t, 3.1, i * 0.8))
        edge = "/" if x < prev else ("|" if x == prev else "\\")
        fold = "(" if wave(t, 3.1, i * 0.8 + 1.5) > 0.5 else ")"
        mid = (10 - x) // 2
        body = " " * (mid - 1) + fold + " " * (10 - x - mid) if mid > 1 else " " * (10 - x)
        c.draw(edge + body, row, x)
        prev = x
    c.draw("".join("~-._"[(i + int(t * 6)) % 4] for i in range(10 - prev)), 23, prev)


def staff_x(tilt: float, k: int) -> int:
    """Column of the staff k rows above the fist (negative k: below it)."""
    return STAFF_X + round(tilt * k / 6)


def draw_staff(c: Canvas, tilt: float) -> int:
    """Draw the staff pivoting in the fist; returns the column of the orb on top."""
    for row in range(4, H):
        k = HAND_ROW - row
        x, below = staff_x(tilt, k), staff_x(tilt, k - 1)
        c.draw("|" if x == below else ("/" if x > below else "\\"), row, x)
    return staff_x(tilt, HAND_ROW - ORB_ROW)


def draw_orb(c: Canvas, x: int, pose: Pose, t: float):
    c.draw(f" .-.\n( {ORBS.get(pose.mood, 'o')} )\n '-'", ORB_ROW - 1, x - 2)
    r, p = ORB_ROW, pose.power
    if pose.orb == "glow":  # a halo breathing in and out
        c.draw(".", r - 1, x - 4).draw(".", r - 1, x + 4).draw("'", r + 1, x - 4).draw("'", r + 1, x + 4)
        if wave(t, 3) > 1 - p:
            c.draw(".", r - 2, x - 6).draw(".", r - 2, x + 6).draw(":", r, x - 7).draw(":", r, x + 7)
    elif pose.orb in ("twinkle", "sparks"):  # glints popping up around the orb
        fast = pose.orb == "sparks"
        rng = random.Random(int(t * (12 if fast else 4)))
        for _ in range(round((7 if fast else 3) * p)):
            dr, dc = rng.randint(-2, 2), rng.choice([-1, 1]) * rng.randint(4, 8 if fast else 6)
            c.draw(rng.choice("*'`.,+\"" if fast else "+.*'"), r + dr, x + dc)
    elif pose.orb == "flare" and wave(t, 7) > 1 - p:  # rays bursting out
        c.draw("\\   |   /", r - 2, x - 4, solid=False).draw("--", r, x - 6).draw("--", r, x + 5)
        c.draw("/       \\", r + 2, x - 4, solid=False)


def draw_bubbles(c: Canvas, p: float):
    """Thought bubbles drifting up from under the hat, then a cloud filling with dots."""
    if p >= 0:
        c.draw("o", 6, 14)
    if p >= 0.2:
        c.draw("O", 4, 11)
    if p >= 0.4:
        dots = " ".join("." * (int((p - 0.4) / 0.6 * 4) > i) or " " for i in range(3))
        c.draw(" .-~~~~-.\n(  " + dots + "  )\n '-~~~~-'", 0, 1)


def draw(t: float, pose: Pose) -> str:
    c = Canvas(W, H)
    draw_wind(c, t)
    draw_cape(c, t)
    orb_x = draw_staff(c, pose.tilt)
    c.draw(ROBE, 14, 0)
    c.draw(HAT, 2, 0)
    c.draw(HAT_TIPS[int(t * 2.5 * (0.6 + gust(t))) % 3], 0, 0)
    c.draw(FACE, 9, 0)
    for side, col in ((0, 18), (1, 33)):
        brow = BROWS[pose.mood][side]
        c.draw(brow.lstrip("^"), 9 if brow.startswith("^") else 10, col)
        c.draw(eye(pose.lid or LIDS[pose.mood], pose.look), 11, col + 1)
    # the beard: each row trails further downwind the lower it hangs, with a wave rolling down it
    g = gust(t)
    for i, width in enumerate(BEARD_WIDTHS):
        line = "(" * (width // 2) + ")" * (width // 2)
        if i == 0:
            line = line[:9] + (pose.mouth or MOUTHS[pose.mood]) + line[15:]
        elif i == len(BEARD_WIDTHS) - 1:
            line = "\\" + line[1:-1] + "/"
        drift = -round(i / 7 * 3.4 * g * wave(t, 3.1, i * 0.9))
        c.draw(line, 14 + i, 17 + (24 - width) // 2 + drift)
    c.draw("(#)", HAND_ROW, STAFF_X - 1)
    draw_orb(c, orb_x, pose, t)
    if pose.bubbles >= 0:
        draw_bubbles(c, pose.bubbles)
    if pose.words:
        c.draw(pose.words, 13 - round(pose.words_at * 4), 44)
    return str(c)


# ----------------------------------------------------------------- animation
def idle_pose(mood: str, t: float) -> Pose:
    """How the wizard stands around in each mood. Uses global time so fidgets blend in without a jump."""
    if mood == "angry":  # shakes his staff in fits, sparks flying
        return Pose(mood, tilt=2.6 * math.sin(t * 8) * max(0.0, math.sin(t * 1.3)), orb="sparks")
    if mood == "suspicious":  # shifty eyes
        return Pose(mood, look=(0, 1, 1, 0, -1, -1)[int(t / 0.8) % 6])
    if mood == "thoughtful":
        return Pose(mood, look=2, orb="glow", power=0.3)
    if mood == "impressed":
        return Pose(mood, orb="flare", power=0.35)
    if mood == "warm":
        return Pose(mood, orb="glow", power=0.6)
    if mood in ("pleased", "amused"):
        return Pose(mood, orb="twinkle")
    return Pose(mood)


def clip(pose_at, length=0.0) -> Clip:
    """A Clip drawn from pose_at(cue) -> Pose; one-shot when given a length."""
    return Clip(lambda cue: draw(cue.t, pose_at(cue)), loop=not length, length=length)


def idle_with(**changes):
    """Pose function: the current mood's idle pose with some fields changed."""
    return lambda cue: replace(idle_pose(cue.mood, cue.t), **changes)


def floating(word: str, length: float, **changes) -> Clip:
    """One-shot: a word drifts up beside his head."""
    return clip(lambda cue: replace(idle_pose(cue.mood, cue.t), words=word, words_at=cue.progress, **changes),
                length)


def look_around(cue) -> Pose:
    look = next(look for until, look in ((0.3, 0), (1.0, -1), (1.3, 0), (2.1, 1), (9, 0)) if cue.local < until)
    return replace(idle_pose(cue.mood, cue.t), look=look)


def chuckle(cue) -> Pose:
    pose = replace(idle_pose(cue.mood, cue.t), words="ho ho", words_at=cue.progress)
    return replace(pose, mouth="\\_oo_/" if int(cue.local * 6) % 2 else "\\____/")


def blink_between(cue) -> Pose:
    """Generic mood change: blink, and open the eyes on the new expression."""
    mood = cue.mood if cue.progress >= 0.5 else cue.prev_mood
    return replace(idle_pose(mood, cue.t), lid="closed" if 0.25 < cue.progress < 0.75 else "")


def flare_up(cue) -> Pose:
    """Into anger: the staff whips back and forth and the orb bursts."""
    p = cue.progress
    return Pose(cue.mood, tilt=3 * math.sin(p * 3 * math.pi) * (1 - p), orb="flare" if p > 0.3 else "sparks",
                words="!!", words_at=p)


def calm_down(cue) -> Pose:
    """Out of anger: eyes shut, a long breath, the sparks die out."""
    p = cue.progress
    return replace(idle_pose(cue.mood, cue.t), lid="closed" if p < 0.6 else "", orb="sparks", power=1 - p,
                   words="hmph", words_at=p)


def wonder(cue) -> Pose:
    return Pose(cue.mood, orb="flare", power=1.0, words="oh!", words_at=cue.progress)


def ponder(cue) -> Pose:
    """While the model thinks: eyes up, thought bubbles, the orb softly glowing."""
    return Pose(cue.mood, look=2, orb="glow", power=0.5, bubbles=(cue.local * 0.4) % 1)


BLINK = Fidget(clip(idle_with(lid="closed"), 0.15), every=(2.5, 6.0))
LOOK = Fidget(clip(look_around, 2.6), every=(6.0, 12.0))
HMPH = Fidget(floating("hmph", 1.4), every=(5.0, 10.0))

