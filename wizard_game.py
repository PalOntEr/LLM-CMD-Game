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

