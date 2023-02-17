from re import S
import sys
import math
from typing import List
from enum import Enum
import itertools

def debug(s: str):
    print(s, file=sys.stderr, flush=True)
# Auto-generated code below aims at helping you parse
# the standard input according to the problem statement.

# base_x: The corner of the map representing your base
BASE_X, BASE_Y = [int(i) for i in input().split()]
HEROES_PER_PLAYER = int(input())  # Always 3

class EntityType(Enum):
    MONSTER = 0
    MY_HERO = 1
    OPP_HERO = 2

class Entity:
    # _id: Unique identifier
    # _type: 0=monster, 1=your hero, 2=opponent hero
    # x: Position of this entity
    # shield_life: Ignore for this league; Count down until shield spell fades
    # is_controlled: Ignore for this league; Equals 1 when this entity is under a control spell
    # health: Remaining health of this monster
    # vx: Trajectory of this monster
    # near_base: 0=monster with no target yet, 1=monster targeting a base
    # threat_for: Given this monster's trajectory, is it a threat to 1=your base, 2=your opponent's base, 0=neither
    def __init__(self, s: str):
        [
            self.id,
            self.type,
            self.x,
            self.y,
            self.shield_life,
            self.is_controlled,
            self.health,
            self.vx,
            self.vy,
            self.near_base,
            self.threat_for,
        ] = [int(_) for _ in s.split()]
        self.type = EntityType(self.type)

    def calc_distance(self):
        self.base_distance: float = ((self.x - BASE_X)**2 + (self.y - BASE_Y)**2 ) ** 0.5

    def set_command(self, s: str):
        self.command = s

    def execute(self):
        print(self.command)


# game loop
while True:
    for i in range(2):
        # health: Each player's base health
        # mana: Ignore in the first league; Spend ten mana to cast a spell
        health, mana = [int(j) for j in input().split()]

    entity_count = int(input())  # Amount of heros and monsters you can see
    entities: List[Entity] = [Entity(input()) for _ in range(entity_count)]
    monsters: List[Entity] = [e for e in entities if e.type == EntityType.MONSTER]
    threats: List[Entity] = [e for e in monsters if e.threat_for == 1]
    heroes: List[Entity] = [e for e in entities if e.type == EntityType.MY_HERO]

    [e.calc_distance() for e in threats]
    [e.calc_distance() for e in heroes]
    threats.sort(key= lambda e: e.base_distance, reverse=True)
    heroes.sort(key= lambda e: e.base_distance)

    debug(threats)
    debug(heroes)
    for h in heroes:
        h.set_command("WAIT")

    for i in range(HEROES_PER_PLAYER):

        # Write an action using print
        # To debug: print("Debug messages...", file=sys.stderr, flush=True)

        # In the first league: MOVE <x> <y> | WAIT; In later leagues: | SPELL <spellParams>;

        if threats:
            t1: Entity = threats.pop()
            heroes[i].set_command(f"MOVE {t1.x} {t1.y}")
        else:
            heroes[i].set_command(heroes[0].command)

    for h in heroes:
        h.execute()
