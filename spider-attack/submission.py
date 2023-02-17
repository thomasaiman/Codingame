from re import S
import sys
import math
from typing import List, SupportsFloat as Numeric
from enum import Enum
import itertools

def debug(s: str):
    print(s, file=sys.stderr, flush=True)


class Point:
    __slots__ = ('x', 'y')

    def __init__(self, x: Numeric , y: Numeric):
        self.x, self.y = int(x), int(y)

    def dist(self, other: 'Point'):
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

    def midpoint(self, other: 'Point'):
        x, y = (self.x + other.x)/2, (self.y + other.y)/2
        return Point(x, y)

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


# Auto-generated code below aims at helping you parse
# the standard input according to the problem statement.

# base_x: The corner of the map representing your base
BASE_X, BASE_Y = [int(i) for i in input().split()]
HEROES_PER_PLAYER = int(input())  # Always 3

BASE_VISION, HERO_VISION = 6000, 2200
BASE = Point(BASE_X, BASE_Y)
MAP_X, MAP_Y = 17630, 9000
MAP_MID = Point(MAP_X/2, MAP_Y/2)

# HERO1_HOME = BASE.midpoint(MAP_MID)
MIRROR = 1 if BASE_X == 0 else -1
BASE_VISION *= MIRROR


HERO1_HOME = Point(
    x = BASE.x + BASE_VISION*math.cos(math.pi/4),
    y = BASE.y + BASE_VISION*math.sin(math.pi/4)
)

HERO2_HOME = Point(
    x = BASE.x + BASE_VISION*math.cos(math.pi/8),
    y = BASE.y + BASE_VISION*math.sin(math.pi/8)
)

HERO3_HOME = Point(
    x = BASE.x + BASE_VISION*math.cos(math.pi*3/8),
    y = BASE.y + BASE_VISION*math.sin(math.pi*3/8)
)

BASE_VISION *= MIRROR

HERO_HOMES = [HERO1_HOME, HERO2_HOME, HERO3_HOME]



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
    threats.sort(key= lambda e: e.base_distance)
    heroes.sort(key= lambda e: e.base_distance)

    debug(threats)
    debug(heroes)
    for i in range(HEROES_PER_PLAYER):
        heroes[i].set_command(f"MOVE {HERO_HOMES[i].x} {HERO_HOMES[i].y}")

    if threats:
        t1: Entity = threats[0]
        for i in range(HEROES_PER_PLAYER):
            heroes[i].set_command(f"MOVE {t1.x} {t1.y}")
            # Write an action using print
            # To debug: print("Debug messages...", file=sys.stderr, flush=True)

            # In the first league: MOVE <x> <y> | WAIT; In later leagues: | SPELL <spellParams>;


    for h in heroes:
        h.execute()
