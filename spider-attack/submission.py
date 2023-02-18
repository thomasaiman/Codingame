import sys
import math
from typing import List, Union
from enum import Enum
# import itertools
from collections import namedtuple

Numeric = Union[int, float]

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

    def vector_to(self, other: 'Point') -> 'Vector':
        return Vector(other.x - self.x, other.y - self.y)

    def add_vector(self, v: 'Vector') -> 'Point':
        return Point(self.x + v.x, self.y + v.y)


class Vector:
    __slots__ = ('x', 'y')

    def __init__(self, x: Numeric , y: Numeric):
        self.x, self.y = int(x), int(y)

    def add(self, v: 'Vector') -> 'Vector':
        return Vector(self.x + v.x, self.y + v.y)

    def scale(self, c: Numeric) -> 'Vector':
        return Vector(self.x * c, self.y * c)

    def norm(self):
        return math.sqrt(self.x**2 + self.y**2)
    

class EntityType(Enum):
    MONSTER = 0
    MY_HERO = 1
    OPP_HERO = 2


class ThreatFor(Enum):
    NOT_MONSTER = -1
    NEITHER = 0
    MY_BASE = 1
    OPP_BASE = 2


EntityBase = namedtuple('Entity', (
    'id',
    'etype',
    'loc',
    'shield_life',
    'is_contolled',
    'health',
    'vel',
    'near_base',
    'threat_for'
    )
)

def entityFactory(s: str) -> 'Entity':
    (       
        id,
        etype,
        x,
        y,
        shield_life,
        is_controlled,
        health,
        vx,
        vy,
        near_base,
        threat_for,
    ) = [int(i) for i in s.split()]
    etype = EntityType(etype)
    loc = Point(x,y)
    vel = Vector(vx, vy)
    is_controlled = bool(is_controlled)
    near_base = bool(near_base)
    threat_for = ThreatFor(threat_for)

    return Entity(id, etype, loc, shield_life, is_controlled, health, vel, near_base, threat_for)
 

class Entity(EntityBase):
    # _id: Unique identifier
    # _type: 0=monster, 1=your hero, 2=opponent hero
    # x: Position of this entity
    # shield_life: Ignore for this league; Count down until shield spell fades
    # is_controlled: Ignore for this league; Equals 1 when this entity is under a control spell
    # health: Remaining health of this monster
    # vx: Trajectory of this monster
    # near_base: 0=monster with no target yet, 1=monster targeting a base
    # threat_for: Given this monster's trajectory, is it a threat to 1=your base, 2=your opponent's base, 0=neither
    def __init__(self, *args, **kwargs):
        super().__init__()
    
    def dist(self, other: Point):
        return self.loc.dist(other)
    
class Monster(Entity):
    pass
    

class Hero(Entity):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target = None
        
    def set_command(self, s: str):
        self.command = s

    def execute(self):
        print(self.command)

    def set_target(self, e: Entity):
        self.target = e
    
    def defend(self):
        if self.target == None:
            return

        t = self.target
        if t.dist(BASE) < 2000 and self.loc.dist(t.loc) < 1280:
            self.set_command(f"SPELL WIND {MAP_MID.x} {MAP_MID.y}")
        else:
            self.set_command(f"MOVE {t.loc.x} {t.loc.y}")

class OppHero(Hero):
    pass



# Auto-generated code below aims at helping you parse
# the standard input according to the problem statement.

# base_x: The corner of the map representing your base
BASE_X, BASE_Y = [int(i) for i in input().split()]
HEROES_PER_PLAYER = int(input())  # Always 3

BASE_VISION, HERO_VISION, MONSTER_VISION = 6000, 2200, 5000
WIND_RANGE, SHIELD_RANGE, CONTROL_RANGE = 1280, 2200, 2200
WIND_MOVE = 2200
HERO_SPEED, MELEE_RANGE = 800, 800
MONSTER_SPEED = 400

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
    entities: List[Entity] = [entityFactory(input()) for _ in range(entity_count)]
    heroes: List[Hero] = [Hero(*e) for e in entities if e.etype == EntityType.MY_HERO]
    enemies: List[OppHero] = [OppHero(*e) for e in entities if e.etype == EntityType.OPP_HERO]
    monsters: List[Monster] = [Monster(*e) for e in entities if e.etype == EntityType.MONSTER]


    threats: List[Entity] = [e for e in monsters if e.threat_for == ThreatFor.MY_BASE]
    threats.sort(key= lambda e: e.dist(BASE))
    threats = threats[:3]

    for i in range(HEROES_PER_PLAYER):
        heroes[i].set_command(f"MOVE {HERO_HOMES[i].x} {HERO_HOMES[i].y}")


    free_heroes: List[Hero] = [h for h in heroes]
    for t in threats:
        free_heroes.sort(key= lambda e: e.loc.dist(t.loc), reverse=True)
        h: Hero = free_heroes.pop()
        h.set_target(t)

    
    heroes.sort(key=lambda e:e.id)
    for h in heroes:
        h.defend()
        h.execute()
