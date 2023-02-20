import sys
import math
from typing import List, Union
from enum import Enum
from collections import namedtuple

Numeric = Union[int, float]

def debug(s: str):
    print(s, file=sys.stderr, flush=True)


class Point:
    __slots__ = ('x', 'y')

    def __init__(self, x: Numeric , y: Numeric):
        self.x, self.y = x, y

    def __str__(self):
        return f"Point({self.x} {self.y})"

    def dist(self, other: 'Point'):
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

    def midpoint(self, other: 'Point'):
        x, y = (self.x + other.x)/2, (self.y + other.y)/2
        return Point(x, y)

    def vector_to(self, other: 'Point') -> 'Vector':
        return Vector(other.x - self.x, other.y - self.y)

    def add_vector(self, v: 'Vector') -> 'Point':
        return Point(self.x + v.x, self.y + v.y)

    @staticmethod
    def centroid(points: List['Point']):
        return Point(
            x=sum(p.x for p in points)/len(points),
            y=sum(p.y for p in points)/len(points)
        )

class Vector:
    __slots__ = ('x', 'y')

    def __init__(self, x: Numeric , y: Numeric):
        self.x, self.y = x, y

    def add(self, v: 'Vector') -> 'Vector':
        return Vector(self.x + v.x, self.y + v.y)

    def scale(self, c: Numeric) -> 'Vector':
        return Vector(self.x * c, self.y * c)

    def norm(self):
        return math.sqrt(self.x**2 + self.y**2)
    
    def normalize(self):
        n: float = self.norm()
        return Vector(self.x/n, self.y/n)

    def __str__(self):
        return f"Vector({self.x} {self.y})"

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
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.atk_neighbors = []
        self.wind_neighbors = []
        
    def find_atk_neighbors(self, monsters: List['Monster']):
        pass

class Hero(Entity):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target = None
        self.opp_neighbors = []

    def _wind(self, point:Point):
        self.command = f"{CommandType.WIND.value} {int(point.x)} {int(point.y)}"
    
    def _move(self, point:Point):
        self.command = f"{CommandType.MOVE.value} {int(point.x)} {int(point.y)}"

    def _shield(self, id:int):
        self.command = f"{CommandType.SHIELD.value} {id}"

    def _control(self, id:int, point:Point):
        self.command = f"{CommandType.CONTROL.value} {id} {int(point.x)} {int(point.y)}"

    def execute(self):
        print(self.command)

    def set_target(self, e: Entity):
        self.target = e
    
    def defensive_wind(self, t: Entity):
        v: Vector = BASE.vector_to(t.loc)
        self._wind(self.loc.add_vector(v))

    def offensive_wind(self, t: Entity):
        v: Vector = t.loc.vector_to(OPP_BASE)
        self._wind(self.loc.add_vector(v))

    def attack_monster(self, t: Entity):
        v: Vector = t.loc.vector_to(self.loc)
        if v.norm() < MELEE_RANGE and MANA_ME > 50 and self.shield_life==0:
            self._shield(self.id)
            return
        elif v.norm() > MELEE_RANGE+HERO_SPEED:
            p:Point = t.loc.add_vector(t.vel)
        else:
            centroid: Point = CENTROID
            #find successful attack spot closest to centroid
            v2: Vector = t.loc.vector_to(centroid)
            v2: Vector = v2.normalize().scale(MELEE_RANGE*0.99)
            debug(f"{self.id} atk {t.id} v2 = {v2} {v2.norm()}")
            p:Point = t.loc.add_vector(v2)

        self._move(p)



    def defend(self):
        if self.shield_life==0 and MANA_ME > 40 and self.opp_neighbors:
            self._shield(self.id)
            return

        if self.shield_life>0 and MANA_ME > 40 and self.opp_neighbors:
            if self.opp_neighbors[0].shield_life == 0:
                t = self.opp_neighbors[0]
                v: Vector = t.loc.vector_to(JAIL)
                self._control(t.id, t.loc.add_vector(v))
                return

        if self.target == None:
            return
        
        t: Entity = self.target
        
        if t.etype == EntityType.MONSTER:
            if t.dist(BASE) < (MONSTER_VISION-WIND_MOVE) and self.loc.dist(t.loc) < WIND_RANGE and t.shield_life==0 and MANA_ME>10:
                self.defensive_wind(t)
            elif MANA_ME > 100 and self.loc.dist(t.loc) < WIND_RANGE and t.shield_life==0:
                self.offensive_wind(t)
            else:
                self.attack_monster(t)
        else:
            if MANA_ME > 50 and self.loc.dist(t.loc) < WIND_RANGE and t.shield_life==0:
                self.defensive_wind(t)
                # v: Vector = BASE.vector_to(t.loc)
                # self._control(t.id, t.loc.add_vector(v))


class AtkHero(Hero):
    pass

class DefHero(Hero):
    pass
class OppHero(Hero):
    pass

class CommandType(Enum):
    MOVE = "MOVE"
    WIND = "SPELL WIND"
    SHIELD = "SPELL SHIELD"
    CONTROL = "SPELL CONTROL"


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
ATTACK_DAMAGE = 2

BASE = Point(BASE_X, BASE_Y)
MAP_X, MAP_Y = 17630, 9000
MAP_MID = Point(MAP_X/2, MAP_Y/2)
if BASE_X == 0:
    OPP_BASE = Point(MAP_X, MAP_Y)
else:
    OPP_BASE = Point(0, 0)

# HERO1_HOME = BASE.midpoint(MAP_MID)
MIRROR = 1 if BASE_X == 0 else -1

_v1 = Vector(math.cos(math.pi/4), math.sin(math.pi/4))
_v2 = Vector(math.cos(math.pi/8), math.sin(math.pi/8))
_v3 = Vector(math.cos(math.pi*3/8), math.sin(math.pi*3/8))
_v4 = Vector(BASE_VISION, MAP_Y)
_scale = (BASE_VISION + HERO_VISION/2) * MIRROR

HERO1_HOME = BASE.add_vector(_v1.scale(_scale*0.75))
HERO2_HOME = BASE.add_vector(_v1.scale(_scale*1))
HERO3_HOME = OPP_BASE.add_vector(_v1.scale(_scale*-1))
JAIL = BASE.add_vector(_v4)

HERO_HOMES = [HERO1_HOME, HERO2_HOME, HERO3_HOME]



# game loop
while True:
    HEALTH_ME, MANA_ME = [int(j) for j in input().split()]
    HEALTH_OPP, MANA_OPP = [int(j) for j in input().split()]
    
    entity_count = int(input())  # Amount of heros and monsters you can see
    entities: List[Entity] = [entityFactory(input()) for _ in range(entity_count)]
    heroes: List[Hero] = [Hero(*e) for e in entities if e.etype == EntityType.MY_HERO]
    atk_heroes : List[AtkHero] = [AtkHero(*heroes[0])]
    def_heroes : List[DefHero] = [DefHero(*heroes[1:])]

    opps: List[OppHero] = [OppHero(*e) for e in entities if e.etype == EntityType.OPP_HERO]
    monsters: List[Monster] = [Monster(*e) for e in entities if e.etype == EntityType.MONSTER]
    threats = monsters + opps

    for h in def_heroes:
        for o in opps:
            if h.dist(o.loc) < CONTROL_RANGE:
                h.opp_neighbors.append(o)

    # def attackers_needed(m:Monster):
    #     time: float = m.dist(BASE)/MONSTER_SPEED
    #     n_attacks: float = m.health / ATTACK_DAMAGE
    #     return math.ceil(n_attacks/time) + (1 if m.dist(BASE) < 2000 else 0)

    # for m in monsters:
    #     for i in range (attackers_needed(m)):
    #         threats.append(m)

    close_monsters = [m for m in monsters if m.dist(BASE)<MONSTER_VISION]
    if len(close_monsters) > 1:
        CENTROID = Point.centroid([m.loc for m in close_monsters])
    else:
        CENTROID = HERO1_HOME

    # threats: List[Entity] = [e for e in monsters if e.threat_for == ThreatFor.MY_BASE]

    def tsort(e: Entity):
        # prioritize incoming threats, otherwise closest monster
        # return (not (e.threat_for==ThreatFor.MY_BASE), e.dist(BASE))
        return (not (e.etype==EntityType.OPP_HERO),e.dist(BASE))

    threats.sort(key= lambda e: tsort(e))
    threats = threats[:3]

    for i in range(HEROES_PER_PLAYER):
        heroes[i]._move(HERO_HOMES[i])


    free_heroes: List[Hero] = [h for h in heroes]
    for t in threats:
        free_heroes.sort(key= lambda e: e.loc.dist(t.loc), reverse=True)
        h: Hero = free_heroes.pop()
        h.set_target(t)

    
    heroes.sort(key=lambda e:e.id)
    for h in heroes:
        h.defend()
        h.execute()
