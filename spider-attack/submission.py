import sys
import math
from typing import List, Union, Dict, Iterator, Tuple, Optional
from enum import Enum
from collections import namedtuple, OrderedDict
from itertools import chain

Numeric = Union[int, float]

def debug(s: str):
    print(s, file=sys.stderr, flush=True)


class Point:
    __slots__ = ('x', 'y')

    def __init__(self, x: Numeric , y: Numeric):
        self.x, self.y = x, y

    def __str__(self):
        return f"Point({self.x} {self.y})"

    def __hash__(self):
        return hash((self.x, self.y))

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


def circle_intersection(p1:Point, p2:Point, r: Numeric) -> Tuple[Point, Point]:
    """Calculate the intersection points of two circles with the same radius."""
    x1,y1,x2,y2 = p1.x, p1.y, p2.x, p2.y
    d = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)  # distance between circle centers
    if d > 2 * r:  # circles do not overlap
        raise ValueError('no overlap')
    else:
        a = (r**2 - (d/2)**2) ** 0.5
        xm = (x1 + x2) / 2
        ym = (y1 + y2) / 2
        xs1 = xm + a * (y2 - y1) / d
        xs2 = xm - a * (y2 - y1) / d
        ys1 = ym - a * (x2 - x1) / d
        ys2 = ym + a * (x2 - x1) / d
        return (Point(xs1, ys1), Point(xs2, ys2))



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
    'is_controlled',
    'health',
    'vel',
    'near_base',
    'threat_for'
    )
)

def entityBaseFactory(s: str) -> EntityBase:
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

    return EntityBase(id, etype, loc, shield_life, is_controlled, health, vel, near_base, threat_for)

class EntityTracker:
    def __init__(self):
        self.heroes: Dict[int, Hero ] = {}
        self.opps: Dict[int, OppHero] = {}
        self.monsters: Dict[int, Monster] = {}

    def update(self, n: int):
        [m.advance() for m in self.monsters.values()]
        for m_id in list(self.monsters.keys()):
            m:Monster = self.monsters[m_id]
            c:Monster = m.counterpart()
            if not (c.id in self.monsters):
                self.monsters[c.id] = c

        for e in self.all_entities():
            e.is_winded = False
        self.prune_monsters()
        self.opps = {}
        updates: List[EntityBase] = [entityBaseFactory(input()) for _ in range(n)]
        for eb in updates:
            if eb.etype == EntityType.MONSTER:
                e: Entity = Monster(eb)
                self.monsters[e.id] = e
            elif eb.etype == EntityType.MY_HERO:
                e: Entity = Hero(eb)
                self.heroes[e.id] = e
            elif eb.etype == EntityType.OPP_HERO:
                e: Entity = OppHero(eb)
                self.opps[e.id] = e
            else:
                raise ValueError()
        
        debug(str(list(self.monsters.keys())))
        return

    def prune_monsters(self):
        to_remove: List[int] = []
        for id, m in self.monsters.items():
            if (not m.on_map()) or (m.expect_seen()):
                to_remove.append(id)
        
        [self.monsters.pop(id) for id in to_remove]

    def all_entities(self) -> Iterator['Entity']:
        return chain(self.heroes.values(), self.opps.values(), self.monsters.values())
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
    def __init__(self, eb: EntityBase):
        self.id = eb.id
        self.etype = eb.etype
        self.loc = eb.loc
        self.shield_life = eb.shield_life
        self.is_controlled = eb.is_controlled
        self.health = eb.health
        self.vel = eb.vel
        self.near_base = eb.near_base
        self.threat_for = eb.threat_for
        self.is_winded = False

    def advance(self):
        raise NotImplementedError()

    def dist(self, other: Union[Point,'Entity']):
        if type(other) is Point:
            p: Point = other
        elif hasattr(other, 'loc'):
            p: Point = other.loc # type: ignore
        else:
            raise ValueError()
        return self.loc.dist(p)

class Monster(Entity):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.atk_neighbors = []
        self.wind_neighbors = []

    def counterpart(self) -> 'Monster':
        counter_eb: EntityBase = EntityBase(
            id = (self.id-1 if self.id%2 else self.id+1),
            etype = self.etype,
            loc = Point(MAP_X-self.loc.x, MAP_Y-self.loc.y),
            shield_life = 0,
            is_controlled = False,
            health = self.health,
            vel = self.vel.scale(-1),
            near_base = False,
            threat_for = ThreatFor.NEITHER
        )
        return Monster(counter_eb)

    def find_atk_neighbors(self, monsters: List['Monster']):
        pass

    def attackers_needed(self):
        time: float = self.dist(BASE)/MONSTER_SPEED
        n_attacks: float = self.health / ATTACK_DAMAGE
        return math.ceil(n_attacks/time)

    def advance(self):
        self.loc = self.loc.add_vector(self.vel)
        self.shield_life = max(self.shield_life-1, 0)

    def on_map(self) -> bool:
        return ((0 < self.loc.x < MAP_X) and (0 < self.loc.y < MAP_Y))

    def expect_seen(self) -> bool:
        if self.loc.dist(BASE) < BASE_VISION:
            return True
        for h in ENTITIES.heroes.values():
            if self.loc.dist(h.loc) < HERO_VISION:
                return True
        else:
            return False

    def is_handled(self) -> bool:
        raise NotImplementedError()
class Hero(Entity):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target = None
        self.home = HERO1_HOME

    def _wind(self, point:Point):
        self.command = f"{CommandType.WIND.value} {int(point.x)} {int(point.y)}"
        for e in ENTITIES.all_entities():
            if self.dist(e) < WIND_RANGE:
                e.is_winded = True

    def _move(self, point:Point):
        self.command = f"{CommandType.MOVE.value} {int(point.x)} {int(point.y)}"

    def _shield(self, id:int):
        self.command = f"{CommandType.SHIELD.value} {id}"

    def _control(self, id:int, point:Point):
        self.command = f"{CommandType.CONTROL.value} {id} {int(point.x)} {int(point.y)}"

    def execute(self):
        raise NotImplementedError()

    def set_target(self, e: Entity):
        self.target = e

    def defensive_wind(self, t: Entity):
        v: Vector = BASE.vector_to(t.loc)
        self._wind(self.loc.add_vector(v))
        t.is_winded = True
        debug(f"wind {t.id}")

    def offensive_wind(self, t: Entity):
        v: Vector = t.loc.vector_to(OPP_BASE)
        self._wind(self.loc.add_vector(v))
        debug(f"wind {t.id}")

    def attack_monster(self, t: Entity):
        v: Vector = t.loc.vector_to(self.loc)
        if v.norm() > MELEE_RANGE+HERO_SPEED:
            p:Point = t.loc.add_vector(t.vel)
        else:
            centroid: Point = self.home
            #find successful attack spot closest to centroid
            v2: Vector = t.loc.vector_to(centroid)
            v2: Vector = v2.normalize().scale(MELEE_RANGE*0.99)
            debug(f"{self.id} atk {t.id} v2 = {v2} {v2.norm()}")
            p:Point = t.loc.add_vector(v2)

        self._move(p)

    def opp_neighbors(self) -> List['OppHero']:
        return [o for o in opps if o.dist(self.loc)<CONTROL_RANGE]

    def optimal_attack(self) -> Optional[Point]:
        SEARCH_RANGE = HERO_SPEED + MELEE_RANGE - 10 # for rounding errors
        neighbors: List[Monster] = [m for m in ENTITIES.monsters.values() if m.dist(self) < SEARCH_RANGE]
        if not neighbors:
            return None
        debug(f" hero {self.id} sees monsters {[m.id for m in neighbors]}")
        candidates: Dict[Point, List[Monster]] = {}
        for m in neighbors:
            p: Point = Point.centroid([self.loc, m.loc])
            candidates[p] = []
        for i,m1 in enumerate(neighbors):
            for m2 in neighbors[i+1:]:
                if m1.dist(m2) > SEARCH_RANGE:
                    continue
                else:
                    p1,p2 = circle_intersection(m1.loc, m2.loc, SEARCH_RANGE/2)
                    if self.dist(p1) < SEARCH_RANGE/2:
                        candidates[p1] = [m1,m2]
                    if self.dist(p2) < SEARCH_RANGE/2:
                        candidates[p2] = [m1, m2]

        for p in candidates.keys():
            for m in neighbors:
                if m.dist(p) < SEARCH_RANGE/2:
                    candidates[p].append(m)
        
        result: Point = max(candidates.keys(), key=lambda k:len(candidates[k]))
        debug(f"{result} attacks monsters {[m.id for m in candidates[result]]}")
        return result

        


class AtkHero(Hero):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.home = HERO3_HOME
        self.atk_mode: bool = ATK_MODE

    def execute(self):
        debug(f'atk={self.atk_mode}')
        if self.instakill():
            debug("gib")
        elif self.atk_mode == True:
            self.attack()
        else:
            self.gather()

        print(self.command)

    def instakill(self) -> bool:
        for m in ENTITIES.monsters.values():
            if (m.dist(OPP_BASE) < WIND_MOVE+400) and (self.dist(m) < WIND_RANGE) and m.shield_life==0:
                self.offensive_wind(m)
                return True
        else:
            return False

    def attack(self):
        self._move(self.home)
        if self.loc.dist(self.home) > 500:
            self._move(self.home)
            return
        for m in monsters:
            if m.shield_life>0:
                continue
            if m.threat_for == ThreatFor.OPP_BASE:
                if m.dist(OPP_BASE) < 12*MONSTER_SPEED+300:
                    if self.dist(m.loc) < SHIELD_RANGE:
                        self._shield(m.id)
                        return
            elif m.dist(OPP_BASE) < MONSTER_VISION+WIND_RANGE:
                if m.dist(self.loc) < WIND_RANGE:
                    self.offensive_wind(m)
                    return
        else:
            for o in opps:
                if o.shield_life==0 and self.dist(o.loc)<CONTROL_RANGE:
                    self._control(o.id, BASE)
                    return


    def gather(self):
        p: Optional[Point] = self.optimal_attack()
        if p and p.dist(BASE)>BASE_VISION:
            self._move(p)
        elif monsters:
            target: Monster = min(monsters, key=lambda m: m.loc.dist(HERO3_HOME))
            self.attack_monster(target)
        else:
            self._move(self.home)
        



class MidHero(Hero):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.home = HERO2_HOME

    def execute(self):
        self.defend()
        print(self.command)
    
    def defend(self):
        on: List[OppHero] = self.opp_neighbors()
        if self.shield_life==0 and MANA_ME > 40 and on:
            self._shield(self.id)
            return

        if self.shield_life>0 and MANA_ME > 40 and on:
            if on[0].shield_life == 0 and on[0].loc.dist(self.loc) < WIND_RANGE:
                t = on[0]
                debug(f"opploc={on[0].loc} dist={on[0].loc.dist(self.loc)}")
                v: Vector = t.loc.vector_to(JAIL)
                self.defensive_wind(t)
                return


        if not monsters:
            self._move(self.home)
            return

        monsters.sort(key=lambda m:m.dist(BASE))
        t: Entity = monsters[0]
        if t.dist(BASE) > BASE_VISION+HERO_VISION:
            self._move(self.home)
            return

        if t.etype == EntityType.MONSTER:
            if t.dist(BASE) < (MONSTER_VISION-WIND_MOVE) \
                and self.loc.dist(t.loc) < WIND_RANGE \
                and t.shield_life==0 \
                and MANA_ME>10 \
                and t.is_winded == False:
                    self.defensive_wind(t)
            # elif MANA_ME > 100 and self.loc.dist(t.loc) < WIND_RANGE and t.shield_life==0:
            #     self.offensive_wind(t)
            else:
                self.attack_monster(t)


class DefHero(Hero):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.home = HERO1_HOME

    def execute(self):
        self.defend()
        print(self.command)
    
    def defend(self):
        on: List[OppHero] = self.opp_neighbors()
        if self.shield_life==0 and MANA_ME > 40 and on:
            self._shield(self.id)
            return

        if not monsters:
            self._move(self.home)
            return

        monsters.sort(key=lambda m:m.dist(BASE))
        t: Entity = monsters[0]
        if t.dist(BASE) > BASE_VISION+HERO_VISION:
            self._move(self.home)
            return

        if t.dist(BASE) < (MONSTER_VISION-WIND_MOVE) \
            and self.loc.dist(t.loc) < WIND_RANGE \
            and t.shield_life==0 \
            and MANA_ME>10 \
            and t.is_winded == False:
                self.defensive_wind(t)
            # elif MANA_ME > 100 and self.loc.dist(t.loc) < WIND_RANGE and t.shield_life==0:
            #     self.offensive_wind(t)
        else:
            self.attack_monster(t)


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
_scale = (BASE_VISION) * MIRROR

HERO1_HOME = BASE.add_vector(_v1.scale(_scale*0.75))
HERO2_HOME = BASE.add_vector(_v2.scale(_scale*1.5))
HERO3_HOME = OPP_BASE.add_vector(_v1.scale(_scale*-0.75))
JAIL = BASE.add_vector(_v4)


HERO_HOMES = [HERO1_HOME, HERO2_HOME, HERO3_HOME]
ATK_MODE = False


ENTITIES = EntityTracker()

# game loop
while True:
    HEALTH_ME, MANA_ME = [int(j) for j in input().split()]
    HEALTH_OPP, MANA_OPP = [int(j) for j in input().split()]

    if MANA_ME > 150:
        ATK_MODE = True
    elif MANA_ME < 50:
        ATK_MODE = False

    entity_count = int(input())  # Amount of heros and monsters you can see
    ENTITIES.update(entity_count)

    opps: List[OppHero] = list(ENTITIES.opps.values())
    monsters: List[Monster] = list(ENTITIES.monsters.values())
    heroes: List[Hero] = list(ENTITIES.heroes.values())



    heroes.sort(key=lambda e:e.dist(BASE))
    h1, h2, h3 = DefHero(heroes[0]), MidHero(heroes[1]), AtkHero(heroes[2])
    heroes = [h1,h2,h3]

    heroes.sort(key=lambda e:e.id)
    for h in heroes:
        h.execute()
