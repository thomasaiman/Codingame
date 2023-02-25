import sys
import pprint
import copy
import pprint as pp
from collections import namedtuple, deque, Counter, defaultdict, abc
from functools import lru_cache
import logging
import heapq
import time
import contextlib
import gc
import itertools
from typing import Tuple, List, Dict
logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

WATER = '.'
LAND = 'x'
TRAIL = 't'
carddirs = dict(zip('NSEW', [(-1, 0), (1, 0), (0, 1), (0, -1)]))
vec2dir = dict(zip([(-1, 0), (1, 0), (0, 1), (0, -1)], 'NSEW'))


class FrozenDict(abc.Mapping):
    """Don't forget the docstrings!!"""
    __slots__ = ('_d', '_hash')

    def __init__(self, *args, **kwargs):
        self._d = dict(*args, **kwargs)
        self._hash = hash(frozenset(self._d.items()))

    def __str__(self):
        return str(self._d)

    def __repr__(self):
        return repr(self._d)

    def __iter__(self):
        return iter(self._d)

    def __len__(self):
        return len(self._d)

    def __getitem__(self, key):
        return self._d[key]

    def __hash__(self):
        return self._hash

import signal

class Timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message
    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.setitimer(signal.ITIMER_REAL, self.seconds)
    def __exit__(self, type, value, traceback):
        signal.alarm(0)



IJTuple = namedtuple('IJTuple', ['i', 'j'])
class Point(IJTuple):

    @lru_cache(maxsize=None)
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, *args, **kwargs)

    def add_dir(self, d):
        diff = carddirs[d]
        return Point(self.i + diff[0], self.j + diff[1])

    def subtract_dir(self, d):
        diff = carddirs[d]
        return Point(self.i - diff[0], self.j - diff[1])

    def add_dirs(self, ds):
        p = self
        for d in ds:
            p = p.add_dir(d)
        return p

    def game_str(self):
        return f'{self.j} {self.i}'

    def add_vec(self, i: int, j: int):
        return Point(self.i + i, self.j + j)

    @staticmethod
    def from_game_str(s: str):
        x, y = [int(_) for _ in s.split()]
        return Point(y, x)

    def dir_to(self, p):
        vec = (p.i - self.i, p.j - self.j)
        d = vec2dir[vec]
        return d

    def manhattan_dist(self, p):
        return abs(self.i-p.i) + abs(self.j-p.j)

    def euclidean_dist(self, p):
        return ((self.i - p.i)**2 + abs(self.j - p.j)**2)**(1/2)

class Moves(list):
    revmap = dict(zip('NSEW', 'SNWE'))

    def backtrace(self, p):
        yield p
        for d in reversed(self):
            rd = Moves.revmap[d]
            p = p.add_dir(rd)
            yield p


class State:
    __slots__ = ('visited', 'location', 'health', 'grid', 'mines')

    def __init__(self, location, health, grid, visited=None, mines=None):
        if visited == None:
            visited = set()
        visited.add(location)
        self.location = location
        self.visited = visited
        self.health = health
        self.grid: Grid = grid
        self.mines = mines or FrozenDict() # mine pt : [potential drop pts]

    def healthcheck(self, health: int):
        if self.health == health:
            return self
        else:
            return None

    def move(self, d) -> None:
        nxt = self.location.add_dir(d)
        if (not nxt in self.visited) and (self.grid.get(nxt) == WATER):
            self.visited.add(nxt)
            self.location = nxt
            return self
        else:
            return None

    def test_move(self, d):
        nxt = self.location.add_dir(d)
        if (not nxt in self.visited) and (self.grid.get(nxt) == WATER):
            return True
        else:
            return False

    def explosion(self, tgt: Point):
        pt = self.location
        if pt == tgt:
            self.health -= 2
        elif pt in self.grid.explosion_area(tgt):
            self.health -= 1
        return self

    def surface(self, sector: int):
        if self.location in self.grid.sector_points(sector):
            self.health -= 1
            self.visited = {self.location}
            return self
        else:
            return None

    def silence(self):
        new = []
        new.append(self)
        for d in 'NSEW':
            s1 = self
            for i in range(1, 5):
                ok = s1.test_move(d)
                if ok:
                    s1 = s1.copy()
                    s1.move(d)
                    new.append(s1)
                else:
                    break
        return new

    def sonar(self):
        raise NotImplementedError('dont use this')

    def mine(self):
        mines = dict(self.mines)
        tgts = self.grid.get_neighbors(self.location)
        for tgt in tgts:
            dct = {self.location: 1/len(tgts)}
            old = mines.get(tgt, None)
            if old:
                dct.update(old)
            mines[tgt] = FrozenDict(dct)
        self.mines = FrozenDict(mines)
        return self

    def torpedo(self, tgt:Point):
        if self.location in self.grid.torpedo_points(tgt):
            return self
        else:
            return None

    def trigger(self, tgt1):
        if tgt1 not in self.mines:
            return None
        mines = dict(self.mines)
        locs = dict(mines[tgt1]) #places opponent could have been at time of drop
        ptgt1 = 1 / len(locs)
        for location in locs:
            tgts = self.grid.get_neighbors(location)
            ptgt2 = (1 - ptgt1)/len(tgts)
            for tgt in tgts:
                locs2 = dict(mines.get(tgt, {}))
                if tgt != tgt1:
                    locs2[location] = ptgt2
                elif tgt == tgt1:
                    locs2[location] = 0

                if locs2[location] == 0:
                    locs2.pop(location)
                mines[tgt] = FrozenDict(locs2)
                if len(mines[tgt]) == 0:
                    mines.pop(tgt)

        self.mines = FrozenDict(mines)
        return self

    def copy(self):
        s = State(location=self.location, health=self.health, grid=self.grid, visited=self.visited.copy(),
                  mines=self.mines)
        return s

    def __eq__(self, other):
        if self.visited == other.visited and self.health == other.health and self.location == other.location:
            return True
        else:
            return False

    def __hash__(self):
        tup = (
               frozenset(self.visited),
               self.health,
               self.location,
               self.mines,
        )
        return hash(tup)


class Grid:
    def __init__(self, array, warm_cache = False):
        self.array = array
        self.height = len(self.array)
        self.width = len(self.array[0])
        if warm_cache:
            self.warm_cache()

    def __str__(self):
        sz = max(len(self.get(p)) for p in self.all_points())
        rows = [' '.join(val.ljust(sz) for val in row) for row in self.array]
        return '\n' + '\n'.join(rows)

    def get(self, p: Point):
        if 0<=p.i<self.height and 0<=p.j<self.width:
            return self.array[p.i][p.j]
        else:
            return LAND

    def set(self, p: Point, val):
        self.array[p.i][p.j] = val

    def draw(self, pts, chars):
        grid = copy.deepcopy(self)
        if len(chars) == 1:
            for p in pts:
                grid.set(p, chars[0])
        else:
            for pt, char in zip(pts, chars):
                grid.set(pt, char)
        return grid

    def dijkstra(self, a, b, weights=None):
        # if a==b return []
        if not weights:
            weights = {}
        q = [(0, a, None)]
        done = {}
        path = []
        while b not in done:
            if not q:
                return []
            cost, node, prev = heapq.heappop(q)
            if node in done:
                continue
            neighbors = self.get_neighbors(node)
            for n in neighbors:
                if n in done:
                    continue
                elif self.get(n) != WATER:
                    continue
                else:
                    nxt = (cost + weights.get(n, 0) + 0.1, n, node)
                    heapq.heappush(q, nxt)
            done[node] = (cost, prev)
        node = b
        while node != a:
            path.append(node)
            cost, prev = done[node]
            node = prev
        path.reverse()
        return path

    def BFS(self, a, depth, char=WATER) -> dict:
        q = [(0, a, None)]
        done = {}
        while q:
            cost, node, prev = heapq.heappop(q)
            if cost > depth:
                break
            if node in done:
                continue
            neighbors = self.get_neighbors(node)
            for n in neighbors:
                if n in done:
                    continue  # not necessary but saves some heap pushes
                elif self.get(n) != char:
                    continue
                else:
                    nxt = (cost + 1, n, node)
                    heapq.heappush(q, nxt)
            done[node] = (cost, prev)
        return done

    def nearest_water(self, pt: Point):
        wpts = self.water_points()
        if pt in wpts:
            return pt
        center = Point(7,7)
        dists = [(pt.euclidean_dist(wpt), pt.euclidean_dist(center), wpt) for wpt in wpts]
        closest = min(dists)[-1]
        return closest

    def check_path(self, p: Point, path: Moves):
        pts = list(path.backtrace(p))
        for pt in pts:
            if self.get(pt) != WATER:
                return False
        if len(set(pts)) < len(pts):
            return False
        else:
            return True

    @lru_cache(maxsize=2)
    def water_points(self):
        ps = set()
        for p in self.all_points():
            if self.get(p) == WATER:
                ps.add(p)
        return frozenset(ps)

    @lru_cache(maxsize=1)
    def land_points(self):
        pts = frozenset(p for p in self.all_points() if self.get(p) == LAND)
        return pts

    @lru_cache(maxsize=9)
    def sector_points(self, sector: int):
        sector = sector
        srow = (sector - 1) // 3
        scol = (sector - 1) % 3
        irange = range(srow * 5, (srow + 1) * 5)
        jrange = range(scol * 5, (scol + 1) * 5)
        ps = frozenset(Point(i, j) for i in irange for j in jrange)
        return ps

    def all_points(self):
        for i in range(self.height):
            for j in range(self.width):
                p = Point(i, j)
                yield p

    def lookdir(self, p, d):
        id, jd = carddirs[d]
        contents = self.array[p.i + id][p.j + jd]
        return contents

    def isthmus(self, p):
        edges = {d: self.lookdir(p, d) == '.' for d in 'NSEW'}
        if (edges['N'] == edges['S'] == False) or (edges['E'] == edges['W'] == False):
            return True
        else:
            return False

    @lru_cache(maxsize=255)
    def get_neighbors(self, p):
        pts = []
        for d in 'NSEW':
            pt = p.add_dir(d)
            if self.get(pt) == WATER:
                pts.append(pt)
        return frozenset(pts)

    @lru_cache(maxsize=500)
    def explosion_area(self, p):
        pts = list(self.get_neighbors(p))
        for dd in ['NE', 'NW', 'SE', 'SW']:
            p2 = p.add_dirs(dd)
            if self.get(p2) != LAND:
                pts.append(p2)
        pts.append(p)
        return frozenset(pts)

    @lru_cache(maxsize=500)
    def torpedo_points(self, tgt):
        return frozenset(self.BFS(tgt, depth=4).keys())

    def warm_cache(self):
        for pt in self.water_points():
            self.explosion_area(pt)
            self.torpedo_points(pt)
            self.silence_possibles(pt)
        for i in range(1,10):
            self.sector_points(i)
        self.land_points()

    def lakes(self, waters):
        waters = set(waters)
        lakes = []
        while waters:
            lake = set()
            todo = [waters.pop()]
            while todo:
                p = todo.pop()
                lake.add(p)
                for n in self.get_neighbors(p):
                    if n in waters:
                        waters.remove(n)
                        todo.append(n)
            lakes.append(lake)
        lakes.sort(key=lambda x: len(x))
        return lakes

    def firstpos(self):
        lakes = self.lakes(self.water_points())
        big_lake = lakes[-1]
        pt = Point(7,7)
        g = self.draw(self.land_points(), WATER)
        depths = g.BFS(pt, 15)
        depths = sorted( depths.items(), key= lambda x: x[1])
        for pt, _ in depths:
            if pt in big_lake:
                return pt

    @lru_cache(maxsize=500)
    def silence_possibles(self, pt: Point):
        s = State(location=pt, health=6, grid=self)
        slist = s.silence()
        pts = frozenset(s.location for s in slist)
        return pts

    def silence_then_move(self, pt: Point):
        pts = set()
        spts = self.silence_possibles(pt)
        pts.update(spts)
        for pt2 in spts:
            pts.update(self.get_neighbors(pt2))
        pts.remove(pt)
        return frozenset(pts)

    def move_then_silence(self, pt: Point):
        pts = set()
        for n in self.get_neighbors(pt):
            pts.update(self.silence_possibles(n))
        pts.difference_update({pt})
        return frozenset(pts)

    def waypoints(self):
        #clockwise sector medians
        tgts = [Point(2,2), Point(2,7), Point(2,12),
                Point(7,12),Point(12,12), Point(12,7),
                Point(12,2), Point(7,2), Point(7,7)]
        pts = [self.nearest_water(tgt) for tgt in tgts]
        return pts


class TrailGrid(Grid):
    def __init__(self, grid:Grid):
        self.array = copy.deepcopy(grid.array)
        self.height = len(self.array)
        self.width = len(self.array[0])

    def silence_possibles(self, pt: Point):
        s = State(location=pt, health=6, grid=self)
        slist = s.silence()
        pts = frozenset(s.location for s in slist)
        return pts

    def get_neighbors(self, p):
        pts = []
        for d in 'NSEW':
            pt = p.add_dir(d)
            if self.get(pt) == WATER:
                pts.append(pt)
        return frozenset(pts)


class BoatTracker:
    def __init__(self, grid: Grid):
        self.grid: Grid = grid
        # self.possibles = self.grid.water_points()
        self.moves = Moves()
        self.sonar_pending: int = None
        self.action_queue: list = []
        self.silence_used: bool = False
        self.health = 6
        self.mines = []
        self.states = defaultdict(int)
        self.new_states = defaultdict(int)
        self.limbo= None
        self.t_limit = time.process_time() + 1
        for p in self.grid.water_points():
            s = State(location=p, health=6, grid=self.grid)
            self.states[s] += 1
        self.cooldowns = {'TORPEDO': 3,
                          'SILENCE': 6,
                          'MINE': 3,
                          'SONAR': 4}
        self.update_location_map()
        self.update_minemap()
        self.explosions_caused = []

    def __str__(self):
        c = self.possibles
        pts = c.keys()
        chars = [str(round(c[k] * 100)) for k in pts]
        g = self.grid.draw(pts=pts, chars=chars)

        c = self.minemap
        pts = c.keys()
        chars = [str(round(c[k] * 100)) for k in pts]
        g2 = self.grid.draw(pts=pts, chars=chars)
        return '\n'.join([str(g), '-'*30, str(g2)])

    def update_location_map(self):
        t1 = time.process_time()
        if t1>self.t_limit:
            return
        try:
            with Timeout(seconds=self.t_limit-t1):
                dct = {}
                total_weight = sum(self.states.values())
                for s, w in self.states.items():
                    dct.setdefault(s.location, 0)
                    dct[s.location] += (w / total_weight)
        except TimeoutError as e:
            logging.warning(e)
            return False
        self.possibles = dct
        self.total_weight=total_weight

        i_mean, j_mean = 0, 0
        for pt, w in self.possibles.items():
            i_mean += pt.i * w
            j_mean += pt.j * w

        i_var, j_var = 0,0
        for pt, w in self.possibles.items():
            i_var += (pt.i - i_mean)**2 * w
            j_var += (pt.i - j_mean)**2 * w
        i_std, j_std = i_var**(1/2), j_var**(1/2)

        logging.info(f'weighted location {i_mean} {j_mean}, with std {i_std} {j_std}')
        self._location_mean = Point(round(i_mean), round(j_mean))
        self._location_std = (i_std, j_std)

        return True

    def update_minemap(self):
        t1 = time.process_time()
        if len(self.states) > 300:
            return False
        if t1>self.t_limit:
            return False
        try:
            with Timeout(seconds=self.t_limit-t1):
                c = {}
                for s,w in self.states.items():
                    state_wt = w / self.total_weight
                    for tgt, locs in s.mines.items():
                        mine_wt = sum(locs.values())
                        Pmine = state_wt*mine_wt
                        tea = self.grid.explosion_area(tgt)
                        for pt in tea:
                            c.setdefault(pt, 0)
                            c[pt] += Pmine
                        c[tgt] += Pmine #double count center for double damage
        except TimeoutError as e:
            logging.warning(e)
            return False
        self.minemap = c
        return True

    def reduce_states(self):
        dct = {}
        for s, w in self.states.items():
            k = (s.location, s.health)
            v = [s.visited, 0, dict()]
            dct.setdefault(k, v)
            dct[k][0].intersection_update(v[0])
            dct[k][1] += w
            for tgt,locs in s.mines.items():
                dct[k][2].setdefault(tgt,dict())
                dct[k][2][tgt].update(locs)

        self.states = {}
        for k,v in dct.items():
            s = State(location=k[0], health=k[1], grid=self.grid, visited=v[0])
            mines = v[2]
            mines = {tgt:FrozenDict(locs) for tgt,locs in mines.items()}
            s.mines = FrozenDict(mines)
            self.states[s] = v[1]
        logging.info(f'Consolidated to {len(self.states)}')

    @property
    def location_mean(self):
        return self._location_mean

    @property
    def location_std(self):
        return self._location_std

    def hit_probability(self, tgt: Point):
        tea = self.grid.explosion_area(tgt)
        c = self.possibles
        dmg = 0
        for pt in tea:
            dmg += c.get(pt, 0)
        dmg += c.get(pt, 0)
        P = dmg
        return P

    def need_sonar(self):
        sdict = {}
        for sec in range(1, 10):
            sector_pts = self.grid.sector_points(sec)
            sdict[sec] = sum(self.possibles.get(pt, 0) for pt in sector_pts)

        slist = sorted([(pts, sector) for sector, pts in sdict.items()])
        P, sec = slist[-1]
        if 0.6>P>0.4:
            return sec
        else:
            return None

    def get_explosions(self):
        x = self.explosions_caused
        self.explosions_caused = []
        return x

    def explosion(self, tgt: Point):
        pass

    def mine(self):
        self.cooldowns['MINE'] = 3
        # for s in self.states:
        #     s.mine()

    def sonar_parse(self, result: str):
        if result != 'NA':
            pts = self.grid.sector_points(self.sonar_pending)
            new_states = defaultdict(int)
            if result == 'Y':
                for s, w in self.states.items():
                    if s.location in pts:
                        new_states[s] += w
            elif result == 'N':
                for s, w in self.states.items():
                    if s.location not in pts:
                        new_states[s] += w

            self.states = new_states
            self.sonar_pending = None

    def healthcheck(self, new_health: int):
        self.health = new_health

    def move(self, d):
        for ability, cd in self.cooldowns.items():
            if cd > 0:
                self.cooldowns[ability] -= 1
                break
        return True

    def sonar(self):
        self.cooldowns['SONAR'] = 4

    def surface(self, sector: int):
        pass

    def silence(self):
        self.silence_used = True
        self.cooldowns['SILENCE'] = 6

    def torpedo(self, tgt: Point):
        self.cooldowns['TORPEDO'] = 3

    def trigger(self, tgt: Point):
        pass

    def parse_orders(self, s: str):
        orders = s.split('|')
        for o in orders:
            o = o.strip()
            o = o.split(maxsplit=1)
            atype = o[0]

            if atype == 'EXPLOSION':
                tgt = Point.from_game_str(o[1])
                action = [atype, tgt]
                self.action_queue.append(action)
            if atype == 'SURFACE':
                sector = int(o[1])
                action = [atype, sector]
                self.action_queue.append(action)
            if atype == 'MOVE':
                d = o[1]
                action = [atype, d]
                self.action_queue.append(action)
            if atype == 'SILENCE':
                action = [atype]
                self.action_queue.append(action)
            if atype == 'MINE':
                action = [atype]
                self.action_queue.append(action)
            if atype == 'SONAR':
                action = [atype]
                self.action_queue.append(action)
            if atype == 'TORPEDO':
                tgt = Point.from_game_str(o[1])
                action = [atype, tgt]
                self.action_queue.append(action)
                self.parse_orders(f'EXPLOSION {tgt.game_str()}')
                self.explosions_caused.append(f'EXPLOSION {tgt.game_str()}')
            if atype == 'TRIGGER':
                tgt = Point.from_game_str(o[1])
                action = [atype, tgt]
                self.action_queue.append(action)
                self.parse_orders(f'EXPLOSION {tgt.game_str()}')
                self.explosions_caused.append(f'EXPLOSION {tgt.game_str()}')
            if atype == 'HEALTHCHECK':
                health = int(o[1])
                action = [atype, health]
                self.action_queue.append(action)

    def process_batch(self):
        action = self.action_queue[0]
        atype = action[0]
        args = action[1:] if len(action)>1 else []
        methodname = atype.lower()
        if atype != 'SONAR':
            counter = 0
            while self.states:
                counter += 1
                if counter % 100 == 0:
                    return False
                    # if time.process_time() > self.t_limit:
                    #     return False
                s, w = self.states.popitem()
                s1 = s.__getattribute__(methodname)(*args)
                if type(s1) == State:
                    self.new_states[s1] += w
                elif type(s1) == list:
                    for s2 in s1:
                        self.new_states[s2] += w
                elif type(s1) == type(None):
                    pass
                else:
                    raise ValueError(s1)



            self.states = self.new_states
            self.new_states = defaultdict(int)
        self.action_queue.pop(0)
        self.__getattribute__(methodname)(*args)
        logging.info(f'finished {action}. states = {len(self.states)}.')
        return True

    def process_queue(self, new_health) -> list:
        rlist = reversed(self.action_queue)
        for action in rlist:
            if action[0] == 'HEALTHCHECK':
                break
            elif action[0] == 'EXPLOSION':
                self.parse_orders(f'HEALTHCHECK {new_health}')
                break
        t0 = time.process_time()
        self.t_limit = t0 + 0.037
        while self.action_queue:
            if self.action_queue[0][0]=='SILENCE' and len(self.states)>1000:
                self.reduce_states()
            t1 = time.process_time()
            if t1 > self.t_limit:
                logging.info(f'leaving {self.action_queue[0]}. remaining = {len(self.states)}, done={len(self.new_states)}.')
                return self.get_explosions()
            self.process_batch()
            # try:
            #     with Timeout(seconds=0.0125):
            #         gc.collect()
            # except TimeoutError as e:
            #     logging.info('GC Timeout')


        if len(self.states) < 1:
            raise ValueError(f'lost track of submarine.')
        self.update_location_map()
        t1 = time.process_time()
        logging.info(f'possibles updated, states = {len(self.states)}, time={t1-t0}')
        self.update_minemap()
        t1 = time.process_time()
        logging.info(f'mines updated, states = {len(self.states)}, time={t1-t0}')
        return None


class MyBoat:  # my pathfinding, decision making
    def __init__(self, grid: Grid):
        self.grid: Grid = grid
        self.open_water = set(grid.water_points())
        self.cooldowns = {'TORPEDO': 3,
                          'SILENCE': 6,
                          'MINE': 3,
                          'SONAR': 4}
        self.mines = set()
        self.health = 6

        self.waypoints = [grid.nearest_water(Point(7,7))] #grid.waypoints()
        self.p: Point = self.waypoints[0]
        self.trail_grid = TrailGrid(self.grid.draw([self.p], TRAIL))
        self.open_water.remove(self.p)
        self.d = 'E'

        self.state = State(location=self.p, health=6, grid=self.grid)
        self.tracker = BoatTracker(grid)

    def __str__(self):
        mine_grid = self.grid.draw(self.mines, 'M')
        return '\n'.join([str(self.trail_grid), '-'*30, str(mine_grid)])

    def update_state(self, cooldowns, health):
        self.cooldowns.update(cooldowns)
        self.health = health
        self.state.health = health

    def lakes(self):
        return self.grid.lakes(self.open_water)

    def left(self):
        dirs = 'NESW'
        return dirs[dirs.index(self.d) - 1]

    def right(self):
        dirs = 'NESW'
        return dirs[(dirs.index(self.d) + 1) % 4]

    def straight(self):
        return self.d

    def back(self):
        dirs = 'NESW'
        return dirs[(dirs.index(self.d) + 2) % 4]

    def reachable_points(self):
        pass

    def mine(self, d):
        self.mines.add(self.state.location.add_dir(d))
        self.cooldowns['MINE']=3
        self.tracker.parse_orders(f'MINE')
        self.tracker.process_queue(new_health=self.state.health)

    def move(self, d, powerup = None):
        self.d = d
        self.p = self.p.add_dir(d)
        self.open_water.remove(self.p)
        self.trail_grid.set(self.p, TRAIL)
        self.state.move(d)
        self.tracker.parse_orders(f'MOVE {d}')
        if powerup:
            self.cooldowns[powerup] -= 1

    def surface(self):
        self.open_water = set(self.grid.water_points())
        self.open_water.remove(self.p)
        self.trail_grid=TrailGrid(self.grid.draw([self.p], TRAIL))
        self.moves = Moves()
        self.health -= 1
        sector = next(sec for sec in range(10) if self.p in self.grid.sector_points(sec))
        self.state.surface(sector=sector)

        self.tracker.parse_orders(f'SURFACE {sector}')

    def torpedo(self, tgt):
        assert self.state.torpedo(tgt)
        self.state.explosion(tgt)
        self.tracker.parse_orders(f'TORPEDO {tgt.game_str()}')
        self.cooldowns['TORPEDO']=3

    def trigger(self, tgt):
        self.mines.remove(tgt)
        self.state.explosion(tgt)
        self.tracker.parse_orders(f'TRIGGER {tgt.game_str()}')

    def silence(self, d, n):
        logging.debug(f'SILENCING {d} {n}')
        for i in range(n):
            self.move(d)

        self.tracker.parse_orders('SILENCE')
        self.cooldowns['SILENCE']=6

    def sonar(self):
        self.cooldowns['SONAR']=4


class ActionChooser:
    def __init__(self, me: MyBoat = None, me_est: BoatTracker = None, opponent_est: BoatTracker = None):
        # self.me = me
        # self.me_est = me_est
        # self.opponent_est = opponent_est
        pass

    def nav_to(self, start: Point, end: Point):
        vec = (end.i - start.i, end.j - start.j)
        if abs(vec[0]) > abs(vec[1]):
            big, small = (vec[0], 0), (0, vec[1])
        else:
            big, small = (0, vec[1]), (vec[0], 0)

        silence_sz = max(map(abs, big))
        silence_d = vec2dir[tuple(x / silence_sz for x in big)] if silence_sz>0 else 'N'
        if small == (0, 0):
            move_d = silence_d
            silence_sz -= 1
        else:
            move_d = vec2dir[small]

        return (move_d, silence_d, silence_sz)

    def kill_check(self, me: MyBoat, opponent_est: BoatTracker):
        msg = []
        #opp health, opp possibles, my health, torpedo cooldown, silence cooldown
        if me.cooldowns['TORPEDO'] == 0 and me.cooldowns['SILENCE'] == 1:
            powerup = 'SILENCE'
        elif me.cooldowns['TORPEDO'] <= 1:
            powerup = 'TORPEDO'
        else:
            return None

        opp_health = opponent_est.health
        if opp_health > 4:
            return None

        opp_possibles = opponent_est.possibles.keys()
        if len(opp_possibles) == 1 and me.mines.issuperset(opp_possibles):
            mine = next(iter(opp_possibles))
            opp_health -= 2
            msg.append(f'TRIGGER {mine.game_str()}')
        else:
            for mine in me.mines:
                if me.grid.explosion_area(mine).issuperset(opp_possibles):
                    opp_health -= 1
                    msg.append(f'TRIGGER {mine.game_str()}')
                    break

        logging.info(f'{opp_possibles}')
        if opp_health == 2 and len(opp_possibles) == 1:
            tgts = set(opp_possibles)
        elif opp_health == 1:
            tgts = set(me.grid.water_points())
            for p in opp_possibles:
                tgts.intersection_update(me.grid.explosion_area(p))
                if not tgts:
                    return None
        else:
            return None

        move_to = set()
        for tgt in tgts:
            move_to.update(me.grid.torpedo_points(tgt))

        if me.health > 1:
            g = me.grid
            msg.append('SURFACE')
        else:
            g = me.trail_grid

        if me.cooldowns['SILENCE']==0:
            reachable = g.silence_then_move(me.p).union(g.move_then_silence(me.p))
        elif me.cooldowns['SILENCE']==1 and me.cooldowns['TORPEDO']==0:
            reachable = g.move_then_silence(me.p)
        else:
            reachable = g.get_neighbors(me.p)

        reachable = reachable.intersection(move_to)
        if not reachable:
            return None

        logging.info('Kill check successful!')
        nxt_loc = next(iter(reachable))

        if nxt_loc in g.get_neighbors(me.p):
            move_d = me.p.dir_to(nxt_loc)
            msg.append(f'MOVE {move_d} {powerup}')
        elif nxt_loc in g.move_then_silence(me.p):
            (move_d, silence_d, silence_sz) = self.nav_to(start = me.p, end = nxt_loc)
            if not me.state.test_move(move_d):
                move_d, silence_d = silence_d, move_d
            msg.append(f'MOVE {move_d} {powerup}')
            msg.append(f'SILENCE {silence_d} {silence_sz}')
        elif nxt_loc in g.silence_then_move(me.p):
            (move_d, silence_d, silence_sz) = self.nav_to(start = me.p, end = nxt_loc)
            msg.append(f'SILENCE {silence_d} {silence_sz}')
            msg.append(f'MOVE {move_d} {powerup}')

        my_tgts = me.grid.torpedo_points(nxt_loc).intersection(tgts)
        my_tgt: Point = next(iter(my_tgts))
        msg.append(f'TORPEDO {my_tgt.game_str()}')

        print('|'.join(msg))
        return msg

    def snipe_check(self, me: MyBoat, opponent_est: BoatTracker) -> List[str]:
        actions = []
        # opp health, opp possibles, my health, torpedo cooldown, silence cooldown
        if me.cooldowns['TORPEDO'] == 0 and me.cooldowns['SILENCE'] == 1:
            powerup = 'SILENCE'
        elif me.cooldowns['TORPEDO'] <= 1:
            powerup = 'TORPEDO'
        else:
            return None

        opp_possibles = opponent_est.possibles.keys()

        logging.info(f'{opp_possibles}')
        if len(opp_possibles) == 1:
            tgts = set(opp_possibles)
        else:
            return None

        move_to = set()
        for tgt in tgts:
            move_to.update(me.grid.torpedo_points(tgt))
        move_to.remove(next(iter(tgts))) #dont move right on top of the opponent

        g = me.trail_grid

        if me.cooldowns['SILENCE'] == 0:
            reachable = g.silence_then_move(me.p).union(g.move_then_silence(me.p))
        elif me.cooldowns['SILENCE'] == 1 and me.cooldowns['TORPEDO'] == 0:
            reachable = g.move_then_silence(me.p)
        else:
            reachable = g.get_neighbors(me.p)

        reachable = reachable.intersection(move_to)
        if not reachable:
            return None

        logging.info('Snipe check successful!')
        nxt_loc = next(iter(reachable))
        logging.debug(f'nxt_loc {nxt_loc}')
        if nxt_loc in g.get_neighbors(me.p):
            move_d = me.p.dir_to(nxt_loc)
            actions.append(f'MOVE {move_d} {powerup}')
            me.move(d=move_d, powerup=powerup)
        elif nxt_loc in g.move_then_silence(me.p):
            (move_d, silence_d, silence_sz) = self.nav_to(start=me.p, end=nxt_loc)
            if not me.state.test_move(move_d):
                move_d, silence_d = silence_d, move_d
            actions.append(f'MOVE {move_d} {powerup}')
            me.move(d=move_d, powerup=powerup)
            actions.append(f'SILENCE {silence_d} {silence_sz}')
            me.silence(d=silence_d, n=silence_sz)
        elif nxt_loc in g.silence_then_move(me.p):
            (move_d, silence_d, silence_sz) = self.nav_to(start=me.p, end=nxt_loc)
            actions.append(f'SILENCE {silence_d} {silence_sz}')
            me.silence(d=silence_d, n=silence_sz)
            actions.append(f'MOVE {move_d} {powerup}')
            me.move(d=move_d, powerup=powerup)


        my_tgts = me.grid.torpedo_points(nxt_loc).intersection(tgts)
        my_tgt: Point = next(iter(my_tgts))
        actions.append(f'TORPEDO {my_tgt.game_str()}')
        me.torpedo(tgt)

        return actions

    def pick_dir(self, me: MyBoat, opponent: BoatTracker, player_est: BoatTracker) -> str:
        water = me.open_water.copy()
        opp_mines = [k for k,v in opponent.minemap.items() if v>=1]
        water.difference_update(opp_mines)
        lakes = me.grid.lakes(water)

        #stay away from land and use mine lakes
        for d in [me.left(), me.straight(), me.right(), me.back()]:
            p2 = me.p.add_dir(d)
            if player_est.hit_probability(p2) * opponent.minemap.get(p2,0) > 0.3:
                continue
            if p2 in lakes[-1] and len(me.grid.get_neighbors(p2))==4:
                return d

        #use mine lakes
        while lakes:
            for d in [me.left(), me.straight(), me.right(), me.back()]:
                p2 = me.p.add_dir(d)
                if player_est.hit_probability(p2)*opponent.minemap.get(p2,0) > 0.3:
                    continue
                if p2 in lakes[-1]:
                    return d
            lakes.pop()

        #use any lake
        lakes = me.lakes()
        while lakes:
            for d in [me.left(), me.straight(), me.right(), me.back()]:
                p2 = me.p.add_dir(d)
                if p2 in lakes[-1]:
                    return d
            lakes.pop()
        return None

    def pick_dir2(self, me: MyBoat, opponent: BoatTracker, player_est: BoatTracker) -> str:
        res=[]
        candidates = me.grid.get_neighbors(me.p).intersection(me.open_water)
        logging.debug(me.grid.get_neighbors(me.p))
        if not candidates:
            return None
        for pt in candidates:
            p= me.p
            d= me.p.dir_to(pt)
            me.p = pt
            tracker = copy.deepcopy(player_est)
            tracker.parse_orders(f'MOVE {d}')
            tracker.process_queue(new_health=me.health)
            n = len(tracker.states)
            dmg_received = self.estimate_dmg_to_me(me, opponent, tracker)
            dmg_given, _ = self.best_torpedo(me, opponent)
            score = (dmg_given*(4- me.cooldowns['TORPEDO'])/4) * n / (dmg_received + 1)
            res.append((score,d))
            me.p=p

        res.sort()
        logging.debug(res)
        return res[-1][1]

    def pick_mine_pt(self, me: MyBoat, opponent: BoatTracker) -> Point:
        lp = me.grid.land_points()
        for p in me.grid.get_neighbors(me.p):
            if len(me.grid.get_neighbors(p))<4:
                continue
            if me.mines.intersection(me.grid.explosion_area(p)):
                continue
            return p
        else:
            return None

    def estimate_dmg_to_me(self,me:MyBoat, opponent: BoatTracker, player_est: BoatTracker) -> float:
        if opponent.cooldowns['TORPEDO'] <= 1:
            my_probs = [(player_est.hit_probability(pt), pt) for pt in me.grid.explosion_area(me.p)]
            my_probs.sort()
            opp_shots = {}
            for dmg, tgt in my_probs:
                trange = me.grid.torpedo_points(tgt)
                trange = set(itertools.chain.from_iterable(me.grid.get_neighbors(pt) for pt in trange))
                opp_pts = trange.intersection(opponent.possibles.keys())
                for opp_pt in opp_pts:
                    if not opp_pt in opp_shots:
                        opp_shots[opp_pt] = dmg
            est_torpedo_dmg = sum(opponent.possibles[k]*v for k,v in opp_shots.items()) #between 0 and 2
        else:
            est_torpedo_dmg = 0

        est_mine_dmg = [player_est.hit_probability(p2) * opponent.minemap.get(p2,0) for p2 in me.grid.explosion_area(me.p)]
        est_mine_dmg = sum(est_mine_dmg)
        logging.debug(f'est_mine_dmg {est_mine_dmg} est_torpedo_dmg {est_torpedo_dmg}')
        return est_mine_dmg + est_torpedo_dmg

    def choose_silence(self,me:MyBoat, opponent: BoatTracker, player_est: BoatTracker) -> Tuple[str, int]:
        #save silence for battles, dont spam it
        est_dmg= self.estimate_dmg_to_me(me, opponent, player_est)
        if est_dmg > 1:
            d = self.pick_dir(me, opponent, player_est)
            if d:
                cap = hash(me.state)%5 #pseudorandom
                s = copy.deepcopy(me.state)
                count = 0
                while count < cap:
                    # d2 = self.pick_dir(me, opponent, player_est)
                    # if d2 == d:
                    #     me.move(d)
                    s = s.move(d)
                    if s:
                        count += 1
                    else:
                        break
                # return f'SILENCE {d} {count}'
                return d,count
        return None, None

    def best_torpedo(self, me:MyBoat, opponent: BoatTracker) -> Tuple[float, Point]:
        trange = list(me.grid.torpedo_points(me.p))
        trange.remove(me.p)
        sea = me.grid.explosion_area(me.p)

        expected_dmg = [(opponent.hit_probability(pt) + (-1 if pt in sea else 0),
                         pt)
                        for pt in trange]
        expected_dmg.sort()
        return expected_dmg[-1]

    def best_mine(self, me: MyBoat, opponent: BoatTracker) -> Tuple[float, Point]:
        trange = list(me.mines)
        if trange:
            sea = me.grid.explosion_area(me.p)
            expected_dmg = [(opponent.hit_probability(pt) + (-2 if pt in sea else 0),
                            pt)
                            for pt in trange]
            expected_dmg.sort()
            return expected_dmg[-1]
        else:
            return (0, None)

    def needs_charge(self, me:MyBoat):
        for ability, cd in me.cooldowns.items():
            if cd > 0:
                return ability
        return 'TORPEDO'

    def mine_strategy(self, me:MyBoat, opponent: BoatTracker, player_est: BoatTracker):
        actions = []

        if me.cooldowns['TORPEDO'] == 0:
            t_pt = self.best_torpedo(me, opponent)
            if t_pt:
                actions.append(f'TORPEDO {t_pt.game_str()}')
                opponent.parse_orders(f'EXPLOSION {t_pt.game_str()}')
                me.cooldowns['TORPEDO'] = 3

        if me.cooldowns['SILENCE'] == 0:
            msg = self.choose_silence(me, opponent, player_est)
            if msg:
                actions.append(msg)
                me.cooldowns['SILENCE'] = 6

        if me.cooldowns['MINE'] == 0:
            t_pt = self.pick_mine_pt(me, opponent)
            if t_pt:
                md = me.p.dir_to(t_pt)
                actions.append(f'MINE {md}')
                me.cooldowns['MINE'] = 3
                me.mines.add(t_pt)

        mine: Point = self.best_mine(me, opponent)
        if mine:
            actions.append(f'TRIGGER {mine.game_str()}')
            opponent.parse_orders(f'EXPLOSION {mine.game_str()}')
            me.mines.remove(mine)


        while me.waypoints:
            path = me.trail_grid.dijkstra(me.p, me.waypoints[0], weights=opponent.minemap)
            if path:
                break
            else:
                me.waypoints.pop(0)
        if path:
            d = me.p.dir_to(path[0])
            actions.append(f'MOVE {d} {self.needs_charge(me)}')
            me.move(d)
        else:
            actions += self.wait_strategy(me, opponent, player_est)
        return actions

    def wait_strategy(self, me:MyBoat, opponent: BoatTracker, player_est: BoatTracker):
        actions = []
        snipe_actions = self.snipe_check(me,opponent)
        if snipe_actions:
            actions += snipe_actions

        if me.cooldowns['TORPEDO'] == 0:
            P, t_pt = self.best_torpedo(me, opponent)
            if P>=1:
                actions.append(f'TORPEDO {t_pt.game_str()}')
                me.torpedo(t_pt)


        if me.cooldowns['SILENCE'] == 0:
            d, n = self.choose_silence(me, opponent, player_est)
            if d:
                me.silence(d,n)
                actions.append(f'SILENCE {d} {n}')


        if me.cooldowns['MINE'] == 0:
            t_pt = self.pick_mine_pt(me, opponent)
            if t_pt:
                md = me.p.dir_to(t_pt)
                actions.append(f'MINE {md}')
                me.mine(md)

        Dbest, tgt= self.best_mine(me, opponent)
        if Dbest > 0.75:
            actions.append(f'TRIGGER {tgt.game_str()}')
            me.trigger(tgt)

        if not any('MOVE' in a for a in actions):
            d = self.pick_dir(me, opponent, player_est)
            if d:
                powerup = self.needs_charge(me)
                actions.append(f'MOVE {d} {powerup}')
                me.move(d, powerup)

        if me.cooldowns['TORPEDO'] == 0:
            P, t_pt = self.best_torpedo(me, opponent)
            if P>=1:
                actions.append(f'TORPEDO {t_pt.game_str()}')
                me.torpedo(t_pt)

        return actions

    def act(self, me:MyBoat, opponent: BoatTracker, player_est: BoatTracker):
        actions = []
        self.kill_check(me, opponent)
        logging.debug('kill check done')

        sonar_sector = opponent.need_sonar()
        if sonar_sector and me.cooldowns['SONAR']==0:
            actions.append(f'SONAR {sonar_sector}')
            opponent.sonar_pending = sonar_sector
            me.sonar()

        if False:
            strategy = 'mines'
            actions += self.mine_strategy(me, opponent, player_est)
        elif False:
            strategy = 'chase'
            actions += self.chase_strategy(me, opponent, player_est)
        else:
            strategy = 'wait'
            actions += self.wait_strategy(me, opponent, player_est)


        if not actions:
            actions.append('SURFACE')
            me.surface()

        msg = ' | '.join(actions)
        return msg


class Game:
    def __init__(self, grid):
        self.grid = grid
        self.me = MyBoat(grid)
        self.opponent_est = BoatTracker(grid)
        self.me_est = self.me.tracker
        self.action_chooser = ActionChooser(self.me, self.me_est, self.opponent_est)
        print(self.me.p.game_str())

    def orders_transform(self, s: str, player: MyBoat):
        out = []
        orders = s.split('|')
        for o in orders:
            o = o.strip()
            atype = o.split()[0]
            if atype == 'MOVE':
                out.append(' '.join(o.split()[0:2]))
            if atype == 'TORPEDO':
                out.append(o)
            if atype == 'SURFACE':
                pt = player.p
                sector = next(i for i in range(1,10) if pt in self.grid.sector_points(i))
                out.append(o + ' ' + str(sector))
            if atype == 'SILENCE':
                out.append(atype)
            if atype == 'SONAR':
                out.append(o)
            if atype == 'MINE':
                out.append(atype)
            if atype == 'TRIGGER':
                out.append(o)
        return '|'.join(out)

    def get_input(self):
            s = input()
            if s:
                return s
            else:
                sys.exit('Game Over')

    def run(self):
        gc.disable()
        while True:
            t0 = time.process_time()
            x, y, my_life, opp_life, torpedo_cooldown, sonar_cooldown, silence_cooldown, mine_cooldown = [int(i) for i
                                                                                                          in
                                                                                                          self.get_input().split()]
            sonar_result = self.get_input()
            opponent_orders = self.get_input()
            t1 = time.process_time() - t0
            logging.info(f'Got input after {t1}')

            self.opponent_est.sonar_parse(sonar_result)
            self.opponent_est.parse_orders(opponent_orders)
            self.opponent_est.process_queue(new_health=opp_life)
            logging.info(str(self.opponent_est.action_queue))
            logging.debug(f'Opponent cooldowns: {self.opponent_est.cooldowns}')
            # logging.info(str(self.opponent_est))
            t1 = time.process_time() - t0
            logging.info(f'Opponent estimate after {t1}')

            opp_explosions_caused = self.opponent_est.get_explosions()
            for ex in opp_explosions_caused:
                self.me_est.parse_orders(ex)
            self.me_est.process_queue(new_health=my_life)

            my_cooldowns = dict(zip(['TORPEDO', 'SONAR', 'SILENCE', 'MINE'],
                                    [torpedo_cooldown, sonar_cooldown, silence_cooldown, mine_cooldown]))
            self.me.update_state(my_cooldowns, my_life)


            my_orders = self.action_chooser.act(self.me, self.opponent_est, self.me_est)
            # self.me.parse_orders(my_orders)
            # self.me.process_queue()
            t1 = time.process_time() - t0
            logging.info(f'Action choice after {t1}')

            my_explosions_caused = self.me_est.get_explosions()
            for ex in my_explosions_caused:
                self.opponent_est.parse_orders(ex)


            if t1 < 0.05:
                self.me_est.process_queue(new_health=self.me.state.health)
                t1 = time.process_time() - t0
                logging.info(f'Self estimate after {t1}')

            if t1 < 0.05:
                try:
                    with Timeout(seconds = 0.04):
                        gc.collect()
                except TimeoutError as e:
                    logging.info('GC Timeout')
                t1 = time.process_time() - t0
                logging.info(f'GC after {t1}')

            sys.stderr.flush()
            print(my_orders)
            # print(gc.get_stats())
            sys.stdout.flush()
            t1 = time.process_time() - t0
            logging.info(f'Printed after {t1}')


if __name__ == '__main__':
    t0 = time.process_time()
    # setup
    width, height, my_id = [int(i) for i in input().split()]
    logging.info(str((width, height, my_id)))
    grid = Grid([list(input()) for i in range(height)], warm_cache=True)

    game = Game(grid)
    t1 = time.process_time()
    logging.info(f'setup time {t1-t0}')
    game.run()


#TODO: eval movements for max confusion
