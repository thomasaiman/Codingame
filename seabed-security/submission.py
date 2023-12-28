from typing import List, NamedTuple, Dict, Union, Tuple, Optional, Any
import sys
import math
from itertools import chain

Numeric = Union[int, float]

def debug(*s: Any):
    print(*s, file=sys.stderr, flush=True)

# Define the data structures as namedtuples
class Vector(NamedTuple):
    x: Numeric
    y: Numeric

    def __add__(self, other: "Vector") -> "Vector":
        return Vector(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vector") -> "Vector":
        return Vector(self.x - other.x, self.y - other.y)

    def __mul__(self, scale: Numeric) -> "Vector":
        return Vector(self.x*scale, self.y*scale)

    def __div__(self, scale: Numeric) -> "Vector":
        return Vector(self.x/scale, self.y/scale)

# class Point(Vector):
#     pass
Point = Vector

class LineSegment(NamedTuple):
    p1: Point
    p2: Point

class Circle(NamedTuple):
    center: Point
    radius: Numeric

class Sweep(NamedTuple):
    start: Numeric 
    d_angle: Numeric

    @staticmethod
    def _normalize_angle(phi: Numeric) -> Numeric:
        if phi > math.pi:
            return phi - 2*math.pi
        elif phi < -math.pi:
            return phi + 2*math.pi
        else:
            return phi

    def contains_angle(self, phi: Numeric) -> bool:
        phi = Sweep._normalize_angle(phi)
        start = Sweep._normalize_angle(self.start)
        right, left = sorted((start, start+self.d_angle))
        debug("sweep", "right", right, "left", left)
        if right <= phi <= left:
            return True
        else:
            return False

def norm(v: Vector) -> float:
    return math.hypot(*v)

def normalize(v: Vector) -> Vector:
    n = norm(v)
    return Vector(v.x/ n, v.y/n)

def dot(v1: Vector, v2: Vector) -> Numeric:
    return v1.x*v2.x + v1.y*v2.y

def angle(v1: Vector) -> float:
    return math.atan2(v1.y, v1.x) # radians between +pi and -pi

def relative_angle(v1 : Vector, v2: Vector) -> float:
    # from v1 to v2
    diff = angle(v2) - angle(v1)
    if diff > math.pi:
        return -1 * (2*math.pi - diff) # -2*math.pi + diff = diff - 2*math.pi
    elif diff < -math.pi:
        return (2*math.pi + diff)
    else:
        return diff
    
def vec_from_angle(angle: Numeric) -> Vector:
    return Vector(math.cos(angle), math.sin(angle))

def dist(p1: Point, p2: Point) -> Numeric:
    return norm(p1-p2)

def point_to_line_distance(point: Point, line: LineSegment) -> Numeric:
    if dist(line.p1, line.p2) < 1e-2:
        return dist(line.p1, point)
    # https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
    x0, y0 = point.x, point.y
    x1, y1 = line.p1.x, line.p1.y
    x2, y2 = line.p2.x, line.p2.y

    numerator: Numeric = abs((x2-x1)*(y1-y0) - (x1-x0)*(y2-y1))
    denominator: Numeric = dist(line.p1, line.p2)
    return numerator/denominator

def circle_tangents_from_point(c: Circle, p0: Point) -> Tuple[Point, Point]:
    # find points as if circle were centered at (0,0)
    # https://en.wikipedia.org/wiki/Tangent_lines_to_circles
    d0 = dist(c.center, p0)
    if d0 < c.radius:
        raise ValueError(f"{p0} is inside {c}")
    p0 = p0 - c.center
    t1: Numeric = (c.radius**2)/(d0**2)
    t2: Numeric = (c.radius) / (d0**2) * math.sqrt(d0**2 - c.radius**2)
    p1 = Point(t1*p0.x + (t2*-1*p0.y), t1*p0.y + t2*p0.x)
    p2 = Point(t1*p0.x - (t2*-1*p0.y), t1*p0.y - t2*p0.x)
    # translate results to actual circle location
    p1 = p1 + c.center
    p2 = p2 + c.center
    return (p1, p2)

def collision_avoidance_angles(p_avoider: Point, s_avoider: Numeric, p_chaser: Point, v_chaser: Vector, radius: Numeric) -> Sweep:
    # change reference frame: static chaser, moving avoider, chaser is on (x, 0), avoider is at (0,0)
    p_chaser2 = Point(dist(p_avoider, p_chaser), 0)
    p_avoider2 = Point(0,0)
    debug(p_chaser2)
    pt1, pt2 = circle_tangents_from_point(Circle(p_chaser2, radius), Point(0,0))
    (vec_t1, vec_t2) = (p - p_avoider2 for p in (pt1, pt2))
    (angle_t1, angle_t2) = sorted(angle(v) for v in (vec_t1, vec_t2)) # rays that are tangent to that circle
    debug(math.degrees(angle_t1), math.degrees(angle_t2)) # should be +- the same
    # solve for v_avoider such that angle(v_avoider-v_chaser) = angle_t1 or angle_t2
    # solve SSA triangle: sin(A)/a = sin(B)/b = sin(C)/c ; b>a so only one solution
    side_a = norm(v_chaser)
    side_b = s_avoider
    angle_B = angle_t2 # if this isn't positive we screwed up / the left side of the chaser
    angle_A = math.asin(side_a * math.sin(angle_B)/side_b)  # sin(A)  = sin(B)/b * a
    angle_C = math.pi - angle_A - angle_B # relative to vector (v_chaser)
    debug(math.degrees(angle_C))
    debug(math.degrees(angle(v_chaser)))
    debug(math.degrees(angle(p_avoider-p_chaser)))
    avoider_angle_t2 = angle(v_chaser) - angle_C # smaller angle is on the left from avoider's view facing chaser
    avoider_angle_t1 = angle(v_chaser) + angle_C # bigger angle is on the right
    debug(math.degrees(avoider_angle_t1), math.degrees(avoider_angle_t2))
    return Sweep(start=avoider_angle_t1, d_angle=2*(math.pi-angle_C))


class FishDetail(NamedTuple):
    color: int
    type: int

class Fish(NamedTuple):
    fish_id: int
    pos: Vector
    speed: Vector
    detail: FishDetail

class RadarBlip(NamedTuple):
    fish_id: int
    dir: str

class Drone(NamedTuple):
    drone_id: int
    pos: Vector
    dead: bool
    battery: int
    scans: List[int]

fish_details: Dict[int, FishDetail] = {}

fish_count = int(input())
for _ in range(fish_count):
    fish_id, color, _type = map(int, input().split())
    fish_details[fish_id] = FishDetail(color, _type)

X_MAX, Y_MAX = 10000, 10000
DRONE_SPEED = 600
DRONE_SINK_SPEED = 300
DRONE_FLOAT_SPEED = 300
LIGHT_RADIUS = 800
LIGHT_POWER_RADIUS = 2000
SURFACE_SAVE_DEPTH = 500
FISH_SPEED = 200
FISH_BOUNCE_RADIUS = 600
FISH_FRIGHTENED_SPEED = 400
FISH_FRIGHTENED_RADIUS = 1400
TYPE0_Y_MIN, TYPE0_Y_MAX = 2500, 5000
TYPE1_Y_MIN, TYPE1_Y_MAX = 5000, 7500
TYPE2_Y_MIN, TYPE2_Y_MAX = 7500, 10000
MONSTER_Y_MIN, MONSTER_Y_MAX = 2500, 10000
MONSTER_PROVOKE_RADIUS = LIGHT_RADIUS
MONSTER_DETECT_RADIUS = LIGHT_RADIUS + 300
MONSTER_CHASE_SPEED = 540
MONSTER_ATTACK_RADIUS = 500

# game loop
while True:
    my_scans: List[int] = []
    foe_scans: List[int] = []
    drone_by_id: Dict[int, Drone] = {}
    my_drones: List[Drone] = []
    foe_drones: List[Drone] = []
    visible_fish: List[Fish] = []
    my_radar_blips: Dict[int, List[RadarBlip]] = {}

    my_score = int(input())
    foe_score = int(input())

    my_scan_count = int(input())
    for _ in range(my_scan_count):
        fish_id = int(input())
        my_scans.append(fish_id)

    foe_scan_count = int(input())
    for _ in range(foe_scan_count):
        fish_id = int(input())
        foe_scans.append(fish_id)

    my_drone_count = int(input())
    for _ in range(my_drone_count):
        drone_id, drone_x, drone_y, dead, battery = map(int, input().split())
        debug("drone input: " , drone_id, drone_x, drone_y, dead, battery)
        pos = Vector(drone_x, drone_y)
        drone = Drone(drone_id, pos, dead == 1, battery, [])
        drone_by_id[drone_id] = drone
        my_drones.append(drone)
        my_radar_blips[drone_id] = []

    foe_drone_count = int(input())
    for _ in range(foe_drone_count):
        drone_id, drone_x, drone_y, dead, battery = map(int, input().split())
        pos = Vector(drone_x, drone_y)
        drone = Drone(drone_id, pos, dead == 1, battery, [])
        drone_by_id[drone_id] = drone
        foe_drones.append(drone)

    drone_scan_count = int(input())
    for _ in range(drone_scan_count):
        drone_id, fish_id = map(int, input().split())
        drone_by_id[drone_id].scans.append(fish_id)

    visible_fish_count = int(input())
    for _ in range(visible_fish_count):
        fish_id, fish_x, fish_y, fish_vx, fish_vy = map(int, input().split())
        pos = Vector(fish_x, fish_y)
        speed = Vector(fish_vx, fish_vy)
        visible_fish.append(Fish(fish_id, pos, speed, fish_details[fish_id]))

    my_radar_blip_count = int(input())
    for _ in range(my_radar_blip_count):
        drone_id, fish_id, dir = input().split()
        drone_id = int(drone_id)
        fish_id = int(fish_id)
        my_radar_blips[drone_id].append(RadarBlip(fish_id, dir))

    found_ids = set(chain(my_scans, *(drone.scans for drone in my_drones)))
    remaining_fish = set(
        blip.fish_id
        for blipsList in my_radar_blips.values()
        for blip in blipsList
        if fish_details[blip.fish_id].type != -1
    )

    targeted_ids = set()
    visible_monsters = [fish for fish in visible_fish if fish.detail.type == -1]
    debug("visible_monsters:", visible_monsters)
    for drone in my_drones:
        debug("drone is dead? " , drone.dead)
        if drone.dead:
            print("WAIT 0")
            continue
        x = drone.pos.x
        y = drone.pos.y
        # TODO: Implement logic on where to move here
        target_id: Optional[int] = next(
            (fish_id for fish_id in fish_details.keys()
             if fish_id not in found_ids
             and fish_id in remaining_fish
             and fish_id not in targeted_ids
            ), None)
        if target_id is None:
            target_x = drone.pos.x
            target_y = 0
        else:
            targeted_ids.add(target_id)
            dbs: List[RadarBlip] = my_radar_blips[drone.drone_id]
            blip: RadarBlip = next(b for b in dbs if b.fish_id == target_id)
            target_x = x + (1000 if blip.dir[1] == "R" else -1000)
            target_y = y + (1000 if blip.dir[0] == "B" else -1000)

        # check for monsters and avoid them
        p_target = Point(target_x, target_y)
        l_target = LineSegment(drone.pos, p_target)
        for monster in visible_monsters:
            debug("drone:",drone.drone_id,"dist to monster:", monster.fish_id, "=", dist(drone.pos, monster.pos))
            m_next: Point = monster.pos + normalize(drone.pos - monster.pos)*MONSTER_CHASE_SPEED

            if (dist(m_next, drone.pos)) < MONSTER_ATTACK_RADIUS:
                debug("Escaping", monster)
                debug("expected m_next: " , m_next)
                debug("dist to m_next:", dist(m_next, drone.pos))
                sweep: Sweep = collision_avoidance_angles(
                    p_avoider=drone.pos,
                    s_avoider=DRONE_SPEED,
                    p_chaser=monster.pos,
                    v_chaser=normalize(drone.pos - monster.pos)*MONSTER_CHASE_SPEED,
                    radius=MONSTER_ATTACK_RADIUS+10
                )
                target_angle = angle(p_target - drone.pos)
                if not sweep.contains_angle(target_angle):
                    debug("fleeing and continuing to:", p_target)
                    continue
                else:
                    move_options: List[Point] = [drone.pos + vec_from_angle(angle)*DRONE_SPEED for angle in [sweep.start, sweep.start+sweep.d_angle]]
                    best_move : Point = min(move_options, key = lambda p: dist(p, p_target))
                    p_target = best_move
                    debug("fleeing to:", p_target)

                continue

            elif point_to_line_distance(monster.pos, l_target) < (MONSTER_PROVOKE_RADIUS+10):
                debug("Avoiding" , monster)
                if dist(monster.pos, p_target) < (MONSTER_PROVOKE_RADIUS+10):
                    p_target = (normalize(p_target - drone.pos) * 3000) + drone.pos
                options: Tuple[Point, Point] = circle_tangents_from_point(Circle(monster.pos, MONSTER_PROVOKE_RADIUS+10), drone.pos)
                vecs: List[Vector] = [p - drone.pos for p in options]
                ang1,ang2 = sorted(angle(vec) for vec in vecs)
                if ang2 - ang1 < math.pi:
                    sweep = Sweep(start=ang1, d_angle=ang2-ang1)
                else:
                    start = ang2 - 2*math.pi
                    sweep = Sweep(start=start, d_angle=ang1-start)

                target_angle = angle(p_target - drone.pos)
                if not sweep.contains_angle(target_angle):
                    debug("avoiding and continuing to:", p_target)
                    continue
                else:
                    move_options: List[Point] = [drone.pos + vec_from_angle(angle)*DRONE_SPEED for angle in [sweep.start, sweep.start+sweep.d_angle]]
                    best_move : Point = min(move_options, key = lambda p: dist(p, p_target))
                    p_target = best_move
                    debug("avoiding via tangent to:", p_target)

                continue


        target_x, target_y = int(p_target.x) , int(p_target.y)
        light = 1
        debug("Targeting", target_id, blip)
        print(f"MOVE {target_x} {target_y} {light}")