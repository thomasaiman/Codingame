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

    def __truediv__(self, scale: Numeric) -> "Vector":
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

from dataclasses import dataclass
@dataclass
class Sweep:
    start: Numeric # radians 0 to 2pi
    d_angle: Numeric # radians ... ? 0 to 2pi? or uncapped?

    def __init__(self, start: Numeric, d_angle: Numeric):
        self.start = start%(2*math.pi)
        self.d_angle = d_angle

    def contains_angle(self, phi: Numeric) -> bool:
        if self.d_angle > (2*math.pi):
            return True

        phi %= math.pi*2
        right, left = self.start, (self.start+self.d_angle)%(2*math.pi)
        if self.d_angle < 0:
            right, left = left, right
        debug("sweep", "right", right, "left", left)
        if right < left: # doesn't cross 0/2pi line
            return (right <= phi <= left)
        else: # crosses 0/2pi line
            return not (left < phi < right)

def norm(v: Vector) -> float:
    return math.hypot(*v)

def normalize(v: Vector) -> Vector:
    n = norm(v)
    return Vector(v.x/ n, v.y/n)

def dot(v1: Vector, v2: Vector) -> Numeric:
    return v1.x*v2.x + v1.y*v2.y

def angle(v1: Vector) -> float:
    return math.atan2(v1.y, v1.x) # radians between -pi and +pi

def relative_angle(v1 : Vector, v2: Vector) -> float: # radians between -pi and +pi
    # from v1 to v2
    diff = angle(v2) - angle(v1)
    if diff > math.pi:
        return -1 * (2*math.pi - diff) # -2*math.pi + diff = diff - 2*math.pi
    elif diff <= -math.pi:
        return (2*math.pi + diff)
    else:
        return diff

def vec_from_angle(angle: Numeric) -> Vector:
    return Vector(math.cos(angle), math.sin(angle))

def dist(p1: Point, p2: Point) -> Numeric:
    return norm(p1-p2)

def point_to_line_distance(point: Point, line: LineSegment) -> Numeric:
    denominator: Numeric = dist(line.p1, line.p2)
    if denominator < 1e-8:
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


if __name__=="__main__":
    x1 = circle_tangents_from_point(
        Circle(Point(501,0), 500),
        Point(0,0)
    )
    print([math.degrees(angle(v)) for v in x1])
    print('*'*50)
    x = collision_avoidance_angles(
        Point(0,0),
        600,
        Point(501,0),
        Vector(-540, 0),
        500
    )
    print([math.degrees(x.start), math.degrees(x.start+x.d_angle)])
    print(x.contains_angle(0))
    print(x.contains_angle(math.pi))

    print('-'*50)
    x = collision_avoidance_angles(
        Point(7272,7708),
        600,
        Point(6477,7843),
        Vector(271, 467),
        500
    )
    print([math.degrees(_) for _ in x])
    print(x.contains_angle(math.radians(45)))
    print(x.contains_angle(math.pi))
    # print('-'*50)
    # x = collision_avoidance_angles(
    #     Point(500,0),
    #     600,
    #     Point(-1,0),
    #     Vector(540, 0),
    #     500
    # )
    # print([math.degrees(_) for _ in x])
    # print('-'*50)
    # x = collision_avoidance_angles(
    #     Point(0 + 456,0 + 123),
    #     600,
    #     Point(501/math.sqrt(2) + 456 ,501/math.sqrt(2) + 123),
    #     Vector(-540/math.sqrt(2), -540/math.sqrt(2)),
    #     500
    # )
    # print([math.degrees(_) for _ in x])

    ## seed=-4698798113703483000
