from geometry import *
import unittest

class VectorArithmeticTestCase(unittest.TestCase):
    def testAdd(self):  # test method names begin with 'test'
        v1:Vector = Vector(1,2)
        v2: Vector = Vector(3,4)
        self.assertEqual((v1+v2), Vector(4,6))
        self.assertEqual((v1+v2), (v2+v1))
    def testSubtract(self):
        v1:Vector = Vector(1,2)
        v2: Vector = Vector(3,4)
        self.assertEqual((v1-v2), Vector(-2,-2))
    def testMultiply(self):
        v1:Vector = Vector(1,2)
        scale = 1.23
        self.assertEqual((v1*scale), Vector(1.23,2.46))
    def testDivide(self):
        v1: Vector = Vector(1,2)
        scale = 3
        self.assertEqual(v1/scale, Vector(1/3,2/3))


class SweepTestCase(unittest.TestCase):
    def testInit(self):  # test method names begin with 'test'
        start = math.radians(360+90)
        d_angle = math.radians(400)
        x = Sweep(start, d_angle)
        self.assertEqual(x, Sweep(math.pi/2, d_angle))
        self.assertEqual(x, Sweep(-math.pi*3/2, d_angle))

    def testContainment(self):
        x = Sweep(math.pi*3/2, math.pi/2)
        self.assertTrue(x.contains_angle(math.pi*7/4))
        self.assertTrue(x.contains_angle(math.pi*-1/4))
        self.assertFalse(x.contains_angle(math.pi))

    def testContainmentWraparound(self):
        x = Sweep(math.pi*3/2, math.pi) # from 3pi/2 around and up to pi/2
        self.assertTrue(x.contains_angle(math.pi*1/4))
        self.assertTrue(x.contains_angle(math.pi*-1/4))
        self.assertFalse(x.contains_angle(math.pi*3/4))
        self.assertFalse(x.contains_angle(math.pi*-3/4))
        self.assertFalse(x.contains_angle(math.pi*5/4))

    def testContainmentNegative(self):
        x = Sweep(math.pi*3/2, -math.pi/2)
        self.assertFalse(x.contains_angle(math.pi*7/4))
        self.assertTrue(x.contains_angle(math.pi))

    def testContainmentAbove2Pi(self):
        x = Sweep(math.pi*3/2, 5*math.pi)
        self.assertTrue(x.contains_angle(math.pi*7/4))
        self.assertTrue(x.contains_angle(1e6))
        self.assertTrue(x.contains_angle(-math.pi))

class AngleTestCase(unittest.TestCase):
    def testAngle(self): # currently bounded +- pi
        self.assertEqual(angle(Vector(1,1)), math.pi/4)
        self.assertEqual(angle(Vector(1,-1)), -math.pi/4)
        self.assertEqual(angle(Vector(-1,1)), math.pi*3/4)
        self.assertEqual(angle(Vector(-1,-1)), -math.pi*3/4)
        self.assertEqual(angle(Vector(-1, 0)), math.pi)
        self.assertEqual(angle(Vector(-1, -1e-19)), -math.pi)

    def testRelativeAngle(self): # currently bounded (-pi, pi]
        v1, v2, v3, v4 = Vector(1,1), Vector(-1,1), Vector(-1, -1), Vector(1,-1)
        self.assertEqual(relative_angle(v1, v2), math.pi/2)
        self.assertEqual(relative_angle(v2, v1), -math.pi/2)
        self.assertEqual(relative_angle(v1, v4), -math.pi/2)
        self.assertEqual(relative_angle(v1, v3), math.pi)
        self.assertEqual(relative_angle(v4, v3), -math.pi/2)
        self.assertEqual(relative_angle(v2, v3), math.pi/2)
        self.assertEqual(relative_angle(v3, v2), -math.pi/2)

class DistanceTestCase(unittest.TestCase):
    def testNorm(self):
        self.assertEqual(norm(Vector(0,0)), 0)
        self.assertEqual(norm(Vector(-3,4)), 5)

    def testDistZero(self):
        self.assertEqual(dist(Point(1,1), Point(1,1)), 0)
        self.assertEqual(dist(Point(1,1), Point(4,5)), 5)

    def testPointToLineDistance(self):
        p0 = Point(0,0)
        line0 = LineSegment(p0, p0)
        self.assertEqual(point_to_line_distance(p0, line0), 0)

        p1 = Point(0,0)
        line1 = LineSegment(Point(1,0), Point(0,1))
        self.assertAlmostEqual(
            point_to_line_distance(p1, line1),
            norm(Vector(0.5, 0.5))
        )

        p2 = Point(1,1)
        line2 = LineSegment(Point(0,0), Point(1,0))
        self.assertEqual(
            point_to_line_distance(p2, line2),
            1
        )


class CircleTangentTestCase(unittest.TestCase):
    def testTangent(self):
        c = Circle(Point(0,0), norm(Vector(1,1))) # intersects Point(1,1) with slope -1
        # tangent line at Point(1,1) should intersect Point(2,0)
        p0 = Point(2,0)
        p1, p2 = circle_tangents_from_point(c, p0)
        p1, p2 = sorted((p1,p2))
        self.assertTrue(dist(p1, Point(1,-1)) < 1e-9) # funky to handle floating point errors
        self.assertTrue(dist(p2, Point(1,1)) < 1e-9)


class CollisionTestCase(unittest.TestCase):

    """ Copied from Codingame referee and translated to Python
    https://github.com/CodinGame/FallChallenge2023-SeabedSecurity/blob/main/src/main/java/com/codingame/game/Game.java#L1067
    """
    @staticmethod
    def collision(p1 : Point, v1: Vector, r1: Numeric, p2: Point, v2: Vector, r2: Numeric) -> bool:
        # Check instant collision
        if (dist(p1, p2) <= r1+r2):
            print(0.0); return True

        # Both units are motionless
        if (norm(v1) == 0 and norm(v2) == 0):
            print("Both motionless"); return False

        # Change referencial
        x, y = p1.x, p1.y
        ux, uy = p2.x, p2.y
        x2, y2 = x-ux, y-uy
        r = r1 + r2
        vx2, vy2 = v1.x - v2.x, v1.y - v2.y

        # Resolving: sqrt((x + t*vx)^2 + (y + t*vy)^2) = radius <=> t^2*(vx^2 + vy^2) + t*2*(x*vx + y*vy) + x^2 + y^2 - radius^2 = 0
        # at^2 + bt + c = 0;
        # a = vx^2 + vy^2
        # b = 2*(x*vx + y*vy)
        # c = x^2 + y^2 - radius^2

        a = vx2 * vx2 + vy2 * vy2;

        if (a <= 0.0) :
            print("a<=0"); return False

        b = 2.0 * (x2 * vx2 + y2 * vy2)
        c = x2 * x2 + y2 * y2 - r * r
        delta = b * b - 4.0 * a * c

        if (delta < 0.0):
            print("delta<0"); return False

        t = (-b - math.sqrt(delta)) / (2.0 * a)
        t2 = (-b + math.sqrt(delta)) / (2.0 * a)
        print("t=", t, "t2=", t2)

        if (0 < t <= 1):
            return True
        else:
            return False

    def testCollision(self):
        self.assertTrue(self.collision(
            p1=Point(0,0),
            v1=Vector(1,0),
            r1 = 1,
            p2=Point(1.1, 0),
            v2=Vector(0, 0),
            r2=0,
        ))

        self.assertFalse(self.collision(
            p1=Point(0,0),
            v1=Vector(1,0),
            r1 = 1,
            p2=Point(2.1, 0),
            v2=Vector(0, 0),
            r2=0
        ))

        self.assertTrue(self.collision(
            p1=Point(0,0),
            v1=Vector(1,0),
            r1 = 1,
            p2=Point(1.1, 0),
            v2=Vector(0, 1),
            r2=0
        ))

        self.assertTrue(self.collision(
            p1=Point(0,0),
            v1=Vector(1,0),
            r1 = 1,
            p2=Point(200, 0),
            v2=Vector(-500,0),
            r2=0
        ))

    def testAvoidanceAngles(self):
        pass # TODO


# class
if __name__ == '__main__':
    unittest.main()