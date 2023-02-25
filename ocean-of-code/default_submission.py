import sys
import math

# Auto-generated code below aims at helping you parse
# the standard input according to the problem statement.

width, height, my_id = [int(i) for i in input().split()]
for i in range(height):
    line = input()

# Write an action using print
# To debug: print("Debug messages...", file=sys.stderr, flush=True)

print("7 7")

# game loop
while True:
    x, y, my_life, opp_life, torpedo_cooldown, sonar_cooldown, silence_cooldown, mine_cooldown = [int(i) for i in input().split()]
    sonar_result = input()
    opponent_orders = input()

    # Write an action using print
    # To debug: print("Debug messages...", file=sys.stderr, flush=True)

    print("MOVE N TORPEDO")
