
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.floors import FLOOR_CODES, FLOOR_WIDTH, FLOOR_HEIGHT, locations_for_floor
from modules.pathfinding import (
    get_floor_graph,
    nearest_walkable_node,
    dijkstra,
    compute_route,
    simplify_path,
    generate_directions,
    _is_inside_room,
    _room_rects,
    _on_building_perimeter,
)

PASS = 0
FAIL = 0


def check(name, condition):
    global PASS, FAIL
    if condition:
        print(f"  [PASS] {name}")
        PASS += 1
    else:
        print(f"  [FAIL] {name}")
        FAIL += 1


FLOOR = "ground"
locations_df = locations_for_floor(FLOOR)
rects = _room_rects(locations_df)


print("1. Room-rectangle blocking is correct")
reception = locations_df[locations_df["name"] == "Reception"].iloc[0]
check("a room's own centre is blocked (inside the room)",
      _is_inside_room(reception["x"], reception["y"], rects))
check("a point far outside any room is walkable",
      not _is_inside_room(9, 10, rects))


print("\n2. Walkable graph is built and connected")
coords, graph = get_floor_graph(FLOOR, FLOOR_WIDTH, FLOOR_HEIGHT)
check("graph has walkable nodes", len(coords) > 0)
check("every node has at least one edge", all(len(graph[i]) > 0 for i in graph))


print("\n3. Dijkstra finds a path between two corridor nodes")
start_idx = nearest_walkable_node(9, 0, coords)
end_idx = nearest_walkable_node(19, 20, coords)
path, dist = dijkstra(graph, start_idx, end_idx)
check("a path was found", path is not None)
check("path distance is positive and finite", dist is not None and dist > 0)


print("\n4. compute_route: same-corridor route (Reception -> Emergency Dept, ground floor)")
reception = locations_df[locations_df["name"] == "Reception"].iloc[0]
emergency = locations_df[locations_df["name"] == "Emergency Department"].iloc[0]
route = compute_route(
    reception["x"], reception["y"], emergency["x"], emergency["y"],
    FLOOR, FLOOR_WIDTH, FLOOR_HEIGHT,
)
check("a route was found (no error)", "error" not in route)
check("route is not immediately 'arrived' (rooms are far apart)", not route.get("arrived", True))
check("route distance is >= straight-line distance (walls aren't skipped)",
      route["route_distance"] >= route["straight_line_distance"] - 1e-6)


print("\n5. Route does not cut through a room it should go around")

pharmacy = locations_df[locations_df["name"] == "Pharmacy"].iloc[0]
route2 = compute_route(
    reception["x"], reception["y"], pharmacy["x"], pharmacy["y"],
    FLOOR, FLOOR_WIDTH, FLOOR_HEIGHT,
)
inside_a_room_midway = False
mid_points = route2["points"][1:-1]
for (px, py) in mid_points:
    if _is_inside_room(px, py, rects):
        inside_a_room_midway = True
check("no intermediate route point lies inside a room", not inside_a_room_midway)


print("\n6. Already-at-destination is detected")
route3 = compute_route(
    reception["x"], reception["y"], reception["x"] + 0.5, reception["y"],
    FLOOR, FLOOR_WIDTH, FLOOR_HEIGHT,
)
check("very close points are reported as 'arrived'", route3.get("arrived") is True)


print("\n7. simplify_path collapses a straight run of points")
straight_run = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]
simplified = simplify_path(straight_run)
check("a straight run simplifies to just its endpoints", simplified == [(0, 0), (4, 0)])

bent_run = [(0, 0), (2, 0), (4, 0), (4, 2), (4, 4)]
simplified_bent = simplify_path(bent_run)
check("a single 90-degree corner keeps exactly 3 points", simplified_bent == [(0, 0), (4, 0), (4, 4)])


print("\n8. Turn detection matches compass intuition (north->east = right, east->north = left)")
north_then_east = [(0, 0), (0, 5), (5, 5)]
steps_ne = generate_directions(north_then_east, "Test")
texts_ne = [s["text"] for s in steps_ne]
check("north then east produces a 'Turn right' step", "Turn right" in texts_ne)

east_then_north = [(0, 0), (5, 0), (5, 5)]
steps_en = generate_directions(east_then_north, "Test")
texts_en = [s["text"] for s in steps_en]
check("east then north produces a 'Turn left' step", "Turn left" in texts_en)


print("\n9. Directions always start with 'You are here' and end with arrival")
check("first step is 'You are here'", steps_ne[0]["text"] == "You are here")
check("last step mentions the destination has been reached", "reached" in steps_ne[-1]["text"])


print("\n10. A long straight corridor produces one 'straight' instruction, not many")
long_straight = [(x, 0) for x in range(0, 21)]  # 21 tiny grid steps in a line
steps_long = generate_directions(long_straight, "End")
straight_count = sum(1 for s in steps_long if s["text"] in ("Go straight", "Continue straight"))
check(f"only one straight instruction was generated (got {straight_count})", straight_count == 1)


print("\n11. compute_route works on every floor")
ok = True
for f in FLOOR_CODES:
    locs = locations_for_floor(f)
    a = locs.iloc[0]
    b = locs.iloc[-1]
    r = compute_route(a["x"], a["y"], b["x"], b["y"], f, FLOOR_WIDTH, FLOOR_HEIGHT)
    if "error" in r:
        ok = False
        print(f"  problem on floor '{f}': {r['error']}")
check("every floor produces a usable route between its first and last room", ok)


print("\n12. Unreachable destination (off the floor) reports an error, not a crash")
route_far = compute_route(4, 4, 1000, 1000, FLOOR, FLOOR_WIDTH, FLOOR_HEIGHT)
check("a destination far off the floor still returns a result (nearest-node snapping)",
      "error" in route_far or "points" in route_far)


print("\n13. No walkable node sits on the building's own exterior wall")

check("_on_building_perimeter flags the four outer edges",
      _on_building_perimeter(0, 5, FLOOR_WIDTH, FLOOR_HEIGHT)
      and _on_building_perimeter(FLOOR_WIDTH, 5, FLOOR_WIDTH, FLOOR_HEIGHT)
      and _on_building_perimeter(5, 0, FLOOR_WIDTH, FLOOR_HEIGHT)
      and _on_building_perimeter(5, FLOOR_HEIGHT, FLOOR_WIDTH, FLOOR_HEIGHT))
check("_on_building_perimeter does not flag a genuine interior point",
      not _on_building_perimeter(9, 10, FLOOR_WIDTH, FLOOR_HEIGHT))

no_perimeter_nodes = True
for f in FLOOR_CODES:
    coords_f, _ = get_floor_graph(f, FLOOR_WIDTH, FLOOR_HEIGHT)
    for (px, py) in coords_f:
        if px <= 0 or px >= FLOOR_WIDTH or py <= 0 or py >= FLOOR_HEIGHT:
            no_perimeter_nodes = False
check("no floor's walkable graph contains a node on the outer wall", no_perimeter_nodes)


print("\n14. Regression: user next to an exterior-wall-flush room routes")
print("    through the real internal corridor, not along the outer wall")
icu_floor = "2"
icu_locs = locations_for_floor(icu_floor)
icu = icu_locs[icu_locs["name"] == "ICU"].iloc[0]
route_icu = compute_route(19.2, 2.1, icu["x"], icu["y"], icu_floor, FLOOR_WIDTH, FLOOR_HEIGHT)
mid_icu = route_icu["points"][1:-1]
check("route to ICU never touches the outer wall",
      not any(px <= 0 or px >= FLOOR_WIDTH or py <= 0 or py >= FLOOR_HEIGHT for px, py in mid_icu))
check("route to ICU passes through the middle corridor band (y around 8)",
      any(7 <= py <= 11 for _, py in mid_icu))


print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
