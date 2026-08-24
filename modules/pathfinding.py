"""
modules/pathfinding.py
=======================
INTRIX v2.2 — indoor path & directions ("HOW DO I GET THERE?").

This module is deliberately separate from app.py (UI) and from
trilateration.py / simulation.py (positioning). Positioning still only
answers "WHERE AM I?" — this file starts from that answer and finds a
walkable route to a destination.

WHERE THE WALKABLE GRAPH COMES FROM
------------------------------------
INTRIX did not previously have a corridor/graph representation — only
room rectangles in data/locations.csv (x, y, width, height per room,
loaded via modules.floors.locations_for_floor). Rather than invent a
second, hand-maintained floor-plan format, this module derives the
walkable graph directly from those same rectangles:

    a grid point is WALKABLE  <=>  it does not fall strictly inside
                                    any room's rectangle on that floor

Every point that is not "inside a room" is corridor space — the gaps
between rooms and the space around them. That reproduces the hallways
implied by the existing map without duplicating the floor data:
edit data/locations.csv (move/resize/add a room) and the corridor
network is regenerated automatically, no graph to hand-edit.

GRAPH DEFINITION (tune here if a floor's layout changes significantly)
------------------------------------------------------------------------
- GRID_RESOLUTION_M: spacing, in metres, between walkable graph nodes.
  1.0 m matches these ~30 x 20 m floors — coarse enough to stay fast,
  fine enough to route through the 2-3 m gaps between rooms.
- A node exists at every grid point that is not inside a room.
- Each node connects to its up/down/left/right/diagonal neighbours;
  edge weight = straight-line distance between them. A diagonal edge
  is only added when both of the neighbours it would "cut across" are
  also walkable, so a route can never clip through a room corner.
- Dijkstra's algorithm (modules.pathfinding.dijkstra) finds the
  shortest walkable route between any two nodes on this graph.

CONTINUOUS POSITION <-> GRAPH NODE
------------------------------------
modules.trilateration produces a continuous (x, y) estimate that is
generally NOT exactly on a grid node, and a destination's stored (x, y)
is usually a room's centre, which sits *inside* the room rather than in
a corridor. nearest_walkable_node() snaps any continuous point to the
closest walkable node; that node is what Dijkstra actually runs
between. The short hop from the raw (x, y) point to that snapped node
is assumed to be an unobstructed walk within the room/point's own
immediate space (e.g. from where you're standing to your room's door).
"""

import functools
import heapq
import math

import numpy as np

from modules.floors import locations_for_floor

GRID_RESOLUTION_M = 1.0     # metres between walkable graph nodes
ARRIVAL_THRESHOLD_M = 1.5   # "you have arrived" distance, in metres

STRAIGHT_ANGLE_DEG = 20     # |heading change| below this -> "straight"
SLIGHT_TURN_ANGLE_DEG = 60  # below this -> "slight" turn, else full turn
TURN_AROUND_ANGLE_DEG = 150 # above this -> "turn around"
SIMPLIFY_ANGLE_DEG = 8      # collinear-point removal tolerance for display


# ---------------------------------------------------------------------
# Graph construction (reuses data/locations.csv via modules.floors)
# ---------------------------------------------------------------------

def _room_rects(locations_df):
    rects = []
    for _, room in locations_df.iterrows():
        x0 = room["x"] - room["width"] / 2
        x1 = room["x"] + room["width"] / 2
        y0 = room["y"] - room["height"] / 2
        y1 = room["y"] + room["height"] / 2
        rects.append((x0, x1, y0, y1))
    return tuple(rects)


def _is_inside_room(x, y, rects):
    for x0, x1, y0, y1 in rects:
        if x0 < x < x1 and y0 < y < y1:
            return True
    return False


@functools.lru_cache(maxsize=None)
def _rects_for_floor(floor_code):
    return _room_rects(locations_for_floor(floor_code))


def _segment_crosses_room(p1, p2, rects, samples=40):
    """True if the straight segment p1->p2 passes through any room."""
    x1, y1 = p1
    x2, y2 = p2
    for s in range(1, samples):
        f = s / samples
        if _is_inside_room(x1 + (x2 - x1) * f, y1 + (y2 - y1) * f, rects):
            return True
    return False


def _string_pull(points, rects):
    """Shortens a dense grid path into the fewest possible straight
    segments that still avoid every room, by greedily "pulling a string
    taut" between the current point and the farthest point still in a
    clear line of sight. This is what turns a staircase of many small
    grid steps into the few long, natural-looking segments a person
    would actually walk (and a few clean turns instead of many)."""
    if len(points) <= 2:
        return list(points)

    pulled = [points[0]]
    i = 0
    n = len(points)
    while i < n - 1:
        j = n - 1
        while j > i + 1 and _segment_crosses_room(points[i], points[j], rects):
            j -= 1
        pulled.append(points[j])
        i = j
    return pulled


@functools.lru_cache(maxsize=None)
def _build_graph(floor_code, floor_width, floor_height, resolution):
    """Builds (and caches) the walkable node/edge graph for one floor.

    Returns (coords, graph) where coords[i] = (x, y) of node i, and
    graph[i] = [(neighbour_index, edge_weight_m), ...].
    """
    locations_df = locations_for_floor(floor_code)
    rects = _room_rects(locations_df)

    n_cols = int(round(floor_width / resolution)) + 1
    n_rows = int(round(floor_height / resolution)) + 1

    node_index = {}
    coords = []
    for j in range(n_rows):
        y = round(j * resolution, 6)
        for i in range(n_cols):
            x = round(i * resolution, 6)
            if _is_inside_room(x, y, rects):
                continue
            node_index[(i, j)] = len(coords)
            coords.append((x, y))

    graph = {idx: [] for idx in range(len(coords))}
    neighbor_offsets = [(1, 0), (-1, 0), (0, 1), (0, -1),
                         (1, 1), (1, -1), (-1, 1), (-1, -1)]

    for (i, j), idx in node_index.items():
        for di, dj in neighbor_offsets:
            n_idx = node_index.get((i + di, j + dj))
            if n_idx is None:
                continue
            if di != 0 and dj != 0:
                # Don't let a diagonal move cut across a room corner.
                if node_index.get((i + di, j)) is None or node_index.get((i, j + dj)) is None:
                    continue
            weight = math.hypot(di * resolution, dj * resolution)
            graph[idx].append((n_idx, weight))

    return coords, graph


def get_floor_graph(floor_code, floor_width, floor_height, resolution=GRID_RESOLUTION_M):
    """Public accessor for the (cached) walkable graph of a floor."""
    return _build_graph(floor_code, float(floor_width), float(floor_height), float(resolution))


def nearest_walkable_node(x, y, coords, graph=None, tie_tolerance=0.05):
    """Snaps a continuous (x, y) point to the closest walkable graph node.

    This is how the trilateration output (WHERE AM I?) and a
    destination's stored coordinates both connect to the pathfinding
    graph, since neither is guaranteed to land exactly on a node.

    A room's centre is often equidistant from more than one boundary
    node (e.g. its right-hand door and a nearby stretch of outer wall
    can both be exactly 4 m away). When `graph` is supplied, ties
    within `tie_tolerance` metres of the true minimum are broken in
    favour of the node with more graph connections — in practice this
    prefers a real corridor "doorway" over a dead-end sliver of outer
    wall that merely happens to be just as close.
    """
    if not coords:
        return None
    arr = np.asarray(coords)
    dist = np.sqrt((arr[:, 0] - x) ** 2 + (arr[:, 1] - y) ** 2)
    min_dist = dist.min()

    if graph is None:
        return int(np.argmin(dist))

    candidates = np.flatnonzero(dist <= min_dist + tie_tolerance)
    best_idx = int(candidates[0])
    best_degree = len(graph.get(best_idx, []))
    for idx in candidates[1:]:
        degree = len(graph.get(int(idx), []))
        if degree > best_degree:
            best_idx = int(idx)
            best_degree = degree
    return best_idx


# ---------------------------------------------------------------------
# Dijkstra shortest path
# ---------------------------------------------------------------------

def dijkstra(graph, start_idx, end_idx):
    """Shortest path between two node indices on `graph`.

    Returns (path_as_list_of_node_indices, total_distance) or
    (None, None) if no path exists.
    """
    if start_idx is None or end_idx is None:
        return None, None
    if start_idx == end_idx:
        return [start_idx], 0.0

    dist = {start_idx: 0.0}
    prev = {}
    visited = set()
    heap = [(0.0, start_idx)]

    while heap:
        d, u = heapq.heappop(heap)
        if u in visited:
            continue
        visited.add(u)
        if u == end_idx:
            break
        for v, w in graph.get(u, []):
            nd = d + w
            if nd < dist.get(v, math.inf):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))

    if end_idx not in dist:
        return None, None

    path = [end_idx]
    while path[-1] != start_idx:
        path.append(prev[path[-1]])
    path.reverse()
    return path, dist[end_idx]


# ---------------------------------------------------------------------
# Route computation: continuous position -> graph -> continuous destination
# ---------------------------------------------------------------------

def compute_route(user_x, user_y, dest_x, dest_y, floor_code, floor_width, floor_height,
                   resolution=GRID_RESOLUTION_M):
    """Computes a walkable route from the user's estimated position to a
    destination point, on the given floor.

    Returns a dict:
      - {"error": "<human-readable reason>"}  when no route could be produced
      - {"arrived": True, "points": [...], "route_distance": 0.0,
         "straight_line_distance": d}          when already at the destination
      - {"arrived": False, "points": [...], "route_distance": d_route,
         "straight_line_distance": d_line}     for a normal route

    "points" is the ordered list of (x, y) waypoints from the user's
    actual position to the actual destination point (graph nodes in
    between), suitable for drawing directly on the existing map.
    """
    straight_line = math.hypot(dest_x - user_x, dest_y - user_y)

    if straight_line <= ARRIVAL_THRESHOLD_M:
        return {
            "arrived": True,
            "points": [(user_x, user_y), (dest_x, dest_y)],
            "route_distance": 0.0,
            "straight_line_distance": straight_line,
        }

    coords, graph = get_floor_graph(floor_code, floor_width, floor_height, resolution)
    if not coords:
        return {"error": "This floor doesn't have any walkable area defined yet."}

    start_idx = nearest_walkable_node(user_x, user_y, coords, graph=graph)
    end_idx = nearest_walkable_node(dest_x, dest_y, coords, graph=graph)

    node_path, node_distance = dijkstra(graph, start_idx, end_idx)
    if node_path is None:
        return {"error": "No walkable route could be found to this destination."}

    node_points = [coords[i] for i in node_path]
    full_points = [(user_x, user_y)] + node_points + [(dest_x, dest_y)]

    # Short hops from the raw (x, y) points onto the snapped graph nodes.
    entry_hop = math.hypot(node_points[0][0] - user_x, node_points[0][1] - user_y)
    exit_hop = math.hypot(dest_x - node_points[-1][0], dest_y - node_points[-1][1])
    route_distance = entry_hop + node_distance + exit_hop

    # The raw path follows the grid one step at a time, which can zig-zag
    # slightly even along an essentially straight/diagonal corridor.
    # String-pulling collapses it to the fewest straight segments that
    # still never cross a room, for a cleaner map line and shorter,
    # more natural turn-by-turn directions. Distance above is still the
    # true shortest-path distance from Dijkstra, unaffected by this.
    display_points = _string_pull(full_points, _rects_for_floor(floor_code))

    return {
        "arrived": False,
        "points": display_points,
        "route_distance": route_distance,
        "straight_line_distance": straight_line,
    }


# ---------------------------------------------------------------------
# Path simplification (for both display and directions)
# ---------------------------------------------------------------------

def _vec(a, b):
    return (b[0] - a[0], b[1] - a[1])


def _vlen(v):
    return math.hypot(v[0], v[1])


def _signed_angle_deg(v1, v2):
    """Angle to rotate v1 onto v2, in degrees. Positive = counter-clockwise
    (a LEFT turn when walking, since x is +right and y is +up, matching
    this project's map/coordinate convention)."""
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    cross = v1[0] * v2[1] - v1[1] * v2[0]
    return math.degrees(math.atan2(cross, dot))


def simplify_path(points, angle_eps_deg=SIMPLIFY_ANGLE_DEG):
    """Collapses a dense route (many small grid steps) into just its
    corner points, by dropping any point that is (nearly) collinear
    with its neighbours. A straight corridor of A->B->C->D->E becomes
    a single A->E segment instead of four tiny ones."""
    pts = [p for p in points]
    if len(pts) <= 2:
        return pts

    simplified = [pts[0]]
    for i in range(1, len(pts) - 1):
        prev = simplified[-1]
        cur = pts[i]
        nxt = pts[i + 1]
        v1 = _vec(prev, cur)
        v2 = _vec(cur, nxt)
        if _vlen(v1) < 1e-9 or _vlen(v2) < 1e-9:
            continue
        if abs(_signed_angle_deg(v1, v2)) > angle_eps_deg:
            simplified.append(cur)
    simplified.append(pts[-1])
    return simplified


# ---------------------------------------------------------------------
# Direction generation (route geometry -> human-readable instructions)
# ---------------------------------------------------------------------

_TURN_TEXT = {
    "straight": ("⬆️", "Continue straight"),
    "left": ("⬅️", "Turn left"),
    "right": ("➡️", "Turn right"),
    "slight_left": ("↖️", "Slight left"),
    "slight_right": ("↗️", "Slight right"),
    "turn_around": ("🔄", "Turn around"),
}


def _classify_turn(angle_deg):
    a = angle_deg
    abs_a = abs(a)
    if abs_a < STRAIGHT_ANGLE_DEG:
        return "straight"
    if abs_a > TURN_AROUND_ANGLE_DEG:
        return "turn_around"
    if abs_a < SLIGHT_TURN_ANGLE_DEG:
        return "slight_left" if a > 0 else "slight_right"
    return "left" if a > 0 else "right"


def generate_directions(points, destination_name=None):
    """Converts route coordinates into a short list of step-by-step,
    human-readable directions. Nothing here is hard-coded per
    destination — it is generated purely from the geometry of `points`.

    Returns a list of {"icon": str, "text": str} steps, always starting
    with "You are here" and ending with an arrival message.
    """
    dest_text = f"You have reached {destination_name}." if destination_name else "You have reached your destination."

    if len(points) < 2:
        return [{"icon": "📍", "text": "You are here"},
                {"icon": "📍", "text": dest_text}]

    simplified = simplify_path(points)

    steps = [{"icon": "📍", "text": "You are here"}]
    prev_vec = None
    for i in range(1, len(simplified)):
        vec = _vec(simplified[i - 1], simplified[i])
        if _vlen(vec) < 1e-6:
            continue
        if prev_vec is None:
            steps.append({"icon": "⬆️", "text": "Go straight"})
        else:
            turn = _classify_turn(_signed_angle_deg(prev_vec, vec))
            icon, text = _TURN_TEXT[turn]
            if turn == "straight":
                # Don't repeat "keep going straight" for every grid step.
                if steps[-1]["text"] not in ("Go straight", "Continue straight"):
                    steps.append({"icon": icon, "text": text})
            else:
                steps.append({"icon": icon, "text": text})
        prev_vec = vec

    steps.append({"icon": "📍", "text": dest_text})
    return steps
