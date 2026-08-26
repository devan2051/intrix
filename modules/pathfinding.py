import functools
import heapq
import math

import numpy as np

from modules.floors import locations_for_floor

GRID_RESOLUTION_M = 1.0     
ARRIVAL_THRESHOLD_M = 1.5   
PERIMETER_EPSILON_M = 1e-6  

STRAIGHT_ANGLE_DEG = 20     
SLIGHT_TURN_ANGLE_DEG = 60  
TURN_AROUND_ANGLE_DEG = 150 
SIMPLIFY_ANGLE_DEG = 8      


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


def _on_building_perimeter(x, y, floor_width, floor_height, eps=PERIMETER_EPSILON_M):
    return (
        x <= eps or x >= floor_width - eps
        or y <= eps or y >= floor_height - eps
    )


@functools.lru_cache(maxsize=None)
def _rects_for_floor(floor_code):
    return _room_rects(locations_for_floor(floor_code))


def _segment_crosses_room(p1, p2, rects, samples=40):
    x1, y1 = p1
    x2, y2 = p2
    for s in range(1, samples):
        f = s / samples
        if _is_inside_room(x1 + (x2 - x1) * f, y1 + (y2 - y1) * f, rects):
            return True
    return False


def _string_pull(points, rects):
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
            if _on_building_perimeter(x, y, floor_width, floor_height):
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
    return _build_graph(floor_code, float(floor_width), float(floor_height), float(resolution))


def nearest_walkable_node(x, y, coords, graph=None, tie_tolerance=0.05):
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

def dijkstra(graph, start_idx, end_idx):
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


def compute_route(user_x, user_y, dest_x, dest_y, floor_code, floor_width, floor_height,
                   resolution=GRID_RESOLUTION_M):
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

    entry_hop = math.hypot(node_points[0][0] - user_x, node_points[0][1] - user_y)
    exit_hop = math.hypot(dest_x - node_points[-1][0], dest_y - node_points[-1][1])
    route_distance = entry_hop + node_distance + exit_hop

    display_points = _string_pull(full_points, _rects_for_floor(floor_code))

    return {
        "arrived": False,
        "points": display_points,
        "route_distance": route_distance,
        "straight_line_distance": straight_line,
    }


def _vec(a, b):
    return (b[0] - a[0], b[1] - a[1])


def _vlen(v):
    return math.hypot(v[0], v[1])


def _signed_angle_deg(v1, v2):
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    cross = v1[0] * v2[1] - v1[1] * v2[0]
    return math.degrees(math.atan2(cross, dot))


def simplify_path(points, angle_eps_deg=SIMPLIFY_ANGLE_DEG):
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
                if steps[-1]["text"] not in ("Go straight", "Continue straight"):
                    steps.append({"icon": icon, "text": text})
            else:
                steps.append({"icon": icon, "text": text})
        prev_vec = vec

    steps.append({"icon": "📍", "text": dest_text})
    return steps
