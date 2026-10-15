"""Conservative doorway-to-interior clearance on a 2D occupancy grid."""

from collections import deque
import math

def circular_clearance_overlap(point, obj, radius):
    """Distance to an oriented rectangle, including rounded corner clearance."""
    theta = math.radians(obj.get("rotation", {}).get("z", 0))
    dx = point[0] - obj["position"]["x"]
    dy = point[1] - obj["position"]["y"]
    x = dx * math.cos(theta) + dy * math.sin(theta)
    y = -dx * math.sin(theta) + dy * math.cos(theta)
    gap_x = max(abs(x) - obj["dimensions"]["width"] / 2, 0)
    gap_y = max(abs(y) - obj["dimensions"]["length"] / 2, 0)
    return gap_x * gap_x + gap_y * gap_y <= radius * radius


def door_access(
    room: dict, clearance_m: float = 0.65, resolution_m: float = 0.10
) -> dict:
    """A circular pedestrian must enter and travel 1 m into the room.

    Obstacles are inflated oriented boxes; wall swing and mesh concavities are
    not modelled. The returned method documents this limitation explicitly.
    """
    origin, dims = room["position"], room["dimensions"]
    radius = clearance_m / 2
    walls = {wall["id"]: wall for wall in room.get("walls", [])}
    nx = math.ceil(dims["width"] / resolution_m)
    ny = math.ceil(dims["length"] / resolution_m)
    obstacles = [
        o
        for o in room.get("objects", [])
        if o["position"]["z"] < 1.8
        and o["position"]["z"] + o["dimensions"]["height"] > 0.05
    ]

    def point(cell):
        return (
            origin["x"] + (cell[0] + 0.5) * resolution_m,
            origin["y"] + (cell[1] + 0.5) * resolution_m,
        )

    def free(cell):
        x, y = point(cell)
        if not (
            origin["x"] + radius <= x <= origin["x"] + dims["width"] - radius
            and origin["y"] + radius <= y <= origin["y"] + dims["length"] - radius
        ):
            return False
        return not any(circular_clearance_overlap((x, y), o, radius) for o in obstacles)

    free_cells = {(x, y) for x in range(nx) for y in range(ny) if free((x, y))}
    evidence = []
    for door in room.get("doors", []):
        wall = walls[door["wall_id"]]
        a, b = wall["start_point"], wall["end_point"]
        fraction = door["position_on_wall"]
        # Sage stores position_on_wall as a fraction of the wall length.
        centre = (
            a["x"] + (b["x"] - a["x"]) * fraction,
            a["y"] + (b["y"] - a["y"]) * fraction,
        )
        start = (
            min(free_cells, key=lambda c: math.dist(point(c), centre))
            if free_cells
            else None
        )
        enters = (
            start is not None
            and door["width"] >= clearance_m
            and math.dist(point(start), centre) <= radius + 2 * resolution_m
        )
        reached = set([start]) if enters else set()
        queue = deque(reached)
        while queue:
            x, y = queue.popleft()
            for neighbor in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if neighbor in free_cells and neighbor not in reached:
                    reached.add(neighbor)
                    queue.append(neighbor)
        distance = max((math.dist(point(c), centre) for c in reached), default=0)
        evidence.append(
            {
                "door_id": door["id"],
                "pass": bool(enters and distance >= 1.0),
                "reachable_distance_m": round(distance, 3),
            }
        )
    return {
        "pass": bool(evidence) and all(e["pass"] for e in evidence),
        "method": "circular_clearance_oriented_boxes_grid_v2",
        "clearance_m": clearance_m,
        "resolution_m": resolution_m,
        "doors": evidence,
    }
