"""Distances, walking time and the budget -> radius conversion."""

from __future__ import annotations

import math

EARTH_RADIUS_M = 6_371_008.8

WALK_SPEED_KMH = 4.2
# streets are not straight lines: inflate the straight-line distance
DETOUR_FACTOR = 1.3
# minutes kept aside for "finding the door", crossings, ordering at the counter
BUFFER_MIN = 5

MIN_RADIUS_M = 300
MAX_RADIUS_M = 6000

# how long a visit to each category typically takes, minutes
TYPICAL_VISIT_MIN = {
    "park": 25,
    "cafe": 30,
    "restaurant": 50,
    "museum": 50,
    "landmark": 10,
    "viewpoint": 10,
    "playground": 30,
    "mall": 40,
    "theatre": 90,
    "library": 30,
    "sport": 60,
    "other": 20,
}
DEFAULT_VISIT_MIN = 20


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = phi2 - phi1
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def _metres_per_minute() -> float:
    return WALK_SPEED_KMH * 1000 / 60


def walking_minutes(distance_m: float) -> float:
    """Walking time for a straight-line distance, detour included."""
    return distance_m * DETOUR_FACTOR / _metres_per_minute()


def distance_for_minutes(minutes: float) -> float:
    """Straight-line distance reachable on foot in `minutes`."""
    return max(0.0, minutes) * _metres_per_minute() / DETOUR_FACTOR


def typical_visit_minutes(categories: list[str] | None) -> int:
    """Visit length to plan for, given the categories the user asked about."""
    if not categories:
        return DEFAULT_VISIT_MIN
    known = [TYPICAL_VISIT_MIN[c] for c in categories if c in TYPICAL_VISIT_MIN]
    if not known:
        return DEFAULT_VISIT_MIN
    # plan for the shortest fitting option, so a tight budget still returns places
    return min(known)


def radius_from_budget(
    budget_min: float,
    typical_visit_min: int = DEFAULT_VISIT_MIN,
    return_to_start: bool = False,
) -> float:
    """Search radius in metres for a time budget.

    walk_min = budget - typical visit - buffer; halved when the user wants to
    come back to the starting point. Clamped to [MIN_RADIUS_M, MAX_RADIUS_M] so
    a tiny budget still yields something nearby instead of an empty list.
    """
    walk_min = budget_min - typical_visit_min - BUFFER_MIN
    if return_to_start:
        walk_min /= 2
    radius = distance_for_minutes(walk_min)
    return float(min(MAX_RADIUS_M, max(MIN_RADIUS_M, radius)))


def fits_in_budget(
    distance_m: float,
    visit_min: float,
    budget_min: float,
    return_to_start: bool = False,
    slack_min: float = 5.0,
) -> bool:
    """Can the user walk there, spend `visit_min`, and stay inside the budget?"""
    legs = 2 if return_to_start else 1
    total = walking_minutes(distance_m) * legs + visit_min
    return total <= budget_min + slack_min


def round_coords(lat: float, lon: float, digits: int = 3) -> tuple[float, float]:
    """Privacy: user coordinates are only ever logged/shown rounded."""
    return round(lat, digits), round(lon, digits)
