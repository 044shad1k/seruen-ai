"""Mini-route: order 2-3 picks and count the walking time between them."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from seruen.core import geo
from seruen.core.generate import Pick


@dataclass
class Leg:
    from_name: str
    to_name: str
    distance_m: int
    walk_min: int

    def as_dict(self) -> dict:
        return asdict(self)


MIN_STOP_VISIT_MIN = 10
ROUTE_BUFFER_MIN = 5


@dataclass
class Route:
    order: list[str] = field(default_factory=list)  # poi ids, in walking order
    legs: list[Leg] = field(default_factory=list)
    total_walk_min: int = 0
    total_visit_min: int = 0
    return_leg: Leg | None = None
    stop_visit_min: dict[str, int] = field(default_factory=dict)
    trimmed: bool = False  # visits were shortened to fit the budget

    @property
    def total_min(self) -> int:
        return self.total_walk_min + self.total_visit_min

    def fits(self, budget_min: int, slack_min: int = 5) -> bool:
        return self.total_min <= budget_min + slack_min

    def as_dict(self) -> dict:
        return {
            "order": self.order,
            "legs": [leg.as_dict() for leg in self.legs],
            "total_walk_min": self.total_walk_min,
            "total_visit_min": self.total_visit_min,
            "total_min": self.total_min,
            "return_leg": self.return_leg.as_dict() if self.return_leg else None,
            "stop_visit_min": self.stop_visit_min,
            "trimmed": self.trimmed,
        }


def build_route(
    picks: list[Pick],
    start_lat: float,
    start_lon: float,
    return_to_start: bool = False,
    budget_min: int | None = None,
) -> Route | None:
    """Order the stops and make the result fit the time budget.

    With a tight budget we first shorten the stops (down to 10 minutes each),
    then drop the last stop. If even two stops cannot fit, there is no honest
    mini-route to offer and we return None.
    """
    for count in range(len(picks), 1, -1):
        route = _greedy_route(picks[:count], start_lat, start_lon, return_to_start)
        if budget_min is None or route.fits(budget_min):
            return route
        if _trim_to_budget(route, picks[:count], budget_min):
            return route
    return None


def _trim_to_budget(route: Route, picks: list[Pick], budget_min: int) -> bool:
    """Shorten every stop equally so the route fits. False if it cannot."""
    available = budget_min - route.total_walk_min - ROUTE_BUFFER_MIN
    per_stop = available // max(1, len(picks))
    if per_stop < MIN_STOP_VISIT_MIN:
        return False
    route.total_visit_min = 0
    for pick in picks:
        minutes = int(min(pick.visit_minutes, per_stop))
        route.stop_visit_min[pick.poi_id] = minutes
        route.total_visit_min += minutes
    route.trimmed = True
    return True


def _greedy_route(
    picks: list[Pick],
    start_lat: float,
    start_lon: float,
    return_to_start: bool,
) -> Route:
    """Nearest-neighbour ordering from the user's position.

    Two or three stops is the whole problem space here, so a greedy walk is both
    optimal enough and trivially explainable in the UI.
    """
    remaining = list(picks)
    current = (start_lat, start_lon)
    current_name = "старт"
    route = Route()

    while remaining:
        nearest = min(
            remaining, key=lambda p: geo.haversine_m(current[0], current[1], p.lat, p.lon)
        )
        distance = geo.haversine_m(current[0], current[1], nearest.lat, nearest.lon)
        walk = geo.walking_minutes(distance)
        route.legs.append(
            Leg(
                from_name=current_name,
                to_name=nearest.name,
                distance_m=round(distance),
                walk_min=round(walk),
            )
        )
        route.total_walk_min += round(walk)
        route.total_visit_min += nearest.visit_minutes
        route.stop_visit_min[nearest.poi_id] = nearest.visit_minutes
        route.order.append(nearest.poi_id)
        current = (nearest.lat, nearest.lon)
        current_name = nearest.name
        remaining.remove(nearest)

    if return_to_start:
        distance = geo.haversine_m(current[0], current[1], start_lat, start_lon)
        walk = geo.walking_minutes(distance)
        route.return_leg = Leg(
            from_name=current_name,
            to_name="старт",
            distance_m=round(distance),
            walk_min=round(walk),
        )
        route.total_walk_min += round(walk)

    return route


def describe_route(route: Route, language: str = "ru") -> str:
    """One-line route summary for the card list."""
    if not route.legs:
        return ""
    stops = " → ".join([route.legs[0].from_name] + [leg.to_name for leg in route.legs])
    if route.return_leg:
        stops += " → старт"
    templates = {
        "ru": "Маршрут: {stops}. Пешком {walk} мин + на месте {visit} мин, итого ≈{total} мин.",
        "en": "Route: {stops}. {walk} min walking + {visit} min at the stops, ≈{total} min total.",
        "kk": "Маршрут: {stops}. Жаяу {walk} мин + орында {visit} мин, жиыны ≈{total} мин.",
    }
    template = templates.get(language, templates["ru"])
    text = template.format(
        stops=stops,
        walk=route.total_walk_min,
        visit=route.total_visit_min,
        total=route.total_min,
    )
    if route.trimmed:
        trimmed_note = {
            "ru": " Время на остановках сокращено, чтобы уложиться в бюджет.",
            "en": " Stop durations were shortened to fit the budget.",
            "kk": " Бюджетке сыю үшін аялдамадағы уақыт қысқартылды.",
        }
        text += trimmed_note.get(language, trimmed_note["ru"])
    return text
